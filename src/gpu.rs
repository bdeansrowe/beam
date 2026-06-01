use anyhow::{anyhow, Result};
use bytemuck::{Pod, Zeroable};
use std::rc::Rc;
use wgpu::*;
use winit::{dpi::PhysicalSize, window::Window};

use crate::bvh::{build_trivial_scene, HitRecord, Material, MaterialType, Ray, Vertex, TriangleRecord};

// ── Camera uniform — mirrors WGSL `struct Camera` in ray_gen.wgsl ─────────────
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CameraUniform {
    origin:     [f32; 4],
    lower_left: [f32; 4],
    horizontal: [f32; 4],
    vertical:   [f32; 4],
    dims:       [u32; 4],  // [0]=width [1]=height [2]=frame [3]=pad
}

impl CameraUniform {
    fn new(width: u32, height: u32, frame: u32) -> Self {
        let half_h = (60.0_f32.to_radians() * 0.5).tan();
        let half_w = (width as f32 / height as f32) * half_h;
        CameraUniform {
            origin:     [0.0, 0.0, 3.0, 0.0],
            lower_left: [-half_w, -half_h, 2.0, 0.0],
            horizontal: [2.0 * half_w, 0.0, 0.0, 0.0],
            vertical:   [0.0, 2.0 * half_h, 0.0, 0.0],
            dims:       [width, height, frame, 0],
        }
    }
}

// ── GPU state ──────────────────────────────────────────────────────────────────
pub struct GpuState {
    surface:  Surface<'static>,
    device:   Device,
    queue:    Queue,
    config:   SurfaceConfiguration,
    pub size: PhysicalSize<u32>,

    // Ray generation (Step 3)
    camera_buf:       Buffer,
    #[allow(dead_code)]
    ray_buf:          Buffer,
    ray_gen_pipeline: ComputePipeline,
    ray_gen_bg0:      BindGroup,
    ray_gen_bg1:      BindGroup,

    // BVH scene buffers (Step 5)
    #[allow(dead_code)]
    bvh_node_buf:      Buffer,
    #[allow(dead_code)]
    tlas_instance_buf: Buffer,
    #[allow(dead_code)]
    sphere_buf:        Buffer,

    // Geometry buffers (Step 5.5)
    #[allow(dead_code)]
    vertex_buf:   Buffer,
    #[allow(dead_code)]
    geometry_buf: Buffer,
    // Material buffer (Step 6)
    #[allow(dead_code)]
    material_buf: Buffer,

    // Hit record buffer (Step 6b)
    #[allow(dead_code)]
    hit_buf: Buffer,

    // Shading pipelines (Step 6b/6d)
    shade_diffuse_pipeline:  ComputePipeline,
    shade_metallic_pipeline: ComputePipeline,
    shade_glass_pipeline:    ComputePipeline,
    shade_scene_bg0:         BindGroup,
    shade_bg1:               BindGroup,

    // Sphere intersection + HDR output (Step 4/5)
    #[allow(dead_code)]
    hdr_texture:        Texture,
    intersect_pipeline: ComputePipeline,
    intersect_bg1:      BindGroup,
    scene_bg0:          BindGroup,

    // Blit to canvas (Step 4)
    blit_pipeline: RenderPipeline,
    blit_bg0:      BindGroup,

    frame: u32,
}

impl GpuState {
    pub async fn new(window: Rc<Window>) -> Result<Self> {
        let size = {
            #[cfg(target_arch = "wasm32")]
            {
                use winit::dpi::PhysicalSize;
                let canvas = web_sys::window().unwrap().document().unwrap()
                    .get_element_by_id("canvas").unwrap();
                let canvas: web_sys::HtmlCanvasElement =
                    wasm_bindgen::JsCast::dyn_into(canvas).unwrap();
                PhysicalSize::new(canvas.width(), canvas.height())
            }
            #[cfg(not(target_arch = "wasm32"))]
            { window.inner_size() }
        };

        log::info!("GpuState::new — canvas size: {}×{}", size.width, size.height);

        let instance = Instance::new(&InstanceDescriptor {
            backends: Backends::BROWSER_WEBGPU,
            ..Default::default()
        });
        let surface = instance.create_surface(window)?;
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference:       PowerPreference::HighPerformance,
                compatible_surface:     Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| anyhow!("No adapter found: {:?}", e))?;

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label:                 Some("Main Device"),
                required_features:     Features::empty(),
                required_limits:       Limits::default(),
                experimental_features: ExperimentalFeatures::default(),
                memory_hints:          Default::default(),
                trace:                 Trace::Off,
            })
            .await?;

        let surface_caps   = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats.iter()
            .find(|f| f.is_srgb()).copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = SurfaceConfiguration {
            usage:                         TextureUsages::RENDER_ATTACHMENT,
            format:                        surface_format,
            width:                         size.width,
            height:                        size.height,
            present_mode:                  PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode:                    surface_caps.alpha_modes[0],
            view_formats:                  vec![],
        };
        surface.configure(&device, &config);

        let (ray_gen_pipeline, ray_gen_bg0, ray_gen_bg1, camera_buf, ray_buf) =
            Self::create_ray_gen(&device, size.width, size.height);

        let (bvh_node_buf, tlas_instance_buf, sphere_buf, vertex_buf, geometry_buf, material_buf) =
            Self::create_bvh_buffers(&device);

        let hit_buf = device.create_buffer(&BufferDescriptor {
            label:              Some("Hit Buffer"),
            size:               (size.width * size.height) as u64
                                    * std::mem::size_of::<HitRecord>() as u64,
            usage:              BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let (hdr_texture, hdr_view, intersect_pipeline, intersect_bg1, scene_bg0) =
            Self::create_intersect(
                &device, &ray_buf,
                &hit_buf,
                &bvh_node_buf, &tlas_instance_buf, &sphere_buf,
                &vertex_buf, &geometry_buf, &material_buf,
                size.width, size.height,
            ).await;

        let (shade_diffuse_pipeline, shade_metallic_pipeline, shade_glass_pipeline, shade_scene_bg0, shade_bg1) =
            Self::create_shade_pipelines(
                &device,
                &ray_buf,
                &hit_buf,
                &sphere_buf,
                &vertex_buf,
                &geometry_buf,
                &material_buf,
                &hdr_view,
            ).await;

        let (blit_pipeline, blit_bg0) =
            Self::create_blit(&device, &config, &hdr_view);

        log::info!(
            "Step 6d ready: {}×{} rays, glass BSDF online (Fresnel/Snell/Beer/medium-stack)",
            size.width, size.height,
        );

        Ok(Self {
            surface, device, queue, config, size,
            camera_buf, ray_buf, ray_gen_pipeline, ray_gen_bg0, ray_gen_bg1,
            bvh_node_buf, tlas_instance_buf, sphere_buf,
            vertex_buf, geometry_buf, material_buf,
            hit_buf,
            shade_diffuse_pipeline, shade_metallic_pipeline, shade_glass_pipeline, shade_scene_bg0, shade_bg1,
            hdr_texture, intersect_pipeline, intersect_bg1, scene_bg0,
            blit_pipeline, blit_bg0,
            frame: 0,
        })
    }

    fn upload_slice<T: Pod>(device: &Device, label: &str, data: &[T]) -> Buffer {
        let bytes = bytemuck::cast_slice(data);
        let buf = device.create_buffer(&BufferDescriptor {
            label:              Some(label),
            size:               bytes.len() as u64,
            usage:              BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        buf.slice(..).get_mapped_range_mut().copy_from_slice(bytes);
        buf.unmap();
        buf
    }

    fn create_bvh_buffers(device: &Device) -> (Buffer, Buffer, Buffer, Buffer, Buffer, Buffer) {
        let (nodes, instances, spheres) = build_trivial_scene();
        let bvh_node_buf      = Self::upload_slice(device, "BVH Nodes",      &nodes);
        let tlas_instance_buf = Self::upload_slice(device, "TLAS Instances", &instances);
        let sphere_buf        = Self::upload_slice(device, "Spheres",        &spheres);
        let vertex_buf        = Self::upload_slice(device, "Vertices",       &[Vertex::zeroed()]);
        let geometry_buf      = Self::upload_slice(device, "Geometry",       &[TriangleRecord::zeroed()]);
        let material_buf      = Self::upload_slice(device, "Materials", &[
            // Index 0: air — medium-stack placeholder; never shaded directly.
            Material {
                base_color:    [0.0, 0.0, 0.0, 0.0],
                emission:      [0.0, 0.0, 0.0, 0.0],
                absorption:    [0.0, 0.0, 0.0, 0.0],
                material_type: MaterialType::Diffuse,
                ior:           1.0,
                roughness:     0.0,
                _pad:          0.0,
            },
            // Index 1: tinted glass sphere — visibility test.
            Material {
                base_color:    [0.2, 0.6, 1.0, 1.0],
                emission:      [0.0, 0.0, 0.0, 0.0],
                absorption:    [0.0, 0.0, 0.0, 0.0],
                material_type: MaterialType::Glass,
                ior:           1.5,
                roughness:     0.0,
                _pad:          0.0,
            },
        ]);
        (bvh_node_buf, tlas_instance_buf, sphere_buf, vertex_buf, geometry_buf, material_buf)
    }

    fn create_ray_gen(
        device: &Device, width: u32, height: u32,
    ) -> (ComputePipeline, BindGroup, BindGroup, Buffer, Buffer) {
        let bg0_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label:   Some("Ray Gen BG0"),
            entries: &[BindGroupLayoutEntry {
                binding: 0, visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false, min_binding_size: None,
                },
                count: None,
            }],
        });
        let bg1_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label:   Some("Ray Gen BG1"),
            entries: &[BindGroupLayoutEntry {
                binding: 0, visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false, min_binding_size: None,
                },
                count: None,
            }],
        });

        let camera_buf = device.create_buffer(&BufferDescriptor {
            label: Some("Camera Uniform"),
            size:  std::mem::size_of::<CameraUniform>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ray_buf = device.create_buffer(&BufferDescriptor {
            label: Some("Ray Buffer"),
            size:  (width * height) as u64 * std::mem::size_of::<Ray>() as u64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let ray_gen_bg0 = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Ray Gen BG0"), layout: &bg0_layout,
            entries: &[BindGroupEntry { binding: 0, resource: camera_buf.as_entire_binding() }],
        });
        let ray_gen_bg1 = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Ray Gen BG1"), layout: &bg1_layout,
            entries: &[BindGroupEntry { binding: 0, resource: ray_buf.as_entire_binding() }],
        });

        let ray_gen_src = format!("{}\n{}", include_str!("common_common.wgsl"), include_str!("ray_gen.wgsl"));
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label:  Some("Ray Gen"),
            source: ShaderSource::Wgsl(std::borrow::Cow::Owned(ray_gen_src)),
        });
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Ray Gen Layout"),
            bind_group_layouts: &[&bg0_layout, &bg1_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Ray Gen"), layout: Some(&layout), module: &shader,
            entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        });

        (pipeline, ray_gen_bg0, ray_gen_bg1, camera_buf, ray_buf)
    }

    async fn create_intersect(
        device:            &Device,
        ray_buf:           &Buffer,
        hit_buf:           &Buffer,
        bvh_node_buf:      &Buffer,
        tlas_instance_buf: &Buffer,
        sphere_buf:        &Buffer,
        vertex_buf:        &Buffer,
        geometry_buf:      &Buffer,
        material_buf:      &Buffer,
        width:  u32,
        height: u32,
    ) -> (Texture, TextureView, ComputePipeline, BindGroup, BindGroup) {
        let hdr_texture = device.create_texture(&TextureDescriptor {
            label:            Some("HDR Texture"),
            size:             Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count:  1,
            sample_count:     1,
            dimension:        TextureDimension::D2,
            format:           TextureFormat::Rgba16Float,
            usage:            TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats:     &[],
        });
        let hdr_view = hdr_texture.create_view(&TextureViewDescriptor::default());

        // BG0 — scene-global: BVH nodes, TLAS instances, spheres, vertices, geometry
        let bg0_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label:   Some("Intersect BG0"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0, visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1, visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2, visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3, visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4, visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 5, visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let scene_bg0 = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Intersect BG0"), layout: &bg0_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: bvh_node_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: tlas_instance_buf.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: sphere_buf.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: vertex_buf.as_entire_binding() },
                BindGroupEntry { binding: 4, resource: geometry_buf.as_entire_binding() },
                BindGroupEntry { binding: 5, resource: material_buf.as_entire_binding() },
            ],
        });

        // BG1 — per-pass: ray buffer, HDR output, hit records
        let bg1_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label:   Some("Intersect BG1"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0, visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1, visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access:         StorageTextureAccess::WriteOnly,
                        format:         TextureFormat::Rgba16Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2, visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false, min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let intersect_bg1 = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Intersect BG1"), layout: &bg1_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: ray_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&hdr_view) },
                BindGroupEntry { binding: 2, resource: hit_buf.as_entire_binding() },
            ],
        });

        let intersect_src = format!("{}\n{}", include_str!("common_common.wgsl"), include_str!("intersect.wgsl"));
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label:  Some("Intersect"),
            source: ShaderSource::Wgsl(std::borrow::Cow::Owned(intersect_src)),
        });

        // Dx: surface any WGSL compilation errors or warnings immediately
        let comp_info = shader.get_compilation_info().await;
        for msg in &comp_info.messages {
            match msg.message_type {
                CompilationMessageType::Error =>
                    log::error!("intersect.wgsl: {}", msg.message),
                CompilationMessageType::Warning =>
                    log::warn!("intersect.wgsl: {}", msg.message),
                _ => {}
            }
        }

        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Intersect Layout"),
            bind_group_layouts: &[&bg0_layout, &bg1_layout],
            push_constant_ranges: &[],
        });
        device.push_error_scope(ErrorFilter::Validation);
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Intersect"), layout: Some(&layout), module: &shader,
            entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        });
        if let Some(err) = device.pop_error_scope().await {
            log::error!("Intersect pipeline validation error: {:?}", err);
        }

        (hdr_texture, hdr_view, pipeline, intersect_bg1, scene_bg0)
    }

    fn create_blit(
        device: &Device, config: &SurfaceConfiguration, hdr_view: &TextureView,
    ) -> (RenderPipeline, BindGroup) {
        let bg0_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label:   Some("Blit BG0"),
            entries: &[BindGroupLayoutEntry {
                binding: 0, visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type:    TextureSampleType::Float { filterable: false },
                    view_dimension: TextureViewDimension::D2,
                    multisampled:   false,
                },
                count: None,
            }],
        });
        let blit_bg0 = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Blit BG0"), layout: &bg0_layout,
            entries: &[BindGroupEntry {
                binding: 0, resource: BindingResource::TextureView(hdr_view),
            }],
        });

        let shader = device.create_shader_module(include_wgsl!("shader.wgsl"));
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Blit Layout"),
            bind_group_layouts: &[&bg0_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("Blit Pipeline"), layout: Some(&layout),
            vertex: VertexState {
                module: &shader, entry_point: Some("vs_main"),
                buffers: &[], compilation_options: Default::default(),
            },
            fragment: Some(FragmentState {
                module: &shader, entry_point: Some("fs_main"),
                targets: &[Some(ColorTargetState {
                    format:     config.format,
                    blend:      Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive:     PrimitiveState::default(),
            depth_stencil: None,
            multisample:   MultisampleState::default(),
            multiview:     None,
            cache:         None,
        });

        (pipeline, blit_bg0)
    }

    async fn create_shade_pipelines(
        device:       &Device,
        ray_buf:      &Buffer,
        hit_buf:      &Buffer,
        sphere_buf:   &Buffer,
        vertex_buf:   &Buffer,
        geometry_buf: &Buffer,
        material_buf: &Buffer,
        hdr_view:     &TextureView,
    ) -> (ComputePipeline, ComputePipeline, ComputePipeline, BindGroup, BindGroup) {
        // BG0 — scene resources for shading (bindings 2-5; 0/1 are intersect-only)
        let bg0_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label:   Some("Shade BG0"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 2, visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3, visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4, visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 5, visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let shade_scene_bg0 = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Shade BG0"), layout: &bg0_layout,
            entries: &[
                BindGroupEntry { binding: 2, resource: sphere_buf.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: vertex_buf.as_entire_binding() },
                BindGroupEntry { binding: 4, resource: geometry_buf.as_entire_binding() },
                BindGroupEntry { binding: 5, resource: material_buf.as_entire_binding() },
            ],
        });

        // BG1 — per-pass: hit records (read-only), HDR output, rays (read-only)
        let bg1_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label:   Some("Shade BG1"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0, visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1, visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access:         StorageTextureAccess::WriteOnly,
                        format:         TextureFormat::Rgba16Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2, visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false, min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let shade_bg1 = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Shade BG1"), layout: &bg1_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: hit_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: BindingResource::TextureView(hdr_view) },
                BindGroupEntry { binding: 2, resource: ray_buf.as_entire_binding() },
            ],
        });

        // Compose common_common.wgsl + shade_common.wgsl + shade_<variant>.wgsl
        let common_common = include_str!("common_common.wgsl");
        let shade_common  = include_str!("shade_common.wgsl");
        let diffuse       = include_str!("shade_diffuse.wgsl");
        let metallic      = include_str!("shade_metallic.wgsl");
        let glass         = include_str!("shade_glass.wgsl");

        let diffuse_src  = format!("{}\n{}\n{}", common_common, shade_common, diffuse);
        let metallic_src = format!("{}\n{}\n{}", common_common, shade_common, metallic);
        let glass_src    = format!("{}\n{}\n{}", common_common, shade_common, glass);

        let diffuse_module = device.create_shader_module(ShaderModuleDescriptor {
            label:  Some("Shade Diffuse"),
            source: ShaderSource::Wgsl(std::borrow::Cow::Owned(diffuse_src)),
        });
        let diffuse_info = diffuse_module.get_compilation_info().await;
        for msg in &diffuse_info.messages {
            match msg.message_type {
                CompilationMessageType::Error   => log::error!("shade_diffuse: {}", msg.message),
                CompilationMessageType::Warning => log::warn!("shade_diffuse: {}",  msg.message),
                _ => {}
            }
        }

        let metallic_module = device.create_shader_module(ShaderModuleDescriptor {
            label:  Some("Shade Metallic"),
            source: ShaderSource::Wgsl(std::borrow::Cow::Owned(metallic_src)),
        });
        let metallic_info = metallic_module.get_compilation_info().await;
        for msg in &metallic_info.messages {
            match msg.message_type {
                CompilationMessageType::Error   => log::error!("shade_metallic: {}", msg.message),
                CompilationMessageType::Warning => log::warn!("shade_metallic: {}",  msg.message),
                _ => {}
            }
        }

        let glass_module = device.create_shader_module(ShaderModuleDescriptor {
            label:  Some("Shade Glass"),
            source: ShaderSource::Wgsl(std::borrow::Cow::Owned(glass_src)),
        });
        let glass_info = glass_module.get_compilation_info().await;
        for msg in &glass_info.messages {
            match msg.message_type {
                CompilationMessageType::Error   => log::error!("shade_glass: {}", msg.message),
                CompilationMessageType::Warning => log::warn!("shade_glass: {}",  msg.message),
                _ => {}
            }
        }

        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Shade Layout"),
            bind_group_layouts: &[&bg0_layout, &bg1_layout],
            push_constant_ranges: &[],
        });

        device.push_error_scope(ErrorFilter::Validation);
        let shade_diffuse_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Shade Diffuse"), layout: Some(&layout), module: &diffuse_module,
            entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        });
        if let Some(err) = device.pop_error_scope().await {
            log::error!("Shade Diffuse pipeline validation error: {:?}", err);
        }

        device.push_error_scope(ErrorFilter::Validation);
        let shade_metallic_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Shade Metallic"), layout: Some(&layout), module: &metallic_module,
            entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        });
        if let Some(err) = device.pop_error_scope().await {
            log::error!("Shade Metallic pipeline validation error: {:?}", err);
        }

        device.push_error_scope(ErrorFilter::Validation);
        let shade_glass_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Shade Glass"), layout: Some(&layout), module: &glass_module,
            entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        });
        if let Some(err) = device.pop_error_scope().await {
            log::error!("Shade Glass pipeline validation error: {:?}", err);
        }

        (shade_diffuse_pipeline, shade_metallic_pipeline, shade_glass_pipeline, shade_scene_bg0, shade_bg1)
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width  = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    pub fn reconfigure(&mut self) {
        self.surface.configure(&self.device, &self.config);
    }

    pub fn render(&mut self) -> Result<(), SurfaceError> {
        let cam = CameraUniform::new(self.size.width, self.size.height, self.frame);
        self.queue.write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&cam));

        let output  = self.surface.get_current_texture()?;
        let view    = output.texture.create_view(&TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Frame Encoder"),
        });

        // 1. Ray generation
        {
            let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Ray Gen"), timestamp_writes: None,
            });
            cpass.set_pipeline(&self.ray_gen_pipeline);
            cpass.set_bind_group(0, &self.ray_gen_bg0, &[]);
            cpass.set_bind_group(1, &self.ray_gen_bg1, &[]);
            cpass.dispatch_workgroups(
                (self.size.width + 7) / 8,
                (self.size.height + 7) / 8,
                1,
            );
        }

        // 2. BVH traversal → HDR texture
        {
            let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Intersect"), timestamp_writes: None,
            });
            cpass.set_pipeline(&self.intersect_pipeline);
            cpass.set_bind_group(0, &self.scene_bg0, &[]);
            cpass.set_bind_group(1, &self.intersect_bg1, &[]);
            cpass.dispatch_workgroups(
                (self.size.width + 7) / 8,
                (self.size.height + 7) / 8,
                1,
            );
        }

        // 3. Diffuse shading → HDR texture
        {
            let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Shade Diffuse"), timestamp_writes: None,
            });
            cpass.set_pipeline(&self.shade_diffuse_pipeline);
            cpass.set_bind_group(0, &self.shade_scene_bg0, &[]);
            cpass.set_bind_group(1, &self.shade_bg1, &[]);
            cpass.dispatch_workgroups(
                (self.size.width + 7) / 8,
                (self.size.height + 7) / 8,
                1,
            );
        }

        // 4. Metallic shading → HDR texture
        {
            let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Shade Metallic"), timestamp_writes: None,
            });
            cpass.set_pipeline(&self.shade_metallic_pipeline);
            cpass.set_bind_group(0, &self.shade_scene_bg0, &[]);
            cpass.set_bind_group(1, &self.shade_bg1, &[]);
            cpass.dispatch_workgroups(
                (self.size.width + 7) / 8,
                (self.size.height + 7) / 8,
                1,
            );
        }

        // 5. Glass shading → HDR texture
        {
            let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Shade Glass"), timestamp_writes: None,
            });
            cpass.set_pipeline(&self.shade_glass_pipeline);
            cpass.set_bind_group(0, &self.shade_scene_bg0, &[]);
            cpass.set_bind_group(1, &self.shade_bg1, &[]);
            cpass.dispatch_workgroups(
                (self.size.width + 7) / 8,
                (self.size.height + 7) / 8,
                1,
            );
        }

        // 6. Blit HDR texture → canvas
        {
            let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
                label:                    Some("Blit"),
                color_attachments:        &[Some(RenderPassColorAttachment {
                    view: &view, resolve_target: None, depth_slice: None,
                    ops: Operations {
                        load:  LoadOp::Clear(Color::BLACK),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes:         None,
                occlusion_query_set:      None,
            });
            rpass.set_pipeline(&self.blit_pipeline);
            rpass.set_bind_group(0, &self.blit_bg0, &[]);
            rpass.draw(0..3, 0..1);
        }

        self.frame += 1;
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }
}
