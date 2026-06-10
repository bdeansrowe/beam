use anyhow::{anyhow, Result};
use bytemuck::{Pod, Zeroable};
use std::cell::RefCell;
use std::rc::Rc;
use wgpu::*;
use wgpu::util::DeviceExt;
use winit::{dpi::PhysicalSize, window::Window};

use crate::bvh::{build_trivial_scene, HitRecord, LightUniform, Material, MaterialType, PixelState, Ray, Vertex, TriangleRecord};

const BLOOM_AMPLIFICATION: u32 = 256;

// ── Camera uniform — mirrors WGSL `struct Camera` in ray_gen.wgsl ─────────────
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CameraUniform {
    origin:     [f32; 4],
    lower_left: [f32; 4],
    horizontal: [f32; 4],
    vertical:   [f32; 4],
    dims:      [u32; 2],  // [0]=width [1]=height
    _dims_pad: [u32; 2],
}

impl CameraUniform {
    fn new(width: u32, height: u32) -> Self {
        let half_h = (60.0_f32.to_radians() * 0.5).tan();
        let half_w = (width as f32 / height as f32) * half_h;
        CameraUniform {
            origin:     [0.0, 0.0, 3.0, 0.0],
            lower_left: [-half_w, -half_h, 2.0, 0.0],
            horizontal: [2.0 * half_w, 0.0, 0.0, 0.0],
            vertical:   [0.0, 2.0 * half_h, 0.0, 0.0],
            dims:      [width, height],
            _dims_pad: [0, 0],
        }
    }
}

// ── Frame uniform — BG0 binding 7 across all pipeline BG0s ───────────────────
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct FrameUniform {
    frame:  u32,
    dims:   [u32; 2],
    bounce: u32,
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
    frame_buf:        Buffer,
    #[allow(dead_code)]
    ray_buf:          Buffer,
    ray_gen_pipeline: ComputePipeline,
    ray_gen_bg0:      BindGroup,
    ray_gen_bg1:      BindGroup,

    // Sky mask (Step 9)
    #[allow(dead_code)]
    sky_mask_buf:          Buffer,
    sky_mask_init_pipeline: ComputePipeline,
    sky_mask_init_bg0:      BindGroup,
    sky_mask_init_bg1:      BindGroup,

    // Background supersampling — frame-0 pre-loop, frozen sky pixels
    background_preshader_pipeline: ComputePipeline,
    background_preshader_bg0:      BindGroup,
    background_preshader_bg1:      BindGroup,

    // Background in-loop — per-bounce escaped-ray contribution
    background_shader_pipeline: ComputePipeline,
    background_shader_bg0:      BindGroup,
    background_shader_bg1:      BindGroup,

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
    // Light buffer (Step 7 / B06-4)
    #[allow(dead_code)]
    light_buf: Buffer,

    // Hit record buffer (Step 6b)
    #[allow(dead_code)]
    hit_buf: Buffer,

    // Shading pipelines (Step 6b/6d)
    shade_diffuse_pipeline:  ComputePipeline,
    shade_metallic_pipeline: ComputePipeline,
    shade_glass_pipeline:    ComputePipeline,
    shade_direct_pipeline:   ComputePipeline,
    shade_scene_bg0:         BindGroup,
    shade_direct_bg0:        BindGroup,
    shade_bg1_rw:            BindGroup,  // all material shaders: rays read-write

    // Scratch buffer (B08 — one vec4<f32> per pixel)
    #[allow(dead_code)]
    scratch_buf:  Buffer,
    // Pixel state buffer (B12 — replaces accum_buf; carries PixelState per pixel)
    #[allow(dead_code)]
    pixel_buf:    Buffer,
    // Display texture — written by resolve pass, read by blit
    #[allow(dead_code)]
    display_tex:  Texture,
    // Frame-start clear pass (B08)
    clear_pipeline: ComputePipeline,
    clear_bg0:      BindGroup,
    clear_bg1:      BindGroup,

    // Russian roulette termination pass (B08 Step 5)
    roulette_pipeline: ComputePipeline,
    roulette_bg0:      BindGroup,
    roulette_bg1:      BindGroup,

    intersect_pipeline: ComputePipeline,
    intersect_bg1:      BindGroup,
    scene_bg0:          BindGroup,

    // Accumulate: additive sum (B09)
    accumulate_pipeline: ComputePipeline,
    accum_bg0:           BindGroup,
    accum_bg1:           BindGroup,

    // Variance pass (B12 — per-pixel variance from sq/accum)
    variance_pipeline: ComputePipeline,
    variance_bg0:      BindGroup,
    variance_bg1:      BindGroup,

    // Bloom ray generation (B13)
    bloom_ray_gen_pipeline: ComputePipeline,
    bloom_ray_gen_bg0:      BindGroup,
    bloom_ray_gen_bg1:      BindGroup,
    // Bloom ray buffer (B13 — K=4096 slots × 256 rays × sizeof(Ray) bytes/ray)
    #[allow(dead_code)]
    bloom_slot_buf:         Buffer,
    // Bloom hit buffer (B14 — separate hit records for bloom bounce loop)
    #[allow(dead_code)]
    bloom_hit_buf:          Buffer,
    // Bloom index buffer (B14 — slot → pixel reverse mapping)
    #[allow(dead_code)]
    bloom_index_buf:        Buffer,
    // Bloom scratch buffer (B14 — flat ray-result storage, slot × 256 × vec4<f32>)
    #[allow(dead_code)]
    bloom_scratch_buf:      Buffer,

    // Bloom bounce pipelines (B14)
    bloom_intersect_pipeline: ComputePipeline,
    bloom_roulette_pipeline:  ComputePipeline,
    bloom_diffuse_pipeline:   ComputePipeline,
    bloom_metallic_pipeline:  ComputePipeline,
    bloom_glass_pipeline:     ComputePipeline,
    bloom_direct_pipeline:    ComputePipeline,
    bloom_intersect_bg1:      BindGroup,
    bloom_shade_bg1:          BindGroup,
    bloom_roulette_bg1:       BindGroup,

    // Selection pass (B12 — top-K bloom slot promotion)
    bloom_counter_buf:  Buffer,
    selection_pipeline: ComputePipeline,
    selection_bg0:      BindGroup,
    selection_bg1:      BindGroup,

    // Bloom scratch clear (B14c — zero bloom_scratch_buf each frame before bloom bounce loop)
    clear_bloom_scratch_pipeline: ComputePipeline,
    clear_bloom_scratch_bg1:      BindGroup,

    // Bloom postshader (B14 — collapse 256 rays/slot into scratch_buf)
    bloom_postshader_pipeline: ComputePipeline,
    bloom_postshader_bg0:      BindGroup,
    bloom_postshader_bg1:      BindGroup,

    // Resolve: divide accum by frame count, write display_tex (B09)
    resolve_pipeline: ComputePipeline,
    resolve_bg0:      BindGroup,
    resolve_bg1:      BindGroup,

    // Blit to canvas (Step 4)
    blit_pipeline: RenderPipeline,
    blit_bg0:      BindGroup,

    frame: u32,
    bloom_slot_capacity: u32,

    // Ray counter (HUD Mrays/frame)
    ray_counter_buf:         Buffer,
    ray_counter_staging_buf: Rc<Buffer>,
    last_mrays_frame:        Rc<RefCell<f32>>,
    counter_ready:           Rc<RefCell<bool>>,
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
        let bloom_slot_capacity = (size.width * size.height) / BLOOM_AMPLIFICATION;

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
                label:             Some("Main Device"),
                required_features: Features::empty(),
                // BG0 has 9 storage buffers in the intersect stage (default limit is 8).
                // The adapter reports supporting 10, so this is safe on target hardware.
                // shade_direct is at 8 storage buffers per shader stage (default limit).
                // intersect dropped below the default limit after scratch_buf was removed
                // from its BG1; shade_direct drives the requirement to keep this at 10.
                required_limits: Limits {
                    max_storage_buffers_per_shader_stage: 10,
                    ..Limits::default()
                },
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

        let frame_buf = device.create_buffer(&BufferDescriptor {
            label:              Some("Frame Uniform"),
            size:               std::mem::size_of::<FrameUniform>() as u64,
            usage:              BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sky_mask_buf = device.create_buffer(&BufferDescriptor {
            label:              Some("Sky Mask Buffer"),
            size:               (size.width * size.height) as u64 * 4,
            usage:              BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let (ray_gen_pipeline, ray_gen_bg0, ray_gen_bg1, camera_buf, ray_buf) =
            Self::create_ray_gen(&device, size.width, size.height, &frame_buf, &sky_mask_buf);

        let (bvh_node_buf, tlas_instance_buf, sphere_buf, vertex_buf, geometry_buf, material_buf) =
            Self::create_bvh_buffers(&device);

        let (sky_mask_init_pipeline, sky_mask_init_bg0, sky_mask_init_bg1) =
            Self::create_sky_mask_init(
                &device, &camera_buf, &frame_buf,
                &bvh_node_buf, &tlas_instance_buf, &sphere_buf,
                &sky_mask_buf,
            ).await;

        let light_buf = Self::upload_slice(&device, "Light Buffer", &[LightUniform {
            // position:  [0.0, 0.0, 5.0, 0.0],
            position:  [2.0, 4.0, 2.0, 0.0],
            color:     [1.0, 0.95, 0.88, 0.0],
            intensity: 20.0,
            _pad:      [0.0; 7],
        }]);

        let hit_buf = device.create_buffer(&BufferDescriptor {
            label:              Some("Hit Buffer"),
            size:               (size.width * size.height) as u64
                                    * std::mem::size_of::<HitRecord>() as u64,
            usage:              BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let ray_counter_buf = device.create_buffer(&BufferDescriptor {
            label:              Some("Ray Counter"),
            size:               4,
            usage:              BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ray_counter_staging_buf = Rc::new(device.create_buffer(&BufferDescriptor {
            label:              Some("Ray Counter Staging"),
            size:               4,
            usage:              BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        let (scratch_buf, intersect_pipeline, intersect_bg1, scene_bg0) =
            Self::create_intersect(
                &device, &ray_buf,
                &hit_buf,
                &bvh_node_buf, &tlas_instance_buf, &sphere_buf,
                &vertex_buf, &geometry_buf, &material_buf,
                &ray_counter_buf,
                &frame_buf,
                size.width, size.height,
            ).await;

        let (background_preshader_pipeline, background_preshader_bg0, background_preshader_bg1) =
            Self::create_background_preshader(
                &device, &camera_buf, &frame_buf, &sky_mask_buf, &scratch_buf,
            ).await;

        let (background_shader_pipeline, background_shader_bg0, background_shader_bg1) =
            Self::create_background_shader(
                &device, &frame_buf, &hit_buf, &scratch_buf, &ray_buf,
            ).await;

        let (shade_diffuse_pipeline, shade_metallic_pipeline, shade_glass_pipeline,
             shade_direct_pipeline,
             shade_scene_bg0, shade_direct_bg0, shade_bg1_rw) =
            Self::create_shade_pipelines(
                &device,
                &ray_buf,
                &hit_buf,
                &bvh_node_buf,
                &tlas_instance_buf,
                &sphere_buf,
                &vertex_buf,
                &geometry_buf,
                &material_buf,
                &light_buf,
                &frame_buf,
                &scratch_buf,
            ).await;

        let (clear_pipeline, clear_bg0, clear_bg1) =
            Self::create_clear_pass(&device, &frame_buf, &scratch_buf).await;

        let (roulette_pipeline, roulette_bg0, roulette_bg1) =
            Self::create_roulette_pass(&device, &frame_buf, &ray_buf).await;

        let (accumulate_pipeline, accum_bg0, accum_bg1, pixel_buf) =
            Self::create_accumulate(&device, &frame_buf, &scratch_buf, size.width, size.height).await;

        let bloom_counter_buf = device.create_buffer(&BufferDescriptor {
            label:              Some("Bloom Counter"),
            size:               4,
            usage:              BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bloom_slot_buf = device.create_buffer(&BufferDescriptor {
            label:              Some("Bloom Slot Buffer"),
            size:               bloom_slot_capacity as u64 * BLOOM_AMPLIFICATION as u64 * std::mem::size_of::<Ray>() as u64,
            usage:              BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let bloom_hit_buf = device.create_buffer(&BufferDescriptor {
            label:              Some("Bloom Hit Buffer"),
            size:               bloom_slot_capacity as u64 * BLOOM_AMPLIFICATION as u64 * std::mem::size_of::<HitRecord>() as u64,
            usage:              BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let bloom_index_buf = device.create_buffer(&BufferDescriptor {
            label:              Some("Bloom Index Buffer"),
            size:               bloom_slot_capacity as u64 * 4,
            usage:              BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let bloom_scratch_buf = device.create_buffer(&BufferDescriptor {
            label:              Some("Bloom Scratch Buffer"),
            size:               bloom_slot_capacity as u64 * BLOOM_AMPLIFICATION as u64 * 16,
            usage:              BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let (clear_bloom_scratch_pipeline, clear_bloom_scratch_bg1) =
            Self::create_clear_bloom_scratch(&device, &bloom_scratch_buf).await;

        let (bloom_intersect_pipeline, bloom_roulette_pipeline,
             bloom_diffuse_pipeline, bloom_metallic_pipeline,
             bloom_glass_pipeline, bloom_direct_pipeline,
             bloom_intersect_bg1, bloom_shade_bg1, bloom_roulette_bg1) =
            Self::create_bloom_bounce_pipelines(
                &device, &bloom_slot_buf, &bloom_hit_buf, &bloom_scratch_buf, &ray_counter_buf,
            ).await;

        let (bloom_ray_gen_pipeline, bloom_ray_gen_bg0, bloom_ray_gen_bg1) =
            Self::create_bloom_ray_gen(
                &device, &camera_buf, &frame_buf, &pixel_buf, &bloom_slot_buf,
            ).await;

        let (bloom_postshader_pipeline, bloom_postshader_bg0, bloom_postshader_bg1) =
            Self::create_bloom_postshader(
                &device, &frame_buf, &bloom_index_buf, &bloom_scratch_buf, &scratch_buf, &pixel_buf,
            ).await;

        let (variance_pipeline, variance_bg0, variance_bg1) =
            Self::create_variance_pass(&device, &frame_buf, &pixel_buf).await;

        let (selection_pipeline, selection_bg0, selection_bg1) =
            Self::create_selection_pass(&device, &frame_buf, &pixel_buf, &bloom_counter_buf, &bloom_index_buf).await;

        let (resolve_pipeline, resolve_bg0, resolve_bg1, display_tex, display_tex_view) =
            Self::create_resolve(&device, &frame_buf, &pixel_buf, &sky_mask_buf, size.width, size.height).await;

        let (blit_pipeline, blit_bg0) =
            Self::create_blit(&device, &config, &display_tex_view);

        let last_mrays_frame = Rc::new(RefCell::new(0.0f32));
        let counter_ready    = Rc::new(RefCell::new(true));

        log::info!(
            "B14d ready: {}×{} — bloom_postshader occupancy guard",
            size.width, size.height,
        );

        Ok(Self {
            surface, device, queue, config, size,
            camera_buf, frame_buf, ray_buf, ray_gen_pipeline, ray_gen_bg0, ray_gen_bg1,
            sky_mask_buf, sky_mask_init_pipeline, sky_mask_init_bg0, sky_mask_init_bg1,
            background_preshader_pipeline, background_preshader_bg0, background_preshader_bg1,
            background_shader_pipeline, background_shader_bg0, background_shader_bg1,
            bvh_node_buf, tlas_instance_buf, sphere_buf,
            vertex_buf, geometry_buf, material_buf, light_buf,
            hit_buf,
            shade_diffuse_pipeline, shade_metallic_pipeline, shade_glass_pipeline,
            shade_direct_pipeline,
            shade_scene_bg0, shade_direct_bg0, shade_bg1_rw,
            clear_pipeline, clear_bg0, clear_bg1,
            roulette_pipeline, roulette_bg0, roulette_bg1,
            scratch_buf, pixel_buf, display_tex,
            intersect_pipeline, intersect_bg1, scene_bg0,
            accumulate_pipeline, accum_bg0, accum_bg1,
            variance_pipeline, variance_bg0, variance_bg1,
            bloom_intersect_pipeline, bloom_roulette_pipeline,
            bloom_diffuse_pipeline, bloom_metallic_pipeline,
            bloom_glass_pipeline, bloom_direct_pipeline,
            bloom_intersect_bg1, bloom_shade_bg1, bloom_roulette_bg1,
            bloom_ray_gen_pipeline, bloom_ray_gen_bg0, bloom_ray_gen_bg1,
            bloom_slot_buf, bloom_hit_buf, bloom_index_buf, bloom_scratch_buf,
            bloom_counter_buf, selection_pipeline, selection_bg0, selection_bg1,
            clear_bloom_scratch_pipeline, clear_bloom_scratch_bg1,
            bloom_postshader_pipeline, bloom_postshader_bg0, bloom_postshader_bg1,
            resolve_pipeline, resolve_bg0, resolve_bg1,
            blit_pipeline, blit_bg0,
            frame: 0,
            bloom_slot_capacity,
            ray_counter_buf, ray_counter_staging_buf,
            last_mrays_frame, counter_ready,
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
            // Index 1: clear glass sphere.
            Material {
                base_color:    [1.0, 1.0, 1.0, 1.0],
                emission:      [0.0, 0.0, 0.0, 0.0],
                absorption:    [0.0, 0.0, 0.0, 0.0],
                material_type: MaterialType::Glass,
                ior:           1.3333333,
                // ior:           1.5,
                roughness:     0.0,
                _pad:          0.0,
            },
            // Index 2: warm tan diffuse sphere.
            Material {
                base_color:    [0.72, 0.60, 0.45, 1.0],
                emission:      [0.0, 0.0, 0.0, 0.0],
                absorption:    [0.0, 0.0, 0.0, 0.0],
                material_type: MaterialType::Diffuse,
                ior:           1.0,
                roughness:     0.0,
                _pad:          0.0,
            },
            // Index 3: metallic
            Material {
                base_color:    [1.0, 1.0, 1.0, 1.0],
                emission:      [0.0, 0.0, 0.0, 0.0],
                absorption:    [0.0, 0.0, 0.0, 0.0],
                material_type: MaterialType::Metallic,
                ior:           1.0,
                roughness:     0.0,
                _pad:          0.0,
            },
            // Index 4: glass air-bubble
            Material {
                base_color:    [1.0, 1.0, 1.0, 1.0],
                emission:      [0.0, 0.0, 0.0, 0.0],
                absorption:    [0.0, 0.0, 0.0, 0.0],
                material_type: MaterialType::Glass,
                ior:           1.0,
                roughness:     0.0,
                _pad:          0.0,
            },
        ]);
        (bvh_node_buf, tlas_instance_buf, sphere_buf, vertex_buf, geometry_buf, material_buf)
    }

    fn create_ray_gen(
        device: &Device, width: u32, height: u32, frame_buf: &Buffer, sky_mask_buf: &Buffer,
    ) -> (ComputePipeline, BindGroup, BindGroup, Buffer, Buffer) {
        let bg0_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label:   Some("Ray Gen BG0"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0, visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1, visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let bg1_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label:   Some("Ray Gen BG1"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0, visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
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
            ],
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
            entries: &[
                BindGroupEntry { binding: 0, resource: camera_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: frame_buf.as_entire_binding() },
            ],
        });
        let ray_gen_bg1 = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Ray Gen BG1"), layout: &bg1_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: ray_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: sky_mask_buf.as_entire_binding() },
            ],
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

    async fn create_sky_mask_init(
        device:            &Device,
        camera_buf:        &Buffer,
        frame_buf:         &Buffer,
        bvh_node_buf:      &Buffer,
        tlas_instance_buf: &Buffer,
        sphere_buf:        &Buffer,
        sky_mask_buf:      &Buffer,
    ) -> (ComputePipeline, BindGroup, BindGroup) {
        let bg0_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label:   Some("Sky Mask Init BG0"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0, visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1, visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
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
            ],
        });
        let bg1_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label:   Some("Sky Mask Init BG1"),
            entries: &[BindGroupLayoutEntry {
                binding: 0, visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false, min_binding_size: None,
                },
                count: None,
            }],
        });

        let sky_mask_init_bg0 = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Sky Mask Init BG0"), layout: &bg0_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: camera_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: frame_buf.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: bvh_node_buf.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: tlas_instance_buf.as_entire_binding() },
                BindGroupEntry { binding: 4, resource: sphere_buf.as_entire_binding() },
            ],
        });
        let sky_mask_init_bg1 = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Sky Mask Init BG1"), layout: &bg1_layout,
            entries: &[BindGroupEntry { binding: 0, resource: sky_mask_buf.as_entire_binding() }],
        });

        let src = format!("{}\n{}", include_str!("common_common.wgsl"), include_str!("sky_mask_init.wgsl"));
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label:  Some("Sky Mask Init"),
            source: ShaderSource::Wgsl(std::borrow::Cow::Owned(src)),
        });
        let info = shader.get_compilation_info().await;
        for msg in &info.messages {
            match msg.message_type {
                CompilationMessageType::Error   => log::error!("sky_mask_init.wgsl: {}", msg.message),
                CompilationMessageType::Warning => log::warn!("sky_mask_init.wgsl: {}",  msg.message),
                _ => {}
            }
        }

        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Sky Mask Init Layout"),
            bind_group_layouts: &[&bg0_layout, &bg1_layout],
            push_constant_ranges: &[],
        });
        device.push_error_scope(ErrorFilter::Validation);
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Sky Mask Init"), layout: Some(&layout), module: &shader,
            entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        });
        if let Some(err) = device.pop_error_scope().await {
            log::error!("Sky Mask Init pipeline validation error: {:?}", err);
        }

        (pipeline, sky_mask_init_bg0, sky_mask_init_bg1)
    }

    async fn create_background_preshader(
        device:       &Device,
        camera_buf:   &Buffer,
        frame_buf:    &Buffer,
        sky_mask_buf: &Buffer,
        scratch_buf:  &Buffer,
    ) -> (ComputePipeline, BindGroup, BindGroup) {
        let bg0_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label:   Some("Background Preshader BG0"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0, visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1, visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let bg1_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label:   Some("Background Preshader BG1"),
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
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false, min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let background_preshader_bg0 = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Background Preshader BG0"), layout: &bg0_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: camera_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: frame_buf.as_entire_binding() },
            ],
        });
        let background_preshader_bg1 = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Background Preshader BG1"), layout: &bg1_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: sky_mask_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: scratch_buf.as_entire_binding() },
            ],
        });

        let src = format!("{}\n{}", include_str!("common_common.wgsl"), include_str!("background_preshader.wgsl"));
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label:  Some("Background Preshader"),
            source: ShaderSource::Wgsl(std::borrow::Cow::Owned(src)),
        });
        let info = shader.get_compilation_info().await;
        for msg in &info.messages {
            match msg.message_type {
                CompilationMessageType::Error   => log::error!("background_preshader.wgsl: {}", msg.message),
                CompilationMessageType::Warning => log::warn!("background_preshader.wgsl: {}",  msg.message),
                _ => {}
            }
        }

        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Background Preshader Layout"),
            bind_group_layouts: &[&bg0_layout, &bg1_layout],
            push_constant_ranges: &[],
        });
        device.push_error_scope(ErrorFilter::Validation);
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Background Preshader"), layout: Some(&layout), module: &shader,
            entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        });
        if let Some(err) = device.pop_error_scope().await {
            log::error!("Background Preshader pipeline validation error: {:?}", err);
        }

        (pipeline, background_preshader_bg0, background_preshader_bg1)
    }

    async fn create_background_shader(
        device:      &Device,
        frame_buf:   &Buffer,
        hit_buf:     &Buffer,
        scratch_buf: &Buffer,
        ray_buf:     &Buffer,
    ) -> (ComputePipeline, BindGroup, BindGroup) {
        let bg0_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label:   Some("Background Shader BG0"),
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
            label:   Some("Background Shader BG1"),
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
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false, min_binding_size: None,
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

        let background_shader_bg0 = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Background Shader BG0"), layout: &bg0_layout,
            entries: &[BindGroupEntry { binding: 0, resource: frame_buf.as_entire_binding() }],
        });
        let background_shader_bg1 = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Background Shader BG1"), layout: &bg1_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: hit_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: scratch_buf.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: ray_buf.as_entire_binding() },
            ],
        });

        let src = format!("{}\n{}", include_str!("common_common.wgsl"), include_str!("background_shader.wgsl"));
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label:  Some("Background Shader"),
            source: ShaderSource::Wgsl(std::borrow::Cow::Owned(src)),
        });
        let info = shader.get_compilation_info().await;
        for msg in &info.messages {
            match msg.message_type {
                CompilationMessageType::Error   => log::error!("background_shader.wgsl: {}", msg.message),
                CompilationMessageType::Warning => log::warn!("background_shader.wgsl: {}",  msg.message),
                _ => {}
            }
        }

        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Background Shader Layout"),
            bind_group_layouts: &[&bg0_layout, &bg1_layout],
            push_constant_ranges: &[],
        });
        device.push_error_scope(ErrorFilter::Validation);
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Background Shader"), layout: Some(&layout), module: &shader,
            entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        });
        if let Some(err) = device.pop_error_scope().await {
            log::error!("Background Shader pipeline validation error: {:?}", err);
        }

        (pipeline, background_shader_bg0, background_shader_bg1)
    }

    async fn create_clear_pass(
        device:     &Device,
        frame_buf:  &Buffer,
        scratch_buf: &Buffer,
    ) -> (ComputePipeline, BindGroup, BindGroup) {
        let bg0_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label:   Some("Clear BG0"),
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
            label:   Some("Clear BG1"),
            entries: &[BindGroupLayoutEntry {
                binding: 0, visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false, min_binding_size: None,
                },
                count: None,
            }],
        });
        let clear_bg0 = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Clear BG0"), layout: &bg0_layout,
            entries: &[BindGroupEntry { binding: 0, resource: frame_buf.as_entire_binding() }],
        });
        let clear_bg1 = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Clear BG1"), layout: &bg1_layout,
            entries: &[BindGroupEntry { binding: 0, resource: scratch_buf.as_entire_binding() }],
        });
        let src = format!("{}\n{}", include_str!("common_common.wgsl"), include_str!("clear_pass.wgsl"));
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label:  Some("Clear Pass"),
            source: ShaderSource::Wgsl(std::borrow::Cow::Owned(src)),
        });
        let clear_info = shader.get_compilation_info().await;
        for msg in &clear_info.messages {
            match msg.message_type {
                CompilationMessageType::Error   => log::error!("clear_pass.wgsl: {}", msg.message),
                CompilationMessageType::Warning => log::warn!("clear_pass.wgsl: {}",  msg.message),
                _ => {}
            }
        }
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Clear Layout"),
            bind_group_layouts: &[&bg0_layout, &bg1_layout],
            push_constant_ranges: &[],
        });
        device.push_error_scope(ErrorFilter::Validation);
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Clear Pass"), layout: Some(&layout), module: &shader,
            entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        });
        if let Some(err) = device.pop_error_scope().await {
            log::error!("Clear Pass pipeline validation error: {:?}", err);
        }
        (pipeline, clear_bg0, clear_bg1)
    }

    async fn create_clear_bloom_scratch(
        device:          &Device,
        bloom_scratch_buf: &Buffer,
    ) -> (ComputePipeline, BindGroup) {
        let bg1_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label:   Some("Clear Bloom Scratch BG1"),
            entries: &[BindGroupLayoutEntry {
                binding: 0, visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false, min_binding_size: None,
                },
                count: None,
            }],
        });
        let clear_bloom_scratch_bg1 = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Clear Bloom Scratch BG1"), layout: &bg1_layout,
            entries: &[BindGroupEntry { binding: 0, resource: bloom_scratch_buf.as_entire_binding() }],
        });
        let src = format!("{}\n{}", include_str!("common_common.wgsl"), include_str!("clear_bloom_scratch.wgsl"));
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label:  Some("Clear Bloom Scratch"),
            source: ShaderSource::Wgsl(std::borrow::Cow::Owned(src)),
        });
        let shader_info = shader.get_compilation_info().await;
        for msg in &shader_info.messages {
            match msg.message_type {
                CompilationMessageType::Error   => log::error!("clear_bloom_scratch.wgsl: {}", msg.message),
                CompilationMessageType::Warning => log::warn!("clear_bloom_scratch.wgsl: {}",  msg.message),
                _ => {}
            }
        }
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Clear Bloom Scratch Layout"),
            bind_group_layouts: &[&bg1_layout],
            push_constant_ranges: &[],
        });
        device.push_error_scope(ErrorFilter::Validation);
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Clear Bloom Scratch"), layout: Some(&layout), module: &shader,
            entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        });
        if let Some(err) = device.pop_error_scope().await {
            log::error!("Clear Bloom Scratch pipeline validation error: {:?}", err);
        }
        (pipeline, clear_bloom_scratch_bg1)
    }

    async fn create_roulette_pass(
        device:    &Device,
        frame_buf: &Buffer,
        ray_buf:   &Buffer,
    ) -> (ComputePipeline, BindGroup, BindGroup) {
        let bg0_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label:   Some("Roulette BG0"),
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
            label:   Some("Roulette BG1"),
            entries: &[BindGroupLayoutEntry {
                binding: 0, visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false, min_binding_size: None,
                },
                count: None,
            }],
        });
        let roulette_bg0 = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Roulette BG0"), layout: &bg0_layout,
            entries: &[BindGroupEntry { binding: 0, resource: frame_buf.as_entire_binding() }],
        });
        let roulette_bg1 = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Roulette BG1"), layout: &bg1_layout,
            entries: &[BindGroupEntry { binding: 0, resource: ray_buf.as_entire_binding() }],
        });
        let src = format!("{}\n{}\n{}", include_str!("common_common.wgsl"), include_str!("roulette_variant_canvas.wgsl"), include_str!("roulette_pass.wgsl"));
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label:  Some("Roulette Pass"),
            source: ShaderSource::Wgsl(std::borrow::Cow::Owned(src)),
        });
        let roulette_info = shader.get_compilation_info().await;
        for msg in &roulette_info.messages {
            match msg.message_type {
                CompilationMessageType::Error   => log::error!("roulette_pass.wgsl: {}", msg.message),
                CompilationMessageType::Warning => log::warn!("roulette_pass.wgsl: {}",  msg.message),
                _ => {}
            }
        }
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Roulette Layout"),
            bind_group_layouts: &[&bg0_layout, &bg1_layout],
            push_constant_ranges: &[],
        });
        device.push_error_scope(ErrorFilter::Validation);
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Roulette Pass"), layout: Some(&layout), module: &shader,
            entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        });
        if let Some(err) = device.pop_error_scope().await {
            log::error!("Roulette Pass pipeline validation error: {:?}", err);
        }
        (pipeline, roulette_bg0, roulette_bg1)
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
        ray_counter_buf:   &Buffer,
        frame_buf:         &Buffer,
        width:  u32,
        height: u32,
    ) -> (Buffer, ComputePipeline, BindGroup, BindGroup) {
        let scratch_buf = device.create_buffer(&BufferDescriptor {
            label:              Some("Scratch Buffer"),
            size:               (width * height) as u64 * 16u64,
            usage:              BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

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
                BindGroupLayoutEntry {
                    binding: 7, visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
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
                BindGroupEntry { binding: 7, resource: frame_buf.as_entire_binding() },
            ],
        });

        // BG1 — per-pass: rays (miss termination sentinel), hit records, ray counter.
        // scratch_buf removed — background_shader owns sky-pixel writes on frame 0.
        let bg1_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label:   Some("Intersect BG1"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0, visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false, min_binding_size: None,
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
                BindGroupLayoutEntry {
                    binding: 3, visibility: ShaderStages::COMPUTE,
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
                BindGroupEntry { binding: 2, resource: hit_buf.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: ray_counter_buf.as_entire_binding() },
            ],
        });

        let intersect_src = format!("{}\n{}\n{}\n{}", include_str!("common_common.wgsl"), include_str!("intersect_common.wgsl"), include_str!("intersect_variant_canvas.wgsl"), include_str!("intersect.wgsl"));
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

        (scratch_buf, pipeline, intersect_bg1, scene_bg0)
    }

    async fn create_accumulate(
        device:      &Device,
        frame_buf:   &Buffer,
        scratch_buf: &Buffer,
        width:       u32,
        height:      u32,
    ) -> (ComputePipeline, BindGroup, BindGroup, Buffer) {
        let n = (width * height) as usize;
        let init_pixels: Vec<PixelState> = (0..n).map(|_| PixelState {
            accum:      [0.0; 4],
            sq:         [0.0; 4],
            variance:   0.0,
            bloom_slot: -1,
            _pad:       [0.0; 2],
        }).collect();
        let pixel_buf = device.create_buffer_init(&util::BufferInitDescriptor {
            label:    Some("Pixel State Buffer"),
            contents: bytemuck::cast_slice(&init_pixels),
            usage:    BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        let bg0_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label:   Some("Accumulate BG0"),
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
            label:   Some("Accumulate BG1"),
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
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false, min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let accum_bg0 = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Accumulate BG0"), layout: &bg0_layout,
            entries: &[BindGroupEntry { binding: 0, resource: frame_buf.as_entire_binding() }],
        });
        let accum_bg1 = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Accumulate BG1"), layout: &bg1_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: scratch_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: pixel_buf.as_entire_binding() },
            ],
        });

        let accum_src = format!("{}\n{}", include_str!("common_common.wgsl"), include_str!("accumulate.wgsl"));
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label:  Some("Accumulate"),
            source: ShaderSource::Wgsl(std::borrow::Cow::Owned(accum_src)),
        });
        let accum_info = shader.get_compilation_info().await;
        for msg in &accum_info.messages {
            match msg.message_type {
                CompilationMessageType::Error   => log::error!("accumulate.wgsl: {}", msg.message),
                CompilationMessageType::Warning => log::warn!("accumulate.wgsl: {}",  msg.message),
                _ => {}
            }
        }

        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Accumulate Layout"),
            bind_group_layouts: &[&bg0_layout, &bg1_layout],
            push_constant_ranges: &[],
        });
        device.push_error_scope(ErrorFilter::Validation);
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Accumulate"), layout: Some(&layout), module: &shader,
            entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        });
        if let Some(err) = device.pop_error_scope().await {
            log::error!("Accumulate pipeline validation error: {:?}", err);
        }

        (pipeline, accum_bg0, accum_bg1, pixel_buf)
    }

    async fn create_variance_pass(
        device:    &Device,
        frame_buf: &Buffer,
        pixel_buf: &Buffer,
    ) -> (ComputePipeline, BindGroup, BindGroup) {
        let bg0_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label:   Some("Variance BG0"),
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
            label:   Some("Variance BG1"),
            entries: &[BindGroupLayoutEntry {
                binding: 0, visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false, min_binding_size: None,
                },
                count: None,
            }],
        });

        let variance_bg0 = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Variance BG0"), layout: &bg0_layout,
            entries: &[BindGroupEntry { binding: 0, resource: frame_buf.as_entire_binding() }],
        });
        let variance_bg1 = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Variance BG1"), layout: &bg1_layout,
            entries: &[BindGroupEntry { binding: 0, resource: pixel_buf.as_entire_binding() }],
        });

        let src = format!("{}\n{}", include_str!("common_common.wgsl"), include_str!("variance_pass.wgsl"));
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label:  Some("Variance Pass"),
            source: ShaderSource::Wgsl(std::borrow::Cow::Owned(src)),
        });
        let info = shader.get_compilation_info().await;
        for msg in &info.messages {
            match msg.message_type {
                CompilationMessageType::Error   => log::error!("variance_pass.wgsl: {}", msg.message),
                CompilationMessageType::Warning => log::warn!("variance_pass.wgsl: {}",  msg.message),
                _ => {}
            }
        }

        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Variance Layout"),
            bind_group_layouts: &[&bg0_layout, &bg1_layout],
            push_constant_ranges: &[],
        });
        device.push_error_scope(ErrorFilter::Validation);
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Variance Pass"), layout: Some(&layout), module: &shader,
            entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        });
        if let Some(err) = device.pop_error_scope().await {
            log::error!("Variance Pass pipeline validation error: {:?}", err);
        }

        (pipeline, variance_bg0, variance_bg1)
    }

    async fn create_selection_pass(
        device:            &Device,
        frame_buf:         &Buffer,
        pixel_buf:         &Buffer,
        bloom_counter_buf: &Buffer,
        bloom_index_buf:   &Buffer,
    ) -> (ComputePipeline, BindGroup, BindGroup) {
        let bg0_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label:   Some("Selection BG0"),
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
            label:   Some("Selection BG1"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0, visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false, min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1, visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false, min_binding_size: None,
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

        let selection_bg0 = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Selection BG0"), layout: &bg0_layout,
            entries: &[BindGroupEntry { binding: 0, resource: frame_buf.as_entire_binding() }],
        });
        let selection_bg1 = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Selection BG1"), layout: &bg1_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: pixel_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: bloom_counter_buf.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: bloom_index_buf.as_entire_binding() },
            ],
        });

        let src = format!("{}\n{}", include_str!("common_common.wgsl"), include_str!("selection_pass.wgsl"));
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label:  Some("Selection Pass"),
            source: ShaderSource::Wgsl(std::borrow::Cow::Owned(src)),
        });
        let info = shader.get_compilation_info().await;
        for msg in &info.messages {
            match msg.message_type {
                CompilationMessageType::Error   => log::error!("selection_pass.wgsl: {}", msg.message),
                CompilationMessageType::Warning => log::warn!("selection_pass.wgsl: {}",  msg.message),
                _ => {}
            }
        }

        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Selection Layout"),
            bind_group_layouts: &[&bg0_layout, &bg1_layout],
            push_constant_ranges: &[],
        });
        device.push_error_scope(ErrorFilter::Validation);
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Selection Pass"), layout: Some(&layout), module: &shader,
            entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        });
        if let Some(err) = device.pop_error_scope().await {
            log::error!("Selection Pass pipeline validation error: {:?}", err);
        }

        (pipeline, selection_bg0, selection_bg1)
    }

    async fn create_resolve(
        device:        &Device,
        frame_buf:     &Buffer,
        pixel_buf:     &Buffer,
        sky_mask_buf:  &Buffer,
        width:         u32,
        height:        u32,
    ) -> (ComputePipeline, BindGroup, BindGroup, Texture, TextureView) {
        let display_tex = device.create_texture(&TextureDescriptor {
            label:           Some("Display Texture"),
            size:            Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count:    1,
            dimension:       TextureDimension::D2,
            format:          TextureFormat::Rgba16Float,
            usage:           TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats:    &[],
        });
        let display_tex_view = display_tex.create_view(&TextureViewDescriptor::default());

        let bg0_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label:   Some("Resolve BG0"),
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
            label:   Some("Resolve BG1"),
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
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let resolve_bg0 = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Resolve BG0"), layout: &bg0_layout,
            entries: &[BindGroupEntry { binding: 0, resource: frame_buf.as_entire_binding() }],
        });
        let resolve_bg1 = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Resolve BG1"), layout: &bg1_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: pixel_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&display_tex_view) },
                BindGroupEntry { binding: 2, resource: sky_mask_buf.as_entire_binding() },
            ],
        });

        let resolve_src = format!("{}\n{}", include_str!("common_common.wgsl"), include_str!("resolve.wgsl"));
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label:  Some("Resolve"),
            source: ShaderSource::Wgsl(std::borrow::Cow::Owned(resolve_src)),
        });
        let resolve_info = shader.get_compilation_info().await;
        for msg in &resolve_info.messages {
            match msg.message_type {
                CompilationMessageType::Error   => log::error!("resolve.wgsl: {}", msg.message),
                CompilationMessageType::Warning => log::warn!("resolve.wgsl: {}",  msg.message),
                _ => {}
            }
        }

        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Resolve Layout"),
            bind_group_layouts: &[&bg0_layout, &bg1_layout],
            push_constant_ranges: &[],
        });
        device.push_error_scope(ErrorFilter::Validation);
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Resolve"), layout: Some(&layout), module: &shader,
            entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        });
        if let Some(err) = device.pop_error_scope().await {
            log::error!("Resolve pipeline validation error: {:?}", err);
        }

        (pipeline, resolve_bg0, resolve_bg1, display_tex, display_tex_view)
    }

    fn create_blit(
        device: &Device, config: &SurfaceConfiguration, display_tex_view: &TextureView,
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
            entries: &[BindGroupEntry { binding: 0, resource: BindingResource::TextureView(display_tex_view) }],
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

    async fn create_bloom_ray_gen(
        device:        &Device,
        camera_buf:    &Buffer,
        frame_buf:     &Buffer,
        pixel_buf:     &Buffer,
        bloom_slot_buf: &Buffer,
    ) -> (ComputePipeline, BindGroup, BindGroup) {
        let bg0_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label:   Some("Bloom Ray Gen BG0"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0, visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1, visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let bg1_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label:   Some("Bloom Ray Gen BG1"),
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
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false, min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bloom_ray_gen_bg0 = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Bloom Ray Gen BG0"), layout: &bg0_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: camera_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: frame_buf.as_entire_binding() },
            ],
        });
        let bloom_ray_gen_bg1 = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Bloom Ray Gen BG1"), layout: &bg1_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: pixel_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: bloom_slot_buf.as_entire_binding() },
            ],
        });

        let src = format!("{}\n{}", include_str!("common_common.wgsl"), include_str!("bloom_ray_gen.wgsl"));
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label:  Some("Bloom Ray Gen"),
            source: ShaderSource::Wgsl(std::borrow::Cow::Owned(src)),
        });
        let info = shader.get_compilation_info().await;
        for msg in &info.messages {
            match msg.message_type {
                CompilationMessageType::Error   => log::error!("bloom_ray_gen.wgsl: {}", msg.message),
                CompilationMessageType::Warning => log::warn!("bloom_ray_gen.wgsl: {}",  msg.message),
                _ => {}
            }
        }

        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Bloom Ray Gen Layout"),
            bind_group_layouts: &[&bg0_layout, &bg1_layout],
            push_constant_ranges: &[],
        });
        device.push_error_scope(ErrorFilter::Validation);
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Bloom Ray Gen"), layout: Some(&layout), module: &shader,
            entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        });
        if let Some(err) = device.pop_error_scope().await {
            log::error!("Bloom Ray Gen pipeline validation error: {:?}", err);
        }

        (pipeline, bloom_ray_gen_bg0, bloom_ray_gen_bg1)
    }

    async fn create_shade_pipelines(
        device:            &Device,
        ray_buf:           &Buffer,
        hit_buf:           &Buffer,
        bvh_node_buf:      &Buffer,
        tlas_instance_buf: &Buffer,
        sphere_buf:        &Buffer,
        vertex_buf:        &Buffer,
        geometry_buf:      &Buffer,
        material_buf:      &Buffer,
        light_buf:         &Buffer,
        frame_buf:         &Buffer,
        scratch_buf:       &Buffer,
    ) -> (ComputePipeline, ComputePipeline, ComputePipeline, ComputePipeline,
          BindGroup, BindGroup, BindGroup) {
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
                BindGroupLayoutEntry {
                    binding: 6, visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false, min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 7, visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
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
                BindGroupEntry { binding: 6, resource: light_buf.as_entire_binding() },
                BindGroupEntry { binding: 7, resource: frame_buf.as_entire_binding() },
            ],
        });

        // BG1 — single layout: all material shaders write continuation rays (read_write).
        let buf_entry = |binding: u32, read_only: bool| BindGroupLayoutEntry {
            binding, visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only },
                has_dynamic_offset: false, min_binding_size: None,
            },
            count: None,
        };
        let bg1_rw_entries: [BindGroupLayoutEntry; 3] = [
            buf_entry(0, true),   // hit_records — read-only
            buf_entry(1, false),  // scratch_buf — read_write
            buf_entry(2, false),  // rays — read_write (all shaders write continuation rays)
        ];
        let bg1_rw_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label:   Some("Shade BG1 RW"),
            entries: &bg1_rw_entries,
        });
        let shade_bg1_rw = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Shade BG1 RW"), layout: &bg1_rw_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: hit_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: scratch_buf.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: ray_buf.as_entire_binding() },
            ],
        });

        // Compose common_common + shade_common + mesh_common + <variant_canvas> + shade_<material>
        let common_common    = include_str!("common_common.wgsl");
        let shade_common     = include_str!("shade_common.wgsl");
        let mesh_common      = include_str!("mesh_common.wgsl");
        let canvas_diffuse   = include_str!("shade_diffuse_variant_canvas.wgsl");
        let canvas_metallic  = include_str!("shade_metallic_variant_canvas.wgsl");
        let canvas_glass     = include_str!("shade_glass_variant_canvas.wgsl");
        let canvas_direct    = include_str!("shade_direct_variant_canvas.wgsl");
        let diffuse          = include_str!("shade_diffuse.wgsl");
        let metallic         = include_str!("shade_metallic.wgsl");
        let glass            = include_str!("shade_glass.wgsl");

        let diffuse_src  = format!("{}\n{}\n{}\n{}\n{}", common_common, shade_common, mesh_common, canvas_diffuse,  diffuse);
        let metallic_src = format!("{}\n{}\n{}\n{}\n{}", common_common, shade_common, mesh_common, canvas_metallic, metallic);
        let glass_src    = format!("{}\n{}\n{}\n{}\n{}", common_common, shade_common, mesh_common, canvas_glass,    glass);

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

        let shade_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Shade Layout"),
            bind_group_layouts: &[&bg0_layout, &bg1_rw_layout],
            push_constant_ranges: &[],
        });

        device.push_error_scope(ErrorFilter::Validation);
        let shade_diffuse_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Shade Diffuse"), layout: Some(&shade_layout), module: &diffuse_module,
            entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        });
        if let Some(err) = device.pop_error_scope().await {
            log::error!("Shade Diffuse pipeline validation error: {:?}", err);
        }

        device.push_error_scope(ErrorFilter::Validation);
        let shade_metallic_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Shade Metallic"), layout: Some(&shade_layout), module: &metallic_module,
            entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        });
        if let Some(err) = device.pop_error_scope().await {
            log::error!("Shade Metallic pipeline validation error: {:?}", err);
        }

        device.push_error_scope(ErrorFilter::Validation);
        let shade_glass_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Shade Glass"), layout: Some(&shade_layout), module: &glass_module,
            entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        });
        if let Some(err) = device.pop_error_scope().await {
            log::error!("Shade Glass pipeline validation error: {:?}", err);
        }

        // ── shade_direct: BG0 for shadow ray BVH traversal ───────────────────────
        // Omits bindings 3/4 (vertices, geometry) — shadow rays never touch mesh data.
        let sd_storage_ro = |binding: u32| BindGroupLayoutEntry {
            binding, visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false, min_binding_size: None,
            },
            count: None,
        };
        let shade_direct_bg0_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label:   Some("Shade Direct BG0"),
            entries: &[
                sd_storage_ro(0), sd_storage_ro(1), sd_storage_ro(2),
                sd_storage_ro(5), sd_storage_ro(6),
                BindGroupLayoutEntry {
                    binding: 7, visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let shade_direct_bg0 = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Shade Direct BG0"), layout: &shade_direct_bg0_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: bvh_node_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: tlas_instance_buf.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: sphere_buf.as_entire_binding() },
                BindGroupEntry { binding: 5, resource: material_buf.as_entire_binding() },
                BindGroupEntry { binding: 6, resource: light_buf.as_entire_binding() },
                BindGroupEntry { binding: 7, resource: frame_buf.as_entire_binding() },
            ],
        });

        let direct_src = format!("{}\n{}\n{}\n{}\n{}", common_common, shade_common, mesh_common, canvas_direct, include_str!("shade_direct.wgsl"));
        let direct_module = device.create_shader_module(ShaderModuleDescriptor {
            label:  Some("Shade Direct"),
            source: ShaderSource::Wgsl(std::borrow::Cow::Owned(direct_src)),
        });
        let direct_info = direct_module.get_compilation_info().await;
        for msg in &direct_info.messages {
            match msg.message_type {
                CompilationMessageType::Error   => log::error!("shade_direct: {}", msg.message),
                CompilationMessageType::Warning => log::warn!("shade_direct: {}",  msg.message),
                _ => {}
            }
        }

        let shade_direct_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Shade Direct Layout"),
            bind_group_layouts: &[&shade_direct_bg0_layout, &bg1_rw_layout],
            push_constant_ranges: &[],
        });
        device.push_error_scope(ErrorFilter::Validation);
        let shade_direct_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Shade Direct"), layout: Some(&shade_direct_layout), module: &direct_module,
            entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        });
        if let Some(err) = device.pop_error_scope().await {
            log::error!("Shade Direct pipeline validation error: {:?}", err);
        }

        (shade_diffuse_pipeline, shade_metallic_pipeline, shade_glass_pipeline,
         shade_direct_pipeline,
         shade_scene_bg0, shade_direct_bg0, shade_bg1_rw)
    }

    async fn create_bloom_bounce_pipelines(
        device:            &Device,
        bloom_slot_buf:    &Buffer,
        bloom_hit_buf:     &Buffer,
        bloom_scratch_buf: &Buffer,
        ray_counter_buf:   &Buffer,
    ) -> (ComputePipeline, ComputePipeline, ComputePipeline, ComputePipeline,
          ComputePipeline, ComputePipeline,
          BindGroup, BindGroup, BindGroup) {
        let storage_ro = |binding: u32| BindGroupLayoutEntry {
            binding, visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false, min_binding_size: None,
            },
            count: None,
        };
        let storage_rw = |binding: u32| BindGroupLayoutEntry {
            binding, visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false, min_binding_size: None,
            },
            count: None,
        };
        let uniform_b = |binding: u32| BindGroupLayoutEntry {
            binding, visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false, min_binding_size: None,
            },
            count: None,
        };

        // ── BG0 layouts — re-created to match mainline layouts ──────────────
        let intersect_bg0_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label:   Some("Bloom Intersect BG0"),
            entries: &[
                storage_ro(0), storage_ro(1), storage_ro(2),
                storage_ro(3), storage_ro(4), storage_ro(5),
                uniform_b(7),
            ],
        });
        let shade_bg0_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label:   Some("Bloom Shade BG0"),
            entries: &[
                storage_ro(2), storage_ro(3), storage_ro(4),
                storage_ro(5), storage_ro(6), uniform_b(7),
            ],
        });
        let shade_direct_bg0_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label:   Some("Bloom Direct BG0"),
            entries: &[
                storage_ro(0), storage_ro(1), storage_ro(2),
                storage_ro(5), storage_ro(6), uniform_b(7),
            ],
        });
        let roulette_bg0_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label:   Some("Bloom Roulette BG0"),
            entries: &[uniform_b(0)],
        });

        // ── Bloom BG1 layouts and bind groups ────────────────────────────────
        let bloom_intersect_bg1_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label:   Some("Bloom Intersect BG1"),
            entries: &[storage_rw(0), storage_rw(2), storage_rw(3)],
        });
        let bloom_intersect_bg1 = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Bloom Intersect BG1"), layout: &bloom_intersect_bg1_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: bloom_slot_buf.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: bloom_hit_buf.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: ray_counter_buf.as_entire_binding() },
            ],
        });

        let bloom_shade_bg1_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label:   Some("Bloom Shade BG1"),
            entries: &[storage_ro(0), storage_rw(1), storage_rw(2)],
        });
        let bloom_shade_bg1 = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Bloom Shade BG1"), layout: &bloom_shade_bg1_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: bloom_hit_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: bloom_scratch_buf.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: bloom_slot_buf.as_entire_binding() },
            ],
        });

        let bloom_roulette_bg1_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label:   Some("Bloom Roulette BG1"),
            entries: &[storage_rw(0)],
        });
        let bloom_roulette_bg1 = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Bloom Roulette BG1"), layout: &bloom_roulette_bg1_layout,
            entries: &[BindGroupEntry { binding: 0, resource: bloom_slot_buf.as_entire_binding() }],
        });

        // ── Pipeline layouts ─────────────────────────────────────────────────
        let bloom_intersect_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Bloom Intersect Layout"),
            bind_group_layouts: &[&intersect_bg0_layout, &bloom_intersect_bg1_layout],
            push_constant_ranges: &[],
        });
        let bloom_shade_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Bloom Shade Layout"),
            bind_group_layouts: &[&shade_bg0_layout, &bloom_shade_bg1_layout],
            push_constant_ranges: &[],
        });
        let bloom_direct_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Bloom Direct Layout"),
            bind_group_layouts: &[&shade_direct_bg0_layout, &bloom_shade_bg1_layout],
            push_constant_ranges: &[],
        });
        let bloom_roulette_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Bloom Roulette Layout"),
            bind_group_layouts: &[&roulette_bg0_layout, &bloom_roulette_bg1_layout],
            push_constant_ranges: &[],
        });

        // ── Shader source composition ─────────────────────────────────────────
        let common_common  = include_str!("common_common.wgsl");
        let shade_common   = include_str!("shade_common.wgsl");
        let mesh_common    = include_str!("mesh_common.wgsl");

        let intersect_src = format!("{}\n{}\n{}\n{}",
            common_common,
            include_str!("intersect_common.wgsl"),
            include_str!("intersect_variant_bloom.wgsl"),
            include_str!("intersect.wgsl"));
        let roulette_src = format!("{}\n{}\n{}",
            common_common,
            include_str!("roulette_variant_bloom.wgsl"),
            include_str!("roulette_pass.wgsl"));
        let diffuse_src = format!("{}\n{}\n{}\n{}\n{}",
            common_common, shade_common, mesh_common,
            include_str!("shade_diffuse_variant_bloom.wgsl"),
            include_str!("shade_diffuse.wgsl"));
        let metallic_src = format!("{}\n{}\n{}\n{}\n{}",
            common_common, shade_common, mesh_common,
            include_str!("shade_metallic_variant_bloom.wgsl"),
            include_str!("shade_metallic.wgsl"));
        let glass_src = format!("{}\n{}\n{}\n{}\n{}",
            common_common, shade_common, mesh_common,
            include_str!("shade_glass_variant_bloom.wgsl"),
            include_str!("shade_glass.wgsl"));
        let direct_src = format!("{}\n{}\n{}\n{}\n{}",
            common_common, shade_common, mesh_common,
            include_str!("shade_direct_variant_bloom.wgsl"),
            include_str!("shade_direct.wgsl"));

        // ── Shader modules ────────────────────────────────────────────────────
        let intersect_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Bloom Intersect"), source: ShaderSource::Wgsl(std::borrow::Cow::Owned(intersect_src)),
        });
        for msg in &intersect_module.get_compilation_info().await.messages {
            match msg.message_type {
                CompilationMessageType::Error   => log::error!("bloom_intersect: {}", msg.message),
                CompilationMessageType::Warning => log::warn!("bloom_intersect: {}",  msg.message),
                _ => {}
            }
        }

        let roulette_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Bloom Roulette"), source: ShaderSource::Wgsl(std::borrow::Cow::Owned(roulette_src)),
        });
        for msg in &roulette_module.get_compilation_info().await.messages {
            match msg.message_type {
                CompilationMessageType::Error   => log::error!("bloom_roulette: {}", msg.message),
                CompilationMessageType::Warning => log::warn!("bloom_roulette: {}",  msg.message),
                _ => {}
            }
        }

        let diffuse_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Bloom Diffuse"), source: ShaderSource::Wgsl(std::borrow::Cow::Owned(diffuse_src)),
        });
        for msg in &diffuse_module.get_compilation_info().await.messages {
            match msg.message_type {
                CompilationMessageType::Error   => log::error!("bloom_diffuse: {}", msg.message),
                CompilationMessageType::Warning => log::warn!("bloom_diffuse: {}",  msg.message),
                _ => {}
            }
        }

        let metallic_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Bloom Metallic"), source: ShaderSource::Wgsl(std::borrow::Cow::Owned(metallic_src)),
        });
        for msg in &metallic_module.get_compilation_info().await.messages {
            match msg.message_type {
                CompilationMessageType::Error   => log::error!("bloom_metallic: {}", msg.message),
                CompilationMessageType::Warning => log::warn!("bloom_metallic: {}",  msg.message),
                _ => {}
            }
        }

        let glass_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Bloom Glass"), source: ShaderSource::Wgsl(std::borrow::Cow::Owned(glass_src)),
        });
        for msg in &glass_module.get_compilation_info().await.messages {
            match msg.message_type {
                CompilationMessageType::Error   => log::error!("bloom_glass: {}", msg.message),
                CompilationMessageType::Warning => log::warn!("bloom_glass: {}",  msg.message),
                _ => {}
            }
        }

        let direct_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Bloom Direct"), source: ShaderSource::Wgsl(std::borrow::Cow::Owned(direct_src)),
        });
        for msg in &direct_module.get_compilation_info().await.messages {
            match msg.message_type {
                CompilationMessageType::Error   => log::error!("bloom_direct: {}", msg.message),
                CompilationMessageType::Warning => log::warn!("bloom_direct: {}",  msg.message),
                _ => {}
            }
        }

        // ── Pipelines ─────────────────────────────────────────────────────────
        device.push_error_scope(ErrorFilter::Validation);
        let bloom_intersect_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Bloom Intersect"), layout: Some(&bloom_intersect_layout),
            module: &intersect_module, entry_point: Some("main"),
            compilation_options: Default::default(), cache: None,
        });
        if let Some(err) = device.pop_error_scope().await {
            log::error!("Bloom Intersect pipeline error: {:?}", err);
        }

        device.push_error_scope(ErrorFilter::Validation);
        let bloom_roulette_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Bloom Roulette"), layout: Some(&bloom_roulette_layout),
            module: &roulette_module, entry_point: Some("main"),
            compilation_options: Default::default(), cache: None,
        });
        if let Some(err) = device.pop_error_scope().await {
            log::error!("Bloom Roulette pipeline error: {:?}", err);
        }

        device.push_error_scope(ErrorFilter::Validation);
        let bloom_diffuse_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Bloom Diffuse"), layout: Some(&bloom_shade_layout),
            module: &diffuse_module, entry_point: Some("main"),
            compilation_options: Default::default(), cache: None,
        });
        if let Some(err) = device.pop_error_scope().await {
            log::error!("Bloom Diffuse pipeline error: {:?}", err);
        }

        device.push_error_scope(ErrorFilter::Validation);
        let bloom_metallic_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Bloom Metallic"), layout: Some(&bloom_shade_layout),
            module: &metallic_module, entry_point: Some("main"),
            compilation_options: Default::default(), cache: None,
        });
        if let Some(err) = device.pop_error_scope().await {
            log::error!("Bloom Metallic pipeline error: {:?}", err);
        }

        device.push_error_scope(ErrorFilter::Validation);
        let bloom_glass_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Bloom Glass"), layout: Some(&bloom_shade_layout),
            module: &glass_module, entry_point: Some("main"),
            compilation_options: Default::default(), cache: None,
        });
        if let Some(err) = device.pop_error_scope().await {
            log::error!("Bloom Glass pipeline error: {:?}", err);
        }

        device.push_error_scope(ErrorFilter::Validation);
        let bloom_direct_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Bloom Direct"), layout: Some(&bloom_direct_layout),
            module: &direct_module, entry_point: Some("main"),
            compilation_options: Default::default(), cache: None,
        });
        if let Some(err) = device.pop_error_scope().await {
            log::error!("Bloom Direct pipeline error: {:?}", err);
        }

        (bloom_intersect_pipeline, bloom_roulette_pipeline,
         bloom_diffuse_pipeline, bloom_metallic_pipeline,
         bloom_glass_pipeline, bloom_direct_pipeline,
         bloom_intersect_bg1, bloom_shade_bg1, bloom_roulette_bg1)
    }

    async fn create_bloom_postshader(
        device:            &Device,
        frame_buf:         &Buffer,
        bloom_index_buf:   &Buffer,
        bloom_scratch_buf: &Buffer,
        scratch_buf:       &Buffer,
        pixel_buf:         &Buffer,
    ) -> (ComputePipeline, BindGroup, BindGroup) {
        let bg0_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label:   Some("Bloom Postshader BG0"),
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
            label:   Some("Bloom Postshader BG1"),
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
                        ty: BufferBindingType::Storage { read_only: false },
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
            ],
        });

        let bloom_postshader_bg0 = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Bloom Postshader BG0"), layout: &bg0_layout,
            entries: &[BindGroupEntry { binding: 0, resource: frame_buf.as_entire_binding() }],
        });
        let bloom_postshader_bg1 = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Bloom Postshader BG1"), layout: &bg1_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: bloom_index_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: bloom_scratch_buf.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: scratch_buf.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: pixel_buf.as_entire_binding() },
            ],
        });

        let src = format!("{}\n{}", include_str!("common_common.wgsl"), include_str!("bloom_postshader.wgsl"));
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label:  Some("Bloom Postshader"),
            source: ShaderSource::Wgsl(std::borrow::Cow::Owned(src)),
        });
        let info = shader.get_compilation_info().await;
        for msg in &info.messages {
            match msg.message_type {
                CompilationMessageType::Error   => log::error!("bloom_postshader.wgsl: {}", msg.message),
                CompilationMessageType::Warning => log::warn!("bloom_postshader.wgsl: {}",  msg.message),
                _ => {}
            }
        }

        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Bloom Postshader Layout"),
            bind_group_layouts: &[&bg0_layout, &bg1_layout],
            push_constant_ranges: &[],
        });
        device.push_error_scope(ErrorFilter::Validation);
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Bloom Postshader"), layout: Some(&layout), module: &shader,
            entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        });
        if let Some(err) = device.pop_error_scope().await {
            log::error!("Bloom Postshader pipeline validation error: {:?}", err);
        }

        (pipeline, bloom_postshader_bg0, bloom_postshader_bg1)
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
        let cam = CameraUniform::new(self.size.width, self.size.height);
        self.queue.write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&cam));

        // Reset ray counter before any intersect dispatches this frame.
        self.queue.write_buffer(&self.ray_counter_buf, 0, bytemuck::bytes_of(&0u32));

        let wx = (self.size.width  + 7) / 8;
        let wy = (self.size.height + 7) / 8;

        let output = self.surface.get_current_texture()?;
        let view   = output.texture.create_view(&TextureViewDescriptor::default());

        // write_buffer is flushed on each submit(), so per-bounce frame_data updates
        // require a separate submit per phase — writes issued before a submit are seen
        // by that submit's commands, not by commands in later submits.

        // ── Pre-loop: clear + ray gen (bounce field is unused by both passes) ──
        self.queue.write_buffer(
            &self.frame_buf, 0,
            bytemuck::bytes_of(&FrameUniform {
                frame:  self.frame,
                dims:   [self.size.width, self.size.height],
                bounce: 0,
            }),
        );
        {
            let mut enc = self.device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Pre-Loop Encoder"),
            });
            if self.frame == 0 {
                let mut cpass = enc.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Sky Mask Init"), timestamp_writes: None,
                });
                cpass.set_pipeline(&self.sky_mask_init_pipeline);
                cpass.set_bind_group(0, &self.sky_mask_init_bg0, &[]);
                cpass.set_bind_group(1, &self.sky_mask_init_bg1, &[]);
                cpass.dispatch_workgroups(wx, wy, 1);
            }
            {
                let mut cpass = enc.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Clear Pass"), timestamp_writes: None,
                });
                cpass.set_pipeline(&self.clear_pipeline);
                cpass.set_bind_group(0, &self.clear_bg0, &[]);
                cpass.set_bind_group(1, &self.clear_bg1, &[]);
                cpass.dispatch_workgroups(wx, wy, 1);
            }
            {
                let mut cpass = enc.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Clear Bloom Scratch"), timestamp_writes: None,
                });
                cpass.set_pipeline(&self.clear_bloom_scratch_pipeline);
                cpass.set_bind_group(0, &self.clear_bloom_scratch_bg1, &[]);
                cpass.dispatch_workgroups(self.bloom_slot_capacity, 1, 1);
            }
            if self.frame == 0 {
                let mut cpass = enc.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Background Preshader"), timestamp_writes: None,
                });
                cpass.set_pipeline(&self.background_preshader_pipeline);
                cpass.set_bind_group(0, &self.background_preshader_bg0, &[]);
                cpass.set_bind_group(1, &self.background_preshader_bg1, &[]);
                cpass.dispatch_workgroups(wx, wy, 1);
            }
            {
                let mut cpass = enc.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Ray Gen"), timestamp_writes: None,
                });
                cpass.set_pipeline(&self.ray_gen_pipeline);
                cpass.set_bind_group(0, &self.ray_gen_bg0, &[]);
                cpass.set_bind_group(1, &self.ray_gen_bg1, &[]);
                cpass.dispatch_workgroups(wx, wy, 1);
            }
            {
                let mut cpass = enc.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Bloom Ray Gen"), timestamp_writes: None,
                });
                cpass.set_pipeline(&self.bloom_ray_gen_pipeline);
                cpass.set_bind_group(0, &self.bloom_ray_gen_bg0, &[]);
                cpass.set_bind_group(1, &self.bloom_ray_gen_bg1, &[]);
                cpass.dispatch_workgroups(wx, wy, 1);
            }
            self.queue.submit([enc.finish()]);
        }

        // ── Bounce loop: 8 path-tracing bounces ──────────────────────────────
        for bounce in 0u32..8u32 {
            self.queue.write_buffer(
                &self.frame_buf, 0,
                bytemuck::bytes_of(&FrameUniform {
                    frame:  self.frame,
                    dims:   [self.size.width, self.size.height],
                    bounce,
                }),
            );
            let mut enc = self.device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Bounce Encoder"),
            });
            {
                let mut cpass = enc.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Intersect"), timestamp_writes: None,
                });
                cpass.set_pipeline(&self.intersect_pipeline);
                cpass.set_bind_group(0, &self.scene_bg0, &[]);
                cpass.set_bind_group(1, &self.intersect_bg1, &[]);
                cpass.dispatch_workgroups(wx, wy, 1);
            }
            {
                let mut cpass = enc.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Background Shader"), timestamp_writes: None,
                });
                cpass.set_pipeline(&self.background_shader_pipeline);
                cpass.set_bind_group(0, &self.background_shader_bg0, &[]);
                cpass.set_bind_group(1, &self.background_shader_bg1, &[]);
                cpass.dispatch_workgroups(wx, wy, 1);
            }
            {
                let mut cpass = enc.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Shade Diffuse"), timestamp_writes: None,
                });
                cpass.set_pipeline(&self.shade_diffuse_pipeline);
                cpass.set_bind_group(0, &self.shade_scene_bg0, &[]);
                cpass.set_bind_group(1, &self.shade_bg1_rw, &[]);
                cpass.dispatch_workgroups(wx, wy, 1);
            }
            {
                let mut cpass = enc.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Shade Metallic"), timestamp_writes: None,
                });
                cpass.set_pipeline(&self.shade_metallic_pipeline);
                cpass.set_bind_group(0, &self.shade_scene_bg0, &[]);
                cpass.set_bind_group(1, &self.shade_bg1_rw, &[]);
                cpass.dispatch_workgroups(wx, wy, 1);
            }
            {
                let mut cpass = enc.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Shade Glass"), timestamp_writes: None,
                });
                cpass.set_pipeline(&self.shade_glass_pipeline);
                cpass.set_bind_group(0, &self.shade_scene_bg0, &[]);
                cpass.set_bind_group(1, &self.shade_bg1_rw, &[]);
                cpass.dispatch_workgroups(wx, wy, 1);
            }
            {
                let mut cpass = enc.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Roulette"), timestamp_writes: None,
                });
                cpass.set_pipeline(&self.roulette_pipeline);
                cpass.set_bind_group(0, &self.roulette_bg0, &[]);
                cpass.set_bind_group(1, &self.roulette_bg1, &[]);
                cpass.dispatch_workgroups(wx, wy, 1);
            }
            {
                let mut cpass = enc.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Shade Direct"), timestamp_writes: None,
                });
                cpass.set_pipeline(&self.shade_direct_pipeline);
                cpass.set_bind_group(0, &self.shade_direct_bg0, &[]);
                cpass.set_bind_group(1, &self.shade_bg1_rw, &[]);
                cpass.dispatch_workgroups(wx, wy, 1);
            }
            self.queue.submit([enc.finish()]);
        }

        // ── Bloom bounce loop: 8 bounces through bloom_slot_buf ──────────────
        let bsc = self.bloom_slot_capacity;
        for bounce in 0u32..8u32 {
            self.queue.write_buffer(
                &self.frame_buf, 0,
                bytemuck::bytes_of(&FrameUniform {
                    frame:  self.frame,
                    dims:   [self.size.width, self.size.height],
                    bounce,
                }),
            );
            let mut enc = self.device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Bloom Bounce Encoder"),
            });
            {
                let mut cpass = enc.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Bloom Intersect"), timestamp_writes: None,
                });
                cpass.set_pipeline(&self.bloom_intersect_pipeline);
                cpass.set_bind_group(0, &self.scene_bg0, &[]);
                cpass.set_bind_group(1, &self.bloom_intersect_bg1, &[]);
                cpass.dispatch_workgroups(bsc, 1, 1);
            }
            {
                let mut cpass = enc.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Bloom Shade Diffuse"), timestamp_writes: None,
                });
                cpass.set_pipeline(&self.bloom_diffuse_pipeline);
                cpass.set_bind_group(0, &self.shade_scene_bg0, &[]);
                cpass.set_bind_group(1, &self.bloom_shade_bg1, &[]);
                cpass.dispatch_workgroups(bsc, 1, 1);
            }
            {
                let mut cpass = enc.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Bloom Shade Metallic"), timestamp_writes: None,
                });
                cpass.set_pipeline(&self.bloom_metallic_pipeline);
                cpass.set_bind_group(0, &self.shade_scene_bg0, &[]);
                cpass.set_bind_group(1, &self.bloom_shade_bg1, &[]);
                cpass.dispatch_workgroups(bsc, 1, 1);
            }
            {
                let mut cpass = enc.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Bloom Shade Glass"), timestamp_writes: None,
                });
                cpass.set_pipeline(&self.bloom_glass_pipeline);
                cpass.set_bind_group(0, &self.shade_scene_bg0, &[]);
                cpass.set_bind_group(1, &self.bloom_shade_bg1, &[]);
                cpass.dispatch_workgroups(bsc, 1, 1);
            }
            {
                let mut cpass = enc.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Bloom Roulette"), timestamp_writes: None,
                });
                cpass.set_pipeline(&self.bloom_roulette_pipeline);
                cpass.set_bind_group(0, &self.roulette_bg0, &[]);
                cpass.set_bind_group(1, &self.bloom_roulette_bg1, &[]);
                cpass.dispatch_workgroups(bsc, 1, 1);
            }
            {
                let mut cpass = enc.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Bloom Shade Direct"), timestamp_writes: None,
                });
                cpass.set_pipeline(&self.bloom_direct_pipeline);
                cpass.set_bind_group(0, &self.shade_direct_bg0, &[]);
                cpass.set_bind_group(1, &self.bloom_shade_bg1, &[]);
                cpass.dispatch_workgroups(bsc, 1, 1);
            }
            self.queue.submit([enc.finish()]);
        }

        // ── Bloom postshader: collapse 256 rays/slot into scratch_buf ────────
        {
            let mut enc = self.device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Bloom Postshader Encoder"),
            });
            {
                let mut cpass = enc.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Bloom Postshader"), timestamp_writes: None,
                });
                cpass.set_pipeline(&self.bloom_postshader_pipeline);
                cpass.set_bind_group(0, &self.bloom_postshader_bg0, &[]);
                cpass.set_bind_group(1, &self.bloom_postshader_bg1, &[]);
                cpass.dispatch_workgroups(bsc, 1, 1);
            }
            self.queue.submit([enc.finish()]);
        }

        // ── Post-loop: accumulate + variance + resolve + blit ─────────────────
        {
            let mut enc = self.device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Post-Loop Encoder"),
            });
            {
                let mut cpass = enc.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Accumulate"), timestamp_writes: None,
                });
                cpass.set_pipeline(&self.accumulate_pipeline);
                cpass.set_bind_group(0, &self.accum_bg0, &[]);
                cpass.set_bind_group(1, &self.accum_bg1, &[]);
                cpass.dispatch_workgroups(wx, wy, 1);
            }
            {
                let mut cpass = enc.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Variance Pass"), timestamp_writes: None,
                });
                cpass.set_pipeline(&self.variance_pipeline);
                cpass.set_bind_group(0, &self.variance_bg0, &[]);
                cpass.set_bind_group(1, &self.variance_bg1, &[]);
                cpass.dispatch_workgroups(wx, wy, 1);
            }
            self.queue.submit([enc.finish()]);
        }
        // Reset bloom counter before selection so slot indices start from 0 each frame.
        self.queue.write_buffer(&self.bloom_counter_buf, 0, bytemuck::bytes_of(&0u32));
        {
            let mut enc = self.device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Selection Encoder"),
            });
            {
                let mut cpass = enc.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Selection Pass"), timestamp_writes: None,
                });
                cpass.set_pipeline(&self.selection_pipeline);
                cpass.set_bind_group(0, &self.selection_bg0, &[]);
                cpass.set_bind_group(1, &self.selection_bg1, &[]);
                cpass.dispatch_workgroups(wx, wy, 1);
            }
            self.queue.submit([enc.finish()]);
        }
        {
            let mut enc = self.device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Resolve Encoder"),
            });
            {
                let mut cpass = enc.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Resolve"), timestamp_writes: None,
                });
                cpass.set_pipeline(&self.resolve_pipeline);
                cpass.set_bind_group(0, &self.resolve_bg0, &[]);
                cpass.set_bind_group(1, &self.resolve_bg1, &[]);
                cpass.dispatch_workgroups(wx, wy, 1);
            }
            {
                let mut rpass = enc.begin_render_pass(&RenderPassDescriptor {
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
            self.queue.submit([enc.finish()]);
        }

        // After post-loop: copy ray counter to staging and read back asynchronously.
        // Skipped if the previous frame's map hasn't completed yet (one frame of latency is fine).
        if *self.counter_ready.borrow() {
            *self.counter_ready.borrow_mut() = false;
            let mut enc = self.device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Counter Readback"),
            });
            enc.copy_buffer_to_buffer(
                &self.ray_counter_buf, 0,
                &*self.ray_counter_staging_buf, 0,
                4,
            );
            self.queue.submit([enc.finish()]);

            let staging  = Rc::clone(&self.ray_counter_staging_buf);
            let mrays    = Rc::clone(&self.last_mrays_frame);
            let ready    = Rc::clone(&self.counter_ready);
            self.ray_counter_staging_buf.slice(..).map_async(MapMode::Read, move |result| {
                if result.is_ok() {
                    let view  = staging.slice(..).get_mapped_range();
                    let count = u32::from_le_bytes([view[0], view[1], view[2], view[3]]);
                    drop(view);
                    staging.unmap();
                    *mrays.borrow_mut() = count as f32 / 1_000_000.0;
                }
                *ready.borrow_mut() = true;
            });
        }

        self.frame += 1;

        // Expose frame counter and Mrays/frame to the JS HUD overlay.
        #[cfg(target_arch = "wasm32")]
        if let Some(window) = web_sys::window() {
            let _ = js_sys::Reflect::set(
                window.as_ref(),
                &wasm_bindgen::JsValue::from_str("beamFrame"),
                &wasm_bindgen::JsValue::from_f64(self.frame as f64),
            );
            let _ = js_sys::Reflect::set(
                window.as_ref(),
                &wasm_bindgen::JsValue::from_str("beamMrays"),
                &wasm_bindgen::JsValue::from_f64(*self.last_mrays_frame.borrow() as f64),
            );
        }

        output.present();
        Ok(())
    }
}
