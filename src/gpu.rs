use anyhow::{anyhow, Result};
use bytemuck::{Pod, Zeroable};
use std::rc::Rc;
use wgpu::*;
use winit::{dpi::PhysicalSize, window::Window};

// ── Camera uniform — mirrors WGSL `struct Camera` in ray_gen.wgsl ─────────────
// Layout: five vec4s = 80 bytes. Must stay in sync with the WGSL struct.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CameraUniform {
    origin:     [f32; 4],  // .xyz = camera position
    lower_left: [f32; 4],  // .xyz = image-plane lower-left corner
    horizontal: [f32; 4],  // .xyz = image-plane horizontal span
    vertical:   [f32; 4],  // .xyz = image-plane vertical span
    dims:       [u32; 4],  // [0]=width [1]=height [2]=frame [3]=pad
}

impl CameraUniform {
    fn new(width: u32, height: u32, frame: u32) -> Self {
        // Pinhole: origin at (0,0,3), looking toward origin, 60° vertical FOV,
        // forward = -Z, right = +X, up = +Y.
        let half_h = (60.0_f32.to_radians() * 0.5).tan();
        let half_w = (width as f32 / height as f32) * half_h;

        // lower_left = origin + forward(0,0,-1) - half_w*right - half_h*up
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
    surface:          Surface<'static>,
    device:           Device,
    queue:            Queue,
    config:           SurfaceConfiguration,
    pub size:         PhysicalSize<u32>,

    // Raster pipeline (RGB triangle — scaffold, replaced in Step 4)
    render_pipeline:  RenderPipeline,

    // Ray generation (Step 3)
    camera_buf:       Buffer,
    ray_buf:          Buffer,
    ray_gen_pipeline: ComputePipeline,
    ray_gen_bg0:      BindGroup,
    ray_gen_bg1:      BindGroup,
    frame:            u32,
}

impl GpuState {
    pub async fn new(window: Rc<Window>) -> Result<Self> {
        let size = {
            #[cfg(target_arch = "wasm32")]
            {
                use winit::dpi::PhysicalSize;
                let canvas = web_sys::window()
                    .unwrap()
                    .document()
                    .unwrap()
                    .get_element_by_id("canvas")
                    .unwrap();
                let canvas: web_sys::HtmlCanvasElement =
                    wasm_bindgen::JsCast::dyn_into(canvas).unwrap();
                PhysicalSize::new(canvas.width(), canvas.height())
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                window.inner_size()
            }
        };

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
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
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

        let render_pipeline = Self::create_render_pipeline(&device, &config);

        let (ray_gen_pipeline, ray_gen_bg0, ray_gen_bg1, camera_buf, ray_buf) =
            Self::create_ray_gen(&device, size.width, size.height);

        log::info!(
            "Ray gen ready: {}×{} = {} rays ({} KB)",
            size.width, size.height,
            size.width * size.height,
            size.width * size.height * 32 / 1024,
        );

        Ok(Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            camera_buf,
            ray_buf,
            ray_gen_pipeline,
            ray_gen_bg0,
            ray_gen_bg1,
            frame: 0,
        })
    }

    fn create_render_pipeline(device: &Device, config: &SurfaceConfiguration) -> RenderPipeline {
        let shader = device.create_shader_module(include_wgsl!("shader.wgsl"));
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label:                Some("Raster Layout"),
            bind_group_layouts:   &[],
            push_constant_ranges: &[],
        });
        device.create_render_pipeline(&RenderPipelineDescriptor {
            label:  Some("Raster Pipeline"),
            layout: Some(&layout),
            vertex: VertexState {
                module:              &shader,
                entry_point:         Some("vs_main"),
                buffers:             &[],
                compilation_options: Default::default(),
            },
            fragment: Some(FragmentState {
                module:              &shader,
                entry_point:         Some("fs_main"),
                targets:             &[Some(ColorTargetState {
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
        })
    }

    fn create_ray_gen(
        device: &Device,
        width: u32,
        height: u32,
    ) -> (ComputePipeline, BindGroup, BindGroup, Buffer, Buffer) {
        let bg0_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label:   Some("Ray Gen BG0"),
            entries: &[BindGroupLayoutEntry {
                binding:    0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty:                 BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size:   None,
                },
                count: None,
            }],
        });

        let bg1_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label:   Some("Ray Gen BG1"),
            entries: &[BindGroupLayoutEntry {
                binding:    0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty:                 BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size:   None,
                },
                count: None,
            }],
        });

        let camera_buf = device.create_buffer(&BufferDescriptor {
            label:              Some("Camera Uniform"),
            size:               std::mem::size_of::<CameraUniform>() as u64,
            usage:              BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let ray_buf = device.create_buffer(&BufferDescriptor {
            label:              Some("Ray Buffer"),
            size:               (width * height * 32) as u64,  // 32 bytes per Ray
            usage:              BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let ray_gen_bg0 = device.create_bind_group(&BindGroupDescriptor {
            label:   Some("Ray Gen BG0"),
            layout:  &bg0_layout,
            entries: &[BindGroupEntry {
                binding:  0,
                resource: camera_buf.as_entire_binding(),
            }],
        });

        let ray_gen_bg1 = device.create_bind_group(&BindGroupDescriptor {
            label:   Some("Ray Gen BG1"),
            layout:  &bg1_layout,
            entries: &[BindGroupEntry {
                binding:  0,
                resource: ray_buf.as_entire_binding(),
            }],
        });

        let shader = device.create_shader_module(include_wgsl!("ray_gen.wgsl"));
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label:                Some("Ray Gen Layout"),
            bind_group_layouts:   &[&bg0_layout, &bg1_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label:               Some("Ray Gen Pipeline"),
            layout:              Some(&pipeline_layout),
            module:              &shader,
            entry_point:         Some("main"),
            compilation_options: Default::default(),
            cache:               None,
        });

        (pipeline, ray_gen_bg0, ray_gen_bg1, camera_buf, ray_buf)
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

        // Ray generation compute pass
        {
            let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label:            Some("Ray Gen"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.ray_gen_pipeline);
            cpass.set_bind_group(0, &self.ray_gen_bg0, &[]);
            cpass.set_bind_group(1, &self.ray_gen_bg1, &[]);
            cpass.dispatch_workgroups(
                (self.size.width  + 7) / 8,
                (self.size.height + 7) / 8,
                1,
            );
        }

        // Raster pass — RGB triangle scaffold, replaced in Step 4
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label:                    Some("Raster Pass"),
                color_attachments:        &[Some(RenderPassColorAttachment {
                    view:           &view,
                    resolve_target: None,
                    depth_slice:    None,
                    ops: Operations {
                        load:  LoadOp::Clear(Color { r: 0.05, g: 0.05, b: 0.1, a: 1.0 }),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes:         None,
                occlusion_query_set:      None,
            });
            pass.set_pipeline(&self.render_pipeline);
            pass.draw(0..3, 0..1);
        }

        self.frame += 1;
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }
}
