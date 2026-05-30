use std::cell::RefCell;
use std::rc::Rc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

#[cfg(target_arch = "wasm32")]
use winit::platform::web::WindowAttributesExtWebSys;

use crate::gpu::GpuState;

#[derive(Default)]
struct App {
    window: Option<Rc<Window>>,
    gpu: Rc<RefCell<Option<GpuState>>>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() { return; }

        let attrs = Window::default_attributes()
            .with_title("wgpu + WASM");

        // On WASM, attach to the existing <canvas_id="canvas"> in the index.html
        #[cfg(target_arch = "wasm32")]
        let attrs = {
            use wasm_bindgen::JsCast;
            let canvas = web_sys::window()
                .unwrap()
                .document()
                .unwrap()
                .get_element_by_id("canvas")
                .expect("No <canvas id='canvas'> in index.html")
                .dyn_into::<web_sys::HtmlCanvasElement>()
                .unwrap();
            attrs.with_canvas(Some(canvas))
        };

        let window = Rc::new(
            event_loop
                .create_window(attrs)
                .unwrap());
        self.window = Some(window.clone());

        // Spawn async GPU init — hands control back to JS event loop
        // while wgpu negotiates the adapter/device asynchronously
        let gpu_slot = self.gpu.clone();
        wasm_bindgen_futures::spawn_local(async move {
            match GpuState::new(window).await {
                Ok(gpu) => {
                    *gpu_slot.borrow_mut() = Some(gpu);
                    log::info!("GPU ready");
                }
                Err(e) => log::error!("GPU init failed: {:?}", e),
            }
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(gpu) = self.gpu.borrow_mut().as_mut() {
                    gpu.resize(size);
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(gpu) = self.gpu.borrow_mut().as_mut() {
                    match gpu.render() {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost) => gpu.reconfigure(),
                        Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                        Err(e) => log::error!("Render error: {:?}", e),
                    }
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

pub fn run() -> anyhow::Result<()> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::default();
    event_loop.run_app(&mut app)?;
    
    Ok(())
}