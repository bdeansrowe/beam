use wasm_bindgen::prelude::*;

mod app;
mod bvh;
mod gpu;

// Expose a clean entry point to JavaScript
#[wasm_bindgen(start)]
pub fn start() {
    // Redirect panics to the browser console
    console_error_panic_hook::set_once();

    // Init logging → browser console
    console_log::init_with_level(log::Level::Debug).expect("Failed to init logger");

    log::info!("WASM module loaded");

    // Kick off the run loop
    app::run().unwrap();
}
