# wgpu + WASM Starter

A minimal Rust project that renders via **WebGPU** in the browser using **wgpu** and **wasm-bindgen**.

## Project layout

```
.
├── Cargo.toml
├── build.sh
├── src/
│   ├── lib.rs        ← WASM entry point (#[wasm_bindgen(start)])
│   ├── app.rs        ← winit event loop
│   ├── gpu.rs        ← wgpu device / surface / pipeline
│   └── shader.wgsl   ← WGSL vertex + fragment shader
└── web/
    └── index.html    ← host page (canvas + ES module import)
```

## Prerequisites

```bash
# 1. Rust (stable)
curl https://sh.rustup.rs -sSf | sh

# 2. WASM target
rustup target add wasm32-unknown-unknown

# 3. wasm-pack
cargo install wasm-pack

# 4. A local HTTP server (pick one)
cargo install basic-http-server
# — or —
pip install httpserver
```

> **Browser requirement:** Chrome 113+ or Edge 113+ with WebGPU enabled.  
> Firefox requires the `dom.webgpu.enabled` flag (nightly recommended).

## Build & run

```bash
chmod +x build.sh
./build.sh
# Open http://localhost:8080
```

Or manually:

```bash
wasm-pack build --target web --out-dir web/pkg --release
cd web && python3 -m http.server 8080
```

## What you get

On success you'll see a dark background with an **RGB triangle** rendered entirely through WebGPU. From here you can:

- Edit `src/shader.wgsl` to change the GPU program.
- Add vertex buffers / uniform buffers in `gpu.rs`.
- Use `wasm-bindgen` in `lib.rs` to expose Rust functions to JS.

## Key crate versions

| Crate | Version | Notes |
|---|---|---|
| `wgpu` | 22 | `webgpu` feature required for browser target |
| `wasm-bindgen` | 0.2 | JS ↔ Rust bridge |
| `winit` | 0.30 | `web-sys` feature for canvas attachment |
| `console_error_panic_hook` | 0.1 | Readable panics in browser console |
| `console_log` | 1 | `log::info!()` → browser console |

## Troubleshooting

| Symptom | Fix |
|---|---|
| Black screen / no triangle | Open DevTools → Console for errors |
| `WebGPU is not supported` | Use Chrome 113+; check `chrome://flags/#enable-unsafe-webgpu` |
| `No adapter found` | Ensure hardware acceleration is enabled in browser settings |
| CORS error on script import | You **must** serve via HTTP, not `file://` |
