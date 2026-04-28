#!/usr/bin/env bash
set -euo pipefail

# ── Prerequisites ─────────────────────────────────────────────────────────────
# cargo install wasm-pack
# cargo install basic-http-server   (or use: python3 -m http.server 8080)

# ── Build ─────────────────────────────────────────────────────────────────────
echo "→ Building WASM package..."
wasm-pack build \
  --target web \
  --out-dir web/pkg \
  --release

echo "→ Build complete. Output: web/pkg/"

# ── Serve ─────────────────────────────────────────────────────────────────────
echo "→ Serving on http://localhost:9000 ..."
cd web
basic-http-server --addr 127.0.0.1:9000 .

# Alternative (no install needed):
# python3 -m http.server 8080
