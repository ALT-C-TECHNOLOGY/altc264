#!/bin/sh
# The threaded WebAssembly FFmpeg driver (MT=1 build-ffmpeg-wasm.sh) under
# Node, run like the native driver (for bench.sh).
exec node "${FFMPEG_WASM:-work/bench/bench-ffmpeg-wasm-mt.js}" "$@"
