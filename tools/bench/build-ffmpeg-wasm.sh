#!/bin/sh
# Builds FFmpeg's h264 decoder and the decode-only timing driver for
# WebAssembly with Emscripten (single-threaded, no assembly, SIMD128 from the
# compiler, as in-browser FFmpeg ports are), into DIR (default: work/bench).
# Needs emcc on PATH (Emscripten SDK). Run with Node:
#   node DIR/bench-ffmpeg-wasm.js CLIP.264 --base-only --threads 1
set -eu
TAG=n9.0.2
S=$(cd "$(dirname "$0")" && pwd)
D=${1:-$S/../../work/bench}
J=$(getconf _NPROCESSORS_ONLN)
command -v emcc >/dev/null || { echo "emcc not found: install and activate the Emscripten SDK" >&2; exit 1; }
mkdir -p "$D" && D=$(cd "$D" && pwd) && cd "$D"
[ -d ffmpeg-$TAG-wasm ] || git clone -q --depth 1 -b $TAG https://git.ffmpeg.org/ffmpeg.git ffmpeg-$TAG-wasm
cd ffmpeg-$TAG-wasm
emconfigure ./configure --prefix="$D/install-$TAG-wasm" --target-os=none --arch=x86_32 \
	--enable-cross-compile --disable-asm --disable-inline-asm --disable-stripping \
	--cc=emcc --cxx=em++ --ar=emar --ranlib=emranlib --nm=emnm \
	--disable-everything --disable-programs --disable-doc --disable-network --disable-autodetect \
	--disable-swresample --disable-swscale --disable-avformat --disable-avfilter --disable-avdevice \
	--disable-pthreads --disable-w32threads --disable-os2threads \
	--enable-decoder=h264 --enable-parser=h264 --enable-static --disable-shared \
	--extra-cflags="-O3 -msimd128" > "$D/configure-$TAG-wasm.log" 2>&1
emmake make -s -j"$J" > "$D/make-$TAG-wasm.log" 2>&1
make -s install > /dev/null
cd ..
emcc -O3 -msimd128 -std=gnu11 -I"install-$TAG-wasm/include" "$S/ffmpeg-driver.c" \
	"install-$TAG-wasm/lib/libavcodec.a" "install-$TAG-wasm/lib/libavutil.a" -lm \
	-sALLOW_MEMORY_GROWTH=1 -sNODERAWFS=1 -sEXIT_RUNTIME=1 -sENVIRONMENT=node -o bench-ffmpeg-wasm.js
echo "built $D/bench-ffmpeg-wasm.js (FFmpeg $TAG, WebAssembly)"
