#!/bin/sh
# Builds FFmpeg's h264 decoder and the decode-only timing driver for
# WebAssembly with Emscripten (no assembly, SIMD128 from the compiler, as
# in-browser FFmpeg ports are), into DIR (default: work/bench).
# MT=1 builds the threaded driver (bench-ffmpeg-wasm-mt.js).
# Needs emcc on PATH (Emscripten SDK). Run with Node:
#   node DIR/bench-ffmpeg-wasm.js CLIP.264 --base-only --threads 1
#   node DIR/bench-ffmpeg-wasm-mt.js CLIP.264 --base-only --threads 8
set -eu
TAG=n9.0.2
S=$(cd "$(dirname "$0")" && pwd)
D=${1:-$S/../../work/bench}
J=$(getconf _NPROCESSORS_ONLN)
if [ "${MT:-0}" = 1 ]; then
	V=wasm-mt
	THREADS="--enable-pthreads"
	FLAGS="-pthread"
	LINK="-pthread -sPTHREAD_POOL_SIZE=16 -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=4GB"
else
	V=wasm
	THREADS="--disable-pthreads"
	FLAGS=""
	LINK="-sALLOW_MEMORY_GROWTH=1"
fi
command -v emcc >/dev/null || { echo "emcc not found: install and activate the Emscripten SDK" >&2; exit 1; }
mkdir -p "$D" && D=$(cd "$D" && pwd) && cd "$D"
[ -d ffmpeg-$TAG-$V ] || git clone -q --depth 1 -b $TAG https://git.ffmpeg.org/ffmpeg.git ffmpeg-$TAG-$V
cd ffmpeg-$TAG-$V
emconfigure ./configure --prefix="$D/install-$TAG-$V" --target-os=none --arch=x86_32 \
	--enable-cross-compile --disable-asm --disable-inline-asm --disable-stripping \
	--cc=emcc --cxx=em++ --ar=emar --ranlib=emranlib --nm=emnm \
	--disable-everything --disable-programs --disable-doc --disable-network --disable-autodetect \
	--disable-swresample --disable-swscale --disable-avformat --disable-avfilter --disable-avdevice \
	$THREADS --disable-w32threads --disable-os2threads \
	--enable-decoder=h264 --enable-parser=h264 --enable-static --disable-shared \
	--extra-cflags="-O3 -msimd128 $FLAGS" --extra-ldflags="$FLAGS" > "$D/configure-$TAG-$V.log" 2>&1
emmake make -s -j"$J" > "$D/make-$TAG-$V.log" 2>&1
make -s install > /dev/null
cd ..
emcc -O3 -msimd128 $FLAGS -std=gnu11 -I"install-$TAG-$V/include" "$S/ffmpeg-driver.c" \
	"install-$TAG-$V/lib/libavcodec.a" "install-$TAG-$V/lib/libavutil.a" -lm \
	$LINK -sNODERAWFS=1 -sEXIT_RUNTIME=1 -sENVIRONMENT=node -o bench-ffmpeg-$V.js
echo "built $D/bench-ffmpeg-$V.js (FFmpeg $TAG, WebAssembly)"
