#!/bin/sh
# Builds FFmpeg's h264 decoder (static, with its assembly) and the decode-only
# timing driver into DIR (default: work/bench). See BENCHMARKS.md.
#   tools/bench/build-ffmpeg.sh [DIR]
set -eu
TAG=n9.0.2
S=$(cd "$(dirname "$0")" && pwd)
D=${1:-$S/../../work/bench}
J=$(getconf _NPROCESSORS_ONLN)
mkdir -p "$D" && D=$(cd "$D" && pwd) && cd "$D"
[ -d ffmpeg-$TAG ] || git clone -q --depth 1 -b $TAG https://git.ffmpeg.org/ffmpeg.git ffmpeg-$TAG
cd ffmpeg-$TAG
./configure --prefix="$D/install-$TAG" --disable-everything --disable-programs \
	--disable-doc --disable-network --disable-autodetect --disable-swresample --disable-swscale \
	--disable-avformat --disable-avfilter --disable-avdevice --enable-decoder=h264 \
	--enable-parser=h264 --enable-pthreads --enable-static --disable-shared > "$D/configure-$TAG.log" 2>&1
make -s -j"$J" > "$D/make-$TAG.log" 2>&1
make -s install > /dev/null
cd ..
cc -O2 -std=gnu11 -Wall -I"install-$TAG/include" "$S/ffmpeg-driver.c" \
	"install-$TAG/lib/libavcodec.a" "install-$TAG/lib/libavutil.a" -lm -lpthread -o bench-ffmpeg
echo "built $D/bench-ffmpeg (FFmpeg $TAG)"
