#!/bin/sh
# Makes the MVC benchmark clip bbb3d-mvc-10s.264: 10 s (7:04 to 7:14, a busy
# action scene) of the Blender Foundation's stereo Big Buck Bunny (CC-BY 3.0, peach.blender.org),
# left eye as the base view and right eye inter-view predicted, encoded with
# JMVC 8.5, the ITU-T reference MVC encoder. See BENCHMARKS.md.
#   tools/build-reference.sh            # once: JMVC encoder and assembler
#   tools/bench/make-mvc-sample.sh [DIR]
# Needs curl, unzip and ffmpeg (to unpack the source frames). The clip is also attached to the GitHub release.
set -eu
S=$(cd "$(dirname "$0")" && pwd)
BIN=$S/../../third_party/install/bin
D=${1:-$S/../../work/mvc-sample}
ZIP=bbb_sunflower_1080p_30fps_stereo_abl.mp4.zip
ZIP_SHA256=e7f103cf7ac21ed9497e3d2628680e29e31186c8994ae1a7cc97f52467c4819c
OUT_SHA256=981c8576e2bd1d6d0d0a8bd2d4d1ef0e3d01fefca12999189ab3c9f59950a5d6
sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }
for t in jmvc-encoder jmvc-assembler; do
	[ -x "$BIN/$t" ] || { echo "$BIN/$t missing: run tools/build-reference.sh" >&2; exit 1; }
done
mkdir -p "$D" && cd "$D"
[ -f $ZIP ] || curl -fL -o $ZIP "https://download.blender.org/demo/movies/BBB/$ZIP"
[ "$(sha256 $ZIP)" = $ZIP_SHA256 ] || { echo "$ZIP: unexpected checksum" >&2; exit 1; }
unzip -o -q $ZIP
# The source stacks the eyes, left on top (1920x2160). Frames 12720 to 13019
# (7:04 to 7:14 at 30 fps), chosen by frame number so the cut is exact.
SRC=${ZIP%.zip}
CUT="select=between(n\,12720\,13019)"
ffmpeg -nostdin -loglevel error -y -i "$SRC" -vf "$CUT,crop=1920:1080:0:0" -fps_mode passthrough -pix_fmt yuv420p -f rawvideo bbb3d_0.yuv
ffmpeg -nostdin -loglevel error -y -i "$SRC" -vf "$CUT,crop=1920:1080:0:1080" -fps_mode passthrough -pix_fmt yuv420p -f rawvideo bbb3d_1.yuv
# One view per encoder run, then the assembler interleaves them.
cp "$S/mvc-sample.cfg" .
"$BIN/jmvc-encoder" -vf mvc-sample.cfg 0 > encode-view0.log
"$BIN/jmvc-encoder" -vf mvc-sample.cfg 1 > encode-view1.log
printf 'OutputFile bbb3d-mvc-10s.264\nNumberOfViews 2\nInputFile0 bbb3d_0.264\nInputFile1 bbb3d_1.264\n' > assembler.cfg
"$BIN/jmvc-assembler" -vf assembler.cfg > assemble.log
got=$(sha256 bbb3d-mvc-10s.264)
echo "$D/bbb3d-mvc-10s.264  sha256 $got"
if [ "$got" = "$OUT_SHA256" ]; then
	echo "identical to the published clip"
else
	echo "differs from the published clip ($OUT_SHA256): a different compiler or platform can change the encoder's decisions; the stream is still valid MVC" >&2
fi
