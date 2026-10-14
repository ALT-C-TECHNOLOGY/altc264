#!/bin/sh
# Builds the reference decoders into third_party/install/bin: JM (ldecod),
# the ITU-T/ISO reference, and JMVC for MVC streams with more than two
# views. Only `cargo xtask conformance-manifest`, `conformance-run` and
# `verify` need them; `conformance-check` does not. Works on macOS and
# Linux (x86_64, aarch64). Re-run after changing a pin.
set -eu

JM_TAG=JM-19.1
JMVC_TAG=JMVC_8_5

root=$(cd "$(dirname "$0")/.." && pwd)
tp="$root/third_party"
inst="$tp/install"
jobs=$(getconf _NPROCESSORS_ONLN)
mkdir -p "$tp" "$inst/bin"

fetch() { # dir url rev
	[ -d "$tp/$1" ] || git clone -q "$2" "$tp/$1"
	git -C "$tp/$1" fetch -q --tags origin "$3" 2>/dev/null || true
	git -C "$tp/$1" checkout -q "$3"
}

echo "== jm (reference decoder, ldecod only)"
[ -d "$tp/jm" ] && git -C "$tp/jm" checkout -q -- CMakeLists.txt source/app/ldecod/mbuffer.c # undo our patches before switching pins
fetch jm https://vcgit.hhi.fraunhofer.de/jvet/JM.git "$JM_TAG"
# Upstream forces -msse4.1 on every Unix target and -Werror on clang; neither
# builds on arm64 with current compilers. Both patches are idempotent.
sed -i.orig -e 's/^if( UNIX OR MINGW )$/if( (UNIX OR MINGW) AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86|AMD64|i.86" )/' \
	-e '/bb_enable_warnings/s/ warnings-as-errors//' "$tp/jm/CMakeLists.txt"
# JM divides the MVC DPB limit (H.264 G.10.2.1 f) by the number of views and
# then rejects conformant subset SPSs whose max_dec_frame_buffering exceeds
# that per-view figure. Make it a warning; JM then sizes the DPB from the VUI.
sed -i.orig 's|error ("max_dec_frame_buffering larger than MaxDpbSize", 500);|fprintf(stderr, "Warning: max_dec_frame_buffering larger than MaxDpbSize\\n");|' \
	"$tp/jm/source/app/ldecod/mbuffer.c"
(mkdir -p "$tp/jm/build" && cd "$tp/jm/build" && cmake .. -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1 &&
	make -s -j"$jobs" ldecod >/dev/null 2>&1)
cp "$(find "$tp/jm/bin" -name ldecod -type f | head -1)" "$inst/bin/ldecod"
cp "$tp/jm/cfg/decoder.cfg" "$inst/bin/decoder.cfg"

echo "== jmvc (MVC reference decoder for streams with more than two views)"
[ -d "$tp/jmvc" ] && git -C "$tp/jmvc" checkout -q -- .
fetch jmvc https://vcgit.hhi.fraunhofer.de/jvet/jmvc.git "$JMVC_TAG"
# Build fixes for current compilers and macOS; no decoding changes.
git -C "$tp/jmvc" apply "$root/tools/patches/jmvc-8.5.patch"
jl="$tp/jmvc/JMVC/H264Extension/build/linux"
for d in lib/H264AVCVideoIoLib lib/H264AVCCommonLib lib/H264AVCDecoderLib test/H264AVCDecoderLibTest; do
	make -s -C "$jl/$d" release -j"$jobs" >/dev/null 2>&1
done
cp "$tp/jmvc/bin/H264AVCDecoderLibTestStatic" "$inst/bin/jmvc-decoder"
ls "$inst/bin"
