#!/bin/sh
# Adds the libaltc264 decoder wrapper to an FFmpeg source tree (idempotent).
#   integrations/ffmpeg/apply.sh /path/to/ffmpeg
# Tested with FFmpeg 8.1.3. It edits configure and the libavcodec build files
# next to the libdav1d entries, so other versions may need those edits by hand.
set -eu
src=$(cd "$(dirname "$0")" && pwd)
ff=${1:?usage: apply.sh FFMPEG_DIR}

cp "$src/libaltc264.c" "$ff/libavcodec/libaltc264.c"

grep -q 'libaltc264' "$ff/configure" || {
	sed -i.bak \
		-e 's|^  --enable-libdav1d        enable AV1 decoding via libdav1d \[no\]$|&\n  --enable-libaltc264      enable H.264/MVC decoding via libaltc264 [no]|' \
		-e 's|^    libdav1d$|    libaltc264\n&|' \
		-e 's|^libdav1d_decoder_deps="libdav1d"$|libaltc264_decoder_deps="libaltc264"\n&|' \
		-e 's|^enabled libdav1d          \&\& require_pkg_config libdav1d.*$|enabled libaltc264        \&\& require_pkg_config libaltc264 altc264 altc264.h altc264_open\n&|' \
		"$ff/configure"
	rm -f "$ff/configure.bak"
}
grep -q 'libaltc264' "$ff/libavcodec/Makefile" || {
	sed -i.bak 's|^OBJS-$(CONFIG_LIBDAV1D_DECODER) .*$|OBJS-$(CONFIG_LIBALTC264_DECODER)         += libaltc264.o\n&|' "$ff/libavcodec/Makefile"
	rm -f "$ff/libavcodec/Makefile.bak"
}
grep -q 'libaltc264' "$ff/libavcodec/allcodecs.c" || {
	sed -i.bak 's|^extern const FFCodec ff_libdav1d_decoder;$|extern const FFCodec ff_libaltc264_decoder;\n&|' "$ff/libavcodec/allcodecs.c"
	rm -f "$ff/libavcodec/allcodecs.c.bak"
}
grep -c libaltc264 "$ff/configure" "$ff/libavcodec/Makefile" "$ff/libavcodec/allcodecs.c"
