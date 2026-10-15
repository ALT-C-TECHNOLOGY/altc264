#!/bin/sh
# altc264's threaded WebAssembly CLI under Node, run like the native CLI (for
# bench.sh): file arguments become paths under /work, the current directory.
#   cargo build --release -p altc264-cli --target wasm32-wasip1-threads
#   tools/bench/bench.sh tools/bench/wasm-altc264.sh tools/bench/wasm-ffmpeg.sh CLIP.264
S=$(cd "$(dirname "$0")" && pwd)
W=${ALTC264_WASM:-target/wasm32-wasip1-threads/release/altc264.wasm}
for a in "$@"; do
	shift
	case $a in
	/dev/*) ;;
	/*) a=/work/${a#"$PWD"/} ;;
	*) [ -e "$a" ] && a=/work/$a ;;
	esac
	set -- "$@" "$a"
done
exec node --no-warnings "$S/../wasi-run.mjs" "$W" "$@"
