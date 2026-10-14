#!/bin/sh
# Best of 5 interleaved runs, at 1 and 8 threads: altc264 against FFmpeg's
# h264 decoder (decoding the base view only, as FFmpeg cannot decode MVC's
# second view), and for MVC clips altc264 decoding both views. Prints frames
# (access units) per second. See BENCHMARKS.md.
#   tools/bench/bench.sh ALTC264 BENCH_FFMPEG CLIP.264 ["PIN"]
# PIN prefixes the single-thread runs, e.g. "taskset -c 0" to keep them on
# one performance core of a CPU with mixed core types.
A=$1; F=$2; C=$3; P=${4:-}
fps() { "$@" 2>&1 | grep -oE '"fps":[0-9.]+|\(([0-9.]+) fps\)' | tail -1 | grep -oE '[0-9.]+'; }
best() { tr ' ' '\n' | awk 'BEGIN { b = 0 } { if ($1 > b) b = $1 } END { print b }'; }
mvc=$("$A" nals "$C" 2>/dev/null | grep -qE "SubsetSps|SliceExtension" && echo 1 || echo 0)
a1=; f1=; m1=; a8=; f8=; m8=
for i in 1 2 3 4 5; do
	a1="$a1 $(fps env ALTC_THREADS=0 $P "$A" decode "$C" /dev/null)"
	f1="$f1 $(fps $P "$F" "$C" --base-only --threads 1)"
	a8="$a8 $(fps env ALTC_THREADS=8 "$A" decode "$C" /dev/null)"
	f8="$f8 $(fps "$F" "$C" --base-only --threads 8)"
	if [ "$mvc" = 1 ]; then
		m1="$m1 $(fps env ALTC_THREADS=0 $P "$A" decode "$C" /dev/null /dev/null)"
		m8="$m8 $(fps env ALTC_THREADS=8 "$A" decode "$C" /dev/null /dev/null)"
	fi
done
echo "base view, 1 thread:  altc264 $(echo $a1 | best)  ffmpeg $(echo $f1 | best)"
echo "base view, 8 threads: altc264 $(echo $a8 | best)  ffmpeg $(echo $f8 | best)"
if [ "$mvc" = 1 ]; then
	echo "both views, 1 thread:  altc264 $(echo $m1 | best)"
	echo "both views, 8 threads: altc264 $(echo $m8 | best)"
fi
