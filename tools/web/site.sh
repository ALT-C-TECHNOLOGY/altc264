#!/bin/sh
# Assembles the player as a static site in DIR (for GitHub Pages): the page,
# its scripts, both cores from build.sh, and the sample clips from CLIPS
# (bbb-1080-10s.mp4 and bbb3d-mvc-10s.mkv, from the benchmark-clips release).
#   tools/web/build.sh && tools/web/site.sh DIR CLIPS
set -eu
cd "$(dirname "$0")/../.."
out=${1:?usage: site.sh DIR CLIPS} clips=${2:?usage: site.sh DIR CLIPS}
mkdir -p "$out/clips"
for f in index.html wasi.js pool.js pool-worker.js player-worker.js source.js coi-serviceworker.js coi-serviceworker.LICENSE; do
	cp "tools/web/$f" "$out/"
done
cp work/web/core.wasm work/web/core-st.wasm "$out/"
cp "$clips/bbb-1080-10s.mp4" "$clips/bbb3d-mvc-10s.mkv" "$out/clips/"
echo "site in $out"
