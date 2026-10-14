# Benchmarks

altc264 against FFmpeg's H.264 decoder (libavcodec, FFmpeg 9.0.2, with its
hand-written assembly), decoding two public clips at 1 and 8 threads.
Measured October 2026 with altc264 0.1.0.

## Results

Frames per second (best of 5), with altc264's speed relative to FFmpeg in brackets.

### 2D: Big Buck Bunny, 1080p

| CPU | 1 thread: altc264 vs FFmpeg | 8 threads: altc264 vs FFmpeg |
|---|---|---|
| Apple M4 Pro | 132.5 vs 109.7 (**1.21×**) | 578.1 vs 370.1 (**1.56×**) |
| Intel Core Ultra X9 388H | 108.7 vs 114.8 (0.95×) | 360.3 vs 289.6 (**1.24×**) |
| AMD Ryzen AI 9 HX 370 | 95.7 vs 108.6 (0.88×) | 409.8 vs 351.0 (**1.17×**) |
| Qualcomm QCM6490 | 51.8 vs 49.1 (**1.05×**) | 191.2 vs 167.3 (**1.14×**) |
| WebAssembly (Node 22, on the M4 Pro) | 77.7 vs 62.5 (**1.24×**) | not supported yet |

### 3D (MVC): Big Buck Bunny 3D, 1080p

FFmpeg's decoder cannot decode the second view of an MVC stream, so the
comparison is on the first view.

| CPU | 1 thread: altc264 vs FFmpeg | 8 threads: altc264 vs FFmpeg |
|---|---|---|
| Apple M4 Pro | 174.9 vs 148.5 (**1.18×**) | 817.0 vs 525.4 (**1.55×**) |
| Intel Core Ultra X9 388H | 151.2 vs 164.5 (0.92×) | 539.0 vs 467.6 (**1.15×**) |
| AMD Ryzen AI 9 HX 370 | 128.2 vs 154.6 (0.83×) | 603.9 vs 509.3 (**1.19×**) |
| Qualcomm QCM6490 | 72.1 vs 68.9 (**1.05×**) | 246.0 vs 239.7 (**1.03×**) |
| WebAssembly (Node 22, on the M4 Pro) | 97.1 vs 81.3 (**1.19×**) | not supported yet |

altc264 decoding both views:

| CPU | 1 thread | 8 threads |
|---|---|---|
| Apple M4 Pro | 86.7 | 546.1 |
| Intel Core Ultra X9 388H | 73.5 | 300.8 |
| AMD Ryzen AI 9 HX 370 | 61.7 | 339.0 |
| Qualcomm QCM6490 | 36.1 | 123.9 |
| WebAssembly (Node 22, on the M4 Pro) | 48.2 | not supported yet |

### Reading the numbers

- WebAssembly runs single-threaded: threads in WebAssembly are not supported
  yet, by either build.

## Clips

| Clip | Source | SHA-256 |
|---|---|---|
| 2D: Big Buck Bunny 1080p, 10 s, H.264 High, 30 fps, 24.5 Mbps | [test-videos.co.uk](https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/1080/Big_Buck_Bunny_1080_10s_30MB.mp4) (MP4, with the video track extracted to Annex B) | `5b603ec1db36cc15d5e48ab9ea6e72ffd6a65e325f2bb218e9ed298974b453bc` (MP4) |
| 3D: Big Buck Bunny 3D 1080p, 10 s, MVC (Multiview High), 30 fps, 36.5 Mbps | Attached to the [`benchmark-clips` release](https://github.com/ALT-C-TECHNOLOGY/altc264/releases/tag/benchmark-clips), or remade with `tools/bench/make-mvc-sample.sh` | `981c8576e2bd1d6d0d0a8bd2d4d1ef0e3d01fefca12999189ab3c9f59950a5d6` |

Extract the 2D clip's video track with
`ffmpeg -i Big_Buck_Bunny_1080_10s_30MB.mp4 -map 0:v:0 -c copy -bsf:v h264_mp4toannexb -f h264 bbb-1080-10s.264`.

The 3D clip is 7:04 to 7:14 of the Blender Foundation's stereo Big
Buck Bunny ([download.blender.org](https://download.blender.org/demo/movies/BBB/),
`bbb_sunflower_1080p_30fps_stereo_abl.mp4`, the eyes stacked left on top),
encoded with JMVC 8.5, the ITU-T reference MVC encoder (configuration in
`tools/bench/mvc-sample.cfg`). The clip was encoded once, on the M4 Pro,
and that one file (the release file, checksum above) was benchmarked on
every platform. JMVC's output depends on the compiler, so rebuilding it on
macOS gives the same file byte for byte, and on other systems a different
but valid stream.

Big Buck Bunny © 2008, 2013 Blender Foundation, [www.bigbuckbunny.org](https://www.bigbuckbunny.org),
licensed under [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/).

## Method

- **Decoders:** altc264's CLI (`altc264 decode CLIP /dev/null`, which skips
  writing the output), and FFmpeg 9.0.2 built with
  `tools/bench/build-ffmpeg.sh`: the h264 decoder only, with its assembly
  (x86 and NEON), timed by `tools/bench/ffmpeg-driver.c`. Both load the
  whole file before the clock starts and time decoding alone.
- **Threads:** 1 thread is altc264's calling-thread decoder
  (`ALTC_THREADS=0`) and FFmpeg's `thread_count` 1. 8 threads is
  `ALTC_THREADS=8` and FFmpeg's frame and slice threading with
  `thread_count` 8.
- **Runs:** `tools/bench/bench.sh`, best of 5, the two decoders interleaved.
- **Core pinning:** the single-thread runs are pinned to a performance core
  on CPUs with mixed core types (Intel cpu 0, AMD cpu 0, QCM6490 cpu 7, its
  2.7 GHz Cortex-A78). 8-thread runs are not pinned. The QCM6490 has four
  Cortex-A78 and four Cortex-A55 cores, and its 8-thread runs use both.
- **Machines:** the M4 Pro on macOS 27, the Intel and AMD laptops on Fedora
  Linux (kernel 7.2), and the QCM6490 on Ubuntu 24.04 (kernel 6.8), running a
  static aarch64-musl build. altc264 is built with `cargo build --release`
  and Rust 1.98 on every machine, with no CPU-specific flags.
- **WebAssembly:** both decoders single-threaded under Node 22: altc264's
  `wasm32-wasip1` build with SIMD128 (`tools/wasi-run.mjs`), and FFmpeg 9.0.2
  compiled with Emscripten by `tools/bench/build-ffmpeg-wasm.sh` (`-O3
  -msimd128`, no assembly, as in-browser FFmpeg ports are).

### Reproducing

```sh
tools/bench/build-ffmpeg.sh                      # -> work/bench/bench-ffmpeg
cargo build --release -p altc264-cli
tools/bench/bench.sh target/release/altc264 work/bench/bench-ffmpeg bbb-1080-10s.264 "taskset -c 0"
tools/bench/bench.sh target/release/altc264 work/bench/bench-ffmpeg bbb3d-mvc-10s.264 "taskset -c 0"
```