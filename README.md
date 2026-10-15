<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/ALT-C-TECHNOLOGY/altc264/main/.github/assets/altc-wordmark-dark.svg">
    <img alt="altc" src="https://raw.githubusercontent.com/ALT-C-TECHNOLOGY/altc264/main/.github/assets/altc-wordmark-light.svg" width="260">
  </picture>
</p>

<h1 align="center">altc264</h1>

<p align="center">
  An H.264/AVC decoder in Rust with full MVC (stereo 3D) support.
</p>

<p align="center">
  <a href="https://github.com/ALT-C-TECHNOLOGY/altc264/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/ALT-C-TECHNOLOGY/altc264/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/ALT-C-TECHNOLOGY/altc264/actions/workflows/nightly.yml"><img alt="Conformance" src="https://github.com/ALT-C-TECHNOLOGY/altc264/actions/workflows/nightly.yml/badge.svg"></a>
  <a href="https://codecov.io/gh/ALT-C-TECHNOLOGY/altc264"><img alt="Coverage" src="https://codecov.io/gh/ALT-C-TECHNOLOGY/altc264/graph/badge.svg"></a>
  <a href="https://github.com/ALT-C-TECHNOLOGY/altc264/blob/main/LICENSE"><img alt="Licence: MIT" src="https://img.shields.io/badge/licence-MIT-blue.svg"></a>
  <img alt="MSRV 1.88" src="https://img.shields.io/badge/MSRV-1.88-orange.svg">
</p>

altc264 decodes H.264 from Baseline to the professional 4:4:4 profiles, and
both views of Blu-ray 3D and other MVC streams, fast enough for real-time
stereo playback on XR headsets. It is bit-exact against the ITU-T H.264.1
conformance suites, which CI checks on every change. Live browser demo
available [here](https://alt-c-technology.github.io/altc264/).

## Features

- **Profiles:** Baseline, Main, High, High 10, High 4:2:2 and High 4:4:4,
  including intra-only, lossless, interlaced (field and MBAFF) and 9 to
  14-bit video. Extended profile SP and SI slices (not data partitioning).
- **MVC:** Stereo High and Multiview High, every view, with side-by-side,
  top-and-bottom or frame-packed output and Blu-ray 3D plane offsets for
  subtitles.
- **Fast:** frame threading, and SIMD for NEON, SSE2/AVX2 and WebAssembly,
  with a portable fallback.
- **Easy to embed:** a Rust crate, a C library, and FFmpeg and GStreamer
  integrations.

## Platforms

| Platform | Architectures | SIMD |
|---|---|---|
| Apple: macOS, iOS, iPadOS, tvOS, visionOS | arm64, and x86_64 on macOS | NEON, SSE2/AVX2 on Intel |
| Android | arm64-v8a, x86_64 | NEON, SSE2/AVX2 |
| Linux | x86_64, aarch64 | SSE2/AVX2, NEON |
| Windows | x86_64, arm64 | SSE2/AVX2, NEON |
| WebAssembly | wasm32 (WASI) | SIMD128, threaded where the host allows it |

Other targets build with the portable fallback.

## Conformance

**All 270** ITU-T H.264.1 conformance streams are bit-exact on every CPU
backend, in every view the reference decoders can produce. The expected output of each stream is in
[`conformance/manifest.json`](https://github.com/ALT-C-TECHNOLOGY/altc264/blob/main/conformance/manifest.json).

## Performance

Against FFmpeg 9.0.2's H.264 decoder, on this public 1080p clip of [Big Buck Bunny](https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/1080/Big_Buck_Bunny_1080_10s_30MB.mp4).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/ALT-C-TECHNOLOGY/altc264/main/.github/assets/benchmark-dark.svg">
  <img alt="Frames per second decoding Big Buck Bunny 1080p, altc264 against FFmpeg 9.0.2, at 1 and 8 threads" src="https://raw.githubusercontent.com/ALT-C-TECHNOLOGY/altc264/main/.github/assets/benchmark-light.svg">
</picture>

### 3D (MVC)

Frames per second decoding both views of a 3D clip of Big Buck Bunny ([`bbb3d-mvc-10s.264`](https://github.com/ALT-C-TECHNOLOGY/altc264/releases/tag/benchmark-clips), 1080p).

| CPU | 1 thread | 8 threads |
|---|---:|---:|
| Apple M4 Pro | 87 | 546 |
| Intel Core Ultra X9 388H | 74 | 301 |
| AMD Ryzen AI 9 HX 370 | 62 | 339 |
| Qualcomm QCM6490 | 36 | 124 |
| WebAssembly (SIMD128) | 48 | 306 |

Full details in [BENCHMARKS.md](https://github.com/ALT-C-TECHNOLOGY/altc264/blob/main/BENCHMARKS.md).

## Usage

### Rust

```rust
use altc264::decoder::Decoder;

let mut dec = Decoder::with_auto_threads();
dec.decode_annex_b(&data);
dec.flush();
while let Some(frame) = dec.next_frame() {
    let y = frame.plane(0).unwrap(); // rows of luma samples
}
```

### C

```sh
cargo cinstall --release -p altc264-capi --prefix /usr/local
```

See [`crates/altc264-capi/include/altc264.h`](https://github.com/ALT-C-TECHNOLOGY/altc264/blob/main/crates/altc264-capi/include/altc264.h).

### FFmpeg and GStreamer

- FFmpeg: [`integrations/ffmpeg`](https://github.com/ALT-C-TECHNOLOGY/altc264/tree/main/integrations/ffmpeg) adds a `libaltc264`
  decoder (`./apply.sh /path/to/ffmpeg`, then `--enable-libaltc264`).
  Tested with FFmpeg 8.1.3 and 9.0.2.
- GStreamer: [`integrations/gstreamer`](https://github.com/ALT-C-TECHNOLOGY/altc264/tree/main/integrations/gstreamer) is an
  `altc264dec` element.

### Command line

```sh
cargo install --path crates/altc264-cli
altc264 decode input.264 left.yuv right.yuv
```

## Contributing

See [CONTRIBUTING.md](https://github.com/ALT-C-TECHNOLOGY/altc264/blob/main/CONTRIBUTING.md), including its [AI policy](https://github.com/ALT-C-TECHNOLOGY/altc264/blob/main/CONTRIBUTING.md#ai-tools).
Report security issues privately, as described in [SECURITY.md](https://github.com/ALT-C-TECHNOLOGY/altc264/blob/main/SECURITY.md).

## Licence

See [LICENSE](https://github.com/ALT-C-TECHNOLOGY/altc264/blob/main/LICENSE). The FFmpeg wrapper in
`integrations/ffmpeg` is LGPL-2.1-or-later, like FFmpeg.

The licence covers this code only. H.264 is subject to patents in many
jurisdictions, whoever distributes binaries is responsible for any
licensing they require.
