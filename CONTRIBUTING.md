# Contributing to altc264

Thanks for helping. Bug reports, conformance findings and patches are all
welcome.

## Before you start

- For anything larger than a small fix, open an issue first so we can agree
  on the approach.
- Decoder output must stay bit-exact. Every change is checked against the
  ITU-T H.264.1 conformance suites in CI.

## Building and testing

```sh
cargo build --release
cargo test --workspace
```

Conformance (downloads about 3 GB from the ITU once; keeps 0.5 GB):

```sh
cargo xtask conformance --bitstreams-only   # fetch the suites into work/conformance
cargo xtask conformance-check --smoke       # a fast subset, as CI runs on every PR
cargo xtask conformance-check               # every stream
```

`conformance-check` compares each view's output with the expected hashes in
`conformance/manifest.json`. It needs no reference decoder. If your change
makes a stream newly bit-exact, regenerate the manifest (this needs the
reference YUVs and JM, `tools/build-reference.sh`):

```sh
cargo xtask conformance
cargo xtask conformance-manifest
```

Other useful checks:

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
RUSTFLAGS="--cfg altc_scalar" cargo test -p altc264   # the portable SIMD fallback
```

## Pull requests

- Keep each pull request to one change, with tests where they make sense.
- CI must pass: formatting, clippy, tests on Linux and macOS, the scalar and
  wasm builds, and the conformance smoke subset.
- Performance changes: include before and after timings, and the machine
  you measured on.

## Developer Certificate of Origin

All commits must be signed off, certifying the
[Developer Certificate of Origin](https://developercertificate.org/): that you
wrote the change, or otherwise have the right to submit it under the
project's licence.

```sh
git commit -s
```

This adds a `Signed-off-by: Your Name <you@example.com>` line, which must
match the commit author. A check on every pull request enforces it.

## Licence

By contributing you agree that your contributions are licensed under the
MIT licence in `LICENSE`, except for `integrations/ffmpeg/libaltc264.c`,
which is LGPL-2.1-or-later like FFmpeg (see `integrations/ffmpeg/LICENSE.md`).
