// Runs a WASI command module (such as the CLI or the unit tests built for
// wasm32-wasip1) under node, with the current directory preopened as /work:
//   cargo build --release -p altc264-cli --target wasm32-wasip1
//   node tools/wasi-run.mjs target/wasm32-wasip1/release/altc264.wasm \
//     decode /work/in.264 /work/out.yuv
// Add RUSTFLAGS="-C target-feature=+simd128" for the SIMD128 kernels.
import { readFile } from 'node:fs/promises';
import { WASI } from 'node:wasi';
import { argv, env, exit } from 'node:process';

const [, , wasmPath, ...args] = argv;
const wasi = new WASI({
  version: 'preview1',
  args: [wasmPath, ...args],
  env: { ALTC_THREADS: env.ALTC_THREADS ?? '0' },
  preopens: { '/work': process.cwd() },
  returnOnExit: true,
});
const module = await WebAssembly.compile(await readFile(wasmPath));
const instance = await WebAssembly.instantiate(module, wasi.getImportObject());
exit(wasi.start(instance));
