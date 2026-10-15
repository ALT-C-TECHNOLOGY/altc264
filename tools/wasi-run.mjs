// Runs a WASI command module (such as the CLI or the unit tests built for
// wasm32-wasip1) under node, with the current directory preopened as /work:
//   cargo build --release -p altc264-cli --target wasm32-wasip1
//   node tools/wasi-run.mjs target/wasm32-wasip1/release/altc264.wasm \
//     decode /work/in.264 /work/out.yuv
// Add RUSTFLAGS="-C target-feature=+simd128" for the SIMD128 kernels.
// wasm32-wasip1-threads modules get threads: one worker each, from a pool.
import { writeSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { WASI } from 'node:wasi';
import { argv, env, exit } from 'node:process';
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';

const SELF = new URL(import.meta.url);

function newWasi(args) {
  return new WASI({
    version: 'preview1',
    args,
    env: { ALTC_THREADS: env.ALTC_THREADS ?? '0' },
    // /dev too, so `decode ... /dev/null` discards frames as natively (benchmarks).
    preopens: { '/work': process.cwd(), '/dev': '/dev' },
    returnOnExit: true,
  });
}

// The limits of the module's imported memory, from its import section.
function importedMemory(bytes) {
  let p = 8;
  const u32 = () => {
    let v = 0;
    for (let s = 0; ; s += 7) {
      const b = bytes[p++];
      v += (b & 0x7f) * 2 ** s;
      if (b < 0x80) return v;
    }
  };
  const name = () => {
    const n = u32();
    p += n;
  };
  while (p < bytes.length) {
    const id = bytes[p++];
    const size = u32();
    const end = p + size;
    if (id === 2) {
      for (let n = u32(); n > 0; n--) {
        name();
        name();
        const kind = bytes[p++];
        if (kind === 2) {
          const flags = bytes[p++];
          const initial = u32();
          const maximum = flags & 1 ? u32() : undefined;
          return { initial, maximum, shared: (flags & 2) !== 0 };
        }
        if (kind === 0) u32();
        else if (kind === 1) (p++, bytes[p++] & 1 ? (u32(), u32()) : u32());
        else if (kind === 3) p += 2;
        else if (kind === 4) (p++, u32());
      }
    }
    p = end;
  }
  return null;
}

// Workers are made before the program starts, as Emscripten's pthreads are.
// WASM_THREAD_POOL sets the pool size.
const POOL = +env.WASM_THREAD_POOL || 32;
// Per pool slot: state, thread id, start argument.
const [IDLE, CLAIMED, GO] = [1, 2, 3];

// Imports for one thread: WASI on the shared memory, and thread-spawn.
function imports(wasi, module, memory, ctl, args) {
  const spawn = (startArg) => {
    for (let i = 1; i <= POOL; i++) {
      if (Atomics.compareExchange(ctl, 3 * i, IDLE, CLAIMED) === IDLE) {
        // Thread ids start at 1.
        const tid = Atomics.add(ctl, 0, 1) + 1;
        ctl[3 * i + 1] = tid;
        ctl[3 * i + 2] = startArg;
        Atomics.store(ctl, 3 * i, GO);
        Atomics.notify(ctl, 3 * i);
        return tid;
      }
    }
    // Pool used up: a worker of its own.
    const tid = Atomics.add(ctl, 0, 1) + 1;
    worker({ module, memory, ctl, args, tid, startArg });
    return tid;
  };
  return { ...wasi.getImportObject(), env: { memory }, wasi: { 'thread-spawn': spawn } };
}

function worker(data) {
  const w = new Worker(SELF, { workerData: data });
  w.on('error', (e) => {
    console.error(e);
    exit(1);
  });
  w.unref();
  return w;
}

if (isMainThread) {
  const [, , wasmPath, ...rest] = argv;
  const args = [wasmPath, ...rest];
  const bytes = await readFile(wasmPath);
  const module = await WebAssembly.compile(bytes);
  const wasi = newWasi(args);
  const limits = importedMemory(bytes);
  if (!limits) {
    const instance = await WebAssembly.instantiate(module, wasi.getImportObject());
    exit(wasi.start(instance));
  }
  const memory = new WebAssembly.Memory(limits);
  const ctl = new Int32Array(new SharedArrayBuffer(4 * 3 * (POOL + 1)));
  await Promise.all(
    Array.from({ length: POOL }, (_, k) => {
      const w = worker({ module, memory, ctl, slot: k + 1, args });
      return new Promise((ready) => w.once('message', ready));
    }),
  );
  const instance = await WebAssembly.instantiate(module, imports(wasi, module, memory, ctl, args));
  const code = wasi.start(instance);
  if (env.WASM_MEMORY_REPORT) console.error(`wasm memory ${memory.buffer.byteLength / 2 ** 20} MiB`);
  exit(code);
} else {
  const { module, memory, ctl, slot, args, tid, startArg } = workerData;
  const wasi = newWasi(args);
  const instance = await WebAssembly.instantiate(module, imports(wasi, module, memory, ctl, args));
  // WASI on this thread's instance without running _start.
  wasi.initialize({ exports: { memory } });
  const run = (tid, startArg) => {
    try {
      instance.exports.wasi_thread_start(tid, startArg);
    } catch (e) {
      // A trap or proc_exit ends the process (exit() would end only the worker).
      if (!(e instanceof WebAssembly.RuntimeError) && e?.code === 0) exit(0);
      writeSync(2, `thread ${tid}: ${e?.stack ?? e}\n`);
      process.kill(process.pid, 'SIGTERM');
    }
  };
  if (slot === undefined) {
    run(tid, startArg);
    process.exit();
  }
  Atomics.store(ctl, 3 * slot, IDLE);
  parentPort.postMessage('ready');
  for (;;) {
    for (let s; (s = Atomics.load(ctl, 3 * slot)) !== GO; ) Atomics.wait(ctl, 3 * slot, s);
    run(ctl[3 * slot + 1], ctl[3 * slot + 2]);
    // Back to the pool.
    Atomics.store(ctl, 3 * slot, IDLE);
  }
}
