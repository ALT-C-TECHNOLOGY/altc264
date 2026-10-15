// Threads for a wasm32-wasip1-threads module: a pool of workers made before
// it runs (pool-worker.js), handed threads by wasi.thread-spawn.
import { Wasi } from './wasi.js';

const [IDLE, CLAIMED, GO] = [1, 2, 3];

export function imports(wasi, memory, ctl, size) {
  const spawn = (startArg) => {
    for (let i = 1; i <= size; i++) {
      if (Atomics.compareExchange(ctl, 3 * i, IDLE, CLAIMED) === IDLE) {
        const tid = Atomics.add(ctl, 0, 1) + 1;
        ctl[3 * i + 1] = tid;
        ctl[3 * i + 2] = startArg;
        Atomics.store(ctl, 3 * i, GO);
        Atomics.notify(ctl, 3 * i);
        return tid;
      }
    }
    // A worker made now could not start while this thread blocks.
    wasi.log(2, `all ${size} pool threads busy\n`);
    return -6;
  };
  return { wasi_snapshot_preview1: wasi.imports(), env: { memory }, wasi: { 'thread-spawn': spawn } };
}

// Stubs for a module's other function imports, for threads that never call them.
export function stubs(module, given) {
  for (const { module: m, name, kind } of WebAssembly.Module.imports(module)) {
    if (kind !== 'function' || given[m]?.[name]) continue;
    given[m] ??= {};
    given[m][name] = () => {
      throw new Error(`${m}.${name} called on a pool thread`);
    };
  }
  return given;
}

// The largest shared memory this browser gives, from the module's maximum
// down, and its maximum in pages.
export function newMemory(limits) {
  for (let max = limits.maximum; max >= limits.initial; max >>= 1) {
    try {
      return [new WebAssembly.Memory({ ...limits, maximum: max }), max];
    } catch {}
  }
  throw new Error('no shared memory');
}

// Pool workers log here: the thread that made them may be blocked.
export const logs = new BroadcastChannel('altc264-log');

// The pool workers' scripts as blob URLs, downloaded once. Loaded per worker,
// WebKit sends some of them past the service worker that isolates the page
// on GitHub Pages, then refuses them for lacking its headers.
let threadScript = null;
function threadUrl() {
  const blob = async (file, from, to) => {
    let text = await (await fetch(new URL(file, import.meta.url))).text();
    if (from) {
      if (!text.includes(from)) throw new Error(`${file} does not import ${from}`);
      text = text.replace(from, to);
    }
    return URL.createObjectURL(new Blob([text], { type: 'text/javascript' }));
  };
  return (threadScript ??= (async () => {
    const wasi = await blob('./wasi.js');
    const pool = await blob('./pool.js', "from './wasi.js'", `from '${wasi}'`);
    return blob('./pool-worker.js', "from './pool.js'", `from '${pool}'`);
  })());
}

export async function startPool(module, memory, size, args = [], env = {}) {
  const ctl = new Int32Array(new SharedArrayBuffer(4 * 3 * (size + 1)));
  const workers = [];
  const url = await threadUrl();
  await Promise.all(
    Array.from({ length: size }, (_, k) => {
      const w = new Worker(url, { type: 'module' });
      workers.push(w);
      w.postMessage({ module, memory, ctl, slot: k + 1, size, args, env });
      return new Promise((ready, fail) => {
        w.onmessage = ready;
        w.onerror = (e) => fail(new Error(`pool thread ${k + 1} failed to start: ${e.message || 'no message'} (${e.filename || '?'}:${e.lineno ?? '?'})`));
      });
    }),
  );
  // busy(): the pool threads running now.
  const busy = () => Array.from({ length: size }, (_, k) => Atomics.load(ctl, 3 * (k + 1))).filter((s) => s !== IDLE).length;
  return { ctl, size, busy, stop: () => workers.forEach((w) => w.terminate()) };
}

export async function poolWorker({ module, memory, ctl, slot, size, args, env }) {
  const wasi = new Wasi({ args, env, log: (fd, text) => logs.postMessage({ log: text, fd }) });
  const instance = await WebAssembly.instantiate(module, stubs(module, imports(wasi, memory, ctl, size)));
  wasi.bind(memory);
  Atomics.store(ctl, 3 * slot, IDLE);
  postMessage({ ready: true });
  for (;;) {
    for (let s; (s = Atomics.load(ctl, 3 * slot)) !== GO; ) Atomics.wait(ctl, 3 * slot, s);
    try {
      instance.exports.wasi_thread_start(ctl[3 * slot + 1], ctl[3 * slot + 2]);
    } catch (e) {
      logs.postMessage({ log: `thread ${ctl[3 * slot + 1]}: ${e.stack ?? e}\n`, fd: 2, fatal: true });
      return;
    }
    Atomics.store(ctl, 3 * slot, IDLE);
  }
}
