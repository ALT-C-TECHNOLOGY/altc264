// The player's decoder: the FFmpeg + libaltc264 core (core.c) in this
// worker, its threads from the pool. Sends video frames and audio chunks.
import { Wasi, importedMemory } from './wasi.js';
import { imports, newMemory, startPool } from './pool.js';
import { source } from './source.js';

let core, memory, pool = null, src = null, fps = 30, dropBefore = -1, busy = 0;

const view = () => new DataView(memory.buffer);

// bytes: the wasm the page downloaded, the threaded build where the browser
// allows threads (else the single-threaded one).
async function init({ bytes, threaded }) {
  const host = {
    read(pos, len, buf) {
      const b = src.read(pos, len);
      new Uint8Array(memory.buffer, buf, b.length).set(b);
      return b.length;
    },
  };
  const wasi = new Wasi({ args: ['core'], env: {}, log: (fd, text) => postMessage({ log: text }) });
  let instance = null;
  if (threaded) {
    postMessage({ progress: 'Compiling the decoder…' });
    const module = await WebAssembly.compile(bytes);
    try {
      [memory] = newMemory(importedMemory(new Uint8Array(bytes)));
      postMessage({ progress: 'Starting the decoder threads…' });
      pool = await startPool(module, memory, 48);
      instance = await WebAssembly.instantiate(module, { ...imports(wasi, memory, pool.ctl, pool.size), host });
    } catch (e) {
      postMessage({ log: `threaded build unavailable (${e.message}): single-threaded\n` });
    }
  }
  if (!instance) {
    postMessage({ progress: 'Compiling the decoder…' });
    const st = threaded ? await (await fetch('core-st.wasm')).arrayBuffer() : bytes;
    const module = await WebAssembly.compile(st);
    instance = await WebAssembly.instantiate(module, { wasi_snapshot_preview1: wasi.imports(), host });
    memory = instance.exports.memory;
  }
  wasi.bind(memory);
  core = instance.exports;
  core._initialize();
  return { threaded: !!pool };
}

function cstring(p) {
  const b = new Uint8Array(memory.buffer, p);
  return new TextDecoder().decode(b.slice(0, b.indexOf(0)));
}

function open({ input, threads, rate }) {
  src = source(input);
  const r = core.core_open(src.size, threads, rate);
  if (r === -1) return { refused: 'This file could not be opened. The player plays MKV, MP4 and TS files.' };
  if (r === -2) {
    const codec = cstring(core.core_video_codec()).toUpperCase();
    return { refused: codec ? `This video is ${codec}. The player only plays H.264 and MVC video.` : 'This file has no video.' };
  }
  if (r < 0) return { refused: `This file could not be played (error ${r}).` };
  fps = core.core_fps() || 30;
  dropBefore = -1;
  return { duration: core.core_duration(), fps: core.core_fps(), audio: cstring(core.core_audio_codec()) };
}

// The core's output struct (see core.c).
function video(o) {
  const v = view();
  const w = v.getInt32(o + 24, true), h = v.getInt32(o + 28, true);
  const cw = v.getInt32(o + 32, true), ch = v.getInt32(o + 36, true);
  const sbs = v.getInt32(o + 40, true) === 1, t = v.getFloat64(o + 48, true), bits = v.getInt32(o + 72, true);
  if (t < dropBefore) return null;
  const planes = [0, 1, 2].map((k) => {
    const [pw, ph] = k ? [cw, ch] : [w, h];
    const data = v.getUint32(o + 4 * k, true), stride = v.getInt32(o + 12 + 4 * k, true);
    const out = new Uint8Array(pw * ph);
    for (let y = 0; y < ph; y++) {
      if (bits <= 8) out.set(new Uint8Array(memory.buffer, data + y * stride, pw), y * pw);
      else {
        const row = new Uint16Array(memory.buffer.slice(data + y * stride, data + y * stride + 2 * pw));
        for (let x = 0; x < pw; x++) out[y * pw + x] = row[x] >> (bits - 8);
      }
    }
    return out;
  });
  return { t, w, h, cw, ch, sbs, planes };
}

function audio(o) {
  const v = view();
  const p = v.getUint32(o + 56, true), n = v.getInt32(o + 60, true), t = v.getFloat64(o + 64, true);
  if (t < dropBefore) return null;
  return { t, pcm: new Float32Array(new Float32Array(memory.buffer, p, 2 * n)) };
}

// Decodes until `want` video frames are out or the stream ends; unless
// `wait`, stops early (starved) rather than wait on the network.
let starved = false;
function pull(want, wait) {
  const t0 = performance.now(), w0 = src.waited();
  const o = core.core_out();
  let n = 0;
  starved = false;
  while (n < want) {
    if (!wait && src.starved()) {
      starved = true;
      break;
    }
    const kind = core.core_next();
    if (kind === 0) {
      postMessage({ end: true });
      break;
    }
    if (kind === 1) {
      const f = video(o);
      if (!f) continue;
      postMessage({ frame: f }, f.planes.map((p) => p.buffer));
      n++;
    } else {
      const a = audio(o);
      if (a) postMessage({ audio: a }, [a.pcm.buffer]);
    }
  }
  busy += performance.now() - t0 - (src.waited() - w0);
  return n;
}

onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.init) postMessage({ init: await init(m.init) });
    else if (m.open) postMessage({ opened: open(m.open) });
    else if (m.pull) {
      const pulled = src ? pull(m.pull, m.wait) : 0;
      postMessage({ pulled, busy, starved, buffered: src?.buffered() ?? 0, threads: pool?.busy() ?? 0 });
    } else if (m.status) postMessage({ starved: src?.starved() ?? false, buffered: src?.buffered() ?? 0 });
    else if (m.seek != null) {
      core.core_seek(m.seek);
      dropBefore = m.seek - 0.5 / fps;
      postMessage({ seeked: m.seek });
    }
  } catch (err) {
    postMessage({ error: `${err}\n${err.stack ?? ''}` });
  }
};
