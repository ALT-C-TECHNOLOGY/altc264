// Reads of a File (in a worker) or a URL (HTTP range requests), in cached
// 1 MiB blocks. A URL is downloaded in the background up to WINDOW bytes
// past the last read, so playback rarely waits on the network.
const BLOCK = 1 << 20;
const WINDOW = 128 << 20;
const PARALLEL = 4;
const AHEAD = 8 << 20;

export function source(input) {
  let size, fetchBlock, fetchAhead = null;
  if (typeof input === 'string') {
    const head = new XMLHttpRequest();
    head.open('HEAD', input, false);
    head.send();
    if (head.status >= 400) throw new Error(`HTTP ${head.status}`);
    size = +head.getResponseHeader('Content-Length');
    const range = (pos, len) => `bytes=${pos}-${pos + len - 1}`;
    fetchBlock = (pos, len) => {
      const x = new XMLHttpRequest();
      x.open('GET', input, false);
      x.responseType = 'arraybuffer';
      x.setRequestHeader('Range', range(pos, len));
      x.send();
      if (x.status !== 206) throw new Error(`the server does not support range requests (HTTP ${x.status})`);
      return new Uint8Array(x.response);
    };
    fetchAhead = async (pos, len, signal) => {
      const r = await fetch(input, { headers: { Range: range(pos, len) }, signal });
      if (r.status !== 206) throw new Error(`HTTP ${r.status}`);
      return new Uint8Array(await r.arrayBuffer());
    };
  } else {
    size = input.size;
    const reader = new FileReaderSync();
    fetchBlock = (pos, len) => new Uint8Array(reader.readAsArrayBuffer(input.slice(pos, pos + len)));
  }
  const blocks = Math.ceil(size / BLOCK);
  const len = (i) => Math.min(BLOCK, size - i * BLOCK);
  const cache = new Map();
  // Background fetches in flight, by block, to abort when read directly.
  const pending = new Map();
  let last = 0;
  let waited = 0;
  // Blocks before the last read's are dropped first.
  const trim = () => {
    const limit = (WINDOW / BLOCK) * 1.25;
    for (const i of cache.keys()) {
      if (cache.size <= limit) break;
      if (i < last) cache.delete(i);
    }
  };
  const fill = () => {
    if (!fetchAhead) return;
    const end = Math.min(blocks, last + WINDOW / BLOCK);
    for (let i = last; i < end && pending.size < PARALLEL; i++) {
      if (cache.has(i) || pending.has(i)) continue;
      const abort = new AbortController();
      pending.set(i, abort);
      fetchAhead(i * BLOCK, len(i), abort.signal)
        .then((b) => cache.set(i, b))
        .catch(() => {})
        .finally(() => {
          if (pending.get(i) === abort) pending.delete(i);
          fill();
        });
    }
  };
  const block = (i) => {
    let b = cache.get(i);
    if (!b) {
      // Its result could not arrive until this read returns.
      pending.get(i)?.abort();
      pending.delete(i);
      const t0 = performance.now();
      b = fetchBlock(i * BLOCK, len(i));
      waited += performance.now() - t0;
      cache.set(i, b);
    }
    last = i;
    trim();
    fill();
    return b;
  };
  const read = (pos, n) => {
    n = Math.max(0, Math.min(n, size - pos));
    const out = new Uint8Array(n);
    for (let o = 0; o < n; ) {
      const i = Math.floor((pos + o) / BLOCK);
      const b = block(i);
      const from = pos + o - i * BLOCK;
      const k = Math.min(n - o, b.length - from);
      out.set(b.subarray(from, from + k), o);
      o += k;
    }
    return out;
  };
  // The fraction of the file downloaded without a gap from the last read.
  const buffered = () => {
    let i = last;
    while (i < blocks && cache.has(i)) i++;
    return Math.min(1, (i * BLOCK) / size);
  };
  // Whether the next reads could wait on the network: under AHEAD bytes
  // downloaded past the last read, before the end of the file. One decode
  // step can read several MiB (the core's 256 KiB reads plus the decoder's
  // frame delay).
  const starved = () => {
    if (!fetchAhead) return false;
    for (let i = last + 1; i <= last + AHEAD / BLOCK; i++) if (i < blocks && !cache.has(i)) return true;
    return false;
  };
  return { size, read, buffered, starved, waited: () => waited, network: !!fetchAhead, };
}
