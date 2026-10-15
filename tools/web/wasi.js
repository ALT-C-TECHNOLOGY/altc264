// The WASI preview1 calls the player core makes: environment, clocks,
// sleeping and console output. It opens no files (the core reads through
// host.read).
const ESUCCESS = 0, EBADF = 8, ESPIPE = 70;

export class Wasi {
  constructor({ args = [], env = {}, log }) {
    this.args = args;
    this.env = Object.entries(env).map(([k, v]) => `${k}=${v}`);
    this.log = log;
    this.dec = new TextDecoder();
    this.sleeper = self.SharedArrayBuffer ? new Int32Array(new SharedArrayBuffer(4)) : null;
  }

  bind(memory) {
    this.memory = memory;
  }

  view() {
    return new DataView(this.memory.buffer);
  }

  bytes(p, n) {
    return new Uint8Array(this.memory.buffer, p, n);
  }

  strings(list, ptrs, buf) {
    const enc = new TextEncoder();
    const v = this.view();
    for (const s of list) {
      const b = enc.encode(s + '\0');
      v.setUint32(ptrs, buf, true);
      this.bytes(buf, b.length).set(b);
      ptrs += 4;
      buf += b.length;
    }
    return ESUCCESS;
  }

  sizes(list, countPtr, sizePtr) {
    const v = this.view();
    v.setUint32(countPtr, list.length, true);
    v.setUint32(sizePtr, list.reduce((n, s) => n + new TextEncoder().encode(s).length + 1, 0), true);
    return ESUCCESS;
  }

  imports() {
    const w = this;
    return {
      args_sizes_get: (c, s) => w.sizes(w.args, c, s),
      args_get: (p, b) => w.strings(w.args, p, b),
      environ_sizes_get: (c, s) => w.sizes(w.env, c, s),
      environ_get: (p, b) => w.strings(w.env, p, b),
      clock_time_get(id, _precision, p) {
        const ms = id === 0 ? Date.now() : performance.timeOrigin + performance.now();
        w.view().setBigUint64(p, BigInt(Math.round(ms * 1e6)), true);
        return ESUCCESS;
      },
      fd_close: (fd) => (fd < 3 ? ESUCCESS : EBADF),
      fd_fdstat_get(fd, p) {
        if (fd > 2) return EBADF;
        const v = w.view();
        v.setUint8(p, 2);
        v.setUint16(p + 2, 0, true);
        v.setBigUint64(p + 8, 0xffffffffffffffffn, true);
        v.setBigUint64(p + 16, 0xffffffffffffffffn, true);
        return ESUCCESS;
      },
      fd_fdstat_set_flags: (fd) => (fd < 3 ? ESUCCESS : EBADF),
      fd_prestat_get: () => EBADF,
      fd_prestat_dir_name: () => EBADF,
      fd_seek: (fd) => (fd < 3 ? ESPIPE : EBADF),
      fd_read(fd, _iovs, _n, nread) {
        if (fd > 2) return EBADF;
        w.view().setUint32(nread, 0, true);
        return ESUCCESS;
      },
      fd_write(fd, iovs, n, nwritten) {
        if (fd > 2) return EBADF;
        const v = w.view();
        let total = 0;
        for (let i = 0; i < n; i++) {
          const buf = v.getUint32(iovs + 8 * i, true);
          const len = v.getUint32(iovs + 8 * i + 4, true);
          w.log(fd, w.dec.decode(w.bytes(buf, len).slice()));
          total += len;
        }
        v.setUint32(nwritten, total, true);
        return ESUCCESS;
      },
      path_open: () => EBADF,
      // Clock subscriptions only: sleeps for the shortest.
      poll_oneoff(subs, events, n, nevents) {
        const v = w.view();
        let ms = Infinity;
        for (let i = 0; i < n; i++) {
          const s = subs + 48 * i;
          if (v.getUint8(s + 8) !== 0) continue;
          const t = Number(v.getBigUint64(s + 24, true)) / 1e6;
          ms = Math.min(ms, v.getUint16(s + 40, true) & 1 ? t - (performance.timeOrigin + performance.now()) : t);
        }
        if (ms > 0 && ms < Infinity) {
          if (w.sleeper) Atomics.wait(w.sleeper, 0, 0, ms);
          else for (const end = performance.now() + ms; performance.now() < end; );
        }
        for (let i = 0; i < n; i++) {
          const e = events + 32 * i;
          v.setBigUint64(e, v.getBigUint64(subs + 48 * i, true), true);
          v.setUint16(e + 8, 0, true);
          v.setUint8(e + 10, v.getUint8(subs + 48 * i + 8));
        }
        v.setUint32(nevents, n, true);
        return ESUCCESS;
      },
      proc_exit(code) {
        throw new Error(`exit ${code}`);
      },
      sched_yield: () => ESUCCESS,
    };
  }
}

// The limits of a module's imported memory, from its import section.
export function importedMemory(bytes) {
  let p = 8;
  const u32 = () => {
    let v = 0;
    for (let s = 0; ; s += 7) {
      const b = bytes[p++];
      v += (b & 0x7f) * 2 ** s;
      if (b < 0x80) return v;
    }
  };
  const skip = () => {
    const n = u32();
    p += n;
  };
  while (p < bytes.length) {
    const id = bytes[p++];
    const end = u32() + p;
    if (id === 2) {
      for (let n = u32(); n > 0; n--) {
        skip();
        skip();
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
