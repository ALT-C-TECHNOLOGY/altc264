// Serves the player over HTTPS on the LAN, cross-origin isolated (as wasm
// threads need), with a self-signed certificate made on first run. The
// sample clips are read from work/clips (the benchmark-clips release files).
//   tools/web/build.sh && node tools/web/server.mjs [PORT]
import { execFileSync } from 'node:child_process';
import { createReadStream, existsSync, mkdirSync, readFileSync, statSync } from 'node:fs';
import { createServer } from 'node:https';
import { networkInterfaces } from 'node:os';
import { fileURLToPath } from 'node:url';

export const ROOT = fileURLToPath(new URL('../..', import.meta.url)).replace(/\/$/, '');

const WEB = `${ROOT}/tools/web`;
const CLIPS = Object.fromEntries(['bbb-1080-10s.mp4', 'bbb3d-mvc-10s.mkv'].map((c) => [c, `${ROOT}/work/clips/${c}`]));
const ROUTES = {
  ...Object.fromEntries(['index.html', 'wasi.js', 'pool.js', 'pool-worker.js', 'player-worker.js', 'source.js', 'coi-serviceworker.js']
    .map((f) => [`/${f}`, `${WEB}/${f}`])),
  '/': `${WEB}/index.html`,
  '/core.wasm': `${ROOT}/work/web/core.wasm`,
  '/core-st.wasm': `${ROOT}/work/web/core-st.wasm`,
  ...Object.fromEntries(Object.entries(CLIPS).map(([c, path]) => [`/clips/${c}`, path])),
};
const TYPES = { html: 'text/html', js: 'text/javascript', wasm: 'application/wasm', mp4: 'video/mp4', mkv: 'video/x-matroska' };

function certificate() {
  const dir = `${ROOT}/work/web/cert`;
  if (!existsSync(`${dir}/cert.pem`)) {
    mkdirSync(dir, { recursive: true });
    execFileSync('openssl', ['req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '825',
      '-keyout', `${dir}/key.pem`, '-out', `${dir}/cert.pem`, '-subj', '/CN=altc264 test'], { stdio: 'ignore' });
  }
  return { key: readFileSync(`${dir}/key.pem`), cert: readFileSync(`${dir}/cert.pem`) };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const port = +process.argv[2] || 8443;
  createServer(certificate(), (req, res) => {
    const path = ROUTES[new URL(req.url, 'https://x').pathname];
    if (!path || !existsSync(path)) {
      res.writeHead(404).end();
      return;
    }
    const size = statSync(path).size;
    const range = /bytes=(\d+)-(\d*)/.exec(req.headers.range ?? '');
    const [start, end] = range ? [+range[1], Math.min(size - 1, range[2] ? +range[2] : size - 1)] : [0, size - 1];
    if (start > end) {
      res.writeHead(416, { 'Content-Range': `bytes */${size}` }).end();
      return;
    }
    res.writeHead(range ? 206 : 200, {
      'Content-Type': TYPES[path.split('.').pop()] ?? 'application/octet-stream',
      'Content-Length': end - start + 1,
      'Accept-Ranges': 'bytes',
      ...(range ? { 'Content-Range': `bytes ${start}-${end}/${size}` } : {}),
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Resource-Policy': 'same-origin',
      'Cache-Control': 'no-store',
    });
    if (req.method === 'HEAD') res.end();
    else createReadStream(path, { start, end }).pipe(res);
  }).listen(port, '0.0.0.0', () => {
    const ips = Object.values(networkInterfaces()).flat().filter((i) => i.family === 'IPv4' && !i.internal);
    for (const i of ips) console.log(`https://${i.address}:${port}/`);
  });
}
