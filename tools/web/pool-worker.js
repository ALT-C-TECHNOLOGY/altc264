import { poolWorker } from './pool.js';

onmessage = (e) => poolWorker(e.data);
