//! Micro-benchmark of the CABAC decision engine: decodes bins from
//! pseudo-random data through a residual-like context pattern.
//!
//! cargo run --release -p altc264 --example cabac_bench
use altc264::cabac::{Ctx, Decoder};
use std::time::Instant;

fn main() {
    let mut seed = 0x1234_5678u32;
    let data: Vec<u8> = (0..8 << 20)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed as u8
        })
        .collect();
    let mut best = f64::MAX;
    for _ in 0..5 {
        let mut ctx = [Ctx::new(20, true); 64];
        let mut d = Decoder::new(&data, 0);
        let start = Instant::now();
        let (mut bins, mut ones) = (0u64, 0u64);
        // Significance-map-like: sig/last pairs over 16 positions, then levels.
        while d.bit_pos() < data.len() * 8 - 1024 {
            for i in 0..15 {
                bins += 1;
                if d.decision(&mut ctx[i]) {
                    ones += 1;
                    bins += 1;
                    if d.decision(&mut ctx[16 + i]) {
                        break;
                    }
                }
            }
            for k in 0..4 {
                bins += 2;
                ones += d.decision(&mut ctx[32 + k]) as u64 + d.bypass() as u64;
            }
        }
        let ns = start.elapsed().as_secs_f64() * 1e9 / bins as f64;
        best = best.min(ns);
        std::hint::black_box(ones);
    }
    println!("{best:.3} ns/bin");
}
