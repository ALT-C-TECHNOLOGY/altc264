//! A mutation fuzzer: decodes damaged copies of streams (bit flips, byte
//! runs, truncation, dropped NAL units) and reports any panic.
//!
//! cargo run --release -p altc264 --example fuzz -- [--threads N] [--rounds N] FILE...
use std::panic::{AssertUnwindSafe, catch_unwind};

use altc264::decoder::Decoder;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

fn mutate(data: &[u8], rng: &mut Rng) -> Vec<u8> {
    let mut d = data.to_vec();
    for _ in 0..1 + rng.below(8) {
        if d.is_empty() {
            break;
        }
        match rng.below(5) {
            0 => {
                let i = rng.below(d.len());
                d[i] ^= 1 << rng.below(8);
            }
            1 => {
                let i = rng.below(d.len());
                let n = rng.below(64).min(d.len() - i);
                let v = rng.next() as u8;
                d[i..i + n].fill(v);
            }
            2 => {
                let keep = rng.below(d.len());
                d.truncate(keep.max(1));
            }
            3 => {
                // Drop a range (often whole NAL units).
                let i = rng.below(d.len());
                let n = rng.below(4096).min(d.len() - i);
                d.drain(i..i + n);
            }
            _ => {
                let i = rng.below(d.len());
                let n = rng.below(256).min(d.len() - i);
                let chunk: Vec<u8> = d[i..i + n].to_vec();
                let j = rng.below(d.len());
                d.splice(j..j, chunk);
            }
        }
    }
    d
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut opt = |name: &str, default: usize| {
        args.iter().position(|a| a == name).map_or(default, |i| {
            let v = args[i + 1].parse().unwrap();
            args.drain(i..i + 2);
            v
        })
    };
    let threads = opt("--threads", 0);
    let rounds = opt("--rounds", 200);
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut panics = 0;
    for path in &args {
        let data = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        // Keep inputs small so many rounds run quickly.
        let data = &data[..data.len().min(2 << 20)];
        for round in 0..rounds {
            let input = mutate(data, &mut rng);
            let r = catch_unwind(AssertUnwindSafe(|| {
                let mut dec = Decoder::with_threads(threads);
                dec.decode_annex_b(&input);
                dec.flush();
                let mut n = 0;
                while dec.next_frame().is_some() {
                    n += 1;
                }
                n
            }));
            if r.is_err() {
                panics += 1;
                let out = format!("work/tmp/fuzz-crash-{panics}.264");
                let _ = std::fs::write(&out, &input);
                eprintln!("panic: {path} round {round}, input saved to {out}");
            }
        }
    }
    println!("{panics} panics");
}
