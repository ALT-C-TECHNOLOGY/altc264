//! Prints the output order (view, POC, pts) of a stream's frames.
//!
//! cargo run --release -p altc264 --example output_order FILE.264 [N]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&args[1]).expect("read");
    let n: usize = args
        .get(2)
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX);
    let mut dec = altc264::decoder::Decoder::new();
    dec.decode_annex_b(&data);
    dec.flush();
    let mut k = 0;
    while let Some(f) = dec.next_frame() {
        if k < n {
            println!("{k}\tview {}\tpoc {}", f.view_id, f.poc);
        }
        k += 1;
    }
    println!("{k} frames");
}
