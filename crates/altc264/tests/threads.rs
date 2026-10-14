//! Threaded decoding gives the same frames as single-threaded decoding.
//! Uses the ITU conformance streams in work/conformance (see `cargo xtask
//! conformance`) and skips when they are absent. Small enough for Miri:
//! cargo +nightly miri test -p altc264 --test threads (with
//! MIRIFLAGS=-Zmiri-disable-isolation) checks the unsafe slice and
//! deblocking parallelism for data races and aliasing errors.
use altc264::decoder::Decoder;

fn frames(data: &[u8], threads: usize) -> Vec<Vec<u8>> {
    frames_with(data, threads, true)
}

fn frames_with(data: &[u8], threads: usize, frame_threading: bool) -> Vec<Vec<u8>> {
    let mut dec = Decoder::with_threads(threads);
    dec.frame_threading = frame_threading;
    dec.decode_annex_b(data);
    dec.flush();
    let mut out = Vec::new();
    while let Some(f) = dec.next_frame() {
        let mut v = Vec::new();
        for i in 0..3 {
            if let Some(p) = f.plane(i) {
                for row in p.rows() {
                    v.extend_from_slice(row);
                }
            }
        }
        out.push(v);
    }
    out
}

/// About the first quarter of a stream, ending at a NAL unit boundary (a
/// slice cut short decodes differently with frame threading, which keeps
/// the rows it has already handed to other pictures).
fn quarter(data: &[u8]) -> &[u8] {
    let end = (3..data.len() / 4)
        .rev()
        .find(|&i| data[i - 3..i] == [0, 0, 1])
        .map_or(data.len() / 4, |i| i - 3);
    &data[..end]
}

fn stream(name: &str) -> Option<Vec<u8>> {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../work/conformance/AVCv1/");
    std::fs::read(format!("{root}{name}")).ok()
}

#[test]
fn parallel_slices_match_sequential() {
    // Frame threading off: the slices of one picture at a time.
    // Three slices per picture; with 3 threads every slice and row of
    // macroblocks of a picture is decoded concurrently.
    let Some(data) = stream("SVA_Base_B/SVA_Base_B.264") else {
        eprintln!("conformance streams not downloaded; skipping");
        return;
    };
    // Under Miri a few pictures are enough.
    let data = if cfg!(miri) {
        quarter(&data)
    } else {
        &data[..]
    };
    let want = frames(data, 0);
    assert!(!want.is_empty());
    assert_eq!(frames_with(data, 3, false), want);
    // Frame threading: several pictures at once, each reading the rows of
    // its references as they are finished.
    assert_eq!(frames(data, 3), want);
}

#[test]
fn frame_threads_match_sequential_with_b_pictures() {
    // CABAC with B slices: direct prediction reads the co-located rows of
    // pictures still being decoded.
    let Some(data) = stream("CABA3_SVA_B/CABA3_SVA_B.264") else {
        eprintln!("conformance streams not downloaded; skipping");
        return;
    };
    let data = if cfg!(miri) {
        quarter(&data)
    } else {
        &data[..]
    };
    let want = frames(data, 0);
    assert!(!want.is_empty());
    assert_eq!(frames(data, 4), want);
}
