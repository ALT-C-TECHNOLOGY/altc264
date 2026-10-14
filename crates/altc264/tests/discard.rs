//! A seek (Decoder::discard) mid-stream, then decoding from an I picture,
//! gives the same frames as a new decoder given only the parameter sets
//! and then the same data: nothing from before the seek is output or used,
//! and the parameter sets received before it are kept. Uses the ITU
//! conformance streams in work/conformance and skips when they are absent.
use altc264::decoder::Decoder;

fn frames(dec: &mut Decoder) -> Vec<Vec<u8>> {
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

/// (start of the start code, NAL unit type) of each NAL unit.
fn nals(data: &[u8]) -> Vec<(usize, u8)> {
    let mut v = Vec::new();
    let mut i = 0;
    while i + 3 < data.len() {
        if data[i..i + 3] == [0, 0, 1] {
            let start = if i > 0 && data[i - 1] == 0 { i - 1 } else { i };
            v.push((start, data[i + 3] & 31));
            i += 3;
        } else {
            i += 1;
        }
    }
    v
}

/// Whether the slice NAL unit at `at` (its start code) is an I slice.
fn i_slice(data: &[u8], at: usize) -> bool {
    let p = at + if data[at + 2] == 1 { 4 } else { 5 };
    // first_mb_in_slice and slice_type, both ue(v), from the first bytes.
    let mut bit = 0usize;
    let mut read = || {
        let b = data[p + bit / 8] >> (7 - bit % 8) & 1;
        bit += 1;
        b as u32
    };
    let mut ue = || {
        let mut zeros = 0;
        while read() == 0 {
            zeros += 1;
        }
        let mut v = 0;
        for _ in 0..zeros {
            v = v << 1 | read();
        }
        (1 << zeros) - 1 + v
    };
    let _first_mb = ue();
    ue() % 5 == 2
}

/// Byte offsets where an access unit starts with an IDR or I picture of
/// the base view, after the first: the non-VCL NAL units before its first
/// slice belong to it.
fn restart_points(data: &[u8], n: &[(usize, u8)]) -> Vec<usize> {
    let mut v = Vec::new();
    for (k, &(at, t)) in n.iter().enumerate() {
        let vcl = t == 5 || t == 1;
        let first_of_au = k == 0 || !matches!(n[k - 1].1, 1 | 5 | 20);
        if vcl && first_of_au && (t == 5 || i_slice(data, at)) {
            let mut s = k;
            while s > 0 && matches!(n[s - 1].1, 6..=9 | 14 | 15) {
                s -= 1;
            }
            if n[s].0 > 0 {
                v.push(n[s].0);
            }
        }
    }
    v
}

fn check(path: &str) -> bool {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../work/conformance/");
    let full = if path.starts_with('/') {
        path.to_string()
    } else {
        format!("{root}{path}")
    };
    let Ok(data) = std::fs::read(full) else {
        return false;
    };
    let n = nals(&data);
    let points = restart_points(&data, &n);
    // Streams with a single I picture have nowhere to restart.
    let Some(&p) = points.get(points.len() / 2) else {
        eprintln!("{path}: no I picture after the first; skipped");
        return false;
    };
    // Parameter sets before the restart point, as a container's header
    // would have given them.
    let mut params = Vec::new();
    for (k, &(at, t)) in n.iter().enumerate() {
        if at >= p {
            break;
        }
        if matches!(t, 7 | 8 | 15) {
            let end = n.get(k + 1).map_or(data.len(), |x| x.0);
            params.extend_from_slice(&data[at..end]);
        }
    }
    // Cut the stream before the restart point mid-GOP: halfway, at a NAL
    // unit boundary.
    let cut = n.iter().map(|x| x.0).rfind(|&a| a <= p / 2).unwrap_or(0);
    for threads in [0, 1, 4] {
        let mut fresh = Decoder::with_threads(threads);
        fresh.decode_annex_b(&params);
        fresh.decode_annex_b(&data[p..]);
        let want = frames(&mut fresh);
        assert!(
            !want.is_empty(),
            "{path}: no frames after the restart point"
        );

        let mut dec = Decoder::with_threads(threads);
        dec.decode_annex_b(&data[..cut]);
        // Take what is ready, as a player would before seeking.
        while dec.next_frame().is_some() {}
        dec.discard();
        dec.decode_annex_b(&data[p..]);
        let got = frames(&mut dec);
        assert_eq!(
            got.len(),
            want.len(),
            "{path}: frame count, {threads} threads"
        );
        assert!(got == want, "{path}: frames differ, {threads} threads");
    }
    true
}

#[test]
fn discard_then_restart_matches_a_new_decoder() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../work/conformance/");
    if std::fs::metadata(root).is_err() {
        eprintln!("conformance streams not downloaded; skipping");
        return;
    }
    let streams = [
        "AVCv1/SVA_Base_B/SVA_Base_B.264",
        "AVCv1/CABA3_TOSHIBA_E/CABA3_TOSHIBA_E.264",
        "AVCv1/CVSE3_Sony_H/CVSE3_Sony_H.jsv",
        "AVCv1/MR1_BT_A/MR1_BT_A.h264",
        "FRExt/HPCV_BRCM_A/HPCV_BRCM_A.264",
        "MVC/MVCDS-5.264",
        "MVC/MVCICT-1.264",
        "MVC/MVCNV-2.264",
    ];
    let ran: Vec<_> = streams.iter().filter(|s| check(s)).collect();
    // More streams (absolute paths, comma separated), e.g. local clips.
    if let Ok(extra) = std::env::var("ALTC_DISCARD_STREAMS") {
        for p in extra.split(',').filter(|p| !p.is_empty()) {
            assert!(check(p), "{p}: not checked");
        }
    }
    eprintln!("ran: {ran:?}");
    assert!(
        ran.len() >= 3,
        "too few streams with a restart point: {ran:?}"
    );
}
