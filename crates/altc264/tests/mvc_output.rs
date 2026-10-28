//! MVC output when decoding ends or starts part-way through a GOP, as when
//! a player or transcoder drains the decoder at a segment boundary or seeks:
//! the frames come out in output order with the timestamps of their access
//! units, the views of each access unit together, as the base view alone
//! would. Uses the ITU conformance streams in work/conformance and skips
//! when they are absent.
use altc264::decoder::Decoder;
use altc264::nal::AnnexBIter;
use altc264::stereo::StereoLayout;

fn stream(name: &str) -> Option<Vec<u8>> {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../work/conformance/MVC/");
    std::fs::read(format!("{root}{name}")).ok()
}

/// Coded slice extension: a non-base view component.
const EXT: u8 = 20;

fn nal_type(nal: &[u8]) -> u8 {
    nal[0] & 31
}

/// The NAL units of each access unit of a two-view stream whose access
/// units all end with the non-base view: one starts at the first NAL unit
/// after a slice extension that is not one (7.4.1.2.3).
fn access_units(data: &[u8]) -> Vec<Vec<&[u8]>> {
    let mut aus: Vec<Vec<&[u8]>> = Vec::new();
    let mut after_ext = true;
    for nal in AnnexBIter::new(data) {
        let t = nal_type(nal);
        if after_ext && t != EXT {
            aus.push(Vec::new());
        }
        after_ext = t == EXT;
        aus.last_mut().unwrap().push(nal);
    }
    for au in &aus {
        assert!(au.iter().any(|n| matches!(nal_type(n), 1 | 5)));
        assert!(au.iter().any(|n| nal_type(n) == EXT));
    }
    aus
}

/// Packets as mkvmerge stores an MVC elementary stream: each holds the
/// non-base view of the previous access unit, then the base view of its
/// own, so a cut between packets leaves the last access unit without its
/// non-base view and a seek starts with one whose base view is missing.
fn mkvmerge_packets<'a>(aus: &[Vec<&'a [u8]>]) -> Vec<Vec<&'a [u8]>> {
    (0..aus.len())
        .map(|k| {
            let mut p: Vec<&[u8]> = match k {
                0 => Vec::new(),
                _ => aus[k - 1]
                    .iter()
                    .copied()
                    .filter(|n| nal_type(n) == EXT)
                    .collect(),
            };
            p.extend(aus[k].iter().copied().filter(|n| nal_type(n) != EXT));
            p
        })
        .collect()
}

/// Parameter sets, as a container's codec configuration.
fn parameter_sets<'a>(aus: &[Vec<&'a [u8]>]) -> Vec<&'a [u8]> {
    aus.iter()
        .flatten()
        .copied()
        .filter(|n| matches!(nal_type(n), 7 | 8 | 15))
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Out {
    view: usize,
    poc: i32,
    pts: Option<i64>,
}

#[derive(Clone, Copy)]
enum Views {
    Base,
    Separate,
    SideBySide,
}

/// Decodes `packets` (timestamp, NAL units), with a seek (Decoder::discard)
/// before the packet at each index in `seeks`, and drains the decoder: the
/// frames after the last seek.
fn decode(
    params: &[&[u8]],
    packets: &[(i64, Vec<&[u8]>)],
    seeks: &[usize],
    views: Views,
    threads: usize,
) -> Vec<Out> {
    let mut dec = Decoder::with_threads(threads);
    dec.decode_all_views = !matches!(views, Views::Base);
    if let Views::SideBySide = views {
        dec.stereo_layout = Some(StereoLayout::SideBySide);
    }
    for nal in params {
        dec.decode_nal(nal);
    }
    let mut out = Vec::new();
    let take = |dec: &mut Decoder, out: &mut Vec<Out>| {
        while let Some(f) = dec.next_frame() {
            out.push(Out {
                view: f.view_index,
                poc: f.poc,
                pts: f.pts,
            });
        }
    };
    for (i, (pts, nals)) in packets.iter().enumerate() {
        if seeks.contains(&i) {
            dec.discard();
            // Which frames were ready before the seek depends on timing.
            out.clear();
        }
        dec.set_timestamp(Some(*pts));
        for nal in nals {
            dec.decode_nal(nal);
        }
        take(&mut dec, &mut out);
    }
    dec.flush();
    take(&mut dec, &mut out);
    assert_eq!(dec.errors(), 0, "slice errors");
    out
}

/// Checks the MVC output of `packets` against the base view's: the same
/// frames in the same order with the same timestamps, one packed frame per
/// access unit side by side, and separately each base frame followed by
/// the non-base frame of its access unit (when `has_second(pts)`).
fn check(
    what: &str,
    params: &[&[u8]],
    packets: &[(i64, Vec<&[u8]>)],
    seeks: &[usize],
    has_second: impl Fn(i64) -> bool,
) {
    for threads in [0, 1, 4] {
        let base = decode(params, packets, seeks, Views::Base, threads);
        assert!(!base.is_empty(), "{what}: no frames");
        let sbs = decode(params, packets, seeks, Views::SideBySide, threads);
        let pts: Vec<_> = base.iter().map(|f| f.pts).collect();
        let sbs_pts: Vec<_> = sbs.iter().map(|f| f.pts).collect();
        assert_eq!(sbs_pts, pts, "{what}, {threads} threads: side by side");
        let separate = decode(params, packets, seeks, Views::Separate, threads);
        let want: Vec<Out> = base
            .iter()
            .flat_map(|&b| {
                let second = has_second(b.pts.unwrap()).then_some(Out { view: 1, ..b });
                std::iter::once(b).chain(second)
            })
            .collect();
        assert_eq!(separate, want, "{what}, {threads} threads: separate views");
    }
}

/// Access unit packets with their index as timestamp, the last one cut
/// after its base view when `cut` is set.
fn packets<'a>(aus: &[Vec<&'a [u8]>], n: usize, cut: bool) -> Vec<(i64, Vec<&'a [u8]>)> {
    let mut p: Vec<(i64, Vec<&[u8]>)> = aus[..n]
        .iter()
        .enumerate()
        .map(|(i, au)| (i as i64, au.clone()))
        .collect();
    if cut {
        p[n - 1].1.retain(|nal| nal_type(nal) != EXT);
    }
    p
}

fn drained_mid_gop(name: &str, cuts: impl Iterator<Item = usize>) {
    let Some(data) = stream(name) else {
        eprintln!("conformance streams not downloaded; skipping");
        return;
    };
    let aus = access_units(&data);
    let params = parameter_sets(&aus);
    for n in cuts {
        for cut in [false, true] {
            let what = format!("{name} drained after {n} access units (cut: {cut})");
            let last = n as i64 - 1;
            check(&what, &params, &packets(&aus, n, cut), &[], |pts| {
                !cut || pts != last
            });
        }
    }
}

#[test]
fn drained_mid_gop_outputs_in_order() {
    // Hierarchical B pictures, 9 access units: output order 0 8 4 2 6 1 3 5 7
    // of decoding order, so every cut leaves pictures waiting in the DPB.
    drained_mid_gop("MVCDS-4.264", 1..=9);
    // An I picture every 4 access units: every place in that pattern, twice
    // (kept short for instrumented scalar builds).
    drained_mid_gop("MVCDS2.264", 1..=8);
}

#[test]
fn views_of_an_access_unit_share_its_timestamp() {
    let Some(data) = stream("MVCDS2.264") else {
        eprintln!("conformance streams not downloaded; skipping");
        return;
    };
    let aus = access_units(&data);
    let params = parameter_sets(&aus);
    let packets: Vec<(i64, Vec<&[u8]>)> = mkvmerge_packets(&aus)
        .into_iter()
        .enumerate()
        .map(|(i, p)| (i as i64, p))
        .collect();
    // Drained between packets: the last access unit has no non-base view.
    for n in 1..=8 {
        let last = n as i64 - 1;
        check(
            &format!("mkvmerge packets, drained after {n}"),
            &params,
            &packets[..n],
            &[],
            |pts| pts != last,
        );
    }
}

#[test]
fn seek_into_split_access_units() {
    let Some(data) = stream("MVCDS2.264") else {
        eprintln!("conformance streams not downloaded; skipping");
        return;
    };
    let aus = access_units(&data);
    let params = parameter_sets(&aus);
    let packets: Vec<(i64, Vec<&[u8]>)> = mkvmerge_packets(&aus)
        .into_iter()
        .enumerate()
        .map(|(i, p)| (i as i64, p))
        .collect();
    // From packet 10, seek to the I picture of access unit 15 (POC 16),
    // which starts with the non-base view of access unit 14: it cannot be
    // decoded without its base view and is not output.
    for end in [16, 19, 21, 27] {
        let mut p = packets[..10].to_vec();
        p.extend(packets[15..end].iter().cloned());
        let last = end as i64 - 1;
        check(
            &format!("seek to packet 15, drained after {end}"),
            &params,
            &p,
            &[10],
            |pts| pts != last,
        );
    }
}
