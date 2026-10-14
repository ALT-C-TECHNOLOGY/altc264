//! `altc264` command-line tool.

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::process::ExitCode;

/// println! that exits quietly when stdout is closed (e.g. piped into `head`).
macro_rules! out {
    ($($t:tt)*) => {
        if writeln!(std::io::stdout().lock(), $($t)*).is_err() {
            std::process::exit(0);
        }
    };
}

use altc264::nal::{AnnexBIter, NalHeader, NalType, unescape_into};
use altc264::poc::{PocState, first_slice_of_new_picture, structure};
use altc264::ps::{Pps, Sps, SubsetSps};
use altc264::slice::SliceHeader;

const USAGE: &str = "\
usage: altc264 nals FILE.264       print one line per NAL unit
       altc264 headers FILE.264    parse parameter sets, print each distinct one
                                   (exit status 1 on any parse or trailing-bit error)
       altc264 slices FILE.264     print one line per slice header:
                                   nal index, nal type, view, first_mb, slice_type, pps,
                                   frame_num, poc lsb, qp delta, header end bit (from NAL start),
                                   structure (F frame, M MBAFF frame, T/B top/bottom field)
       altc264 pocs FILE.264       print one line per picture in decoding order:
                                   view, frame_num, first slice type, PicOrderCnt
       altc264 decode FILE.264 OUT.yuv [OUT2.yuv ...]
                                   decode to planar YUV (4:0:0 written as 4:2:0 grey);
                                   the Nth output receives MVC view order index N-1;
                                   ALTC_STEREO=sbs|sbs-half|tab|tab-half|fp packs
                                   the two views of each access unit into OUT
                                   (ALTC_BASE_RIGHT: the base view is the right eye)
       altc264 offsets FILE.264 [SEQ]
                                   Blu-ray 3D plane offsets per output frame: one
                                   column per offset sequence, or only SEQ
                                   (signed: plane_offset_direction_flag 1 is negative)";

fn read(path: &str) -> Option<Vec<u8>> {
    std::fs::read(path)
        .map_err(|e| eprintln!("{path}: {e}"))
        .ok()
}

fn nals(path: &str) -> ExitCode {
    let Some(data) = read(path) else {
        return ExitCode::FAILURE;
    };
    for (i, nal) in AnnexBIter::new(&data).enumerate() {
        match NalHeader::parse(nal) {
            Ok(h) => {
                let view = h.mvc.map(|m| {
                    format!(
                        " view={} anchor={} inter_view={}",
                        m.view_id, m.anchor_pic as u8, m.inter_view as u8
                    )
                });
                out!(
                    "{i}\t{:?}\tref_idc={}\tlen={}{}",
                    h.nal_type,
                    h.ref_idc,
                    nal.len(),
                    view.unwrap_or_default()
                );
            }
            Err(e) => out!("{i}\terror={e:?}\tlen={}", nal.len()),
        }
    }
    ExitCode::SUCCESS
}

fn describe_sps(s: &Sps) -> String {
    let (w, h) = s.cropped_size();
    let restriction = s
        .vui
        .as_ref()
        .and_then(|v| v.bitstream_restriction.as_ref());
    format!(
        "profile={} level={} chroma={} depth={}/{} {w}x{h} frame_mbs_only={} poc={:?} refs={} scaling={} reorder={:?} dpb={:?}",
        s.profile_idc,
        s.level_idc,
        s.chroma_format_idc,
        s.bit_depth_luma,
        s.bit_depth_chroma,
        s.frame_mbs_only,
        s.poc,
        s.max_num_ref_frames,
        s.scaling.is_some(),
        restriction.map(|r| r.max_num_reorder_frames),
        restriction.map(|r| r.max_dec_frame_buffering),
    )
}

fn headers(path: &str) -> ExitCode {
    let Some(data) = read(path) else {
        return ExitCode::FAILURE;
    };
    let mut sps: HashMap<u8, Sps> = HashMap::new();
    let mut subset: HashMap<u8, SubsetSps> = HashMap::new();
    let mut seen = HashSet::new();
    let (mut errors, mut counts) = (0usize, [0usize; 3]);
    let mut rbsp = Vec::new();
    for (i, nal) in AnnexBIter::new(&data).enumerate() {
        let Ok(h) = NalHeader::parse(nal) else {
            continue;
        };
        if !matches!(h.nal_type, NalType::Sps | NalType::SubsetSps | NalType::Pps) {
            continue;
        }
        if !seen.insert(nal.to_vec()) {
            continue; // repeated identical parameter set
        }
        rbsp.clear();
        unescape_into(&nal[h.len..], &mut rbsp);
        let (line, ok) = match h.nal_type {
            NalType::Sps => {
                counts[0] += 1;
                match Sps::parse(&rbsp) {
                    Ok(s) => {
                        let r = (format!("SPS {} {}", s.id, describe_sps(&s)), s.trailing_ok);
                        sps.insert(s.id, s);
                        r
                    }
                    Err(e) => (format!("SPS error: {e}"), false),
                }
            }
            NalType::SubsetSps => {
                counts[1] += 1;
                match SubsetSps::parse(&rbsp) {
                    Ok(s) => {
                        let refs: Vec<_> = s
                            .mvc
                            .refs
                            .iter()
                            .skip(1)
                            .map(|r| (&r.anchor_l0, &r.non_anchor_l0))
                            .collect();
                        let r = (
                            format!(
                                "SubsetSPS {} {} views={:?} inter_view_l0(anchor, non_anchor)={refs:?} mvc_vui={}",
                                s.sps.id,
                                describe_sps(&s.sps),
                                s.mvc.view_ids,
                                if s.mvc.vui_discarded {
                                    "discarded"
                                } else if s.mvc.vui.is_some() {
                                    "yes"
                                } else {
                                    "no"
                                }
                            ),
                            // A dropped MVC VUI is a known encoder quirk, not a parser problem.
                            s.sps.trailing_ok || s.mvc.vui_discarded,
                        );
                        subset.insert(s.sps.id, s);
                        r
                    }
                    Err(e) => (format!("SubsetSPS error: {e}"), false),
                }
            }
            _ => {
                counts[2] += 1;
                match Pps::parse(&rbsp) {
                    Ok(p) => {
                        // A PPS may serve the base view (SPS) and non-base views (subset SPS).
                        let targets: Vec<(&str, &Sps)> = [
                            ("sps", sps.get(&p.sps_id)),
                            ("subset", subset.get(&p.sps_id).map(|s| &s.sps)),
                        ]
                        .into_iter()
                        .filter_map(|(k, s)| Some((k, s?)))
                        .collect();
                        let mut ok = p.trailing_ok;
                        let mut ext_desc = Vec::new();
                        for (k, s) in &targets {
                            match p.ext(s) {
                                Ok(x) => {
                                    ok &= x.trailing_ok;
                                    ext_desc.push(format!(
                                        "{k}: 8x8={} cqp2={}",
                                        x.transform_8x8_mode, x.second_chroma_qp_index_offset
                                    ));
                                }
                                Err(e) => {
                                    ok = false;
                                    ext_desc.push(format!("{k}: error {e}"));
                                }
                            }
                        }
                        if targets.is_empty() {
                            ext_desc.push(format!("no SPS {} yet", p.sps_id));
                        }
                        (
                            format!(
                                "PPS {} sps={} cabac={} slice_groups={} refs={:?} wp={}/{} qp={} cqp={} [{}]",
                                p.id,
                                p.sps_id,
                                p.entropy_coding_mode,
                                p.num_slice_groups,
                                p.num_ref_idx_default_active,
                                p.weighted_pred,
                                p.weighted_bipred_idc,
                                26 + p.pic_init_qp_minus26,
                                p.chroma_qp_index_offset,
                                ext_desc.join(", ")
                            ),
                            ok,
                        )
                    }
                    Err(e) => (format!("PPS error: {e}"), false),
                }
            }
        };
        if !ok {
            errors += 1;
        }
        out!("{i}\t{}{line}", if ok { "" } else { "!! " });
    }
    out!(
        "distinct: {} SPS, {} subset SPS, {} PPS; {errors} problems",
        counts[0],
        counts[1],
        counts[2]
    );
    if errors > 0 {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

fn slices(path: &str) -> ExitCode {
    let Some(data) = read(path) else {
        return ExitCode::FAILURE;
    };
    let mut sps: HashMap<u8, Sps> = HashMap::new();
    let mut subset: HashMap<u8, Sps> = HashMap::new();
    let mut pps: HashMap<u8, Pps> = HashMap::new();
    let mut errors = 0usize;
    let mut rbsp = Vec::new();
    for (i, nal) in AnnexBIter::new(&data).enumerate() {
        let Ok(h) = NalHeader::parse(nal) else {
            continue;
        };
        rbsp.clear();
        unescape_into(&nal[h.len..], &mut rbsp);
        match h.nal_type {
            NalType::Sps => {
                if let Ok(s) = Sps::parse(&rbsp) {
                    sps.insert(s.id, s);
                }
            }
            NalType::SubsetSps => {
                if let Ok(s) = SubsetSps::parse(&rbsp) {
                    subset.insert(s.sps.id, s.sps);
                }
            }
            NalType::Pps => {
                if let Ok(p) = Pps::parse(&rbsp) {
                    pps.insert(p.id, p);
                }
            }
            NalType::Slice | NalType::SliceIdr | NalType::SliceExtension => {
                let sets = if h.nal_type == NalType::SliceExtension {
                    &subset
                } else {
                    &sps
                };
                let r = SliceHeader::parse(&rbsp, &h, |id| {
                    let p = pps.get(&id)?;
                    Some((sets.get(&p.sps_id)?, p))
                });
                let view = h.mvc.map_or(0, |m| m.view_id);
                match r {
                    Ok(s) => out!(
                        "{i}\t{}\t{view}\t{}\t{:?}{}\t{}\t{}\t{}\t{}\t{}\t{}",
                        h.nal_type as u8,
                        s.first_mb_in_slice,
                        s.slice_type,
                        if s.slice_type_fixed { "*" } else { "" },
                        s.pps_id,
                        s.frame_num,
                        s.pic_order_cnt_lsb,
                        s.slice_qp_delta,
                        s.data_bit_offset + h.len * 8,
                        match (s.field_pic, s.bottom_field) {
                            (true, false) => "T",
                            (true, true) => "B",
                            _ if sets
                                .get(&pps[&s.pps_id].sps_id)
                                .is_some_and(|q| q.mb_adaptive_frame_field) =>
                                "M",
                            _ => "F",
                        }
                    ),
                    Err(e) => {
                        errors += 1;
                        out!("{i}\t{}\t{view}\terror: {e}", h.nal_type as u8);
                    }
                }
            }
            _ => {}
        }
    }
    if errors > 0 {
        eprintln!("{errors} slice header errors");
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

/// Parameter-set bookkeeping shared by the slice-level commands.
#[derive(Default)]
struct Sets {
    sps: HashMap<u8, Sps>,
    subset: HashMap<u8, Sps>,
    pps: HashMap<u8, Pps>,
}

impl Sets {
    /// Handles a parameter set NAL; returns false for other NAL types.
    fn update(&mut self, nal_type: NalType, rbsp: &[u8]) -> bool {
        match nal_type {
            NalType::Sps => {
                if let Ok(s) = Sps::parse(rbsp) {
                    self.sps.insert(s.id, s);
                }
            }
            NalType::SubsetSps => {
                if let Ok(s) = SubsetSps::parse(rbsp) {
                    self.subset.insert(s.sps.id, s.sps);
                }
            }
            NalType::Pps => {
                if let Ok(p) = Pps::parse(rbsp) {
                    self.pps.insert(p.id, p);
                }
            }
            _ => return false,
        }
        true
    }

    /// Parses a slice header, returning it with the SPS it activates.
    fn slice(&self, h: &NalHeader, rbsp: &[u8]) -> Result<(SliceHeader, Sps), altc264::Error> {
        let sets = if h.nal_type == NalType::SliceExtension {
            &self.subset
        } else {
            &self.sps
        };
        let mut used = None;
        let s = SliceHeader::parse(rbsp, h, |id| {
            let p = self.pps.get(&id)?;
            let s = sets.get(&p.sps_id)?;
            used = Some(s);
            Some((s, p))
        })?;
        Ok((s, used.unwrap().clone()))
    }
}

fn pocs(path: &str) -> ExitCode {
    let Some(data) = read(path) else {
        return ExitCode::FAILURE;
    };
    let mut sets = Sets::default();
    // Per view: POC state and the previous slice (with its nal_ref_idc).
    let mut views: HashMap<u16, (PocState, Option<(SliceHeader, u8)>)> = HashMap::new();
    let mut rbsp = Vec::new();
    let mut errors = 0;
    for nal in AnnexBIter::new(&data) {
        let Ok(h) = NalHeader::parse(nal) else {
            continue;
        };
        rbsp.clear();
        unescape_into(&nal[h.len..], &mut rbsp);
        if sets.update(h.nal_type, &rbsp)
            || !matches!(
                h.nal_type,
                NalType::Slice | NalType::SliceIdr | NalType::SliceExtension
            )
        {
            continue;
        }
        let (s, sps) = match sets.slice(&h, &rbsp) {
            Ok(v) => v,
            Err(_) => {
                errors += 1;
                continue;
            }
        };
        let view = h.mvc.map_or(0, |m| m.view_id);
        let (state, prev) = views.entry(view).or_default();
        let new_pic = match prev {
            Some((p, p_ref)) => first_slice_of_new_picture(&sps, p, *p_ref, &s, h.ref_idc),
            None => true,
        };
        if new_pic {
            let poc = state.next(&sps, &s, h.ref_idc);
            out!(
                "{view}\t{}\t{:?}\t{}",
                s.frame_num,
                s.slice_type,
                poc.pic(structure(&s))
            );
        }
        *prev = Some((s, h.ref_idc));
    }
    if errors > 0 {
        eprintln!("{errors} slice header errors");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn decode(path: &str, out_paths: &[String]) -> ExitCode {
    let Some(data) = read(path) else {
        return ExitCode::FAILURE;
    };
    // Frames for /dev/null are not written (for timing the decoder alone).
    let mut outs = Vec::new();
    let discard: Vec<bool> = out_paths.iter().map(|p| p == "/dev/null").collect();
    for p in out_paths {
        match std::fs::File::create(p) {
            Ok(f) => outs.push(std::io::BufWriter::new(f)),
            Err(e) => {
                eprintln!("{p}: {e}");
                return ExitCode::FAILURE;
            }
        }
    }
    // ALTC_THREADS=n picks a thread count (0: the calling thread); one per
    // CPU by default.
    let mut dec = match std::env::var("ALTC_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        Some(n) => altc264::decoder::Decoder::with_threads(n),
        None => altc264::decoder::Decoder::with_auto_threads(),
    };
    // Without a second output there is no need to decode non-base views,
    // unless they are packed into the one output.
    use altc264::stereo::StereoLayout::*;
    dec.stereo_layout = match std::env::var("ALTC_STEREO").as_deref() {
        Ok("sbs") => Some(SideBySide),
        Ok("sbs-half") => Some(SideBySideHalf),
        Ok("tab") => Some(TopAndBottom),
        Ok("tab-half") => Some(TopAndBottomHalf),
        Ok("fp") => Some(FramePacking),
        _ => None,
    };
    dec.decode_all_views = outs.len() > 1 || dec.stereo_layout.is_some();
    dec.base_view_right = std::env::var_os("ALTC_BASE_RIGHT").is_some();
    let start = std::time::Instant::now();
    let mut frames = vec![0usize; outs.len()];
    let mut size = (0, 0);
    let mut write = |f: altc264::decoder::Frame| -> std::io::Result<()> {
        let v = f.view_index;
        let Some(out) = outs.get_mut(v) else {
            return Ok(());
        };
        frames[v] += 1;
        size = (f.width, f.height);
        if discard[v] {
            return Ok(());
        }
        // Samples of more than 8 bits are written as 16-bit little endian,
        // as JM writes them.
        let high = f.bit_depth.iter().any(|&b| b > 8);
        let mut buf = Vec::new();
        // GBR frames (matrix_coefficients 0) are written as R, G, B like JM.
        let gbr = f.colour.is_some_and(|c| c.2 == 0) && f.plane(1).is_some();
        let order = if gbr { [2, 0, 1] } else { [0, 1, 2] };
        for i in order {
            match f.plane(i) {
                Some(p) if high => {
                    buf.clear();
                    for row in p.rows16() {
                        buf.extend(row.iter().flat_map(|v| v.to_le_bytes()));
                    }
                    out.write_all(&buf)?;
                }
                Some(p) => {
                    for row in p.rows() {
                        out.write_all(row)?;
                    }
                }
                None if high => {
                    let grey = (1u16 << (f.bit_depth[1] - 1)).to_le_bytes();
                    let n = (f.width / 2) * (f.height / 2);
                    out.write_all(&grey.repeat(n))?;
                }
                None => out.write_all(&vec![128u8; (f.width / 2) * (f.height / 2)])?,
            }
        }
        Ok(())
    };
    let mut drain = |dec: &mut altc264::decoder::Decoder| {
        while let Some(f) = dec.next_frame() {
            if let Err(e) = write(f) {
                eprintln!("{}: {e}", out_paths[0]);
                return false;
            }
        }
        true
    };
    for nal in AnnexBIter::new(&data) {
        dec.decode_nal(nal);
        if !drain(&mut dec) {
            return ExitCode::FAILURE;
        }
    }
    dec.flush();
    if !drain(&mut dec) {
        return ExitCode::FAILURE;
    }
    let secs = start.elapsed().as_secs_f64();
    let other: Vec<String> = frames[1..].iter().map(|n| n.to_string()).collect();
    eprintln!(
        "{} frames {}x{} in {secs:.3} s ({:.1} fps), {} second-view frames, {} slice errors{}",
        frames[0],
        size.0,
        size.1,
        frames[0] as f64 / secs,
        if other.is_empty() {
            "0".into()
        } else {
            other.join("/")
        },
        dec.errors(),
        dec.last_error()
            .map(|e| format!(", last: {e}"))
            .unwrap_or_default()
    );
    if dec.errors() > 0 {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

fn offsets(path: &str, seq: Option<usize>) -> ExitCode {
    let Some(data) = read(path) else {
        return ExitCode::FAILURE;
    };
    let mut dec = altc264::decoder::Decoder::new();
    dec.decode_all_views = false;
    let mut n = 0usize;
    let mut print = |f: altc264::decoder::Frame| {
        let Some(o) = f.plane_offsets else {
            out!("{n}\tpoc {}\t-", f.poc);
            n += 1;
            return;
        };
        let v = |p: &altc264::sei::PlaneOffset| {
            if p.direction {
                -(p.value as i32)
            } else {
                p.value as i32
            }
        };
        let cols: Vec<String> = match seq {
            Some(k) => o.get(k).map(|p| v(p).to_string()).into_iter().collect(),
            None => o.iter().map(|p| v(p).to_string()).collect(),
        };
        out!("{n}\tpoc {}\t{}", f.poc, cols.join(" "));
        n += 1;
    };
    for nal in AnnexBIter::new(&data) {
        dec.decode_nal(nal);
        while let Some(f) = dec.next_frame() {
            print(f);
        }
    }
    dec.flush();
    while let Some(f) = dec.next_frame() {
        print(f);
    }
    ExitCode::SUCCESS
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [cmd, path] if cmd == "nals" => nals(path),
        [cmd, path] if cmd == "headers" => headers(path),
        [cmd, path] if cmd == "slices" => slices(path),
        [cmd, path] if cmd == "pocs" => pocs(path),
        [cmd, path] if cmd == "offsets" => offsets(path, None),
        [cmd, path, k] if cmd == "offsets" => offsets(path, k.parse().ok()),
        [cmd, path, outs @ ..] if cmd == "decode" && !outs.is_empty() => decode(path, outs),
        _ => {
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
}
