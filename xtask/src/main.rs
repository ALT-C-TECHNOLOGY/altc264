//! Project tasks: conformance testing, reference verification and stream tools.
//!
//! All outputs go under `work/` (gitignored). The reference decoders (JM,
//! JMVC), needed only by `conformance-manifest`, `conformance-run` and
//! `verify`, come from `tools/build-reference.sh`, which installs into
//! `third_party/install/bin`.

use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};

use serde_json::{Value, json};

type Result<T = ()> = std::result::Result<T, String>;

const USAGE: &str = "\
usage: cargo xtask <command>

  conformance [--bitstreams-only]      download ITU-T H.264.1 suites -> work/conformance/index.json
                                       (--bitstreams-only drops the reference YUVs after unpacking)
  conformance-check [--smoke] [--threads N]
                                       decode the suites and compare with conformance/manifest.json
                                       (bitstreams only; no reference decoder needed)
  conformance-manifest                 regenerate conformance/manifest.json from the reference YUVs
                                       and JM (maintainers)
  conformance-run [FILTER]             compare frame by frame with the reference YUVs or JM
  coverage [--streams smoke|full|none] [--html] [--lcov FILE]
                                       line coverage of the unit tests and the conformance streams
                                       (needs cargo-llvm-cov); prints a summary by default
  clip FILE.mkv START DURATION NAME    cut the video track to work/clips/NAME.264 (Annex B, MVC kept)
  check-headers FILE.264               compare base-view slice headers with ffmpeg's trace_headers
  check-poc CLIP.264                   compare picture order counts with the JM log (run verify first)
  verify CLIP.264                      compare every view with the JM (or JMVC) reference decoder";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let r = match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["clip", file, start, dur, name] => clip(Path::new(file), start, dur, name),
        ["conformance"] => conformance(false),
        ["conformance", "--bitstreams-only"] => conformance(true),
        ["conformance-check", rest @ ..] => conformance_check(rest),
        ["conformance-manifest"] => conformance_manifest(),
        ["coverage", rest @ ..] => coverage(rest),
        ["conformance-run"] => conformance_run(""),
        ["conformance-run", filter] => conformance_run(filter),
        ["check-headers", file] => check_headers(Path::new(file)),
        ["check-poc", clip] => check_poc(Path::new(clip)),
        ["verify", clip] => verify(Path::new(clip)),
        _ => Err(USAGE.into()),
    };
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

fn work(sub: &str) -> Result<PathBuf> {
    let p = root().join("work").join(sub);
    fs::create_dir_all(&p).map_err(|e| format!("{}: {e}", p.display()))?;
    Ok(p)
}

fn tool(name: &str) -> Result<PathBuf> {
    let p = root().join("third_party/install/bin").join(name);
    if p.exists() {
        Ok(p)
    } else {
        Err(format!(
            "{} missing; run tools/build-reference.sh",
            p.display()
        ))
    }
}

fn run_json(cmd: &mut Command) -> Result<Value> {
    let out = cmd
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("{cmd:?}: {e}"))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(err.lines().last().unwrap_or("failed").trim().to_string());
    }
    serde_json::from_slice(&out.stdout).map_err(|e| format!("bad JSON from {cmd:?}: {e}"))
}

// ---------------------------------------------------------------- clip

fn clip(file: &Path, start: &str, dur: &str, name: &str) -> Result {
    let out = work("clips")?.join(format!("{name}.264"));
    let st = Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-ss", start, "-i"])
        .arg(file)
        .args([
            "-t",
            dur,
            "-map",
            "0:v:0",
            "-c",
            "copy",
            "-bsf:v",
            "h264_mp4toannexb",
            "-f",
            "h264",
        ])
        .arg(&out)
        .status()
        .map_err(|e| e.to_string())?;
    if !st.success() {
        return Err("ffmpeg failed".into());
    }
    let meta = json!({ "source": file, "start": start, "duration": dur });
    fs::write(
        out.with_extension("json"),
        serde_json::to_string_pretty(&meta).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    println!("{}", out.display());
    Ok(())
}

// ---------------------------------------------------------------- conformance

const CONFORMANCE_URL: &str = "https://www.itu.int/wftp3/av-arch/jvt-site/draft_conformance";
const CONFORMANCE_SETS: &[&str] = &["AVCv1", "FRExt", "MVC", "Professional_profiles"];
const BITSTREAM_EXTENSIONS: &[&str] = &["264", "26l", "jsv", "jvt", "avc", "h264", "bit", "bits"];

fn curl(url: &str, out: Option<&Path>) -> Result<Vec<u8>> {
    let mut cmd = Command::new("curl");
    cmd.args(["-sfL", "--retry", "3", url]);
    if let Some(p) = out {
        cmd.arg("-o").arg(p);
    }
    let o = cmd.output().map_err(|e| format!("curl: {e}"))?;
    if !o.status.success() {
        return Err(format!("download failed: {url}"));
    }
    Ok(o.stdout)
}

/// Downloads and unpacks each suite (skipping files already present), then
/// writes an index pairing every bitstream with the reference YUV shipped next to it.
fn conformance(bitstreams_only: bool) -> Result {
    let base = work("conformance")?;
    let mut index = Vec::new();
    for set in CONFORMANCE_SETS {
        let dir = work(&format!("conformance/{set}"))?;
        let listing = String::from_utf8_lossy(&curl(&format!("{CONFORMANCE_URL}/{set}/"), None)?)
            .into_owned();
        let files: Vec<&str> = listing
            .split("HREF=\"")
            .skip(1)
            .filter_map(|s| s.split('"').next())
            .filter(|h| !h.ends_with('/'))
            .filter_map(|h| h.rsplit('/').next())
            .collect();
        // The server is slow per connection, so fetch several files at once.
        let next = std::sync::atomic::AtomicUsize::new(0);
        let errors = std::sync::Mutex::new(Vec::new());
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    while let Some(f) =
                        files.get(next.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
                    {
                        if let Err(e) = fetch_conformance_file(&dir, set, f, bitstreams_only) {
                            errors.lock().unwrap().push(e);
                        }
                    }
                });
            }
        });
        if let Some(e) = errors.into_inner().unwrap().into_iter().next() {
            return Err(e);
        }
        let mut found = Vec::new();
        collect_files(&dir, &mut found);
        found.retain(|p| !p.to_string_lossy().contains(".unzipping"));
        found.sort();
        for p in &found {
            let ext = p
                .extension()
                .and_then(|x| x.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if !BITSTREAM_EXTENSIONS.contains(&ext.as_str()) {
                continue;
            }
            let folder = p.parent().unwrap();
            let mut yuv: Vec<PathBuf> = fs::read_dir(folder)
                .into_iter()
                .flatten()
                .flatten()
                .map(|e| e.path())
                .filter(|q| {
                    q.extension()
                        .and_then(|x| x.to_str())
                        .is_some_and(|x| x.eq_ignore_ascii_case("yuv"))
                })
                .collect();
            yuv.sort();
            index.push(json!({
                "set": set,
                "name": p.file_stem().map(|s| s.to_string_lossy()),
                "bitstream": p.strip_prefix(&base).unwrap_or(p),
                "reference_yuv": yuv.iter().map(|q| q.strip_prefix(&base).unwrap_or(q).to_path_buf()).collect::<Vec<_>>(),
            }));
        }
    }
    let out = base.join("index.json");
    fs::write(&out, serde_json::to_string_pretty(&index).unwrap()).map_err(|e| e.to_string())?;
    for set in CONFORMANCE_SETS {
        let n = index.iter().filter(|e| e["set"] == *set).count();
        let with_ref = index
            .iter()
            .filter(|e| {
                e["set"] == *set && e["reference_yuv"].as_array().is_some_and(|a| !a.is_empty())
            })
            .count();
        println!("{set:<22} {n:>4} bitstreams, {with_ref:>4} with reference YUV");
    }
    println!("-> {}", out.display());
    Ok(())
}

/// Downloads one file of a suite and unpacks it if it is a zip. Both steps
/// go through temporary names so an interrupted run leaves nothing half-done.
fn fetch_conformance_file(dir: &Path, set: &str, f: &str, bitstreams_only: bool) -> Result {
    let ext = f.rsplit_once('.').map_or("", |x| x.1).to_ascii_lowercase();
    if bitstreams_only && ext == "yuv" {
        return Ok(());
    }
    let dst = dir.join(f);
    if !dst.exists() {
        eprintln!("{set}/{f}");
        let part = dir.join(format!("{f}.part"));
        curl(&format!("{CONFORMANCE_URL}/{set}/{f}"), Some(&part))?;
        fs::rename(&part, &dst).map_err(|e| e.to_string())?;
    }
    if f.to_ascii_lowercase().ends_with(".zip") {
        let unpacked = dir.join(f.rsplit_once('.').unwrap().0);
        if !unpacked.exists() {
            let tmp = dir.join(format!("{f}.unzipping"));
            let _ = fs::remove_dir_all(&tmp);
            let st = Command::new("unzip")
                .args(["-q", "-o"])
                .arg(&dst)
                .arg("-d")
                .arg(&tmp)
                .status()
                .map_err(|e| e.to_string())?;
            if st.success() {
                if bitstreams_only {
                    remove_yuv(&tmp);
                }
                fs::rename(&tmp, &unpacked).map_err(|e| e.to_string())?;
                if bitstreams_only {
                    // The zip is only kept so a rerun skips the download.
                    let _ = fs::write(&dst, b"");
                }
            } else {
                eprintln!("warning: could not unzip {f}");
            }
        }
    }
    Ok(())
}

/// Decodes every indexed stream that ships reference YUV and compares it
/// frame by frame. Writes work/conformance/results.json.
fn conformance_run(filter: &str) -> Result {
    let base = work("conformance")?;
    let index: Value = serde_json::from_slice(
        &fs::read(base.join("index.json")).map_err(|_| "run `cargo xtask conformance` first")?,
    )
    .map_err(|e| e.to_string())?;
    build_cli()?;
    let bin = cli_bin();
    let tmp = work("tmp/conformance")?;
    let mut results = Vec::new();
    let (mut exact, mut total) = (0, 0);
    for e in index.as_array().into_iter().flatten() {
        let name = e["name"].as_str().unwrap_or_default();
        if !name.contains(filter) {
            continue;
        }
        let stream = base.join(e["bitstream"].as_str().unwrap_or_default());
        let mut refs = e["reference_yuv"].as_array().cloned().unwrap_or_default();
        // Streams that ship no reference YUV are checked against JM.
        let from_jm = refs.is_empty();
        if from_jm {
            match jm_conformance_yuv(name, &stream) {
                Ok(p) => refs.push(json!(p.to_string_lossy())),
                Err(err) => {
                    eprintln!("{name}: no reference ({err})");
                    continue;
                }
            }
        }
        total += 1;
        let out = tmp.join(format!("{name}.yuv"));
        let run = Command::new(&bin)
            .arg("decode")
            .arg(&stream)
            .arg(&out)
            .output()
            .map_err(|e| e.to_string())?;
        let log = String::from_utf8_lossy(&run.stderr).trim().to_string();
        // "N frames WxH in ..."
        let dims = log
            .split_whitespace()
            .nth(2)
            .and_then(|d| d.split_once('x'))
            .and_then(|(w, h)| Some((w.parse::<usize>().ok()?, h.parse::<usize>().ok()?)));
        let got = fs::read(&out).unwrap_or_default();
        let _ = fs::remove_file(&out);
        // The frame size follows from the output, so any chroma format works.
        let frames = log
            .split_whitespace()
            .next()
            .and_then(|n| n.parse::<usize>().ok());
        let frame_size = match (dims, frames) {
            (Some((w, h)), Some(f)) if f > 0 && got.len() % f == 0 && got.len() / f >= w * h => {
                got.len() / f
            }
            (Some((w, h)), _) => w * h * 3 / 2,
            _ => 0,
        };
        // Some suites ship several YUV files (e.g. per layer); match any.
        let mut best: Option<(usize, usize, String)> = None;
        if dims.is_some_and(|(w, h)| w > 0 && h > 0) {
            let fs_ = frame_size;
            for r in &refs {
                let want = fs::read(base.join(r.as_str().unwrap_or_default())).unwrap_or_default();
                let n = want.len() / fs_;
                let ok = (0..n)
                    .filter(|&i| {
                        got.get(i * fs_..(i + 1) * fs_) == want.get(i * fs_..(i + 1) * fs_)
                    })
                    .count();
                if best.as_ref().is_none_or(|b| ok > b.0) {
                    best = Some((ok, n, r.as_str().unwrap_or_default().to_string()));
                }
            }
        }
        let (ok, n, _) = best.clone().unwrap_or((0, 0, String::new()));
        let status = if n > 0 && ok == n && got.len() == n * frame_size {
            exact += 1;
            if from_jm { "exact (JM)" } else { "exact" }.to_string()
        } else if log.contains("unsupported") {
            format!(
                "unsupported ({})",
                log.rsplit("last: ").next().unwrap_or("")
            )
        } else {
            format!("{ok}/{n} frames")
        };
        println!(
            "{:<22} {:<28} {status}",
            e["set"].as_str().unwrap_or_default(),
            name
        );
        results.push(json!({ "set": e["set"], "name": name, "status": status, "frames_ok": ok, "frames": n, "log": log }));
    }
    println!("{exact}/{total} bit-exact");
    fs::write(
        base.join("results.json"),
        serde_json::to_string_pretty(&results).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

fn remove_yuv(dir: &Path) {
    let mut files = Vec::new();
    collect_files(dir, &mut files);
    for f in files {
        if f.extension()
            .and_then(|x| x.to_str())
            .is_some_and(|x| x.eq_ignore_ascii_case("yuv"))
        {
            let _ = fs::remove_file(f);
        }
    }
}

fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect_files(&p, out);
        } else {
            out.push(p);
        }
    }
}

// ---------------------------------------------------------------- check-headers

/// Key fields of one slice header: first_mb, slice_type, pps, frame_num,
/// poc lsb, qp delta and the bit where the header ends (from the NAL start).
type SliceKey = [i64; 7];

/// Base-view slice headers as seen by ffmpeg's CBS parser. Access units in
/// which CBS reported a failure are dropped: it cannot parse PPSs that refer
/// to a subset SPS and then prints partial or misparsed slice headers.
fn ffmpeg_slices(file: &Path) -> Result<Vec<SliceKey>> {
    let out = Command::new("ffmpeg")
        .args(["-v", "repeat+trace", "-f", "h264", "-i"])
        .arg(file)
        .args(["-c", "copy", "-bsf:v", "trace_headers", "-f", "null", "-"])
        .output()
        .map_err(|e| e.to_string())?;
    let text = String::from_utf8_lossy(&out.stderr);
    let mut slices = Vec::new();
    let mut au: Vec<SliceKey> = Vec::new();
    let mut au_failed = false;
    let mut cur: Option<(SliceKey, bool)> = None;
    let end_slice = |cur: &mut Option<(SliceKey, bool)>, au: &mut Vec<SliceKey>| {
        if let Some((k, true)) = cur.take() {
            au.push(k);
        }
    };
    for line in text.lines() {
        // Other components (e.g. the h264 parser) log in between; skip them.
        if !line.starts_with("[trace_headers") {
            continue;
        }
        let Some(rest) = line.split_once("] ").map(|(_, r)| r) else {
            continue;
        };
        if rest.starts_with("Access Unit Delimiter") {
            end_slice(&mut cur, &mut au);
            if !au_failed {
                slices.append(&mut au);
            }
            au.clear();
            au_failed = false;
            continue;
        }
        if rest.starts_with("Failed")
            || rest.contains("out of range")
            || rest.contains("not available")
        {
            au_failed = true;
        }
        if rest.starts_with("Slice Header") {
            end_slice(&mut cur, &mut au);
            cur = Some(([0; 7], false));
            continue;
        }
        if !rest.starts_with(|c: char| c.is_ascii_digit()) {
            if !rest.starts_with(' ') {
                end_slice(&mut cur, &mut au);
            }
            continue;
        }
        let Some((key, is_base)) = cur.as_mut() else {
            continue;
        };
        // "<pos> <name> <bits> = <value>"
        let mut it = rest.split_whitespace();
        let (Some(pos), Some(name), Some(bits)) = (it.next(), it.next(), it.next()) else {
            continue;
        };
        let value = rest
            .rsplit("= ")
            .next()
            .and_then(|v| v.trim().parse::<i64>().ok())
            .unwrap_or(0);
        let (Ok(pos), name) = (pos.parse::<i64>(), name.split('[').next().unwrap_or(name)) else {
            continue;
        };
        match name {
            "nal_unit_type" => *is_base = value == 1 || value == 5,
            "first_mb_in_slice" => key[0] = value,
            "slice_type" => key[1] = value,
            "pic_parameter_set_id" => key[2] = value,
            "frame_num" => key[3] = value,
            "pic_order_cnt_lsb" => key[4] = value,
            "slice_qp_delta" => key[5] = value,
            _ => {}
        }
        if name != "cabac_alignment_one_bit" {
            key[6] = key[6].max(pos + bits.len() as i64);
        }
    }
    end_slice(&mut cur, &mut au);
    if !au_failed {
        slices.append(&mut au);
    }
    Ok(slices)
}

fn our_slices(file: &Path) -> Result<Vec<SliceKey>> {
    let bin = cli_bin();
    build_cli()?;
    let out = Command::new(bin)
        .arg("slices")
        .arg(file)
        .output()
        .map_err(|e| e.to_string())?;
    let mut slices = Vec::new();
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 10 || !(f[1] == "1" || f[1] == "5") {
            continue;
        }
        let ty = f[4].trim_end_matches('*');
        let fixed = f[4].ends_with('*') as i64;
        let ty = ["P", "B", "I", "Sp", "Si"]
            .iter()
            .position(|t| *t == ty)
            .unwrap_or(9) as i64
            + 5 * fixed;
        let n = |s: &str| s.parse::<i64>().unwrap_or(-1);
        slices.push([n(f[3]), ty, n(f[5]), n(f[6]), n(f[7]), n(f[8]), n(f[9])]);
    }
    Ok(slices)
}

fn check_headers(file: &Path) -> Result {
    let theirs = ffmpeg_slices(file)?;
    let ours = our_slices(file)?;
    let names = [
        "first_mb",
        "slice_type",
        "pps",
        "frame_num",
        "poc_lsb",
        "qp_delta",
        "header_end",
    ];
    // ffmpeg's raw H.264 path drops some pictures (e.g. leading B pictures
    // before the first keyframe), so match its slices as an ordered subsequence.
    const WINDOW: usize = 256;
    let (mut j, mut bad, mut skipped) = (0, 0, 0);
    for (i, b) in theirs.iter().enumerate() {
        match ours[j.min(ours.len())..]
            .iter()
            .take(WINDOW)
            .position(|a| a == b)
        {
            Some(k) => {
                skipped += k;
                j += k + 1;
            }
            None => {
                bad += 1;
                if bad <= 5 {
                    let a = ours.get(j).copied().unwrap_or([-1; 7]);
                    let diff: Vec<String> = (0..7)
                        .filter(|&k| a[k] != b[k])
                        .map(|k| format!("{}: ours {} ffmpeg {}", names[k], a[k], b[k]))
                        .collect();
                    println!(
                        "ffmpeg slice {i} has no match; next of ours differs in {}",
                        diff.join(", ")
                    );
                }
                j += 1;
            }
        }
    }
    skipped += ours.len().saturating_sub(j);
    println!(
        "{} base slices, ffmpeg {}: {} matched, {bad} unmatched, {skipped} not seen by ffmpeg",
        ours.len(),
        theirs.len(),
        theirs.len() - bad
    );
    if bad == 0 {
        Ok(())
    } else {
        Err("slice headers differ".into())
    }
}

// ---------------------------------------------------------------- check-poc

/// POC per decoded view component, in decoding order, from JM's log lines
/// such as "00003( P )        6     1    14  ...".
fn jm_pocs(log: &str) -> Vec<i64> {
    log.lines()
        .filter(|l| {
            l.len() > 6
                && l.as_bytes()[..5].iter().all(u8::is_ascii_digit)
                && l.as_bytes()[5] == b'('
        })
        .filter_map(|l| l.split_once(')')?.1.split_whitespace().next()?.parse().ok())
        .collect()
}

fn check_poc(clip: &Path) -> Result {
    let stem = clip.file_stem().unwrap().to_string_lossy().to_string();
    let log_path = root().join("work/ref").join(&stem).join("jm.log");
    let log = fs::read_to_string(&log_path).map_err(|_| {
        format!(
            "{} missing; run `cargo xtask verify` first",
            log_path.display()
        )
    })?;
    let want = jm_pocs(&log);
    build_cli()?;
    let out = Command::new(cli_bin())
        .arg("pocs")
        .arg(clip)
        .output()
        .map_err(|e| e.to_string())?;
    let got: Vec<i64> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.split('\t').nth(3)?.parse().ok())
        .collect();
    let bad: Vec<usize> = (0..got.len().max(want.len()))
        .filter(|&i| got.get(i) != want.get(i))
        .collect();
    println!(
        "{} view components (JM {}), {} POC mismatches",
        got.len(),
        want.len(),
        bad.len()
    );
    for &i in bad.iter().take(5) {
        println!("  #{i}: ours {:?} JM {:?}", got.get(i), want.get(i));
    }
    if bad.is_empty() {
        Ok(())
    } else {
        Err("POCs differ".into())
    }
}

/// Path of the altc264 CLI: $ALTC_BIN (a frozen copy, so long runs are not
/// affected by edits) or the workspace's release build, built on demand.
fn cli_bin() -> PathBuf {
    std::env::var_os("ALTC_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| root().join("target/release/altc264"))
}

fn build_cli() -> Result {
    if std::env::var_os("ALTC_BIN").is_some() {
        return Ok(());
    }
    let st = Command::new("cargo")
        .args(["build", "-q", "--release", "-p", "altc264-cli"])
        .current_dir(root())
        .status()
        .map_err(|e| e.to_string())?;
    if st.success() {
        Ok(())
    } else {
        Err("building altc264-cli failed".into())
    }
}

// ---------------------------------------------------------------- verify

fn dims(clip: &Path) -> Result<(usize, usize)> {
    let v = run_json(
        Command::new("ffprobe")
            .args([
                "-v",
                "quiet",
                "-f",
                "h264",
                "-of",
                "json",
                "-show_entries",
                "stream=width,height",
            ])
            .arg(clip),
    )?;
    let s = &v["streams"][0];
    match (s["width"].as_u64(), s["height"].as_u64()) {
        (Some(w), Some(h)) => Ok((w as usize, h as usize)),
        _ => Err("could not read dimensions".into()),
    }
}

/// MVC view_ids in view order, from the first subset SPS; `[0]` for 2D streams.
fn stream_views(clip: &Path) -> Result<Vec<u16>> {
    let data = fs::read(clip).map_err(|e| e.to_string())?;
    let mut rbsp = Vec::new();
    for nal in altc264::nal::AnnexBIter::new(&data) {
        if nal.first().is_some_and(|b| b & 31 == 15) {
            rbsp.clear();
            altc264::nal::unescape_into(&nal[1..], &mut rbsp);
            if let Ok(s) = altc264::ps::SubsetSps::parse(&rbsp) {
                return Ok(s.mvc.view_ids.clone());
            }
        }
    }
    Ok(vec![0])
}

/// Decodes a conformance stream with JM into a cached YUV file (base view
/// only for MVC streams).
fn jm_conformance_yuv(name: &str, stream: &Path) -> Result<PathBuf> {
    let dir = work(&format!("conformance-jm/{name}"))?;
    let out = dir.join("jm.yuv");
    let fresh = match (fs::metadata(&out), fs::metadata(stream)) {
        (Ok(a), Ok(b)) => a.len() > 0 && a.modified().ok() >= b.modified().ok(),
        _ => false,
    };
    if fresh {
        return Ok(out);
    }
    // JM's option parser splits on spaces, so hand it relative names via a symlink.
    let link = dir.join("input.264");
    let _ = fs::remove_file(&link);
    let abs = fs::canonicalize(stream).map_err(|e| e.to_string())?;
    std::os::unix::fs::symlink(&abs, &link).map_err(|e| e.to_string())?;
    let log = File::create(dir.join("jm.log")).map_err(|e| e.to_string())?;
    let st = Command::new(tool("ldecod")?)
        .current_dir(&dir)
        .arg("-d")
        .arg(tool("decoder.cfg")?)
        .args([
            "-p",
            "InputFile=input.264",
            "-p",
            "OutputFile=jm.yuv",
            "-p",
            "RefFile=none",
        ])
        .stdout(log)
        .status()
        .map_err(|e| e.to_string())?;
    let _ = fs::remove_file(&link);
    if !st.success() || fs::metadata(&out).map_or(true, |m| m.len() == 0) {
        let _ = fs::remove_file(&out);
        return Err("JM failed".into());
    }
    Ok(out)
}

/// Decodes `clip` with JM once and caches per-view frame hashes in
/// `work/ref/<stem>/hashes.json`. The YUV is deleted afterwards (about 1.5 GB per
/// 10 s of 1080p stereo) unless ALTC_KEEP_REF is set, for debugging diffs.
fn jm_reference(clip: &Path, frame_size: usize, num_views: usize) -> Result<Vec<Vec<u64>>> {
    let stem = clip.file_stem().unwrap().to_string_lossy().to_string();
    let dir = work(&format!("ref/{stem}"))?;
    let cache = dir.join("hashes.json");
    let fresh = match (fs::metadata(&cache), fs::metadata(clip)) {
        (Ok(a), Ok(b)) => a.modified().ok() >= b.modified().ok(),
        _ => false,
    };
    if fresh {
        let v: Value = serde_json::from_slice(&fs::read(&cache).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        let tool_ok = v["reference"].as_str().unwrap_or("jm") == reference_tool(num_views);
        if v["frame_size"].as_u64() == Some(frame_size as u64) && tool_ok {
            return serde_json::from_value(v["views"].clone()).map_err(|e| e.to_string());
        }
    }
    if reference_tool(num_views) == "jmvc" {
        return jmvc_reference(clip, &dir, frame_size, num_views);
    }
    eprintln!("decoding reference with JM (slow)...");
    // JM's option parser splits on spaces, so hand it relative names via a symlink.
    let link = dir.join("input.264");
    let _ = fs::remove_file(&link);
    let abs = fs::canonicalize(clip).map_err(|e| e.to_string())?;
    std::os::unix::fs::symlink(&abs, &link).map_err(|e| e.to_string())?;
    let log = File::create(dir.join("jm.log")).map_err(|e| e.to_string())?;
    let st = Command::new(tool("ldecod")?)
        .current_dir(&dir)
        .arg("-d")
        .arg(tool("decoder.cfg")?)
        .args([
            "-p",
            "InputFile=input.264",
            "-p",
            "OutputFile=jm.yuv",
            "-p",
            "RefFile=none",
            "-p",
            "DecodeAllLayers=1",
        ])
        .stdout(log)
        .status()
        .map_err(|e| e.to_string())?;
    if !st.success() {
        return Err(format!("JM failed; see {}", dir.join("jm.log").display()));
    }
    // JM names outputs jm_ViewId0000.yuv / jm_ViewId0001.yuv for MVC, jm.yuv otherwise.
    // JM names view files by view_id (jm_ViewId0000.yuv, jm_ViewId0002.yuv, ...).
    let mut outputs: Vec<PathBuf> = fs::read_dir(&dir)
        .map_err(|e| e.to_string())?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("jm_ViewId") && n.ends_with(".yuv"))
        })
        // JM also leaves empty files for view ids it does not output.
        .filter(|p| fs::metadata(p).is_ok_and(|m| m.len() > 0))
        .collect();
    outputs.sort();
    let outputs = if outputs.is_empty() {
        vec![dir.join("jm.yuv")]
    } else {
        outputs
    };
    let views = outputs
        .iter()
        .map(|p| frame_hashes(p, frame_size))
        .collect::<Result<Vec<_>>>()?;
    if std::env::var_os("ALTC_KEEP_REF").is_none() {
        for p in &outputs {
            let _ = fs::remove_file(p);
        }
    }
    let _ = fs::remove_file(&link);
    let v = json!({ "frame_size": frame_size, "views": views });
    fs::write(&cache, v.to_string()).map_err(|e| e.to_string())?;
    Ok(views)
}

/// JM outputs at most two views; JMVC decodes any number.
fn reference_tool(num_views: usize) -> &'static str {
    if num_views > 2 { "jmvc" } else { "jm" }
}

/// Like jm_reference, with the JMVC decoder. Its outputs are named
/// jmvc_<view_id>.yuv.
fn jmvc_reference(
    clip: &Path,
    dir: &Path,
    frame_size: usize,
    num_views: usize,
) -> Result<Vec<Vec<u64>>> {
    eprintln!("decoding reference with JMVC (slow)...");
    let abs = fs::canonicalize(clip).map_err(|e| e.to_string())?;
    let log = File::create(dir.join("jmvc.log")).map_err(|e| e.to_string())?;
    let st = Command::new(tool("jmvc-decoder")?)
        .current_dir(dir)
        .arg(&abs)
        .arg("jmvc.yuv")
        .arg(num_views.to_string())
        .stdout(log)
        .status()
        .map_err(|e| e.to_string())?;
    if !st.success() {
        return Err(format!(
            "JMVC failed; see {}",
            dir.join("jmvc.log").display()
        ));
    }
    let mut outputs: Vec<(u32, PathBuf)> = fs::read_dir(dir)
        .map_err(|e| e.to_string())?
        .flatten()
        .filter_map(|e| {
            let n = e.file_name().to_str()?.to_string();
            let id = n
                .strip_prefix("jmvc_")?
                .strip_suffix(".yuv")?
                .parse()
                .ok()?;
            Some((id, e.path()))
        })
        .filter(|(_, p)| fs::metadata(p).is_ok_and(|m| m.len() > 0))
        .collect();
    outputs.sort();
    let views = outputs
        .iter()
        .map(|(_, p)| frame_hashes(p, frame_size))
        .collect::<Result<Vec<_>>>()?;
    if std::env::var_os("ALTC_KEEP_REF").is_none() {
        for (_, p) in &outputs {
            let _ = fs::remove_file(p);
        }
    }
    let v = json!({ "frame_size": frame_size, "views": views, "reference": "jmvc" });
    fs::write(dir.join("hashes.json"), v.to_string()).map_err(|e| e.to_string())?;
    Ok(views)
}

/// FNV-1a over 64-bit words: stable across Rust versions, so hashes can be cached.
fn hash_frame(buf: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    let (words, rest) = buf.as_chunks::<8>();
    for w in words {
        h = (h ^ u64::from_le_bytes(*w)).wrapping_mul(0x100000001b3);
    }
    for &b in rest {
        h = (h ^ b as u64).wrapping_mul(0x100000001b3);
    }
    h
}

fn frame_hashes(path: &Path, frame_size: usize) -> Result<Vec<u64>> {
    let mut r = BufReader::with_capacity(
        1 << 22,
        File::open(path).map_err(|e| format!("{}: {e}", path.display()))?,
    );
    let mut buf = vec![0u8; frame_size];
    let mut out = Vec::new();
    while r.read_exact(&mut buf).is_ok() {
        out.push(hash_frame(&buf));
    }
    Ok(out)
}

struct Comparison {
    frames: usize,
    /// Decoder frame i is compared with reference frame i - offset. Nonzero
    /// when a decoder emits or drops undecodable leading pictures (clips cut
    /// at an open-GOP I picture), which the spec leaves unconstrained.
    offset: isize,
    exact: usize,
    pixel_ok: usize,
    first_bad: Vec<usize>,
}

const MAX_OFFSET: isize = 8;

fn compare(got: &[u64], want: &[u64]) -> Comparison {
    let matches = |k: isize| {
        got.iter()
            .enumerate()
            .filter(|&(i, h)| {
                usize::try_from(i as isize - k)
                    .ok()
                    .and_then(|j| want.get(j))
                    == Some(h)
            })
            .count()
    };
    // Prefer zero, then the smallest shift, among equally good alignments.
    let offset = (-MAX_OFFSET..=MAX_OFFSET)
        .max_by_key(|&k| (matches(k), std::cmp::Reverse(k.abs())))
        .unwrap();
    let set: HashSet<u64> = want.iter().copied().collect();
    Comparison {
        frames: got.len(),
        offset,
        exact: matches(offset),
        pixel_ok: got.iter().filter(|h| set.contains(h)).count(),
        first_bad: (0..got.len())
            .filter(|&i| {
                usize::try_from(i as isize - offset)
                    .ok()
                    .and_then(|j| want.get(j))
                    != Some(&got[i])
            })
            .take(5)
            .collect(),
    }
}

fn verify(clip: &Path) -> Result {
    let (w, h) = dims(clip)?;
    let frame_size = w * h * 3 / 2;
    let view_ids = stream_views(clip)?;
    let want = jm_reference(clip, frame_size, view_ids.len())?;
    println!(
        "{} reference: {} views, {} frames, {w}x{h}",
        reference_tool(view_ids.len()).to_uppercase(),
        want.len(),
        want[0].len()
    );

    let tmp = work("tmp/verify")?;
    build_cli()?;
    println!(
        "{:<5} {:>7} {:>16} {:>16}  first mismatches",
        "view", "frames", "right position", "right pixels"
    );
    // Per-process names: several verify runs may share work/tmp.
    let pid = std::process::id();
    let outs: Vec<PathBuf> = (0..view_ids.len().max(2))
        .map(|v| tmp.join(format!("altc264_{pid}_v{v}.yuv")))
        .collect();
    let o = Command::new(cli_bin())
        .arg("decode")
        .arg(clip)
        .args(&outs)
        .output()
        .map_err(|e| e.to_string())?;
    let mut summary = Vec::new();
    if !o.status.success() {
        let log = String::from_utf8_lossy(&o.stderr).trim().to_string();
        println!("{:<5} crashed: {log}", "-");
        summary.push(json!({ "error": log }));
    } else {
        // References list the views that have pictures in view_id order;
        // outputs are in view order.
        let mut got = Vec::new();
        for (v, o) in outs.iter().enumerate() {
            let h = frame_hashes(o, frame_size)?;
            if v == 0 || !h.is_empty() {
                got.push((view_ids.get(v).copied().unwrap_or(v as u16), h));
            }
        }
        got.sort_by_key(|(id, _)| *id);
        for (v, want) in want.iter().enumerate() {
            let empty = Vec::new();
            let got = got.get(v).map_or(&empty, |(_, h)| h);
            let c = compare(got, want);
            let n = want.len();
            println!(
                "{v:<5} {:>7} {:>16} {:>16}  {:?}",
                c.frames,
                if c.offset == 0 {
                    format!("{}/{n}", c.exact)
                } else {
                    format!("{}/{n} @{:+}", c.exact, c.offset)
                },
                format!("{}/{n}", c.pixel_ok),
                c.first_bad
            );
            summary.push(json!({ "view": v, "frames": c.frames, "reference_frames": n,
                "offset": c.offset, "exact": c.exact, "pixel_ok": c.pixel_ok, "first_bad": c.first_bad }));
        }
    }
    for o in &outs {
        let _ = fs::remove_file(o);
    }
    let stem = clip.file_stem().unwrap().to_string_lossy();
    let out = work("verify")?.join(format!("{stem}.json"));
    fs::write(&out, serde_json::to_string_pretty(&summary).unwrap()).map_err(|e| e.to_string())?;
    Ok(())
}

// ---------------------------------------------------------------- conformance manifest

/// Expected output of every conformance stream, committed so a check needs
/// only the bitstreams: no reference YUVs (35 GB) and no reference decoder.
const MANIFEST: &str = "conformance/manifest.json";

/// Streams per suite in the `--smoke` subset: the smallest bit-exact ones.
const SMOKE: &[(&str, usize)] = &[
    ("AVCv1", 14),
    ("FRExt", 10),
    ("MVC", 8),
    ("Professional_profiles", 6),
];

/// Where the downloaded suites are: `ALTC_CONFORMANCE_DIR`, else work/conformance.
fn conformance_dir() -> Result<PathBuf> {
    match std::env::var_os("ALTC_CONFORMANCE_DIR") {
        Some(d) => Ok(PathBuf::from(d)),
        None => work("conformance"),
    }
}

/// FNV-1a over a file, a 64-bit word at a time (the tail byte by byte):
/// stable and dependency-free, and enough to see any change in decoded
/// output. Not a cryptographic hash. Returns (bytes, hex digest).
fn hash_file(path: &Path) -> Result<(u64, String)> {
    hash_file_prefix(path, u64::MAX)
}

/// Streams whose reference YUV runs past the bitstream: (name, bytes of the
/// reference that it covers). alphaconformanceG combines 40 frames of
/// foreground and alpha, while the supplied foreground YUV has all 43 of the
/// separate foreground stream (its readme); the 40 match exactly.
const REFERENCE_PREFIX: &[(&str, u64)] = &[("alphaconformanceG", 40 * 960 * 544 * 3 / 2)];

/// `hash_file` over at most the first `limit` bytes.
fn hash_file_prefix(path: &Path, limit: u64) -> Result<(u64, String)> {
    let mut f = BufReader::with_capacity(
        1 << 20,
        File::open(path).map_err(|e| format!("{}: {e}", path.display()))?,
    );
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut n = 0u64;
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let want = (limit - n).min(buf.len() as u64) as usize;
        let k = f.read(&mut buf[..want]).map_err(|e| e.to_string())?;
        if k == 0 {
            break;
        }
        let (words, rest) = buf[..k].as_chunks::<8>();
        for w in words {
            h ^= u64::from_le_bytes(*w);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        for &b in rest {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        n += k as u64;
    }
    Ok((n, format!("{h:016x}")))
}

/// Decodes `stream` into one file per view under `dir` and returns their
/// (bytes, hash), view 0 first, with the decoder's stderr.
fn decode_hashes(
    stream: &Path,
    dir: &Path,
    views: usize,
    threads: Option<&str>,
) -> Result<(Vec<(u64, String)>, String)> {
    let outs: Vec<PathBuf> = (0..views.max(1))
        .map(|v| dir.join(format!("out_v{v}.yuv")))
        .collect();
    let mut cmd = match std::env::var("ALTC_RUNNER") {
        // e.g. "wasmtime run --dir=/" for a wasm build.
        Ok(r) if !r.trim().is_empty() => {
            let mut p = r.split_whitespace();
            let mut c = Command::new(p.next().unwrap());
            c.args(p).arg(cli_bin());
            c
        }
        _ => Command::new(cli_bin()),
    };
    cmd.arg("decode").arg(stream).args(&outs);
    if let Some(t) = threads {
        cmd.env("ALTC_THREADS", t);
    }
    let o = cmd.output().map_err(|e| e.to_string())?;
    let log = String::from_utf8_lossy(&o.stderr).trim().to_string();
    let mut hashes = Vec::new();
    for p in &outs {
        hashes.push(if p.exists() {
            hash_file(p)?
        } else {
            (0, String::new())
        });
        let _ = fs::remove_file(p);
    }
    if !o.status.success() {
        return Err(if log.is_empty() {
            "decoder failed".into()
        } else {
            log
        });
    }
    Ok((hashes, log))
}

/// Runs JM (two views at most) or JMVC on an MVC stream and returns the
/// per-view output files in view_id order.
fn jm_all_views(stream: &Path, dir: &Path, num_views: usize) -> Result<Vec<PathBuf>> {
    let abs = fs::canonicalize(stream).map_err(|e| e.to_string())?;
    let mut outputs: Vec<PathBuf> = if num_views > 2 {
        let st = Command::new(tool("jmvc-decoder")?)
            .current_dir(dir)
            .arg(&abs)
            .arg("jmvc.yuv")
            .arg(num_views.to_string())
            .stdout(Stdio::null())
            .status()
            .map_err(|e| e.to_string())?;
        if !st.success() {
            return Err("JMVC failed".into());
        }
        let mut v: Vec<(u32, PathBuf)> = fs::read_dir(dir)
            .map_err(|e| e.to_string())?
            .flatten()
            .filter_map(|e| {
                let n = e.file_name().to_str()?.to_string();
                Some((
                    n.strip_prefix("jmvc_")?
                        .strip_suffix(".yuv")?
                        .parse()
                        .ok()?,
                    e.path(),
                ))
            })
            .collect();
        v.sort();
        v.into_iter().map(|(_, p)| p).collect()
    } else {
        let link = dir.join("input.264");
        let _ = fs::remove_file(&link);
        std::os::unix::fs::symlink(&abs, &link).map_err(|e| e.to_string())?;
        let st = Command::new(tool("ldecod")?)
            .current_dir(dir)
            .arg("-d")
            .arg(tool("decoder.cfg")?)
            .args([
                "-p",
                "InputFile=input.264",
                "-p",
                "OutputFile=jm.yuv",
                "-p",
                "RefFile=none",
                "-p",
                "DecodeAllLayers=1",
            ])
            .stdout(Stdio::null())
            .status()
            .map_err(|e| e.to_string())?;
        let _ = fs::remove_file(&link);
        if !st.success() {
            return Err("JM failed".into());
        }
        let mut v: Vec<PathBuf> = fs::read_dir(dir)
            .map_err(|e| e.to_string())?
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("jm_ViewId") && n.ends_with(".yuv"))
            })
            .collect();
        v.sort();
        v
    };
    // JM also leaves empty files for view ids it does not output.
    outputs.retain(|p| fs::metadata(p).is_ok_and(|m| m.len() > 0));
    if outputs.is_empty() {
        return Err("reference decoder wrote no views".into());
    }
    Ok(outputs)
}

/// Rebuilds conformance/manifest.json: for every indexed stream, the hash
/// of each view's reference output (the ITU-supplied YUV, else JM's or
/// JMVC's), and whether altc264 currently matches it.
fn conformance_manifest() -> Result {
    let base = conformance_dir()?;
    let index: Value = serde_json::from_slice(
        &fs::read(base.join("index.json")).map_err(|_| "run `cargo xtask conformance` first")?,
    )
    .map_err(|e| e.to_string())?;
    build_cli()?;
    let tmp = work("tmp/manifest")?;
    let mut streams = Vec::new();
    for e in index.as_array().into_iter().flatten() {
        let set = e["set"].as_str().unwrap_or_default();
        let name = e["name"].as_str().unwrap_or_default();
        let rel = e["bitstream"].as_str().unwrap_or_default();
        let stream = base.join(rel);
        let (bytes, hash) = hash_file(&stream)?;
        let ids = stream_views(&stream)?;
        let refs: Vec<&str> = e["reference_yuv"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect();
        // Each view lists the outputs it may match (a few suites ship more
        // than one YUV for a stream, e.g. with and without an SEI applied).
        let (reference, views): (&str, Vec<Vec<(u64, String)>>) = if !refs.is_empty() {
            let limit = REFERENCE_PREFIX
                .iter()
                .find(|(n, _)| *n == name)
                .map_or(u64::MAX, |p| p.1);
            let cands = refs
                .iter()
                .map(|r| hash_file_prefix(&base.join(r), limit))
                .collect::<Result<_>>()?;
            ("itu", vec![cands])
        } else if ids.len() > 1 {
            let dir = tmp.join(name);
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
            // JMVC for more than two views; JM, which outputs two, if JMVC
            // cannot read the stream.
            let mut tool = if ids.len() > 2 { "jmvc" } else { "jm" };
            let outs = match jm_all_views(&stream, &dir, ids.len()) {
                Ok(o) => o,
                Err(_) if ids.len() > 2 => match jm_all_views(&stream, &dir, 2) {
                    Ok(o) => {
                        tool = "jm";
                        o
                    }
                    Err(err) => {
                        eprintln!("{name}: no reference ({err}); skipped");
                        continue;
                    }
                },
                Err(err) => {
                    eprintln!("{name}: no reference ({err}); skipped");
                    continue;
                }
            };
            // The references are in view_id order, the decoder's outputs in
            // view order: list the expected outputs in view order.
            let mut by_id = ids.clone();
            by_id.sort();
            let mut ordered: Vec<&PathBuf> = Vec::new();
            for id in &ids {
                if let Some(p) = by_id.iter().position(|x| x == id).and_then(|i| outs.get(i)) {
                    ordered.push(p);
                }
            }
            if ordered.len() != outs.len() {
                ordered = outs.iter().collect();
            }
            let v = ordered
                .iter()
                .map(|p| hash_file(p).map(|h| vec![h]))
                .collect::<Result<_>>()?;
            let _ = fs::remove_dir_all(&dir);
            (tool, v)
        } else {
            match jm_conformance_yuv(name, &stream) {
                Ok(p) => ("jm", vec![vec![hash_file(&p)?]]),
                Err(err) => {
                    eprintln!("{name}: no reference ({err}); skipped");
                    continue;
                }
            }
        };
        let expect = match decode_hashes(&stream, &tmp, views.len(), None) {
            Ok((got, _))
                if got.iter().zip(&views).all(|(g, c)| c.contains(g))
                    && got.len() == views.len() =>
            {
                "exact"
            }
            Ok(_) => "differs",
            Err(log) if log.contains("unsupported") => "unsupported",
            Err(_) => "fails",
        };
        println!(
            "{set:<22} {name:<28} {reference:<5} {} view(s)  {expect}",
            views.len()
        );
        let mut entry = json!({
            "set": set,
            "name": name,
            "bitstream": rel,
            "bytes": bytes,
            "hash": hash,
            "reference": reference,
            "views": views.iter().map(|c| c.iter().map(|(b, h)| json!({ "bytes": b, "hash": h })).collect::<Vec<_>>()).collect::<Vec<_>>(),
            "expect": expect,
        });
        // Streams whose subset SPS lists more views than have pictures are
        // fully checked; note only those where altc264 outputs views that
        // no reference decoder could.
        if views.len() < ids.len() {
            let ours = decode_hashes(&stream, &tmp, ids.len(), None)
                .map_or(0, |(h, _)| h.iter().filter(|(b, _)| *b > 0).count());
            if ours > views.len() {
                entry["note"] = json!(format!(
                    "altc264 outputs {ours} views; the reference decoders only {}, so only those are checked",
                    views.len()
                ));
            }
        }
        streams.push(entry);
    }
    // The smoke subset: the smallest bit-exact streams of each suite.
    for (set, n) in SMOKE {
        let mut exact: Vec<usize> = (0..streams.len())
            .filter(|&i| streams[i]["set"] == *set && streams[i]["expect"] == "exact")
            .collect();
        exact.sort_by_key(|&i| streams[i]["bytes"].as_u64());
        for &i in exact.iter().take(*n) {
            streams[i]["smoke"] = json!(true);
        }
    }
    let exact = streams.iter().filter(|s| s["expect"] == "exact").count();
    let out = root().join(MANIFEST);
    fs::create_dir_all(out.parent().unwrap()).map_err(|e| e.to_string())?;
    let doc = json!({
        "about": "Expected decoder output for the ITU-T H.264.1 conformance streams. Regenerate with `cargo xtask conformance-manifest`; check with `cargo xtask conformance-check`. Hashes are FNV-1a (see xtask).",
        "source": CONFORMANCE_URL,
        "streams": streams,
    });
    fs::write(&out, serde_json::to_string_pretty(&doc).unwrap() + "\n")
        .map_err(|e| e.to_string())?;
    println!("{exact}/{} bit-exact -> {}", streams.len(), out.display());
    Ok(())
}

/// Decodes every stream in the manifest (or the `--smoke` subset) and
/// compares each view's output hash. Fails if a stream expected to be
/// bit-exact is not, or is missing; reports streams that newly match.
fn conformance_check(args: &[&str]) -> Result {
    let mut smoke = false;
    let mut threads: Option<String> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match *a {
            "--smoke" => smoke = true,
            "--threads" => threads = it.next().map(|s| s.to_string()),
            _ => return Err(USAGE.into()),
        }
    }
    let manifest: Value = serde_json::from_slice(
        &fs::read(root().join(MANIFEST)).map_err(|e| format!("{MANIFEST}: {e}"))?,
    )
    .map_err(|e| e.to_string())?;
    let base = conformance_dir()?;
    build_cli()?;
    let streams: Vec<&Value> = manifest["streams"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|s| !smoke || s["smoke"] == true)
        .collect();
    // Several decodes at once: one thread each unless --threads is given.
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    let per = threads
        .as_deref()
        .and_then(|t| t.parse::<usize>().ok())
        .unwrap_or(1)
        .max(1);
    let jobs = (cores / per).max(1);
    let threads = threads.or(Some("1".into()));
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results = std::sync::Mutex::new(Vec::new());
    let tmp = work("tmp/check")?;
    std::thread::scope(|scope| {
        for j in 0..jobs {
            let (next, results, streams, base, tmp, threads) =
                (&next, &results, &streams, &base, &tmp, &threads);
            scope.spawn(move || {
                let dir = tmp.join(format!("job{j}_{}", std::process::id()));
                let _ = fs::create_dir_all(&dir);
                while let Some(s) =
                    streams.get(next.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
                {
                    let stream = base.join(s["bitstream"].as_str().unwrap_or_default());
                    let views = s["views"].as_array().cloned().unwrap_or_default();
                    let status = if !stream.exists() {
                        "missing".to_string()
                    } else if hash_file(&stream).ok().map(|h| h.1)
                        != s["hash"].as_str().map(String::from)
                    {
                        "bitstream differs from the manifest's".to_string()
                    } else {
                        match decode_hashes(&stream, &dir, views.len(), threads.as_deref()) {
                            Ok((got, _)) => {
                                let ok = got.len() == views.len()
                                    && got.iter().zip(&views).all(|((b, h), c)| {
                                        c.as_array().into_iter().flatten().any(|x| {
                                            x["bytes"].as_u64() == Some(*b)
                                                && x["hash"].as_str() == Some(h)
                                        })
                                    });
                                if ok { "exact".into() } else { "differs".into() }
                            }
                            Err(log) if log.contains("unsupported") => "unsupported".into(),
                            Err(log) => format!("fails: {}", log.lines().last().unwrap_or("")),
                        }
                    };
                    results.lock().unwrap().push((*s, status));
                }
                let _ = fs::remove_dir_all(&dir);
            });
        }
    });
    let mut results = results.into_inner().unwrap();
    results.sort_by_key(|(s, _)| {
        (
            s["set"].as_str().unwrap_or_default(),
            s["name"].as_str().unwrap_or_default(),
        )
    });
    let (mut exact, mut regressions, mut improved) = (0, 0, 0);
    for (s, status) in &results {
        let expect = s["expect"].as_str().unwrap_or_default();
        let (set, name) = (
            s["set"].as_str().unwrap_or_default(),
            s["name"].as_str().unwrap_or_default(),
        );
        if status == "exact" {
            exact += 1;
            if expect != "exact" {
                improved += 1;
                println!("{set:<22} {name:<28} now bit-exact (manifest: {expect})");
            }
        } else if expect == "exact" {
            regressions += 1;
            println!("{set:<22} {name:<28} REGRESSION: {status}");
        } else {
            println!("{set:<22} {name:<28} {status} (expected: {expect})");
        }
    }
    println!(
        "{exact}/{} bit-exact{}",
        results.len(),
        if smoke { " (smoke subset)" } else { "" }
    );
    if improved > 0 {
        println!("{improved} stream(s) now match: regenerate the manifest to lock them in");
    }
    if regressions > 0 {
        return Err(format!("{regressions} regression(s)"));
    }
    Ok(())
}

// ---------------------------------------------------------------- coverage

/// Coverage of what actually verifies the decoder: the unit and
/// integration tests, and the conformance streams decoded by an
/// instrumented CLI (`--streams`: the smoke subset by default, every
/// stream, or none). Measures the code built for this machine only, so
/// each CPU family reports its own SIMD paths; CI uploads one report per
/// family and Codecov merges them.
fn coverage(args: &[&str]) -> Result {
    let mut streams = "smoke";
    let mut format: Vec<String> = vec!["--summary-only".into()];
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match *a {
            "--streams" => streams = it.next().copied().unwrap_or(""),
            "--html" => format = vec!["--html".into()],
            "--lcov" => {
                let out = it.next().ok_or("--lcov needs a file")?;
                format = vec!["--lcov".into(), "--output-path".into(), out.to_string()];
            }
            _ => return Err(USAGE.into()),
        }
    }
    if !matches!(streams, "smoke" | "full" | "none") {
        return Err("--streams is smoke, full or none".into());
    }
    // Instrumented builds in a directory of their own; every step, the
    // report included, runs with the environment cargo-llvm-cov exports.
    let target = root().join("target/coverage");
    let shown = Command::new("cargo")
        .args(["llvm-cov", "show-env", "--export-prefix", "--release"])
        .env("CARGO_TARGET_DIR", &target)
        .current_dir(root())
        .output()
        .map_err(|e| {
            format!("cargo llvm-cov: {e} (install with `cargo install cargo-llvm-cov`)")
        })?;
    if !shown.status.success() {
        return Err(String::from_utf8_lossy(&shown.stderr).trim().to_string());
    }
    let mut env: Vec<(String, String)> = String::from_utf8_lossy(&shown.stdout)
        .lines()
        .filter_map(|l| l.strip_prefix("export ")?.split_once('='))
        .map(|(k, v)| (k.to_string(), v.trim_matches('\'').to_string()))
        .collect();
    env.push((
        "CARGO_TARGET_DIR".into(),
        target.to_string_lossy().into_owned(),
    ));
    let run = |args: &[&str]| -> Result<std::process::Output> {
        Command::new("cargo")
            .args(args)
            .envs(env.iter().map(|(k, v)| (k, v)))
            .current_dir(root())
            .output()
            .map_err(|e| e.to_string())
    };
    let clean = run(&["llvm-cov", "clean", "--workspace"])?;
    if !clean.status.success() {
        return Err(String::from_utf8_lossy(&clean.stderr).trim().to_string());
    }
    let cargo = |args: &[&str]| -> Result {
        let st = Command::new("cargo")
            .args(args)
            .envs(env.iter().map(|(k, v)| (k, v)))
            .current_dir(root())
            .status()
            .map_err(|e| e.to_string())?;
        if st.success() {
            Ok(())
        } else {
            Err(format!("cargo {} failed", args.join(" ")))
        }
    };
    cargo(&[
        "test",
        "-q",
        "--workspace",
        "--exclude",
        "xtask",
        "--release",
    ])?;
    if streams != "none" {
        cargo(&["build", "-q", "--release", "-p", "altc264-cli"])?;
        let mut check = Command::new(std::env::current_exe().map_err(|e| e.to_string())?);
        check.arg("conformance-check");
        if streams == "smoke" {
            check.arg("--smoke");
        }
        let st = check
            .envs(env.iter().map(|(k, v)| (k, v)))
            .env("ALTC_BIN", target.join("release/altc264"))
            .status()
            .map_err(|e| e.to_string())?;
        if !st.success() {
            return Err("conformance check failed under coverage".into());
        }
    }
    let mut report = vec![
        "llvm-cov",
        "report",
        "--release",
        "--ignore-filename-regex",
        "(xtask|tools)/",
    ];
    report.extend(format.iter().map(String::as_str));
    let out = run(&report)?;
    print!("{}", String::from_utf8_lossy(&out.stdout));
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).trim().to_string());
    }
    Ok(())
}
