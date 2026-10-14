//! include/altc264.h is the C API exactly as cbindgen generates it (what
//! cargo-c installs, less its version defines). Users of the header, such
//! as video-rs's check that FFmpeg's wrapper was built against the current
//! API, rely on it being current: this fails when it is not.
//! `ALTC264_BLESS=1 cargo test -p altc264-capi --test header` rewrites it.

#[test]
fn header_is_current() {
    let dir = env!("CARGO_MANIFEST_DIR");
    let config = cbindgen::Config::from_file(format!("{dir}/cbindgen.toml")).unwrap();
    let mut out = Vec::new();
    cbindgen::Builder::new()
        .with_crate(dir)
        .with_config(config)
        .generate()
        .expect("cbindgen")
        .write(&mut out);
    let path = format!("{dir}/include/altc264.h");
    if std::env::var_os("ALTC264_BLESS").is_some() {
        std::fs::write(&path, &out).unwrap();
        return;
    }
    let have = std::fs::read(&path).unwrap_or_default();
    assert!(
        have == out,
        "include/altc264.h is stale: run ALTC264_BLESS=1 cargo test -p altc264-capi --test header"
    );
}
