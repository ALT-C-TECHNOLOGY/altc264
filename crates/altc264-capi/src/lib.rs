//! C API for the altc264 decoder. The generated header is
//! `include/altc264.h` (see cbindgen.toml).
//!
//! Usage: open a decoder, optionally give it the container's extradata
//! (avcC, or Annex B parameter sets), feed packets with `altc264_send_data`
//! and fetch pictures with `altc264_get_picture` until it returns
//! `ALTC264_EAGAIN`. At the end of the stream call `altc264_drain` and fetch
//! the remaining pictures. Every picture must be released with
//! `altc264_picture_unref`.
#![allow(clippy::missing_safety_doc)]

use std::ffi::{CStr, c_char, c_int};

use altc264::decoder::{Decoder, Frame};
use altc264::nal::AvcConfig;
use altc264::stereo::StereoLayout;

/// No picture is available yet; send more data.
pub const ALTC264_EAGAIN: c_int = 1;
/// Success.
pub const ALTC264_OK: c_int = 0;
/// Invalid argument.
pub const ALTC264_EINVAL: c_int = -1;

/// Decoder settings.
#[repr(C)]
pub struct Altc264Settings {
    /// Decoding threads: negative for one per CPU (the default), 0 to
    /// decode on the calling thread, n > 1 to decode several pictures at
    /// once (frame threading).
    pub threads: c_int,
    /// Non-zero: decode every MVC view; zero: base view only.
    pub all_views: c_int,
    /// ALTC264_STEREO_*: output the two views of each access unit as one
    /// packed frame (with all_views), or SEPARATE for one frame per view.
    pub stereo_layout: c_int,
    /// Non-zero: the base view is the right eye (default: left). Blu-ray
    /// signals it in the playlist; Matroska in StereoMode (block_rl).
    pub base_view_right: c_int,
}

/// One picture per view.
pub const ALTC264_STEREO_SEPARATE: c_int = 0;
/// Left | right, full resolution (double width).
pub const ALTC264_STEREO_SIDE_BY_SIDE: c_int = 1;
/// Left | right, each half width.
pub const ALTC264_STEREO_SIDE_BY_SIDE_HALF: c_int = 2;
/// Left above right, full resolution (double height).
pub const ALTC264_STEREO_TOP_AND_BOTTOM: c_int = 3;
/// Left above right, each half height.
pub const ALTC264_STEREO_TOP_AND_BOTTOM_HALF: c_int = 4;
/// HDMI 1.4 frame packing (1920x2205 for 1080p).
pub const ALTC264_STEREO_FRAME_PACKING: c_int = 5;

fn layout(v: c_int) -> Option<StereoLayout> {
    use StereoLayout::*;
    match v {
        ALTC264_STEREO_SIDE_BY_SIDE => Some(SideBySide),
        ALTC264_STEREO_SIDE_BY_SIDE_HALF => Some(SideBySideHalf),
        ALTC264_STEREO_TOP_AND_BOTTOM => Some(TopAndBottom),
        ALTC264_STEREO_TOP_AND_BOTTOM_HALF => Some(TopAndBottomHalf),
        ALTC264_STEREO_FRAME_PACKING => Some(FramePacking),
        _ => None,
    }
}

/// Graphics plane offset of one Blu-ray 3D offset sequence for a frame.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Altc264PlaneOffset {
    /// plane_offset_direction_flag.
    pub direction: u8,
    /// plane_offset_value, in pixels.
    pub value: u8,
}

/// A decoded picture (planar YUV, cropped). The planes point into the
/// decoder's reference buffer and stay valid until altc264_picture_unref.
#[repr(C)]
pub struct Altc264Picture {
    /// Y, Cb, Cr planes (Cb and Cr are NULL for 4:0:0): uint8_t samples, or
    /// native-endian uint16_t samples when a bit depth exceeds 8.
    pub data: [*const u8; 3],
    /// Distance between rows, in bytes.
    pub stride: [isize; 3],
    /// Bits per luma and chroma sample (8 to 14).
    pub bit_depth_luma: c_int,
    pub bit_depth_chroma: c_int,
    pub width: c_int,
    pub height: c_int,
    pub chroma_width: c_int,
    pub chroma_height: c_int,
    /// MVC view_id (0 for the base view).
    pub view_id: c_int,
    pub poc: c_int,
    /// Timestamp passed with the packet that started this picture's access
    /// unit.
    pub pts: i64,
    pub has_pts: c_int,
    /// Blu-ray 3D plane offsets, one per offset sequence, or NULL.
    pub plane_offsets: *const Altc264PlaneOffset,
    pub num_plane_offsets: c_int,
    /// Owned by the decoder library; released by altc264_picture_unref.
    pub opaque: *mut core::ffi::c_void,
}

struct Owned {
    frame: Frame,
    offsets: Vec<Altc264PlaneOffset>,
}

/// Opaque decoder handle.
pub struct Altc264Decoder {
    dec: Decoder,
    /// Length field size of AVCC packets; 0 for Annex B.
    length_size: usize,
}

fn new_decoder(
    (threads, all_views, stereo, base_right): (usize, bool, Option<StereoLayout>, bool),
) -> Decoder {
    let mut d = Decoder::with_threads(threads);
    d.decode_all_views = all_views;
    d.stereo_layout = stereo;
    d.base_view_right = base_right;
    d
}

/// Library version as a static string.
#[unsafe(no_mangle)]
pub extern "C" fn altc264_version() -> *const c_char {
    const V: &CStr =
        match CStr::from_bytes_with_nul(concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes()) {
            Ok(v) => v,
            Err(_) => panic!("version contains NUL"),
        };
    V.as_ptr()
}

/// Fills `s` with the default settings: all views, one thread per CPU.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn altc264_default_settings(s: *mut Altc264Settings) {
    if let Some(s) = unsafe { s.as_mut() } {
        *s = Altc264Settings {
            threads: std::thread::available_parallelism().map_or(1, |n| n.get()) as c_int,
            all_views: 1,
            stereo_layout: ALTC264_STEREO_SEPARATE,
            base_view_right: 0,
        };
    }
}

/// Creates a decoder. `settings` may be NULL for the defaults.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn altc264_open(
    out: *mut *mut Altc264Decoder,
    settings: *const Altc264Settings,
) -> c_int {
    let Some(out) = (unsafe { out.as_mut() }) else {
        return ALTC264_EINVAL;
    };
    let mut defaults = Altc264Settings {
        threads: 0,
        all_views: 0,
        stereo_layout: 0,
        base_view_right: 0,
    };
    let settings = match unsafe { settings.as_ref() } {
        Some(s) => s,
        None => {
            unsafe { altc264_default_settings(&mut defaults) };
            &defaults
        }
    };
    let settings = (
        if settings.threads < 0 {
            std::thread::available_parallelism().map_or(1, |n| n.get())
        } else {
            settings.threads as usize
        },
        settings.all_views != 0,
        layout(settings.stereo_layout),
        settings.base_view_right != 0,
    );
    let d = Box::new(Altc264Decoder {
        dec: new_decoder(settings),
        length_size: 0,
    });
    *out = Box::into_raw(d);
    ALTC264_OK
}

/// Destroys a decoder and sets `*dec` to NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn altc264_close(dec: *mut *mut Altc264Decoder) {
    if let Some(p) = unsafe { dec.as_mut() }
        && !p.is_null()
    {
        drop(unsafe { Box::from_raw(*p) });
        *p = std::ptr::null_mut();
    }
}

/// Codec extradata: an avcC record (then packets are length-prefixed) or
/// Annex B parameter sets.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn altc264_set_extradata(
    dec: *mut Altc264Decoder,
    data: *const u8,
    size: usize,
) -> c_int {
    let Some(d) = (unsafe { dec.as_mut() }) else {
        return ALTC264_EINVAL;
    };
    if data.is_null() || size == 0 {
        return ALTC264_OK;
    }
    let b = unsafe { std::slice::from_raw_parts(data, size) };
    if let Some(cfg) = AvcConfig::parse(b) {
        d.length_size = cfg.length_size;
        for nal in &cfg.nals {
            d.dec.decode_nal(nal);
        }
    } else {
        d.dec.decode_annex_b(b);
    }
    ALTC264_OK
}

/// Decodes one packet (an access unit or part of one). `has_pts` says
/// whether `pts` is valid; it is returned with the pictures of the access
/// units this packet starts.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn altc264_send_data(
    dec: *mut Altc264Decoder,
    data: *const u8,
    size: usize,
    pts: i64,
    has_pts: c_int,
) -> c_int {
    let Some(d) = (unsafe { dec.as_mut() }) else {
        return ALTC264_EINVAL;
    };
    if data.is_null() && size > 0 {
        return ALTC264_EINVAL;
    }
    let b = if size == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(data, size) }
    };
    d.dec.set_timestamp((has_pts != 0).then_some(pts));
    if d.length_size > 0 {
        d.dec.decode_length_prefixed(b, d.length_size);
    } else {
        d.dec.decode_annex_b(b);
    }
    ALTC264_OK
}

/// End of stream: makes every remaining picture available.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn altc264_drain(dec: *mut Altc264Decoder) {
    if let Some(d) = unsafe { dec.as_mut() } {
        d.dec.flush();
    }
}

/// Seeking: discards every picture decoded or in progress and every
/// picture not yet returned. Decoding restarts with the next data as at the
/// start of a stream, keeping the parameter sets received so far (in-band
/// or from altc264_set_extradata), the settings and the threads.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn altc264_reset(dec: *mut Altc264Decoder) {
    if let Some(d) = unsafe { dec.as_mut() } {
        d.dec.discard();
    }
}

/// Number of slices that failed to decode so far.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn altc264_error_count(dec: *const Altc264Decoder) -> u64 {
    unsafe { dec.as_ref() }.map_or(0, |d| d.dec.errors())
}

/// Returns ALTC264_OK and fills `pic`, or ALTC264_EAGAIN when no picture
/// is ready.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn altc264_get_picture(
    dec: *mut Altc264Decoder,
    pic: *mut Altc264Picture,
) -> c_int {
    let (Some(d), Some(pic)) = (unsafe { dec.as_mut() }, unsafe { pic.as_mut() }) else {
        return ALTC264_EINVAL;
    };
    let Some(frame) = d.dec.next_frame() else {
        return ALTC264_EAGAIN;
    };
    let offsets: Vec<Altc264PlaneOffset> = frame
        .plane_offsets
        .iter()
        .flatten()
        .map(|o| Altc264PlaneOffset {
            direction: o.direction as u8,
            value: o.value,
        })
        .collect();
    let mut owned = Box::new(Owned { frame, offsets });
    let f = &owned.frame;
    let high = f.bit_depth.iter().any(|&b| b > 8);
    let plane = |i: usize| {
        f.plane(i).map_or((std::ptr::null(), 0), |p| {
            if high {
                (p.data16.as_ptr().cast(), 2 * p.stride as isize)
            } else {
                (p.data.as_ptr(), p.stride as isize)
            }
        })
    };
    let planes = [plane(0), plane(1), plane(2)];
    *pic = Altc264Picture {
        data: planes.map(|p| p.0),
        stride: planes.map(|p| p.1),
        bit_depth_luma: f.bit_depth[0] as c_int,
        bit_depth_chroma: f.bit_depth[1] as c_int,
        width: f.width as c_int,
        height: f.height as c_int,
        chroma_width: f.chroma_width as c_int,
        chroma_height: f.chroma_height as c_int,
        view_id: f.view_id as c_int,
        poc: f.poc,
        pts: f.pts.unwrap_or(0),
        has_pts: f.pts.is_some() as c_int,
        plane_offsets: if f.plane_offsets.is_some() {
            owned.offsets.as_ptr()
        } else {
            std::ptr::null()
        },
        num_plane_offsets: if f.plane_offsets.is_some() {
            owned.offsets.len() as c_int
        } else {
            0
        },
        opaque: std::ptr::null_mut(),
    };
    pic.opaque = (&mut *owned as *mut Owned).cast();
    let _ = Box::into_raw(owned);
    ALTC264_OK
}

/// Releases a picture from altc264_get_picture.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn altc264_picture_unref(pic: *mut Altc264Picture) {
    if let Some(p) = unsafe { pic.as_mut() }
        && !p.opaque.is_null()
    {
        drop(unsafe { Box::from_raw(p.opaque.cast::<Owned>()) });
        p.opaque = std::ptr::null_mut();
        p.data = [std::ptr::null(); 3];
        p.plane_offsets = std::ptr::null();
    }
}
