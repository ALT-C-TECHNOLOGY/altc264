use std::sync::{LazyLock, Mutex};

use altc264::decoder::{Decoder, Frame};
use altc264::nal::AvcConfig;
use altc264::stereo::StereoLayout as Layout;
use gst::glib;
use gst::prelude::*;
use gst::subclass::prelude::*;
use gst_video::prelude::*;
use gst_video::subclass::prelude::*;

use super::{BaseView, StereoLayout};

static CAT: LazyLock<gst::DebugCategory> = LazyLock::new(|| {
    gst::DebugCategory::new(
        "altc264dec",
        gst::DebugColorFlags::empty(),
        Some("altc264 H.264 / MVC decoder"),
    )
});

/// UUID of the per-frame Blu-ray 3D graphics plane offsets attached as
/// user data unregistered SEI meta: one byte per offset sequence, bit 7 the
/// direction flag, bits 0-6 the offset (as the FFmpeg wrapper exports).
const PLANE_OFFSETS_UUID: [u8; 16] = [
    0xa1, 0x7c, 0x26, 0x4e, 0x3d, 0x2b, 0x4f, 0x0e, 0x9b, 0x52, 0x6c, 0x21, 0x64, 0x70, 0x6c, 0x6f,
];

#[derive(Clone, Copy)]
struct Settings {
    threads: u32,
    layout: StereoLayout,
    base_view: BaseView,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            threads: 0,
            layout: StereoLayout::SideBySide,
            base_view: BaseView::Auto,
        }
    }
}

struct State {
    dec: Decoder,
    /// Length field size of AVC samples; 0 for byte-stream.
    length_size: usize,
    /// The negotiated output: (format, width, height, multiview mode).
    output: Option<(
        gst_video::VideoFormat,
        u32,
        u32,
        gst_video::VideoMultiviewMode,
    )>,
    input_state: gst_video::VideoCodecState<'static, gst_video::video_codec_state::Readable>,
}

#[derive(Default)]
pub struct Altc264Dec {
    settings: Mutex<Settings>,
    state: Mutex<Option<State>>,
}

impl Altc264Dec {
    fn new_decoder(&self, base_right: bool) -> Decoder {
        let settings = *self.settings.lock().unwrap();
        let threads = if settings.threads == 0 {
            std::thread::available_parallelism().map_or(1, |n| n.get())
        } else {
            settings.threads as usize
        };
        let mut dec = Decoder::with_threads(threads);
        dec.decode_all_views = settings.layout != StereoLayout::Mono;
        dec.stereo_layout = match settings.layout {
            StereoLayout::Mono => None,
            StereoLayout::SideBySide => Some(Layout::SideBySide),
            StereoLayout::SideBySideHalf => Some(Layout::SideBySideHalf),
            StereoLayout::TopBottom => Some(Layout::TopAndBottom),
            StereoLayout::TopBottomHalf => Some(Layout::TopAndBottomHalf),
        };
        dec.base_view_right = match settings.base_view {
            BaseView::Auto => base_right,
            BaseView::Left => false,
            BaseView::Right => true,
        };
        dec
    }

    /// Whether upstream says the right view comes first (Matroska block_rl).
    fn upstream_right_first(caps: Option<&gst::CapsRef>) -> bool {
        // A GstVideoMultiviewFlagsSet serializes as "flags:mask:names",
        // flags in hex; GST_VIDEO_MULTIVIEW_FLAGS_RIGHT_VIEW_FIRST is bit 0.
        caps.and_then(|c| c.structure(0))
            .and_then(|s| s.value("multiview-flags").ok())
            .and_then(|v| v.serialize().ok())
            .and_then(|s| u32::from_str_radix(s.split(':').next()?, 16).ok())
            .is_some_and(|flags| flags & 1 != 0)
    }

    /// Sends decoded frames downstream, matching them to their codec frames
    /// by the system frame number carried as the timestamp.
    fn output_frames(&self, state: &mut State) -> Result<gst::FlowSuccess, gst::FlowError> {
        let obj = self.obj();
        while let Some(f) = state.dec.next_frame() {
            let Some(number) = f.pts else { continue };
            let Some(mut frame) = obj.frame(number as i32) else {
                gst::warning!(CAT, imp = self, "no codec frame {number}");
                continue;
            };
            self.negotiate_for(state, &f)?;
            obj.allocate_output_frame(&mut frame, None)?;
            let (format, w, h, _) = state.output.unwrap();
            let info = gst_video::VideoInfo::builder(format, w, h)
                .build()
                .map_err(|_| gst::FlowError::NotNegotiated)?;
            {
                let buffer = frame.output_buffer_mut().ok_or(gst::FlowError::Error)?;
                if let Some(offsets) = &f.plane_offsets {
                    let mut data = Vec::with_capacity(offsets.len());
                    for o in offsets {
                        data.push((o.direction as u8) << 7 | (o.value & 0x7f));
                    }
                    gst_video::video_meta::VideoSeiUserDataUnregisteredMeta::add(
                        buffer,
                        &PLANE_OFFSETS_UUID,
                        &data,
                    );
                }
                let mut v = gst_video::VideoFrameRef::from_buffer_ref_writable(buffer, &info)
                    .map_err(|_| gst::FlowError::Error)?;
                copy_planes(&f, &mut v);
            }
            obj.finish_frame(frame)?;
        }
        Ok(gst::FlowSuccess::Ok)
    }

    /// (Re)negotiates the output caps when the frame size or kind changes.
    fn negotiate_for(&self, state: &mut State, f: &Frame) -> Result<(), gst::FlowError> {
        let format = output_format(f).0;
        // 2D streams come out unpacked even with a stereo layout set.
        let mode = match f.packed {
            None => gst_video::VideoMultiviewMode::Mono,
            Some(Layout::SideBySide | Layout::SideBySideHalf) => {
                gst_video::VideoMultiviewMode::SideBySide
            }
            Some(_) => gst_video::VideoMultiviewMode::TopBottom,
        };
        let key = (format, f.width as u32, f.height as u32, mode);
        if state.output == Some(key) {
            return Ok(());
        }
        let obj = self.obj();
        let mut out = obj.set_output_state(format, key.1, key.2, Some(&state.input_state))?;
        if mode != gst_video::VideoMultiviewMode::Mono {
            let mut flags = gst_video::VideoMultiviewFlags::empty();
            if matches!(
                f.packed,
                Some(Layout::SideBySideHalf | Layout::TopAndBottomHalf)
            ) {
                flags |= gst_video::VideoMultiviewFlags::HALF_ASPECT;
            }
            let info = gst_video::VideoInfo::builder_from_info(out.info())
                .multiview_mode(mode)
                .multiview_flags(flags)
                .views(2)
                .build()
                .map_err(|_| gst::FlowError::NotNegotiated)?;
            out.set_info(info);
        }
        obj.negotiate(out)?;
        state.output = Some(key);
        Ok(())
    }
}

/// The output format of a frame and how far its samples shift to fill the
/// format's sample size (negative: right). Samples above 8 bits go in the
/// nearest 10, 12 or 16-bit format.
fn output_format(f: &Frame) -> (gst_video::VideoFormat, i32) {
    use gst_video::VideoFormat as F;
    let chroma = f.plane(1).map(|c| c.height == f.height);
    let depth = f.bit_depth[0].max(if chroma.is_some() { f.bit_depth[1] } else { 0 }) as i32;
    if depth <= 8 {
        let format = match chroma {
            None => F::Gray8,
            Some(true) => F::Y42b,
            Some(false) => F::I420,
        };
        return (format, 0);
    }
    let container = if chroma.is_none() {
        16
    } else if depth <= 10 {
        10
    } else {
        12
    };
    let format = match (chroma, container) {
        (None, _) => F::Gray16Le,
        (Some(true), 10) => F::I42210le,
        (Some(true), _) => F::I42212le,
        (Some(false), 10) => F::I42010le,
        (Some(false), _) => F::I42012le,
    };
    (format, container - depth)
}

fn copy_planes(f: &Frame, v: &mut gst_video::VideoFrameRef<&mut gst::BufferRef>) {
    let (_, shift) = output_format(f);
    if f.bit_depth.iter().any(|&b| b > 8) {
        for i in 0..v.n_planes() as usize {
            let Some(src) = f.plane(i) else { continue };
            let stride = v.plane_stride()[i] as usize;
            let Ok(dst) = v.plane_data_mut(i as u32) else {
                continue;
            };
            for (y, row) in src.rows16().enumerate() {
                let out = &mut dst[y * stride..y * stride + 2 * row.len()];
                for (o, &s) in out.chunks_exact_mut(2).zip(row) {
                    let s = if shift >= 0 { s << shift } else { s >> -shift };
                    o.copy_from_slice(&s.to_le_bytes());
                }
            }
        }
        return;
    }
    for i in 0..v.n_planes() as usize {
        let Some(src) = f.plane(i) else { continue };
        let stride = v.plane_stride()[i] as usize;
        let Ok(dst) = v.plane_data_mut(i as u32) else {
            continue;
        };
        for (y, row) in src.rows().enumerate() {
            dst[y * stride..y * stride + row.len()].copy_from_slice(row);
        }
    }
}

#[glib::object_subclass]
impl ObjectSubclass for Altc264Dec {
    const NAME: &'static str = "GstAltc264Dec";
    type Type = super::Altc264Dec;
    type ParentType = gst_video::VideoDecoder;
}

impl ObjectImpl for Altc264Dec {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: LazyLock<Vec<glib::ParamSpec>> = LazyLock::new(|| {
            vec![
                glib::ParamSpecUInt::builder("threads")
                    .nick("Threads")
                    .blurb("Decoding threads (0 = one per CPU)")
                    .default_value(0)
                    .mutable_ready()
                    .build(),
                glib::ParamSpecEnum::builder_with_default(
                    "stereo-layout",
                    StereoLayout::SideBySide,
                )
                .nick("Stereo layout")
                .blurb("How the two views of an MVC (3D) stream are output")
                .mutable_ready()
                .build(),
                glib::ParamSpecEnum::builder_with_default("base-view", BaseView::Auto)
                    .nick("Base view")
                    .blurb("Which eye the MVC base view is")
                    .mutable_ready()
                    .build(),
            ]
        });
        PROPERTIES.as_ref()
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        let mut s = self.settings.lock().unwrap();
        match pspec.name() {
            "threads" => s.threads = value.get().unwrap(),
            "stereo-layout" => s.layout = value.get().unwrap(),
            "base-view" => s.base_view = value.get().unwrap(),
            _ => unreachable!(),
        }
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        let s = self.settings.lock().unwrap();
        match pspec.name() {
            "threads" => s.threads.to_value(),
            "stereo-layout" => s.layout.to_value(),
            "base-view" => s.base_view.to_value(),
            _ => unreachable!(),
        }
    }
}

impl GstObjectImpl for Altc264Dec {}

impl ElementImpl for Altc264Dec {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static META: LazyLock<gst::subclass::ElementMetadata> = LazyLock::new(|| {
            gst::subclass::ElementMetadata::new(
                "altc264 H.264 / MVC decoder",
                "Codec/Decoder/Video",
                "Decodes H.264 video, including MVC stereo 3D",
                "altc264 developers",
            )
        });
        Some(&*META)
    }

    fn pad_templates() -> &'static [gst::PadTemplate] {
        static TEMPLATES: LazyLock<Vec<gst::PadTemplate>> = LazyLock::new(|| {
            let sink = gst::Caps::builder("video/x-h264")
                .field("stream-format", gst::List::new(["byte-stream", "avc"]))
                .field("alignment", "au")
                .build();
            let src = gst_video::VideoCapsBuilder::new()
                .format_list([
                    gst_video::VideoFormat::I420,
                    gst_video::VideoFormat::Y42b,
                    gst_video::VideoFormat::Gray8,
                    gst_video::VideoFormat::I42010le,
                    gst_video::VideoFormat::I42210le,
                    gst_video::VideoFormat::I42012le,
                    gst_video::VideoFormat::I42212le,
                    gst_video::VideoFormat::Gray16Le,
                ])
                .build();
            vec![
                gst::PadTemplate::new(
                    "sink",
                    gst::PadDirection::Sink,
                    gst::PadPresence::Always,
                    &sink,
                )
                .unwrap(),
                gst::PadTemplate::new(
                    "src",
                    gst::PadDirection::Src,
                    gst::PadPresence::Always,
                    &src,
                )
                .unwrap(),
            ]
        });
        TEMPLATES.as_ref()
    }
}

impl VideoDecoderImpl for Altc264Dec {
    fn stop(&self) -> Result<(), gst::ErrorMessage> {
        *self.state.lock().unwrap() = None;
        Ok(())
    }

    fn set_format(
        &self,
        input_state: &gst_video::VideoCodecState<'static, gst_video::video_codec_state::Readable>,
    ) -> Result<(), gst::LoggableError> {
        let caps = input_state.caps();
        let right = Self::upstream_right_first(caps);
        let mut dec = self.new_decoder(right);
        let mut length_size = 0;
        if let Some(cd) = input_state.codec_data() {
            let map = cd
                .map_readable()
                .map_err(|_| gst::loggable_error!(CAT, "codec_data"))?;
            match AvcConfig::parse(&map) {
                Some(cfg) => {
                    length_size = cfg.length_size;
                    for nal in &cfg.nals {
                        dec.decode_nal(nal);
                    }
                }
                None => dec.decode_annex_b(&map),
            }
        }
        *self.state.lock().unwrap() = Some(State {
            dec,
            length_size,
            output: None,
            input_state: input_state.clone(),
        });
        Ok(())
    }

    fn handle_frame(
        &self,
        frame: gst_video::VideoCodecFrame,
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
        let mut guard = self.state.lock().unwrap();
        let state = guard.as_mut().ok_or(gst::FlowError::NotNegotiated)?;
        let number = frame.system_frame_number();
        {
            let input = frame.input_buffer().ok_or(gst::FlowError::Error)?;
            let map = input.map_readable().map_err(|_| gst::FlowError::Error)?;
            state.dec.set_timestamp(Some(number as i64));
            if state.length_size > 0 {
                state.dec.decode_length_prefixed(&map, state.length_size);
            } else {
                state.dec.decode_annex_b(&map);
            }
        }
        drop(frame);
        self.output_frames(state)
    }

    fn drain(&self) -> Result<gst::FlowSuccess, gst::FlowError> {
        let mut guard = self.state.lock().unwrap();
        let Some(state) = guard.as_mut() else {
            return Ok(gst::FlowSuccess::Ok);
        };
        state.dec.flush();
        self.output_frames(state)
    }

    fn finish(&self) -> Result<gst::FlowSuccess, gst::FlowError> {
        self.drain()
    }

    fn flush(&self) -> bool {
        // Seeking: drop every picture; the parameter sets received so far
        // (codec_data and in-band) are kept for the next keyframe.
        let mut guard = self.state.lock().unwrap();
        if let Some(state) = guard.as_mut() {
            state.dec.discard();
        }
        true
    }
}
