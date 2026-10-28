//! Top-level decoder: parameter set management, picture assembly, the DPB
//! and output in display order, for single-view and MVC streams.
//!
//! MVC (Annex G): each view has its own POC state, frame_num tracking and
//! DPB (marking only ever touches a view's own pictures, G.8.3). The base
//! view picture of each access unit is the inter-view prediction reference
//! for the non-base view.
//!
//! Threading: the front end parses parameter sets and slice headers and
//! decides picture boundaries; each view is decoded by its own worker. A
//! non-base view picture needs only its own view's references and the base
//! picture of the same access unit, so view 1 of access unit n decodes in
//! parallel with view 0 of access unit n + 1. With `threads == 0` the same
//! workers run inline on the calling thread.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, SyncSender, channel, sync_channel};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use crate::Error;
use crate::deblock::{SliceFilter, deblock_mbaff, deblock_mbaff_t, deblock_picture, deblock_rows};
use crate::dpb::{
    Decoded, Dpb, Entry, GopMeta, Output, OutputInfo, PicStructure, RefLists, RefPic, dpb_size,
};
use crate::nal::{AnnexBIter, MvcExtension, NalHeader, NalType, unescape_into};
use crate::picture::{Parity, Picture, Pixel, Plane, RawPlane};
use crate::poc::{Poc, PocState, first_slice_of_new_picture, structure};
use crate::pool::{JobQueue, Pool};
use crate::progress::PicState;
use crate::ps::{Pps, Sps, SubsetSps};
use crate::sei::{OffsetMetadata, PlaneOffset};
use crate::slice::{DecRefPicMarking, Mmco, SliceHeader, SliceType};
use crate::slice_decoder::{
    InfoView, MbDone, MbInfo, SharedPic, SliceArea, SliceCtx, decode_slice,
};
use crate::transform::LevelScale;

/// A decoded output frame. It normally shares the decoder's picture buffer,
/// so it costs no copy; the planes are views of the cropped area.
pub struct Frame {
    pub width: usize,
    pub height: usize,
    /// Chroma plane size (0 for 4:0:0).
    pub chroma_width: usize,
    pub chroma_height: usize,
    /// BitDepthY and BitDepthC: samples are 16-bit (FramePlane::data16)
    /// when either exceeds 8.
    pub bit_depth: [u32; 2],
    /// VUI (colour_primaries, transfer_characteristics, matrix_coefficients).
    /// matrix_coefficients 0 means the planes hold G, B and R.
    pub colour: Option<(u8, u8, u8)>,
    /// VUI video_full_range_flag.
    pub full_range: bool,
    pub poc: i32,
    pub view_id: u16,
    /// MVC view order index (0 for the base view).
    pub view_index: usize,
    /// Blu-ray 3D graphics plane offsets for this frame, one per offset
    /// sequence (subtitle tracks refer to a sequence by its 3d-plane number).
    pub plane_offsets: Option<Vec<PlaneOffset>>,
    /// The timestamp given with Decoder::set_timestamp for this frame's
    /// access unit, carried through reordering.
    pub pts: Option<i64>,
    /// Both views of a stereo access unit packed in this layout
    /// (Decoder::stereo_layout).
    pub packed: Option<crate::stereo::StereoLayout>,
    storage: Storage,
}

enum Storage {
    /// A decoded picture and the top-left luma sample of its cropped area.
    Shared {
        pic: Arc<Picture>,
        crop: (usize, usize),
        /// Done once the picture is decoded (frame threading).
        state: Arc<PicState>,
    },
    /// Tightly packed planes (stereo packing).
    Owned(Vec<Vec<u8>>),
}

/// One plane of a [`Frame`]: rows of `width` samples, `stride` apart.
#[derive(Clone, Copy)]
pub struct FramePlane<'a> {
    /// 8-bit samples: starts at the first visible sample and holds at least
    /// `(height - 1) * stride + width` samples. Empty for high bit depths.
    pub data: &'a [u8],
    /// As `data` for frames of more than 8 bits per sample (else empty).
    pub data16: &'a [u16],
    pub stride: usize,
    pub width: usize,
    pub height: usize,
}

impl<'a> FramePlane<'a> {
    /// A row of 8-bit samples.
    pub fn row(&self, y: usize) -> &'a [u8] {
        &self.data[y * self.stride..y * self.stride + self.width]
    }

    pub fn rows(&self) -> impl Iterator<Item = &'a [u8]> + '_ {
        (0..self.height).map(|y| self.row(y))
    }

    /// A row of high bit depth samples.
    pub fn row16(&self, y: usize) -> &'a [u16] {
        &self.data16[y * self.stride..y * self.stride + self.width]
    }

    pub fn rows16(&self) -> impl Iterator<Item = &'a [u16]> + '_ {
        (0..self.height).map(|y| self.row16(y))
    }
}

impl Frame {
    /// Waits until the frame's picture is decoded (frame threading hands
    /// frames over in output order, possibly before).
    fn wait(&self) {
        if let Storage::Shared { state, .. } = &self.storage {
            state.wait_done();
        }
    }

    /// Whether the frame's picture is decoded.
    fn decoded(&self) -> bool {
        match &self.storage {
            Storage::Shared { state, .. } => state.is_done(),
            Storage::Owned(_) => true,
        }
    }

    /// Plane 0 (Y), 1 (Cb) or 2 (Cr); None for chroma of 4:0:0 frames.
    pub fn plane(&self, i: usize) -> Option<FramePlane<'_>> {
        let (w, h) = if i == 0 {
            (self.width, self.height)
        } else {
            (self.chroma_width, self.chroma_height)
        };
        match &self.storage {
            Storage::Shared { pic, crop, .. } => {
                let p = pic.planes.get(i)?;
                let (x0, y0) = if i == 0 {
                    *crop
                } else {
                    (crop.0 * pic.mb_c.0 / 16, crop.1 * pic.mb_c.1 / 16)
                };
                let o = y0 * p.stride + x0;
                Some(FramePlane {
                    data: p.data.get(o..).unwrap_or_default(),
                    data16: p.data16.get(o..).unwrap_or_default(),
                    stride: p.stride,
                    width: w,
                    height: h,
                })
            }
            Storage::Owned(planes) => Some(FramePlane {
                data: planes.get(i)?,
                data16: &[],
                stride: w,
                width: w,
                height: h,
            }),
        }
    }

    /// A frame of tightly packed planes, without metadata.
    #[cfg(test)]
    pub(crate) fn owned(w: usize, h: usize, cw: usize, ch: usize, planes: Vec<Vec<u8>>) -> Self {
        Frame {
            width: w,
            height: h,
            chroma_width: cw,
            chroma_height: ch,
            bit_depth: [8, 8],
            colour: None,
            full_range: false,
            poc: 0,
            view_id: 0,
            view_index: 0,
            plane_offsets: None,
            pts: None,
            packed: None,
            storage: Storage::Owned(planes),
        }
    }

    /// This frame's metadata with other, tightly packed planes.
    pub(crate) fn with_planes(
        &self,
        w: usize,
        h: usize,
        cw: usize,
        ch: usize,
        planes: Vec<Vec<u8>>,
        packed: Option<crate::stereo::StereoLayout>,
    ) -> Self {
        Frame {
            width: w,
            height: h,
            chroma_width: cw,
            chroma_height: ch,
            bit_depth: self.bit_depth,
            colour: self.colour,
            full_range: self.full_range,
            poc: self.poc,
            view_id: self.view_id,
            view_index: self.view_index,
            plane_offsets: self.plane_offsets.clone(),
            pts: self.pts,
            packed,
            storage: Storage::Owned(planes),
        }
    }
}

fn output_info(sps: &Sps, view_id: u16) -> OutputInfo {
    let (w, h) = sps.cropped_size();
    let [l, _, t, _] = sps.crop.unwrap_or([0; 4]);
    let (sub_w, sub_h) = match sps.chroma_array_type() {
        1 => (2, 2),
        2 => (2, 1),
        _ => (1, 1),
    };
    OutputInfo {
        width: w as usize,
        height: h as usize,
        crop_x: sub_w * l as usize,
        // CropUnitY doubles without frame_mbs_only (7-19).
        crop_y: sub_h * (2 - sps.frame_mbs_only as usize) * t as usize,
        view_id,
        pts: None,
        au: 0,
        hidden: false,
        colour: sps.vui.as_ref().and_then(|v| v.colour_description),
        full_range: sps.vui.as_ref().is_some_and(|v| v.video_full_range),
    }
}

fn to_frame(o: Output, view_index: usize, plane_offsets: Option<Vec<PlaneOffset>>) -> Frame {
    let info = o.out;
    let (w, h) = (info.width, info.height);
    let (cw, ch) = match o.pic.mb_c {
        (0, _) => (0, 0),
        (mw, mh) => (w * mw / 16, h * mh / 16),
    };
    Frame {
        width: w,
        height: h,
        chroma_width: cw,
        chroma_height: ch,
        bit_depth: o.pic.bit_depth,
        colour: info.colour,
        full_range: info.full_range,
        poc: o.poc,
        view_id: info.view_id,
        view_index,
        plane_offsets,
        pts: info.pts,
        packed: None,
        storage: Storage::Shared {
            pic: o.pic,
            crop: (info.crop_x, info.crop_y),
            state: o.state,
        },
    }
}

/// Spare RBSP buffers: a slice's goes back once the slice is decoded, so
/// the front end does not allocate (and grow) one per slice.
type RbspPool = Arc<Mutex<Vec<Vec<u8>>>>;

/// A slice's RBSP, returned to its pool when dropped.
struct Rbsp {
    buf: Vec<u8>,
    pool: RbspPool,
}

impl std::ops::Deref for Rbsp {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &self.buf
    }
}

impl Drop for Rbsp {
    fn drop(&mut self) {
        if let Ok(mut p) = self.pool.lock()
            && p.len() < 64
        {
            p.push(std::mem::take(&mut self.buf));
        }
    }
}

/// One slice handed to a view worker, with the parameter sets it uses.
struct SliceJob {
    nal: NalHeader,
    header: SliceHeader,
    rbsp: Rbsp,
    sps: Arc<Sps>,
    pps: Arc<Pps>,
    subset: Option<Arc<SubsetSps>>,
    /// First slice of a new picture (decided by the front end).
    new_picture: bool,
    /// Access unit number, shared by the views of an access unit.
    au: u64,
    /// Output period of the access unit: it goes up at each access unit
    /// with an IDR picture or a memory_management_control_operation 5,
    /// which output every earlier picture first (C.4.4). Within a period,
    /// pictures are output in PicOrderCnt order.
    period: u64,
    view_id: u16,
    pts: Option<i64>,
    /// With new_picture: where this picture is published for the views
    /// that follow it in the access unit.
    publish: Option<Publisher>,
    /// The pictures of the preceding views of the access unit (view_id, slot).
    earlier_views: Vec<(u16, Arc<PicSlot>)>,
}

enum Msg {
    Slice(Box<SliceJob>),
    /// The current picture is complete.
    Finish,
    /// End of stream: finish and output everything.
    Flush,
    /// A seek: drop every picture without output and start again, as at
    /// the start of a stream (parameter sets are kept by the Decoder).
    Discard,
    /// Offset metadata for the GOP starting at the current access unit.
    Metadata(Arc<OffsetMetadata>),
    /// Replies once everything sent before has been handled.
    Sync(std::sync::mpsc::Sender<()>),
}

/// A view component of the current access unit, once decoded, for
/// inter-view prediction by the views decoded after it (G.8.2.1). None when
/// the picture is not an inter-view reference (inter_view_flag 0) or failed.
#[derive(Default)]
struct PicSlot {
    pic: Mutex<Option<Option<RefPic>>>,
    ready: Condvar,
}

impl PicSlot {
    fn wait(&self) -> Option<RefPic> {
        let mut p = self.pic.lock().unwrap();
        loop {
            if let Some(p) = &*p {
                return p.clone();
            }
            p = self.ready.wait(p).unwrap();
        }
    }
}

/// The producing side of a PicSlot; publishes None if dropped unset, so a
/// waiting view never blocks on a picture that will not come.
struct Publisher(Arc<PicSlot>);

impl Publisher {
    fn set(self, pic: Option<RefPic>) {
        *self.0.pic.lock().unwrap() = Some(pic);
        self.0.ready.notify_all();
    }
}

impl Drop for Publisher {
    fn drop(&mut self) {
        let mut p = self.0.pic.lock().unwrap();
        if p.is_none() {
            *p = Some(None);
            self.0.ready.notify_all();
        }
    }
}

/// Shared error accounting.
#[derive(Default)]
struct Errors {
    count: AtomicU64,
    last: Mutex<Option<Error>>,
}

impl Errors {
    fn add(&self, e: Error) {
        self.count.fetch_add(1, Ordering::Relaxed);
        *self.last.lock().unwrap() = Some(e);
    }
}

struct CurrentPicture {
    pic: Picture,
    info: Vec<MbInfo>,
    sps: Arc<Sps>,
    first: SliceHeader,
    ref_idc: u8,
    /// inter_view_flag of the view component (from the prefix NAL unit for
    /// the base view; 1 when absent).
    inter_view_ref: bool,
    publish: Option<Publisher>,
    /// PicOrderCnt(CurrPic).
    poc: i32,
    /// Top and bottom field order counts.
    field_poc: Poc,
    structure: PicStructure,
    slices: u32,
    filters: Vec<SliceFilter>,
    out: OutputInfo,
    /// Output period of the access unit (SliceJob::period).
    period: u64,
    meta: Option<GopMeta>,
    /// Slices waiting to be decoded when the picture is complete.
    pending: Vec<PendingSlice>,
    /// Where concealed macroblocks are copied from (the first reference).
    conceal: Option<RefPic>,
}

/// A slice with everything its decoding needs, prepared on arrival.
struct PendingSlice {
    job: SliceJob,
    ext: crate::ps::PpsExt,
    ls: Arc<LevelScale>,
    ref_lists: [Vec<Option<RefPic>>; 2],
    /// MBAFF: the field lists for field macroblocks, by parity.
    field_lists: Option<Box<[RefLists; 2]>>,
    slice_num: u32,
    poc: i32,
    field_poc: [i32; 2],
}

impl PendingSlice {
    fn decode<T: Pixel>(&self, area: SliceArea<T>) -> Result<(), Error> {
        let ctx = SliceCtx {
            sps: &self.job.sps,
            pps: &self.job.pps,
            ext: &self.ext,
            ls: &self.ls,
            header: &self.job.header,
            slice_num: self.slice_num,
            ref_lists: &self.ref_lists,
            field_lists: self.field_lists.as_deref(),
            poc: self.poc,
            field_poc: self.field_poc,
            mb_field: std::cell::Cell::new(None),
            mc_refs: crate::slice_decoder::McRefs::new(&self.ref_lists),
            mc_field_refs: self
                .field_lists
                .as_deref()
                .map(|f| [0, 1].map(|p| crate::slice_decoder::McRefs::new(&f[p]))),
            implicit: SliceCtx::implicit_table(&self.job.header, &self.job.pps, &self.ref_lists),
        };
        decode_slice(&self.job.rbsp, &ctx, area)
    }
}

/// Decodes the slices of a picture, several at a time with a pool. Each
/// slice owns the macroblocks from its first_mb_in_slice
/// to the next slice's; anything unusual (overlapping or out-of-order
/// slices) is decoded one slice at a time, in arrival order.
fn decode_slices(
    pic: &mut Picture,
    info: &mut [MbInfo],
    slices: &[PendingSlice],
    pool: Option<&Pool>,
    errors: &Errors,
) -> bool {
    let fail = |_: &PendingSlice, e: Error| errors.add(e);
    let total = pic.width_mbs * pic.height_mbs;
    // MBAFF slices start at a macroblock pair.
    let firsts: Vec<usize> = slices
        .iter()
        .map(|s| {
            let mbaff = s.job.sps.mb_adaptive_frame_field && !s.job.header.field_pic;
            s.job.header.first_mb_in_slice as usize * (1 + mbaff as usize)
        })
        .collect();
    if pic.high() {
        decode_slices_t(
            &mut SharedPic::<u16>::new(pic),
            info,
            slices,
            pool,
            &fail,
            total,
            &firsts,
        )
    } else {
        decode_slices_t(
            &mut SharedPic::<u8>::new(pic),
            info,
            slices,
            pool,
            &fail,
            total,
            &firsts,
        )
    }
}

/// decode_slices for one sample type.
fn decode_slices_t<T: Pixel>(
    shared: &mut SharedPic<T>,
    info: &mut [MbInfo],
    slices: &[PendingSlice],
    pool: Option<&Pool>,
    fail: &(dyn Fn(&PendingSlice, Error) + Sync),
    total: usize,
    firsts: &[usize],
) -> bool {
    let shared = &*shared;
    let ordered =
        firsts.windows(2).all(|w| w[0] < w[1]) && firsts.last().is_some_and(|&f| f < total);
    // Slice groups interleave the slices' macroblocks, so they run in turn.
    let fmo = slices.iter().any(|s| s.job.pps.num_slice_groups > 1);
    // In raster order each slice decodes a run from its first macroblock:
    // what it leaves undecoded runs from where it stopped to the next one,
    // marked once all are decoded. Otherwise every macroblock is marked
    // first, as info may be another picture's.
    let runs = ordered && !fmo;
    if !runs {
        unmark(info);
    }
    let mut reached: Vec<usize> = firsts.to_vec();
    // Whether every macroblock was decoded (runs from 0 without gaps), and
    // so none needs concealing.
    let mark_gaps = |info: &mut [MbInfo], reached: &[usize]| {
        if !runs {
            return false;
        }
        unmark(&mut info[..firsts[0]]);
        for (k, &r) in reached.iter().enumerate() {
            let end = firsts.get(k + 1).copied().unwrap_or(total);
            unmark(&mut info[r.min(end)..end]);
        }
        firsts[0] == 0
            && reached
                .iter()
                .enumerate()
                .all(|(k, &r)| r >= firsts.get(k + 1).copied().unwrap_or(total))
    };
    let pool = pool.filter(|_| slices.len() > 1 && runs);
    let Some(pool) = pool else {
        for (k, sl) in slices.iter().enumerate() {
            // Slices in turn: each has the picture to itself. In raster
            // order, each ends where the next begins, as with the pool and
            // frame threads (a damaged slice does not run into the next).
            let end = match firsts.get(k + 1) {
                Some(&next) if ordered && !fmo => next,
                _ => total,
            };
            let r = &mut reached[k];
            let mut done = |addr: usize, _: &[MbInfo]| *r = addr + 1;
            let area = SliceArea {
                pic: shared,
                end,
                info: InfoView::new(&mut *info),
                info_first: 0,
                on_mb: runs.then_some(&mut done as MbDone),
                exclusive: true,
            };
            if let Err(e) = sl.decode(area) {
                fail(sl, e);
            }
        }
        return mark_gaps(info, &reached);
    };
    {
        let mut dones: Vec<_> = reached
            .iter_mut()
            .map(|r| move |addr: usize, _: &[MbInfo]| *r = addr + 1)
            .collect();
        let mut rest = &mut info[firsts[0]..];
        let mut areas = Vec::with_capacity(slices.len());
        for ((k, &first), done) in firsts.iter().enumerate().zip(&mut dones) {
            let end = firsts.get(k + 1).copied().unwrap_or(total);
            let (own, r) = std::mem::take(&mut rest).split_at_mut(end - first);
            rest = r;
            areas.push((
                k,
                SliceArea {
                    pic: shared,
                    info: InfoView::new(own),
                    info_first: first,
                    end,
                    on_mb: Some(done as MbDone),
                    // Slices at once may share rows.
                    exclusive: false,
                },
            ));
        }
        let queue = Mutex::new(areas.into_iter());
        pool.broadcast(|_| {
            loop {
                let Some((k, area)) = queue.lock().unwrap().next() else {
                    break;
                };
                if let Err(e) = slices[k].decode(area) {
                    fail(&slices[k], e);
                }
            }
        });
    }
    mark_gaps(info, &reached)
}

/// Error concealment: macroblocks that were never decoded take the
/// co-located samples of `src`, or mid-grey without one, and are then
/// treated as missing (not deblocked, no co-located motion). Those a
/// failed slice decoded before its error are kept: with frame threading
/// they may be final already, read by other pictures, and every thread
/// count must give the same output.
/// Runs `f` on colour plane `k` of a picture coded with separate colour
/// planes, as a monochrome picture with that plane's macroblocks, or else
/// on the whole picture.
fn with_colour_plane(
    pic: &mut Picture,
    info: &mut [MbInfo],
    separate: bool,
    k: usize,
    f: impl FnOnce(&mut Picture, &mut [MbInfo]),
) {
    if !separate {
        return f(pic, info);
    }
    let n = info.len() / 3;
    let mut mono = Picture {
        width_mbs: pic.width_mbs,
        height_mbs: pic.height_mbs,
        planes: vec![std::mem::replace(&mut pic.planes[k], Plane::new(0, 0))],
        mb_c: (0, 0),
        bit_depth: pic.bit_depth,
    };
    f(&mut mono, &mut info[k * n..(k + 1) * n]);
    pic.planes[k] = mono.planes.pop().unwrap();
}

/// `src_plane` is the first plane of `src` matching the planes of `pic`
/// (the colour plane, with separate colour planes).
fn conceal(
    pic: &mut Picture,
    info: &mut [MbInfo],
    src: Option<&RefPic>,
    mbaff: bool,
    src_plane: usize,
) {
    let geo = Geometry::of(pic);
    if pic.high() {
        let planes: Vec<RawPlane<u16>> = pic.planes.iter_mut().map(RawPlane::new).collect();
        conceal_t(
            &planes,
            &geo,
            &mut InfoView::new(info),
            src,
            mbaff,
            src_plane,
            0,
        );
    } else {
        let planes: Vec<RawPlane<u8>> = pic.planes.iter_mut().map(RawPlane::new).collect();
        conceal_t(
            &planes,
            &geo,
            &mut InfoView::new(info),
            src,
            mbaff,
            src_plane,
            0,
        );
    }
}

/// The shape of a picture's planes.
#[derive(Clone, Copy)]
struct Geometry {
    width_mbs: usize,
    height_mbs: usize,
    mb_c: (usize, usize),
    bit_depth: [u32; 2],
    /// Plane sizes in samples.
    sizes: [(usize, usize); 3],
}

impl Geometry {
    fn of(pic: &Picture) -> Self {
        let mut sizes = [(0, 0); 3];
        for (s, p) in sizes.iter_mut().zip(&pic.planes) {
            *s = (p.width, p.height);
        }
        Self {
            width_mbs: pic.width_mbs,
            height_mbs: pic.height_mbs,
            mb_c: pic.mb_c,
            bit_depth: pic.bit_depth,
            sizes,
        }
    }

    fn grey(&self, plane: usize) -> i32 {
        1 << (self.bit_depth[(plane > 0) as usize] - 1)
    }
}

/// Conceals the macroblocks that were not decoded, from address `from` on
/// (those before it are all decoded).
#[allow(clippy::too_many_arguments)]
fn conceal_t<T: Pixel>(
    planes: &[RawPlane<T>],
    geo: &Geometry,
    info: &mut InfoView,
    src: Option<&RefPic>,
    mbaff: bool,
    src_plane: usize,
    from: usize,
) {
    let w = geo.width_mbs;
    let mut waited = false;
    for addr in from..info.len() {
        let m = info.get_mut(addr);
        if m.slice != u32::MAX {
            continue;
        }
        // The source is read once something needs concealing.
        if let Some(r) = src
            && !std::mem::replace(&mut waited, true)
        {
            r.state().wait_done();
        }
        // Never decoded: may hold another picture's (see new_info).
        *m = MbInfo::default();
        // Frame macroblock geometry also covers missing MBAFF pairs.
        let (mx, my) = if mbaff {
            (addr / 2 % w, addr / 2 / w * 2 + addr % 2)
        } else {
            (addr % w, addr / w)
        };
        for (i, pl) in planes.iter().enumerate() {
            let (mw, mh) = if i == 0 { (16, 16) } else { geo.mb_c };
            let (x0, y0) = (mx * mw, my * mh);
            let from = src
                .map(|r| r.plane(i + src_plane))
                .filter(|v| (v.width, v.height) == geo.sizes[i]);
            let grey = T::new(geo.grey(i));
            let mut row = [grey; 16];
            for y in y0..y0 + mh {
                if let Some(v) = &from {
                    row[..mw].copy_from_slice(v.samples::<T>(y * v.stride + x0, mw));
                }
                pl.write(y * pl.stride + x0, &row[..mw]);
            }
        }
    }
}

/// Marks every macroblock as not decoded (in this picture).
fn unmark(info: &mut [MbInfo]) {
    for m in info {
        m.slice = u32::MAX;
    }
}

/// The planes of a picture being decoded on a frame thread, written
/// through raw pointers taken before the picture was shared.
enum RawPlanes {
    U8(Vec<RawPlane<u8>>),
    U16(Vec<RawPlane<u16>>),
}

impl RawPlanes {
    fn new(pic: &mut Picture) -> Self {
        if pic.high() {
            Self::U16(pic.planes.iter_mut().map(RawPlane::new).collect())
        } else {
            Self::U8(pic.planes.iter_mut().map(RawPlane::new).collect())
        }
    }
}

/// A frame picture decoded on its own thread while the pictures before and
/// after it are decoded on others (frame threading): its slices in order,
/// each row of macroblocks deblocked as soon as the row below it is
/// decoded, and the rows that no later deblocking changes published for
/// the pictures referencing this one.
struct FrameJob {
    /// Keeps the planes alive.
    _pic: Arc<Picture>,
    planes: RawPlanes,
    geo: Geometry,
    state: Arc<PicState>,
    slices: Vec<PendingSlice>,
    filters: Vec<SliceFilter>,
    chroma_array_type: u8,
    mbaff: bool,
    conceal: Option<RefPic>,
    errors: Arc<Errors>,
}

/// Finishes a picture's state however its job ends, so no reader waits
/// forever (a panicking job leaves no macroblocks).
struct FinishGuard(Arc<PicState>);

impl Drop for FinishGuard {
    fn drop(&mut self) {
        if !self.0.is_done() {
            self.0.finish();
        }
    }
}

impl FrameJob {
    fn run(mut self) {
        let _guard = FinishGuard(self.state.clone());
        match std::mem::replace(&mut self.planes, RawPlanes::U8(Vec::new())) {
            RawPlanes::U8(p) => self.run_t(&p),
            RawPlanes::U16(p) => self.run_t(&p),
        }
    }

    fn run_t<T: Pixel>(self, planes: &[RawPlane<T>]) {
        let g = self.geo;
        let (w, h) = (g.width_mbs, g.height_mbs);
        let shared = SharedPic::from_raw(planes, w, h, g.mb_c, g.bit_depth);
        // SAFETY: this job alone writes the picture's macroblocks, those of
        // rows it has not published (publish_info_row) before finishing.
        let (ptr, len) = unsafe { self.state.info_ptr() };
        let mut info = unsafe { InfoView::from_raw(ptr, len) };
        let (filters, state, cat) = (&self.filters, &self.state, self.chroma_array_type);
        // Rows are deblocked as they complete (not MBAFF: there, at the end).
        let incremental = !self.mbaff;
        let per_row = if self.mbaff { 2 * w } else { w };
        let mut seen = vec![false; w * h];
        let mut count = vec![0usize; h];
        let mut deblocked = 0;
        // A row is deblocked as soon as it is decoded, but the next row's
        // intra prediction reads its bottom samples unfiltered: those stay
        // in the picture, the filtered ones aside until the next row is
        // decoded and deblocked in turn (`aside`).
        let size = |i: usize| if i == 0 { (16, 16) } else { g.mb_c };
        let bottom = |i: usize, r: usize| ((r + 1) * size(i).1 - 1) * planes[i].stride;
        let mut keep: Vec<[Vec<T>; 2]> = (0..planes.len())
            .map(|i| {
                [
                    vec![T::default(); w * size(i).0],
                    vec![T::default(); w * size(i).0],
                ]
            })
            .collect();
        let mut aside = false;
        let restore = |keep: &[[Vec<T>; 2]], r: usize| {
            for (i, k) in keep.iter().enumerate() {
                planes[i].write(bottom(i, r), &k[1]);
            }
        };
        {
            let mut row_done = |addr: usize, inf: &[MbInfo]| {
                if std::mem::replace(&mut seen[addr], true) {
                    return;
                }
                let r = addr / per_row;
                count[r] += 1;
                if count[r] == per_row {
                    state.publish_info_row(r);
                }
                if !incremental {
                    return;
                }
                while deblocked < h && count[deblocked] == w {
                    let r = deblocked;
                    if std::mem::take(&mut aside) {
                        restore(&keep, r - 1);
                    }
                    for (i, k) in keep.iter_mut().enumerate() {
                        planes[i].read(bottom(i, r), &mut k[0]);
                    }
                    deblock_rows(planes, w, inf, filters, cat, false, g.bit_depth, r..r + 1);
                    deblocked += 1;
                    if deblocked < h {
                        for (i, k) in keep.iter_mut().enumerate() {
                            planes[i].read(bottom(i, r), &mut k[1]);
                            planes[i].write(bottom(i, r), &k[0]);
                        }
                        aside = true;
                        // Deblocking the next row changes up to 3 rows above
                        // it (chroma: 1, or 3 in 4:4:4).
                        state.publish_rows(16 * deblocked - 3);
                    }
                }
            };
            for (k, sl) in self.slices.iter().enumerate() {
                // A slice ends where the next begins (they are in raster
                // order): a damaged one must not run on into the next one's
                // macroblocks, rewriting rows other pictures may have read.
                let end = self.slices.get(k + 1).map_or(w * h, |n| {
                    n.job.header.first_mb_in_slice as usize * (1 + self.mbaff as usize)
                });
                let area = SliceArea {
                    pic: &shared,
                    info: info.reborrow(),
                    info_first: 0,
                    end,
                    on_mb: Some(&mut row_done),
                    // Slices in turn; other threads only read final rows,
                    // above the macroblock row being decoded.
                    exclusive: true,
                };
                if let Err(e) = sl.decode(area) {
                    self.errors.add(e);
                }
            }
        }
        if aside {
            restore(&keep, deblocked - 1);
        }
        // The info may be another picture's where nothing was decoded.
        for (addr, _) in seen.iter().enumerate().filter(|(_, s)| !**s) {
            info.get_mut(addr).slice = u32::MAX;
        }
        // What is still missing is concealed (rows already final are whole).
        let src = self.conceal.as_ref();
        conceal_t(planes, &g, &mut info, src, self.mbaff, 0, deblocked * w);
        let info = info.as_slice();
        if self.mbaff {
            // SAFETY: no row is published before the picture is done, so
            // nothing reads the planes.
            unsafe { deblock_mbaff_t(planes, w, h, g.bit_depth, info, filters, cat) };
        } else {
            deblock_rows(
                planes,
                w,
                info,
                filters,
                cat,
                false,
                g.bit_depth,
                deblocked..h,
            );
        }
        self.state.finish();
    }
}

/// Adds `x` to `list` of things to reuse once nothing else holds them,
/// keeping the list bounded.
fn keep_free<T>(list: &mut Vec<Arc<T>>, x: &Arc<T>) {
    const KEEP: usize = 40;
    if list.len() >= KEEP {
        // The oldest is likeliest free; drop one still in use otherwise.
        let i = list
            .iter()
            .position(|p| Arc::strong_count(p) == 1)
            .unwrap_or(0);
        list.remove(i);
    }
    list.push(x.clone());
}

/// The place of a view's frame in output order. Every view outputs its
/// pictures in PicOrderCnt order within an output period (C.4.5.3), and
/// the view components of an access unit have the same PicOrderCnt (G.8.1),
/// so the frames of an access unit share it; the access unit, in decoding
/// order, tells apart pictures with the same order count.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct OutputOrder {
    period: u64,
    poc: i32,
    au: u64,
}

/// A frame a view decoder has output.
struct ViewFrame {
    voidx: usize,
    order: OutputOrder,
    /// The access unit decoded when the DPB output it.
    out_au: u64,
    frame: Frame,
}

/// Decodes the pictures of one view (one view order index).
struct ViewDecoder {
    voidx: usize,
    dpb: Dpb,
    poc: PocState,
    prev_ref_frame_num: u32,
    /// Decoded pictures whose buffers new pictures reuse when free.
    recycle: Vec<Arc<Picture>>,
    /// Likewise for their macroblock info.
    recycle_info: Vec<Arc<PicState>>,
    /// The last frame decoded (with its progress), for concealment.
    last_pic: Option<(Arc<Picture>, Arc<PicState>)>,
    cur: Option<CurrentPicture>,
    /// Inter-view references for the current picture: (view_id, picture)
    /// of the preceding views of the access unit.
    inter_view: Vec<(u16, RefPic)>,
    next_id: Arc<AtomicU32>,
    errors: Arc<Errors>,
    /// Output frames.
    frames: Sender<ViewFrame>,
    /// The access unit of the last picture decoded.
    last_au: u64,
    /// (access unit, output period) of the pictures not yet output.
    periods: Vec<(u64, u64)>,
    /// Offset metadata of the GOP being decoded.
    gop: Option<GopMeta>,
    /// (GOP, display index) of the last output frame.
    out_pos: (u64, usize),
    /// Decoding started at a non-IDR picture with this POC: pictures before
    /// it in output order (leading pictures whose references precede the
    /// start) are not output, as JM and other decoders do.
    skip_before: Option<i32>,
    started: bool,
    /// Threads for decoding the slices and deblocking rows of a picture in
    /// parallel; None for one.
    pool: Option<Pool>,
    /// Frame threading: threads decoding several pictures at once, shared
    /// by the views.
    jobs: Option<Arc<JobQueue>>,
    /// The level scales of the last slice's scaling matrices.
    level_scale: Option<(crate::ps::ScalingMatrices, Arc<LevelScale>)>,
}

impl ViewDecoder {
    fn handle(&mut self, msg: Msg) {
        match msg {
            Msg::Slice(job) => self.slice(*job),
            Msg::Finish => self.finish_picture(),
            Msg::Sync(done) => {
                let _ = done.send(());
            }
            Msg::Flush => {
                self.finish_picture();
                self.dpb.flush();
                self.drain();
            }
            Msg::Discard => {
                // A picture still being received is finished (frame jobs
                // waiting on it must end) but not output; nor is anything
                // its storing would bump.
                self.dpb.discard();
                self.periods.clear();
                self.skip_before = Some(i32::MAX);
                self.finish_picture();
                self.dpb.discard();
                self.poc = PocState::default();
                self.prev_ref_frame_num = 0;
                self.last_pic = None;
                self.inter_view.clear();
                self.gop = None;
                self.out_pos = (0, 0);
                self.skip_before = None;
                self.started = false;
            }
            Msg::Metadata(m) => {
                // The metadata arrives with the GOP's first access unit, after
                // its base picture's slices: tag that picture and what follows.
                let gop = self.gop.as_ref().map_or(1, |g| g.gop + 1);
                self.gop = Some(GopMeta { gop, data: m });
                if let Some(c) = &mut self.cur {
                    c.meta = self.gop.clone();
                }
            }
        }
    }

    fn slice(&mut self, mut job: SliceJob) {
        if job.new_picture || self.cur.is_none() {
            self.finish_picture();
            self.start_picture(&job);
            self.cur.as_mut().unwrap().publish = job.publish.take();
        }
        let (sps, pps) = (&*job.sps, &*job.pps);
        let ext = match pps.ext(sps) {
            Ok(e) => e,
            Err(e) => return self.errors.add(e),
        };
        // The same scaling matrices (the usual case) share their scales.
        let ls = match &self.level_scale {
            Some((m, ls)) if *m == ext.scaling => ls.clone(),
            _ => {
                let ls: Arc<LevelScale> = LevelScale::new(&ext.scaling).into();
                self.level_scale = Some((ext.scaling.clone(), ls.clone()));
                ls
            }
        };
        let anchor = job.nal.mvc.is_some_and(|m| m.anchor_pic);
        let ref_lists = self.ref_lists(sps, job.subset.as_deref(), &job.header, anchor);
        let cur = self.cur.as_mut().unwrap();
        let h = &job.header;
        cur.filters.push(SliceFilter {
            disable_idc: h.disable_deblocking_filter_idc,
            offset_a: (h.slice_alpha_c0_offset_div2 as i32) << 1,
            offset_b: (h.slice_beta_offset_div2 as i32) << 1,
            switching: matches!(h.slice_type, SliceType::Sp | SliceType::Si),
            chroma_qp_offset: [
                pps.chroma_qp_index_offset,
                ext.second_chroma_qp_index_offset,
            ],
        });
        // 8.4.2.1: for field macroblocks of MBAFF frames, entry 2i (2i + 1)
        // of a list is the field of frame i with the same (opposite) parity.
        let field_lists = (sps.mb_adaptive_frame_field && !h.field_pic).then(|| {
            Box::new([Parity::Top, Parity::Bottom].map(|p| {
                ref_lists.clone().map(|l| {
                    l.iter()
                        .flat_map(|r| [p, p.other()].map(|f| r.as_ref().map(|r| r.field(f))))
                        .collect()
                })
            }))
        });
        let sl = PendingSlice {
            job,
            ext,
            ls,
            ref_lists,
            field_lists,
            slice_num: cur.slices,
            poc: cur.poc,
            field_poc: [cur.field_poc.top, cur.field_poc.bottom],
        };
        cur.slices += 1;
        if cur.conceal.is_none() {
            cur.conceal = sl.ref_lists.iter().flatten().flatten().next().cloned();
        }
        // Decoded once the picture is complete, when each slice's extent
        // is known, the same way whatever the threads.
        cur.pending.push(sl);
    }

    /// A picture to decode into: a decoded one nothing refers to any more,
    /// or a new one. Its samples are all written (or concealed) before use.
    fn new_picture(&mut self, w: usize, h: usize, cat: u8, bit_depth: [u32; 2]) -> Picture {
        let fits = |p: &Picture| {
            (p.width_mbs, p.height_mbs, p.bit_depth) == (w, h, bit_depth)
                && p.planes.len() == Picture::new(0, 0, cat).planes.len()
                && p.mb_c == Picture::new(0, 0, cat).mb_c
        };
        // Only this list holds it: nothing can share it again.
        if let Some(i) = self
            .recycle
            .iter()
            .position(|p| Arc::strong_count(p) == 1 && fits(p))
            && let Ok(p) = Arc::try_unwrap(self.recycle.swap_remove(i))
        {
            return p;
        }
        Picture::with_depth(w, h, cat, bit_depth)
    }

    /// Remembers a decoded picture so its buffers can be reused once the
    /// DPB and the frames handed out let go of it.
    fn keep(&mut self, pic: &Arc<Picture>, state: &Arc<PicState>) {
        keep_free(&mut self.recycle, pic);
        keep_free(&mut self.recycle_info, state);
    }

    /// Macroblock info for `n` macroblocks, none of them decoded: a decoded
    /// picture's that nothing refers to any more, or new. Only `slice` is
    /// reset; a macroblock's other fields are written when it is decoded,
    /// or reset when it is concealed.
    /// Macroblock info for a picture of `n` macroblocks: possibly another
    /// picture's, reused (see the comment below).
    fn new_info(&mut self, n: usize) -> Vec<MbInfo> {
        if let Some(i) = self
            .recycle_info
            .iter()
            .position(|s| Arc::strong_count(s) == 1 && s.len() == n)
            && let Some(info) = Arc::try_unwrap(self.recycle_info.swap_remove(i))
                .ok()
                .and_then(PicState::into_info)
                .filter(|v| v.len() == n)
        {
            // Decoding marks what its slices leave undecoded (see
            // decode_slices_t and FrameJob).
            return info;
        }
        vec![MbInfo::default(); n]
    }

    fn start_picture(&mut self, job: &SliceJob) {
        let (sps, header, ref_idc) = (&*job.sps, &job.header, job.nal.ref_idc);
        let max_frame_num = 1u32 << sps.log2_max_frame_num;
        // A stream may start at a non-IDR picture (one cut from a longer
        // one): the DPB then takes its size from the first SPS too, not
        // the 16 frames it starts with, which would hold back output and
        // keep that many pictures.
        if header.idr || !self.started {
            self.dpb.size = dpb_size(sps);
        }
        if !header.idr
            && header.frame_num != self.prev_ref_frame_num
            && header.frame_num != (self.prev_ref_frame_num + 1) % max_frame_num
        {
            // 8.2.5.2 (also used as concealment when gaps are not allowed).
            let (w, h) = (
                sps.pic_width_in_mbs as usize,
                sps.frame_height_in_mbs() as usize,
            );
            // Non-existing frames share the samples of the last frame (only
            // read by streams in error), once it is decoded.
            let template = self
                .last_pic
                .clone()
                .map(|(p, state)| {
                    state.wait_done();
                    p
                })
                .filter(|p| p.height_mbs == h)
                .unwrap_or_else(|| {
                    Arc::new(Picture::with_depth(
                        w,
                        h,
                        sps.picture_chroma_type(),
                        sps.bit_depths(),
                    ))
                });
            // Reserve one picture id per inferred frame.
            let gaps =
                (header.frame_num + max_frame_num - self.prev_ref_frame_num - 1) % max_frame_num;
            let mut id = self.next_id.fetch_add(gaps + 1, Ordering::Relaxed);
            self.dpb.fill_gaps(
                self.prev_ref_frame_num,
                header.frame_num,
                max_frame_num,
                sps.max_num_ref_frames,
                &template,
                &mut id,
            );
            self.prev_ref_frame_num = (header.frame_num + max_frame_num - 1) % max_frame_num;
        }
        self.inter_view = job
            .earlier_views
            .iter()
            .filter_map(|(v, slot)| Some((*v, slot.wait()?)))
            .collect();
        let field_poc = self.poc.next(sps, header, ref_idc);
        let poc = field_poc.pic(structure(header));
        let pic_structure = match (header.field_pic, header.bottom_field) {
            (false, _) => PicStructure::Frame,
            (true, false) => PicStructure::Field(Parity::Top),
            (true, true) => PicStructure::Field(Parity::Bottom),
        };
        if header.idr {
            self.skip_before = None;
        } else if !self.started {
            self.skip_before = Some(poc);
        }
        self.started = true;
        let (w, h) = (
            sps.pic_width_in_mbs as usize,
            sps.frame_height_in_mbs() as usize / (1 + header.field_pic as usize),
        );
        self.cur = Some(CurrentPicture {
            pic: self.new_picture(w, h, sps.picture_chroma_type(), sps.bit_depths()),
            info: self.new_info(w * h * sps.colour_planes()),
            sps: job.sps.clone(),
            first: header.clone(),
            ref_idc,
            inter_view_ref: job.nal.mvc.is_none_or(|m| m.inter_view),
            publish: None,
            poc,
            field_poc,
            structure: pic_structure,
            slices: 0,
            filters: Vec::new(),
            out: OutputInfo {
                pts: job.pts,
                au: job.au,
                ..output_info(sps, job.view_id)
            },
            period: job.period,
            meta: self.gop.clone(),
            pending: Vec::new(),
            conceal: None,
        });
    }

    fn ref_lists(
        &self,
        sps: &Sps,
        subset: Option<&SubsetSps>,
        header: &SliceHeader,
        anchor: bool,
    ) -> [Vec<Option<RefPic>>; 2] {
        let max_frame_num = 1u32 << sps.log2_max_frame_num;
        let n = [
            header.num_ref_idx_active[0] as usize,
            header.num_ref_idx_active[1] as usize,
        ];
        let dpb = &self.dpb;
        // Inter-view prediction references per list (G.8.2.1): the view
        // components of the same access unit named by the subset SPS.
        let inter_view = |list: usize| -> Vec<RefPic> {
            let Some(sub) = subset else { return Vec::new() };
            let Some(refs) = sub.mvc.refs.get(self.voidx) else {
                return Vec::new();
            };
            let ids = match (anchor, list) {
                (true, 0) => &refs.anchor_l0,
                (true, _) => &refs.anchor_l1,
                (false, 0) => &refs.non_anchor_l0,
                (false, _) => &refs.non_anchor_l1,
            };
            ids.iter()
                .filter_map(|&v| {
                    self.inter_view
                        .iter()
                        .find(|(id, _)| *id == v)
                        .map(|(_, p)| p.clone())
                })
                .collect()
        };
        // An IDR view component of a non-base view starts from an empty list (G.8.2).
        let cur = self
            .cur
            .as_ref()
            .map_or(PicStructure::Frame, |c| c.structure);
        let temporal = |init: Vec<Entry>| {
            if header.idr && self.voidx > 0 {
                None
            } else {
                Some(init)
            }
        };
        match header.slice_type {
            SliceType::P | SliceType::Sp => {
                let init = dpb.init_list_p(cur, header.frame_num, max_frame_num);
                [
                    dpb.build_list(
                        temporal(init),
                        &inter_view(0),
                        &header.ref_pic_list_modification[0],
                        n[0],
                        cur,
                        header.frame_num,
                        max_frame_num,
                    ),
                    Vec::new(),
                ]
            }
            SliceType::B => {
                let poc = self.cur.as_ref().map_or(0, |c| c.poc);
                let [l0, l1] = dpb.init_lists_b(cur, poc);
                [
                    dpb.build_list(
                        temporal(l0),
                        &inter_view(0),
                        &header.ref_pic_list_modification[0],
                        n[0],
                        cur,
                        header.frame_num,
                        max_frame_num,
                    ),
                    dpb.build_list(
                        temporal(l1),
                        &inter_view(1),
                        &header.ref_pic_list_modification[1],
                        n[1],
                        cur,
                        header.frame_num,
                        max_frame_num,
                    ),
                ]
            }
            _ => [Vec::new(), Vec::new()],
        }
    }

    fn finish_picture(&mut self) {
        let Some(mut c) = self.cur.take() else { return };
        // Decided now: skip_before may be cleared by a later IDR picture
        // while this one still waits in the DPB.
        c.out.hidden = self.skip_before.is_some_and(|t| c.poc < t);
        // Intra pictures conceal from the previous frame.
        let src = c.conceal.take().or_else(|| {
            self.last_pic
                .as_ref()
                .filter(|_| c.structure == PicStructure::Frame)
                .map(|(p, state)| RefPic::frame(p.clone(), state.clone(), 0, 0))
        });
        let mbaff = c.sps.mb_adaptive_frame_field && c.structure == PicStructure::Frame;
        let cat = c.sps.chroma_array_type();
        // Frame threading takes frames whose slices come in raster order
        // (field pairs, slice groups and separate colour planes are decoded
        // here, after the pictures they reference).
        let firsts: Vec<u32> = c
            .pending
            .iter()
            .map(|s| s.job.header.first_mb_in_slice)
            .collect();
        let raster = firsts.windows(2).all(|w| w[0] < w[1])
            && c.pending.iter().all(|s| s.job.pps.num_slice_groups == 1);
        let threaded = self.jobs.is_some()
            && c.structure == PicStructure::Frame
            && !c.sps.separate_colour_plane
            && raster;
        let (pic, info) = if threaded {
            let planes = RawPlanes::new(&mut c.pic);
            let geo = Geometry::of(&c.pic);
            // Info rows: rows of macroblocks, or of pairs in MBAFF frames.
            let per_row = c.pic.width_mbs * (1 + mbaff as usize);
            // Moving the picture keeps its sample buffers where they are.
            let pic = Arc::new(c.pic);
            let state = PicState::pending(c.info.len() / per_row, per_row, c.info);
            self.keep(&pic, &state);
            let job = FrameJob {
                _pic: pic.clone(),
                planes,
                geo,
                state: state.clone(),
                slices: std::mem::take(&mut c.pending),
                filters: std::mem::take(&mut c.filters),
                chroma_array_type: cat,
                mbaff,
                conceal: src,
                errors: self.errors.clone(),
            };
            self.jobs.as_ref().unwrap().submit(move || job.run());
            (pic, state)
        } else {
            self.decode_here(&mut c, src.as_ref(), mbaff, cat);
            let pic = Arc::new(c.pic);
            let state = PicState::done(c.info);
            self.keep(&pic, &state);
            (pic, state)
        };
        let max_frame_num = 1u32 << c.sps.log2_max_frame_num;
        let mmco5 = matches!(&c.first.dec_ref_pic_marking, Some(DecRefPicMarking::Adaptive(ops)) if ops.contains(&Mmco::ForgetAll));
        let id = self.next_id.fetch_add(1, Ordering::Relaxed) + 1;
        if let Some(p) = c.publish.take() {
            // A view component with inter_view_flag 0 is not used for
            // inter-view prediction (G.7.4.1.1); publishing None tells the
            // following views not to wait for it.
            p.set(c.inter_view_ref.then(|| RefPic {
                inter_view: true,
                ..RefPic::frame(pic.clone(), info.clone(), id, c.poc)
            }));
        }
        if c.structure == PicStructure::Frame {
            self.last_pic = Some((pic.clone(), info.clone()));
        }
        // After mmco 5, tempPicOrderCnt is subtracted (8.2.1).
        let mut poc = [c.field_poc.top, c.field_poc.bottom];
        if mmco5 {
            for v in &mut poc {
                *v -= c.poc;
            }
        }
        let entry = Decoded {
            pic,
            info,
            structure: c.structure,
            mbaff: c.sps.mb_adaptive_frame_field && c.structure == PicStructure::Frame,
            id,
            frame_num: c.first.frame_num,
            poc,
            out: c.out,
            meta: c.meta,
        };
        if c.ref_idc != 0 {
            self.prev_ref_frame_num = if mmco5 { 0 } else { c.first.frame_num };
        }
        self.last_au = c.out.au;
        self.periods.push((c.out.au, c.period));
        self.dpb.store(
            entry,
            c.ref_idc,
            c.first.idr,
            c.first.dec_ref_pic_marking.as_ref(),
            max_frame_num,
            c.sps.max_num_ref_frames,
        );
        self.drain();
    }

    /// Decodes the current picture on this thread (with the slice pool).
    fn decode_here(&self, c: &mut CurrentPicture, src: Option<&RefPic>, mbaff: bool, cat: u8) {
        // Slices by colour plane (one group without separate colour planes).
        let sep = c.sps.separate_colour_plane;
        let mut groups: [Vec<PendingSlice>; 3] = Default::default();
        for sl in std::mem::take(&mut c.pending) {
            groups[sl.job.header.colour_plane_id as usize].push(sl);
        }
        // Per colour plane, whether all macroblocks were decoded.
        let mut complete = [false; 3];
        for (k, slices) in groups.iter().enumerate() {
            if !slices.is_empty() || sep {
                with_colour_plane(&mut c.pic, &mut c.info, sep, k, |pic, info| {
                    let pool = self.pool.as_ref();
                    complete[k] = decode_slices(pic, info, slices, pool, &self.errors);
                });
            }
        }
        drop(groups);
        for (k, &complete) in complete.iter().enumerate().take(c.sps.colour_planes()) {
            let (filters, field) = (&c.filters, c.structure != PicStructure::Frame);
            with_colour_plane(&mut c.pic, &mut c.info, sep, k, |pic, info| {
                if !complete {
                    conceal(pic, info, src, mbaff, k);
                }
                if mbaff {
                    deblock_mbaff(pic, info, filters, cat);
                } else {
                    deblock_picture(pic, info, filters, cat, field, self.pool.as_ref());
                }
            });
        }
    }

    fn drain(&mut self) {
        while let Some(o) = self.dpb.output.pop_front() {
            // Frames of a GOP are output contiguously in display order, so
            // the display index restarts at each new GOP.
            let offsets = o.meta.as_ref().map(|m| {
                if self.out_pos.0 != m.gop {
                    self.out_pos = (m.gop, 0);
                } else {
                    self.out_pos.1 += 1;
                }
                let i = self.out_pos.1;
                m.data
                    .offsets
                    .iter()
                    .map(|seq| {
                        seq.get(i).or(seq.last()).copied().unwrap_or(PlaneOffset {
                            direction: false,
                            value: 0,
                        })
                    })
                    .collect()
            });
            if o.out.hidden {
                continue;
            }
            let order = OutputOrder {
                period: self
                    .periods
                    .iter()
                    .find(|p| p.0 == o.out.au)
                    .map_or(0, |p| p.1),
                poc: o.poc,
                au: o.out.au,
            };
            let _ = self.frames.send(ViewFrame {
                voidx: self.voidx,
                order,
                out_au: self.last_au,
                frame: to_frame(o, self.voidx, offsets),
            });
        }
        // Pictures output, or no longer stored, are done with.
        let dpb = &self.dpb;
        self.periods.retain(|&(au, _)| dpb.holds(au));
    }
}

enum Worker {
    Inline(Box<ViewDecoder>),
    Thread {
        tx: SyncSender<Msg>,
        handle: Option<JoinHandle<()>>,
    },
}

impl Worker {
    fn send(&mut self, msg: Msg) {
        match self {
            Worker::Inline(v) => v.handle(msg),
            Worker::Thread { tx, .. } => {
                let _ = tx.send(msg);
            }
        }
    }
}

impl Drop for Decoder {
    // Stops the view threads and waits for them, so their threads and
    // pictures are released when the decoder is.
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        for w in std::mem::take(&mut self.workers) {
            if let Worker::Thread { tx, mut handle } = w {
                drop(tx);
                if let Some(h) = handle.take() {
                    let _ = h.join();
                }
            }
        }
    }
}

/// Access units a view's frame waits for the other views before it is
/// output anyway (see `next_frame`).
const MAX_PENDING: u64 = 32;

pub struct Decoder {
    sps: Vec<Option<Arc<Sps>>>,
    subset: Vec<Option<Arc<SubsetSps>>>,
    pps: Vec<Option<Arc<Pps>>>,
    /// Previous slice per view (header, nal_ref_idc), for picture boundaries.
    last: Vec<Option<(SliceHeader, u8)>>,
    /// View order index of the picture in progress.
    cur_view: Option<usize>,
    /// MVC header of the last prefix NAL unit, for the next base slice.
    prefix: Option<MvcExtension>,
    au: u64,
    workers: Vec<Worker>,
    // Set on drop: view threads stop instead of decoding what is queued.
    stop: Arc<AtomicBool>,
    #[cfg(test)]
    live_views: Arc<std::sync::atomic::AtomicUsize>,
    /// The pictures started so far in the current access unit.
    au_views: Vec<(u16, Arc<PicSlot>)>,
    next_id: Arc<AtomicU32>,
    errors: Arc<Errors>,
    frames_tx: Sender<ViewFrame>,
    frames_rx: Receiver<ViewFrame>,
    /// Per view, its frames not yet output, in its output order.
    pending: Vec<VecDeque<ViewFrame>>,
    /// The access unit / MAX_PENDING at the last check for frames waiting
    /// too long (see `next_frame`).
    checkpoint: u64,
    /// Per view, the last access unit with a picture of it.
    pic_au: Vec<u64>,
    /// Output period of the current access unit (SliceJob::period), and
    /// the access unit that started it.
    period: u64,
    period_au: u64,
    /// Frames in output order, decided but not yet returned: one view's,
    /// or (stereo_layout) an access unit's base and non-base view to pack.
    ready: VecDeque<[Option<Frame>; 2]>,
    flushed: bool,
    threads: usize,
    rbsp: Vec<u8>,
    rbsp_pool: RbspPool,
    /// Decode non-base views (MVC). Off: base view only, as a 2D decoder.
    pub decode_all_views: bool,
    /// With several views: output each access unit's base and first
    /// non-base view as one frame in this layout, instead of one frame per
    /// view.
    pub stereo_layout: Option<crate::stereo::StereoLayout>,
    /// The base view is the right eye (Blu-ray: from the playlist, carried
    /// by containers such as Matroska's StereoMode); it is the left one by
    /// default. Decides the order of the views in stereo_layout.
    pub base_view_right: bool,
    /// With threads > 1: decode several pictures at once (frame threading,
    /// the default), which delays output by a few pictures; off, threads
    /// only work on the slices and deblocking of one picture at a time.
    /// Set before decoding starts.
    pub frame_threading: bool,
    jobs: Option<Option<Arc<JobQueue>>>,
    pts: Option<i64>,
    /// The timestamp of the current access unit: the one set when its base
    /// view picture started.
    au_pts: Option<i64>,
}

impl Default for Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Decoder {
    /// A decoder that runs on the calling thread.
    pub fn new() -> Self {
        Self::with_threads(0)
    }

    /// A decoder using one thread per CPU (see [`Self::with_threads`]).
    pub fn with_auto_threads() -> Self {
        Self::with_threads(std::thread::available_parallelism().map_or(1, |n| n.get()))
    }

    /// A decoder with `threads` threads: 0 decodes on the calling thread;
    /// 1 on a worker thread per view; more decode several pictures at once
    /// (frame threading), or the slices of one picture with
    /// `frame_threading` off. Frame threading keeps half as many threads
    /// again to run while others wait on references, `threads` running at
    /// a time. On WebAssembly without thread support (wasm32-wasip1, or
    /// wasm32 built without `atomics`), every count decodes on the calling
    /// thread, as 0 does.
    pub fn with_threads(threads: usize) -> Self {
        let threads = if cfg!(target_family = "wasm") && !wasm_threads() {
            0
        } else {
            threads
        };
        let (frames_tx, frames_rx) = channel();
        Self {
            sps: vec![None; 32],
            subset: vec![None; 32],
            pps: vec![None; 256],
            last: Vec::new(),
            cur_view: None,
            prefix: None,
            au: 0,
            workers: Vec::new(),
            stop: Arc::new(AtomicBool::new(false)),
            #[cfg(test)]
            live_views: Arc::default(),
            au_views: Vec::new(),
            next_id: Arc::new(AtomicU32::new(0)),
            errors: Arc::new(Errors::default()),
            frames_tx,
            frames_rx,
            pending: Vec::new(),
            checkpoint: 0,
            pic_au: Vec::new(),
            period: 0,
            period_au: 0,
            ready: VecDeque::new(),
            flushed: false,
            threads,
            rbsp: Vec::new(),
            rbsp_pool: RbspPool::default(),
            decode_all_views: true,
            stereo_layout: None,
            base_view_right: false,
            frame_threading: true,
            jobs: None,
            pts: None,
            au_pts: None,
        }
    }

    /// The frame threads, made on first use.
    fn jobs(&mut self) -> &Option<Arc<JobQueue>> {
        let (threads, on) = (self.threads, self.frame_threading);
        self.jobs.get_or_insert_with(|| {
            // Workers to spare for those waiting on references, and one
            // picture more than workers, so a worker finds the next ready.
            let workers = threads + threads / 2;
            (threads > 1 && on).then(|| Arc::new(JobQueue::new(workers, threads, workers + 1)))
        })
    }

    /// Number of slices that failed to decode (their pictures are still output).
    pub fn errors(&self) -> u64 {
        self.errors.count.load(Ordering::Relaxed)
    }

    pub fn last_error(&self) -> Option<Error> {
        *self.errors.last.lock().unwrap()
    }

    fn worker(&mut self, voidx: usize) -> &mut Worker {
        while self.workers.len() <= voidx {
            let vd = ViewDecoder {
                voidx: self.workers.len(),
                dpb: Dpb::new(),
                poc: PocState::default(),
                prev_ref_frame_num: 0,
                last_pic: None,
                recycle: Vec::new(),
                recycle_info: Vec::new(),
                level_scale: None,
                cur: None,
                inter_view: Vec::new(),
                next_id: self.next_id.clone(),
                errors: self.errors.clone(),
                frames: self.frames_tx.clone(),
                gop: None,
                out_pos: (0, 0),
                last_au: 0,
                periods: Vec::new(),
                skip_before: None,
                started: false,
                pool: (self.threads > 1).then(|| Pool::new(self.threads - 1)),
                jobs: self.jobs().clone(),
            };
            let w = if self.threads == 0 {
                Worker::Inline(Box::new(vd))
            } else {
                let (tx, rx) = sync_channel::<Msg>(64);
                let mut vd = vd;
                let stop = self.stop.clone();
                #[cfg(test)]
                let live = self.live_views.clone();
                #[cfg(test)]
                live.fetch_add(1, Ordering::SeqCst);
                let handle = std::thread::Builder::new()
                    .name(format!("altc264-view{}", self.workers.len()))
                    .spawn(move || {
                        #[cfg(test)]
                        let _live = LiveView(live);
                        while let Ok(m) = rx.recv() {
                            if stop.load(Ordering::Relaxed) {
                                break;
                            }
                            vd.handle(m);
                        }
                    })
                    .expect("spawn decoder thread");
                Worker::Thread {
                    tx,
                    handle: Some(handle),
                }
            };
            self.workers.push(w);
        }
        &mut self.workers[voidx]
    }

    /// Sets the timestamp attached to access units that start from now on
    /// (for example the PTS of the next container packet): every view of an
    /// access unit has the timestamp set when its base view picture started.
    pub fn set_timestamp(&mut self, pts: Option<i64>) {
        self.pts = pts;
    }

    /// Decodes length-prefixed NAL units (ISO/IEC 14496-15 "AVCC" samples
    /// as in MP4 and Matroska), with `length_size` bytes per length field.
    pub fn decode_length_prefixed(&mut self, mut data: &[u8], length_size: usize) {
        while data.len() > length_size {
            let n = data[..length_size]
                .iter()
                .fold(0usize, |v, &b| v << 8 | b as usize);
            let Some(nal) = data.get(length_size..length_size + n) else {
                return;
            };
            self.decode_nal(nal);
            data = &data[length_size + n..];
        }
    }

    /// Decodes all NAL units of an Annex B byte stream.
    pub fn decode_annex_b(&mut self, data: &[u8]) {
        for nal in AnnexBIter::new(data) {
            self.decode_nal(nal);
        }
    }

    /// Decodes one NAL unit (without start code).
    pub fn decode_nal(&mut self, nal: &[u8]) {
        self.flushed = false;
        let Ok(h) = NalHeader::parse(nal) else { return };
        let mut rbsp = std::mem::take(&mut self.rbsp);
        if rbsp.capacity() == 0
            && let Some(b) = self.rbsp_pool.lock().unwrap().pop()
        {
            rbsp = b;
        }
        rbsp.clear();
        unescape_into(&nal[h.len..], &mut rbsp);
        match h.nal_type {
            NalType::Sps => {
                if let Ok(s) = Sps::parse(&rbsp) {
                    let id = s.id as usize;
                    self.sps[id] = Some(Arc::new(s));
                }
            }
            NalType::SubsetSps => {
                if let Ok(s) = SubsetSps::parse(&rbsp) {
                    let id = s.sps.id as usize;
                    self.subset[id] = Some(Arc::new(s));
                }
            }
            NalType::Pps => {
                if let Ok(p) = Pps::parse(&rbsp) {
                    let id = p.id as usize;
                    self.pps[id] = Some(Arc::new(p));
                }
            }
            NalType::Prefix => self.prefix = h.mvc,
            NalType::Slice | NalType::SliceIdr => {
                let mut h = h;
                h.mvc = self.prefix.take();
                return self.slice(h, rbsp);
            }
            NalType::SliceExtension if self.decode_all_views => return self.slice(h, rbsp),
            NalType::Sei => {
                // Blu-ray 3D offset metadata applies to the base view's frames.
                if let Some(m) = crate::sei::offset_metadata(&rbsp)
                    && self.au > 0
                {
                    self.worker(0).send(Msg::Metadata(Arc::new(m)));
                }
            }
            NalType::EndOfSequence | NalType::EndOfStream => self.finish_current(),
            _ => {}
        }
        self.rbsp = rbsp;
    }

    fn finish_current(&mut self) {
        if let Some(v) = self.cur_view.take() {
            self.worker(v).send(Msg::Finish);
        }
    }

    fn slice(&mut self, h: NalHeader, rbsp: Vec<u8>) {
        let ext_nal = h.nal_type == NalType::SliceExtension;
        let mut used = None;
        let parsed = SliceHeader::parse(&rbsp, &h, |id| {
            let pps = self.pps[id as usize].as_ref()?;
            if ext_nal {
                let sub = self.subset[pps.sps_id as usize].as_ref()?;
                used = Some((pps.clone(), Arc::new(sub.sps.clone()), Some(sub.clone())));
                Some((&sub.sps, &**pps))
            } else {
                let sps = self.sps[pps.sps_id as usize].as_ref()?;
                used = Some((pps.clone(), sps.clone(), None));
                Some((&**sps, &**pps))
            }
        });
        let header = match parsed {
            Ok(v) => v,
            Err(e) => return self.errors.add(e),
        };
        let (pps, sps, subset) = used.unwrap();
        let view_id = h.mvc.map_or(0, |m| m.view_id);
        let voidx = match &subset {
            Some(s) => match s.mvc.view_ids.iter().position(|&v| v == view_id) {
                Some(i) if i > 0 => i,
                _ => return self.errors.add(Error::Invalid("view_id not in subset SPS")),
            },
            None => 0,
        };
        if self.last.len() <= voidx {
            self.last.resize(voidx + 1, None);
        }
        let new_picture = self.cur_view != Some(voidx)
            || match &self.last[voidx] {
                Some((prev, prev_ref)) => {
                    first_slice_of_new_picture(&sps, prev, *prev_ref, &header, h.ref_idc)
                }
                None => true,
            };
        if new_picture {
            // The previous picture is complete; finishing it before the next
            // view starts lets the base picture reach the inter-view slot.
            self.finish_current();
            if voidx == 0 {
                self.au += 1;
                self.au_views.clear();
                self.au_pts = self.pts;
            }
            if self.pic_au.len() <= voidx {
                self.pic_au.resize(voidx + 1, 0);
            }
            self.pic_au[voidx] = self.au;
            // An IDR access unit (non_idr_flag is the same in all its view
            // components, G.7.4.1.1) or mmco 5 outputs every picture before
            // it and starts the order counts again (C.4.4): a new period.
            let mmco5 = matches!(&header.dec_ref_pic_marking,
                Some(DecRefPicMarking::Adaptive(ops)) if ops.contains(&Mmco::ForgetAll));
            if (header.idr || mmco5) && self.period_au != self.au {
                self.period += 1;
                self.period_au = self.au;
            }
        }
        let earlier_views = self
            .au_views
            .iter()
            .filter(|(v, _)| *v != view_id)
            .cloned()
            .collect();
        let publish = new_picture.then(|| {
            let slot = Arc::new(PicSlot::default());
            self.au_views.retain(|(v, _)| *v != view_id);
            self.au_views.push((view_id, slot.clone()));
            Publisher(slot)
        });
        self.cur_view = Some(voidx);
        self.last[voidx] = Some((header.clone(), h.ref_idc));
        let job = SliceJob {
            nal: h,
            header,
            rbsp: Rbsp {
                buf: rbsp,
                pool: self.rbsp_pool.clone(),
            },
            sps,
            pps,
            subset,
            new_picture,
            au: self.au,
            period: self.period,
            view_id,
            // Every view of an access unit has its timestamp, wherever the
            // container put the non-base views.
            pts: self.au_pts,
            publish,
            earlier_views,
        };
        self.worker(voidx).send(Msg::Slice(Box::new(job)));
    }

    /// Drops every picture decoded or in progress and every frame not yet
    /// returned, for a seek: decoding starts again with the next data as at
    /// the start of a stream, keeping the parameter sets received so far,
    /// the settings and the threads. Frames from before it are never
    /// returned.
    pub fn discard(&mut self) {
        for w in &mut self.workers {
            w.send(Msg::Discard);
        }
        self.sync_workers();
        while self.frames_rx.try_recv().is_ok() {}
        self.last.iter_mut().for_each(|l| *l = None);
        self.cur_view = None;
        self.prefix = None;
        self.au_views.clear();
        self.pending.iter_mut().for_each(VecDeque::clear);
        self.ready.clear();
        self.flushed = false;
        self.pts = None;
        self.au_pts = None;
        // Access units after the seek are told apart from those before.
        self.au += 1;
    }

    /// Ends the stream, making all pending frames available.
    pub fn flush(&mut self) {
        self.cur_view = None;
        self.flushed = true;
        // Pictures decoded after this, which start again, follow those before.
        self.period += 1;
        self.period_au = self.au;
        for w in &mut self.workers {
            w.send(Msg::Flush);
        }
        // Wait for threaded workers by closing their queues.
        for w in std::mem::take(&mut self.workers) {
            if let Worker::Thread { tx, mut handle } = w {
                drop(tx);
                if let Some(h) = handle.take() {
                    let _ = h.join();
                }
            }
        }
    }

    /// The next frame in output order. With several views, the views of an
    /// access unit are returned together, in view order.
    ///
    /// The views' frames are merged in output order: next is the earliest
    /// first waiting frame of a view, with those of the other views in the
    /// same place, which are of the same access unit. It is output once
    /// every other view has a frame waiting (each view's come in its output
    /// order, so none earlier will come), or the stream has ended. Which
    /// frames have arrived from view threads depends on timing; the output
    /// must not: the other rules are applied only once the view decoders
    /// have handled all input so far, as they always have without threads.
    pub fn next_frame(&mut self) -> Option<Frame> {
        let views = self.last.len().max(1);
        if let Some(f) = self.receive(views) {
            return Some(f);
        }
        // Every MAX_PENDING access units, with the view decoders synced: a
        // frame that has waited that long since its DPB output it, for a
        // view with pictures but no frames (damaged streams), is output
        // without it.
        if views > 1 && self.au / MAX_PENDING != self.checkpoint {
            self.checkpoint = self.au / MAX_PENDING;
            self.sync_workers();
            self.receive(views);
            self.merge(views, true);
        } else {
            self.merge(views, false);
        }
        let front = self.ready.front()?;
        if !front.iter().flatten().all(|f| self.take(f)) {
            return None;
        }
        let [a, b] = self.ready.pop_front()?;
        match (self.stereo_layout, b) {
            (Some(layout), b) if views >= 2 => {
                // A missing view repeats the other.
                let base = a.as_ref().or(b.as_ref())?;
                let dep = b.as_ref().unwrap_or(base);
                let (l, r) = if self.base_view_right {
                    (dep, base)
                } else {
                    (base, dep)
                };
                Some(crate::stereo::pack(l, r, layout, base))
            }
            (_, b) => a.or(b),
        }
    }

    /// Moves the frames whose place in output order is decided to `ready`
    /// (see `next_frame`). `synced`: the view decoders have handled all input
    /// so far, at a check for frames waiting too long.
    fn merge(&mut self, views: usize, synced: bool) {
        let overdue = synced;
        let mut synced = synced;
        let first = |q: &VecDeque<ViewFrame>| q.front().map(|f| f.order);
        while let Some(order) = self.pending.iter().filter_map(first).min() {
            let waiting = |v: usize| self.pending.get(v).is_some_and(|q| !q.is_empty());
            // A view with no picture in the access unit or since.
            let none =
                |v: usize| self.au > order.au && self.pic_au.get(v).is_none_or(|&a| a < order.au);
            let ok = self.flushed
                || (0..views).all(|v| waiting(v) || synced && none(v))
                || overdue
                    && self
                        .pending
                        .iter()
                        .filter_map(VecDeque::front)
                        .any(|f| f.order == order && f.out_au + MAX_PENDING <= self.au);
            if !ok {
                if !synced && (0..views).all(|v| waiting(v) || none(v)) {
                    self.sync_workers();
                    self.receive(views);
                    synced = true;
                    continue;
                }
                return;
            }
            let group: Vec<Option<Frame>> = self
                .pending
                .iter_mut()
                .map(|q| {
                    let at = first(q) == Some(order);
                    at.then(|| q.pop_front().map(|f| f.frame)).flatten()
                })
                .collect();
            if self.stereo_layout.is_some() && views >= 2 {
                let mut views = group.into_iter();
                let (base, dep) = (views.next().flatten(), views.next().flatten());
                self.ready.push_back([base, dep]);
            } else {
                self.ready
                    .extend(group.into_iter().flatten().map(|f| [Some(f), None]));
            }
        }
    }

    /// Whether frame `f` can be output: frames still being decoded wait
    /// (frame threading) unless the stream has ended, as blocking here
    /// would stop the caller feeding the decoder.
    fn take(&self, f: &Frame) -> bool {
        if self.flushed {
            f.wait();
        }
        f.decoded()
    }

    /// Takes the frames the view decoders have output into the views'
    /// queues; with one view, returns a frame that needs no merging.
    fn receive(&mut self, views: usize) -> Option<Frame> {
        while let Ok(f) = self.frames_rx.try_recv() {
            if views == 1
                && self.pending.iter().all(VecDeque::is_empty)
                && self.ready.is_empty()
                && self.take(&f.frame)
            {
                return Some(f.frame);
            }
            if self.pending.len() <= f.voidx {
                self.pending.resize_with(f.voidx + 1, VecDeque::new);
            }
            self.pending[f.voidx].push_back(f);
        }
        None
    }

    /// Waits until every view decoder has handled all input sent to it.
    fn sync_workers(&mut self) {
        for w in &mut self.workers {
            if let Worker::Thread { tx, .. } = w {
                let (done_tx, done_rx) = std::sync::mpsc::channel();
                if tx.send(Msg::Sync(done_tx)).is_ok() {
                    let _ = done_rx.recv();
                }
            }
        }
    }
}

/// Whether this WebAssembly build can spawn threads. Stable cfg can't tell.
fn wasm_threads() -> bool {
    static THREADS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *THREADS.get_or_init(|| {
        std::thread::Builder::new()
            .spawn(|| {})
            .is_ok_and(|t| t.join().is_ok())
    })
}

/// Counts a view thread as live until it ends (tests).
#[cfg(test)]
struct LiveView(Arc<std::sync::atomic::AtomicUsize>);

#[cfg(test)]
impl Drop for LiveView {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod drop_tests {
    use super::*;

    // Dropping a decoder mid-stream ends its view threads before drop
    // returns, instead of leaving them decoding the queued access units.
    #[test]
    #[cfg_attr(target_family = "wasm", ignore = "no threads")]
    fn drop_mid_stream_ends_view_threads() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../work/conformance/MVC/MVCDS-6.264"
        );
        let Ok(data) = std::fs::read(path) else {
            eprintln!("skipped: {path} missing (cargo xtask conformance)");
            return;
        };
        for _ in 0..5 {
            let mut dec = Decoder::with_threads(4);
            dec.decode_annex_b(&data[..data.len() / 2]);
            let live = dec.live_views.clone();
            assert!(live.load(Ordering::SeqCst) > 0, "no view threads to test");
            drop(dec);
            assert_eq!(
                live.load(Ordering::SeqCst),
                0,
                "view threads still running after drop"
            );
        }
    }
}
