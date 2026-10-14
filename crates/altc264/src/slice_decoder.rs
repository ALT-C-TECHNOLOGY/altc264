//! Slice data decoding: macroblock parsing (7.3.4, 7.3.5) and reconstruction.
//!
//! Current scope: I and P slices, CABAC, frame pictures without MBAFF, 4:2:0
//! or 4:0:0, 8-bit, Intra_4x4 / Intra_16x16 / I_PCM and all P partitions.
//!
//! Loops index blocks by luma4x4BlkIdx like the spec, so clippy's
//! needless_range_loop is allowed for this module.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use crate::Error;
use crate::bits::BitReader;
use crate::cabac::{Ctx, Decoder, init_contexts};
use crate::cabac_se as se;
use crate::cavlc;
use crate::cavlc_tables::{CBP_INTER_0_3, CBP_INTER_420, CBP_INTRA_0_3, CBP_INTRA_420};
use crate::dpb::{ColInfo, RefPic};
use crate::inter::{self, Weight};
use crate::intra::{self, Edge4, Edge8, EdgeMb};
use crate::mb::{MbType, Partition, PredMode, SubMbType, SubPartition};
use crate::picture::{Parity, Picture, Pixel, Plane, PlaneView};
use crate::progress::PicState;
use crate::ps::{Pps, PpsExt, Sps};
use crate::slice::{SliceHeader, SliceType};
use crate::transform::{self, Coeffs, FIELD_4X4, FIELD_8X8, LevelScale, ZIGZAG_4X4, ZIGZAG_8X8};

/// Raster position (x, y) in 4x4 units of each luma4x4BlkIdx (6.4.3).
const BLK_XY: [(usize, usize); 16] = [
    (0, 0),
    (1, 0),
    (0, 1),
    (1, 1),
    (2, 0),
    (3, 0),
    (2, 1),
    (3, 1),
    (0, 2),
    (1, 2),
    (0, 3),
    (1, 3),
    (2, 2),
    (3, 2),
    (2, 3),
    (3, 3),
];
/// luma4x4BlkIdx of the 4x4 block at raster (x + 4 * y).
const BLK_AT: [usize; 16] = [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15];

/// QPC as a function of qPI >= 30 (Table 8-15).
const QPC_TABLE: [i32; 22] = [
    29, 30, 31, 32, 32, 33, 34, 34, 35, 35, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 39,
];

/// QPC from QPY (Table 8-15); `bd_offset` is QpBdOffsetC.
fn chroma_qp(qp_y: i32, offset: i32, bd_offset: i32) -> i32 {
    let qpi = (qp_y + offset).clamp(-bd_offset, 51);
    if qpi < 30 {
        qpi
    } else {
        QPC_TABLE[(qpi - 30) as usize]
    }
}

// Bits of MbInfo::cbf.
const CBF_LUMA_DC: u32 = 16;
const CBF_CHROMA_DC: u32 = 17; // + iCbCr
const CBF_CHROMA_AC: u32 = 19; // + 16 * iCbCr + blkIdx (4:4:4: luma4x4BlkIdx)

/// The coded_block_flag bit of 4x4 block `blk` of a plane coded like luma.
#[inline]
fn cbf_bit(plane: usize, blk: usize) -> u32 {
    if plane == 0 {
        blk as u32
    } else {
        CBF_CHROMA_AC + 16 * (plane as u32 - 1) + blk as u32
    }
}

/// The coded_block_flag bit of the Intra_16x16 DC block of a plane.
#[inline]
fn cbf_dc_bit(plane: usize) -> u32 {
    if plane == 0 {
        CBF_LUMA_DC
    } else {
        CBF_CHROMA_DC + plane as u32 - 1
    }
}

/// Per-macroblock state kept for neighbour derivations and later stages.
///
/// Laid out by use, a cache line apart: what deblocking and the neighbour
/// checks read for every macroblock first (one line), then the vectors
/// (two) and the intra modes. 64-aligned, so a macroblock's lines are its
/// own. What only the entropy decoding of the neighbours below and to the
/// right reads is in [`MbCtx`].
#[derive(Clone, Copy, Debug)]
#[repr(C, align(64))]
pub struct MbInfo {
    /// Slice the macroblock belongs to, or u32::MAX when not yet decoded.
    pub slice: u32,
    pub mb_type: MbType,
    pub transform_8x8: bool,
    pub cbp_luma: u8,
    pub cbp_chroma: u8,
    pub qp_y: i8,
    pub chroma_pred_mode: u8,
    /// 8x8 blocks predicted in direct mode (B_Direct_8x8, B_Skip, B_Direct_16x16).
    pub direct8x8: u8,
    /// One motion for the whole macroblock, known from direct prediction
    /// (motion_blocks need not compare its blocks).
    pub(crate) uniform_motion: bool,
    /// A field macroblock of an MBAFF frame (mb_field_decoding_flag).
    pub field: bool,
    /// coded_block_flag bits, see CBF_* (luma 4x4 blocks in bits 0..16).
    pub cbf: u64,
    /// Reference index per list and 8x8 block (raster order), -1 when unused.
    pub ref_idx: [[i8; 4]; 2],
    /// Identity of the referenced picture per list and 8x8 block.
    pub ref_id: [[u32; 4]; 2],
    /// Motion vectors per list and 4x4 block (raster order), quarter samples.
    pub mv: [[[i16; 2]; 16]; 2],
    /// Intra4x4PredMode (Intra8x8PredMode replicated) per luma4x4BlkIdx.
    pub intra_modes: [u8; 16],
}

const _: () = assert!(std::mem::size_of::<MbInfo>() == 256);

/// What the entropy decoding of later macroblocks of the slice reads of a
/// macroblock besides its MbInfo: its CABAC mvd and CAVLC coefficient
/// counts. Only the row above is ever read (see `Mbs::ctx`).
#[derive(Clone, Copy, Debug, Default)]
pub struct MbCtx {
    /// Absolute mvd per list, 4x4 block and component (saturated), for CABAC.
    pub mvd: [[[u8; 2]; 16]; 2],
    /// Nonzero coefficient count per luma4x4BlkIdx (CAVLC nC).
    pub nnz: [u8; 16],
    /// Nonzero AC coefficient count per chroma component and 4x4 block (CAVLC nC).
    pub nnz_c: [[u8; 16]; 2],
}

impl MbCtx {
    /// Nonzero coefficient counts of the 4x4 blocks of a plane coded like
    /// luma (Cb and Cr in 4:4:4).
    #[inline]
    fn nnz_of(&self, plane: usize) -> &[u8; 16] {
        if plane == 0 {
            &self.nnz
        } else {
            &self.nnz_c[plane - 1]
        }
    }

    #[inline]
    fn nnz_of_mut(&mut self, plane: usize) -> &mut [u8; 16] {
        if plane == 0 {
            &mut self.nnz
        } else {
            &mut self.nnz_c[plane - 1]
        }
    }
}

/// The macroblock being decoded: its MbInfo and MbCtx, stored apart.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct MbWork {
    pub info: MbInfo,
    pub ctx: MbCtx,
}

impl std::ops::Deref for MbWork {
    type Target = MbInfo;
    #[inline(always)]
    fn deref(&self) -> &MbInfo {
        &self.info
    }
}

impl std::ops::DerefMut for MbWork {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut MbInfo {
        &mut self.info
    }
}

pub const NO_REF: u32 = u32::MAX;

impl Default for MbInfo {
    fn default() -> Self {
        Self {
            slice: u32::MAX,
            mb_type: MbType::INxN,
            transform_8x8: false,
            cbp_luma: 0,
            cbp_chroma: 0,
            qp_y: 0,
            chroma_pred_mode: 0,
            intra_modes: [2; 16],
            cbf: 0,
            mv: [[[0; 2]; 16]; 2],
            ref_idx: [[-1; 4]; 2],
            ref_id: [[NO_REF; 4]; 2],
            direct8x8: 0,
            uniform_motion: false,
            field: false,
        }
    }
}

impl MbInfo {
    fn is_pcm(&self) -> bool {
        matches!(self.mb_type, MbType::IPcm)
    }
    fn is_i16x16(&self) -> bool {
        matches!(self.mb_type, MbType::I16x16 { .. })
    }
    fn is_skip(&self) -> bool {
        matches!(self.mb_type, MbType::Skip)
    }
    pub fn is_intra(&self) -> bool {
        self.mb_type.is_intra()
    }
}

/// Everything fixed for the duration of one slice.
pub struct SliceCtx<'a> {
    pub sps: &'a Sps,
    pub pps: &'a Pps,
    pub ext: &'a PpsExt,
    pub ls: &'a LevelScale,
    pub header: &'a SliceHeader,
    pub slice_num: u32,
    pub ref_lists: &'a [Vec<Option<RefPic>>; 2],
    /// MBAFF frames: the lists of fields for field macroblocks, by the
    /// macroblock's parity (8.4.2.1).
    pub field_lists: Option<&'a [crate::dpb::RefLists; 2]>,
    /// PicOrderCnt(CurrPic), for temporal direct and implicit weights.
    pub poc: i32,
    /// Top and bottom field order counts of the current picture.
    pub field_poc: [i32; 2],
    /// The parity of the current macroblock when it is a field macroblock
    /// of an MBAFF frame.
    pub mb_field: std::cell::Cell<Option<Parity>>,
    /// The references of `ref_lists`, and of the field lists by parity, as
    /// motion compensation reads them.
    pub mc_refs: McRefs<'a>,
    pub mc_field_refs: Option<[McRefs<'a>; 2]>,
    /// Implicit bi-prediction weights w1 of frame macroblocks by refIdxL0 *
    /// list 1 length + refIdxL1, found on first use (i16::MIN until then;
    /// 32 for the default average). Empty without implicit weights.
    pub implicit: Box<[std::cell::Cell<i16>]>,
}

impl SliceCtx<'_> {
    /// [`SliceCtx::implicit`] for this slice.
    pub fn implicit_table(
        header: &SliceHeader,
        pps: &Pps,
        lists: &[Vec<Option<RefPic>>; 2],
    ) -> Box<[std::cell::Cell<i16>]> {
        let implicit = header.slice_type == SliceType::B && pps.weighted_bipred_idc == 2;
        let n = if implicit {
            lists[0].len() * lists[1].len()
        } else {
            0
        };
        (0..n).map(|_| std::cell::Cell::new(i16::MIN)).collect()
    }
}

/// A reference picture as motion compensation reads it: its planes and
/// decoding progress, resolved once per slice rather than per block.
#[derive(Clone, Copy)]
pub struct McRef<'a> {
    /// The planes (repeating plane 0 past the last).
    planes: [PlaneView<'a>; 3],
    state: &'a PicState,
    /// A field of a frame (its rows interleave with the other field's).
    field_of_frame: bool,
    parity: Option<Parity>,
}

impl<'a> McRef<'a> {
    fn new(rp: &'a RefPic) -> Self {
        let n = rp.pic.planes.len();
        McRef {
            planes: [0, 1, 2].map(|i| rp.plane(if i < n { i } else { 0 })),
            state: rp.state(),
            field_of_frame: rp.field_of_frame.is_some(),
            parity: rp.parity,
        }
    }
}

/// [`McRef`]s of a slice's two reference lists, by refIdx, and per list
/// the reference that stands in for a missing one (see `SliceCtx::mc_ref`).
pub struct McRefs<'a> {
    lists: [Vec<Option<McRef<'a>>>; 2],
    stand_in: [Option<McRef<'a>>; 2],
}

impl<'a> McRefs<'a> {
    pub fn new(lists: &'a [Vec<Option<RefPic>>; 2]) -> Self {
        let first = |l: usize| lists[l].iter().flatten().next();
        McRefs {
            lists: [0, 1].map(|l| {
                lists[l]
                    .iter()
                    .map(|r| r.as_ref().map(McRef::new))
                    .collect()
            }),
            stand_in: [0, 1].map(|l| first(l).or_else(|| first(1 - l)).map(McRef::new)),
        }
    }
}

impl<'a> SliceCtx<'a> {
    /// BitDepthY for plane 0, BitDepthC otherwise.
    fn bit_depth(&self, plane: usize) -> u32 {
        if plane == 0 {
            self.sps.bit_depth_luma as u32
        } else {
            self.sps.bit_depth_chroma as u32
        }
    }

    /// The largest sample value of a plane.
    fn max_sample(&self, plane: usize) -> i32 {
        (1 << self.bit_depth(plane)) - 1
    }

    /// Field picture or field macroblock.
    fn field(&self) -> bool {
        self.header.field_pic || self.mb_field.get().is_some()
    }

    /// Inverse 4x4 and 8x8 scans (8.5.6, 8.5.7): field scans for fields.
    fn scan4(&self) -> &'static [u8; 16] {
        if self.field() {
            &FIELD_4X4
        } else {
            &ZIGZAG_4X4
        }
    }

    fn scan8(&self) -> &'static [u8; 64] {
        if self.field() {
            &FIELD_8X8
        } else {
            &ZIGZAG_8X8
        }
    }

    /// Reference picture list `l` for the current macroblock.
    fn list(&self, l: usize) -> &'a [Option<RefPic>] {
        match (self.mb_field.get(), self.field_lists) {
            (Some(p), Some(f)) => &f[p as usize][l],
            _ => &self.ref_lists[l],
        }
    }

    /// The reference that list `l`, refIdx `r` predicts from (None for a
    /// negative r). A missing reference (damaged stream, or a stream
    /// starting at a non-IDR picture) is replaced by any available one,
    /// of list l first; with none there is no picture (Some(None)).
    #[inline(always)]
    fn mc_ref(&self, l: usize, r: i8) -> Option<Option<&McRef<'a>>> {
        let r = usize::try_from(r).ok()?;
        let m = match (self.mb_field.get(), &self.mc_field_refs) {
            (Some(p), Some(f)) => &f[p as usize],
            _ => &self.mc_refs,
        };
        Some(
            m.lists[l]
                .get(r)
                .and_then(|x| x.as_ref())
                .or(m.stand_in[l].as_ref()),
        )
    }

    /// num_ref_idx_lX_active: doubled for field macroblocks (7.4.5.1).
    fn num_active(&self, l: usize) -> u32 {
        self.header.num_ref_idx_active[l] << self.mb_field.get().is_some() as u32
    }

    /// PicOrderCnt of the current picture, or of its field with the
    /// current field macroblock's parity (8.4.1.2.3, 8.4.2.3.1).
    fn cur_poc(&self) -> i32 {
        match self.mb_field.get() {
            Some(p) => self.field_poc[p as usize],
            None => self.poc,
        }
    }

    /// Prepares for macroblock `addr`: field macroblocks of MBAFF frames
    /// use fields as references.
    fn start_mb(&self, addr: usize, field: bool) {
        self.mb_field.set(field.then_some(if addr % 2 == 1 {
            Parity::Bottom
        } else {
            Parity::Top
        }));
    }
}

/// The MbInfo a slice decoding writes and reads, reached one macroblock at
/// a time: with frame threading, other threads read the rows of the
/// picture already published (PicState) while this thread writes others,
/// so no reference may cover them all.
pub struct InfoView<'a> {
    ptr: *mut MbInfo,
    len: usize,
    _mbs: std::marker::PhantomData<&'a mut [MbInfo]>,
}

// SAFETY: as &mut [MbInfo] (the view is the only writer of what it writes).
unsafe impl Send for InfoView<'_> {}

impl<'a> InfoView<'a> {
    pub fn new(info: &'a mut [MbInfo]) -> Self {
        Self {
            ptr: info.as_mut_ptr(),
            len: info.len(),
            _mbs: std::marker::PhantomData,
        }
    }

    /// The view of `len` macroblocks at `ptr`.
    ///
    /// # Safety
    ///
    /// For 'a, only this view writes them, and other threads read only
    /// those it no longer writes (rows published in a PicState).
    pub(crate) unsafe fn from_raw(ptr: *mut MbInfo, len: usize) -> Self {
        Self {
            ptr,
            len,
            _mbs: std::marker::PhantomData,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// The same macroblocks, for a shorter borrow.
    pub(crate) fn reborrow(&mut self) -> InfoView<'_> {
        InfoView {
            ptr: self.ptr,
            len: self.len,
            _mbs: std::marker::PhantomData,
        }
    }

    #[inline(always)]
    pub(crate) fn get(&self, i: usize) -> &MbInfo {
        assert!(i < self.len);
        // SAFETY: in bounds; not written meanwhile (only this view writes).
        unsafe { &*self.ptr.add(i) }
    }

    #[inline(always)]
    pub(crate) fn get_mut(&mut self, i: usize) -> &mut MbInfo {
        assert!(i < self.len);
        // SAFETY: in bounds; other threads do not read what this view
        // writes (see from_raw).
        unsafe { &mut *self.ptr.add(i) }
    }

    /// All of them, to read (by this thread, or rows published).
    #[inline(always)]
    pub(crate) fn as_slice(&self) -> &[MbInfo] {
        // SAFETY: nothing writes them while the slice lives (this thread
        // writes through the view only, which the borrow holds).
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

/// The macroblocks a slice may touch: element 0 is macroblock `first`.
/// Indexed by macroblock address.
pub(crate) struct InfoChunk<'a> {
    data: InfoView<'a>,
    first: usize,
}

impl std::ops::Index<usize> for InfoChunk<'_> {
    type Output = MbInfo;
    #[inline]
    fn index(&self, addr: usize) -> &MbInfo {
        self.data.get(addr - self.first)
    }
}

impl std::ops::IndexMut<usize> for InfoChunk<'_> {
    #[inline]
    fn index_mut(&mut self, addr: usize) -> &mut MbInfo {
        self.data.get_mut(addr - self.first)
    }
}

pub(crate) struct Mbs<'a> {
    info: InfoChunk<'a>,
    width: usize,
    slice: u32,
    /// first_mb_in_slice: without slice groups a slice is a run of
    /// consecutive macroblocks, so a neighbour is in it iff its address is
    /// at least this (6.4.8). Availability never looks at other slices.
    first: usize,
    /// Slice groups: a slice's macroblocks are not consecutive, so
    /// neighbours are in it iff they carry its number.
    fmo: bool,
    /// constrained_intra_pred_flag: inter neighbours are unusable for intra prediction.
    constrained_intra: bool,
    /// An MBAFF frame: neighbours follow Table 6-4.
    mbaff: bool,
    /// mb_field_decoding_flag of the current macroblock pair.
    cur_field: std::cell::Cell<bool>,
    /// The current macroblock is SI: SI neighbours are then usable for intra
    /// prediction under constrained_intra_pred_flag (8.3.1.2).
    cur_si: std::cell::Cell<bool>,
    /// Outside MBAFF frames, the macroblocks around `near_of` by region of
    /// the location (see [`Self::region`]), NONE where there is none.
    near: [usize; 9],
    near_of: usize,
    /// MbCtx of the macroblocks of the last two rows (of pairs in MBAFF
    /// frames), by address modulo its length (a power of two): entropy decoding reads those
    /// of the row above and this one only.
    ctx: Vec<MbCtx>,
}

const NONE: usize = usize::MAX;

/// Outside MBAFF frames, macroblocks A, B, C and D of the current one
/// (6.4.9), where available.
#[derive(Clone, Copy)]
struct Around<'m>([Option<&'m MbInfo>; 4]);

impl Mbs<'_> {
    /// The macroblocks around `addr`, outside MBAFF frames.
    #[inline]
    fn around(&self, addr: usize) -> Option<Around<'_>> {
        if self.mbaff {
            return None;
        }
        let f = |n: Option<usize>| n.map(|n| &self.info[n]);
        Some(Around([
            f(self.a(addr)),
            f(self.b(addr)),
            f(self.c(addr)),
            f(self.d(addr)),
        ]))
    }

    /// The MbCtx of macroblock `addr` (of this row or the one above).
    #[inline(always)]
    fn ctx(&self, addr: usize) -> &MbCtx {
        &self.ctx[addr & (self.ctx.len() - 1)]
    }

    /// Stores the decoded macroblock `addr`.
    #[inline(always)]
    fn store(&mut self, addr: usize, mb: &MbWork) {
        self.info[addr] = mb.info;
        let n = self.ctx.len();
        self.ctx[addr & (n - 1)] = mb.ctx;
    }

    /// Macroblock `cur` is next: caches its neighbours.
    fn enter(&mut self, cur: usize) {
        if self.mbaff {
            return;
        }
        let o = |n: Option<usize>| n.unwrap_or(NONE);
        // One division for the column, which a to d would each work out.
        let (w, x) = (self.width, cur % self.width);
        let (a, b, c, d) = (
            o(self.avail((x != 0).then(|| cur - 1))),
            o(self.avail(cur.checked_sub(w))),
            o(self.avail(cur.checked_sub(w - 1).filter(|_| x + 1 != w))),
            o(self.avail(cur.checked_sub(w + 1).filter(|_| x != 0))),
        );
        self.near = [d, b, c, a, cur, NONE, NONE, NONE, NONE];
        self.near_of = cur;
    }
    /// Index into `near` of the region holding location (x, y) relative to
    /// a w x h macroblock: D, B, C above, A, the macroblock, none.
    #[inline(always)]
    fn region(x: i32, y: i32, w: i32, h: i32) -> usize {
        (x >= 0) as usize + (x >= w) as usize + 3 * ((y >= 0) as usize + (y >= h) as usize)
    }
    #[inline(always)]
    fn cached(&self, cur: usize, k: usize) -> Option<Option<usize>> {
        (cur == self.near_of).then(|| Some(self.near[k]).filter(|&n| n != NONE))
    }
    #[inline]
    fn avail(&self, addr: Option<usize>) -> Option<usize> {
        addr.filter(|&a| a >= self.first && (!self.fmo || self.info[a].slice == self.slice))
    }
    /// mbAddrA / B / C / D: the neighbouring macroblocks containing the
    /// luma locations (-1, 0), (0, -1), (16, -1) and (-1, -1) (6.4.9, and
    /// 6.4.10.1 for MBAFF frames).
    #[inline]
    fn a(&self, cur: usize) -> Option<usize> {
        if let Some(n) = self.cached(cur, 3) {
            return n;
        }
        if self.mbaff {
            return self.locate(cur, -1, 0).map(|n| n.0);
        }
        self.avail((cur % self.width != 0).then(|| cur - 1))
    }
    #[inline]
    fn b(&self, cur: usize) -> Option<usize> {
        if let Some(n) = self.cached(cur, 1) {
            return n;
        }
        if self.mbaff {
            return self.locate(cur, 0, -1).map(|n| n.0);
        }
        self.avail(cur.checked_sub(self.width))
    }
    #[inline]
    fn c(&self, cur: usize) -> Option<usize> {
        if let Some(n) = self.cached(cur, 2) {
            return n;
        }
        if self.mbaff {
            return self.locate(cur, 16, -1).map(|n| n.0);
        }
        self.avail(
            cur.checked_sub(self.width - 1)
                .filter(|_| (cur + 1) % self.width != 0),
        )
    }
    #[inline]
    fn d(&self, cur: usize) -> Option<usize> {
        if let Some(n) = self.cached(cur, 0) {
            return n;
        }
        if self.mbaff {
            return self.locate(cur, -1, -1).map(|n| n.0);
        }
        self.avail(
            cur.checked_sub(self.width + 1)
                .filter(|_| cur % self.width != 0),
        )
    }
    /// The top macroblocks of the left, above, above-right and above-left
    /// macroblock pairs in an MBAFF frame (6.4.10).
    fn pair(&self, cur: usize, dx: i32, dy: i32) -> Option<usize> {
        let (px, py) = (
            (cur / 2 % self.width) as i32 + dx,
            (cur / 2 / self.width) as i32 + dy,
        );
        if px < 0 || px >= self.width as i32 || py < 0 {
            return None;
        }
        self.avail(Some(2 * (py as usize * self.width + px as usize)))
    }
    /// Whether macroblock `n` is a field macroblock; the current one's
    /// flag is set while it is decoded.
    fn is_field(&self, n: usize, cur: usize) -> bool {
        if n == cur || n + 1 == cur && cur % 2 == 1 {
            // The current pair: its flag may not be stored yet.
            self.cur_field.get()
        } else {
            self.info[n].field
        }
    }
    /// Neighbour usable for intra prediction samples (8.3.1.2 and friends).
    #[inline]
    fn intra_ok(&self, n: Option<usize>) -> bool {
        n.is_some_and(|n| {
            let m = &self.info[n];
            !self.constrained_intra
                || m.is_intra() && (!matches!(m.mb_type, MbType::Si) || self.cur_si.get())
        })
    }
    /// Macroblock containing the luma location (x, y) relative to `cur`
    /// (6.4.12), with the location inside that macroblock.
    #[inline(always)]
    fn locate(&self, cur: usize, x: i32, y: i32) -> Option<(usize, usize, usize)> {
        // The current macroblock's neighbours are cached (outside MBAFF).
        if cur == self.near_of {
            let n = self.near[Self::region(x, y, 16, 16)];
            return (n != NONE).then_some((n, (x & 15) as usize, (y & 15) as usize));
        }
        self.locate_slow(cur, x, y)
    }
    #[inline(never)]
    fn locate_slow(&self, cur: usize, x: i32, y: i32) -> Option<(usize, usize, usize)> {
        if self.mbaff {
            return self.locate_mbaff(cur, x, y, 16, 16);
        }
        let n = match (x, y) {
            (..0, ..0) => self.d(cur),
            (0..16, ..0) => self.b(cur),
            (16.., ..0) => self.c(cur),
            (..0, 0..16) => self.a(cur),
            (0..16, 0..16) => Some(cur),
            _ => None,
        }?;
        Some((n, x.rem_euclid(16) as usize, y.rem_euclid(16) as usize))
    }
    /// As locate, for a chroma location (maxW x maxH = MbWidthC x MbHeightC).
    #[inline(always)]
    fn locate_c(
        &self,
        cur: usize,
        x: i32,
        y: i32,
        mw: i32,
        mh: i32,
    ) -> Option<(usize, usize, usize)> {
        // MbWidthC and MbHeightC are 8 or 16: masks are remainders.
        debug_assert!(mw.count_ones() == 1 && mh.count_ones() == 1);
        if cur == self.near_of {
            let n = self.near[Self::region(x, y, mw, mh)];
            return (n != NONE).then(|| (n, (x & (mw - 1)) as usize, (y & (mh - 1)) as usize));
        }
        self.locate_c_slow(cur, x, y, mw, mh)
    }
    #[inline(never)]
    fn locate_c_slow(
        &self,
        cur: usize,
        x: i32,
        y: i32,
        mw: i32,
        mh: i32,
    ) -> Option<(usize, usize, usize)> {
        if self.mbaff {
            return self.locate_mbaff(cur, x, y, mw, mh);
        }
        let at = |n: usize| (n, (x & (mw - 1)) as usize, (y & (mh - 1)) as usize);
        let n = match (x, y) {
            (..0, ..0) => self.d(cur),
            (0.., ..0) if x < mw => self.b(cur),
            (_, ..0) => self.c(cur),
            (..0, _) if y < mh => self.a(cur),
            _ if x < mw && y < mh => Some(cur),
            _ => None,
        }?;
        Some(at(n))
    }
    /// Table 6-4: neighbouring locations in MBAFF frames.
    #[inline(never)]
    fn locate_mbaff(
        &self,
        cur: usize,
        xn: i32,
        yn: i32,
        mw: i32,
        mh: i32,
    ) -> Option<(usize, usize, usize)> {
        mbaff_neighbour(
            cur,
            self.cur_field.get(),
            xn,
            yn,
            mw,
            mh,
            |dx, dy| self.pair(cur, dx, dy),
            |n| self.is_field(n, cur),
        )
    }
}

/// Table 6-4: the macroblock containing the location (xn, yn) relative to
/// macroblock `cur` of an MBAFF frame, and the location inside it.
/// `pair(dx, dy)` gives the top macroblock of an available neighbouring
/// pair (6.4.10) and `is_field(n)` whether macroblock n is a field one.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mbaff_neighbour(
    cur: usize,
    cur_field: bool,
    xn: i32,
    yn: i32,
    mw: i32,
    mh: i32,
    pair: impl Fn(i32, i32) -> Option<usize>,
    is_field: impl Fn(usize) -> bool,
) -> Option<(usize, usize, usize)> {
    {
        let frame = !cur_field;
        let top = cur % 2 == 0;
        let fld = |n: usize| is_field(n);
        let (n, ym) = if yn > mh - 1 {
            return None;
        } else if xn < 0 && yn < 0 {
            if frame {
                if top {
                    (pair(-1, -1)? + 1, yn)
                } else {
                    // Row 15 of the left pair: an odd line in a field pair.
                    let a = pair(-1, 0)?;
                    if fld(a) {
                        (a + 1, (yn + mh) >> 1)
                    } else {
                        (a, yn)
                    }
                }
            } else if top {
                let d = pair(-1, -1)?;
                if fld(d) { (d, yn) } else { (d + 1, 2 * yn) }
            } else {
                (pair(-1, -1)? + 1, yn)
            }
        } else if xn < 0 {
            let a = pair(-1, 0)?;
            match (frame, top, fld(a)) {
                (true, true, false) => (a, yn),
                (true, true, true) => (a + (yn % 2) as usize, yn >> 1),
                (true, false, false) => (a + 1, yn),
                (true, false, true) => (a + (yn % 2) as usize, (yn + mh) >> 1),
                (false, true, false) if yn < mh / 2 => (a, yn << 1),
                (false, true, false) => (a + 1, (yn << 1) - mh),
                (false, true, true) => (a, yn),
                (false, false, false) if yn < mh / 2 => (a, (yn << 1) + 1),
                (false, false, false) => (a + 1, (yn << 1) + 1 - mh),
                (false, false, true) => (a + 1, yn),
            }
        } else if xn < mw {
            if yn >= 0 {
                (cur, yn)
            } else if frame {
                if top {
                    (pair(0, -1)? + 1, yn)
                } else {
                    (cur - 1, yn)
                }
            } else if top {
                let b = pair(0, -1)?;
                if fld(b) { (b, yn) } else { (b + 1, 2 * yn) }
            } else {
                (pair(0, -1)? + 1, yn)
            }
        } else if yn >= 0 {
            return None;
        } else if frame {
            if top {
                (pair(1, -1)? + 1, yn)
            } else {
                return None;
            }
        } else if top {
            let c = pair(1, -1)?;
            if fld(c) { (c, yn) } else { (c + 1, 2 * yn) }
        } else {
            (pair(1, -1)? + 1, yn)
        };
        // mw and mh are 8 or 16: masks are remainders.
        Some((n, (xn & (mw - 1)) as usize, (ym & (mh - 1)) as usize))
    }
}

/// A picture that several slices reconstruct at the same time.
///
/// Slices own disjoint runs of macroblocks (the scheduler checks this), and
/// a slice reads only samples of macroblocks it has already stored itself
/// (its available neighbours). So no sample is ever written by one thread
/// while another reads or writes it, and plain copies through raw pointers
/// are race-free. All such access goes through `Pix`, which asserts that
/// each macroblock is inside its slice's run.
pub struct SharedPic<'a, T: Pixel = u8> {
    /// (data, len, stride) per plane.
    planes: Vec<(*mut T, usize, usize)>,
    width_mbs: usize,
    height_mbs: usize,
    mb_c: (usize, usize),
    bit_depth: [u32; 2],
    _pic: std::marker::PhantomData<&'a mut Picture>,
}

// SAFETY: see the type documentation; access is confined to disjoint
// macroblock runs per thread.
unsafe impl<T: Pixel> Send for SharedPic<'_, T> {}
unsafe impl<T: Pixel> Sync for SharedPic<'_, T> {}

impl<'a, T: Pixel> SharedPic<'a, T> {
    /// `pic` must have samples of type T.
    pub fn new(pic: &'a mut Picture) -> Self {
        assert_eq!(pic.high(), T::HIGH);
        Self {
            planes: pic
                .planes
                .iter_mut()
                .map(|p| {
                    let stride = p.stride;
                    let d = T::of_mut(p);
                    (d.as_mut_ptr(), d.len(), stride)
                })
                .collect(),
            width_mbs: pic.width_mbs,
            height_mbs: pic.height_mbs,
            mb_c: pic.mb_c,
            bit_depth: pic.bit_depth,
            _pic: std::marker::PhantomData,
        }
    }

    /// A picture whose planes are written through raw pointers (frame
    /// threading: the picture is shared with the pictures referencing it,
    /// which only read rows it has finished).
    pub(crate) fn from_raw(
        planes: &'a [crate::picture::RawPlane<T>],
        width_mbs: usize,
        height_mbs: usize,
        mb_c: (usize, usize),
        bit_depth: [u32; 2],
    ) -> Self {
        Self {
            planes: planes.iter().map(|p| (p.ptr, p.len, p.stride)).collect(),
            width_mbs,
            height_mbs,
            mb_c,
            bit_depth,
            _pic: std::marker::PhantomData,
        }
    }

    pub fn total_mbs(&self) -> usize {
        self.width_mbs * self.height_mbs
    }
}

/// Called after each macroblock is stored, with its address and the
/// picture's macroblocks (frame threading deblocks rows as they complete).
pub type MbDone<'a> = &'a mut (dyn FnMut(usize, &[MbInfo]) + Send);

/// What one slice may touch: macroblocks `first..end` of the picture and
/// their MbInfo, `info[0]` being macroblock `info_first`.
pub struct SliceArea<'a, T: Pixel = u8> {
    pub pic: &'a SharedPic<'a, T>,
    pub info: InfoView<'a>,
    pub info_first: usize,
    pub end: usize,
    pub on_mb: Option<MbDone<'a>>,
    /// No other slice of the picture is decoded at the same time, and no
    /// other thread reads the rows of the macroblock row being decoded
    /// (slices in turn, or frame threading): macroblocks may then be
    /// reconstructed in the picture itself.
    pub exclusive: bool,
}

/// One plane of the macroblock being reconstructed, with its neighbours:
/// rows `stride` apart.
pub(crate) struct Win<'a, T> {
    data: &'a mut [T],
    stride: usize,
}

impl<T> Win<'_, T> {
    #[inline(always)]
    fn s(&self) -> &[T] {
        self.data
    }
    #[inline(always)]
    fn m(&mut self) -> &mut [T] {
        self.data
    }
}

/// Where macroblocks are reconstructed. With exclusive access (see
/// SliceArea) and outside MBAFF frames, in the picture itself, where their
/// neighbours already are; otherwise each one is built in `buf` (one
/// macroblock with a border holding its left, top, top-right and corner
/// neighbours, for intra prediction) and then stored into the picture, so
/// a slice touches the picture only in load_border and store.
pub(crate) struct Pix<'a, T: Pixel = u8> {
    pic: &'a SharedPic<'a, T>,
    /// The slice's macroblocks.
    first: usize,
    end: usize,
    /// Macroblocks are addressed in pairs (MBAFF frame).
    mbaff: bool,
    /// The current macroblock is a field macroblock (MBAFF).
    cur_field: bool,
    buf: Picture,
    /// Predictions of the two lists before they are combined (the last two
    /// for Cr when Cb and Cr are predicted together).
    mc: Box<[[T; inter::MAX * inter::MAX]; 4]>,
    /// Reconstructing in the picture (see Pix).
    direct: bool,
    /// Per plane, the current macroblock's window of the picture (direct):
    /// its start and length, and the macroblock's offset in it.
    win: [(usize, usize, usize); 3],
    /// The last macroblock begun and its column and row (outside MBAFF):
    /// saves dividing again.
    at: (usize, usize, usize),
}

impl<'a, T: Pixel> Pix<'a, T> {
    fn new(
        pic: &'a SharedPic<'a, T>,
        first: usize,
        end: usize,
        mbaff: bool,
        exclusive: bool,
    ) -> Self {
        let mut planes = vec![Plane::with_depth(1 + 16 + 8, 1 + 16, T::HIGH)];
        let (cw, ch) = pic.mb_c;
        if pic.planes.len() == 3 {
            // 4:4:4 chroma is predicted like luma, with top-right samples.
            let right = if cw == 16 { 8 } else { 0 };
            for _ in 0..2 {
                planes.push(Plane::with_depth(1 + cw + right, 1 + ch, T::HIGH));
            }
        }
        let buf = Picture {
            width_mbs: 1,
            height_mbs: 1,
            planes,
            mb_c: pic.mb_c,
            bit_depth: pic.bit_depth,
        };
        Self {
            pic,
            first,
            end,
            mbaff,
            cur_field: false,
            buf,
            mc: Box::new([[T::default(); inter::MAX * inter::MAX]; 4]),
            direct: exclusive && !mbaff,
            win: [(0, 0, 0); 3],
            at: (NONE, 0, 0),
        }
    }

    /// Macroblock `addr` is next (direct mode: its windows of the picture,
    /// from its top-left neighbour, when there is one, to its end).
    #[inline]
    fn begin(&mut self, addr: usize) {
        let w = self.pic.width_mbs;
        let (mx, my) = (addr % w, addr / w);
        self.at = (addr, mx, my);
        if !self.direct {
            return;
        }
        for (p, &(_, len, stride)) in self.pic.planes.iter().enumerate() {
            let (mw, mh) = if p == 0 { (16, 16) } else { self.pic.mb_c };
            let (x0, y0) = (mx * mw, my * mh);
            let (r0, c0) = (y0 - (y0 > 0) as usize, x0 - (x0 > 0) as usize);
            let (start, end) = (r0 * stride + c0, (y0 + mh - 1) * stride + x0 + mw);
            assert!(end <= len);
            self.win[p] = (start, end - start, (y0 - r0) * stride + (x0 - c0));
        }
    }

    /// Plane `p` of the current macroblock and the offset of its top-left
    /// sample.
    #[inline(always)]
    fn win(&mut self, p: usize) -> (Win<'_, T>, usize) {
        self.win_mc(p).0
    }

    /// [`Self::win`], and the buffers of predictions to combine.
    #[inline(always)]
    #[allow(clippy::type_complexity)]
    fn win_mc(
        &mut self,
        p: usize,
    ) -> ((Win<'_, T>, usize), &mut [[T; inter::MAX * inter::MAX]; 4]) {
        if self.direct {
            let (start, len, base) = self.win[p];
            let (ptr, _, stride) = self.pic.planes[p];
            // SAFETY: the window lies in the plane (begin), and in direct
            // mode no other thread touches its rows (SliceArea::exclusive).
            let data = unsafe { std::slice::from_raw_parts_mut(ptr.add(start), len) };
            ((Win { data, stride }, base), &mut self.mc)
        } else {
            let pl = &mut self.buf.planes[p];
            let stride = pl.stride;
            (
                (
                    Win {
                        data: T::of_mut(pl),
                        stride,
                    },
                    stride + 1,
                ),
                &mut self.mc,
            )
        }
    }

    /// [`Self::win_mc`] of both chroma planes.
    #[inline(always)]
    #[allow(clippy::type_complexity)]
    fn win_mc2(
        &mut self,
    ) -> (
        [(Win<'_, T>, usize); 2],
        &mut [[T; inter::MAX * inter::MAX]; 4],
    ) {
        if self.direct {
            let w = [1, 2].map(|p| {
                let (start, len, base) = self.win[p];
                let (ptr, _, stride) = self.pic.planes[p];
                // SAFETY: as in win_mc; the two planes are separate buffers.
                let data = unsafe { std::slice::from_raw_parts_mut(ptr.add(start), len) };
                (Win { data, stride }, base)
            });
            (w, &mut self.mc)
        } else {
            let [_, cb, cr] = &mut self.buf.planes[..] else {
                unreachable!()
            };
            let w = [cb, cr].map(|pl| {
                let stride = pl.stride;
                (
                    Win {
                        data: T::of_mut(pl),
                        stride,
                    },
                    stride + 1,
                )
            });
            (w, &mut self.mc)
        }
    }

    /// Top-left sample of macroblock `addr` in plane `p` and the distance
    /// between its rows, in picture rows (2 for field macroblocks).
    fn mb_origin(&self, addr: usize, field: bool, p: usize) -> (usize, usize, usize) {
        let w = self.pic.width_mbs;
        let (mw, mh) = if p == 0 { (16, 16) } else { self.pic.mb_c };
        if !self.mbaff {
            let (x, y) = self.mb_xy(addr);
            return (x * mw, y * mh, 1);
        }
        let (px, py, bottom) = (addr / 2 % w, addr / 2 / w, addr % 2);
        if field {
            (px * mw, py * 2 * mh + bottom, 2)
        } else {
            (px * mw, py * 2 * mh + bottom * mh, 1)
        }
    }

    fn width_mbs(&self) -> usize {
        self.pic.width_mbs
    }

    /// Column and row of macroblock `addr` outside MBAFF frames.
    #[inline(always)]
    fn mb_xy(&self, addr: usize) -> (usize, usize) {
        if addr == self.at.0 {
            return (self.at.1, self.at.2);
        }
        let w = self.pic.width_mbs;
        (addr % w, addr / w)
    }

    /// Copies `out.len()` samples at `off` of picture plane `p` that belong
    /// to macroblock `mb`, which must be one this slice has stored.
    fn read(&self, mb: usize, p: usize, off: usize, out: &mut [T]) {
        let (ptr, len, _) = self.pic.planes[p];
        assert!(mb >= self.first && mb < self.end && off + out.len() <= len);
        // SAFETY: in bounds; macroblock `mb` belongs to this slice (see SharedPic).
        unsafe { std::ptr::copy_nonoverlapping(ptr.add(off), out.as_mut_ptr(), out.len()) }
    }

    /// Copies `h` samples down the column at `off` of picture plane `p`,
    /// within macroblock `mb` (which this slice has stored), to `out` at
    /// steps of `step`.
    fn read_column(&self, mb: usize, p: usize, off: usize, h: usize, out: &mut [T], step: usize) {
        let (ptr, len, stride) = self.pic.planes[p];
        assert!(mb >= self.first && mb < self.end && h > 0);
        assert!(off + (h - 1) * stride < len && (h - 1) * step < out.len());
        for r in 0..h {
            // SAFETY: in bounds (checked above); macroblock `mb` belongs to
            // this slice (see SharedPic).
            unsafe { *out.get_unchecked_mut(r * step) = ptr.add(off + r * stride).read() }
        }
    }

    /// Writes `src` at `off` of picture plane `p`, within macroblock `mb`
    /// of this slice.
    fn write(&self, mb: usize, p: usize, off: usize, src: &[T]) {
        let (ptr, len, _) = self.pic.planes[p];
        assert!(mb >= self.first && mb < self.end && off + src.len() <= len);
        // SAFETY: in bounds; macroblock `mb` belongs to this slice (see SharedPic).
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), ptr.add(off), src.len()) }
    }

    /// Copies the neighbouring samples that intra prediction may use: those
    /// of available macroblocks (in this slice, already decoded).
    fn load_border(&mut self, mbs: &Mbs, addr: usize) {
        if self.direct {
            return; // the neighbours are in place
        }
        if self.mbaff {
            return self.load_border_mbaff(mbs, addr);
        }
        let w_mbs = self.pic.width_mbs;
        let (mx, my) = (addr % w_mbs, addr / w_mbs);
        let (cw, ch) = self.pic.mb_c;
        for p in 0..self.buf.planes.len() {
            let (w, h, right) = if p == 0 || cw == 16 {
                (16, 16, 8)
            } else {
                (cw, ch, 0)
            };
            let stride = self.pic.planes[p].2;
            let (x0, y0) = (mx * w, my * h);
            let mut dst = std::mem::take(T::vec_mut(&mut self.buf.planes[p]));
            let ds = self.buf.planes[p].stride;
            if let Some(b) = mbs.b(addr) {
                self.read(b, p, (y0 - 1) * stride + x0, &mut dst[1..1 + w]);
            }
            if right > 0
                && let Some(c) = mbs.c(addr)
            {
                self.read(
                    c,
                    p,
                    (y0 - 1) * stride + x0 + w,
                    &mut dst[1 + w..1 + w + right],
                );
            }
            if let Some(d) = mbs.d(addr) {
                self.read(d, p, (y0 - 1) * stride + x0 - 1, &mut dst[..1]);
            }
            if let Some(a) = mbs.a(addr) {
                self.read_column(a, p, y0 * stride + x0 - 1, h, &mut dst[ds..], ds);
            }
            *T::vec_mut(&mut self.buf.planes[p]) = dst;
        }
    }

    /// load_border for MBAFF frames: each neighbouring sample is located
    /// with Table 6-4, as the left column may alternate between macroblocks.
    fn load_border_mbaff(&mut self, mbs: &Mbs, addr: usize) {
        let (cw, ch) = self.pic.mb_c;
        for p in 0..self.buf.planes.len() {
            let (w, h, right) = if p == 0 || cw == 16 {
                (16, 16, 8)
            } else {
                (cw, ch, 0)
            };
            let stride = self.pic.planes[p].2;
            let mut dst = std::mem::take(T::vec_mut(&mut self.buf.planes[p]));
            let ds = self.buf.planes[p].stride;
            let mut fetch = |x: i32, y: i32, at: usize| {
                let n = if p == 0 || cw == 16 {
                    mbs.locate(addr, x, y)
                } else {
                    mbs.locate_c(addr, x, y, w as i32, h as i32)
                };
                if let Some((n, xw, yw)) = n {
                    let (x0, y0, step) = self.mb_origin(n, mbs.is_field(n, addr), p);
                    self.read(
                        n,
                        p,
                        (y0 + yw * step) * stride + x0 + xw,
                        &mut dst[at..at + 1],
                    );
                }
            };
            for x in -1..(w + right) as i32 {
                fetch(x, -1, (x + 1) as usize);
            }
            for y in 0..h as i32 {
                fetch(-1, y, (y as usize + 1) * ds);
            }
            *T::vec_mut(&mut self.buf.planes[p]) = dst;
        }
    }

    /// Stores the reconstructed macroblock into the picture.
    fn store(&mut self, addr: usize) {
        if self.direct {
            return; // reconstructed in place
        }
        if self.mbaff {
            return self.store_mbaff(addr);
        }
        let w_mbs = self.pic.width_mbs;
        let (mx, my) = (addr % w_mbs, addr / w_mbs);
        let (cw, ch) = self.pic.mb_c;
        for p in 0..self.buf.planes.len() {
            let (w, h) = if p == 0 { (16, 16) } else { (cw, ch) };
            match w {
                16 => self.store_plane::<16>(addr, p, mx, my, h),
                _ => self.store_plane::<8>(addr, p, mx, my, h),
            }
        }
    }

    fn store_mbaff(&mut self, addr: usize) {
        let field = self.cur_field;
        let (cw, ch) = self.pic.mb_c;
        for p in 0..self.buf.planes.len() {
            let (w, h) = if p == 0 { (16, 16) } else { (cw, ch) };
            let (x0, y0, step) = self.mb_origin(addr, field, p);
            let stride = self.pic.planes[p].2;
            let src = &self.buf.planes[p];
            for r in 0..h {
                let so = (r + 1) * src.stride + 1;
                self.write(
                    addr,
                    p,
                    (y0 + r * step) * stride + x0,
                    &T::of(src)[so..so + w],
                );
            }
        }
    }

    #[inline(always)]
    fn store_plane<const W: usize>(&self, addr: usize, p: usize, mx: usize, my: usize, h: usize) {
        let (ptr, len, stride) = self.pic.planes[p];
        let src = &self.buf.planes[p];
        let (ss, src) = (src.stride, T::of(src));
        let off = my * h * stride + mx * W;
        // One check for the whole block, in both buffers.
        assert!(addr >= self.first && addr < self.end && h > 0);
        assert!(off + (h - 1) * stride + W <= len && h * ss + 1 + W <= src.len());
        for r in 0..h {
            // SAFETY: in bounds (checked above); macroblock `addr` belongs
            // to this slice (see SharedPic).
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.as_ptr().add((r + 1) * ss + 1),
                    ptr.add(off + r * stride),
                    W,
                )
            }
        }
    }
}

/// Coefficient levels of one macroblock, in scan order (levelListIdx).
pub(crate) struct Residual {
    luma: [[i32; 16]; 16],
    luma8: [[i32; 64]; 4],
    luma_dc: [i32; 16],
    chroma_dc: [[i32; 8]; 2],
    chroma_ac: [[[i32; 16]; 8]; 2],
    /// 4:4:4: Cb and Cr, coded like luma.
    c444: Option<Box<[LumaLevels; 2]>>,
}

impl Residual {
    fn new() -> Self {
        Self {
            luma: [[0; 16]; 16],
            luma8: [[0; 64]; 4],
            luma_dc: [0; 16],
            chroma_dc: [[0; 8]; 2],
            chroma_ac: [[[0; 16]; 8]; 2],
            c444: None,
        }
    }

    /// Zeroes what macroblock `mi` wrote: the blocks its coded_block_flag
    /// bits mark, which are the only ones the residual decoding writes.
    /// `kept` says the levels stayed levels (levels_kept): the transforms,
    /// which zero the blocks they read, did not see them.
    fn clear(&mut self, mi: &MbInfo, kept: bool) {
        self.chroma_dc = [[0; 8]; 2];
        let planes = if self.c444.is_some() { 3 } else { 1 };
        for plane in 0..planes {
            let lv = self.plane_mut(plane);
            *lv.luma_dc = [0; 16];
            if !kept {
                continue;
            }
            // Levels go to the 8x8 or the 4x4 blocks, by transform size.
            for b8 in 0..4 {
                if mi.transform_8x8 {
                    if mi.cbf >> cbf_bit(plane, 4 * b8) & 0xf != 0 {
                        lv.luma8[b8] = [0; 64];
                    }
                    continue;
                }
                for k in 4 * b8..4 * b8 + 4 {
                    if mi.cbf >> cbf_bit(plane, k) & 1 != 0 {
                        lv.luma[k] = [0; 16];
                    }
                }
            }
        }
        for comp in 0..2 {
            for blk in 0..8 {
                if kept && mi.cbf >> (CBF_CHROMA_AC + 16 * comp + blk) & 1 != 0 {
                    self.chroma_ac[comp as usize][blk as usize] = [0; 16];
                }
            }
        }
    }

    /// The levels of a plane coded like luma: 0 for luma, 1 and 2 for Cb
    /// and Cr in 4:4:4.
    fn plane_mut(&mut self, plane: usize) -> LumaLevelsMut<'_> {
        match plane {
            0 => LumaLevelsMut {
                luma: &mut self.luma,
                luma8: &mut self.luma8,
                luma_dc: &mut self.luma_dc,
            },
            _ => {
                let c = &mut self.c444.as_mut().expect("4:4:4 levels")[plane - 1];
                LumaLevelsMut {
                    luma: &mut c.luma,
                    luma8: &mut c.luma8,
                    luma_dc: &mut c.luma_dc,
                }
            }
        }
    }
}

/// Levels of a colour component coded like luma (Cb and Cr in 4:4:4).
pub(crate) struct LumaLevels {
    luma: [[i32; 16]; 16],
    luma8: [[i32; 64]; 4],
    luma_dc: [i32; 16],
}

impl LumaLevels {
    fn new() -> Self {
        Self {
            luma: [[0; 16]; 16],
            luma8: [[0; 64]; 4],
            luma_dc: [0; 16],
        }
    }

    fn view_mut(&mut self) -> LumaLevelsMut<'_> {
        LumaLevelsMut {
            luma: &mut self.luma,
            luma8: &mut self.luma8,
            luma_dc: &mut self.luma_dc,
        }
    }
}

/// The levels of one plane coded like luma.
struct LumaLevelsMut<'a> {
    luma: &'a mut [[i32; 16]; 16],
    luma8: &'a mut [[i32; 64]; 4],
    luma_dc: &'a mut [i32; 16],
}

/// Per-slice running state.
struct State {
    qp: i32,
    prev_dq: bool,
    /// Kept zeroed between macroblocks (see Residual::clear), so it is
    /// not cleared whole for each.
    res: Option<Box<Residual>>,
}

/// Macroblock-layer syntax elements, read with CABAC or CAVLC. Context
/// selection needs the neighbourhood, so the readers get it passed in.
pub(crate) trait Syntax {
    /// pcm_alignment_zero_bits and `n` raw 8-bit samples (7.3.5).
    /// pcm_sample_luma and pcm_sample_chroma after pcm_alignment_zero_bit:
    /// `n` samples, the first `n_luma` of `bits[0]` bits, then of `bits[1]`.
    fn pcm(&mut self, n: usize, n_luma: usize, bits: [u32; 2]) -> Result<Vec<u16>, Error>;
    fn transform_8x8(&mut self, mbs: &Mbs, addr: usize) -> bool;
    /// prev_intra*_pred_mode_flag / rem_intra*_pred_mode: None when predicted.
    fn intra_pred_mode(&mut self) -> Option<u8>;
    fn intra_chroma_pred_mode(&mut self, mbs: &Mbs, addr: usize) -> u8;
    /// (CodedBlockPatternLuma, CodedBlockPatternChroma).
    fn coded_block_pattern(
        &mut self,
        s: &SliceCtx,
        mbs: &Mbs,
        addr: usize,
        mi: &MbInfo,
    ) -> Result<(u8, u8), Error>;
    fn mb_qp_delta(&mut self, prev_nonzero: bool) -> i32;
    fn ref_idx(
        &mut self,
        mbs: &Mbs,
        addr: usize,
        mi: &MbInfo,
        list: usize,
        x: usize,
        y: usize,
        num_active: u32,
    ) -> u32;
    /// Both components of mvd_lX for the partition at (x, y).
    fn mvd_pair(
        &mut self,
        mbs: &Mbs,
        addr: usize,
        mi: &MbWork,
        list: usize,
        x: usize,
        y: usize,
    ) -> [i32; 2];
    fn sub_mb_type(&mut self, b_slice: bool) -> u32;
    /// residual( 0, 15 ): fills `res` and the coded_block_flag / nnz state of `mi`.
    fn residual(
        &mut self,
        s: &SliceCtx,
        mbs: &Mbs,
        addr: usize,
        mi: &mut MbWork,
        res: &mut Residual,
    ) -> Result<(), Error>;
}

/// CABAC reader: the arithmetic decoder and its contexts.
pub(crate) struct Cabac<'a> {
    rbsp: &'a [u8],
    d: Decoder<'a>,
    /// From and back to the thread's spare (SPARE_CONTEXTS).
    c: std::mem::ManuallyDrop<Box<se::Contexts>>,
}

// Buffers every slice needs, kept per thread between slices: allocating
// them for each slice costs page faults and system calls where the
// allocator returns them to the system.
thread_local! {
    static SPARE_CONTEXTS: std::cell::Cell<Option<Box<se::Contexts>>> =
        const { std::cell::Cell::new(None) };
    static SPARE_MB_CTX: std::cell::Cell<Vec<MbCtx>> = const { std::cell::Cell::new(Vec::new()) };
}

impl Drop for Cabac<'_> {
    fn drop(&mut self) {
        // SAFETY: `c` is not used after this.
        let c = unsafe { std::mem::ManuallyDrop::take(&mut self.c) };
        SPARE_CONTEXTS.set(Some(c));
    }
}

/// At least `n` default MbCtx, a power of two (indexed by address bits),
/// in the thread's spare buffer when there is one.
fn mb_ctx_ring(n: usize) -> Vec<MbCtx> {
    let mut v = SPARE_MB_CTX.take();
    v.clear();
    v.resize(n.next_power_of_two(), MbCtx::default());
    v
}

impl Drop for Mbs<'_> {
    fn drop(&mut self) {
        SPARE_MB_CTX.set(std::mem::take(&mut self.ctx));
    }
}

impl Syntax for Cabac<'_> {
    fn pcm(&mut self, n: usize, n_luma: usize, bits: [u32; 2]) -> Result<Vec<u16>, Error> {
        let pos = self.d.bit_pos().div_ceil(8);
        let (v, len) = pcm_samples(self.rbsp.get(pos..).unwrap_or_default(), n, n_luma, bits)?;
        // 9.3.1.2: the engine restarts after the samples.
        self.d = Decoder::new(self.rbsp, pos + len);
        Ok(v)
    }

    fn transform_8x8(&mut self, mbs: &Mbs, addr: usize) -> bool {
        let cond = |n: Option<usize>| n.is_some_and(|n| mbs.info[n].transform_8x8) as usize;
        se::transform_size_8x8_flag(
            &mut self.d,
            &mut self.c,
            cond(mbs.a(addr)) + cond(mbs.b(addr)),
        )
    }

    fn intra_pred_mode(&mut self) -> Option<u8> {
        se::intra_pred_mode(&mut self.d, &mut self.c)
    }

    fn intra_chroma_pred_mode(&mut self, mbs: &Mbs, addr: usize) -> u8 {
        let cond = |n: Option<usize>| {
            // Only intra macroblocks other than I_PCM set the mode.
            n.is_some_and(|n| mbs.info[n].chroma_pred_mode != 0) as usize
        };
        se::intra_chroma_pred_mode(
            &mut self.d,
            &mut self.c,
            cond(mbs.a(addr)) + cond(mbs.b(addr)),
        )
    }

    fn coded_block_pattern(
        &mut self,
        s: &SliceCtx,
        mbs: &Mbs,
        addr: usize,
        mi: &MbInfo,
    ) -> Result<(u8, u8), Error> {
        let (a, b) = (mbs.a(addr), mbs.b(addr));
        let info = &mbs.info;
        let has_chroma = matches!(s.sps.chroma_array_type(), 1 | 2);
        if !mbs.mbaff {
            // The neighbours are macroblocks A and B, each reduced to the
            // coded_block_pattern its contexts see: I_PCM counts as all
            // coded, P_Skip and B_Skip (and an unavailable one for luma
            // and chroma respectively) as all coded and none.
            let seen = |n: Option<usize>| {
                n.map_or((15, 0), |n| {
                    let m = &info[n];
                    if m.is_pcm() {
                        (15, 2)
                    } else if m.is_skip() {
                        (0, 0)
                    } else {
                        (m.cbp_luma, m.cbp_chroma)
                    }
                })
            };
            let ((la, ca), (lb, cb)) = (seen(a), seen(b));
            // Left of 8x8 block i: block i - 1, or block i + 1 of A; above
            // it: block i - 2, or block i + 2 of B.
            let luma_inc = |i: usize, bits: u8| {
                let left = if i & 1 != 0 {
                    bits >> (i - 1)
                } else {
                    la >> (i + 1)
                };
                let top = if i >= 2 {
                    bits >> (i - 2)
                } else {
                    lb >> (i + 2)
                };
                (!left & 1) as usize + 2 * (!top & 1) as usize
            };
            let chroma_inc = [
                (ca != 0) as usize + 2 * (cb != 0) as usize,
                (ca == 2) as usize + 2 * (cb == 2) as usize,
            ];
            return Ok(se::coded_block_pattern(
                &mut self.d,
                &mut self.c,
                luma_inc,
                chroma_inc,
                has_chroma,
            ));
        }
        cbp_mbaff(&mut self.d, &mut self.c, mbs, addr, mi, has_chroma)
    }

    fn mb_qp_delta(&mut self, prev_nonzero: bool) -> i32 {
        // 9.3.3.1.1.5
        se::mb_qp_delta(&mut self.d, &mut self.c, prev_nonzero as usize)
    }

    fn ref_idx(
        &mut self,
        mbs: &Mbs,
        addr: usize,
        mi: &MbInfo,
        list: usize,
        x: usize,
        y: usize,
        _num_active: u32,
    ) -> u32 {
        let inc = ref_idx_inc(mbs, addr, mi, list, x as i32, y as i32);
        se::ref_idx(&mut self.d, &mut self.c, inc)
    }

    fn mvd_pair(
        &mut self,
        mbs: &Mbs,
        addr: usize,
        mi: &MbWork,
        list: usize,
        x: usize,
        y: usize,
    ) -> [i32; 2] {
        let inc = mvd_inc(mbs, addr, mi, list, x as i32, y as i32);
        se::mvd_pair(&mut self.d, &mut self.c, inc)
    }

    fn sub_mb_type(&mut self, b_slice: bool) -> u32 {
        if b_slice {
            se::sub_mb_type_b(&mut self.d, &mut self.c)
        } else {
            se::sub_mb_type_p(&mut self.d, &mut self.c)
        }
    }

    fn residual(
        &mut self,
        s: &SliceCtx,
        mbs: &Mbs,
        addr: usize,
        mi: &mut MbWork,
        res: &mut Residual,
    ) -> Result<(), Error> {
        cabac_residual(&mut self.d, &mut self.c, s, mbs, addr, mi, res);
        Ok(())
    }
}

/// CAVLC reader (Exp-Golomb and VLC coded syntax elements).
pub(crate) struct Cavlc<'a> {
    r: BitReader<'a>,
}

impl Cavlc<'_> {
    fn ue(&mut self) -> u32 {
        self.r.ue().unwrap_or(u32::MAX)
    }
}

/// nC for a CAVLC block (9.2.1) from the counts of the left and upper
/// neighbouring blocks (None when the neighbour is unavailable).
fn nc_from(a: Option<u8>, b: Option<u8>) -> i32 {
    match (a, b) {
        (Some(a), Some(b)) => (a as i32 + b as i32 + 1) >> 1,
        (Some(a), None) => a as i32,
        (None, Some(b)) => b as i32,
        (None, None) => 0,
    }
}

impl Syntax for Cavlc<'_> {
    fn pcm(&mut self, n: usize, n_luma: usize, bits: [u32; 2]) -> Result<Vec<u16>, Error> {
        let pos = self.r.pos().div_ceil(8);
        let data = self.r.data().get(pos..).unwrap_or_default();
        let (v, len) = pcm_samples(data, n, n_luma, bits)?;
        self.r.skip((pos + len) * 8 - self.r.pos());
        Ok(v)
    }

    fn transform_8x8(&mut self, _mbs: &Mbs, _addr: usize) -> bool {
        self.r.flag()
    }

    fn intra_pred_mode(&mut self) -> Option<u8> {
        if self.r.flag() {
            None
        } else {
            Some(self.r.u(3) as u8)
        }
    }

    fn intra_chroma_pred_mode(&mut self, _mbs: &Mbs, _addr: usize) -> u8 {
        self.ue().min(3) as u8
    }

    fn coded_block_pattern(
        &mut self,
        s: &SliceCtx,
        _mbs: &Mbs,
        _addr: usize,
        mi: &MbInfo,
    ) -> Result<(u8, u8), Error> {
        // me(v), Table 9-4.
        let code = self.ue() as usize;
        let intra = mi.is_intra();
        let v = if matches!(s.sps.chroma_array_type(), 1 | 2) {
            *if intra {
                &CBP_INTRA_420[..]
            } else {
                &CBP_INTER_420[..]
            }
            .get(code)
            .ok_or(Error::OutOfRange {
                field: "coded_block_pattern",
                value: code as i64,
            })?
        } else {
            *if intra {
                &CBP_INTRA_0_3[..]
            } else {
                &CBP_INTER_0_3[..]
            }
            .get(code)
            .ok_or(Error::OutOfRange {
                field: "coded_block_pattern",
                value: code as i64,
            })?
        };
        Ok((v & 15, v >> 4))
    }

    fn mb_qp_delta(&mut self, _prev_nonzero: bool) -> i32 {
        self.r.se().unwrap_or(i32::MAX)
    }

    fn ref_idx(
        &mut self,
        _mbs: &Mbs,
        _addr: usize,
        _mi: &MbInfo,
        _list: usize,
        _x: usize,
        _y: usize,
        num_active: u32,
    ) -> u32 {
        // te(v) with range num_ref_idx_active - 1 (9.1.2).
        if num_active == 2 {
            !self.r.flag() as u32
        } else {
            self.ue()
        }
    }

    fn mvd_pair(
        &mut self,
        _: &Mbs,
        _: usize,
        _: &MbWork,
        _: usize,
        _: usize,
        _: usize,
    ) -> [i32; 2] {
        [0, 1].map(|_| self.r.se().unwrap_or(0))
    }

    fn sub_mb_type(&mut self, _b_slice: bool) -> u32 {
        self.ue()
    }

    fn residual(
        &mut self,
        s: &SliceCtx,
        mbs: &Mbs,
        addr: usize,
        mi: &mut MbWork,
        res: &mut Residual,
    ) -> Result<(), Error> {
        self.residual_luma(s, mbs, addr, mi, 0, res.plane_mut(0))?;
        if s.sps.chroma_array_type() == 3 {
            // 4:4:4: Cb and Cr coded like luma (7.3.5.3).
            self.residual_luma(s, mbs, addr, mi, 1, res.plane_mut(1))?;
            self.residual_luma(s, mbs, addr, mi, 2, res.plane_mut(2))?;
            return Ok(());
        }
        if !matches!(s.sps.chroma_array_type(), 1 | 2) {
            return Ok(());
        }
        let num_c8x8 = if s.sps.chroma_array_type() == 1 { 1 } else { 2 };
        if mi.cbp_chroma & 3 != 0 {
            let nc = if num_c8x8 == 1 { -1 } else { -2 };
            for comp in 0..2 {
                if cavlc::residual_block(&mut self.r, nc, 4 * num_c8x8, &mut res.chroma_dc[comp])?
                    > 0
                {
                    mi.cbf |= 1 << (CBF_CHROMA_DC + comp as u32);
                }
            }
        }
        if mi.cbp_chroma & 2 != 0 {
            let rows = 2 * num_c8x8;
            let dq = [dequant(s, mi, 1), dequant(s, mi, 2)];
            let mut list = Coeffs::default();
            for comp in 0..2 {
                for blk in 0..4 * num_c8x8 {
                    let (cx, cy) = (blk % 2, blk / 2);
                    let outside = |x: i32, y: i32| {
                        mbs.locate_c(addr, x, y, 8, 4 * rows as i32)
                            .map(|(n, xw, yw)| mbs.ctx(n).nnz_c[comp][yw / 4 * 2 + xw / 4])
                    };
                    let na = if cx > 0 {
                        Some(mi.ctx.nnz_c[comp][blk - 1])
                    } else {
                        outside(-1, 4 * cy as i32)
                    };
                    let nb = if cy > 0 {
                        Some(mi.ctx.nnz_c[comp][blk - 2])
                    } else {
                        outside(4 * cx as i32, -1)
                    };
                    let n = cavlc::residual_list(&mut self.r, nc_from(na, nb), 15, &mut list)?;
                    dq[comp].put4(&mut res.chroma_ac[comp][blk], &list, 1);
                    mi.ctx.nnz_c[comp][blk] = n;
                    if n > 0 {
                        mi.cbf |= 1 << (CBF_CHROMA_AC + 16 * comp as u32 + blk as u32);
                    }
                }
            }
        }
        Ok(())
    }
}

impl Cavlc<'_> {
    /// residual_luma() (7.3.5.3.1) of plane 0 (luma), or of Cb (1) or Cr (2)
    /// in 4:4:4.
    fn residual_luma(
        &mut self,
        s: &SliceCtx,
        mbs: &Mbs,
        addr: usize,
        mi: &mut MbWork,
        plane: usize,
        lv: LumaLevelsMut,
    ) -> Result<(), Error> {
        let dq = dequant(s, mi, plane);
        let mut list = Coeffs::default();
        // nC for 4x4 block `blk` of the plane (9.2.1 with 6.4.11.4).
        let luma_nc = |mi: &MbWork, blk: usize| {
            let (bx, by) = BLK_XY[blk];
            let outside = |x: i32, y: i32| {
                mbs.locate(addr, x, y)
                    .map(|(n, xw, yw)| mbs.ctx(n).nnz_of(plane)[BLK_AT[yw / 4 * 4 + xw / 4]])
            };
            let na = if bx > 0 {
                Some(mi.ctx.nnz_of(plane)[BLK_AT[by * 4 + bx - 1]])
            } else {
                outside(-1, 4 * by as i32)
            };
            let nb = if by > 0 {
                Some(mi.ctx.nnz_of(plane)[BLK_AT[(by - 1) * 4 + bx]])
            } else {
                outside(4 * bx as i32, -1)
            };
            nc_from(na, nb)
        };
        if mi.is_i16x16() {
            let nc = luma_nc(mi, 0);
            if cavlc::residual_block(&mut self.r, nc, 16, lv.luma_dc)? > 0 {
                mi.cbf |= 1 << cbf_dc_bit(plane);
            }
        }
        for b8 in 0..4 {
            if mi.cbp_luma >> b8 & 1 == 0 {
                continue;
            }
            for i4 in 0..4 {
                let blk = 4 * b8 + i4;
                let nc = luma_nc(mi, blk);
                let ac = mi.is_i16x16();
                let n = cavlc::residual_list(&mut self.r, nc, 16 - ac as usize, &mut list)?;
                mi.ctx.nnz_of_mut(plane)[blk] = n;
                if n > 0 {
                    mi.cbf |= 1 << cbf_bit(plane, blk);
                }
                if mi.transform_8x8 {
                    dq.put8(&mut lv.luma8[b8], &list, Some(i4));
                } else {
                    dq.put4(&mut lv.luma[blk], &list, ac as usize);
                }
            }
        }
        Ok(())
    }
}

/// How the levels of a block become the coefficients reconstruction uses:
/// scaled (8.5.12.1, 8.5.13.1) at their raster position, or only placed
/// there (transform bypass, SP and SI macroblocks, which use the levels).
/// Done as the levels are decoded, for their nonzero entries only.
#[derive(Clone, Copy)]
struct Dequant<'a> {
    ls4: &'a [i32; 16],
    ls8: &'a [i32; 64],
    qp: i32,
    scale: bool,
    scan4: &'static [u8; 16],
    scan8: &'static [u8; 64],
    /// Where each scan position's coefficient goes: its raster position,
    /// transposed for 8-bit samples (see transform::add4x4_dq).
    place4: &'static [u8; 16],
    place8: &'static [u8; 64],
}

impl<'a> Dequant<'a> {
    /// `dst` as where a 4x4 block's levels go, from scan index `first`.
    #[inline(always)]
    fn placed4<'b>(&'b self, dst: &'b mut [i32; 16], first: usize) -> se::Placed<'b> {
        // (c * ls + 2^(3 - qP/6)) >> (4 - qP/6), or c * ls << (qP/6 - 4) (8-336, 8-337).
        let q = self.qp / 6;
        let k = if q >= 4 {
            (1 << (q - 4), 0, 0)
        } else {
            (1, 1 << (3 - q), (4 - q) as u32)
        };
        se::Placed {
            dst,
            scan: self.scan4,
            place: self.place4,
            ls: self.ls4,
            k: self.scale.then_some(k),
            first,
        }
    }

    /// Where 8x8 blocks' levels go.
    #[inline(always)]
    fn place8(&self) -> se::Place<'_> {
        let q = self.qp / 6;
        let k = if q >= 6 {
            (1 << (q - 6), 0, 0)
        } else {
            (1, 1 << (5 - q), (6 - q) as u32)
        };
        se::Place {
            scan: self.scan8,
            place: self.place8,
            ls: self.ls8,
            k: self.scale.then_some(k),
            first: 0,
        }
    }

    /// Where 4x4 blocks' levels go, from scan index `first`.
    #[inline(always)]
    fn place4(&self, first: usize) -> se::Place<'_> {
        let q = self.qp / 6;
        let k = if q >= 4 {
            (1 << (q - 4), 0, 0)
        } else {
            (1, 1 << (3 - q), (4 - q) as u32)
        };
        se::Place {
            scan: self.scan4,
            place: self.place4,
            ls: self.ls4,
            k: self.scale.then_some(k),
            first,
        }
    }

    /// `dst` as where an 8x8 block's levels go.
    #[inline(always)]
    fn placed8<'b>(&'b self, dst: &'b mut [i32; 64]) -> se::Placed<'b> {
        // (c * ls + 2^(5 - qP/6)) >> (6 - qP/6), or c * ls << (qP/6 - 6) (8-368, 8-369).
        let q = self.qp / 6;
        let k = if q >= 6 {
            (1 << (q - 6), 0, 0)
        } else {
            (1, 1 << (5 - q), (6 - q) as u32)
        };
        se::Placed {
            dst,
            scan: self.scan8,
            place: self.place8,
            ls: self.ls8,
            k: self.scale.then_some(k),
            first: 0,
        }
    }
}

impl Dequant<'_> {
    /// A 4x4 block whose list starts at scan position `first` (1 for AC).
    #[inline]
    fn put4(&self, dst: &mut [i32; 16], c: &Coeffs, first: usize) {
        for i in 0..c.n {
            let k = c.pos[i] as usize + first;
            let r = self.scan4[k] as usize;
            dst[self.place4[k] as usize] = if self.scale {
                transform::scale_level_4x4(c.level[i], self.ls4[r], self.qp)
            } else {
                c.level[i]
            };
        }
    }

    /// An 8x8 block, or with `interleave` = Some(i4) the 4x4 list i4 of a
    /// CAVLC 8x8 block, whose entry k is 8x8 position 4k + i4 (7.3.5.3.1).
    #[inline]
    fn put8(&self, dst: &mut [i32; 64], c: &Coeffs, interleave: Option<usize>) {
        for i in 0..c.n {
            let p = c.pos[i] as usize;
            let k = match interleave {
                Some(i4) => 4 * p + i4,
                None => p,
            };
            let r = self.scan8[k] as usize;
            dst[self.place8[k] as usize] = if self.scale {
                transform::scale_level_8x8(c.level[i], self.ls8[r], self.qp)
            } else {
                c.level[i]
            };
        }
    }
}

/// Whether the levels of macroblock `mi` are used as levels rather than
/// scaled and transformed: transform bypass (lossless), and P macroblocks
/// of SP slices and SI macroblocks (8.6).
fn levels_kept(s: &SliceCtx, mi: &MbInfo) -> bool {
    let lossless = s.sps.qpprime_y_zero_transform_bypass
        && mi.qp_y as i32 + 6 * (s.bit_depth(0) as i32 - 8) == 0;
    lossless
        || matches!(mi.mb_type, MbType::Si)
        || s.header.slice_type == SliceType::Sp && !mi.is_intra()
}

/// The Dequant of plane `plane` of macroblock `mi` (its qp_y and type set).
#[inline(always)]
fn dequant<'a>(s: &SliceCtx<'a>, mi: &MbInfo, plane: usize) -> Dequant<'a> {
    let qp_y = mi.qp_y as i32;
    let qp = if plane == 0 {
        qp_y + 6 * (s.bit_depth(0) as i32 - 8)
    } else {
        let offset = if plane == 1 {
            s.pps.chroma_qp_index_offset
        } else {
            s.ext.second_chroma_qp_index_offset
        };
        let bd_offset_c = 6 * (s.bit_depth(1) as i32 - 8);
        chroma_qp(qp_y, offset, bd_offset_c) + bd_offset_c
    };
    let intra = mi.is_intra();
    let kept = levels_kept(s, mi);
    let ycbcr = plane + s.header.colour_plane_id as usize;
    let ls: &'a LevelScale = s.ls;
    // 8-bit pictures (both depths 8) use the u8 transforms.
    let eight_bit = s.bit_depth(0) == 8 && s.bit_depth(1) == 8;
    Dequant {
        ls4: &ls.s4[ycbcr + if intra { 0 } else { 3 }][(qp.max(0) % 6) as usize],
        ls8: &ls.s8[(2 * ycbcr + !intra as usize).min(5)][(qp.max(0) % 6) as usize],
        qp,
        scale: !kept,
        scan4: s.scan4(),
        scan8: s.scan8(),
        // Levels kept as levels (lossless, SP) stay in raster order.
        place4: match (eight_bit && !kept, s.field()) {
            (false, _) => s.scan4(),
            (true, false) => &transform::ZIGZAG_4X4_T,
            (true, true) => &transform::FIELD_4X4_T,
        },
        place8: match (eight_bit && !kept, s.field()) {
            (false, _) => s.scan8(),
            (true, false) => &transform::ZIGZAG_8X8_T,
            (true, true) => &transform::FIELD_8X8_T,
        },
    }
}

/// Reads PCM samples from byte-aligned `data`: the samples and the bytes
/// they take.
fn pcm_samples(
    data: &[u8],
    n: usize,
    n_luma: usize,
    bits: [u32; 2],
) -> Result<(Vec<u16>, usize), Error> {
    let total = n_luma * bits[0] as usize + (n - n_luma) * bits[1] as usize;
    let len = total.div_ceil(8);
    if data.len() < len {
        return Err(Error::Truncated);
    }
    if bits == [8, 8] {
        return Ok((data[..n].iter().map(|&v| v as u16).collect(), n));
    }
    let mut r = BitReader::new(&data[..len]);
    let v = (0..n)
        .map(|i| r.u(bits[(i >= n_luma) as usize]) as u16)
        .collect();
    Ok((v, len))
}

/// Slice data with CAVLC (7.3.4).
/// mb_field_decoding_flag of a macroblock pair without one (7.4.4): that
/// of the left pair, else the pair above, in the same slice; else frame.
fn infer_field(mbs: &Mbs, addr: usize) -> bool {
    mbs.pair(addr, -1, 0)
        .or_else(|| mbs.pair(addr, 0, -1))
        .is_some_and(|n| mbs.info[n].field)
}

/// mb_field_decoding_flag with the ctxIdxInc of 9.3.3.1.1.2: the left and
/// above pairs that are field pairs.
fn field_flag(syn: &mut Cabac, mbs: &Mbs, addr: usize) -> bool {
    let cond = |n: Option<usize>| n.is_some_and(|n| mbs.info[n].field) as usize;
    let inc = cond(mbs.pair(addr, -1, 0)) + cond(mbs.pair(addr, 0, -1));
    se::mb_field_decoding_flag(&mut syn.d, &mut syn.c, inc)
}

fn decode_slice_cavlc<T: Pixel>(
    rbsp: &[u8],
    s: &SliceCtx,
    area: SliceArea<T>,
    slice_qp: i32,
) -> Result<(), Error> {
    let h = s.header;
    let mut r = BitReader::new(rbsp);
    r.skip(h.data_bit_offset);
    let mut syn = Cavlc { r };
    let mbaff = s.sps.mb_adaptive_frame_field && !h.field_pic;
    let first = h.first_mb_in_slice as usize * (1 + mbaff as usize);
    let mut mbs = Mbs {
        info: InfoChunk {
            data: area.info,
            first: area.info_first,
        },
        width: area.pic.width_mbs,
        slice: s.slice_num,
        first,
        fmo: s.pps.num_slice_groups > 1,
        constrained_intra: s.pps.constrained_intra_pred,
        mbaff,
        cur_field: std::cell::Cell::new(false),
        cur_si: std::cell::Cell::new(false),
        near: [NONE; 9],
        near_of: NONE,
        ctx: mb_ctx_ring(2 * area.pic.width_mbs * (1 + mbaff as usize)),
    };
    let total = area.end;
    let groups = crate::fmo::mb_to_slice_group_map(s.sps, s.pps, h);
    let next = |addr: usize| crate::fmo::next_mb_addr(groups.as_deref(), addr, total);
    let mut px = Pix::new(area.pic, first, area.end, mbaff, area.exclusive);
    let mut on_mb = area.on_mb;
    let mut addr = first;
    let mut st = State {
        qp: slice_qp,
        prev_dq: false,
        res: None,
    };
    let mut more = true;
    while more {
        let mut prev_skipped = false;
        if !h.slice_type.is_intra() {
            let run = syn.ue();
            if run as usize > total {
                return Err(Error::Invalid("mb_skip_run"));
            }
            if run > 0 {
                more = syn.r.more_rbsp_data();
            }
            for i in 0..run {
                if addr >= total {
                    return Err(Error::Invalid("slice runs past the end of the picture"));
                }
                mbs.enter(addr);
                if mbaff && addr % 2 == 0 {
                    mbs.cur_field.set(infer_field(&mbs, addr));
                    // A skipped top macroblock takes the field flag of a
                    // coded bottom one, which comes next (7.4.4).
                    if i == run - 1 && more {
                        mbs.cur_field.set(syn.r.peek(1) == 1);
                    }
                }
                if h.slice_type == SliceType::B {
                    decode_b_skip(s, &mut px, &mut mbs, addr, &mut st);
                } else {
                    decode_p_skip(s, &mut px, &mut mbs, addr, &mut st);
                }
                if let Some(f) = on_mb.as_mut() {
                    f(addr, mbs.info.data.as_slice());
                }
                addr = next(addr);
            }
            prev_skipped = run > 0;
        }
        if more {
            if addr >= total {
                return Err(Error::Invalid("slice runs past the end of the picture"));
            }
            mbs.enter(addr);
            if mbaff && (addr % 2 == 0 || prev_skipped) {
                mbs.cur_field.set(syn.r.flag());
            }
            let v = syn.ue();
            let mb_type = match h.slice_type {
                SliceType::P | SliceType::Sp => MbType::from_p(v),
                SliceType::B => MbType::from_b(v),
                SliceType::Si => MbType::from_si(v),
                SliceType::I => MbType::from_i(v),
            }
            .ok_or(Error::Invalid("mb_type"))?;
            decode_mb(&mut syn, s, &mut px, &mut mbs, addr, &mut st, mb_type)?;
            if let Some(f) = on_mb.as_mut() {
                f(addr, mbs.info.data.as_slice());
            }
            addr = next(addr);
            more = syn.r.more_rbsp_data();
        }
        if syn.r.overrun() {
            return Err(Error::Truncated);
        }
    }
    Ok(())
}

/// Decodes one slice into its area of the picture.
pub fn decode_slice<T: Pixel>(rbsp: &[u8], s: &SliceCtx, area: SliceArea<T>) -> Result<(), Error> {
    let h = s.header;
    if s.pps.entropy_coding_mode && matches!(h.slice_type, SliceType::Sp | SliceType::Si) {
        return Err(Error::Invalid("SP and SI slices use CAVLC"));
    }
    if area.pic.bit_depth != s.sps.bit_depths() {
        return Err(Error::Invalid("bit depth changed within a picture"));
    }
    if T::HIGH && matches!(s.header.slice_type, SliceType::Sp | SliceType::Si) {
        return Err(Error::Invalid("SP and SI slices are 8-bit"));
    }
    let slice_qp = 26 + s.pps.pic_init_qp_minus26 + h.slice_qp_delta;
    if !s.pps.entropy_coding_mode {
        return decode_slice_cavlc(rbsp, s, area, slice_qp);
    }
    let mut ctx = SPARE_CONTEXTS
        .take()
        .unwrap_or_else(|| Box::new([Ctx::default(); 1024]));
    init_contexts(
        &mut ctx,
        h.slice_type.is_intra(),
        h.cabac_init_idc,
        slice_qp,
    );
    let mut syn = Cabac {
        rbsp,
        d: Decoder::new(rbsp, h.data_bit_offset.div_ceil(8)),
        c: std::mem::ManuallyDrop::new(ctx),
    };
    let mbaff = s.sps.mb_adaptive_frame_field && !h.field_pic;
    let first = h.first_mb_in_slice as usize * (1 + mbaff as usize);
    let mut mbs = Mbs {
        info: InfoChunk {
            data: area.info,
            first: area.info_first,
        },
        width: area.pic.width_mbs,
        slice: s.slice_num,
        first,
        fmo: s.pps.num_slice_groups > 1,
        constrained_intra: s.pps.constrained_intra_pred,
        mbaff,
        cur_field: std::cell::Cell::new(false),
        cur_si: std::cell::Cell::new(false),
        near: [NONE; 9],
        near_of: NONE,
        ctx: mb_ctx_ring(2 * area.pic.width_mbs * (1 + mbaff as usize)),
    };
    let total = area.end;
    let groups = crate::fmo::mb_to_slice_group_map(s.sps, s.pps, h);
    let next = |addr: usize| crate::fmo::next_mb_addr(groups.as_deref(), addr, total);
    let mut px = Pix::new(area.pic, first, area.end, mbaff, area.exclusive);
    let mut on_mb = area.on_mb;
    let mut addr = first;
    let mut st = State {
        qp: slice_qp,
        prev_dq: false,
        res: None,
    };
    // MBAFF: the bottom macroblock's mb_skip_flag, read after a skipped top.
    let mut next_skip = None;
    let b_slice = h.slice_type == SliceType::B;
    loop {
        if addr >= total {
            return Err(Error::Invalid("slice runs past the end of the picture"));
        }
        mbs.enter(addr);
        if mbaff && addr % 2 == 0 {
            mbs.cur_field.set(infer_field(&mbs, addr));
        }
        let skip_flag = |syn: &mut Cabac, mbs: &Mbs, at: usize| {
            let cond = |n: Option<usize>| n.is_some_and(|n| !mbs.info[n].is_skip()) as usize;
            let inc = cond(mbs.a(at)) + cond(mbs.b(at));
            se::mb_skip_flag(&mut syn.d, &mut syn.c, b_slice, inc)
        };
        let skip = !h.slice_type.is_intra()
            && match next_skip.take() {
                Some(k) => k,
                None => skip_flag(&mut syn, &mbs, addr),
            };
        if mbaff && addr % 2 == 0 {
            if skip {
                // The bottom's mb_skip_flag and, when coded, its
                // mb_field_decoding_flag come next and decide the pair's
                // field flag (7.3.4, 7.4.4). The top counts as skipped.
                mbs.info[addr].mb_type = MbType::Skip;
                mbs.info[addr].slice = s.slice_num;
                let k = skip_flag(&mut syn, &mbs, addr + 1);
                next_skip = Some(k);
                if !k {
                    mbs.cur_field.set(field_flag(&mut syn, &mbs, addr));
                }
            } else {
                mbs.cur_field.set(field_flag(&mut syn, &mbs, addr));
            }
        }
        if skip {
            if b_slice {
                decode_b_skip(s, &mut px, &mut mbs, addr, &mut st);
            } else {
                decode_p_skip(s, &mut px, &mut mbs, addr, &mut st);
            }
        } else if h.slice_type == SliceType::P {
            let v = se::mb_type_p(&mut syn.d, &mut syn.c);
            let mb_type = MbType::from_p(v).ok_or(Error::Invalid("mb_type"))?;
            decode_mb(&mut syn, s, &mut px, &mut mbs, addr, &mut st, mb_type)?;
        } else if b_slice {
            // 9.3.3.1.1.3, ctxIdxOffset 27: neighbour not B_Skip / B_Direct_16x16.
            let cond = |n: Option<usize>| {
                n.is_some_and(|n| {
                    !matches!(mbs.info[n].mb_type, MbType::Skip | MbType::Direct16x16)
                }) as usize
            };
            let inc = cond(mbs.a(addr)) + cond(mbs.b(addr));
            let v = se::mb_type_b(&mut syn.d, &mut syn.c, inc);
            let mb_type = MbType::from_b(v).ok_or(Error::Invalid("mb_type"))?;
            decode_mb(&mut syn, s, &mut px, &mut mbs, addr, &mut st, mb_type)?;
        } else {
            let (a, b) = (mbs.a(addr), mbs.b(addr));
            // 9.3.3.1.1.3, ctxIdxOffset 3: neighbour not I_NxN.
            let cond = |n: Option<usize>| {
                n.is_some_and(|n| !matches!(mbs.info[n].mb_type, MbType::INxN)) as usize
            };
            let v = se::mb_type_i(&mut syn.d, &mut syn.c, cond(a) + cond(b));
            let mb_type = MbType::from_i(v).ok_or(Error::Invalid("mb_type"))?;
            decode_mb(&mut syn, s, &mut px, &mut mbs, addr, &mut st, mb_type)?;
        }
        if let Some(f) = on_mb.as_mut() {
            f(addr, mbs.info.data.as_slice());
        }
        // No end_of_slice_flag after the top macroblock of a pair.
        if !(mbaff && addr % 2 == 0) && syn.d.terminate() {
            break;
        }
        if syn.d.overrun() {
            return Err(Error::Truncated);
        }
        addr = next(addr);
    }
    Ok(())
}

/// Decodes the rest of a macroblock after mb_type.
/// Decodes and reconstructs one macroblock into `px`, then stores it.
fn decode_mb<T: Pixel>(
    syn: &mut impl Syntax,
    s: &SliceCtx,
    px: &mut Pix<T>,
    mbs: &mut Mbs,
    addr: usize,
    st: &mut State,
    mb_type: MbType,
) -> Result<(), Error> {
    px.begin(addr);
    let r = decode_mb_inner(syn, s, px, mbs, addr, st, mb_type);
    px.store(addr);
    r
}

fn decode_mb_inner<T: Pixel>(
    syn: &mut impl Syntax,
    s: &SliceCtx,
    px: &mut Pix<T>,
    mbs: &mut Mbs,
    addr: usize,
    st: &mut State,
    mb_type: MbType,
) -> Result<(), Error> {
    let field = mbs.cur_field.get();
    s.start_mb(addr, field);
    px.cur_field = field;
    mbs.cur_si.set(mb_type == MbType::Si);
    let mut mi = MbWork {
        info: MbInfo {
            slice: mbs.slice,
            mb_type,
            field,
            ..Default::default()
        },
        ctx: MbCtx::default(),
    };

    if mb_type == MbType::IPcm {
        // pcm_alignment_zero_bit, then raw samples (7.3.5).
        let (cw, ch) = px.buf.mb_c;
        let planes = px.buf.planes.len();
        let samples = syn.pcm(
            256 + (planes - 1) * cw * ch,
            256,
            [s.bit_depth(0), s.bit_depth(1)],
        )?;
        for p in 0..planes {
            let (w, h, from) = if p == 0 {
                (16, 16, 0)
            } else {
                (cw, ch, 256 + (p - 1) * cw * ch)
            };
            let (mut pl, base) = px.win(p);
            for row in 0..h {
                let o = base + row * pl.stride;
                for (d, &v) in pl.m()[o..o + w]
                    .iter_mut()
                    .zip(&samples[from + row * w..from + row * w + w])
                {
                    *d = T::new(v as i32);
                }
            }
        }
        // I_PCM counts as all coefficients present for neighbours (9.3.3.1.1.9)
        // and has qPp 0 for deblocking (8.7.2.2). QPY,PRED is unchanged.
        mi.cbf = u64::MAX;
        mi.ctx.nnz = [16; 16];
        mi.qp_y = 0;
        mi.cbp_luma = 15;
        mi.cbp_chroma = 2;
        mi.ctx.nnz_c = [[16; 16]; 2];
        mbs.store(addr, &mi);
        st.prev_dq = false;
        return Ok(());
    }

    let intra = mb_type.is_intra();
    let mut parts = Parts::default();
    if intra {
        if mb_type == MbType::INxN && s.ext.transform_8x8_mode {
            mi.transform_8x8 = syn.transform_8x8(mbs, addr);
        }
        if matches!(mb_type, MbType::INxN | MbType::Si) {
            if mi.transform_8x8 {
                parse_intra8x8_modes(syn, mbs, addr, &mut mi);
            } else {
                parse_intra4x4_modes(syn, mbs, addr, &mut mi);
            }
        }
        if matches!(s.sps.chroma_array_type(), 1 | 2) {
            mi.chroma_pred_mode = syn.intra_chroma_pred_mode(mbs, addr);
        }
    } else {
        parts = parse_inter_pred(syn, s, mbs, addr, &mut mi)?;
    }

    if let MbType::I16x16 {
        cbp_luma,
        cbp_chroma,
        ..
    } = mb_type
    {
        mi.cbp_luma = cbp_luma;
        mi.cbp_chroma = cbp_chroma;
    } else {
        (mi.cbp_luma, mi.cbp_chroma) = syn.coded_block_pattern(s, mbs, addr, &mi)?;
        if mi.cbp_luma > 0 && s.ext.transform_8x8_mode && !intra && parts.no_sub_8x8 {
            mi.transform_8x8 = syn.transform_8x8(mbs, addr);
        }
    }

    let mut res = st.res.take().unwrap_or_else(|| Box::new(Residual::new()));
    if s.sps.chroma_array_type() == 3 && res.c444.is_none() {
        res.c444 = Some(Box::new([LumaLevels::new(), LumaLevels::new()]));
    }
    let has_residual = mi.cbp_luma > 0 || mi.cbp_chroma > 0 || mi.is_i16x16();
    if has_residual {
        let dq = syn.mb_qp_delta(st.prev_dq);
        // QpBdOffsetY widens the range of QPY to -QpBdOffsetY..=51 (7.4.5).
        let bdo = 6 * (s.bit_depth(0) as i32 - 8);
        if !(-(26 + bdo / 2)..=25 + bdo / 2).contains(&dq) {
            return Err(Error::OutOfRange {
                field: "mb_qp_delta",
                value: dq as i64,
            });
        }
        st.prev_dq = dq != 0;
        // dq is in range: one wrap at most.
        st.qp += dq;
        if st.qp < -bdo {
            st.qp += 52 + bdo;
        } else if st.qp > 51 {
            st.qp -= 52 + bdo;
        }
        // Levels are scaled as they are decoded, with this QPY.
        mi.qp_y = st.qp as i8;
        syn.residual(s, mbs, addr, &mut mi, &mut res)?;
    } else {
        st.prev_dq = false;
    }
    mi.qp_y = st.qp as i8;
    mbs.store(addr, &mi);
    if intra {
        px.load_border(mbs, addr);
    } else {
        predict_inter(s, px, addr, &mbs.info[addr]);
    }
    // An inter macroblock without levels outside SP slices is its prediction.
    let mi = &mbs.info[addr];
    if intra || mi.cbp_luma | mi.cbp_chroma != 0 || s.header.slice_type == SliceType::Sp {
        reconstruct(s, px, mbs, addr, &mut res);
        res.clear(&mbs.info[addr], levels_kept(s, &mbs.info[addr]));
    }
    st.res = Some(res);
    Ok(())
}

/// mb_pred for Intra_4x4: prev/rem modes and the prediction of 8.3.1.1.
fn parse_intra4x4_modes(syn: &mut impl Syntax, mbs: &Mbs, addr: usize, mi: &mut MbInfo) {
    // 4x4 blocks of neighbouring macroblocks (6.4.11.4).
    let outside = |x: i32, y: i32| {
        mbs.locate(addr, x, y)
            .map(|(n, xw, yw)| (&mbs.info[n], BLK_AT[yw / 4 * 4 + xw / 4]))
    };
    for blk in 0..16 {
        let (bx, by) = BLK_XY[blk];
        let left = if bx > 0 {
            Some((&*mi, BLK_AT[by * 4 + bx - 1]))
        } else {
            outside(-1, 4 * by as i32)
        };
        let top = if by > 0 {
            Some((&*mi, BLK_AT[(by - 1) * 4 + bx]))
        } else {
            outside(4 * bx as i32, -1)
        };
        // dcPredModePredictedFlag: a neighbour is unavailable, or is inter
        // while constrained_intra_pred_flag is set.
        // An SI neighbour also counts as inter for a macroblock that is not SI.
        let usable = |n: &Option<(&MbInfo, usize)>| {
            n.is_some_and(|(m, _)| {
                !mbs.constrained_intra
                    || m.is_intra()
                        && (!matches!(m.mb_type, MbType::Si) || matches!(mi.mb_type, MbType::Si))
            })
        };
        let mode_of = |(m, k): (&MbInfo, usize)| {
            if matches!(m.mb_type, MbType::INxN | MbType::Si) {
                m.intra_modes[k]
            } else {
                2
            }
        };
        let pred = if usable(&left) && usable(&top) {
            mode_of(left.unwrap()).min(mode_of(top.unwrap()))
        } else {
            2
        };
        let mode = match syn.intra_pred_mode() {
            None => pred,
            Some(rem) if rem < pred => rem,
            Some(rem) => rem + 1,
        };
        mi.intra_modes[blk] = mode;
    }
}

/// mb_pred for Intra_8x8: prev/rem modes and the prediction of 8.3.2.1.
/// Modes are stored replicated over the four 4x4 blocks of each 8x8 block.
fn parse_intra8x8_modes(syn: &mut impl Syntax, mbs: &Mbs, addr: usize, mi: &mut MbInfo) {
    for b8 in 0..4 {
        let (x8, y8) = (b8 % 2, b8 / 2);
        // (macroblock, luma8x8BlkIdxN, n of 8-72), with neighbouring
        // macroblocks' blocks from 6.4.11.2.
        let left = if x8 > 0 {
            Some((&*mi, b8 - 1, 1))
        } else {
            mbs.locate(addr, -1, 8 * y8 as i32).map(|(n, _, yw)| {
                let m = &mbs.info[n];
                // A field neighbour of a frame macroblock's lower left block.
                let k = if mbs.mbaff && !mi.field && m.field && b8 == 2 {
                    3
                } else {
                    1
                };
                (m, yw / 8 * 2 + 1, k)
            })
        };
        let top = if y8 > 0 {
            Some((&*mi, b8 - 2, 2))
        } else {
            mbs.locate(addr, 8 * x8 as i32, -1)
                .map(|(n, xw, yw)| (&mbs.info[n], yw / 8 * 2 + xw / 8, 2))
        };
        let usable = |n: &Option<(&MbInfo, usize, usize)>| {
            n.is_some_and(|(m, _, _)| !mbs.constrained_intra || m.is_intra())
        };
        let mode_of = |(m, k, n): (&MbInfo, usize, usize)| match m.mb_type {
            MbType::INxN if m.transform_8x8 => m.intra_modes[4 * k],
            MbType::INxN => m.intra_modes[4 * k + n],
            _ => 2,
        };
        let pred = if usable(&left) && usable(&top) {
            mode_of(left.unwrap()).min(mode_of(top.unwrap()))
        } else {
            2
        };
        let mode = match syn.intra_pred_mode() {
            None => pred,
            Some(rem) if rem < pred => rem,
            Some(rem) => rem + 1,
        };
        mi.intra_modes[4 * b8..4 * b8 + 4].fill(mode);
    }
}

/// What the inter prediction syntax of a macroblock tells the rest of it.
#[derive(Default)]
struct Parts {
    /// noSubMbPartSizeLessThan8x8Flag (7.3.5).
    no_sub_8x8: bool,
}

#[derive(Clone, Copy, Default)]
struct Part {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
}

/// Motion data of a neighbouring partition (8.4.1.3.2).
#[derive(Clone, Copy)]
struct Neighbour {
    avail: bool,
    ref_idx: i8,
    mv: [i16; 2],
}

/// Motion data for luma location (x, y) relative to the current macroblock,
/// which is `cur` (partly decoded: `done` marks its finished 4x4 blocks).
#[inline(always)]
fn neighbour(
    mbs: &Mbs,
    addr: usize,
    cur: &MbInfo,
    done: u16,
    list: usize,
    x: i32,
    y: i32,
) -> Neighbour {
    if !mbs.mbaff && addr == mbs.near_of {
        // Selects rather than branches: availability inside the current
        // macroblock depends on its partitioning, which varies per MB.
        let n = mbs.near[Mbs::region(x, y, 16, 16)];
        let (nx, ny) = ((x & 15) as usize, (y & 15) as usize);
        let (b4, b8) = (nx / 4 + 4 * (ny / 4), nx / 8 + 2 * (ny / 8));
        let inside = n == addr;
        let m = if inside || n == NONE {
            cur
        } else {
            &mbs.info[n]
        };
        let avail = n != NONE && (!inside || done >> b4 & 1 != 0);
        let (r, mv) = (m.ref_idx[list][b8], m.mv[list][b4]);
        // Intra macroblocks keep reference -1 (MbInfo::default).
        let used = avail && r >= 0;
        return Neighbour {
            avail,
            ref_idx: if used { r } else { -1 },
            mv: if used { mv } else { [0, 0] },
        };
    }
    let none = Neighbour {
        avail: false,
        ref_idx: -1,
        mv: [0, 0],
    };
    let Some((n, nx, ny)) = mbs.locate(addr, x, y) else {
        return none;
    };
    let b4 = nx / 4 + 4 * (ny / 4);
    let b8 = nx / 8 + 2 * (ny / 8);
    let m = if n == addr {
        if done >> b4 & 1 == 0 {
            return none; // not yet decoded
        }
        cur
    } else {
        &mbs.info[n]
    };
    // Intra macroblocks keep reference -1 (MbInfo::default).
    if m.ref_idx[list][b8] < 0 {
        return Neighbour {
            avail: true,
            ref_idx: -1,
            mv: [0, 0],
        };
    }
    let (mut ref_idx, mut mv) = (m.ref_idx[list][b8], m.mv[list][b4]);
    // MBAFF: a neighbour of the other kind is scaled to the current
    // macroblock's units (8-214 to 8-217).
    if mbs.mbaff && n != addr && m.field != cur.field {
        if cur.field {
            mv[1] /= 2;
            ref_idx *= 2;
        } else {
            mv[1] *= 2;
            ref_idx >>= 1;
        }
    }
    Neighbour {
        avail: true,
        ref_idx,
        mv,
    }
}

/// Motion data of neighbour A, B, C or D (`k` 0 to 3: locations (-1, 0),
/// (0, -1), (16, -1), (-1, -1)) of a whole macroblock: outside MBAFF frames
/// the blocks of macroblocks A to D next to it, without locating them.
#[inline(always)]
fn neighbour16(mbs: &Mbs, addr: usize, cur: &MbInfo, list: usize, k: usize) -> Neighbour {
    const AT: [(i32, i32); 4] = [(-1, 0), (0, -1), (16, -1), (-1, -1)];
    if mbs.mbaff {
        return neighbour(mbs, addr, cur, 0, list, AT[k].0, AT[k].1);
    }
    // The 4x4 and 8x8 blocks next to the macroblock in each of them.
    const BLK: [(usize, usize); 4] = [(3, 1), (12, 2), (12, 2), (15, 3)];
    let n = match k {
        0 => mbs.a(addr),
        1 => mbs.b(addr),
        2 => mbs.c(addr),
        _ => mbs.d(addr),
    };
    let Some(n) = n else {
        return Neighbour {
            avail: false,
            ref_idx: -1,
            mv: [0, 0],
        };
    };
    let m = &mbs.info[n];
    let (b4, b8) = BLK[k];
    let ref_idx = m.ref_idx[list][b8];
    // Intra macroblocks keep reference -1.
    Neighbour {
        avail: true,
        ref_idx,
        mv: if ref_idx < 0 { [0, 0] } else { m.mv[list][b4] },
    }
}

/// Motion vector prediction for a partition (8.4.1.3). `shape` gives the
/// directional rules: Some((partition, mbPartIdx)) for 16x8 and 8x16.
fn predict_mv(
    mbs: &Mbs,
    addr: usize,
    cur: &MbInfo,
    done: u16,
    list: usize,
    p: Part,
    ref_idx: i8,
    shape: Option<(Partition, usize)>,
) -> [i16; 2] {
    if (p.x, p.y, p.w, p.h) == (0, 0, 16, 16) {
        let n = |k: usize| neighbour16(mbs, addr, cur, list, k);
        let (na, nb, mut nc) = (n(0), n(1), n(2));
        if !nc.avail {
            nc = n(3);
        }
        return median_mv(na, nb, nc, ref_idx);
    }
    let (x, y, w) = (p.x as i32, p.y as i32, p.w as i32);
    let na = neighbour(mbs, addr, cur, done, list, x - 1, y);
    let nb = neighbour(mbs, addr, cur, done, list, x, y - 1);
    let mut nc = neighbour(mbs, addr, cur, done, list, x + w, y - 1);
    if !nc.avail {
        nc = neighbour(mbs, addr, cur, done, list, x - 1, y - 1);
    }
    match shape {
        Some((Partition::P16x8, 0)) if nb.ref_idx == ref_idx => return nb.mv,
        Some((Partition::P16x8, 1)) if na.ref_idx == ref_idx => return na.mv,
        Some((Partition::P8x16, 0)) if na.ref_idx == ref_idx => return na.mv,
        Some((Partition::P8x16, 1)) if nc.ref_idx == ref_idx => return nc.mv,
        _ => {}
    }
    median_mv(na, nb, nc, ref_idx)
}

/// [`neighbour`] outside MBAFF frames, from the macroblocks around.
#[inline(always)]
fn near(ar: &Around, cur: &MbInfo, done: u16, list: usize, x: i32, y: i32) -> Neighbour {
    let (m, b4, b8) = if y < 0 {
        if x < 0 {
            (ar.0[3], 15, 3)
        } else if x < 16 {
            (ar.0[1], x as usize / 4 + 12, x as usize / 8 + 2)
        } else {
            (ar.0[2], 12, 2)
        }
    } else if x < 0 {
        (ar.0[0], 3 + 4 * (y as usize / 4), 1 + 2 * (y as usize / 8))
    } else if x < 16 {
        let b4 = x as usize / 4 + 4 * (y as usize / 4);
        let b8 = x as usize / 8 + 2 * (y as usize / 8);
        ((done >> b4 & 1 != 0).then_some(cur), b4, b8)
    } else {
        (None, 0, 0)
    };
    let avail = m.is_some();
    let m = m.unwrap_or(cur);
    let (r, mv) = (m.ref_idx[list][b8], m.mv[list][b4]);
    // Intra macroblocks keep reference -1 (MbInfo::default).
    let used = avail && r >= 0;
    Neighbour {
        avail,
        ref_idx: if used { r } else { -1 },
        mv: if used { mv } else { [0, 0] },
    }
}

/// [`predict_mv`] outside MBAFF frames.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn predict_mv_near(
    ar: &Around,
    cur: &MbInfo,
    done: u16,
    list: usize,
    p: Part,
    ref_idx: i8,
    shape: Option<(Partition, usize)>,
) -> [i16; 2] {
    let (x, y, w) = (p.x as i32, p.y as i32, p.w as i32);
    let na = near(ar, cur, done, list, x - 1, y);
    let nb = near(ar, cur, done, list, x, y - 1);
    let mut nc = near(ar, cur, done, list, x + w, y - 1);
    if !nc.avail {
        nc = near(ar, cur, done, list, x - 1, y - 1);
    }
    match shape {
        Some((Partition::P16x8, 0)) if nb.ref_idx == ref_idx => return nb.mv,
        Some((Partition::P16x8, 1)) if na.ref_idx == ref_idx => return na.mv,
        Some((Partition::P8x16, 0)) if na.ref_idx == ref_idx => return na.mv,
        Some((Partition::P8x16, 1)) if nc.ref_idx == ref_idx => return nc.mv,
        _ => {}
    }
    median_mv(na, nb, nc, ref_idx)
}

/// [`predict_mv_near`] for sub-macroblock partitions, out of line.
#[inline(never)]
fn predict_mv_sub(
    ar: &Around,
    cur: &MbInfo,
    done: u16,
    list: usize,
    p: Part,
    ref_idx: i8,
) -> [i16; 2] {
    predict_mv_near(ar, cur, done, list, p, ref_idx, None)
}

/// 8.4.1.3.1.
fn median_mv(na: Neighbour, mut nb: Neighbour, mut nc: Neighbour, ref_idx: i8) -> [i16; 2] {
    if !nb.avail && !nc.avail && na.avail {
        nb = na;
        nc = na;
    }
    let (ea, eb, ec) = (
        na.ref_idx == ref_idx,
        nb.ref_idx == ref_idx,
        nc.ref_idx == ref_idx,
    );
    if ea as u8 + eb as u8 + ec as u8 == 1 {
        return if ea {
            na.mv
        } else if eb {
            nb.mv
        } else {
            nc.mv
        };
    }
    let med = |a: i16, b: i16, c: i16| a.max(b).min(a.min(b).max(c));
    [
        med(na.mv[0], nb.mv[0], nc.mv[0]),
        med(na.mv[1], nb.mv[1], nc.mv[1]),
    ]
}

/// The 4x4 blocks of partition `p`, as bits in raster order.
#[inline(always)]
fn part_bits(p: Part) -> u16 {
    let row = (1u16 << (p.w / 4)) - 1;
    (0..p.h / 4).fold(0, |b, y| b | row << (p.x / 4 + p.y + 4 * y))
}

#[inline(always)]
fn set_motion(
    mi: &mut MbInfo,
    list: usize,
    p: Part,
    ref_idx: i8,
    ref_id: u32,
    mv: [i16; 2],
    done: &mut u16,
) {
    // Fixed shapes, so the loops unroll into stores.
    #[inline(always)]
    fn rect<const W: usize, const H: usize>(
        m: &mut [[i16; 2]; 16],
        at: usize,
        mv: [i16; 2],
    ) -> u16 {
        let mut bits = 0;
        for y in 0..H {
            for x in 0..W {
                m[at + 4 * y + x] = mv;
                bits |= 1 << (at + 4 * y + x);
            }
        }
        bits
    }
    let at = p.x / 4 + p.y;
    let m = &mut mi.mv[list];
    *done |= match (p.w, p.h) {
        (16, 16) => rect::<4, 4>(m, 0, mv),
        (16, 8) => rect::<4, 2>(m, at, mv),
        (8, 16) => rect::<2, 4>(m, at, mv),
        (8, 8) => rect::<2, 2>(m, at, mv),
        (8, 4) => rect::<2, 1>(m, at, mv),
        (4, 8) => rect::<1, 2>(m, at, mv),
        _ => rect::<1, 1>(m, at, mv),
    };
    // The 8x8 blocks covered, also of fixed shapes.
    let at8 = p.x / 8 + 2 * (p.y / 8);
    let (r, id) = (&mut mi.ref_idx[list], &mut mi.ref_id[list]);
    match (p.w, p.h) {
        (16, 16) => (*r, *id) = ([ref_idx; 4], [ref_id; 4]),
        (16, 8) => {
            let h = at8 & 2;
            (r[h], r[h + 1], id[h], id[h + 1]) = (ref_idx, ref_idx, ref_id, ref_id);
        }
        (8, 16) => {
            let c = at8 & 1;
            (r[c], r[c + 2], id[c], id[c + 2]) = (ref_idx, ref_idx, ref_id, ref_id);
        }
        _ => (r[at8 & 3], id[at8 & 3]) = (ref_idx, ref_id),
    }
}

/// The reference index of partition `p` alone, for the contexts of the
/// partitions parsed after it (its motion is set once it is known).
#[inline(always)]
fn set_ref(mi: &mut MbInfo, list: usize, p: Part, ref_idx: i8) {
    let (at8, r) = (p.x / 8 + 2 * (p.y / 8), &mut mi.ref_idx[list]);
    match (p.w, p.h) {
        (16, 16) => *r = [ref_idx; 4],
        (16, 8) => (r[at8 & 2], r[(at8 & 2) + 1]) = (ref_idx, ref_idx),
        (8, 16) => (r[at8 & 1], r[(at8 & 1) + 2]) = (ref_idx, ref_idx),
        _ => r[at8 & 3] = ref_idx,
    }
}

fn ref_id(s: &SliceCtx, list: usize, ref_idx: i8) -> u32 {
    s.list(list)
        .get(ref_idx as usize)
        .and_then(|r| r.as_ref())
        .map_or(NO_REF, |r| r.id)
}

/// P_Skip (8.4.1.1): 16x16 L0 prediction with ref_idx 0, no residual.
fn decode_p_skip<T: Pixel>(
    s: &SliceCtx,
    px: &mut Pix<T>,
    mbs: &mut Mbs,
    addr: usize,
    st: &mut State,
) {
    px.begin(addr);
    let field = mbs.cur_field.get();
    s.start_mb(addr, field);
    px.cur_field = field;
    let mut mi = MbInfo {
        slice: mbs.slice,
        mb_type: MbType::Skip,
        qp_y: st.qp as i8,
        field,
        ..Default::default()
    };
    let p = Part {
        x: 0,
        y: 0,
        w: 16,
        h: 16,
    };
    let na = neighbour16(mbs, addr, &mi, 0, 0);
    let nb = neighbour16(mbs, addr, &mi, 0, 1);
    let zero = mbs.a(addr).is_none()
        || mbs.b(addr).is_none()
        || (na.ref_idx == 0 && na.mv == [0, 0])
        || (nb.ref_idx == 0 && nb.mv == [0, 0]);
    let mv = if zero {
        [0, 0]
    } else {
        predict_mv(mbs, addr, &mi, 0, 0, p, 0, None)
    };
    let mut done = 0;
    set_motion(&mut mi, 0, p, 0, ref_id(s, 0, 0), mv, &mut done);
    st.prev_dq = false;
    mbs.store(
        addr,
        &MbWork {
            info: mi,
            ctx: MbCtx::default(),
        },
    );
    predict_inter(s, px, addr, &mbs.info[addr]);
    if s.header.slice_type == SliceType::Sp {
        // Still transformed and requantised, with no levels (8.6).
        reconstruct(s, px, mbs, addr, &mut Residual::new());
    }
    px.store(addr);
}

/// B_Skip: direct prediction of the whole macroblock, no residual.
fn decode_b_skip<T: Pixel>(
    s: &SliceCtx,
    px: &mut Pix<T>,
    mbs: &mut Mbs,
    addr: usize,
    st: &mut State,
) {
    px.begin(addr);
    let field = mbs.cur_field.get();
    s.start_mb(addr, field);
    px.cur_field = field;
    let mut mi = MbInfo {
        slice: mbs.slice,
        mb_type: MbType::Skip,
        qp_y: st.qp as i8,
        field,
        ..Default::default()
    };
    let spatial = SpatialDirect::new(s, mbs, addr, &mi);
    direct_mb(s, addr, &mut mi, &spatial);
    st.prev_dq = false;
    mbs.store(
        addr,
        &MbWork {
            info: mi,
            ctx: MbCtx::default(),
        },
    );
    predict_inter(s, px, addr, &mbs.info[addr]);
    px.store(addr);
}

/// coded_block_pattern in MBAFF frames (out of line: rare).
#[cold]
#[inline(never)]
fn cbp_mbaff(
    d: &mut Decoder,
    c: &mut se::Contexts,
    mbs: &Mbs,
    addr: usize,
    mi: &MbInfo,
    has_chroma: bool,
) -> Result<(u8, u8), Error> {
    let (a, b) = (mbs.a(addr), mbs.b(addr));
    let info = &mbs.info;
    // 9.3.3.1.1.4: the inc grows when the neighbouring 8x8 block has no residual.
    let luma_inc = |i: usize, bits: u8| {
        let (x8, y8) = (i % 2, i / 2);
        let cond = |n: Option<(&MbInfo, usize, bool)>| match n {
            None => 0,
            Some((m, k, cur)) => {
                if cur {
                    (bits >> k & 1 == 0) as usize
                } else if m.is_pcm() {
                    0
                } else {
                    (m.is_skip() || m.cbp_luma >> k & 1 == 0) as usize
                }
            }
        };
        // 8x8 blocks of neighbouring macroblocks (6.4.11.2).
        let outside = |x: i32, y: i32| {
            mbs.locate(addr, x, y)
                .map(|(n, xw, yw)| (&info[n], yw / 8 * 2 + xw / 8, false))
        };
        let left = if x8 > 0 {
            Some((mi, i - 1, true))
        } else {
            outside(-1, 8 * y8 as i32)
        };
        let top = if y8 > 0 {
            Some((mi, i - 2, true))
        } else {
            outside(8 * x8 as i32, -1)
        };
        cond(left) + 2 * cond(top)
    };
    let ccond = |n: Option<usize>, bin: usize| {
        n.is_some_and(|n| {
            let m = &info[n];
            m.is_pcm()
                || (!m.is_skip()
                    && if bin == 0 {
                        m.cbp_chroma != 0
                    } else {
                        m.cbp_chroma == 2
                    })
        }) as usize
    };
    let chroma_inc = [ccond(a, 0) + 2 * ccond(b, 0), ccond(a, 1) + 2 * ccond(b, 1)];
    Ok(se::coded_block_pattern(
        d, c, luma_inc, chroma_inc, has_chroma,
    ))
}

/// ref_idx_lX context (9.3.3.1.1.6) for the partition at (x, y).
#[inline(always)]
fn ref_idx_inc(mbs: &Mbs, addr: usize, cur: &MbInfo, list: usize, x: i32, y: i32) -> usize {
    if !mbs.mbaff {
        // The 8x8 blocks left of and above (x, y): in q, or the edge
        // blocks of A and B (of the same kind: no field limit).
        let used = |m: &MbInfo, b8: usize| {
            !m.is_skip() && m.direct8x8 >> b8 & 1 == 0 && m.ref_idx[list][b8] > 0
        };
        let (x8, y8) = (x as usize / 8, y as usize / 8);
        let edge = |n: Option<usize>, b8: usize| n.is_some_and(|n| used(&mbs.info[n], b8));
        let a = if x8 > 0 {
            used(cur, 2 * y8)
        } else {
            edge(mbs.a(addr), 1 + 2 * y8)
        };
        let b = if y8 > 0 {
            used(cur, x8)
        } else {
            edge(mbs.b(addr), x8 + 2)
        };
        return a as usize + 2 * b as usize;
    }
    ref_idx_inc_mbaff(mbs, addr, cur, list, x, y)
}

/// [`ref_idx_inc`] in MBAFF frames (out of line: rare).
#[cold]
#[inline(never)]
fn ref_idx_inc_mbaff(mbs: &Mbs, addr: usize, cur: &MbInfo, list: usize, x: i32, y: i32) -> usize {
    let cond = |xn: i32, yn: i32| {
        let Some((n, nx, ny)) = mbs.locate(addr, xn, yn) else {
            return 0;
        };
        let m = if n == addr { cur } else { &mbs.info[n] };
        let b8 = nx / 8 + 2 * (ny / 8);
        if m.is_skip() || m.is_intra() || m.direct8x8 >> b8 & 1 != 0 {
            return 0;
        }
        // A field neighbour of a frame macroblock counts in frame units.
        let limit = if !cur.field && m.field { 1 } else { 0 };
        (m.ref_idx[list][b8] > limit) as usize
    };
    cond(x - 1, y) + 2 * cond(x, y - 1)
}

/// mvd_lX context (9.3.3.1.1.7) for component `comp` at (x, y).
/// ctxIdxInc of both mvd components (9.3.3.1.1.7), from the same
/// neighbouring partitions.
#[inline(always)]
fn mvd_inc(mbs: &Mbs, addr: usize, cur: &MbWork, list: usize, x: i32, y: i32) -> [usize; 2] {
    let inc = |a: [u8; 2], b: [u8; 2]| {
        [0, 1].map(|c| match a[c] as u32 + b[c] as u32 {
            33.. => 2,
            3..=32 => 1,
            _ => 0,
        })
    };
    if !mbs.mbaff {
        // The blocks left of and above (x, y): in q, or the edge blocks of
        // macroblocks A and B, of the same kind (no vertical scaling).
        let (bx, by) = (x as usize / 4, y as usize / 4);
        let none = |n: Option<usize>, b4: usize| n.map_or([0, 0], |n| mbs.ctx(n).mvd[list][b4]);
        let a = if bx > 0 {
            cur.ctx.mvd[list][bx - 1 + 4 * by]
        } else {
            none(mbs.a(addr), 3 + 4 * by)
        };
        let b = if by > 0 {
            cur.ctx.mvd[list][bx + 4 * (by - 1)]
        } else {
            none(mbs.b(addr), bx + 12)
        };
        return inc(a, b);
    }
    mvd_inc_mbaff(mbs, addr, cur, list, x, y)
}

/// [`mvd_inc`] in MBAFF frames (out of line: rare).
#[cold]
#[inline(never)]
fn mvd_inc_mbaff(mbs: &Mbs, addr: usize, cur: &MbWork, list: usize, x: i32, y: i32) -> [usize; 2] {
    let abs = |xn: i32, yn: i32| -> [u32; 2] {
        let Some((n, nx, ny)) = mbs.locate(addr, xn, yn) else {
            return [0, 0];
        };
        let (field, mvd) = if n == addr {
            (cur.field, &cur.ctx.mvd)
        } else {
            (mbs.info[n].field, &mbs.ctx(n).mvd)
        };
        // Skipped, intra and direct blocks, and those not using the list,
        // have no mvd: theirs stays 0, as absMvdComp is taken to be.
        let [h, v] = mvd[list][nx / 4 + 4 * (ny / 4)].map(|c| c as u32);
        // MBAFF: vertical components in the current macroblock's units.
        let v = match (cur.field, field) {
            (false, true) => v * 2,
            (true, false) => v / 2,
            _ => v,
        };
        [h, v]
    };
    let (a, b) = (abs(x - 1, y), abs(x, y - 1));
    [0, 1].map(|c| match a[c] + b[c] {
        33.. => 2,
        3..=32 => 1,
        _ => 0,
    })
}

fn store_mvd(mi: &mut MbWork, list: usize, p: Part, mvd: [i32; 2]) {
    let v = [
        mvd[0].unsigned_abs().min(127) as u8,
        mvd[1].unsigned_abs().min(127) as u8,
    ];
    // Fixed shapes, so the loops unroll into stores.
    #[inline(always)]
    fn rect<const W: usize, const H: usize>(m: &mut [[u8; 2]; 16], at: usize, v: [u8; 2]) {
        for y in 0..H {
            for x in 0..W {
                m[at + 4 * y + x] = v;
            }
        }
    }
    let (at, m) = (p.x / 4 + p.y, &mut mi.ctx.mvd[list]);
    match (p.w, p.h) {
        (16, 16) => rect::<4, 4>(m, 0, v),
        (16, 8) => rect::<4, 2>(m, at, v),
        (8, 16) => rect::<2, 4>(m, at, v),
        (8, 8) => rect::<2, 2>(m, at, v),
        (8, 4) => rect::<2, 1>(m, at, v),
        (4, 8) => rect::<1, 2>(m, at, v),
        _ => rect::<1, 1>(m, at, v),
    }
}

/// Entry `idx` of reference list `list`, if present.
fn ref_poc<'a>(s: &SliceCtx<'a>, list: usize, idx: i8) -> Option<&'a RefPic> {
    s.list(list)
        .get(idx.max(0) as usize)
        .and_then(|r| r.as_ref())
}

/// DistScaleFactor (8-197, 8-198), or None when td is 0.
fn dist_scale_factor(cur_poc: i32, poc0: i32, poc1: i32) -> Option<i32> {
    let td = (poc1 - poc0).clamp(-128, 127);
    if td == 0 {
        return None;
    }
    let tb = (cur_poc - poc0).clamp(-128, 127);
    let tx = (16384 + (td / 2).abs()) / td;
    Some(((tb * tx + 32) >> 6).clamp(-1024, 1023))
}

/// Spatial direct reference indices and motion vector predictors, which are
/// shared by every direct block of a macroblock (8.4.1.2.2 steps 1 to 5).
struct SpatialDirect {
    ref_idx: [i8; 2],
    mvp: [[i16; 2]; 2],
    direct_zero: bool,
}

impl SpatialDirect {
    /// Whether colZeroFlag can change a vector: some list has reference 0
    /// and a predictor other than zero.
    #[inline(always)]
    fn col_matters(&self) -> bool {
        (0..2).any(|l| self.ref_idx[l] == 0 && self.mvp[l] != [0, 0])
    }

    fn new(s: &SliceCtx, mbs: &Mbs, addr: usize, cur: &MbInfo) -> Self {
        let mut r = Self {
            ref_idx: [-1; 2],
            mvp: [[0; 2]; 2],
            direct_zero: false,
        };
        if !s.header.direct_spatial_mv_pred {
            return r;
        }
        let min_positive = |x: i8, y: i8| if x >= 0 && y >= 0 { x.min(y) } else { x.max(y) };
        // Neighbours A, B and C (or D) per list, for the references and
        // then the predictors (predict_mv of the whole macroblock).
        let near = [0, 1].map(|list| {
            let na = neighbour16(mbs, addr, cur, list, 0);
            let nb = neighbour16(mbs, addr, cur, list, 1);
            let mut nc = neighbour16(mbs, addr, cur, list, 2);
            if !nc.avail {
                nc = neighbour16(mbs, addr, cur, list, 3);
            }
            (na, nb, nc)
        });
        for (list, &(na, nb, nc)) in near.iter().enumerate() {
            r.ref_idx[list] = min_positive(na.ref_idx, min_positive(nb.ref_idx, nc.ref_idx));
        }
        if r.ref_idx[0] < 0 && r.ref_idx[1] < 0 {
            r.ref_idx = [0, 0];
            r.direct_zero = true;
        } else {
            for (list, &(na, nb, nc)) in near.iter().enumerate() {
                if r.ref_idx[list] >= 0 {
                    r.mvp[list] = median_mv(na, nb, nc, r.ref_idx[list]);
                }
            }
        }
        r
    }
}

/// vertMvScale of 8.4.1.2.1.
#[derive(Clone, Copy, PartialEq, Eq)]
enum VertMvScale {
    OneToOne,
    FrmToFld,
    FldToFrm,
}

/// The co-located 4x4 block (8.4.1.2.1, Tables 8-6 and 8-8) of the 4x4
/// block (x4, y4) of macroblock `addr`: (refIdxCol, mvCol, id of refPicCol,
/// vertMvScale). mvCol is unscaled.
fn colocated(s: &SliceCtx, addr: usize, x4: usize, y4: usize) -> (i8, [i16; 2], u32, VertMvScale) {
    col_pick(col_row(s, addr, y4), x4)
}

/// The co-located 4x4 block in column x4 of a row found by [`col_row`].
#[inline(always)]
fn col_pick(
    row: Option<(&MbInfo, usize, VertMvScale)>,
    x4: usize,
) -> (i8, [i16; 2], u32, VertMvScale) {
    let Some((m, y_m, scale)) = row else {
        return (-1, [0, 0], NO_REF, VertMvScale::OneToOne);
    };
    let b4 = x4 + 4 * (y_m / 4);
    let b8 = x4 / 2 + 2 * (y_m / 8);
    let l = if m.ref_idx[0][b8] >= 0 { 0 } else { 1 };
    (m.ref_idx[l][b8], m.mv[l][b4], m.ref_id[l][b8], scale)
}

/// For the 4x4 blocks in row y4 of macroblock `addr`: the co-located
/// macroblock, if inter, the row within it and vertMvScale (see
/// [`colocated`]).
fn col_row<'a>(
    s: &SliceCtx<'a>,
    addr: usize,
    y4: usize,
) -> Option<(&'a MbInfo, usize, VertMvScale)> {
    use VertMvScale::*;
    // RefPicList1[0]: for a field macroblock, the field of the same
    // parity, whose co-located data is that field or the frame (Table 8-6).
    let col = ref_poc(s, 1, 0)?;
    let w = s.sps.pic_width_in_mbs as usize;
    let y_col = 4 * y4;
    // topAbsDiffPOC < bottomAbsDiffPOC (8-175, 8-176).
    let top_closer = || (col.field_poc[0] - s.poc).abs() < (col.field_poc[1] - s.poc).abs();
    let mbaff = s.sps.mb_adaptive_frame_field && !s.header.field_pic;
    let (info, mb, y_m, scale) = match (&col.col, s.header.field_pic, mbaff, s.mb_field.get()) {
        // A field picture.
        (ColInfo::Field(info), true, ..) => (info, addr, y_col, OneToOne),
        (ColInfo::Frame(info), true, ..) => (
            info,
            2 * w * (addr / w) + addr % w + w * (y_col / 8),
            (2 * y_col) % 16,
            FrmToFld,
        ),
        (ColInfo::Mbaff(info), true, ..) => {
            let x = 2 * addr;
            if info.mb(x).is_some_and(|m| m.field) {
                (info, x + s.header.bottom_field as usize, y_col, OneToOne)
            } else {
                (info, x + y_col / 8, (2 * y_col) % 16, FrmToFld)
            }
        }
        // A frame without MBAFF.
        (ColInfo::Pair { fields, .. }, false, false, _) => (
            &fields[!top_closer() as usize],
            w * (addr / (2 * w)) + addr % w,
            8 * ((addr / w) % 2) + 4 * (y_col / 8),
            FldToFrm,
        ),
        (ColInfo::Frame(info), false, false, _) => (info, addr, y_col, OneToOne),
        // An MBAFF frame, frame macroblock.
        (ColInfo::Pair { fields, .. }, false, true, None) => (
            &fields[!top_closer() as usize],
            addr / 2,
            8 * (addr % 2) + 4 * (y_col / 8),
            FldToFrm,
        ),
        (ColInfo::Mbaff(info) | ColInfo::Frame(info), false, true, None) => {
            if info.mb(addr).is_some_and(|m| m.field) {
                (
                    info,
                    2 * (addr / 2) + !top_closer() as usize,
                    8 * (addr % 2) + 4 * (y_col / 8),
                    FldToFrm,
                )
            } else {
                (info, addr, y_col, OneToOne)
            }
        }
        // An MBAFF frame, field macroblock: the field of the pair has the
        // macroblock's parity.
        (ColInfo::Field(info), false, true, Some(_)) => (info, addr / 2, y_col, OneToOne),
        (ColInfo::Mbaff(info) | ColInfo::Frame(info), false, true, Some(_)) => {
            if info.mb(addr).is_some_and(|m| m.field) {
                (info, addr, y_col, OneToOne)
            } else {
                (info, 2 * (addr / 2) + y_col / 8, (2 * y_col) % 16, FrmToFld)
            }
        }
        _ => return None,
    };
    // With separate colour planes the macroblocks of the same colour plane
    // (stored one plane after another).
    let base = match s.header.colour_plane_id as usize {
        0 => 0,
        id => id * info.len() / s.sps.colour_planes(),
    };
    let m = info.mb(base + mb)?;
    if m.is_intra() || m.slice == u32::MAX {
        return None;
    }
    Some((m, y_m, scale))
}

/// Direct prediction (8.4.1.2) of a whole macroblock (B_Skip, B_Direct_16x16).
/// Spatial with direct_8x8_inference, the 8x8 blocks differ only through
/// colZeroFlag of their corner blocks: when they agree, the motion is set
/// once for the macroblock.
fn direct_mb(s: &SliceCtx, addr: usize, mi: &mut MbInfo, spatial: &SpatialDirect) {
    let mut done = 0u16;
    if !(s.header.direct_spatial_mv_pred && s.sps.direct_8x8_inference) {
        for k in 0..4 {
            direct_8x8(s, addr, mi, k, spatial, &mut done);
        }
        return;
    }
    let refs = spatial.ref_idx;
    // colZeroFlag only changes a list with reference 0 whose predictor is
    // not already zero: otherwise the co-located picture is not read.
    let tests = !spatial.direct_zero
        && spatial.col_matters()
        && ref_poc(s, 1, 0).is_some_and(|r| !r.long_term && !r.inter_view);
    // The vectors of 8x8 block k (8.4.1.2.2): zero for a list unused, or
    // with reference 0 where the co-located corner block barely moves.
    let rows = if tests {
        [col_row(s, addr, 0), col_row(s, addr, 3)]
    } else {
        [None, None]
    };
    let mvs = |k: usize| {
        let col_zero = tests && {
            let (ref_col, mv_col, ..) = col_pick(rows[k / 2], 3 * (k % 2));
            ref_col == 0 && (-1..=1).contains(&mv_col[0]) && (-1..=1).contains(&mv_col[1])
        };
        [0, 1].map(|list| {
            let r = refs[list];
            if spatial.direct_zero || r < 0 || (r == 0 && col_zero) {
                [0, 0]
            } else {
                spatial.mvp[list]
            }
        })
    };
    let all: [[[i16; 2]; 2]; 4] = if tests {
        std::array::from_fn(mvs)
    } else {
        [mvs(0); 4]
    };
    mi.direct8x8 = 0xf;
    let ids = [0, 1].map(|l| {
        if refs[l] >= 0 {
            ref_id(s, l, refs[l])
        } else {
            NO_REF
        }
    });
    if all.iter().all(|m| *m == all[0]) {
        mi.uniform_motion = true;
        let p = Part {
            x: 0,
            y: 0,
            w: 16,
            h: 16,
        };
        for list in 0..2 {
            set_motion(mi, list, p, refs[list], ids[list], all[0][list], &mut done);
        }
        return;
    }
    for (k, m) in all.iter().enumerate() {
        let p = Part {
            x: k % 2 * 8,
            y: k / 2 * 8,
            w: 8,
            h: 8,
        };
        for list in 0..2 {
            set_motion(mi, list, p, refs[list], ids[list], m[list], &mut done);
        }
    }
}

/// Direct prediction (8.4.1.2) of 8x8 block `k`, adding its partitions.
fn direct_8x8(
    s: &SliceCtx,
    addr: usize,
    mi: &mut MbInfo,
    k: usize,
    spatial: &SpatialDirect,
    done: &mut u16,
) {
    let (x8, y8) = (k % 2, k / 2);
    let inference = s.sps.direct_8x8_inference;
    mi.direct8x8 |= 1 << k;
    // With direct_8x8_inference the corner 4x4 block (luma4x4BlkIdx 5 * k)
    // supplies the co-located motion for the whole 8x8 block.
    let all: [(usize, usize); 4] = std::array::from_fn(|j| (2 * x8 + j % 2, 2 * y8 + j / 2));
    let blocks = if inference { &all[..1] } else { &all[..] };
    for &(bx, by) in blocks {
        let (cx, cy) = if inference {
            (3 * x8, 3 * y8)
        } else {
            (bx, by)
        };
        let size = if inference { 8 } else { 4 };
        let p = Part {
            x: bx * 4,
            y: by * 4,
            w: size,
            h: size,
        };
        let (refs, mvs) = if s.header.direct_spatial_mv_pred {
            // colZeroFlag only matters to a list with reference 0.
            let col_zero = !spatial.direct_zero
                && spatial.col_matters()
                && ref_poc(s, 1, 0).is_some_and(|r| !r.long_term && !r.inter_view)
                && {
                    let (ref_col, mv_col, ..) = colocated(s, addr, cx, cy);
                    ref_col == 0 && (-1..=1).contains(&mv_col[0]) && (-1..=1).contains(&mv_col[1])
                };
            let mut mvs = [[0i16; 2]; 2];
            for list in 0..2 {
                let r = spatial.ref_idx[list];
                if !(spatial.direct_zero || r < 0 || (r == 0 && col_zero)) {
                    mvs[list] = spatial.mvp[list];
                }
            }
            (spatial.ref_idx, mvs)
        } else {
            let (ref_col, mv_col, ref_col_id, scale) = colocated(s, addr, cx, cy);
            // Temporal direct (8.4.1.2.3). MapColToList0: the same picture,
            // the field of the frame refPicCol with the current parity, or
            // the frame containing the field refPicCol.
            let target = match scale {
                VertMvScale::OneToOne => ref_col_id,
                VertMvScale::FrmToFld => {
                    let parity = match s.mb_field.get() {
                        Some(p) => p as u32,
                        None => s.header.bottom_field as u32,
                    };
                    (ref_col_id & !3) | (1 + parity)
                }
                VertMvScale::FldToFrm => ref_col_id & !3,
            };
            let ref0 = if ref_col < 0 {
                0
            } else {
                s.list(0)
                    .iter()
                    .position(|r| r.as_ref().is_some_and(|r| r.id == target))
                    .unwrap_or(0) as i8
            };
            let mv_col = match scale {
                VertMvScale::FrmToFld => [mv_col[0], mv_col[1] / 2],
                VertMvScale::FldToFrm => [mv_col[0], mv_col[1] * 2],
                VertMvScale::OneToOne => mv_col,
            };
            let (pic0, pic1) = (ref_poc(s, 0, ref0), ref_poc(s, 1, 0));
            let (poc0, poc1) = (pic0.map_or(0, |r| r.poc), pic1.map_or(0, |r| r.poc));
            let dsf = if pic0.is_some_and(|r| r.long_term) {
                None
            } else {
                dist_scale_factor(s.cur_poc(), poc0, poc1)
            };
            let mvs = match dsf {
                None => [mv_col, [0, 0]],
                Some(f) => {
                    let l0 = [
                        ((f * mv_col[0] as i32 + 128) >> 8) as i16,
                        ((f * mv_col[1] as i32 + 128) >> 8) as i16,
                    ];
                    [l0, [l0[0] - mv_col[0], l0[1] - mv_col[1]]]
                }
            };
            ([ref0, 0], mvs)
        };
        for list in 0..2 {
            set_motion(
                mi,
                list,
                p,
                refs[list],
                if refs[list] >= 0 {
                    ref_id(s, list, refs[list])
                } else {
                    NO_REF
                },
                mvs[list],
                done,
            );
        }
    }
}

/// mb_pred / sub_mb_pred for P and B macroblocks (7.3.5.1, 7.3.5.2)
/// followed by motion vector derivation (8.4.1).
fn parse_inter_pred(
    syn: &mut impl Syntax,
    s: &SliceCtx,
    mbs: &Mbs,
    addr: usize,
    mi: &mut MbWork,
) -> Result<Parts, Error> {
    let b_slice = s.header.slice_type == SliceType::B;
    let num_active = [s.num_active(0), s.num_active(1)];
    fn parse_ref(
        syn: &mut impl Syntax,
        mbs: &Mbs,
        addr: usize,
        mi: &MbInfo,
        list: usize,
        x: usize,
        y: usize,
        num_active: u32,
    ) -> Result<i8, Error> {
        if num_active <= 1 {
            return Ok(0);
        }
        let r = syn.ref_idx(mbs, addr, mi, list, x, y, num_active);
        if r >= num_active {
            return Err(Error::OutOfRange {
                field: "ref_idx",
                value: r as i64,
            });
        }
        Ok(r as i8)
    }
    fn parse_mvd(
        syn: &mut impl Syntax,
        mbs: &Mbs,
        addr: usize,
        mi: &mut MbWork,
        list: usize,
        p: Part,
    ) -> [i32; 2] {
        let v = syn.mvd_pair(mbs, addr, mi, list, p.x, p.y);
        store_mvd(mi, list, p, v);
        v
    }
    #[inline(always)]
    #[allow(clippy::type_complexity)]
    fn inter_parts<Y: Syntax, const N: usize>(
        (syn, s, mbs, addr, mi, partition, modes, num_active, around): (
            &mut Y,
            &SliceCtx,
            &Mbs,
            usize,
            &mut MbWork,
            Partition,
            [PredMode; 2],
            [u32; 2],
            Option<Around>,
        ),
        rects: [Part; N],
    ) -> Result<(), Error> {
        let add =
            |a: [i16; 2], b: [i32; 2]| [(a[0] as i32 + b[0]) as i16, (a[1] as i32 + b[1]) as i16];
        let uses = |k: usize, list: usize| {
            if list == 0 {
                modes[k].uses_l0()
            } else {
                modes[k].uses_l1()
            }
        };
        let mut refs = [[-1i8; 2]; 2];
        for list in 0..2 {
            for (k, p) in rects.iter().enumerate() {
                if uses(k, list) {
                    refs[list][k] =
                        parse_ref(syn, mbs, addr, mi, list, p.x, p.y, num_active[list])?;
                    // Visible to the next partition's context.
                    set_ref(mi, list, *p, refs[list][k]);
                }
            }
        }
        // The mvds of each list follow in partition order, and a list's
        // predictions depend only on its own vectors: each partition is
        // predicted as its mvd is read. `done` marks the partitions before
        // it (an unused list keeps the defaults: no reference, no motion).
        for list in 0..2 {
            let mut done = 0u16;
            for (k, p) in rects.iter().enumerate() {
                if !uses(k, list) {
                    done |= part_bits(*p);
                    continue;
                }
                let shape = (partition != Partition::P16x16).then_some((partition, k));
                let mvd = parse_mvd(syn, mbs, addr, mi, list, *p);
                let r = refs[list][k];
                let mvp = match &around {
                    Some(ar) => predict_mv_near(ar, mi, done, list, *p, r, shape),
                    None => predict_mv(mbs, addr, mi, done, list, *p, r, shape),
                };
                set_motion(
                    mi,
                    list,
                    *p,
                    r,
                    ref_id(s, list, r),
                    add(mvp, mvd),
                    &mut done,
                );
            }
        }

        Ok(())
    }
    let add = |a: [i16; 2], b: [i32; 2]| [(a[0] as i32 + b[0]) as i16, (a[1] as i32 + b[1]) as i16];
    let mut parts = Parts { no_sub_8x8: true };
    match mi.mb_type {
        MbType::Direct16x16 => {
            parts.no_sub_8x8 = s.sps.direct_8x8_inference;
            let spatial = SpatialDirect::new(s, mbs, addr, mi);
            direct_mb(s, addr, mi, &spatial);
        }
        MbType::Inter { partition, modes } if partition != Partition::P8x8 => {
            // Each shape with its rectangles as constants, so the stores
            // and neighbour lookups of each partition fold to fixed ones.
            let p = |x, y, w, h| Part { x, y, w, h };
            let a = (
                syn,
                s,
                mbs,
                addr,
                mi,
                partition,
                modes,
                num_active,
                mbs.around(addr),
            );
            match partition {
                Partition::P16x16 => inter_parts(a, [p(0, 0, 16, 16)])?,
                Partition::P16x8 => inter_parts(a, [p(0, 0, 16, 8), p(0, 8, 16, 8)])?,
                _ => inter_parts(a, [p(0, 0, 8, 16), p(8, 0, 8, 16)])?,
            }
        }
        MbType::Inter { .. } | MbType::P8x8Ref0 => {
            let mut subs = [SubMbType {
                partition: SubPartition::S8x8,
                mode: PredMode::L0,
            }; 4];
            for sub in subs.iter_mut() {
                let v = syn.sub_mb_type(b_slice);
                *sub = if b_slice {
                    SubMbType::from_b(v)
                } else {
                    SubMbType::from_p(v)
                }
                .ok_or(Error::Invalid("sub_mb_type"))?;
                if sub.mode == PredMode::Direct {
                    if !s.sps.direct_8x8_inference {
                        parts.no_sub_8x8 = false;
                    }
                } else if sub.partition != SubPartition::S8x8 {
                    parts.no_sub_8x8 = false;
                }
            }
            for k in 0..4 {
                if subs[k].mode == PredMode::Direct {
                    mi.direct8x8 |= 1 << k;
                }
            }
            let uses = |k: usize, list: usize| {
                if list == 0 {
                    subs[k].mode.uses_l0()
                } else {
                    subs[k].mode.uses_l1()
                }
            };
            let mut refs = [[-1i8; 4]; 2];
            for list in 0..2 {
                for k in 0..4 {
                    let (x, y) = (8 * (k % 2), 8 * (k / 2));
                    if uses(k, list) {
                        refs[list][k] = if mi.mb_type == MbType::P8x8Ref0 {
                            0
                        } else {
                            parse_ref(syn, mbs, addr, mi, list, x, y, num_active[list])?
                        };
                        set_ref(mi, list, Part { x, y, w: 8, h: 8 }, refs[list][k]);
                    }
                }
            }
            let spatial = if subs.iter().any(|t| t.mode == PredMode::Direct) {
                Some(SpatialDirect::new(s, mbs, addr, mi))
            } else {
                None
            };
            let around = mbs.around(addr);
            // As in inter_parts: each partition predicted as its mvd is
            // read, list by list. Direct blocks get both lists' motion in
            // the first pass.
            for list in 0..2 {
                let mut done = 0u16;
                for k in 0..4 {
                    let blk = Part {
                        x: 8 * (k % 2),
                        y: 8 * (k / 2),
                        w: 8,
                        h: 8,
                    };
                    if subs[k].mode == PredMode::Direct {
                        if list == 0 {
                            direct_8x8(s, addr, mi, k, spatial.as_ref().unwrap(), &mut done);
                        }
                        done |= part_bits(blk);
                        continue;
                    }
                    if !uses(k, list) {
                        done |= part_bits(blk);
                        continue;
                    }
                    let (sw, sh) = subs[k].partition.size();
                    let r = refs[list][k];
                    for j in 0..subs[k].partition.count() {
                        let p = sub_rect(k, subs[k].partition, j, sw as usize, sh as usize);
                        let mvd = parse_mvd(syn, mbs, addr, mi, list, p);
                        let mvp = match &around {
                            Some(ar) => predict_mv_sub(ar, mi, done, list, p, r),
                            None => predict_mv(mbs, addr, mi, done, list, p, r, None),
                        };
                        set_motion(mi, list, p, r, ref_id(s, list, r), add(mvp, mvd), &mut done);
                    }
                }
            }
        }
        _ => return Err(Error::Unsupported("macroblock type")),
    }
    Ok(parts)
}

/// Rectangle of sub-macroblock partition `j` of 8x8 block `k`.
fn sub_rect(k: usize, sp: SubPartition, j: usize, w: usize, h: usize) -> Part {
    let (x0, y0) = (8 * (k % 2), 8 * (k / 2));
    let (dx, dy) = match sp {
        SubPartition::S8x8 => (0, 0),
        SubPartition::S8x4 => (0, 4 * j),
        SubPartition::S4x8 => (4 * j, 0),
        SubPartition::S4x4 => (4 * (j % 2), 4 * (j / 2)),
    };
    Part {
        x: x0 + dx,
        y: y0 + dy,
        w,
        h,
    }
}

/// Prediction weights (8.4.3) for a partition using references r0 / r1
/// (negative: list unused). None means default prediction. Returns
/// (w0, w1) for luma, Cb and Cr.
fn weights(s: &SliceCtx, r: [i8; 2]) -> Option<[(Weight, Weight); 3]> {
    let h = s.header;
    let unit = Weight {
        log_wd: 0,
        w: 1,
        o: 0,
    };
    let is_b = h.slice_type == SliceType::B;
    if is_b && s.pps.weighted_bipred_idc == 2 {
        if r[0] < 0 || r[1] < 0 {
            return None;
        }
        let implicit_w1 = || {
            let (p0, p1) = (ref_poc(s, 0, r[0])?, ref_poc(s, 1, r[1])?);
            let dsf = if p0.long_term || p1.long_term {
                None
            } else {
                dist_scale_factor(s.cur_poc(), p0.poc, p1.poc)
            };
            Some(match dsf {
                Some(f) if (-64..=128).contains(&(f >> 2)) => f >> 2,
                _ => 32,
            })
        };
        // Frame macroblocks: from the slice's table.
        let n1 = s.ref_lists[1].len();
        let cell = (s.mb_field.get().is_none() && (r[1] as usize) < n1)
            .then(|| s.implicit.get(r[0] as usize * n1 + r[1] as usize))
            .flatten();
        let w1 = match cell {
            Some(c) if c.get() != i16::MIN => c.get() as i32,
            Some(c) => {
                let w1 = implicit_w1()?;
                c.set(w1 as i16);
                w1
            }
            None => implicit_w1()?,
        };
        let (w0, w1) = (64 - w1, w1);
        // Equal weights are the default average: (32 a + 32 b + 32) >> 6.
        if (w0, w1) == (32, 32) {
            return None;
        }
        let w = (
            Weight {
                log_wd: 5,
                w: w0,
                o: 0,
            },
            Weight {
                log_wd: 5,
                w: w1,
                o: 0,
            },
        );
        return Some([w; 3]);
    }
    let explicit = (s.pps.weighted_pred && matches!(h.slice_type, SliceType::P | SliceType::Sp))
        || (s.pps.weighted_bipred_idc == 1 && is_b);
    if !explicit {
        return None;
    }
    let t = h.pred_weight_table.as_ref()?;
    let mut out = [(unit, unit); 3];
    for list in 0..2 {
        if r[list] < 0 {
            continue;
        }
        // refIdxLXWP: field macroblocks share the weights of their frame (8-296).
        let e = t.lists[list].get((r[list] >> s.mb_field.get().is_some() as i8) as usize)?;
        // Offsets are in units of 8-bit samples (8-301).
        let lw = Weight {
            log_wd: t.luma_log2_denom as u32,
            w: e.luma.0 as i32,
            o: (e.luma.1 as i32) << (s.bit_depth(0) - 8),
        };
        let cw = |k: usize| Weight {
            log_wd: t.chroma_log2_denom as u32,
            w: e.chroma[k].0 as i32,
            o: (e.chroma[k].1 as i32) << (s.bit_depth(1) - 8),
        };
        for (plane, w) in [lw, cw(0), cw(1)].into_iter().enumerate() {
            if list == 0 {
                out[plane].0 = w;
            } else {
                out[plane].1 = w;
            }
        }
    }
    // Default weights of the lists used (w = 2^logWD, no offset) give the
    // default prediction: ((a + 2^(logWD - 1)) >> logWD), and for two lists
    // ((a + b) 2^logWD + 2^logWD) >> (logWD + 1).
    let default = |w: Weight| w.w == 1 << w.log_wd && w.o == 0;
    if out
        .iter()
        .all(|&(w0, w1)| (r[0] < 0 || default(w0)) && (r[1] < 0 || default(w1)))
    {
        return None;
    }
    // The bi-predictive formula takes logWD from w0; keep it for L1-only blocks too.
    if r[0] < 0 {
        for o in &mut out {
            o.0.log_wd = o.1.log_wd;
        }
    }
    Some(out)
}

/// Motion-compensated prediction of all partitions, written into the picture.
/// [`predict_inter`] of 8-bit 4:2:0 frame macroblocks (not MBAFF), the
/// common case, without the general paths' per-block plane and field
/// handling: macroblock (mx, luma row ly0).
fn predict_inter_420<T: Pixel>(s: &SliceCtx, px: &mut Pix<T>, mi: &MbInfo, mx: usize, ly0: usize) {
    let (lx, cx, cy0) = (mx * 16, mx * 8, ly0 / 2);
    // Weights only where the slice has them (8.4.2.3).
    let weighted = match s.header.slice_type {
        SliceType::P | SliceType::Sp => s.pps.weighted_pred,
        SliceType::B => s.pps.weighted_bipred_idc != 0,
        _ => false,
    };
    motion_blocks(mi).for_each(|p| {
        let b4 = p.x / 4 + 4 * (p.y / 4);
        let b8 = p.x / 8 + 2 * (p.y / 8);
        let refs = [mi.ref_idx[0][b8], mi.ref_idx[1][b8]];
        let rp = [0, 1].map(|l| s.mc_ref(l, refs[l]));
        if rp.iter().all(|r| matches!(r, None | Some(None))) && refs.iter().any(|&r| r >= 0) {
            // With no picture to predict from, the prediction is mid-grey.
            for plane in 0..3 {
                let (mut pl, base) = px.win(plane);
                let (x, y, w, h) = if plane == 0 {
                    (p.x, p.y, p.w, p.h)
                } else {
                    (p.x / 2, p.y / 2, p.w / 2, p.h / 2)
                };
                for r in 0..h {
                    let o = base + (y + r) * pl.stride + x;
                    pl.m()[o..o + w].fill(T::new(128));
                }
            }
            return;
        }
        let rp = rp.map(|r| r.flatten());
        let wts = if weighted { weights(s, refs) } else { None };
        let two = rp[0].is_some() && rp[1].is_some();
        // One unweighted prediction goes straight into the picture; of two,
        // the second is averaged into the first there (8-273).
        let into_pic = wts.is_none();
        let (y0, cy) = (ly0 + p.y, cy0 + p.y / 2);
        {
            let ((mut pl, base), buf) = px.win_mc(0);
            let stride = pl.stride;
            let o = base + p.y * stride + p.x;
            for list in 0..2 {
                let Some(rp) = rp[list] else { continue };
                let mv = mi.mv[list][b4];
                wait_ref_rows(rp, 0, mv, (y0, cy), p.h, 8, false);
                let (dst, ds) = if into_pic {
                    (T::u8s_mut(&mut pl.m()[o..]), stride)
                } else {
                    (T::u8s_mut(&mut buf[list][..]), inter::MAX)
                };
                let (x, y) = (
                    (lx + p.x) as i32 + (mv[0] as i32 >> 2),
                    y0 as i32 + (mv[1] as i32 >> 2),
                );
                let (fx, fy) = ((mv[0] & 3) as usize, (mv[1] & 3) as usize);
                let src = &rp.planes[0];
                if into_pic && two && list == 1 {
                    inter::luma_a::<true>(dst, ds, src, x, y, fx, fy, p.w, p.h);
                } else {
                    inter::luma_a::<false>(dst, ds, src, x, y, fx, fy, p.w, p.h);
                }
            }
            if !into_pic {
                inter::combine_t(
                    &mut pl.m()[o..],
                    stride,
                    rp[0].is_some().then_some(&buf[0]),
                    rp[1].is_some().then_some(&buf[1]),
                    wts.map(|w| w[0]),
                    p.w,
                    p.h,
                    255,
                );
            }
        }
        // Cb and Cr: 4:2:0 vectors are luma vectors in 1/8 chroma units.
        let (w, h) = (p.w / 2, p.h / 2);
        let at = |mv: [i16; 2]| {
            let (mv0, mv1) = (mv[0] as i32, mv[1] as i32);
            (
                (cx + p.x / 2) as i32 + (mv0 >> 3),
                cy as i32 + (mv1 >> 3),
                mv0 & 7,
                mv1 & 7,
            )
        };
        let ([(mut cb, base), (mut cr, _)], buf) = px.win_mc2();
        let stride = cb.stride;
        let o = base + p.y / 2 * stride + p.x / 2;
        let (b_cb, b_cr) = buf.split_at_mut(2);
        for list in 0..2 {
            let Some(rp) = rp[list] else { continue };
            let mv = mi.mv[list][b4];
            wait_ref_rows(rp, 1, mv, (y0, cy), h, 8, false);
            let (x, y, fx, fy) = at(mv);
            let (dst, ds) = if into_pic {
                ([&mut cb.m()[o..], &mut cr.m()[o..]], stride)
            } else {
                ([&mut b_cb[list][..], &mut b_cr[list][..]], inter::MAX)
            };
            let dst = dst.map(T::u8s_mut);
            let planes = [&rp.planes[1], &rp.planes[2]];
            if into_pic && two && list == 1 {
                inter::chroma_pair_a::<true>(dst, ds, planes, x, y, fx, fy, w, h);
            } else {
                inter::chroma_pair(dst, ds, planes, x, y, fx, fy, w, h);
            }
        }
        if !into_pic {
            for (plane, out, b) in [(1, &mut cb, &*b_cb), (2, &mut cr, &*b_cr)] {
                inter::combine_t(
                    &mut out.m()[o..],
                    stride,
                    rp[0].is_some().then_some(&b[0]),
                    rp[1].is_some().then_some(&b[1]),
                    wts.map(|w| w[plane]),
                    w,
                    h,
                    255,
                );
            }
        }
    });
}

fn predict_inter<T: Pixel>(s: &SliceCtx, px: &mut Pix<T>, addr: usize, mi: &MbInfo) {
    let w_mbs = px.width_mbs();
    let (cw, ch) = px.buf.mb_c;
    // Position of the macroblock in the reference: a field macroblock of an
    // MBAFF frame covers rows of its pair in the reference field.
    let (mx, ly0, cy0) = if px.mbaff {
        let (x, y, bottom) = (addr / 2 % w_mbs, addr / 2 / w_mbs, addr % 2);
        if mi.field {
            (x, y * 16, y * ch)
        } else {
            (x, y * 32 + bottom * 16, y * 2 * ch + bottom * ch)
        }
    } else {
        let (x, y) = px.mb_xy(addr);
        (x, y * 16, y * ch)
    };
    // Parity of the current field or field macroblock (Table 8-10).
    let cur_parity = if s.header.field_pic {
        Some(s.header.bottom_field)
    } else {
        s.mb_field.get().map(|p| p == Parity::Bottom)
    };
    let planes = px.buf.planes.len();
    if !T::HIGH
        && !px.mbaff
        && cur_parity.is_none()
        && planes == 3
        && (cw, ch) == (8, 8)
        && s.header.colour_plane_id == 0
    {
        return predict_inter_420(s, px, mi, mx, ly0);
    }
    // Prediction depends only on each sample's motion, so blocks are formed
    // from the motion field rather than the coded partitions: B_Skip and
    // direct macroblocks come as 8x8 or 4x4 partitions that often share it.
    motion_blocks(mi).for_each(|p| {
        let b4 = p.x / 4 + 4 * (p.y / 4);
        let b8 = p.x / 8 + 2 * (p.y / 8);
        let refs = [mi.ref_idx[0][b8], mi.ref_idx[1][b8]];
        let wts = weights(s, refs);
        // With no picture to predict from, the prediction is mid-grey.
        let rp = [0, 1].map(|l| s.mc_ref(l, refs[l]));
        if rp.iter().all(|r| matches!(r, None | Some(None))) && refs.iter().any(|&r| r >= 0) {
            for plane in 0..planes {
                let (mut pl, base) = px.win(plane);
                let (x, y, w, h) = if plane == 0 {
                    (p.x, p.y, p.w, p.h)
                } else {
                    (p.x * cw / 16, p.y * ch / 16, p.w * cw / 16, p.h * ch / 16)
                };
                let grey = T::new(1 << (s.bit_depth(plane) - 1));
                for r in 0..h {
                    let o = base + (y + r) * pl.stride + x;
                    pl.m()[o..o + w].fill(grey);
                }
            }
            return;
        }
        let rp = rp.map(|r| r.flatten());
        // One unweighted prediction goes straight into the picture; of two
        // (8-bit), the second is averaged into the first there (8-273).
        let direct = wts.is_none() && rp.iter().filter(|r| r.is_some()).count() == 1;
        let fused = !T::HIGH && wts.is_none() && rp.iter().all(|r| r.is_some());
        // 8-bit Cb and Cr blocks of 4 or 8 columns are predicted together.
        let pair = !T::HIGH && planes == 3 && cw == 8 && p.w >= 8;
        // Chroma position and eighth-sample fraction of a reference (4:2:0
        // vectors are luma vectors in 1/8 chroma units, shifted between
        // fields of opposite parity, Table 8-10; vertically 1/4 chroma
        // samples for 4:2:2, 8-228, 8-232, 8-234).
        let chroma_at = |rp: &McRef, mv: [i16; 2]| {
            let (x, y) = (
                (mx * cw + p.x * cw / 16) as i32,
                (cy0 + p.y * ch / 16) as i32,
            );
            let adjust = match (cur_parity, rp.parity, ch) {
                (Some(true), Some(Parity::Top), 8) => 2,
                (Some(false), Some(Parity::Bottom), 8) => -2,
                _ => 0,
            };
            let mv = [mv[0] as i32, mv[1] as i32 + adjust];
            let (y_int, y_frac) = if ch == 16 {
                (mv[1] >> 2, (mv[1] & 3) << 1)
            } else {
                (mv[1] >> 3, mv[1] & 7)
            };
            (x + (mv[0] >> 3), y + y_int, mv[0] & 7, y_frac)
        };
        for plane in 0..if pair { 1 } else { planes } {
            let ((mut pl, base), buf) = px.win_mc(plane);
            let stride = pl.stride;
            let (o, w, h) = if plane == 0 {
                (base + p.y * stride + p.x, p.w, p.h)
            } else {
                let (x, y) = (p.x * cw / 16, p.y * ch / 16);
                (base + y * stride + x, p.w * cw / 16, p.h * ch / 16)
            };
            for list in 0..2 {
                let Some(rp) = &rp[list] else { continue };
                let mv = mi.mv[list][b4];
                wait_ref_rows(
                    rp,
                    plane,
                    mv,
                    (ly0 + p.y, cy0 + p.y * ch / 16),
                    h,
                    ch,
                    cw == 16,
                );
                // With separate colour planes, the reference plane of this colour.
                let src = &rp.planes[plane + s.header.colour_plane_id as usize];
                let (dst, ds) = if direct || fused {
                    (&mut pl.m()[o..], stride)
                } else {
                    (&mut buf[list][..], inter::MAX)
                };
                let avg = fused && list == 1;
                if plane == 0 || cw == 16 {
                    // 4:4:4 chroma is interpolated like luma (8.4.2.2).
                    let (x, y) = ((mx * 16 + p.x) as i32, (ly0 + p.y) as i32);
                    inter::luma_t(
                        dst,
                        ds,
                        src,
                        x + (mv[0] as i32 >> 2),
                        y + (mv[1] as i32 >> 2),
                        (mv[0] & 3) as usize,
                        (mv[1] & 3) as usize,
                        w,
                        h,
                        s.max_sample(plane),
                        avg,
                    );
                } else {
                    let (x, y, fx, fy) = chroma_at(rp, mv);
                    inter::chroma_t(dst, ds, src, x, y, fx, fy, w, h, avg);
                }
            }
            if !direct && !fused {
                inter::combine_t(
                    &mut pl.m()[o..],
                    stride,
                    rp[0].is_some().then_some(&buf[0]),
                    rp[1].is_some().then_some(&buf[1]),
                    wts.map(|w| w[plane]),
                    w,
                    h,
                    s.max_sample(plane),
                );
            }
        }
        if pair {
            let ([(mut cb, base), (mut cr, _)], buf) = px.win_mc2();
            let stride = cb.stride;
            debug_assert_eq!(stride, cr.stride);
            let o = base + p.y * ch / 16 * stride + p.x / 2;
            let (w, h) = (p.w / 2, p.h * ch / 16);
            let (b_cb, b_cr) = buf.split_at_mut(2);
            for list in 0..2 {
                let Some(rp) = &rp[list] else { continue };
                let mv = mi.mv[list][b4];
                wait_ref_rows(rp, 1, mv, (ly0 + p.y, cy0 + p.y * ch / 16), h, ch, false);
                let (x, y, fx, fy) = chroma_at(rp, mv);
                let (dst, ds) = if direct || fused {
                    ([&mut cb.m()[o..], &mut cr.m()[o..]], stride)
                } else {
                    ([&mut b_cb[list][..], &mut b_cr[list][..]], inter::MAX)
                };
                let dst = dst.map(T::u8s_mut);
                let planes = [&rp.planes[1], &rp.planes[2]];
                if fused && list == 1 {
                    inter::chroma_pair_a::<true>(dst, ds, planes, x, y, fx, fy, w, h);
                } else {
                    inter::chroma_pair(dst, ds, planes, x, y, fx, fy, w, h);
                }
            }
            if !direct && !fused {
                for (plane, out, b) in [(1, &mut cb, &*b_cb), (2, &mut cr, &*b_cr)] {
                    inter::combine_t(
                        &mut out.m()[o..],
                        stride,
                        rp[0].is_some().then_some(&b[0]),
                        rp[1].is_some().then_some(&b[1]),
                        wts.map(|w| w[plane]),
                        w,
                        h,
                        s.max_sample(plane),
                    );
                }
            }
        }
    });
}

/// The largest blocks of a macroblock with uniform motion, of the shapes
/// macroblock partitions take (16x16 down to 4x4).
fn motion_blocks(mi: &MbInfo) -> Blocks {
    // Partitions of uniform motion, as coded (P_Skip has no direct blocks).
    match mi.mb_type {
        _ if mi.uniform_motion => return Blocks::WHOLE,
        MbType::Skip if mi.direct8x8 == 0 => return Blocks::WHOLE,
        MbType::Inter { partition, .. } if partition != Partition::P8x8 => {
            return match partition {
                Partition::P16x16 => Blocks::WHOLE,
                Partition::P16x8 => Blocks::ROWS,
                _ => Blocks::COLUMNS,
            };
        }
        _ => {}
    }
    // Per 4x4 block (raster) the vectors of both lists, per 8x8 block the
    // references: a list a block does not use has vector 0 (the default,
    // which direct prediction also sets), so equal keys are equal motion.
    let v = |l: usize, b: usize| {
        let m = mi.mv[l][b];
        m[0] as u16 as u64 | (m[1] as u16 as u64) << 16
    };
    let keys: [u64; 16] = std::array::from_fn(|b| v(0, b) | v(1, b) << 32);
    let refs: [u16; 4] = std::array::from_fn(|k| {
        mi.ref_idx[0][k] as u8 as u16 | (mi.ref_idx[1][k] as u8 as u16) << 8
    });
    // 4x4 blocks a and b (raster) have the same references and vectors.
    let same = |a: usize, b: usize| {
        keys[a] == keys[b] && refs[a % 4 / 2 + a / 8 * 2] == refs[b % 4 / 2 + b / 8 * 2]
    };
    if keys.iter().all(|&k| k == keys[0]) && refs.iter().all(|&r| r == refs[0]) {
        return Blocks::WHOLE;
    }
    // Top-left 4x4 block of quadrant q, and whether the quadrant is uniform.
    let q0 = |q: usize| q % 2 * 2 + q / 2 * 8;
    let uniform = |q: usize| [1, 4, 5].iter().all(|&d| same(q0(q), q0(q) + d));
    if (0..4).all(uniform) {
        if same(0, 2) && same(8, 10) {
            return Blocks::ROWS;
        }
        if same(0, 8) && same(2, 10) {
            return Blocks::COLUMNS;
        }
    }
    let mut quads = 0u8;
    for q in 0..4 {
        let b = q0(q);
        let shape = if uniform(q) {
            0
        } else if same(b, b + 1) && same(b + 4, b + 5) {
            1
        } else if same(b, b + 4) && same(b + 1, b + 5) {
            2
        } else {
            3
        };
        quads |= shape << (2 * q);
    }
    Blocks(Blocks::QUADS | quads as u16)
}

/// The blocks of [`motion_blocks`], as a code returned in a register: the
/// whole macroblock, two 16x8 or 8x16 halves, or each 8x8 quadrant whole
/// (0), as 8x4 (1), 4x8 (2) or 4x4 blocks (3), two bits each.
#[derive(Clone, Copy)]
struct Blocks(u16);

impl Blocks {
    const WHOLE: Self = Self(0);
    const ROWS: Self = Self(1 << 8);
    const COLUMNS: Self = Self(2 << 8);
    const QUADS: u16 = 3 << 8;

    /// Calls `f` with each block, in raster order of the quadrants.
    #[inline(always)]
    fn for_each(self, mut f: impl FnMut(Part)) {
        let part = |x: usize, y: usize, w: usize, h: usize| Part { x, y, w, h };
        match self.0 >> 8 {
            0 => f(part(0, 0, 16, 16)),
            1 => {
                f(part(0, 0, 16, 8));
                f(part(0, 8, 16, 8));
            }
            2 => {
                f(part(0, 0, 8, 16));
                f(part(8, 0, 8, 16));
            }
            _ => {
                for q in 0..4 {
                    let (x, y) = (q % 2 * 8, q / 2 * 8);
                    match self.0 >> (2 * q) & 3 {
                        0 => f(part(x, y, 8, 8)),
                        1 => {
                            f(part(x, y, 8, 4));
                            f(part(x, y + 4, 8, 4));
                        }
                        2 => {
                            f(part(x, y, 4, 8));
                            f(part(x + 4, y, 4, 8));
                        }
                        _ => {
                            for k in [0, 1, 4, 5] {
                                f(part(x + k % 4 * 4, y + k / 4 * 4, 4, 4));
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Waits until the rows of reference `rp` that a block of plane `plane`
/// interpolates from are final (frame threading). `y0` is the block's top
/// row in the reference view (luma, chroma), `h` its height in the plane.
#[inline(always)]
fn wait_ref_rows(
    rp: &McRef,
    plane: usize,
    mv: [i16; 2],
    y0: (usize, usize),
    h: usize,
    ch: usize,
    c444: bool,
) {
    // A finished reference (always, decoding on one thread) needs nothing.
    if !rp.state.is_done() {
        wait_ref_rows_slow(rp, plane, mv, y0, h, ch, c444);
    }
}

#[cold]
#[inline(never)]
fn wait_ref_rows_slow(
    rp: &McRef,
    plane: usize,
    mv: [i16; 2],
    y0: (usize, usize),
    h: usize,
    ch: usize,
    c444: bool,
) {
    let state = rp.state;
    let mvy = mv[1] as i32;
    // One past the last view row read: 6-tap filtering reads 3 rows below
    // the block, bilinear chroma one; field adjustments of 2 are covered.
    let luma_rows = if plane == 0 || c444 {
        y0.0 as i32 + (mvy >> 2) + h as i32 + 4
    } else {
        let c = y0.1 as i32 + (mvy >> if ch == 16 { 2 } else { 3 }) + h as i32 + 2;
        (c * 16 + ch as i32 - 1) / ch as i32
    };
    let rows = luma_rows.max(1) as usize;
    // A field of a frame: its rows interleave with the other field's.
    let rows = if rp.field_of_frame {
        2 * rows + 1
    } else {
        rows
    };
    state.wait_rows(rows);
}

/// coded_block_flag condTermFlagN (9.3.3.1.1.9) given the neighbour's
/// macroblock (None if unavailable) and whether its transform block exists.
///
/// A macroblock's cbf bits are set only for blocks it codes (all of them
/// for I_PCM), so for most blocks the bit alone is the condition: see
/// [`cbf_bit_cond`].
fn cbf_cond(
    n: Option<&MbInfo>,
    cur_intra: bool,
    block: impl FnOnce(&MbInfo) -> Option<bool>,
) -> usize {
    match n {
        None => cur_intra as usize,
        Some(m) if m.is_pcm() => 1,
        Some(m) => block(m).unwrap_or(false) as usize,
    }
}

/// condTermFlagN of a block whose cbf bit is `bit` in neighbour `n`,
/// unavailable in an intra macroblock counting as coded.
#[inline(always)]
fn cbf_bit_cond(n: Option<&MbInfo>, cur_intra: bool, bit: u32) -> u32 {
    n.map_or(cur_intra as u32, |m| (m.cbf >> bit & 1) as u32)
}

/// residual_luma() (7.3.5.3.1) with CABAC for plane 0 (luma), or Cb (1) or
/// Cr (2) in 4:4:4, whose blocks use ctxBlockCat 6 to 9 and 10 to 13.
#[allow(clippy::too_many_arguments)]
fn cabac_residual_luma(
    d: &mut Decoder,
    c: &mut se::Contexts,
    s: &SliceCtx,
    mbs: &Mbs,
    addr: usize,
    mi: &mut MbWork,
    plane: usize,
    lv: LumaLevelsMut,
) {
    let (a, b) = (mbs.a(addr), mbs.b(addr));
    let info = &mbs.info;
    let intra = mi.is_intra();
    let field = s.field();
    // ctxBlockCat of the DC, AC, 4x4 and 8x8 blocks.
    let cats = [[0, 1, 2, 5], [6, 7, 8, 9], [10, 11, 12, 13]][plane];
    let dq = dequant(s, mi, plane);

    if mi.is_i16x16() {
        let bit = cbf_dc_bit(plane);
        let inc = (cbf_bit_cond(a.map(|n| &info[n]), intra, bit)
            + 2 * cbf_bit_cond(b.map(|n| &info[n]), intra, bit)) as usize;
        if se::coded_block_flag(d, c, cats[0], inc) {
            mi.cbf |= 1 << cbf_dc_bit(plane);
            se::residual_block(d, c, cats[0], field, 16, 0, lv.luma_dc);
        }
    }

    if mi.transform_8x8 {
        // 8x8 blocks, ctxBlockCat 5, 9 or 13. coded_block_flag is inferred
        // to be 1 outside 4:4:4 (7.4.5.3.3).
        let c444 = s.sps.chroma_array_type() == 3;
        if !c444 {
            // All coded blocks in one call.
            let cbp = mi.cbp_luma;
            se::luma8x8(
                d,
                c,
                field,
                cbp,
                dq.place8(),
                lv.luma8,
                mi.ctx.nnz_of_mut(plane),
            );
            for b8 in 0..4 {
                if cbp >> b8 & 1 != 0 {
                    mi.cbf |= 0xf << cbf_bit(plane, 4 * b8);
                }
            }
            return;
        }
        // An 8x8 block of a neighbour, if it has one (9.3.3.1.1.9).
        let block8 = |m: &MbInfo, k8: usize| {
            (!m.is_skip() && m.transform_8x8 && m.cbp_luma >> k8 & 1 != 0)
                .then(|| m.cbf >> cbf_bit(plane, 4 * k8) & 1 != 0)
        };
        let outside8 = |x: i32, y: i32| {
            let n = mbs.locate(addr, x, y);
            cbf_cond(n.map(|n| &info[n.0]), intra, |m| {
                let (_, xw, yw) = n.unwrap();
                block8(m, yw / 8 * 2 + xw / 8)
            })
        };
        for b8 in 0..4 {
            if mi.cbp_luma >> b8 & 1 == 0 {
                continue;
            }
            let coded = !c444 || {
                let (x8, y8) = (b8 % 2, b8 / 2);
                let ca = if x8 > 0 {
                    (mi.cbf >> cbf_bit(plane, 4 * (b8 - 1)) & 1) as usize
                } else {
                    outside8(-1, 8 * y8 as i32)
                };
                let cb = if y8 > 0 {
                    (mi.cbf >> cbf_bit(plane, 4 * (b8 - 2)) & 1) as usize
                } else {
                    outside8(8 * x8 as i32, -1)
                };
                se::coded_block_flag(d, c, cats[3], ca + 2 * cb)
            };
            if coded {
                let n = se::residual_placed(
                    d,
                    c,
                    cats[3],
                    field,
                    64,
                    &mut dq.placed8(&mut lv.luma8[b8]),
                );
                mi.cbf |= 0xf << cbf_bit(plane, 4 * b8);
                mi.ctx.nnz_of_mut(plane)[4 * b8..4 * b8 + 4].fill(n.min(16) as u8);
            }
        }
    }
    // Luma 4x4 blocks (Intra_16x16 AC: cat 1, 15 coefficients; else cat 2).
    let (cat, num) = if mi.is_i16x16() {
        (cats[1], 15)
    } else {
        (cats[2], 16)
    };
    if mi.transform_8x8 || mi.cbp_luma == 0 {
        return;
    }
    // Blocks of neighbouring macroblocks (6.4.11.4).
    let outside = |x: i32, y: i32| {
        let n = mbs.locate(addr, x, y);
        let k = n.map_or(0, |(_, xw, yw)| BLK_AT[yw / 4 * 4 + xw / 4]);
        cbf_bit_cond(n.map(|n| &info[n.0]), intra, cbf_bit(plane, k))
    };
    // coded_block_flag of the blocks on a 5x5 grid, bit x + 1 + 5 (y + 1)
    // for block (x, y): the left and top neighbours in column and row 0.
    let mut grid = 0u32;
    if mbs.mbaff {
        for k in 0..4i32 {
            grid |= outside(-1, 4 * k) << (5 * (k + 1));
            grid |= outside(4 * k, -1) << (k + 1);
        }
    } else {
        // The right column of A and the bottom row of B (raster blocks
        // 4k + 3 and 12 + k).
        let (a, b) = (mbs.a(addr), mbs.b(addr));
        for k in 0..4 {
            let left = cbf_bit_cond(
                a.map(|n| &info[n]),
                intra,
                cbf_bit(plane, BLK_AT[4 * k + 3]),
            );
            let top = cbf_bit_cond(b.map(|n| &info[n]), intra, cbf_bit(plane, BLK_AT[12 + k]));
            grid |= left << (5 * (k + 1)) | top << (k + 1);
        }
    }
    if plane == 0 {
        // All blocks in one call (luma only: Cb and Cr in 4:4:4 below).
        let p = dq.place4(16 - num);
        let cbp = mi.cbp_luma;
        let coded = se::luma4x4(
            d,
            c,
            cat,
            field,
            cbp,
            &mut grid,
            p,
            lv.luma,
            &mut mi.ctx.nnz,
        );
        mi.cbf |= coded as u64;
        return;
    }
    for blk in 0..16 {
        if mi.cbp_luma >> (blk / 4) & 1 == 0 {
            continue;
        }
        let (bx, by) = BLK_XY[blk];
        let at = bx + 1 + 5 * (by + 1);
        let inc = (grid >> (at - 1) & 1) + 2 * (grid >> (at - 5) & 1);
        if se::coded_block_flag(d, c, cat, inc as usize) {
            grid |= 1 << at;
            mi.cbf |= 1 << cbf_bit(plane, blk);
            let out = &mut dq.placed4(&mut lv.luma[blk], 16 - num);
            mi.ctx.nnz_of_mut(plane)[blk] = se::residual_placed(d, c, cat, field, num, out) as u8;
        }
    }
}

fn cabac_residual(
    d: &mut Decoder,
    c: &mut se::Contexts,
    s: &SliceCtx,
    mbs: &Mbs,
    addr: usize,
    mi: &mut MbWork,
    res: &mut Residual,
) {
    let (a, b) = (mbs.a(addr), mbs.b(addr));
    let info = &mbs.info;
    let intra = mi.is_intra();
    let field = s.field();

    cabac_residual_luma(d, c, s, mbs, addr, mi, 0, res.plane_mut(0));
    if s.sps.chroma_array_type() == 3 {
        // 4:4:4: Cb and Cr coded like luma (7.3.5.3).
        cabac_residual_luma(d, c, s, mbs, addr, mi, 1, res.plane_mut(1));
        cabac_residual_luma(d, c, s, mbs, addr, mi, 2, res.plane_mut(2));
        return;
    }
    if !matches!(s.sps.chroma_array_type(), 1 | 2) {
        return;
    }
    let num_c8x8 = if s.sps.chroma_array_type() == 1 { 1 } else { 2 };
    if mi.cbp_chroma & 3 != 0 {
        for comp in 0..2 {
            let bit = CBF_CHROMA_DC + comp as u32;
            let inc = (cbf_bit_cond(a.map(|n| &info[n]), intra, bit)
                + 2 * cbf_bit_cond(b.map(|n| &info[n]), intra, bit)) as usize;
            if se::coded_block_flag(d, c, 3, inc) {
                mi.cbf |= 1 << (CBF_CHROMA_DC + comp as u32);
                se::residual_block(
                    d,
                    c,
                    3,
                    field,
                    4 * num_c8x8,
                    num_c8x8,
                    &mut res.chroma_dc[comp],
                );
            }
        }
    }
    if mi.cbp_chroma & 2 != 0 {
        let rows = 2 * num_c8x8;
        let dq = [dequant(s, mi, 1), dequant(s, mi, 2)];
        for comp in 0..2 {
            let bit = |k: usize| CBF_CHROMA_AC + 16 * comp as u32 + k as u32;
            // Chroma blocks of neighbouring macroblocks (6.4.11.5).
            let outside = |x: i32, y: i32| {
                let n = mbs.locate_c(addr, x, y, 8, 4 * rows as i32);
                let k = n.map_or(0, |(_, xw, yw)| yw / 4 * 2 + xw / 4);
                cbf_bit_cond(n.map(|n| &info[n.0]), intra, bit(k))
            };
            // coded_block_flag of the blocks on a 3-wide grid, bit x + 1 +
            // 3 (y + 1) for block (x, y): the neighbours in column and row 0.
            let mut grid;
            if mbs.mbaff {
                grid = (outside(0, -1) << 1) | (outside(4, -1) << 2);
                for k in 0..rows as i32 {
                    grid |= outside(-1, 4 * k) << (3 * (k + 1));
                }
            } else {
                // The right column of A (blocks 2k + 1) and the bottom row
                // of B (blocks 2 (rows - 1) and the next).
                let (a, b) = (mbs.a(addr).map(|n| &info[n]), mbs.b(addr).map(|n| &info[n]));
                let last = 2 * (rows - 1);
                grid = cbf_bit_cond(b, intra, bit(last)) << 1
                    | cbf_bit_cond(b, intra, bit(last + 1)) << 2;
                for k in 0..rows {
                    grid |= cbf_bit_cond(a, intra, bit(2 * k + 1)) << (3 * (k + 1));
                }
            }
            let p = dq[comp].place4(1);
            let dst = &mut res.chroma_ac[comp];
            let coded = se::chroma_ac(d, c, field, 4 * num_c8x8, &mut grid, p, dst);
            mi.cbf |= (coded as u64) << bit(0);
        }
    }
}

/// Availability of neighbouring luma samples for intra prediction, relative
/// to the macroblock's top-left corner. `decoded(x, y)` says whether a
/// position inside the current macroblock has been reconstructed.
fn luma_avail(
    mbs: &Mbs,
    addr: usize,
    x: i32,
    y: i32,
    decoded: impl Fn(usize, usize) -> bool,
) -> bool {
    match (x, y) {
        (0..16, 0..16) => decoded(x as usize, y as usize),
        _ => mbs.intra_ok(mbs.locate(addr, x, y).map(|n| n.0)),
    }
}

/// Availability of the `rows` samples left of (x, y): in MBAFF frames they
/// may come from both macroblocks of the left pair.
fn left_avail(mbs: &Mbs, addr: usize, x: i32, y: i32, rows: i32) -> bool {
    if !mbs.mbaff {
        return luma_avail(mbs, addr, x - 1, y, |_, _| true);
    }
    (y..y + rows).all(|r| luma_avail(mbs, addr, x - 1, r, |_, _| true))
}

/// Transform bypass (lossless macroblocks): the residual is the levels
/// themselves, accumulated along the prediction direction for vertical
/// (hor = false) and horizontal (hor = true) intra prediction (8.5.15).
/// `r` is an n x n raster block added to `dst`.
fn add_bypass<T: Pixel>(
    dst: &mut [T],
    stride: usize,
    r: &mut [i32],
    n: usize,
    hor: Option<bool>,
    max: i32,
) {
    add_bypass_rect(dst, stride, r, (n, n), hor, max);
}

/// [`add_bypass`] for a w x h block.
fn add_bypass_rect<T: Pixel>(
    dst: &mut [T],
    stride: usize,
    r: &mut [i32],
    (w, h): (usize, usize),
    hor: Option<bool>,
    max: i32,
) {
    match hor {
        Some(false) => {
            for i in 1..h {
                for j in 0..w {
                    r[i * w + j] += r[(i - 1) * w + j];
                }
            }
        }
        Some(true) => {
            for i in 0..h {
                for j in 1..w {
                    r[i * w + j] += r[i * w + j - 1];
                }
            }
        }
        None => {}
    }
    for i in 0..h {
        for j in 0..w {
            let p = &mut dst[i * stride + j];
            *p = T::clip(p.get() + r[i * w + j], max);
        }
    }
}

/// The prediction of a w x h block at `o` when its mode needs neighbours
/// it does not have (a damaged or non-conforming stream): mid-grey, so the
/// output never depends on what the samples held before.
fn unpredicted<T: Pixel>(dst: &mut [T], o: usize, stride: usize, w: usize, h: usize, bd: u32) {
    let grey = T::new(1 << (bd - 1));
    for r in 0..h {
        dst[o + r * stride..o + r * stride + w].fill(grey);
    }
}

/// The DPCM direction of a lossless intra block predicted with `mode`
/// (0 vertical, 1 horizontal), if any.
fn bypass_dir(mode: u8) -> Option<bool> {
    (mode < 2).then_some(mode == 1)
}

/// Adds the residual of an 8x8 luma block (8.5.13).
fn add_luma8x8<T: Pixel>(
    y: &mut Win<T>,
    o: usize,
    coef: &mut [i32; 64],
    max: i32,
    bypass: Option<Option<bool>>,
) {
    if let Some(hor) = bypass {
        let stride = y.stride;
        let mut r = *coef;
        return add_bypass(&mut y.m()[o..], stride, &mut r, 8, hor, max);
    }
    let stride = y.stride;
    transform::add8x8_dq::<T>(&mut y.m()[o..], stride, coef, max);
}

/// Replaces the prediction of a 4x4 luma block with its SP or SI
/// reconstruction (8.6.1.1, 8.6.2.1).
fn sp_luma4x4<T: Pixel>(
    y: &mut Win<T>,
    o: usize,
    levels: &[i32; 16],
    s: &SliceCtx,
    qp: i32,
    qs: i32,
    switching: bool,
) {
    let stride = y.stride;
    let mut pred = [0i32; 16];
    for (k, v) in pred.iter_mut().enumerate() {
        *v = y.s()[o + k / 4 * stride + k % 4].get();
    }
    let mut coef = crate::sp::block(&pred, levels, qp, qs, switching, false);
    let list = if switching { 0 } else { 3 };
    transform::scale_4x4(&mut coef, &s.ls.s4[list][(qs % 6) as usize], qs, false);
    for r in 0..4 {
        y.m()[o + r * stride..o + r * stride + 4].fill(T::default());
    }
    transform::add4x4::<T>(&mut y.m()[o..], stride, &coef, 255);
}

/// Adds the residual of a 4x4 luma block (non-Intra_16x16).
fn add_luma4x4<T: Pixel>(
    y: &mut Win<T>,
    o: usize,
    coef: &mut [i32; 16],
    max: i32,
    bypass: Option<Option<bool>>,
) {
    let stride = y.stride;
    if let Some(hor) = bypass {
        let mut r = *coef;
        return add_bypass(&mut y.m()[o..], stride, &mut r, 4, hor, max);
    }
    transform::add4x4_dq::<T>(&mut y.m()[o..], stride, coef, max);
}

/// Intra prediction and residual of a plane coded like luma: luma, and Cb
/// and Cr in 4:4:4 (8.3.4.5, 8.5.5). `qp` is QP'Y or QP'C, `sp` the QS and
/// switching flag of an SI macroblock.
#[allow(clippy::too_many_arguments)]
fn recon_luma_like<T: Pixel>(
    s: &SliceCtx,
    px: &mut Pix<T>,
    mbs: &Mbs,
    addr: usize,
    plane: usize,
    qp: i32,
    lv: LumaLevelsMut,
    sp: Option<(i32, bool)>,
) {
    let mi = &mbs.info[addr];
    let intra = mi.is_intra();
    let (bd, max) = (s.bit_depth(plane), s.max_sample(plane));
    // Scaling lists: Y, Cb, Cr for intra then inter (Table 7-2); a colour
    // plane coded separately uses those of its colour (8.5.9).
    let ycbcr = plane + s.header.colour_plane_id as usize;
    let list4 = ycbcr + if intra { 0 } else { 3 };
    let lsy = &s.ls.s4[list4][(qp % 6) as usize];
    let cbf = |blk: usize| mi.cbf >> cbf_bit(plane, blk) & 1 != 0;
    // TransformBypassModeFlag: QP'Y is 0 (7.4.2.1.1).
    let lossless = s.sps.qpprime_y_zero_transform_bypass
        && mi.qp_y as i32 + 6 * (s.bit_depth(0) as i32 - 8) == 0;
    let intra_bypass = |mode: u8| lossless.then(|| bypass_dir(mode));
    let inter_bypass = lossless.then_some(None);
    let (mut y, base) = px.win(plane);
    let stride = y.stride;
    match mi.mb_type {
        MbType::INxN if mi.transform_8x8 => {
            for b8 in 0..4 {
                let (px, py) = (8 * (b8 % 2), 8 * (b8 / 2));
                let decoded = |x: usize, yy: usize| x / 8 + 2 * (yy / 8) < b8;
                let has_top = luma_avail(mbs, addr, px as i32, py as i32 - 1, decoded);
                let has_tr = luma_avail(mbs, addr, px as i32 + 8, py as i32 - 1, decoded);
                let has_left = if px == 0 {
                    left_avail(mbs, addr, 0, py as i32, 8)
                } else {
                    luma_avail(mbs, addr, px as i32 - 1, py as i32, decoded)
                };
                let has_corner = luma_avail(mbs, addr, px as i32 - 1, py as i32 - 1, decoded);
                let o = base + py * stride + px;
                let mut e = Edge8 {
                    top: [T::default(); 16],
                    left: [T::default(); 8],
                    corner: T::default(),
                    has_top,
                    has_left,
                    has_corner,
                };
                if has_top {
                    let row = o - stride;
                    e.top[..8].copy_from_slice(&y.s()[row..row + 8]);
                    if has_tr {
                        e.top[8..].copy_from_slice(&y.s()[row + 8..row + 16]);
                    } else {
                        let t7 = e.top[7];
                        e.top[8..].fill(t7); // 8.3.2.2 substitution
                    }
                }
                if has_left {
                    for k in 0..8 {
                        e.left[k] = y.s()[o + k * stride - 1];
                    }
                }
                if has_corner {
                    e.corner = y.s()[o - stride - 1];
                }
                if !intra::pred8x8(&mut y.m()[o..], stride, mi.intra_modes[4 * b8], &e, bd) {
                    unpredicted(y.m(), o, stride, 8, 8, bd);
                }
                if mi.cbp_luma >> b8 & 1 != 0 {
                    let bp = intra_bypass(mi.intra_modes[4 * b8]);
                    add_luma8x8::<T>(&mut y, o, &mut lv.luma8[b8], max, bp);
                }
            }
        }
        _ if !intra && s.header.slice_type == SliceType::Sp => {}
        MbType::INxN | MbType::Si => {
            for blk in 0..16 {
                let (bx, by) = BLK_XY[blk];
                let (px, py) = (bx * 4, by * 4);
                let decoded = |x: usize, yy: usize| BLK_AT[(yy / 4) * 4 + x / 4] < blk;
                let has_top = luma_avail(mbs, addr, px as i32, py as i32 - 1, decoded);
                let has_tr = luma_avail(mbs, addr, px as i32 + 4, py as i32 - 1, decoded);
                let has_left = if px == 0 {
                    left_avail(mbs, addr, 0, py as i32, 4)
                } else {
                    luma_avail(mbs, addr, px as i32 - 1, py as i32, decoded)
                };
                let has_corner = luma_avail(mbs, addr, px as i32 - 1, py as i32 - 1, decoded);
                let o = base + py * stride + px;
                let mut e = Edge4 {
                    top: [T::default(); 8],
                    left: [T::default(); 4],
                    corner: T::default(),
                    has_top,
                    has_left,
                    has_corner,
                };
                if has_top {
                    let row = o - stride;
                    e.top[..4].copy_from_slice(&y.s()[row..row + 4]);
                    if has_tr {
                        e.top[4..].copy_from_slice(&y.s()[row + 4..row + 8]);
                    } else {
                        let t3 = e.top[3];
                        e.top[4..].fill(t3); // 8.3.1.2 substitution
                    }
                }
                if has_left {
                    for k in 0..4 {
                        e.left[k] = y.s()[o + k * stride - 1];
                    }
                }
                if has_corner {
                    e.corner = y.s()[o - stride - 1];
                }
                if !intra::pred4x4(&mut y.m()[o..], stride, mi.intra_modes[blk], &e, bd) {
                    unpredicted(y.m(), o, stride, 4, 4, bd);
                }
                if let Some((qs, switching)) = sp {
                    sp_luma4x4::<T>(&mut y, o, &lv.luma[blk], s, qp, qs, switching);
                } else if cbf(blk) {
                    let bp = intra_bypass(mi.intra_modes[blk]);
                    add_luma4x4::<T>(&mut y, o, &mut lv.luma[blk], max, bp);
                }
            }
        }
        MbType::I16x16 { pred_mode, .. } => {
            let has_top = mbs.intra_ok(mbs.b(addr));
            let has_left = left_avail(mbs, addr, 0, 0, 16);
            let has_corner = mbs.intra_ok(mbs.d(addr));
            let mut top = [T::default(); 16];
            let mut left = [T::default(); 16];
            if has_top {
                top.copy_from_slice(&y.s()[base - stride..base - stride + 16]);
            }
            if has_left {
                for k in 0..16 {
                    left[k] = y.s()[base + k * stride - 1];
                }
            }
            let corner = if has_corner {
                y.s()[base - stride - 1]
            } else {
                T::default()
            };
            let e = EdgeMb {
                top: &top,
                left: &left,
                corner,
                has_top,
                has_left,
                left_half: [has_left; 2],
                has_corner,
            };
            if !intra::pred16x16(&mut y.m()[base..], stride, pred_mode, &e, bd) {
                unpredicted(y.m(), base, stride, 16, 16, bd);
            }
            // DC: inverse zig-zag into the 4x4 c matrix, then 8.5.10.
            let mut dc = [0i32; 16];
            for (k, &r) in s.scan4().iter().enumerate() {
                dc[r as usize] = lv.luma_dc[k];
            }
            if lossless {
                // rMb holds the levels, DC included (8.5.10, 8.5.12).
                let mut r = [0i32; 256];
                for blk in 0..16 {
                    let (bx, by) = BLK_XY[blk];
                    let mut c = lv.luma[blk];
                    c[0] = dc[by * 4 + bx];
                    for (k, v) in c.into_iter().enumerate() {
                        r[(by * 4 + k / 4) * 16 + bx * 4 + k % 4] = v;
                    }
                }
                let hor = bypass_dir(pred_mode);
                return add_bypass(&mut y.m()[base..], stride, &mut r, 16, hor, max);
            }
            transform::luma_dc(&mut dc, lsy[0], qp);
            if !T::HIGH {
                // Blocks 2k and 2k + 1 side by side, a pair in one transform.
                let (pairs, _) = lv.luma.as_chunks_mut::<2>();
                for (k, [a, b]) in pairs.iter_mut().enumerate() {
                    let (bx, by) = BLK_XY[2 * k];
                    let dcs = [dc[by * 4 + bx], dc[by * 4 + bx + 1]];
                    if dcs == [0, 0] && mi.cbf >> cbf_bit(plane, 2 * k) & 3 == 0 {
                        continue;
                    }
                    let dst = T::u8s_mut(&mut y.m()[base + by * 4 * stride + bx * 4..]);
                    transform::add4x4x2_dq(dst, stride, a, b, Some(dcs));
                }
                return;
            }
            for blk in 0..16 {
                let (bx, by) = BLK_XY[blk];
                let dcv = dc[by * 4 + bx];
                if dcv == 0 && !cbf(blk) {
                    continue;
                }
                // The AC coefficients are scaled already (Dequant); the DC
                // takes position 0, which they leave free.
                transform::add4x4_dq_dc::<T>(
                    &mut y.m()[base + by * 4 * stride + bx * 4..],
                    stride,
                    &mut lv.luma[blk],
                    dcv,
                    max,
                );
            }
        }
        MbType::IPcm => {}
        _ if mi.transform_8x8 => {
            // Inter with 8x8 transform: prediction is already in place.
            for b8 in 0..4 {
                if mi.cbp_luma >> b8 & 1 != 0 {
                    add_luma8x8::<T>(
                        &mut y,
                        base + 8 * (b8 / 2) * stride + 8 * (b8 % 2),
                        &mut lv.luma8[b8],
                        max,
                        inter_bypass,
                    );
                }
            }
        }
        _ if !T::HIGH && !lossless => {
            // Inter: the prediction is already in place. Blocks 2k and
            // 2k + 1 are side by side: a pair with levels in one transform.
            let (pairs, _) = lv.luma.as_chunks_mut::<2>();
            for (k, [a, b]) in pairs.iter_mut().enumerate() {
                if mi.cbf >> cbf_bit(plane, 2 * k) & 3 == 0 {
                    continue;
                }
                let (bx, by) = BLK_XY[2 * k];
                let dst = T::u8s_mut(&mut y.m()[base + by * 4 * stride + bx * 4..]);
                transform::add4x4x2_dq(dst, stride, a, b, None);
            }
        }
        _ => {
            // Inter: the prediction is already in place.
            for blk in 0..16 {
                if cbf(blk) {
                    let (bx, by) = BLK_XY[blk];
                    add_luma4x4::<T>(
                        &mut y,
                        base + by * 4 * stride + bx * 4,
                        &mut lv.luma[blk],
                        max,
                        inter_bypass,
                    );
                }
            }
        }
    }
}

/// [`reconstruct`] of an 8-bit 4:2:0 inter macroblock outside SP slices,
/// not lossless, the common case: the prediction is in place and the
/// levels are scaled already (Dequant), but for the chroma DCs.
fn reconstruct_inter_420<T: Pixel>(s: &SliceCtx, px: &mut Pix<T>, mi: &MbInfo, res: &mut Residual) {
    {
        let (mut y, base) = px.win(0);
        let stride = y.stride;
        if mi.transform_8x8 {
            for b8 in 0..4 {
                if mi.cbp_luma >> b8 & 1 != 0 {
                    let o = base + 8 * (b8 / 2) * stride + 8 * (b8 % 2);
                    transform::add8x8_dq::<T>(&mut y.m()[o..], stride, &mut res.luma8[b8], 255);
                }
            }
        } else {
            // Blocks 2k and 2k + 1 are side by side: a pair with levels in
            // one transform.
            let (pairs, _) = res.luma.as_chunks_mut::<2>();
            for (k, [a, b]) in pairs.iter_mut().enumerate() {
                if mi.cbf >> cbf_bit(0, 2 * k) & 3 == 0 {
                    continue;
                }
                let (bx, by) = BLK_XY[2 * k];
                let dst = T::u8s_mut(&mut y.m()[base + by * 4 * stride + bx * 4..]);
                transform::add4x4x2_dq(dst, stride, a, b, None);
            }
        }
    }
    if mi.cbp_chroma == 0 {
        return;
    }
    for comp in 0..2 {
        let offset = if comp == 0 {
            s.pps.chroma_qp_index_offset
        } else {
            s.ext.second_chroma_qp_index_offset
        };
        let qpc = chroma_qp(mi.qp_y as i32, offset, 0);
        let lsc = &s.ls.s4[4 + comp][(qpc % 6) as usize];
        let mut dc = [0i32; 4];
        dc.copy_from_slice(&res.chroma_dc[comp][..4]);
        transform::chroma_dc_420(&mut dc, lsc[0], qpc);
        let (mut pl, base) = px.win(1 + comp);
        let stride = pl.stride;
        // As in reconstruct: a row of two blocks per transform.
        let ac = (mi.cbf >> (CBF_CHROMA_AC + 16 * comp as u32)) as u32;
        let (pairs, _) = res.chroma_ac[comp][..4].as_chunks_mut::<2>();
        for (r, [a, b]) in pairs.iter_mut().enumerate() {
            let dst = T::u8s_mut(&mut pl.m()[base + r * 4 * stride..]);
            let dcs = [dc[2 * r], dc[2 * r + 1]];
            if ac >> (2 * r) & 3 != 0 {
                transform::add4x4x2_dq(dst, stride, a, b, Some(dcs));
            } else if dcs != [0, 0] {
                transform::add_dc8x4(dst, stride, dcs);
            }
        }
    }
}

/// Reconstructs a macroblock from its prediction and levels; the blocks
/// the transforms read are left zeroed (see Residual::clear).
fn reconstruct<T: Pixel>(
    s: &SliceCtx,
    px: &mut Pix<T>,
    mbs: &Mbs,
    addr: usize,
    res: &mut Residual,
) {
    let mi = &mbs.info[addr];
    if !T::HIGH
        && !mi.is_intra()
        && s.header.slice_type != SliceType::Sp
        && res.c444.is_none()
        && px.buf.planes.len() == 3
        && px.buf.mb_c == (8, 8)
        && !(s.sps.qpprime_y_zero_transform_bypass && mi.qp_y == 0)
    {
        return reconstruct_inter_420(s, px, mi, res);
    }
    let (bd, bd_c) = (s.bit_depth(0), s.bit_depth(1));
    let max_c = s.max_sample(1);
    // QP'Y = QPY + QpBdOffsetY scales the levels (8.5.12.1).
    let qp_y = mi.qp_y as i32;
    let qp = qp_y + 6 * (bd as i32 - 8);
    let ls = s.ls;
    let intra = mi.is_intra();
    let list = if intra { 0 } else { 3 };
    // P macroblocks of SP slices and SI macroblocks are rebuilt from the
    // transformed prediction (8.6).
    let sp = matches!(mi.mb_type, MbType::Si) || s.header.slice_type == SliceType::Sp && !intra;
    let switching = matches!(mi.mb_type, MbType::Si) || s.header.sp_for_switch;
    let qs = (26 + s.pps.pic_init_qs_minus26 + s.header.slice_qs_delta).clamp(0, 51);
    if sp && !intra {
        let (mut y, base) = px.win(0);
        for blk in 0..16 {
            let (bx, by) = BLK_XY[blk];
            let o = base + by * 4 * y.stride + bx * 4;
            sp_luma4x4::<T>(&mut y, o, &res.luma[blk], s, qp, qs, switching);
        }
    }

    let si = (matches!(mi.mb_type, MbType::Si)).then_some((qs, switching));
    recon_luma_like::<T>(s, px, mbs, addr, 0, qp, res.plane_mut(0), si);
    if let Some(cc) = &mut res.c444 {
        // 4:4:4: Cb and Cr like luma, with QP'C (8.5.8).
        for comp in 0..2 {
            let offset = if comp == 0 {
                s.pps.chroma_qp_index_offset
            } else {
                s.ext.second_chroma_qp_index_offset
            };
            let bd_offset_c = 6 * (bd_c as i32 - 8);
            let qpc = chroma_qp(qp_y, offset, bd_offset_c) + bd_offset_c;
            recon_luma_like::<T>(s, px, mbs, addr, 1 + comp, qpc, cc[comp].view_mut(), None);
        }
        return;
    }

    if px.buf.planes.len() == 1 {
        return;
    }
    let (cw, ch) = px.buf.mb_c;
    // Inter chroma without levels is its prediction.
    if !intra && !sp && mi.cbp_chroma == 0 {
        return;
    }
    // Neighbours for intra prediction.
    let (has_top, left_half, has_corner) = if intra {
        (
            mbs.intra_ok(mbs.b(addr)),
            [
                left_avail(mbs, addr, 0, 0, 8),
                left_avail(mbs, addr, 0, 8, 8),
            ],
            mbs.intra_ok(mbs.d(addr)),
        )
    } else {
        (false, [false; 2], false)
    };
    let has_left = left_half[0] && left_half[1];
    for comp in 0..2 {
        let offset = if comp == 0 {
            s.pps.chroma_qp_index_offset
        } else {
            s.ext.second_chroma_qp_index_offset
        };
        // QP'C = QPC + QpBdOffsetC (8.5.8).
        let bd_offset_c = 6 * (bd_c as i32 - 8);
        let qpc = chroma_qp(qp_y, offset, bd_offset_c) + bd_offset_c;
        let (mut pl, base) = px.win(1 + comp);
        let stride = pl.stride;
        if intra {
            let mut top = [T::default(); 16];
            let mut left = [T::default(); 16];
            if has_top {
                top[..cw].copy_from_slice(&pl.s()[base - stride..base - stride + cw]);
            }
            if left_half[0] || left_half[1] {
                for k in 0..ch {
                    left[k] = pl.s()[base + k * stride - 1];
                }
            }
            let corner = if has_corner {
                pl.s()[base - stride - 1]
            } else {
                T::default()
            };
            let e = EdgeMb {
                top: &top,
                left: &left,
                corner,
                has_top,
                has_left,
                left_half,
                has_corner,
            };
            if !intra::pred_chroma(
                &mut pl.m()[base..],
                stride,
                mi.chroma_pred_mode,
                &e,
                cw,
                ch,
                s.sps.chroma_format_idc,
                bd_c,
            ) {
                unpredicted(pl.m(), base, stride, cw, ch, bd_c);
            }
        }
        if sp {
            let qsc = chroma_qp(qs, offset, 0);
            let lss = &ls.s4[list + 1 + comp][(qsc % 6) as usize];
            let mut pred = [[0i32; 16]; 4];
            for (blk, p) in pred.iter_mut().enumerate() {
                let o = base + blk / 2 * 4 * stride + blk % 2 * 4;
                for (k, v) in p.iter_mut().enumerate() {
                    *v = pl.s()[o + k / 4 * stride + k % 4].get();
                }
            }
            let mut dc_levels = [0i32; 4];
            dc_levels.copy_from_slice(&res.chroma_dc[comp][..4]);
            let pred_dc = pred.map(|p| p.iter().sum::<i32>());
            let dc = crate::sp::chroma_dc(pred_dc, &dc_levels, qpc, qsc, switching);
            for blk in 0..4 {
                let levels = &res.chroma_ac[comp][blk];
                let mut coef = crate::sp::block(&pred[blk], levels, qpc, qsc, switching, true);
                coef[0] = dc[blk];
                transform::scale_4x4(&mut coef, lss, qsc, true);
                let o = base + blk / 2 * 4 * stride + blk % 2 * 4;
                for r in 0..4 {
                    pl.m()[o + r * stride..o + r * stride + 4].fill(T::default());
                }
                transform::add4x4::<T>(&mut pl.m()[o..], stride, &coef, max_c);
            }
            continue;
        }
        if mi.cbp_chroma == 0 {
            continue;
        }
        if s.sps.qpprime_y_zero_transform_bypass && qp_y + 6 * (s.bit_depth(0) as i32 - 8) == 0 {
            // Transform bypass: dcC and r are the levels (8.5.11, 8.5.12),
            // accumulated for horizontal (1) and vertical (2) prediction.
            let l = &res.chroma_dc[comp];
            let dc = if ch == 16 {
                [l[0], l[2], l[1], l[5], l[3], l[6], l[4], l[7]]
            } else {
                [l[0], l[1], l[2], l[3], 0, 0, 0, 0]
            };
            let mut r = [0i32; 8 * 16];
            for blk in 0..ch / 2 {
                let mut c = res.chroma_ac[comp][blk];
                c[0] = dc[blk];
                let (cx, cy) = (blk % 2, blk / 2);
                for (k, v) in c.into_iter().enumerate() {
                    r[(cy * 4 + k / 4) * 8 + cx * 4 + k % 4] = v;
                }
            }
            let hor = match (intra, mi.chroma_pred_mode) {
                (true, 1) => Some(true),
                (true, 2) => Some(false),
                _ => None,
            };
            let dst = &mut pl.m()[base..];
            add_bypass_rect(dst, stride, &mut r, (cw, ch), hor, max_c);
            continue;
        }
        let lsc = &ls.s4[list + 1 + comp][(qpc % 6) as usize];
        let mut dc = [0i32; 8];
        let blocks = if ch == 16 {
            // 4:2:2: ChromaDCLevel into the 4x2 matrix c (8-305).
            let l = &res.chroma_dc[comp];
            dc = [l[0], l[2], l[1], l[5], l[3], l[6], l[4], l[7]];
            let s4 = &ls.s4[list + 1 + comp];
            transform::chroma_dc_422(&mut dc, |m| s4[m][0], qpc);
            8
        } else {
            let mut d4 = [0i32; 4];
            d4.copy_from_slice(&res.chroma_dc[comp][..4]);
            transform::chroma_dc_420(&mut d4, lsc[0], qpc);
            dc[..4].copy_from_slice(&d4);
            4
        };
        if !T::HIGH {
            // Each row of two blocks in one transform (a block without
            // levels adds nothing, one with only a DC its DC), or when
            // neither has AC levels just their DCs: a test per row, not per
            // block, as whether one has levels is unpredictable.
            let ac = (mi.cbf >> (CBF_CHROMA_AC + 16 * comp as u32)) as u32;
            let (pairs, _) = res.chroma_ac[comp][..blocks].as_chunks_mut::<2>();
            for (r, [a, b]) in pairs.iter_mut().enumerate() {
                let dst = T::u8s_mut(&mut pl.m()[base + r * 4 * stride..]);
                let dcs = [dc[2 * r], dc[2 * r + 1]];
                if ac >> (2 * r) & 3 != 0 {
                    transform::add4x4x2_dq(dst, stride, a, b, Some(dcs));
                } else if dcs != [0, 0] {
                    transform::add_dc8x4(dst, stride, dcs);
                }
            }
            continue;
        }
        for blk in 0..blocks {
            let (cx, cy) = (blk % 2, blk / 2);
            // The AC coefficients are scaled already (Dequant); the DC
            // takes position 0, which they leave free.
            transform::add4x4_dq_dc::<T>(
                &mut pl.m()[base + cy * 4 * stride + cx * 4..],
                stride,
                &mut res.chroma_ac[comp][blk],
                dc[blk],
                max_c,
            );
        }
    }
}
