//! CABAC parsing of macroblock-layer syntax elements (9.3.2, 9.3.3.1).
//!
//! Context increments that depend on neighbouring macroblocks (9.3.3.1.1)
//! are computed by the caller and passed in as `inc`; increments that depend
//! on earlier bins of the same element (9.3.3.1.2) are handled here.

use crate::cabac::{Ctx, Decoder};
use crate::transform::Coeffs;

pub type Contexts = [Ctx; 1024];

/// Fixed context base indices (ctxIdxOffset, Table 9-34).
mod off {
    pub const MB_TYPE_SI_PREFIX: usize = 0;
    pub const MB_TYPE_I: usize = 3;
    pub const MB_SKIP_P: usize = 11;
    pub const MB_TYPE_P_PREFIX: usize = 14;
    pub const MB_TYPE_P_SUFFIX: usize = 17;
    pub const SUB_MB_TYPE_P: usize = 21;
    pub const MB_SKIP_B: usize = 24;
    pub const MB_TYPE_B_PREFIX: usize = 27;
    pub const MB_TYPE_B_SUFFIX: usize = 32;
    pub const SUB_MB_TYPE_B: usize = 36;
    pub const MVD: [usize; 2] = [40, 47];
    pub const REF_IDX: usize = 54;
    pub const MB_QP_DELTA: usize = 60;
    pub const INTRA_CHROMA_PRED_MODE: usize = 64;
    pub const PREV_INTRA_PRED_MODE_FLAG: usize = 68;
    pub const REM_INTRA_PRED_MODE: usize = 69;
    pub const MB_FIELD_DECODING_FLAG: usize = 70;
    pub const CBP_LUMA: usize = 73;
    pub const CBP_CHROMA: usize = 77;
    pub const TRANSFORM_SIZE_8X8_FLAG: usize = 399;
}

/// mb_skip_flag (ctxIdxOffset 11 or 24, inc from 9.3.3.1.1.1).
#[inline]
pub fn mb_skip_flag(d: &mut Decoder, c: &mut Contexts, b_slice: bool, inc: usize) -> bool {
    let base = if b_slice {
        off::MB_SKIP_B
    } else {
        off::MB_SKIP_P
    };
    d.decision_call(&mut c[base + inc])
}

/// mb_field_decoding_flag (inc from 9.3.3.1.1.2).
#[inline]
pub fn mb_field_decoding_flag(d: &mut Decoder, c: &mut Contexts, inc: usize) -> bool {
    d.decision_call(&mut c[off::MB_FIELD_DECODING_FLAG + inc])
}

/// The I macroblock type bins of Table 9-36 after b0 = 1, returning the
/// I-slice mb_type value (1..=25). `base` is ctxIdxOffset 3 in I slices, or
/// the P/B suffix offset 17/32. Bins: b1 = I_PCM (ctxIdx 276), b2 = luma
/// CBP 15, b3 = chroma CBP != 0, [b4 = chroma CBP 2 when b3], then two bins of
/// Intra16x16PredMode. Increments follow Tables 9-39 and 9-41.
fn i_mb_type_tail(d: &mut Decoder, c: &mut Contexts, base: usize, in_i_slice: bool) -> u32 {
    if d.terminate() {
        return 25; // I_PCM
    }
    // (b2, b3, b4 when b3 == 1, prediction mode bins)
    let (luma_inc, chroma_inc, chroma2_inc, pred_inc) = if in_i_slice {
        (3, 4, 5, [6, 7])
    } else {
        (1, 2, 2, [3, 3])
    };
    let luma15 = d.decision_call(&mut c[base + luma_inc]) as u32;
    let chroma = if d.decision_call(&mut c[base + chroma_inc]) {
        1 + d.decision_call(&mut c[base + chroma2_inc]) as u32
    } else {
        0
    };
    let hi = d.decision_call(&mut c[base + pred_inc[0]]) as u32;
    let lo = d.decision_call(&mut c[base + pred_inc[1]]) as u32;
    1 + (hi << 1 | lo) + 4 * chroma + 12 * luma15
}

/// mb_type in an I slice (Table 9-36); inc for bin 0 from 9.3.3.1.1.3.
pub fn mb_type_i(d: &mut Decoder, c: &mut Contexts, inc: usize) -> u32 {
    if !d.decision_call(&mut c[off::MB_TYPE_I + inc]) {
        return 0;
    }
    i_mb_type_tail(d, c, off::MB_TYPE_I, true)
}

/// mb_type in an SI slice: prefix (offset 0) then an I-slice suffix (offset 3).
pub fn mb_type_si(d: &mut Decoder, c: &mut Contexts, prefix_inc: usize, suffix_inc: usize) -> u32 {
    if !d.decision_call(&mut c[off::MB_TYPE_SI_PREFIX + prefix_inc]) {
        return 0;
    }
    1 + mb_type_i(d, c, suffix_inc)
}

/// The I suffix used in P and B slices (offsets 17 and 32): b0 at inc 0.
fn intra_suffix(d: &mut Decoder, c: &mut Contexts, base: usize) -> u32 {
    if !d.decision_call(&mut c[base]) {
        return 0;
    }
    i_mb_type_tail(d, c, base, false)
}

/// mb_type in a P or SP slice (Table 9-37); intra types come back as 5..=30.
pub fn mb_type_p(d: &mut Decoder, c: &mut Contexts) -> u32 {
    let b = off::MB_TYPE_P_PREFIX;
    if d.decision_call(&mut c[b]) {
        return 5 + intra_suffix(d, c, off::MB_TYPE_P_SUFFIX);
    }
    if !d.decision_call(&mut c[b + 1]) {
        // "0 0 0" P_L0_16x16, "0 0 1" P_8x8 (bin 2 inc: b1 != 1 -> 2)
        if d.decision_call(&mut c[b + 2]) { 3 } else { 0 }
    } else {
        // "0 1 1" P_L0_L0_16x8, "0 1 0" P_L0_L0_8x16 (bin 2 inc 3)
        if d.decision_call(&mut c[b + 3]) { 1 } else { 2 }
    }
}

/// mb_type in a B slice (Table 9-37); inc for bin 0 from 9.3.3.1.1.3. Intra
/// types come back as 23..=48.
pub fn mb_type_b(d: &mut Decoder, c: &mut Contexts, inc: usize) -> u32 {
    let b = off::MB_TYPE_B_PREFIX;
    if !d.decision_call(&mut c[b + inc]) {
        return 0; // B_Direct_16x16
    }
    if !d.decision_call(&mut c[b + 3]) {
        // "1 0 x": bin 2 inc is (b1 != 0) ? 4 : 5 = 5
        return 1 + d.decision_call(&mut c[b + 5]) as u32;
    }
    let b2 = d.decision_call(&mut c[b + 4]);
    // All later bins use inc 5.
    let mut bits =
        |n: u32| (0..n).fold(0u32, |v, _| v << 1 | d.decision_call(&mut c[b + 5]) as u32);
    if !b2 {
        return 3 + bits(3); // "1 1 0 x y z" -> 3..=10
    }
    if bits(1) == 0 {
        return 12 + bits(3); // "1 1 1 0 x y z" -> 12..=19
    }
    match bits(2) {
        0b11 => 22, // B_8x8
        0b10 => 11, // B_L1_L0_8x16
        0b00 => 20 + bits(1),
        _ => 23 + intra_suffix(d, c, off::MB_TYPE_B_SUFFIX), // "1 1 1 1 0 1"
    }
}

/// sub_mb_type in a P macroblock (Table 9-38).
pub fn sub_mb_type_p(d: &mut Decoder, c: &mut Contexts) -> u32 {
    let b = off::SUB_MB_TYPE_P;
    if d.decision_call(&mut c[b]) {
        return 0;
    }
    if !d.decision_call(&mut c[b + 1]) {
        return 1;
    }
    if d.decision_call(&mut c[b + 2]) { 2 } else { 3 }
}

/// sub_mb_type in a B macroblock (Table 9-38).
pub fn sub_mb_type_b(d: &mut Decoder, c: &mut Contexts) -> u32 {
    let b = off::SUB_MB_TYPE_B;
    if !d.decision_call(&mut c[b]) {
        return 0;
    }
    if !d.decision_call(&mut c[b + 1]) {
        // "1 0 x": bin 2 inc (b1 != 0) ? 2 : 3 = 3
        return 1 + d.decision_call(&mut c[b + 3]) as u32;
    }
    // bin 2 inc 2, later bins inc 3
    if !d.decision_call(&mut c[b + 2]) {
        let v = (d.decision_call(&mut c[b + 3]) as u32) << 1;
        return 3 + (v | d.decision_call(&mut c[b + 3]) as u32);
    }
    if d.decision_call(&mut c[b + 3]) {
        return 11 + d.decision_call(&mut c[b + 3]) as u32;
    }
    let v = (d.decision_call(&mut c[b + 3]) as u32) << 1;
    7 + (v | d.decision_call(&mut c[b + 3]) as u32)
}

/// ref_idx_lX (unary, inc for bin 0 from 9.3.3.1.1.6). Capped at 31 to stay
/// bounded on corrupt data; legal values are at most 31.
pub fn ref_idx(d: &mut Decoder, c: &mut Contexts, inc: usize) -> u32 {
    let b = off::REF_IDX;
    if !d.decision_call(&mut c[b + inc]) {
        return 0;
    }
    let mut v = 1;
    if !d.decision_call(&mut c[b + 4]) {
        return v;
    }
    v += 1;
    while v < 31 && d.decision_call(&mut c[b + 5]) {
        v += 1;
    }
    v
}

/// Exp-Golomb suffix of a UEGk binarization (9.3.2.3), in bypass bins.
#[inline]
fn eg_suffix(d: &mut Decoder, mut k: u32) -> u32 {
    let mut v = 0u32;
    // A k that reaches 32 can only come from corrupt data.
    while k < 31 && d.bypass() {
        v += 1 << k;
        k += 1;
    }
    v + d.bypass_bits(k)
}

bit_ops_fn! {
/// Both components of mvd_lX (UEG3, signed, uCoff 9), with the incs of
/// 9.3.3.1.1.7: one call keeps the engine state in registers across them.
pub fn mvd_pair[][](d: &mut Decoder, c: &mut Contexts, inc: [usize; 2]) -> [i32; 2] {
    let mut e = *d;
    let v = [mvd(&mut e, c, 0, inc[0]), mvd(&mut e, c, 1, inc[1])];
    *d = e;
    v
}
}

/// mvd_lX component `comp`.
#[inline(always)]
fn mvd(d: &mut Decoder, c: &mut Contexts, comp: usize, inc: usize) -> i32 {
    let b = off::MVD[comp];
    if !d.decision(&mut c[b + inc]) {
        return 0;
    }
    // Truncated unary prefix up to 9: bins 1..=4+ use incs 3, 4, 5, 6, 6...
    let mut prefix = 1;
    while prefix < 9 && d.decision(&mut c[b + [3, 4, 5, 6][(prefix - 1).min(3) as usize]]) {
        prefix += 1;
    }
    let abs = if prefix == 9 {
        9 + eg_suffix(d, 3) as i32
    } else {
        prefix
    };
    if d.bypass() { -abs } else { abs }
}

/// mb_qp_delta (unary of the Table 9-3 mapping; inc for bin 0 from 9.3.3.1.1.5).
pub fn mb_qp_delta(d: &mut Decoder, c: &mut Contexts, inc: usize) -> i32 {
    let b = off::MB_QP_DELTA;
    if !d.decision_call(&mut c[b + inc]) {
        return 0;
    }
    let mut k = 1u32;
    if d.decision_call(&mut c[b + 2]) {
        k += 1;
        // Legal values map to at most 2 * 26 + 1 (high bit depth: 2 * 38).
        while k < 128 && d.decision_call(&mut c[b + 3]) {
            k += 1;
        }
    }
    if k & 1 != 0 {
        k.div_ceil(2) as i32
    } else {
        -((k / 2) as i32)
    }
}

/// intra_chroma_pred_mode (TU, cMax 3; inc for bin 0 from 9.3.3.1.1.8).
pub fn intra_chroma_pred_mode(d: &mut Decoder, c: &mut Contexts, inc: usize) -> u8 {
    let b = off::INTRA_CHROMA_PRED_MODE;
    if !d.decision_call(&mut c[b + inc]) {
        return 0;
    }
    if !d.decision_call(&mut c[b + 3]) {
        return 1;
    }
    if d.decision_call(&mut c[b + 3]) { 3 } else { 2 }
}

/// prev_intra{4x4,8x8}_pred_mode_flag and rem_intra{4x4,8x8}_pred_mode:
/// None when the predicted mode is used, else the 3-bit remainder.
pub fn intra_pred_mode(d: &mut Decoder, c: &mut Contexts) -> Option<u8> {
    if d.decision_call(&mut c[off::PREV_INTRA_PRED_MODE_FLAG]) {
        return None;
    }
    // FL binarization: binIdx 0 is the least significant bit.
    let r = &mut c[off::REM_INTRA_PRED_MODE];
    let mut v = d.decision_call(r) as u8;
    v |= (d.decision_call(r) as u8) << 1;
    v |= (d.decision_call(r) as u8) << 2;
    Some(v)
}

/// coded_block_pattern (9.3.2.6). `luma_inc(i, bits_so_far)` gives the inc
/// for 8x8 block i (9.3.3.1.1.4), which may depend on bins already decoded
/// for the same macroblock. `chroma_inc` gives incs for the two chroma bins
/// (without the +4 for bin 1). `has_chroma` is ChromaArrayType 1 or 2.
pub fn coded_block_pattern(
    d: &mut Decoder,
    c: &mut Contexts,
    mut luma_inc: impl FnMut(usize, u8) -> usize,
    chroma_inc: [usize; 2],
    has_chroma: bool,
) -> (u8, u8) {
    let mut luma = 0u8;
    for i in 0..4 {
        let inc = luma_inc(i, luma);
        luma |= (d.decision_call(&mut c[off::CBP_LUMA + inc]) as u8) << i;
    }
    let mut chroma = 0;
    if has_chroma && d.decision_call(&mut c[off::CBP_CHROMA + chroma_inc[0]]) {
        chroma = 1 + d.decision_call(&mut c[off::CBP_CHROMA + 4 + chroma_inc[1]]) as u8;
    }
    (luma, chroma)
}

/// transform_size_8x8_flag (inc from 9.3.3.1.1.10).
#[inline]
pub fn transform_size_8x8_flag(d: &mut Decoder, c: &mut Contexts, inc: usize) -> bool {
    d.decision_call(&mut c[off::TRANSFORM_SIZE_8X8_FLAG + inc])
}

/// Context base indices for one ctxBlockCat (Tables 9-34 and 9-40), already
/// including ctxIdxBlockCatOffset. `sig` and `last` are [frame, field].
#[derive(Clone, Copy)]
struct CatCtx {
    cbf: u16,
    sig: [u16; 2],
    last: [u16; 2],
    abs: u16,
}

const CATS: [CatCtx; 14] = {
    const CBF_OFF: [u16; 14] = [0, 4, 8, 12, 16, 0, 0, 4, 8, 4, 0, 4, 8, 8];
    const SIG_OFF: [u16; 14] = [0, 15, 29, 44, 47, 0, 0, 15, 29, 0, 0, 15, 29, 0];
    const ABS_OFF: [u16; 14] = [0, 10, 20, 30, 39, 0, 0, 10, 20, 0, 0, 10, 20, 0];
    let mut t = [CatCtx {
        cbf: 0,
        sig: [0; 2],
        last: [0; 2],
        abs: 0,
    }; 14];
    let mut cat = 0;
    while cat < 14 {
        // (coded_block_flag, sig frame, sig field, last frame, last field, abs)
        let (cbf, sf, sd, lf, ld, abs) = match cat {
            0..=4 => (85, 105, 277, 166, 338, 227),
            5 => (1012, 402, 436, 417, 451, 426),
            6..=8 => (460, 484, 776, 572, 864, 952),
            9 => (1012, 660, 675, 690, 699, 708),
            10..=12 => (472, 528, 820, 616, 908, 982),
            _ => (1012, 718, 733, 748, 757, 766),
        };
        let s = SIG_OFF[cat];
        t[cat] = CatCtx {
            cbf: cbf + CBF_OFF[cat],
            sig: [sf + s, sd + s],
            last: [lf + s, ld + s],
            abs: abs + ABS_OFF[cat],
        };
        cat += 1;
    }
    t
};

/// Table 9-43: (significant frame, significant field, last) increments for
/// 8x8 blocks (ctxBlockCat 5, 9, 13), by levelListIdx 0..=62.
#[rustfmt::skip]
const SIG8X8: [[u8; 3]; 63] = [
    [0, 0, 0], [1, 1, 1], [2, 1, 1], [3, 2, 1], [4, 2, 1], [5, 3, 1], [5, 3, 1], [4, 4, 1],
    [4, 5, 1], [3, 6, 1], [3, 7, 1], [4, 7, 1], [4, 7, 1], [4, 8, 1], [5, 4, 1], [5, 5, 1],
    [4, 6, 2], [4, 9, 2], [4, 10, 2], [4, 10, 2], [3, 8, 2], [3, 11, 2], [6, 12, 2], [7, 11, 2],
    [7, 9, 2], [7, 9, 2], [8, 10, 2], [9, 10, 2], [10, 8, 2], [9, 11, 2], [8, 12, 2], [7, 11, 2],
    [7, 9, 3], [6, 9, 3], [11, 10, 3], [12, 10, 3], [13, 8, 3], [11, 11, 3], [6, 12, 3], [7, 11, 3],
    [8, 9, 4], [9, 9, 4], [14, 10, 4], [10, 10, 4], [9, 8, 4], [8, 13, 4], [6, 13, 4], [11, 9, 4],
    [12, 9, 5], [13, 10, 5], [11, 10, 5], [6, 8, 5], [9, 13, 6], [14, 13, 6], [10, 9, 6], [9, 9, 6],
    [11, 10, 7], [12, 10, 7], [13, 14, 7], [11, 14, 7], [14, 14, 8], [10, 14, 8], [12, 14, 8],
];

/// coded_block_flag for a block of category `cat` (inc from 9.3.3.1.1.9).
#[inline]
pub fn coded_block_flag(d: &mut Decoder, c: &mut Contexts, cat: usize, inc: usize) -> bool {
    d.decision(&mut c[CATS[cat].cbf as usize + inc])
}

/// residual_block_cabac() without coded_block_flag (7.3.5.3.3): decodes the
/// levels of one block into `coeff` (indexed by levelListIdx, which must be
/// zeroed by the caller) for `num` coefficients (maxNumCoeff, or the AC count).
/// `num_c8x8` is NumC8x8 for chroma DC (ctxBlockCat 3). Returns the number of
/// nonzero coefficients.
pub fn residual_block(
    d: &mut Decoder,
    c: &mut Contexts,
    cat: usize,
    field: bool,
    num: usize,
    num_c8x8: usize,
    coeff: &mut [i32],
) -> u32 {
    // Each level straight to its levelListIdx (the list is zeroed).
    let mut e = *d;
    let out = &mut Direct(&mut coeff[..num]);
    #[cfg(not(test))]
    let n = by_cat!(cat, field, (&mut e, c, num, num_c8x8, out), 0 3 6 10);
    #[cfg(test)]
    let n = by_cat!(cat, field, (&mut e, c, num, num_c8x8, out), 0 1 2 3 4 5 6 7 8 9 10 11 12 13);
    *d = e;
    n
}

/// Levels at their levelListIdx in a list.
struct Direct<'a>(&'a mut [i32]);

impl Levels for Direct<'_> {
    #[inline(always)]
    fn put(&mut self, p: usize, level: i32) {
        self.0[p] = level;
    }
}

bit_ops_fn! {
/// As [`residual_block`] for the blocks other than DC, each level scaled
/// and stored at its place as it is decoded: the number of nonzero levels.
pub fn residual_placed[][](
    d: &mut Decoder,
    c: &mut Contexts,
    cat: usize,
    field: bool,
    num: usize,
    out: &mut Placed,
) -> u32 {
    let mut e = *d;
    let n = by_cat!(cat, field, (&mut e, c, num, 0, out), 1 2 4 5 7 8 9 11 12 13);
    *d = e;
    n
}
}

/// How the levels of a block are scaled and placed (see [`Placed`]), less
/// the block.
#[derive(Clone, Copy)]
pub struct Place<'a> {
    pub scan: &'a [u8],
    pub place: &'a [u8],
    pub ls: &'a [i32],
    pub k: Option<(i32, i32, u32)>,
    pub first: usize,
}

/// The 4x4 luma blocks of a macroblock (ctxBlockCat 1 or 2): in
/// luma4x4BlkIdx order, those of the 8x8 blocks set in `cbp` get their
/// coded_block_flag (incs from the 5x5 `grid` of 9.3.3.1.1.9, updated)
/// and, when set, their levels placed in `dst` and counted in `nnz`.
/// Returns the coded blocks as bits. One call keeps the engine state in
/// registers across the blocks.
#[allow(clippy::too_many_arguments)]
pub fn luma4x4(
    d: &mut Decoder,
    c: &mut Contexts,
    cat: usize,
    field: bool,
    cbp: u8,
    grid: &mut u32,
    p: Place,
    dst: &mut [[i32; 16]; 16],
    nnz: &mut [u8; 16],
) -> u16 {
    match (cat, field) {
        (1, false) => luma4x4_run::<1, false>(d, c, cbp, grid, p, dst, nnz),
        (1, true) => luma4x4_run::<1, true>(d, c, cbp, grid, p, dst, nnz),
        (2, false) => luma4x4_run::<2, false>(d, c, cbp, grid, p, dst, nnz),
        (2, true) => luma4x4_run::<2, true>(d, c, cbp, grid, p, dst, nnz),
        _ => unreachable!("ctxBlockCat {cat}"),
    }
}

bit_ops_fn! {
// Each instantiation on its own: one function with all four inlined
// allocates registers worse.
#[inline(never)]
fn luma4x4_run[const CAT: usize, const FIELD: bool][CAT, FIELD](
    d: &mut Decoder,
    c: &mut Contexts,
    cbp: u8,
    grid: &mut u32,
    p: Place,
    dst: &mut [[i32; 16]; 16],
    nnz: &mut [u8; 16],
) -> u16 {
    // (x, y) of each luma4x4BlkIdx.
    const XY: [(u32, u32); 16] = [
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
    let cbf_base = CATS[CAT].cbf as usize;
    let num = if CAT == 1 { 15 } else { 16 };
    let mut coded = 0u16;
    // A local copy of the engine state can live in registers.
    let mut e = *d;
    let e = &mut e;
    for blk in 0..16 {
        if cbp >> (blk / 4) & 1 == 0 {
            continue;
        }
        // Bit x + 1 + 5 (y + 1) of the grid is block (x, y).
        let (bx, by) = XY[blk];
        let at = bx + 1 + 5 * (by + 1);
        let inc = (*grid >> (at - 1) & 1) + 2 * (*grid >> (at - 5) & 1);
        if e.decision(&mut c[cbf_base + inc as usize]) {
            *grid |= 1 << at;
            coded |= 1 << blk;
            let mut out = Placed {
                dst: &mut dst[blk],
                scan: p.scan,
                place: p.place,
                ls: p.ls,
                k: p.k,
                first: p.first,
            };
            nnz[blk] = residual::<CAT, FIELD, _>(e, c, num, 0, &mut out) as u8;
        }
    }
    *d = *e;
    coded
}
}

/// The 8x8 luma blocks of a macroblock outside 4:4:4 (ctxBlockCat 5,
/// coded_block_flag inferred): those set in `cbp` decoded into `dst`, their
/// counts (at most 16) in `nnz`, in one call as [`luma4x4`].
pub fn luma8x8(
    d: &mut Decoder,
    c: &mut Contexts,
    field: bool,
    cbp: u8,
    p: Place,
    dst: &mut [[i32; 64]; 4],
    nnz: &mut [u8; 16],
) {
    if field {
        luma8x8_run::<true>(d, c, cbp, p, dst, nnz)
    } else {
        luma8x8_run::<false>(d, c, cbp, p, dst, nnz)
    }
}

bit_ops_fn! {
#[inline(never)]
fn luma8x8_run[const FIELD: bool][FIELD](
    d: &mut Decoder,
    c: &mut Contexts,
    cbp: u8,
    p: Place,
    dst: &mut [[i32; 64]; 4],
    nnz: &mut [u8; 16],
) {
    let mut e = *d;
    let e = &mut e;
    for (b8, block) in dst.iter_mut().enumerate() {
        if cbp >> b8 & 1 == 0 {
            continue;
        }
        let mut out = Placed {
            dst: block,
            scan: p.scan,
            place: p.place,
            ls: p.ls,
            k: p.k,
            first: 0,
        };
        // x86 cores prefer the shared copy (A/B: an inlined one +1.5..3.6%
        // there, A78 -0.5%).
        let n = if cfg!(target_arch = "x86_64") {
            residual_placed(e, c, 5, FIELD, 64, &mut out)
        } else {
            residual::<5, FIELD, _>(e, c, 64, 0, &mut out)
        };
        nnz[4 * b8..4 * b8 + 4].fill(n.min(16) as u8);
    }
    *d = *e;
}
}

/// The `n` (4 or 8) chroma AC blocks of a component (ctxBlockCat 4), as
/// [`luma4x4`]: incs from the 3-wide `grid`, updated.
#[allow(clippy::too_many_arguments)]
pub fn chroma_ac(
    d: &mut Decoder,
    c: &mut Contexts,
    field: bool,
    n: usize,
    grid: &mut u32,
    p: Place,
    dst: &mut [[i32; 16]; 8],
) -> u16 {
    if field {
        chroma_ac_run::<true>(d, c, n, grid, p, dst)
    } else {
        chroma_ac_run::<false>(d, c, n, grid, p, dst)
    }
}

bit_ops_fn! {
#[inline(never)]
fn chroma_ac_run[const FIELD: bool][FIELD](
    d: &mut Decoder,
    c: &mut Contexts,
    n: usize,
    grid: &mut u32,
    p: Place,
    dst: &mut [[i32; 16]; 8],
) -> u16 {
    let cbf_base = CATS[4].cbf as usize;
    let mut coded = 0u16;
    let mut e = *d;
    let e = &mut e;
    for (blk, block) in dst.iter_mut().enumerate().take(n) {
        // Bit x + 1 + 3 (y + 1) of the grid is block (x, y).
        let at = blk % 2 + 1 + 3 * (blk / 2 + 1);
        let inc = (*grid >> (at - 1) & 1) + 2 * (*grid >> (at - 3) & 1);
        if e.decision(&mut c[cbf_base + inc as usize]) {
            *grid |= 1 << at;
            coded |= 1 << blk;
            let mut out = Placed {
                dst: block,
                scan: p.scan,
                place: p.place,
                ls: p.ls,
                k: p.k,
                first: p.first,
            };
            residual::<4, FIELD, _>(e, c, 15, 0, &mut out);
        }
    }
    *d = *e;
    coded
}
}

/// `residual::<cat, field>(args)` for the listed ctxBlockCat values.
macro_rules! by_cat {
    ($cat:expr, $field:expr, $args:tt, $($k:literal)*) => {
        match ($cat, $field) {
            $(($k, false) => residual::<$k, false, _> $args,
              ($k, true) => residual::<$k, true, _> $args,)*
            _ => unreachable!("ctxBlockCat {}", $cat),
        }
    };
}
use by_cat;

/// Where residual decoding puts each nonzero level, by levelListIdx.
pub trait Levels {
    fn put(&mut self, p: usize, level: i32);
}

impl Levels for Coeffs {
    #[inline(always)]
    fn put(&mut self, p: usize, level: i32) {
        self.push(p, level);
    }
}

/// Levels scaled (8.5.12.1, 8.5.13.1) and placed in their block as they are
/// decoded.
pub struct Placed<'a> {
    pub dst: &'a mut [i32],
    /// Per scan index: the raster position (for the scaling list) and where
    /// the coefficient goes.
    pub scan: &'a [u8],
    pub place: &'a [u8],
    pub ls: &'a [i32],
    /// (c * ls * m + r) >> s as (m, r, s); None keeps levels as levels.
    pub k: Option<(i32, i32, u32)>,
    /// Scan index of levelListIdx 0 (1 for AC blocks).
    pub first: usize,
}

impl Levels for Placed<'_> {
    #[inline(always)]
    fn put(&mut self, p: usize, level: i32) {
        let k = p + self.first;
        self.dst[self.place[k] as usize] = match self.k {
            Some((m, r, s)) => (level * self.ls[self.scan[k] as usize] * m + r) >> s,
            None => level,
        };
    }
}

/// residual_block for one ctxBlockCat and frame/field coding, so context
/// offsets are constants.
#[inline(always)]
fn residual<const CAT: usize, const FIELD: bool, L: Levels>(
    d: &mut Decoder,
    c: &mut Contexts,
    num: usize,
    num_c8x8: usize,
    coeff: &mut L,
) -> u32 {
    let cc = const { CATS[CAT] };
    let sig_base = cc.sig[FIELD as usize] as usize;
    let last_base = cc.last[FIELD as usize] as usize;
    let is8x8 = matches!(CAT, 5 | 9 | 13);
    // Significant coefficients, as bits by scan position.
    let mut sig = 0u64;
    let mut i = 0;
    let mut last_found = false;
    // The contexts as fixed-size windows (indices masked into them).
    let (sig_ctx, last) = {
        let (lo, hi) = c.split_at_mut(sig_base.max(last_base));
        let (a, b) = if sig_base < last_base {
            (&mut lo[sig_base..], hi)
        } else {
            (hi, &mut lo[last_base..])
        };
        // Every category has at least 15 of each (indices are below 15).
        let a: &mut [Ctx; 15] = (&mut a[..15]).try_into().unwrap();
        let b: &mut [Ctx; 15] = (&mut b[..15]).try_into().unwrap();
        (a, b)
    };
    while i < num - 1 {
        let (si, li) = if CAT == 3 {
            // NumC8x8 is 1 or 2 (4:2:0, 4:2:2): a shift divides by it.
            let k = (i >> (num_c8x8 / 2)).min(2);
            (k, k)
        } else if is8x8 {
            let t = SIG8X8[i];
            (t[FIELD as usize] as usize, t[2] as usize)
        } else {
            (i, i)
        };
        // One top-up covers the significant and last flags.
        d.ensure();
        if d.decision_filled(&mut sig_ctx[si]) {
            sig |= 1 << i;
            if d.decision_filled(&mut last[li]) {
                last_found = true;
                break;
            }
        }
        i += 1;
    }
    if !last_found {
        // The final coefficient is significant by inference.
        sig |= 1 << (num - 1);
    }
    let n = sig.count_ones() as usize;
    let abs_base = cc.abs as usize;
    let gt1_cap = if CAT == 3 { 3 } else { 4 };
    let (mut eq1, mut gt1) = (0u32, 0u32);
    while sig != 0 {
        // Highest frequency first.
        let p = 63 - sig.leading_zeros();
        sig ^= 1 << p;
        let inc0 = if gt1 != 0 {
            0
        } else {
            (1 + eq1).min(4) as usize
        };
        let mut abs_minus1 = 0u32;
        // One top-up covers this bin and, for a level of 1, the sign.
        d.ensure();
        if d.decision_filled(&mut c[abs_base + inc0]) {
            let inc = 5 + gt1.min(gt1_cap) as usize;
            abs_minus1 = 1;
            while abs_minus1 < 14 && d.decision(&mut c[abs_base + inc]) {
                abs_minus1 += 1;
            }
            if abs_minus1 == 14 {
                abs_minus1 += eg_suffix(d, 0);
            }
        }
        if abs_minus1 == 0 {
            eq1 += 1;
        } else {
            gt1 += 1;
        }
        let level = abs_minus1 as i32 + 1;
        // At most one bin since the last top-up.
        coeff.put(p as usize, if d.bypass_filled() { -level } else { level });
    }
    n as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cabac::encoder::Encoder;

    fn contexts() -> Contexts {
        let mut c = [Ctx::default(); 1024];
        for (i, x) in c.iter_mut().enumerate() {
            *x = Ctx::init((((i * 37) % 60) as i8 - 30, ((i * 11) % 120) as i8), 28);
        }
        c
    }

    /// Encodes `bins` as (ctxIdx or usize::MAX for bypass, value) and returns a decoder over them.
    fn stream(bins: &[(usize, bool)]) -> Vec<u8> {
        let mut c = contexts();
        let mut e = Encoder::new();
        for &(ctx, b) in bins {
            if ctx == usize::MAX {
                e.bypass(b);
            } else if ctx == 276 {
                e.terminate(b);
            } else {
                e.decision(&mut c[ctx], b);
            }
        }
        e.terminate(true);
        e.bytes()
    }

    const BYP: usize = usize::MAX;

    #[test]
    fn decodes_b_mb_types_per_table_9_37() {
        // (bins as (ctxIdx, value)), expected mb_type, with bin-0 inc 1.
        let cases: &[(&[(usize, bool)], u32)] = &[
            (&[(28, false)], 0),
            (&[(28, true), (30, false), (32, true)], 2),
            (
                &[
                    (28, true),
                    (30, true),
                    (31, false),
                    (32, true),
                    (32, false),
                    (32, true),
                ],
                8,
            ),
            (
                &[
                    (28, true),
                    (30, true),
                    (31, true),
                    (32, true),
                    (32, true),
                    (32, false),
                ],
                11,
            ),
            (
                &[
                    (28, true),
                    (30, true),
                    (31, true),
                    (32, false),
                    (32, true),
                    (32, true),
                    (32, true),
                ],
                19,
            ),
            (
                &[
                    (28, true),
                    (30, true),
                    (31, true),
                    (32, true),
                    (32, false),
                    (32, false),
                    (32, true),
                ],
                21,
            ),
            (
                &[
                    (28, true),
                    (30, true),
                    (31, true),
                    (32, true),
                    (32, true),
                    (32, true),
                ],
                22,
            ),
            // intra prefix, then I suffix: b0 (32) = 0 -> I_NxN
            (
                &[
                    (28, true),
                    (30, true),
                    (31, true),
                    (32, true),
                    (32, false),
                    (32, true),
                    (32, false),
                ],
                23,
            ),
        ];
        for (bins, want) in cases {
            let data = stream(bins);
            let mut d = Decoder::new(&data, 0);
            let mut c = contexts();
            assert_eq!(mb_type_b(&mut d, &mut c, 1), *want, "{bins:?}");
            assert!(d.terminate());
        }
    }

    #[test]
    fn decodes_i_mb_types_per_table_9_36() {
        // I_16x16_2_1_1 = 19: 1, term 0, luma 1, chroma 1 then 0, pred 1 0
        let data = stream(&[
            (3, true),
            (276, false),
            (6, true),
            (7, true),
            (8, false),
            (9, true),
            (10, false),
        ]);
        let mut d = Decoder::new(&data, 0);
        assert_eq!(mb_type_i(&mut d, &mut contexts(), 0), 19);
        // I_16x16_3_0_0 = 4: chroma 0 -> pred bins use incs 6 and 7
        let data = stream(&[
            (4, true),
            (276, false),
            (6, false),
            (7, false),
            (9, true),
            (10, true),
        ]);
        let mut d = Decoder::new(&data, 0);
        assert_eq!(mb_type_i(&mut d, &mut contexts(), 1), 4);
        // I_PCM
        let data = stream(&[(3, true), (276, true)]);
        let mut d = Decoder::new(&data, 0);
        assert_eq!(mb_type_i(&mut d, &mut contexts(), 0), 25);
    }

    #[test]
    fn decodes_mvd_with_suffix_and_sign() {
        // |mvd| = 9 + 13: prefix of nine 1s, EG3 suffix 13 = "1" (8) then "0" + 4 bits 0101, sign 1.
        let mut bins = vec![(40 + 2, true), (43, true), (44, true), (45, true)];
        bins.extend([(46, true); 5]);
        bins.extend([
            (BYP, true),
            (BYP, false),
            (BYP, false),
            (BYP, true),
            (BYP, false),
            (BYP, true),
            (BYP, true),
        ]);
        let data = stream(&bins);
        let mut d = Decoder::new(&data, 0);
        assert_eq!(mvd(&mut d, &mut contexts(), 0, 2), -22);
    }

    #[test]
    fn decodes_residual_4x4_block() {
        // cat 2, frame: coefficients at 0 (level -3) and 2 (level 1), last at 2.
        let sig = 105 + 29;
        let last = 166 + 29;
        let abs = 227 + 20;
        let bins = [
            (sig, true),
            (last, false),    // pos 0 significant, not last
            (sig + 1, false), // pos 1
            (sig + 2, true),
            (last + 2, true), // pos 2 significant and last
            // reverse order: pos 2: abs-1 = 0 (inc0 = 1), sign +
            (abs + 1, false),
            (BYP, false),
            // pos 0: abs-1 = 2: bin0 inc min(4, 1 + 1) = 2, bins >0 inc 5 + 0
            (abs + 2, true),
            (abs + 5, true),
            (abs + 5, false),
            (BYP, true),
        ];
        let data = stream(&bins);
        let mut d = Decoder::new(&data, 0);
        let mut coeff = [0i32; 16];
        assert_eq!(
            residual_block(&mut d, &mut contexts(), 2, false, 16, 0, &mut coeff),
            2
        );
        assert_eq!(&coeff[..4], &[-3, 0, 1, 0]);
        assert!(d.terminate());
    }
}
