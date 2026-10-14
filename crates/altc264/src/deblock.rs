//! Deblocking filter (8.7). Run once per picture after all slices are
//! decoded. 8-bit samples go through vector kernels, higher bit depths
//! through the per-line filter.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::picture::{Picture, Pixel, RawPlane as PlaneRef};
use crate::pool::Pool;
use crate::simd::{I16x8 as V, transpose8};
use crate::simd8::{U8x16 as B, transpose_in, transpose_out};
use crate::slice_decoder::MbInfo;

/// Per-slice deblocking parameters (7.4.3), indexed by MbInfo::slice.
#[derive(Clone, Copy, Debug)]
pub struct SliceFilter {
    pub disable_idc: u8,
    pub offset_a: i32,
    pub offset_b: i32,
    /// SP or SI slice: treated like intra for boundary strength.
    pub switching: bool,
    pub chroma_qp_offset: [i32; 2],
}

// Table 8-16.
#[rustfmt::skip]
const ALPHA: [u8; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 5, 6, 7, 8, 9, 10, 12, 13,
    15, 17, 20, 22, 25, 28, 32, 36, 40, 45, 50, 56, 63, 71, 80, 90, 101, 113, 127, 144, 162, 182, 203, 226, 255, 255,
];
#[rustfmt::skip]
const BETA: [u8; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4,
    6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18,
];
// Table 8-17, rows bS = 1, 2, 3.
#[rustfmt::skip]
const TC0: [[u8; 52]; 3] = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 6, 6, 7, 8, 9, 10, 11, 13],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7, 8, 8, 10, 11, 12, 13, 15, 17],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 6, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 20, 23, 25],
];

/// tC0 by bS (0 to 4) and indexA: zero for bS 0 and 4, so a lookup needs
/// no test.
#[cfg(not(all(target_arch = "aarch64", not(altc_scalar))))]
#[cfg(not(any(
    all(target_arch = "aarch64", not(altc_scalar)),
    all(target_arch = "x86_64", not(altc_scalar))
)))]
const TC0_BS: [[u8; 52]; 5] = [[0; 52], TC0[0], TC0[1], TC0[2], [0; 52]];

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

/// Filters one line of samples across an edge: `s[o - k * step]` is p_{k-1}
/// and `s[o + k * step]` is q_k (8.7.2.3, 8.7.2.4).
#[inline]
#[cfg(test)]
fn filter_line(
    s: &mut [u8],
    o: usize,
    step: usize,
    bs: u8,
    index_a: usize,
    alpha: i32,
    beta: i32,
    chroma: bool,
) {
    filter_line_t(s, o, step, bs, index_a, alpha, beta, chroma, 8);
}

/// [`filter_line`] for samples of bit depth `bd`: `alpha` and `beta` are
/// already scaled (8-461, 8-462), tC0 is scaled here (8-464).
#[inline]
fn filter_line_t<T: Pixel>(
    s: &mut [T],
    o: usize,
    step: usize,
    bs: u8,
    index_a: usize,
    alpha: i32,
    beta: i32,
    chroma: bool,
    bd: u32,
) {
    let max = (1 << bd) - 1;
    let tc_scale = 1 << (bd - 8);
    let clip = |v: i32| T::clip(v, max);
    let p0 = s[o - step].get();
    let q0 = s[o].get();
    let p1 = s[o - 2 * step].get();
    let q1 = s[o + step].get();
    if !((p0 - q0).abs() < alpha && (p1 - p0).abs() < beta && (q1 - q0).abs() < beta) {
        return;
    }
    if chroma {
        // chromaStyleFilteringFlag: only p0 and q0 change.
        if bs < 4 {
            let tc = TC0[bs as usize - 1][index_a] as i32 * tc_scale + 1;
            let delta = ((((q0 - p0) << 2) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
            s[o - step] = clip(p0 + delta);
            s[o] = clip(q0 - delta);
        } else {
            s[o - step] = T::new((2 * p1 + p0 + q1 + 2) >> 2);
            s[o] = T::new((2 * q1 + q0 + p1 + 2) >> 2);
        }
        return;
    }
    let p2 = s[o - 3 * step].get();
    let q2 = s[o + 2 * step].get();
    let ap = (p2 - p0).abs();
    let aq = (q2 - q0).abs();
    if bs < 4 {
        let tc0 = TC0[bs as usize - 1][index_a] as i32 * tc_scale;
        let tc = tc0 + (ap < beta) as i32 + (aq < beta) as i32;
        let delta = ((((q0 - p0) << 2) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
        s[o - step] = clip(p0 + delta);
        s[o] = clip(q0 - delta);
        if ap < beta {
            s[o - 2 * step] =
                T::new(p1 + ((p2 + ((p0 + q0 + 1) >> 1) - (p1 << 1)) >> 1).clamp(-tc0, tc0));
        }
        if aq < beta {
            s[o + step] =
                T::new(q1 + ((q2 + ((p0 + q0 + 1) >> 1) - (q1 << 1)) >> 1).clamp(-tc0, tc0));
        }
    } else {
        let p3 = s[o - 4 * step].get();
        let q3 = s[o + 3 * step].get();
        let strong = (p0 - q0).abs() < ((alpha >> 2) + 2);
        if ap < beta && strong {
            s[o - step] = T::new((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3);
            s[o - 2 * step] = T::new((p2 + p1 + p0 + q0 + 2) >> 2);
            s[o - 3 * step] = T::new((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3);
        } else {
            s[o - step] = T::new((2 * p1 + p0 + q1 + 2) >> 2);
        }
        if aq < beta && strong {
            s[o] = T::new((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3);
            s[o + step] = T::new((p0 + q0 + q1 + q2 + 2) >> 2);
            s[o + 2 * step] = T::new((2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) >> 3);
        } else {
            s[o] = T::new((2 * q1 + q0 + p1 + 2) >> 2);
        }
    }
}

/// Filters eight lines across an edge (8.7.2.3, 8.7.2.4): `v[k]` holds
/// sample k of each line, p3..p0 then q0..q3 (chroma uses p1..q1, v[2..6]).
/// `on` masks the lines with bS > 0 and `tc0` holds their tC0; `strong` is
/// bS 4, which is uniform along an edge (it only occurs at macroblock
/// edges with an intra side). Returns whether any line was filtered.
#[inline(always)]
fn filter_lines<const CHROMA: bool>(
    v: &mut [V; 8],
    on: V,
    tc0: V,
    strong: bool,
    alpha: V,
    beta: V,
) -> bool {
    let (c1, c2, c4) = (V::splat(1), V::splat(2), V::splat(4));
    let [p3, p2, p1, p0, q0, q1, q2, q3] = *v;
    let f = on
        .and(p0.sub(q0).abs().lt(alpha))
        .and(p1.sub(p0).abs().lt(beta))
        .and(q1.sub(q0).abs().lt(beta));
    if !f.any() {
        return false;
    }
    let weak_p0 = p1.shl::<1>().add(p0).add(q1).add(c2).shr::<2>();
    let weak_q0 = q1.shl::<1>().add(q0).add(p1).add(c2).shr::<2>();
    if CHROMA {
        if strong {
            v[3] = f.select(weak_p0, p0);
            v[4] = f.select(weak_q0, q0);
        } else {
            let tc = tc0.add(c1);
            let delta = q0
                .sub(p0)
                .shl::<2>()
                .add(p1.sub(q1))
                .add(c4)
                .shr::<3>()
                .clamp_sym(tc);
            v[3] = f.select(p0.add(delta), p0);
            v[4] = f.select(q0.sub(delta), q0);
        }
        return true;
    }
    let ap = p2.sub(p0).abs().lt(beta);
    let aq = q2.sub(q0).abs().lt(beta);
    if strong {
        let s = p0.sub(q0).abs().lt(alpha.shr::<2>().add(c2));
        let sp = f.and(s).and(ap);
        let sq = f.and(s).and(aq);
        let pq = p0.add(q0);
        v[1] = sp.select(
            p3.shl::<1>()
                .add(p2.shl::<1>())
                .add(p2)
                .add(p1)
                .add(pq)
                .add(c4)
                .shr::<3>(),
            p2,
        );
        v[2] = sp.select(p2.add(p1).add(pq).add(c2).shr::<2>(), p1);
        v[3] = sp.select(
            p2.add(p1.shl::<1>())
                .add(pq.shl::<1>())
                .add(q1)
                .add(c4)
                .shr::<3>(),
            f.select(weak_p0, p0),
        );
        v[4] = sq.select(
            p1.add(pq.shl::<1>())
                .add(q1.shl::<1>())
                .add(q2)
                .add(c4)
                .shr::<3>(),
            f.select(weak_q0, q0),
        );
        v[5] = sq.select(pq.add(q1).add(q2).add(c2).shr::<2>(), q1);
        v[6] = sq.select(
            q3.shl::<1>()
                .add(q2.shl::<1>())
                .add(q2)
                .add(q1)
                .add(pq)
                .add(c4)
                .shr::<3>(),
            q2,
        );
    } else {
        // Masks are -1 where set: subtracting them adds one.
        let tc = tc0.sub(ap).sub(aq);
        let delta = q0
            .sub(p0)
            .shl::<2>()
            .add(p1.sub(q1))
            .add(c4)
            .shr::<3>()
            .clamp_sym(tc);
        let avg = p0.add(q0).add(c1).shr::<1>();
        let np1 = p1.add(p2.add(avg).sub(p1.shl::<1>()).shr::<1>().clamp_sym(tc0));
        let nq1 = q1.add(q2.add(avg).sub(q1.shl::<1>()).shr::<1>().clamp_sym(tc0));
        v[2] = f.and(ap).select(np1, p1);
        v[3] = f.select(p0.add(delta), p0);
        v[4] = f.select(q0.sub(delta), q0);
        v[5] = f.and(aq).select(nq1, q1);
    }
    true
}

/// Filters the edge whose q0 samples start at `o`, for any sample type:
/// eight lines at a time for 8-bit samples, else line by line.
fn edge_t<T: Pixel, const N: usize>(
    pl: &PlaneRef<T>,
    o: usize,
    vertical: bool,
    bs: [u8; 4],
    index_a: usize,
    alpha: i32,
    beta: i32,
    chroma: bool,
    bd: u32,
) {
    if !T::HIGH {
        return edge::<N>(&pl.bytes(), o, vertical, bs, index_a, alpha, beta, chroma);
    }
    let stride = pl.stride;
    for i in 0..N {
        let b = bs[i / (N / 4)];
        if b == 0 {
            continue;
        }
        // p3..q3 of line i.
        let (at, step) = if vertical {
            (o + i * stride - 4, 1)
        } else {
            (o + i - 4 * stride, stride)
        };
        let mut line = [T::default(); 8];
        for (k, v) in line.iter_mut().enumerate() {
            *v = pl.load::<1>(at + k * step)[0];
        }
        filter_line_t(&mut line, 4, 1, b, index_a, alpha, beta, chroma, bd);
        for (k, v) in line.iter().enumerate() {
            pl.store::<1>(at + k * step, &[*v]);
        }
    }
}

/// Filters the edge whose q0 samples start at `o`: a horizontal edge (rows
/// of N samples) or a vertical edge (N rows), eight lines at a time held in
/// vectors (a vertical edge's rows transposed).
fn edge<const N: usize>(
    pl: &PlaneRef,
    o: usize,
    vertical: bool,
    bs: [u8; 4],
    index_a: usize,
    alpha: i32,
    beta: i32,
    chroma: bool,
) {
    let stride = pl.stride;
    let (alpha, beta) = (V::splat(alpha as i16), V::splat(beta as i16));
    let strong = bs[0] == 4;
    // Per segment: whether it is filtered (a mask) and its tC0.
    let on_seg = bs.map(|b| -((b > 0) as i16));
    let tc0_seg = bs.map(|b| match b {
        1..4 => TC0[b as usize - 1][index_a] as i16,
        _ => 0,
    });
    // The lanes of chunk ch: four lines per segment (16 lines), or two (8).
    let spread = |v: [i16; 4], ch: usize| {
        if N == 16 {
            V::splat(v[2 * ch]).combine_lo(V::splat(v[2 * ch + 1]))
        } else {
            V::from_array([v[0], v[0], v[1], v[1], v[2], v[2], v[3], v[3]])
        }
    };
    let ld = |at: usize| V::load_u8(&pl.load::<8>(at));
    let st = |at: usize, v: V| {
        let mut b = [0u8; 8];
        v.store_u8(&mut b);
        pl.store::<8>(at, &b);
    };
    for ch in 0..N / 8 {
        let (on, tc0) = (spread(on_seg, ch), spread(tc0_seg, ch));
        if vertical {
            // Rows p3..q3 of eight lines, transposed to samples.
            let row = |i: usize| o + (ch * 8 + i) * stride - 4;
            let mut v: [V; 8] = std::array::from_fn(|i| ld(row(i)));
            transpose8(&mut v);
            let filtered = match chroma {
                true => filter_lines::<true>(&mut v, on, tc0, strong, alpha, beta),
                false => filter_lines::<false>(&mut v, on, tc0, strong, alpha, beta),
            };
            if filtered {
                transpose8(&mut v);
                for (i, &r) in v.iter().enumerate() {
                    st(row(i), r);
                }
            }
        } else {
            let row = |k: usize| o + k * stride - 4 * stride + ch * 8;
            let mut v: [V; 8] = std::array::from_fn(|k| ld(row(k)));
            let filtered = match chroma {
                true => filter_lines::<true>(&mut v, on, tc0, strong, alpha, beta),
                false => filter_lines::<false>(&mut v, on, tc0, strong, alpha, beta),
            };
            if filtered {
                // p2..q2 (p0, q0 for chroma) are the only rows that change.
                let ks = if chroma { 3..5 } else { 1..7 };
                for k in ks {
                    st(row(k), v[k]);
                }
            }
        }
    }
}

/// p0' and q0' of the bS < 4 filters (8-470 to 8-472), `tc` given per
/// lane, in signed bytes: delta = Clip3(-tC, tC, (4 (q0 - p0) + (p1 - q1) +
/// 4) >> 3) is exactly the rounded average of q0 - p0 and (p1 - q1) >> 2
/// (the dropped quarter never carries), and where q0 - p0 saturates the
/// result is clamped to tC either way.
#[inline(always)]
fn delta_pq(p1: B, p0: B, q0: B, q1: B, tc: B) -> (B, B) {
    let bias = B::splat(0x80);
    let d = q0.sub(bias).qsub_s(p0.sub(bias));
    let x = d.avg_s(p1.hsub_s(q1).shr1_s());
    let delta = x.min_s(tc).max_s(B::splat(0).sub(tc));
    (p0.adds_us(delta), q0.adds_us(B::splat(0).sub(delta)))
}

/// [`filter_lines`] for sixteen lines of 8-bit samples, as bytes: the
/// decisions and the p1, q1 changes in 16 lanes (the latter exactly: p1 +
/// Clip3(-tC0, tC0, (p2 + avg - 2 p1) >> 1) is the halving average of p2
/// and avg clamped to p1 -+ tC0), the rest in 16 bits. `alpha2` is
/// (alpha >> 2) + 2.
#[inline(always)]
fn filter16<const CHROMA: bool>(
    v: &mut [B; 8],
    on: B,
    tc0: B,
    strong: bool,
    alpha: B,
    beta: B,
    alpha2: B,
) -> bool {
    let [p3, p2, p1, p0, q0, q1, q2, q3] = *v;
    let f = on
        .and(p0.abd(q0).lt(alpha))
        .and(p1.abd(p0).lt(beta))
        .and(q1.abd(q0).lt(beta));
    if !f.any() {
        return false;
    }
    // (2 p1 + p0 + q1 + 2) >> 2, exactly (the sum of p0 and q1 rounded
    // down, then the average rounded up, never differ from it).
    let weak_p0 = p1.avg(p0.havg(q1));
    let weak_q0 = q1.avg(q0.havg(p1));
    if CHROMA {
        if strong {
            v[3] = f.select(weak_p0, p0);
            v[4] = f.select(weak_q0, q0);
        } else {
            let (np0, nq0) = delta_pq(p1, p0, q0, q1, tc0.adds(B::splat(1)));
            v[3] = f.select(np0, p0);
            v[4] = f.select(nq0, q0);
        }
        return true;
    }
    let ap = p2.abd(p0).lt(beta);
    let aq = q2.abd(q0).lt(beta);
    if strong {
        let s = p0.abd(q0).lt(alpha2);
        let (sp, sq) = (f.and(s).and(ap), f.and(s).and(aq));
        let (c2, c4) = (V::splat(2), V::splat(4));
        let half = |w: fn(B) -> V| {
            let [p3, p2, p1, p0, q0, q1, q2, q3] = [p3, p2, p1, p0, q0, q1, q2, q3].map(w);
            let pq = p0.add(q0);
            [
                p3.shl::<1>()
                    .add(p2.shl::<1>())
                    .add(p2)
                    .add(p1)
                    .add(pq)
                    .add(c4)
                    .shr::<3>(),
                p2.add(p1).add(pq).add(c2).shr::<2>(),
                p2.add(p1.shl::<1>())
                    .add(pq.shl::<1>())
                    .add(q1)
                    .add(c4)
                    .shr::<3>(),
                p1.add(pq.shl::<1>())
                    .add(q1.shl::<1>())
                    .add(q2)
                    .add(c4)
                    .shr::<3>(),
                pq.add(q1).add(q2).add(c2).shr::<2>(),
                q3.shl::<1>()
                    .add(q2.shl::<1>())
                    .add(q2)
                    .add(q1)
                    .add(pq)
                    .add(c4)
                    .shr::<3>(),
            ]
        };
        let (lo, hi) = (half(B::lo16), half(B::hi16));
        let n: [B; 6] = std::array::from_fn(|k| B::pack(lo[k], hi[k]));
        v[1] = sp.select(n[0], p2);
        v[2] = sp.select(n[1], p1);
        v[3] = sp.select(n[2], f.select(weak_p0, p0));
        v[4] = sq.select(n[3], f.select(weak_q0, q0));
        v[5] = sq.select(n[4], q1);
        v[6] = sq.select(n[5], q2);
    } else {
        // Masks are 255 where set: subtracting them adds one.
        let tc = tc0.sub(ap).sub(aq);
        let (np0, nq0) = delta_pq(p1, p0, q0, q1, tc);
        let avg = p0.avg(q0);
        let np1 = p2.havg(avg).max(p1.subs(tc0)).min(p1.adds(tc0));
        let nq1 = q2.havg(avg).max(q1.subs(tc0)).min(q1.adds(tc0));
        v[2] = f.and(ap).select(np1, p1);
        v[3] = f.select(np0, p0);
        v[4] = f.select(nq0, q0);
        v[5] = f.and(aq).select(nq1, q1);
    }
    true
}

/// Per lane, for segments of `n` lanes (2 or 4) with bS `bs` and tC0 by
/// indexA from `index_a` (lanes 0..8, 8..16): whether the segment is
/// filtered (bS > 0, a mask) and its tC0. Built in registers: a vector
/// from bytes stored one by one would wait for the stores.
#[inline(always)]
// Each target ends its own block with a `return`.
#[allow(clippy::needless_return)]
fn lanes(bs: [u8; 4], n: u32, index_a: [usize; 2]) -> (B, B) {
    // NEON looks bS and tC0 up with byte shuffles.
    #[cfg(all(target_arch = "aarch64", not(altc_scalar)))]
    {
        // tC0 for bS 0..4 by indexA (bS 0 and 4: 0), packed per indexA.
        const ROWS: [[u8; 8]; 52] = {
            let mut t = [[0u8; 8]; 52];
            let mut ia = 0;
            while ia < 52 {
                t[ia] = [0, TC0[0][ia], TC0[1][ia], TC0[2][ia], 0, 0, 0, 0];
                ia += 1;
            }
            t
        };
        let table = |ia: usize| ROWS[ia];
        let tables = if n == 4 {
            [table(index_a[0]); 2]
        } else {
            [table(index_a[0]), table(index_a[1])]
        };
        crate::simd8::spread_bs(bs, n == 4, tables)
    }
    // SSE2: tC0 chosen by bS with compares, each half's entries splatted.
    #[cfg(all(target_arch = "x86_64", not(altc_scalar)))]
    {
        let spread = |b: u64| if n == 4 { b * 0x0101_0101 } else { b * 0x0101 };
        let lane = if n == 4 { 32 } else { 16 };
        let half = |k: usize| {
            (0..64 / lane).fold(0u64, |a, j| {
                a | spread(bs[(k * 64 / lane + j) % 4] as u64) << (lane * j)
            })
        };
        let b = if n == 4 {
            B::from_u64(half(0), half(1))
        } else {
            let h = half(0);
            B::from_u64(h, h)
        };
        let ia_hi = if n == 4 { index_a[0] } else { index_a[1] };
        let splat = |v: u8| v as u64 * 0x0101_0101_0101_0101;
        let t = |k: usize| B::from_u64(splat(TC0[k][index_a[0]]), splat(TC0[k][ia_hi]));
        let on = B::splat(0).lt(b);
        let tc = b.lt(B::splat(2)).select(
            t(0),
            b.lt(B::splat(3))
                .select(t(1), b.lt(B::splat(4)).select(t(2), B::splat(0))),
        );
        return (on, tc.and(on));
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", not(altc_scalar)),
        all(target_arch = "x86_64", not(altc_scalar))
    )))]
    {
        let tc = |ia: usize| bs.map(|b| TC0_BS[b.min(4) as usize][ia] as u64);
        let on = bs.map(|b| if b > 0 { 0xff } else { 0 });
        let (t0, t1) = (tc(index_a[0]), tc(index_a[1]));
        if n == 4 {
            // Two segments per half (one index for all lanes).
            let half = |v: [u64; 4], h: usize| {
                (v[2 * h] * 0x0101_0101) | (v[2 * h + 1] * 0x0101_0101) << 32
            };
            (
                B::from_u64(half(on, 0), half(on, 1)),
                B::from_u64(half(t0, 0), half(t0, 1)),
            )
        } else {
            // Four segments of two lanes in each half, one index per half.
            let half = |v: [u64; 4]| (0..4).fold(0u64, |a, k| a | (v[k] * 0x0101) << (16 * k));
            (
                B::from_u64(half(on), half(on)),
                B::from_u64(half(t0), half(t1)),
            )
        }
    }
}

/// Filters a 16-line edge of an 8-bit plane whose q0 samples start at `o`
/// (luma, 4:4:4 chroma with the luma filter, or the vertical 4:2:2 chroma
/// edges with the chroma filter): four lines per bS segment.
fn edge16(
    pl: &PlaneRef,
    o: usize,
    vertical: bool,
    bs: [u8; 4],
    index_a: usize,
    alpha: u8,
    beta: u8,
    chroma: bool,
) {
    let stride = pl.stride;
    let (on, tc0) = lanes(bs, 4, [index_a; 2]);
    let (a, b, a2) = (B::splat(alpha), B::splat(beta), B::splat((alpha >> 2) + 2));
    let strong = bs[0] == 4;
    // Called at each site (not through a closure), so it is inlined and
    // the sample vectors stay in registers.
    macro_rules! filter {
        ($v:expr) => {
            if chroma {
                filter16::<true>($v, on, tc0, strong, a, b, a2)
            } else {
                filter16::<false>($v, on, tc0, strong, a, b, a2)
            }
        };
    }
    if vertical {
        // p3..q3 of each row, transposed to the samples of the 16 lines.
        let mut rows: [[u8; 8]; 16] = pl.load_rows(o - 4, stride);
        let mut v = transpose_in(&rows);
        if filter!(&mut v) {
            transpose_out(v, &mut rows);
            pl.store_rows(o - 4, stride, &rows);
        }
    } else {
        let rows: [[u8; 16]; 8] = pl.load_rows(o - 4 * stride, stride);
        let mut v: [B; 8] = std::array::from_fn(|k| B::load(&rows[k]));
        if filter!(&mut v) {
            // p2..q2 (p0, q0 for chroma) are the only rows that change.
            if chroma {
                let out: [[u8; 16]; 2] = std::array::from_fn(|k| {
                    let mut b = [0u8; 16];
                    v[3 + k].store(&mut b);
                    b
                });
                pl.store_rows(o - stride, stride, &out);
            } else {
                let out: [[u8; 16]; 6] = std::array::from_fn(|k| {
                    let mut b = [0u8; 16];
                    v[1 + k].store(&mut b);
                    b
                });
                pl.store_rows(o - 3 * stride, stride, &out);
            }
        }
    }
}

/// Filters an 8-line chroma edge of Cb and Cr at once (4:2:0 edges, and
/// the horizontal 4:2:2 ones): Cb in lanes 0..8, Cr in 8..16, two lines
/// per bS segment, each component with its (indexA, alpha, beta) and q0
/// samples starting at `o`.
fn chroma_edges(
    pls: [&PlaneRef; 2],
    o: [usize; 2],
    vertical: bool,
    bs: [u8; 4],
    params: [(usize, u8, u8); 2],
) {
    let (on, tc0) = lanes(bs, 2, [params[0].0, params[1].0]);
    // Each component's value in its half, built in registers.
    let halves = |f: fn((usize, u8, u8)) -> u8| {
        let splat = |v: u8| v as u64 * 0x0101_0101_0101_0101;
        B::from_u64(splat(f(params[0])), splat(f(params[1])))
    };
    let (a, b) = (halves(|p| p.1), halves(|p| p.2));
    let strong = bs[0] == 4;
    macro_rules! filter {
        ($v:expr) => {
            filter16::<true>($v, on, tc0, strong, a, b, a)
        };
    }
    if vertical {
        let [cb, cr]: [[[u8; 8]; 8]; 2] = [0, 1].map(|c| pls[c].load_rows(o[c] - 4, pls[c].stride));
        let mut rows: [[u8; 8]; 16] =
            std::array::from_fn(|i| if i < 8 { cb[i] } else { cr[i - 8] });
        let mut v = transpose_in(&rows);
        if filter!(&mut v) {
            transpose_out(v, &mut rows);
            for c in 0..2 {
                let half: [[u8; 8]; 8] = rows[8 * c..8 * c + 8].try_into().unwrap();
                pls[c].store_rows(o[c] - 4, pls[c].stride, &half);
            }
        }
    } else {
        // Rows p1..q1 (the others are not read for chroma).
        let [cb, cr]: [[[u8; 8]; 4]; 2] =
            [0, 1].map(|c| pls[c].load_rows(o[c] - 2 * pls[c].stride, pls[c].stride));
        let mut v = [B::splat(0); 8];
        for k in 0..4 {
            v[2 + k] = B::from_halves(&cb[k], &cr[k]);
        }
        if filter!(&mut v) {
            let mut out = [[[0u8; 8]; 2]; 2];
            for k in 0..2 {
                let (lo, hi) = out.split_at_mut(1);
                v[3 + k].store_halves(&mut lo[0][k], &mut hi[0][k]);
            }
            for c in 0..2 {
                pls[c].store_rows(o[c] - pls[c].stride, pls[c].stride, &out[c]);
            }
        }
    }
}

/// Raster mask (bit x + 4y) of the 4x4 luma blocks whose transform block
/// has nonzero coefficients (the whole 8x8 block with the 8x8 transform).
fn coded_mask(m: &MbInfo) -> u16 {
    let mut c = m.cbf as u16;
    if m.transform_8x8 {
        // Whole 8x8 blocks: the four bits of each luma8x8BlkIdx.
        c |= c >> 1;
        c |= c >> 2;
        c = (c & 0x1111) * 0xf;
    }
    // luma4x4BlkIdx has bits y8 x8 y4 x4, the raster position y8 y4 x8 x4:
    // swap bits 1 and 2 of each index.
    (c & 0xc3c3) | (c & 0x0c0c) << 2 | (c & 0x3030) >> 2
}

/// The bS = 1 motion conditions of 8.7.2.1 (mixedModeEdgeFlag = 0).
/// `rp` and `rq` are raster 4x4 block positions. In field macroblocks the
/// vertical limit of 4 quarter frame samples is 2 quarter field samples.
#[inline(always)]
fn motion_differs(p: &MbInfo, rp: usize, q: &MbInfo, rq: usize, field: bool) -> bool {
    let (p8, q8) = ((rp % 4) / 2 + 2 * (rp / 8), (rq % 4) / 2 + 2 * (rq / 8));
    // Identical motion data (the usual case inside a partition) is the same
    // motion, without sorting out the lists used.
    let raw = |m: &MbInfo, b8: usize, b4: usize| {
        [0, 1].map(|l| (m.ref_idx[l][b8] >= 0, m.ref_id[l][b8], m.mv[l][b4]))
    };
    raw(p, p8, rp) != raw(q, q8, rq) && motion_differs_lists(p, rp, q, rq, field)
}

/// The 4 x 4 bit matrix `m` (bit 4 row + column) transposed.
#[inline(always)]
fn transpose_bits4(m: u16) -> u16 {
    let t = (m ^ (m >> 3)) & 0x0a0a;
    let m = m ^ t ^ (t << 3);
    let t = (m ^ (m >> 6)) & 0x00cc;
    m ^ t ^ (t << 6)
}

/// Bit i of `m` as byte i (255 or 0): each byte of `m` in eight lanes,
/// tested against one bit per lane.
#[inline(always)]
fn expand16(m: u16) -> B {
    const SPLAT: u64 = 0x0101_0101_0101_0101;
    const BITS: u64 = 0x8040_2010_0804_0201;
    let v = B::from_u64((m & 0xff) as u64 * SPLAT, (m >> 8) as u64 * SPLAT);
    B::splat(0).lt(v.and(B::from_u64(BITS, BITS)))
}

/// The motion condition for bS 1 of [`motion_differs`] on all of q's luma
/// edges, per direction as bytes 4e + k (1: differs) for segment k of edge
/// e (the outer edge with the left and top neighbours `nb`, whatever the
/// bytes of a missing one).
///
/// Per segment the motion is the same when the blocks either side use the
/// same pictures for the same lists with vectors less than 4 apart (in
/// quarter samples; 2 vertically in fields), or the same pictures for
/// the other lists with those vectors close (8.7.2.1): the two pairings,
/// evaluated for all 32 segments in vectors. Unused lists have zero
/// vectors (the defaults, which direct prediction also sets), so a pairing
/// whose pictures match compares them only where neither side uses them.
fn motion_bytes(nb: [Option<&MbInfo>; 2], q: &MbInfo, field: bool) -> [B; 2] {
    let ly = if field { 1 } else { 3 };
    let thr = u64::from_le_bytes([3, ly, 3, ly, 3, ly, 3, ly]);
    let (thr, one, zero) = (B::from_u64(thr, thr), B::splat(1), V::splat(0));
    // Per segment of the four edges (vectors p[e], q[e] of the blocks
    // either side): 1 where x or y differ by the threshold or more, from
    // saturated byte differences folded per segment as 16-bit lanes.
    let far = |p: [V; 4], q: [V; 4]| {
        let d = |i: usize| p[i].sub(q[i]).abs();
        let a = B::pack(d(0), d(1)).subs(thr).min(one);
        let b = B::pack(d(2), d(3)).subs(thr).min(one);
        B::pack(a.as_i16(), b.as_i16())
    };
    let uses1 = |m: &MbInfo| m.ref_idx[1] != [-1; 4];
    let two = uses1(q) || nb.iter().flatten().any(|m| uses1(m));
    let rows = |m: &MbInfo, l: usize| {
        let f = m.mv[l].as_flattened();
        [0, 1, 2, 3].map(|y| V::from_array(f[8 * y..8 * y + 8].try_into().unwrap()))
    };
    // For list l: (p, q) of the vertical edges, then of the horizontal ones.
    let sides = |l: usize| {
        let r = rows(q, l);
        let (t0, t1) = (r[0].zip_lo32(r[1]), r[0].zip_hi32(r[1]));
        let (t2, t3) = (r[2].zip_lo32(r[3]), r[2].zip_hi32(r[3]));
        let c = [
            t0.combine_lo(t2),
            t0.combine_hi(t2),
            t1.combine_lo(t3),
            t1.combine_hi(t3),
        ];
        // Column 3 of the left neighbour, from its rows.
        let a3 = nb[0].map_or(zero, |a| {
            let r = rows(a, l);
            r[0].zip_hi32(r[1]).combine_hi(r[2].zip_hi32(r[3]))
        });
        let b3 = nb[1].map_or(zero, |b| rows(b, l)[3]);
        ([[a3, c[0], c[1], c[2]], c], [[b3, r[0], r[1], r[2]], r])
    };
    // The pictures of the lists per 8x8 block (unused: u32::MAX, which no
    // picture id is), and with the lists swapped.
    let pic = |m: &MbInfo, l: usize, b8: usize| {
        if m.ref_idx[l][b8] >= 0 {
            m.ref_id[l][b8]
        } else {
            u32::MAX
        }
    };
    let key = |m: &MbInfo, b8: usize| pic(m, 0, b8) as u64 | (pic(m, 1, b8) as u64) << 32;
    let kq = [0, 1, 2, 3].map(|b| key(q, b));
    // The neighbours' 8x8 blocks next to q (any keys when missing).
    let kn = [(0, [1, 3]), (1, [2, 3])]
        .map(|(d, bs): (usize, [usize; 2])| nb[d].map_or([u64::MAX; 2], |m| bs.map(|b| key(m, b))));
    // Per direction, bit 2e + h (edge e, half h of its segments) of each
    // pairing's picture match, as bytes 4e + 2h and 4e + 2h + 1. Edges 1
    // and 3 lie inside an 8x8 block.
    let refs = |dir: usize| {
        let (mut par, mut cross) = (0u16, 0u16);
        for h in 0..2 {
            let (a, b) = if dir == 0 {
                (2 * h, 2 * h + 1)
            } else {
                (h, h + 2)
            };
            let pairs = [
                (kn[dir][h], kq[a]),
                (kq[a], kq[a]),
                (kq[a], kq[b]),
                (kq[b], kq[b]),
            ];
            for (e, (kp, kq)) in pairs.into_iter().enumerate() {
                par |= ((kp == kq) as u16) << (2 * e + h);
                cross |= ((kp == kq.rotate_left(32)) as u16) << (2 * e + h);
            }
        }
        let pairs = |x: u16| {
            let x = (x | x << 4) & 0x0f0f;
            let x = (x | x << 2) & 0x3333;
            let x = (x | x << 1) & 0x5555;
            expand16(x | x << 1)
        };
        (pairs(par), if two { pairs(cross) } else { B::splat(0) })
    };
    let s0 = sides(0);
    let s1 = two.then(|| sides(1));
    [0, 1].map(|dir| {
        let [p0, q0] = if dir == 0 { s0.0 } else { s0.1 };
        let (par_ref, cross_ref) = refs(dir);
        let same = |x: B| x.lt(one);
        let ok = match s1 {
            None => par_ref.and(same(far(p0, q0))),
            Some(s1) => {
                let [p1, q1] = if dir == 0 { s1.0 } else { s1.1 };
                let par = far(p0, q0).max(far(p1, q1));
                let cross = far(p0, q1).max(far(p1, q0));
                par_ref.and(same(par)).max(cross_ref.and(same(cross)))
            }
        };
        one.sub(ok.min(one))
    })
}

/// [`motion_differs`] by the reference pictures and vectors of the lists used.
#[inline(never)]
fn motion_differs_lists(p: &MbInfo, rp: usize, q: &MbInfo, rq: usize, field: bool) -> bool {
    let (p8, q8) = ((rp % 4) / 2 + 2 * (rp / 8), (rq % 4) / 2 + 2 * (rq / 8));
    // (reference picture, motion vector) per used list.
    let used = |m: &MbInfo, b8: usize, b4: usize| {
        let mut v = [(0u32, [0i16; 2]); 2];
        let mut n = 0;
        for l in 0..2 {
            if m.ref_idx[l][b8] >= 0 {
                v[n] = (m.ref_id[l][b8], m.mv[l][b4]);
                n += 1;
            }
        }
        (v, n)
    };
    let (pv, pn) = used(p, p8, rp);
    let (qv, qn) = used(q, q8, rq);
    // Identical motion (the usual case inside a partition).
    if pn == qn && pv == qv {
        return false;
    }
    if pn != qn {
        return true;
    }
    let limit = if field { 2 } else { 4 };
    let far = |a: [i16; 2], b: [i16; 2]| {
        (a[0] as i32 - b[0] as i32).abs() >= 4 || (a[1] as i32 - b[1] as i32).abs() >= limit
    };
    match pn {
        0 => false,
        1 => pv[0].0 != qv[0].0 || far(pv[0].1, qv[0].1),
        _ => {
            let (a, b) = (pv[0].0, pv[1].0);
            let (c, d) = (qv[0].0, qv[1].0);
            if !((a == c && b == d) || (a == d && b == c)) {
                return true; // different reference pictures
            }
            if a != b {
                // Two different references: compare the vectors for each picture.
                if a == c {
                    far(pv[0].1, qv[0].1) || far(pv[1].1, qv[1].1)
                } else {
                    far(pv[0].1, qv[1].1) || far(pv[1].1, qv[0].1)
                }
            } else {
                // Both vectors use the same picture: differ in both pairings.
                (far(pv[0].1, qv[0].1) || far(pv[1].1, qv[1].1))
                    && (far(pv[0].1, qv[1].1) || far(pv[1].1, qv[0].1))
            }
        }
    }
}

/// indexA, alpha and beta (8.7.2.2) of an edge between sides of QPs qp_p
/// and qp_q, for bit depth `bd`.
#[inline(always)]
fn thresholds(qp_p: i32, qp_q: i32, f: &SliceFilter, bd: u32) -> (usize, i32, i32) {
    let qp_av = (qp_p + qp_q + 1) >> 1;
    let index_a = (qp_av + f.offset_a).clamp(0, 51) as usize;
    let index_b = (qp_av + f.offset_b).clamp(0, 51) as usize;
    (
        index_a,
        (ALPHA[index_a] as i32) << (bd - 8),
        (BETA[index_b] as i32) << (bd - 8),
    )
}

/// Deblocks one macroblock: its left and internal vertical edges, then its
/// top and internal horizontal edges (8.7).
fn deblock_mb<T: Pixel>(
    planes: &[PlaneRef<T>],
    w: usize,
    info: &[MbInfo],
    sf: &[SliceFilter],
    chroma_array_type: u8,
    field: bool,
    bit_depth: [u32; 2],
    addr: usize,
    xy: (usize, usize),
) {
    // alpha and beta scale with the bit depth (8-461, 8-462).
    let [bd, bd_c] = bit_depth;
    let bd_offset_c = 6 * (bd_c as i32 - 8);
    let q = &info[addr];
    if q.slice == u32::MAX {
        return; // never decoded (error concealment is separate)
    }
    let f = sf[q.slice as usize];
    if f.disable_idc == 1 {
        return;
    }
    let (mx, my) = xy;
    let left = (mx > 0)
        .then(|| addr - 1)
        .filter(|&a| info[a].slice != u32::MAX && (f.disable_idc != 2 || info[a].slice == q.slice));
    let top = (my > 0)
        .then(|| addr - w)
        .filter(|&a| info[a].slice != u32::MAX && (f.disable_idc != 2 || info[a].slice == q.slice));

    // With indexA or indexB below 16, alpha or beta is 0 (Table 8-16) and
    // no line of the edge is filtered: low QPs leave whole macroblocks as
    // they are, before any bS is worked out.
    let open = |p: i32, q: i32| {
        let av = (p + q + 1) >> 1;
        av + f.offset_a >= 16 && av + f.offset_b >= 16
    };
    let qps = |m: &MbInfo| {
        let o = sf[m.slice as usize].chroma_qp_offset;
        let y = m.qp_y as i32;
        [
            y,
            chroma_qp(y, o[0], bd_offset_c),
            chroma_qp(y, o[1], bd_offset_c),
        ]
    };
    let n_planes = if chroma_array_type == 0 { 1 } else { 3 };
    let qq = qps(q);
    let side_open = |n: Option<usize>| {
        n.is_some_and(|n| {
            let p = qps(&info[n]);
            (0..n_planes).any(|i| open(p[i], qq[i]))
        })
    };
    if !(0..n_planes).any(|i| open(qq[i], qq[i])) && !side_open(left) && !side_open(top) {
        return;
    }

    // bS (8.7.2.1) for each vertical (dir 0) and horizontal (dir 1)
    // luma edge, per 4-sample segment. mixedModeEdgeFlag is 0 (frames).
    let intra = |m: &MbInfo| m.mb_type.is_intra() || sf[m.slice as usize].switching;
    let q_intra = intra(q);
    let cq = coded_mask(q);
    // Per direction and edge, bS of the four segments in the bytes of a word.
    let mut bs = [[0u32; 4]; 2];
    // Per direction, raster masks of the 4x4 blocks whose edge (left for
    // vertical edges, top for horizontal ones) is filtered with bS from
    // coefficients or motion, and of those with coefficients either side.
    let mut edges = [0u16; 2];
    let mut coded = [0u16; 2];
    let mut nbs = [None; 2];
    for (dir, nb) in [(0, left), (1, top)] {
        // Raster block of segment k on edge e.
        let at = |e: usize, k: usize| if dir == 0 { e + 4 * k } else { k + 4 * e };
        // The blocks along edge 0 and the internal edges.
        let (outer, inner) = if dir == 0 {
            (0x1111, 0xeeee)
        } else {
            (0x000f, 0xfff0)
        };
        if let Some(a) = nb {
            let p = &info[a];
            if q_intra || intra(p) {
                // Horizontal edges between field macroblocks get 3.
                bs[dir][0] = if field && dir == 1 {
                    0x03030303
                } else {
                    0x04040404
                };
            } else {
                // The p block is on the far side of the neighbour.
                let cp = coded_mask(p);
                let cp = if dir == 0 { cp >> 3 } else { cp >> 12 };
                edges[dir] |= outer;
                coded[dir] |= (cq | cp) & outer;
                nbs[dir] = Some(p);
            }
        }
        if q_intra {
            for e in 1..4 {
                if !(e % 2 == 1 && q.transform_8x8 && !(chroma_array_type == 2 && dir == 1)) {
                    bs[dir][e] = 0x03030303;
                }
            }
            continue;
        }
        // Luma skips the edges inside 8x8 transform blocks, but 4:2:2
        // chroma filters horizontal edges at every 4 rows with the
        // strength of the luma edge there (8.7, 8.7.2).
        let mut inner = inner;
        if q.transform_8x8 && !(chroma_array_type == 2 && dir == 1) {
            inner &= (0..4).fold(0, |m, k| m | 1 << at(2, k));
        }
        edges[dir] |= inner;
        let shifted = if dir == 0 { cq << 1 } else { cq << 4 };
        coded[dir] |= (cq | shifted) & inner;
    }
    if edges != [0; 2] {
        // Motion decides where coefficients do not.
        let motion = if edges[0] & !coded[0] == 0 && edges[1] & !coded[1] == 0 {
            [B::splat(0); 2]
        } else {
            motion_bytes(nbs, q, field)
        };
        for dir in 0..2 {
            // Raster bits as bytes 4e + k (segment k of edge e).
            let ek = |m: u16| expand16(if dir == 0 { transpose_bits4(m) } else { m });
            let (present, c) = (ek(edges[dir]), ek(coded[dir]));
            let v = present.and(c.select(B::splat(2), motion[dir]));
            let mut b = [0u8; 16];
            v.store(&mut b);
            for e in 0..4 {
                bs[dir][e] |=
                    u32::from_le_bytes([b[4 * e], b[4 * e + 1], b[4 * e + 2], b[4 * e + 3]]);
            }
        }
    }

    // Nothing to filter (most skipped macroblocks).
    if bs == [[0; 4]; 2] {
        return;
    }

    // Luma, and in 4:4:4 Cb and Cr on the same edges with the luma filter
    // (chromaStyleFilteringFlag 0) but their own QP (8.7.2.2).
    let luma_like = if chroma_array_type == 3 { 3 } else { 1 };
    for (i, plane) in planes.iter().enumerate().take(luma_like) {
        let stride = plane.stride;
        let base = my * 16 * stride + mx * 16;
        let bd = if i == 0 { bd } else { bd_c };
        let qp = |m: &MbInfo| match i {
            0 => m.qp_y as i32,
            _ => {
                let offset = sf[m.slice as usize].chroma_qp_offset[i - 1];
                chroma_qp(m.qp_y as i32, offset, bd_offset_c)
            }
        };
        // Thresholds by side (left, top, inside q), once per macroblock.
        let qq = qp(q);
        let inner = thresholds(qq, qq, &f, bd);
        let side = [left, top].map(|n| n.map(|n| thresholds(qp(&info[n]), qq, &f, bd)));
        for dir in 0..2 {
            for e in 0..4 {
                let th = if e == 0 { side[dir] } else { Some(inner) };
                let Some((index_a, alpha, beta)) = th else {
                    continue;
                };
                if bs[dir][e] == 0 || e % 2 == 1 && q.transform_8x8 || alpha == 0 || beta == 0 {
                    continue;
                }
                let o = if dir == 0 {
                    base + 4 * e
                } else {
                    base + 4 * e * stride
                };
                if T::HIGH {
                    edge_t::<T, 16>(
                        plane,
                        o,
                        dir == 0,
                        bs[dir][e].to_le_bytes(),
                        index_a,
                        alpha,
                        beta,
                        false,
                        bd,
                    );
                } else {
                    let (a, b) = (alpha as u8, beta as u8);
                    edge16(
                        &plane.bytes(),
                        o,
                        dir == 0,
                        bs[dir][e].to_le_bytes(),
                        index_a,
                        a,
                        b,
                        false,
                    );
                }
            }
        }
    }

    if !matches!(chroma_array_type, 1 | 2) {
        return;
    }
    // Chroma edges at 0 and 4 horizontally map to luma edges 0 and 8; with
    // 4:2:2 the 16 rows have edges at 0, 4, 8, 12 like luma.
    let ch = if chroma_array_type == 2 { 16 } else { 8 };
    let edges: [&[(usize, usize)]; 2] = if ch == 16 {
        [&[(0, 0), (1, 2)], &[(0, 0), (1, 1), (2, 2), (3, 3)]]
    } else {
        [&[(0, 0), (1, 2)], &[(0, 0), (1, 2)]]
    };
    // Chroma QPs of q, A and B and the thresholds by side, once.
    let qpc = |m: &MbInfo| {
        let o = sf[m.slice as usize].chroma_qp_offset;
        [0, 1].map(|c| chroma_qp(m.qp_y as i32, o[c], bd_offset_c))
    };
    let qq = qpc(q);
    let inner = [0, 1].map(|c| thresholds(qq[c], qq[c], &f, bd_c));
    let side = [left, top].map(|n| {
        n.map(|n| {
            let qp = qpc(&info[n]);
            [0, 1].map(|c| thresholds(qp[c], qq[c], &f, bd_c))
        })
    });
    for dir in 0..2 {
        for &(ce, le) in edges[dir] {
            let th = if le == 0 { side[dir] } else { Some(inner) };
            let Some(th) = th else { continue };
            let b = bs[dir][le].to_le_bytes();
            if b == [0; 4] || th.iter().all(|&(_, alpha, beta)| alpha == 0 || beta == 0) {
                continue;
            }
            // Per component: indexA, alpha, beta and where q0 starts.
            let params = [0, 1].map(|comp| {
                let (index_a, alpha, beta) = th[comp];
                let stride = planes[1 + comp].stride;
                let base = my * ch * stride + mx * 8;
                let o = if dir == 0 {
                    base + 4 * ce
                } else {
                    base + 4 * ce * stride
                };
                (index_a, alpha, beta, o)
            });
            let vert16 = dir == 0 && ch == 16;
            if !T::HIGH && !vert16 {
                let pls = [&planes[1].bytes(), &planes[2].bytes()];
                let os = params.map(|p| p.3);
                let ps = params.map(|p| (p.0, p.1 as u8, p.2 as u8));
                chroma_edges(pls, os, dir == 0, b, ps);
                continue;
            }
            for (comp, &(index_a, alpha, beta, o)) in params.iter().enumerate() {
                let plane = &planes[1 + comp];
                if !T::HIGH {
                    let (a, bt) = (alpha as u8, beta as u8);
                    edge16(&plane.bytes(), o, true, b, index_a, a, bt, true);
                } else if vert16 {
                    edge_t::<T, 16>(plane, o, true, b, index_a, alpha, beta, true, bd_c);
                } else {
                    edge_t::<T, 8>(plane, o, dir == 0, b, index_a, alpha, beta, true, bd_c);
                }
            }
        }
    }
}

/// Deblocks a picture, with up to `threads` rows of macroblocks at a time.
///
/// Macroblock (x, y) reads and writes its own samples, the three columns
/// next to its left edge and the three rows above its top edge. The latter
/// are last changed by (x + 1, y - 1)'s left edge, so a row may filter
/// macroblock x once the row above has finished x + 1; rows then never
/// touch the same samples at the same time.
pub(crate) fn deblock_picture(
    pic: &mut Picture,
    info: &[MbInfo],
    sf: &[SliceFilter],
    chroma_array_type: u8,
    field: bool,
    pool: Option<&Pool>,
) {
    if pic.high() {
        deblock_picture_t::<u16>(pic, info, sf, chroma_array_type, field, pool);
    } else {
        deblock_picture_t::<u8>(pic, info, sf, chroma_array_type, field, pool);
    }
}

fn deblock_picture_t<T: Pixel>(
    pic: &mut Picture,
    info: &[MbInfo],
    sf: &[SliceFilter],
    chroma_array_type: u8,
    field: bool,
    pool: Option<&Pool>,
) {
    let (w, h) = (pic.width_mbs, pic.height_mbs);
    let bd = pic.bit_depth;
    // Rows of macroblocks at a time: the wavefront below keeps them apart.
    let planes: Vec<PlaneRef<T>> = pic.planes.iter_mut().map(PlaneRef::new).collect();
    let threads = pool.map_or(1, |p| p.size().min(h));
    let Some(pool) = pool.filter(|_| threads > 1) else {
        for y in 0..h {
            for x in 0..w {
                deblock_mb(
                    &planes,
                    w,
                    info,
                    sf,
                    chroma_array_type,
                    field,
                    bd,
                    y * w + x,
                    (x, y),
                );
            }
        }
        return;
    };
    let done: Vec<AtomicUsize> = (0..h).map(|_| AtomicUsize::new(0)).collect();
    let (planes, done) = (&planes, &done);
    pool.broadcast(|t| {
        for y in (t..h).step_by(threads) {
            for x in 0..w {
                if y > 0 {
                    let need = (x + 2).min(w);
                    let mut spins = 0u32;
                    while done[y - 1].load(Ordering::Acquire) < need {
                        spins += 1;
                        if spins < 64 {
                            std::hint::spin_loop();
                        } else {
                            std::thread::yield_now();
                        }
                    }
                }
                deblock_mb(
                    planes,
                    w,
                    info,
                    sf,
                    chroma_array_type,
                    field,
                    bd,
                    y * w + x,
                    (x, y),
                );
                done[y].store(x + 1, Ordering::Release);
            }
        }
    });
}

/// Deblocks the macroblock rows `rows` of a w-macroblock-wide picture (not
/// MBAFF), top to bottom, on one thread: rows above must be done.
#[allow(clippy::too_many_arguments)]
pub(crate) fn deblock_rows<T: Pixel>(
    planes: &[PlaneRef<T>],
    w: usize,
    info: &[MbInfo],
    sf: &[SliceFilter],
    chroma_array_type: u8,
    field: bool,
    bit_depth: [u32; 2],
    rows: std::ops::Range<usize>,
) {
    for y in rows {
        for x in 0..w {
            deblock_mb(
                planes,
                w,
                info,
                sf,
                chroma_array_type,
                field,
                bit_depth,
                y * w + x,
                (x, y),
            );
        }
    }
}

/// bS (8.7.2.1) between the sample p0 at (x, y) of macroblock p.0 and q0
/// of macroblock q.0 in an MBAFF frame.
fn bs_mbaff(
    info: &[MbInfo],
    sf: &[SliceFilter],
    p: (usize, usize, usize),
    q: (usize, usize, usize),
    vertical: bool,
    mb_edge: bool,
) -> u8 {
    let (pm, qm) = (&info[p.0], &info[q.0]);
    let intra = |m: &MbInfo| m.mb_type.is_intra() || sf[m.slice as usize].switching;
    if intra(pm) || intra(qm) {
        return if mb_edge && (vertical || !pm.field && !qm.field) {
            4
        } else {
            3
        };
    }
    let raster = |x: usize, y: usize| x / 4 + 4 * (y / 4);
    if (coded_mask(pm) >> raster(p.1, p.2) | coded_mask(qm) >> raster(q.1, q.2)) & 1 != 0 {
        return 2;
    }
    if pm.field != qm.field {
        return 1; // mixedModeEdgeFlag
    }
    motion_differs(pm, raster(p.1, p.2), qm, raster(q.1, q.2), qm.field) as u8
}

/// Deblocks an MBAFF frame (8.7, 8.7.1): one sample line at a time, as
/// edges between frame and field macroblock pairs change macroblock, QP and
/// strength from line to line. Field macroblocks are stored interleaved,
/// so their rows are 2 picture rows apart (dy of 8.7.1).
pub(crate) fn deblock_mbaff(
    pic: &mut Picture,
    info: &[MbInfo],
    sf: &[SliceFilter],
    chroma_array_type: u8,
) {
    let (w, h, bd) = (pic.width_mbs, pic.height_mbs, pic.bit_depth);
    if pic.high() {
        let planes: Vec<PlaneRef<u16>> = pic.planes.iter_mut().map(PlaneRef::new).collect();
        // SAFETY: from a &mut Picture.
        unsafe { deblock_mbaff_t(&planes, w, h, bd, info, sf, chroma_array_type) };
    } else {
        let planes: Vec<PlaneRef<u8>> = pic.planes.iter_mut().map(PlaneRef::new).collect();
        // SAFETY: from a &mut Picture.
        unsafe { deblock_mbaff_t(&planes, w, h, bd, info, sf, chroma_array_type) };
    }
}

/// deblock_mbaff on planes of a w x h macroblock picture.
///
/// # Safety
/// No other thread may access the planes meanwhile.
pub(crate) unsafe fn deblock_mbaff_t<T: Pixel>(
    planes: &[PlaneRef<T>],
    w: usize,
    h: usize,
    bit_depth: [u32; 2],
    info: &[MbInfo],
    sf: &[SliceFilter],
    chroma_array_type: u8,
) {
    let [bd, bd_c] = bit_depth;
    let bd_offset_c = 6 * (bd_c as i32 - 8);
    for addr in 0..w * h {
        let q = &info[addr];
        if q.slice == u32::MAX {
            continue;
        }
        let f = sf[q.slice as usize];
        if f.disable_idc == 1 {
            continue;
        }
        let (px, py, bottom) = (addr / 2 % w, addr / 2 / w, addr % 2);
        let pair = |dx: i32, dy: i32| {
            let (x, y) = (px as i32 + dx, py as i32 + dy);
            (x >= 0 && x < w as i32 && y >= 0).then(|| 2 * (y as usize * w + x as usize))
        };
        let same_slice = |n: Option<usize>| {
            n.is_some_and(|n| {
                info[n].slice != u32::MAX && (f.disable_idc != 2 || info[n].slice == q.slice)
            })
        };
        let locate = |xn: i32, yn: i32, mw: i32, mh: i32| {
            crate::slice_decoder::mbaff_neighbour(addr, q.field, xn, yn, mw, mh, pair, |n| {
                info[n].field
            })
        };
        let filter_left = same_slice(pair(-1, 0));
        let filter_top = if !q.field && bottom == 1 {
            true // the top macroblock of the same frame pair
        } else {
            same_slice(pair(0, -1))
        };
        // Top-left sample and row distance of a macroblock in a plane.
        let origin = |n: usize, mw: usize, mh: usize| {
            let (x, y, b) = (n / 2 % w, n / 2 / w, n % 2);
            if info[n].field {
                (x * mw, y * 2 * mh + b, 2)
            } else {
                (x * mw, y * 2 * mh + b * mh, 1)
            }
        };
        let qp = |m: &MbInfo, comp: Option<usize>| match comp {
            None => m.qp_y as i32,
            Some(c) => chroma_qp(
                m.qp_y as i32,
                sf[m.slice as usize].chroma_qp_offset[c],
                bd_offset_c,
            ),
        };
        let thresholds = |pm: usize, comp: Option<usize>| {
            let qp_av = (qp(&info[pm], comp) + qp(q, comp) + 1) >> 1;
            let index_a = (qp_av + f.offset_a).clamp(0, 51) as usize;
            let index_b = (qp_av + f.offset_b).clamp(0, 51) as usize;
            let sh = if comp.is_some() { bd_c } else { bd } - 8;
            (
                index_a,
                (ALPHA[index_a] as i32) << sh,
                (BETA[index_b] as i32) << sh,
            )
        };
        // A frame macroblock below a field pair filters its top edge twice,
        // once per field (8.7, step 3c).
        let above_field = !q.field && bottom == 0 && pair(0, -1).is_some_and(|n| info[n].field);

        // Luma strengths: vertical edges per row, horizontal per column.
        let mut bs_v = [[0u8; 16]; 4];
        for e in 0..4 {
            if e == 0 && !filter_left || e % 2 == 1 && q.transform_8x8 {
                continue;
            }
            for k in 0..16 {
                let pn = if e == 0 {
                    locate(-1, k as i32, 16, 16).unwrap()
                } else {
                    (addr, 4 * e - 1, k)
                };
                bs_v[e][k] = bs_mbaff(info, sf, pn, (addr, 4 * e, k), true, e == 0);
            }
        }
        // Horizontal edges: [top or its top field pass, the bottom field
        // pass of the double edge, 4, 8, 12].
        let mut bs_h = [[0u8; 16]; 5];
        let top_mbs: [Option<usize>; 2] = if !filter_top {
            [None, None]
        } else if above_field {
            let b = pair(0, -1).unwrap();
            [Some(b), Some(b + 1)]
        } else {
            [locate(0, -1, 16, 16).map(|n| n.0), None]
        };
        for (pass, pm) in top_mbs.iter().enumerate() {
            let Some(pm) = *pm else { continue };
            for x in 0..16 {
                bs_h[pass][x] = bs_mbaff(info, sf, (pm, x, 15), (addr, x, 0), false, true);
            }
        }
        // 4:2:2 chroma also filters the horizontal edges inside 8x8
        // transform blocks.
        let skip_odd = q.transform_8x8 && chroma_array_type != 2;
        for e in 1..4 {
            if e % 2 == 1 && skip_odd {
                continue;
            }
            for x in 0..16 {
                bs_h[e + 1][x] = bs_mbaff(
                    info,
                    sf,
                    (addr, x, 4 * e - 1),
                    (addr, x, 4 * e),
                    false,
                    false,
                );
            }
        }

        // Luma.
        {
            let stride = planes[0].stride;
            // SAFETY: the caller has exclusive access.
            let plane = unsafe { planes[0].exclusive() };
            let (x0, y0, dy) = origin(addr, 16, 16);
            for e in 0..4 {
                for k in 0..16 {
                    let b = bs_v[e][k];
                    if b == 0 {
                        continue;
                    }
                    let pm = if e == 0 {
                        locate(-1, k as i32, 16, 16).unwrap().0
                    } else {
                        addr
                    };
                    let (ia, alpha, beta) = thresholds(pm, None);
                    let o = (y0 + dy * k) * stride + x0 + 4 * e;
                    filter_line_t(plane, o, 1, b, ia, alpha, beta, false, bd);
                }
            }
            for (slot, (row0, step, pm)) in [
                (0, (y0, if above_field { 2 } else { dy }, top_mbs[0])),
                (1, (y0 + 1, 2, top_mbs[1])),
                (2, (y0 + 4 * dy, dy, Some(addr))),
                (3, (y0 + 8 * dy, dy, Some(addr))),
                (4, (y0 + 12 * dy, dy, Some(addr))),
            ] {
                let Some(pm) = pm else { continue };
                if slot % 2 == 0 && slot > 0 && q.transform_8x8 {
                    continue; // edges 4 and 12 inside an 8x8 transform
                }
                let (ia, alpha, beta) = thresholds(pm, None);
                for x in 0..16 {
                    let b = bs_h[slot][x];
                    if b != 0 {
                        let o = row0 * stride + x0 + x;
                        let s = step * stride;
                        filter_line_t(plane, o, s, b, ia, alpha, beta, false, bd);
                    }
                }
            }
        }

        if !matches!(chroma_array_type, 1 | 2) {
            continue;
        }
        // Chroma uses the strengths of the luma line in the same field; 4:2:2
        // chroma has the luma height, so its lines map one to one.
        let ch = if chroma_array_type == 2 { 16 } else { 8 };
        for comp in 0..2 {
            let stride = planes[1 + comp].stride;
            // SAFETY: the caller has exclusive access.
            let plane = unsafe { planes[1 + comp].exclusive() };
            let (x0, y0, dy) = origin(addr, 8, ch);
            for (ce, le) in [(0usize, 0usize), (1, 2)] {
                for k in 0..ch {
                    let pm = if ce == 0 {
                        match locate(-1, k as i32, 8, ch as i32) {
                            Some(n) => n.0,
                            None => continue,
                        }
                    } else {
                        addr
                    };
                    let idx = if ch == 16 {
                        k
                    } else if q.field && !info[pm].field {
                        2 * k
                    } else {
                        4 * (k >> 1) + (k & 1)
                    };
                    let b = bs_v[le][idx];
                    if b == 0 {
                        continue;
                    }
                    let (ia, alpha, beta) = thresholds(pm, Some(comp));
                    let o = (y0 + dy * k) * stride + x0 + 4 * ce;
                    filter_line_t(plane, o, 1, b, ia, alpha, beta, true, bd_c);
                }
            }
            let inner: &[(usize, usize)] = if ch == 16 {
                &[(2, 4), (3, 8), (4, 12)]
            } else {
                &[(3, 4)]
            };
            let edges = [
                (0, y0, if above_field { 2 } else { dy }, top_mbs[0]),
                (1, y0 + 1, 2, top_mbs[1]),
            ]
            .into_iter()
            .chain(
                inner
                    .iter()
                    .map(|&(slot, row)| (slot, y0 + row * dy, dy, Some(addr))),
            );
            for (slot, row0, step, pm) in edges {
                let Some(pm) = pm else { continue };
                let (ia, alpha, beta) = thresholds(pm, Some(comp));
                for x in 0..8 {
                    let b = bs_h[slot][2 * x];
                    if b != 0 {
                        let o = row0 * stride + x0 + x;
                        let s = step * stride;
                        filter_line_t(plane, o, s, b, ia, alpha, beta, true, bd_c);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_helpers_match_their_definitions() {
        for m in [0u16, 1, 0x8001, 0x1234, 0xffff, 0x0a5c, 0x4321] {
            let t = transpose_bits4(m);
            for r in 0..4 {
                for c in 0..4 {
                    assert_eq!(t >> (4 * c + r) & 1, m >> (4 * r + c) & 1, "{m:#x}");
                }
            }
            let mut b = [0u8; 16];
            expand16(m).store(&mut b);
            for (i, &x) in b.iter().enumerate() {
                assert_eq!(x, if m >> i & 1 != 0 { 255 } else { 0 }, "{m:#x} byte {i}");
            }
        }
    }

    #[test]
    fn byte_lanes_match_the_line_filter() {
        let mut x = 12345u32;
        let mut rnd = |n: u32| {
            x = x.wrapping_mul(1_103_515_245).wrapping_add(12345);
            (x >> 8) % n
        };
        let mut changed = [0; 2];
        for round in 0..20_000 {
            let chroma = round % 2 == 1;
            let index_a = rnd(52) as usize;
            let index_b = rnd(52) as usize;
            let (alpha, beta) = (ALPHA[index_a], BETA[index_b]);
            let strong = rnd(4) == 0;
            let bs: [u8; 4] = std::array::from_fn(|_| if strong { 4 } else { rnd(4) as u8 });
            // Lines near the thresholds: a level, small steps, a jump.
            let lines: [[u8; 8]; 16] = std::array::from_fn(|_| {
                let base = rnd(256) as i32;
                let jump = rnd(alpha as u32 * 2 + 2) as i32 - alpha as i32;
                std::array::from_fn(|k| {
                    let step = rnd(beta as u32 * 2 + 2) as i32 - beta as i32;
                    (base + step + if k >= 4 { jump } else { 0 }).clamp(0, 255) as u8
                })
            });
            let mut want = lines;
            for (i, l) in want.iter_mut().enumerate() {
                let b = bs[i / 4];
                if b > 0 {
                    filter_line(l, 4, 1, b, index_a, alpha as i32, beta as i32, chroma);
                }
            }
            let (on, tc0) = lanes(bs, 4, [index_a; 2]);
            let (a, b, a2) = (B::splat(alpha), B::splat(beta), B::splat((alpha >> 2) + 2));
            let mut v = transpose_in(&lines);
            if chroma {
                filter16::<true>(&mut v, on, tc0, bs[0] == 4, a, b, a2);
            } else {
                filter16::<false>(&mut v, on, tc0, bs[0] == 4, a, b, a2);
            }
            let mut got = [[0u8; 8]; 16];
            transpose_out(v, &mut got);
            assert_eq!(got, want, "chroma {chroma} indexA {index_a} bS {bs:?}");
            changed[strong as usize] += (want != lines) as usize;
        }
        // The filters really ran, weak and strong.
        assert!(changed[0] > 5000 && changed[1] > 1000, "{changed:?}");
    }

    #[test]
    fn edge_kernel_matches_line_filter() {
        let stride = 64;
        let mut seed = 3u32;
        let mut rnd = |n: u32| {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            (seed >> 16) % n
        };
        for case in 0..4000 {
            // Smooth-ish data so the filter conditions are often met.
            let base = rnd(200) as i32;
            let spread = [2, 8, 30, 255][case % 4];
            let mut data: Vec<u8> = (0..stride * 24)
                .map(|_| (base + rnd(spread) as i32).clamp(0, 255) as u8)
                .collect();
            let mut want = data.clone();
            let chroma = case % 3 == 0;
            let vertical = case % 2 == 0;
            let strong = rnd(4) == 0;
            let bs = if strong {
                [4; 4]
            } else {
                [0, 1, 2, 3].map(|_| rnd(4) as u8)
            };
            let index_a = 16 + rnd(36) as usize;
            let (alpha, beta) = (
                ALPHA[index_a] as i32,
                BETA[(index_a + rnd(5) as usize).min(51)] as i32,
            );
            let o = 4 * stride + 8;
            let n = if chroma { 8 } else { 16 };
            for k in 0..n {
                let b = bs[k / (n / 4)];
                if b == 0 {
                    continue;
                }
                let (oo, step) = if vertical {
                    (o + k * stride, 1)
                } else {
                    (o + k, stride)
                };
                filter_line(&mut want, oo, step, b, index_a, alpha, beta, chroma);
            }
            let pl = PlaneRef {
                ptr: data.as_mut_ptr(),
                len: data.len(),
                stride,
            };
            if chroma {
                edge::<8>(&pl, o, vertical, bs, index_a, alpha, beta, true);
            } else {
                edge::<16>(&pl, o, vertical, bs, index_a, alpha, beta, false);
            }
            assert_eq!(data, want, "case {case}");
        }
    }
}
