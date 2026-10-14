//! NEON luma interpolation working on bytes, eight samples at a time: the
//! 6-tap filter with widening multiply-accumulates and one rounding,
//! saturating narrow, and averages with rounding halving adds. Results are
//! identical to the 8 x i16 kernel in `inter`.

use core::arch::aarch64::*;

// SAFETY (all functions): NEON is part of the aarch64 baseline; the callers
// guarantee that every pointer read or written has 8 bytes there.

/// Unrounded 6-tap sum of the eight samples at `p, p + step, ...` (it
/// wraps in u16 but fits i16: -2550..=10710).
#[inline(always)]
unsafe fn raw(p: *const u8, step: usize) -> int16x8_t {
    unsafe {
        let v = |k: usize| vld1_u8(p.add(k * step));
        let a = vaddl_u8(v(0), v(5));
        let b = vaddl_u8(v(1), v(4));
        let c = vaddl_u8(v(2), v(3));
        vreinterpretq_s16_u16(vmlsq_n_u16(vmlaq_n_u16(a, c, 20), b, 5))
    }
}

/// Rounded, clipped half sample: (t + 16) >> 5 in 0..=255.
#[inline(always)]
fn half(t: int16x8_t) -> uint8x8_t {
    unsafe { vqrshrun_n_s16::<5>(t) }
}

/// Sample j from unrounded horizontal sums of six rows: the vertical 6-tap
/// in 32 bits, (sum + 512) >> 10 clipped to 0..=255.
#[inline(always)]
fn centre(r: &[int16x8_t; 6]) -> uint8x8_t {
    unsafe {
        let a = vaddq_s16(r[0], r[5]);
        let b = vaddq_s16(r[1], r[4]);
        let c = vaddq_s16(r[2], r[3]);
        let lo = vmlal_n_s16(
            vmlal_n_s16(vmovl_s16(vget_low_s16(a)), vget_low_s16(b), -5),
            vget_low_s16(c),
            20,
        );
        let hi = vmlal_high_n_s16(vmlal_high_n_s16(vmovl_high_s16(a), b, -5), c, 20);
        vqmovun_s16(vcombine_s16(
            vqrshrn_n_s32::<10>(lo),
            vqrshrn_n_s32::<10>(hi),
        ))
    }
}

/// A 16-sample row as two halves.
#[derive(Clone, Copy)]
struct Wide(int16x8_t, int16x8_t);

/// [`raw`] for sixteen samples, with 16-byte loads.
#[inline(always)]
unsafe fn raw16(p: *const u8, step: usize) -> Wide {
    unsafe {
        let v = |k: usize| vld1q_u8(p.add(k * step));
        let (v0, v1, v2, v3, v4, v5) = (v(0), v(1), v(2), v(3), v(4), v(5));
        let lo = vmlsq_n_u16(
            vmlaq_n_u16(
                vaddl_u8(vget_low_u8(v0), vget_low_u8(v5)),
                vaddl_u8(vget_low_u8(v2), vget_low_u8(v3)),
                20,
            ),
            vaddl_u8(vget_low_u8(v1), vget_low_u8(v4)),
            5,
        );
        let hi = vmlsq_n_u16(
            vmlaq_n_u16(vaddl_high_u8(v0, v5), vaddl_high_u8(v2, v3), 20),
            vaddl_high_u8(v1, v4),
            5,
        );
        Wide(vreinterpretq_s16_u16(lo), vreinterpretq_s16_u16(hi))
    }
}

#[inline(always)]
fn half16(t: Wide) -> uint8x16_t {
    unsafe { vqrshrun_high_n_s16::<5>(vqrshrun_n_s16::<5>(t.0), t.1) }
}

#[inline(always)]
fn centre16(r: &[Wide; 6]) -> uint8x16_t {
    let lo: [int16x8_t; 6] = std::array::from_fn(|k| r[k].0);
    let hi: [int16x8_t; 6] = std::array::from_fn(|k| r[k].1);
    unsafe { vcombine_u8(centre(&lo), centre(&hi)) }
}

/// [`luma`] for 16-wide blocks, a row per vector.
#[inline(always)]
fn luma16<const A: bool>(
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    st: usize,
    fx: usize,
    fy: usize,
    h: usize,
) {
    assert!((1..=16).contains(&h) && src.len() >= (h + 4) * st + 21);
    assert!(dst.len() >= (h - 1) * ds + 16);
    let s = src.as_ptr();
    let d = dst.as_mut_ptr();
    // SAFETY (all blocks below): every load reads 16 samples within the
    // window (columns up to 21, rows up to h + 4), every store 16 in dst.
    let b = |r: usize| half16(unsafe { raw16(s.add((r + 2) * st), 1) });
    let hv = |r: usize, c: usize| half16(unsafe { raw16(s.add(r * st + c + 2), st) });
    let g = |r: usize, c: usize| unsafe { vld1q_u8(s.add((r + 2) * st + c + 2)) };
    let avg = |x: uint8x16_t, y: uint8x16_t| unsafe { vrhaddq_u8(x, y) };
    // With A, averaged with the samples there (default bi-prediction).
    let st16 = |r: usize, v: uint8x16_t| unsafe {
        let v = if A {
            vrhaddq_u8(vld1q_u8(d.add(r * ds)), v)
        } else {
            v
        };
        vst1q_u8(d.add(r * ds), v)
    };
    // Positions sharing a shape share a loop, the offsets of their samples
    // given at run time (Table 8-12), to keep the code small.
    match (fx, fy) {
        (2, 1..=3) | (1 | 3, 2) => {
            // A sliding window of six rows' horizontal sums, in named values
            // (an array indexed at run time would live in memory).
            let raw = |k: usize| unsafe { raw16(s.add(k * st), 1) };
            let (mut w0, mut w1, mut w2, mut w3, mut w4, mut w5) =
                (raw(0), raw(1), raw(2), raw(3), raw(4), raw(5));
            for r in 0..h {
                if r > 0 {
                    (w0, w1, w2, w3, w4) = (w1, w2, w3, w4, w5);
                    w5 = raw(r + 5);
                }
                let j = centre16(&[w0, w1, w2, w3, w4, w5]);
                let v = match (fx, fy) {
                    (2, 2) => j,
                    // j with b or s, the half sample of row r or r + 1.
                    (2, 1) => avg(j, half16(w2)),
                    (2, _) => avg(j, half16(w3)),
                    // j with h or m, of column 0 or 1.
                    _ => avg(j, hv(r, fx >> 1)),
                };
                st16(r, v);
            }
        }
        (0, 0) => (0..h).for_each(|r| st16(r, g(r, 0))),
        (2, 0) => (0..h).for_each(|r| st16(r, b(r))),
        (0, 2) => (0..h).for_each(|r| st16(r, hv(r, 0))),
        // b with G or H (column 0 or 1).
        (_, 0) => (0..h).for_each(|r| st16(r, avg(g(r, fx >> 1), b(r)))),
        // h with G or M (row 0 or 1).
        (0, _) => (0..h).for_each(|r| st16(r, avg(g(r + (fy >> 1), 0), hv(r, 0)))),
        // b or s (row 0 or 1) with h or m (column 0 or 1).
        _ => (0..h).for_each(|r| st16(r, avg(b(r + (fy >> 1)), hv(r, fx >> 1)))),
    }
}

/// `inter::luma_vec` for widths 8 and 16 at fractional positions: `src` is
/// the window of rows 0..h + 5 of W + 5 samples with row stride `st`.
#[inline(always)]
pub(crate) fn luma<const W: usize, const A: bool>(
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    st: usize,
    fx: usize,
    fy: usize,
    h: usize,
) {
    if W == 16 {
        return luma16::<A>(dst, ds, src, st, fx, fy, h);
    }
    // Every load reads 8 samples within the window (columns up to W + 5,
    // rows up to h + 4) and every store 8 within dst.
    assert!(W % 8 == 0 && (1..=16).contains(&h) && src.len() >= (h + 4) * st + W + 5);
    assert!(dst.len() >= (h - 1) * ds + W);
    let s = src.as_ptr();
    let d = dst.as_mut_ptr();
    // SAFETY (all blocks below): within the window and dst, asserted above.
    // Half samples b (horizontal) and h (vertical) and integer samples G of
    // block row r, from column c of the block.
    let b = |r: usize, c: usize| half(unsafe { raw(s.add((r + 2) * st + c), 1) });
    let hv = |r: usize, c: usize| half(unsafe { raw(s.add(r * st + c + 2), st) });
    let g = |r: usize, c: usize| unsafe { vld1_u8(s.add((r + 2) * st + c + 2)) };
    let avg = |x: uint8x8_t, y: uint8x8_t| unsafe { vrhadd_u8(x, y) };
    let st8 = |r: usize, c: usize, v: uint8x8_t| unsafe {
        let v = if A {
            vrhadd_u8(vld1_u8(d.add(r * ds + c)), v)
        } else {
            v
        };
        vst1_u8(d.add(r * ds + c), v)
    };
    // Positions sharing a shape share a loop (see luma16).
    let (gx, gy) = (fx >> 1, fy >> 1);
    macro_rules! each {
        (|$r:ident, $c:ident| $e:expr) => {
            for $r in 0..h {
                for $c in (0..W).step_by(8) {
                    st8($r, $c, $e);
                }
            }
        };
    }
    match (fx, fy) {
        (2, 1..=3) | (1 | 3, 2) => {
            // The centre sample j: the vertical 6-tap over the unrounded
            // horizontal sums of window rows r..r + 6, a sliding window.
            for c in (0..W).step_by(8) {
                // Named values, not an array indexed at run time (see luma16).
                let raw = |k: usize| unsafe { raw(s.add(k * st + c), 1) };
                let (mut w0, mut w1, mut w2, mut w3, mut w4, mut w5) =
                    (raw(0), raw(1), raw(2), raw(3), raw(4), raw(5));
                for r in 0..h {
                    if r > 0 {
                        (w0, w1, w2, w3, w4) = (w1, w2, w3, w4, w5);
                        w5 = raw(r + 5);
                    }
                    let j = centre(&[w0, w1, w2, w3, w4, w5]);
                    let v = match (fx, fy) {
                        (2, 2) => j,
                        (2, 1) => avg(j, half(w2)),
                        (2, _) => avg(j, half(w3)),
                        _ => avg(j, hv(r, c + gx)),
                    };
                    st8(r, c, v);
                }
            }
        }
        (0, 0) => each!(|r, c| g(r, c)),
        (2, 0) => each!(|r, c| b(r, c)),
        (0, 2) => each!(|r, c| hv(r, c)),
        (_, 0) => each!(|r, c| avg(g(r, c + gx), b(r, c))),
        (0, _) => each!(|r, c| avg(g(r + gy, c), hv(r, c))),
        _ => each!(|r, c| avg(b(r + gy, c), hv(r, c + gx))),
    }
}

/// Chroma interpolation (8.4.2.2.2) of the same W x h block (W 8 or 4) of
/// Cb and Cr with eighth-sample fraction (fx, fy): the two planes side by
/// side in one vector, weighted with widening byte multiplies. `s` are the
/// (W + 1) x (h + 1) windows (row stride `st`), `d` the outputs (stride
/// `ds`).
///
/// # Safety
///
/// The windows and outputs must hold those rows.
pub(crate) unsafe fn chroma_pair<const W: usize, const A: bool>(
    d: [*mut u8; 2],
    ds: usize,
    s: [*const u8; 2],
    st: usize,
    fx: i32,
    fy: i32,
    h: usize,
) {
    // SAFETY: NEON is part of the aarch64 baseline.
    let w = |v: i32| unsafe { vdup_n_u8(v as u8) };
    let (wa, wb, wc, wd) = (
        w((8 - fx) * (8 - fy)),
        w(fx * (8 - fy)),
        w((8 - fx) * fy),
        w(fx * fy),
    );
    unsafe {
        if W == 8 {
            let wq = |v: uint8x8_t| vcombine_u8(v, v);
            let (qa, qb, qc, qd) = (wq(wa), wq(wb), wq(wc), wq(wd));
            // Row samples x and x + 1 of both planes.
            let row = |o: usize| {
                (
                    vcombine_u8(vld1_u8(s[0].add(o)), vld1_u8(s[1].add(o))),
                    vcombine_u8(vld1_u8(s[0].add(o + 1)), vld1_u8(s[1].add(o + 1))),
                )
            };
            let (mut t0, mut t1) = row(0);
            for r in 0..h {
                let (b0, b1) = row((r + 1) * st);
                let lo = vmull_u8(vget_low_u8(t0), wa);
                let lo = vmlal_u8(lo, vget_low_u8(t1), wb);
                let lo = vmlal_u8(lo, vget_low_u8(b0), wc);
                let lo = vmlal_u8(lo, vget_low_u8(b1), wd);
                let hi = vmull_high_u8(t0, qa);
                let hi = vmlal_high_u8(hi, t1, qb);
                let hi = vmlal_high_u8(hi, b0, qc);
                let hi = vmlal_high_u8(hi, b1, qd);
                let v = vrshrn_high_n_u16::<6>(vrshrn_n_u16::<6>(lo), hi);
                let v = if A {
                    let o = vcombine_u8(vld1_u8(d[0].add(r * ds)), vld1_u8(d[1].add(r * ds)));
                    vrhaddq_u8(o, v)
                } else {
                    v
                };
                vst1_u8(d[0].add(r * ds), vget_low_u8(v));
                vst1_u8(d[1].add(r * ds), vget_high_u8(v));
                (t0, t1) = (b0, b1);
            }
        } else {
            let ld = |p: *const u8| p.cast::<u32>().read_unaligned();
            let row = |o: usize| {
                let v = |o: usize| {
                    let lanes = ld(s[0].add(o)) as u64 | (ld(s[1].add(o)) as u64) << 32;
                    vcreate_u8(lanes)
                };
                (v(o), v(o + 1))
            };
            let (mut t0, mut t1) = row(0);
            for r in 0..h {
                let (b0, b1) = row((r + 1) * st);
                let v = vmull_u8(t0, wa);
                let v = vmlal_u8(v, t1, wb);
                let v = vmlal_u8(v, b0, wc);
                let v = vmlal_u8(v, b1, wd);
                let v = vrshrn_n_u16::<6>(v);
                let v = if A {
                    let o = ld(d[0].add(r * ds).cast_const()) as u64
                        | (ld(d[1].add(r * ds).cast_const()) as u64) << 32;
                    vrhadd_u8(vcreate_u8(o), v)
                } else {
                    v
                };
                let v = vreinterpret_u32_u8(v);
                d[0].add(r * ds)
                    .cast::<u32>()
                    .write_unaligned(vget_lane_u32::<0>(v));
                d[1].add(r * ds)
                    .cast::<u32>()
                    .write_unaligned(vget_lane_u32::<1>(v));
                (t0, t1) = (b0, b1);
            }
        }
    }
}

/// Weighted prediction (8.4.2.3.2) of w x h samples (w 16, 8 or 4) in 32
/// bits: clip((a * wa + b * wb + k) >> s), with b and wb only for
/// bi-prediction; k is the rounding plus the offset shifted left by s. `a` and `b` have row stride
/// [`crate::inter::MAX`].
///
/// # Safety
///
/// `out` must hold the rows (stride `stride`), and `a` and `b` w x h
/// samples; h is even for w 4.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn weighted<const BI: bool>(
    out: *mut u8,
    stride: usize,
    a: *const u8,
    b: *const u8,
    (wa, wb): (i32, i32),
    k: i32,
    s: u32,
    w: usize,
    h: usize,
) {
    use crate::inter::MAX;
    unsafe {
        let (kk, sh) = (vdupq_n_s32(k), vdupq_n_s32(-(s as i32)));
        let (wa, wb) = (wa as i16, wb as i16);
        let blend = |pa: uint8x8_t, pb: uint8x8_t| {
            let a = vreinterpretq_s16_u16(vmovl_u8(pa));
            let mut lo = vmlal_n_s16(kk, vget_low_s16(a), wa);
            let mut hi = vmlal_high_n_s16(kk, a, wa);
            if BI {
                let b = vreinterpretq_s16_u16(vmovl_u8(pb));
                lo = vmlal_n_s16(lo, vget_low_s16(b), wb);
                hi = vmlal_high_n_s16(hi, b, wb);
            }
            let (lo, hi) = (vshlq_s32(lo, sh), vshlq_s32(hi, sh));
            vqmovun_s16(vcombine_s16(vqmovn_s32(lo), vqmovn_s32(hi)))
        };
        if w == 4 {
            let ld = |p: *const u8, r: usize| {
                let lane = |r: usize| p.add(r * MAX).cast::<u32>().read_unaligned() as u64;
                vcreate_u8(lane(r) | lane(r + 1) << 32)
            };
            for r in (0..h).step_by(2) {
                let v =
                    vreinterpret_u32_u8(blend(ld(a, r), if BI { ld(b, r) } else { vdup_n_u8(0) }));
                out.add(r * stride)
                    .cast::<u32>()
                    .write_unaligned(vget_lane_u32::<0>(v));
                out.add((r + 1) * stride)
                    .cast::<u32>()
                    .write_unaligned(vget_lane_u32::<1>(v));
            }
            return;
        }
        for r in 0..h {
            for c in (0..w).step_by(8) {
                let o = r * MAX + c;
                let pb = if BI { vld1_u8(b.add(o)) } else { vdup_n_u8(0) };
                vst1_u8(out.add(r * stride + c), blend(vld1_u8(a.add(o)), pb));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn matches_the_i16_kernel() {
        let st = 24;
        let mut x = 7u32;
        for h in 1..=16 {
            let src: Vec<u8> = (0..(h + 5) * st)
                .map(|_| {
                    x = x.wrapping_mul(1_103_515_245).wrapping_add(12345);
                    (x >> 16) as u8
                })
                .collect();
            for fx in 0..4 {
                for fy in 0..4 {
                    let (mut a, mut b) = ([0u8; 16 * 20], [0u8; 16 * 20]);
                    super::luma::<16, false>(&mut a, 20, &src, st, fx, fy, h);
                    crate::inter::luma_vec_for_test(&mut b, 20, &src, st, fx, fy, h);
                    assert_eq!(a, b, "h {h} fx {fx} fy {fy}");
                    let (mut a, mut b) = ([0u8; 16 * 20], [0u8; 16 * 20]);
                    super::luma::<8, false>(&mut a, 20, &src, st, fx, fy, h);
                    crate::inter::luma_vec8_for_test(&mut b, 20, &src, st, fx, fy, h);
                    assert_eq!(a, b, "w 8 h {h} fx {fx} fy {fy}");
                }
            }
        }
    }
}
