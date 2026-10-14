//! AVX2 versions of motion-compensation kernels, picked at run time on
//! x86_64 CPUs that have it: a whole 16-sample row per 16 x i16 vector
//! (and weighted prediction, in SSE2). Results are identical to the
//! 8 x i16 kernels in `inter`.

use core::arch::x86_64::*;

/// Whether the CPU has AVX2 (std caches the detection).
#[inline]
pub(crate) fn available() -> bool {
    std::is_x86_feature_detected!("avx2")
}

/// Whether the CPU has SSSE3 (std caches the detection).
#[inline]
pub(crate) fn ssse3() -> bool {
    std::is_x86_feature_detected!("ssse3")
}

/// The byte pairs (p[k], p[k + 1]) and the coefficients of the 6-tap
/// filter for them: 1, -5 | 20, 20 | -5, 1 (low byte first).
#[target_feature(enable = "ssse3")]
#[inline]
fn coef(a: i8, b: i8) -> __m128i {
    _mm_set1_epi16(((b as u8 as i16) << 8) | a as u8 as i16)
}

/// Unrounded horizontal 6-tap sums of the eight samples from `p` (reading
/// p[0..13]) as 8 x i16, by byte pair multiply-adds (-2550..=10710).
///
/// # Safety
/// p[0..13] must be readable.
#[target_feature(enable = "ssse3")]
#[inline]
unsafe fn hraw8(p: *const u8) -> __m128i {
    // p[0..8] and p[5..13] side by side: sample k at k, or k + 3 from 8.
    let v =
        unsafe { _mm_unpacklo_epi64(_mm_loadl_epi64(p.cast()), _mm_loadl_epi64(p.add(5).cast())) };
    const fn pairs(o: i8) -> [i8; 16] {
        let mut m = [0i8; 16];
        let mut i = 0;
        while i < 8 {
            let (a, b) = (i as i8 + o, i as i8 + o + 1);
            m[2 * i] = if a < 8 { a } else { a + 3 };
            m[2 * i + 1] = if b < 8 { b } else { b + 3 };
            i += 1;
        }
        m
    }
    const M0: [i8; 16] = pairs(0);
    const M2: [i8; 16] = pairs(2);
    const M4: [i8; 16] = pairs(4);
    let sh = |m: &[i8; 16]| unsafe { _mm_shuffle_epi8(v, _mm_loadu_si128(m.as_ptr().cast())) };
    let a = _mm_maddubs_epi16(sh(&M0), coef(1, -5));
    let b = _mm_maddubs_epi16(sh(&M2), coef(20, 20));
    let c = _mm_maddubs_epi16(sh(&M4), coef(-5, 1));
    _mm_add_epi16(_mm_add_epi16(a, b), c)
}

/// Unrounded vertical 6-tap sums of the eight samples at `p, p + st, ...,
/// p + 5 st` as 8 x i16.
///
/// # Safety
/// Eight bytes must be readable at each row.
#[target_feature(enable = "ssse3")]
#[inline]
unsafe fn vraw8(p: *const u8, st: usize) -> __m128i {
    let r = |k: usize| unsafe { _mm_loadl_epi64(p.add(k * st).cast()) };
    let a = _mm_maddubs_epi16(_mm_unpacklo_epi8(r(0), r(1)), coef(1, -5));
    let b = _mm_maddubs_epi16(_mm_unpacklo_epi8(r(2), r(3)), coef(20, 20));
    let c = _mm_maddubs_epi16(_mm_unpacklo_epi8(r(4), r(5)), coef(-5, 1));
    _mm_add_epi16(_mm_add_epi16(a, b), c)
}

/// (t + 16) >> 5 clipped to bytes (in the low 8 lanes).
#[target_feature(enable = "ssse3")]
#[inline]
fn half8(t: __m128i) -> __m128i {
    // mulhrs by 1024: (t * 1024 + 2^14) >> 15 = (t + 16) >> 5.
    let v = _mm_mulhrs_epi16(t, _mm_set1_epi16(1024));
    _mm_packus_epi16(v, v)
}

/// `inter::luma_vec` for 8-wide blocks at a fractional position, with
/// SSSE3 byte multiply-adds: `src` is the window of rows 0..h + 5 of 13
/// samples with row stride `st`; with `A`, averaged into `dst`.
///
/// # Safety
/// The CPU must support SSSE3.
#[target_feature(enable = "ssse3")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn luma8<const A: bool>(
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    st: usize,
    fx: usize,
    fy: usize,
    h: usize,
) {
    use crate::simd::I16x8;
    // Every load reads within the window (columns up to 13, rows up to
    // h + 4) and every store 8 within dst.
    assert!((1..=16).contains(&h) && src.len() >= (h + 4) * st + 13);
    assert!(dst.len() >= (h - 1) * ds + 8);
    let s = src.as_ptr();
    let d = dst.as_mut_ptr();
    // SAFETY (all blocks below): within the window and dst, asserted above.
    let b = |r: usize| half8(unsafe { hraw8(s.add((r + 2) * st)) });
    let hv = |r: usize, c: usize| half8(unsafe { vraw8(s.add(r * st + c + 2), st) });
    let g = |r: usize, c: usize| unsafe { _mm_loadl_epi64(s.add((r + 2) * st + c + 2).cast()) };
    let avg = |x: __m128i, y: __m128i| _mm_avg_epu8(x, y);
    let store = |r: usize, v: __m128i| unsafe {
        let p = d.add(r * ds);
        let v = if A {
            _mm_avg_epu8(_mm_loadl_epi64(p.cast()), v)
        } else {
            v
        };
        _mm_storel_epi64(p.cast(), v)
    };
    let (gx, gy) = (fx >> 1, fy >> 1);
    match (fx, fy) {
        (2, 1..=3) | (1 | 3, 2) => {
            // The centre sample j: the vertical 6-tap (32 bits) over the
            // horizontal sums of window rows r..r + 6, a sliding window.
            let raw = |k: usize| I16x8(unsafe { hraw8(s.add(k * st)) });
            let (mut w0, mut w1, mut w2, mut w3, mut w4, mut w5) =
                (raw(0), raw(1), raw(2), raw(3), raw(4), raw(5));
            for r in 0..h {
                if r > 0 {
                    (w0, w1, w2, w3, w4) = (w1, w2, w3, w4, w5);
                    w5 = raw(r + 5);
                }
                let t = crate::simd::tap6_10([w0, w1, w2, w3, w4, w5]).0;
                let j = _mm_packus_epi16(t, t);
                let v = match (fx, fy) {
                    (2, 2) => j,
                    (2, 1) => avg(j, half8(w2.0)),
                    (2, _) => avg(j, half8(w3.0)),
                    _ => avg(j, hv(r, gx)),
                };
                store(r, v);
            }
        }
        (0, 0) => (0..h).for_each(|r| store(r, g(r, 0))),
        (2, 0) => (0..h).for_each(|r| store(r, b(r))),
        (0, 2) => (0..h).for_each(|r| store(r, hv(r, 0))),
        (_, 0) => (0..h).for_each(|r| store(r, avg(g(r, gx), b(r)))),
        (0, _) => (0..h).for_each(|r| store(r, avg(g(r + gy, 0), hv(r, 0)))),
        _ => (0..h).for_each(|r| store(r, avg(b(r + gy), hv(r, gx)))),
    }
}

/// 16 samples at `p`, widened.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn ld(p: *const u8) -> __m256i {
    // SAFETY: the caller guarantees 16 readable bytes at p.
    unsafe { _mm256_cvtepu8_epi16(_mm_loadu_si128(p.cast())) }
}

/// Unrounded 6-tap sum of six vectors of samples `p, p + step, ...`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn raw(p: *const u8, step: usize) -> __m256i {
    // SAFETY: the caller guarantees 16 readable bytes at each p + k * step.
    let v = |k: usize| unsafe { ld(p.add(k * step)) };
    let (a, b, c) = (
        _mm256_add_epi16(v(0), v(5)),
        _mm256_add_epi16(v(1), v(4)),
        _mm256_add_epi16(v(2), v(3)),
    );
    _mm256_add_epi16(
        _mm256_sub_epi16(a, _mm256_mullo_epi16(b, _mm256_set1_epi16(5))),
        _mm256_mullo_epi16(c, _mm256_set1_epi16(20)),
    )
}

/// Clips to 0..=255.
#[target_feature(enable = "avx2")]
#[inline]
fn clip(v: __m256i) -> __m256i {
    _mm256_min_epi16(
        _mm256_max_epi16(v, _mm256_setzero_si256()),
        _mm256_set1_epi16(255),
    )
}

/// Rounded, clipped half sample from an unrounded 6-tap sum.
#[target_feature(enable = "avx2")]
#[inline]
fn round(t: __m256i) -> __m256i {
    clip(_mm256_srai_epi16::<5>(_mm256_add_epi16(
        t,
        _mm256_set1_epi16(16),
    )))
}

#[target_feature(enable = "avx2")]
#[inline]
fn avg(a: __m256i, b: __m256i) -> __m256i {
    _mm256_srai_epi16::<1>(_mm256_add_epi16(
        _mm256_add_epi16(a, b),
        _mm256_set1_epi16(1),
    ))
}

/// ((r0 + r5) - 5 (r1 + r4) + 20 (r2 + r3) + 512) >> 10 in 32 bits, clipped
/// to 0..=255: sample j from unrounded horizontal sums (as simd::tap6_10).
#[target_feature(enable = "avx2")]
#[inline]
fn tap6_10(r: &[__m256i; 6]) -> __m256i {
    let a = _mm256_add_epi16(r[0], r[5]);
    let b = _mm256_add_epi16(r[1], r[4]);
    let c = _mm256_add_epi16(r[2], r[3]);
    // (a, b) pairs times (1, -5) and (c, 0) pairs times (20, 0); the unpacks
    // and the pack work within 128-bit lanes, so the order is kept.
    let k_ab = _mm256_set1_epi32((-5i32 << 16) | 1);
    let k_c = _mm256_set1_epi32(20);
    let zero = _mm256_setzero_si256();
    let round = _mm256_set1_epi32(512);
    let half = |ab: __m256i, c0: __m256i| {
        let t = _mm256_add_epi32(_mm256_madd_epi16(ab, k_ab), _mm256_madd_epi16(c0, k_c));
        _mm256_srai_epi32::<10>(_mm256_add_epi32(t, round))
    };
    let lo = half(_mm256_unpacklo_epi16(a, b), _mm256_unpacklo_epi16(c, zero));
    let hi = half(_mm256_unpackhi_epi16(a, b), _mm256_unpackhi_epi16(c, zero));
    clip(_mm256_packs_epi32(lo, hi))
}

/// Stores 16 samples (0..=255) at `p`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn store<const A: bool>(p: *mut u8, v: __m256i) {
    let b = _mm_packus_epi16(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
    // With A, averaged with the samples there (default bi-prediction).
    // SAFETY: the caller guarantees 16 readable bytes at p.
    let b = if A {
        _mm_avg_epu8(b, unsafe { _mm_loadu_si128(p.cast()) })
    } else {
        b
    };
    // SAFETY: the caller guarantees 16 writable bytes at p.
    unsafe { _mm_storeu_si128(p.cast(), b) }
}

/// `inter::luma_vec` for 16-wide blocks at a fractional position: `src` is
/// the window of rows 0..h + 5 of 21 samples with row stride `st`.
///
/// # Safety
/// The CPU must support AVX2.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn luma16<const A: bool>(
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    st: usize,
    fx: usize,
    fy: usize,
    h: usize,
) {
    // Every load below reads 16 samples within the window (columns up to
    // 21, rows up to h + 4) and every store 16 within dst.
    assert!((1..=16).contains(&h) && src.len() >= (h + 4) * st + 21);
    assert!(dst.len() >= (h - 1) * ds + 16);
    let s = src.as_ptr();
    let d = dst.as_mut_ptr();
    // SAFETY (all blocks below): within the window and dst, asserted above.
    // Half samples b (horizontal) and h (vertical) and integer samples G of
    // block row r, from column c of the block.
    let b = |r: usize| round(unsafe { raw(s.add((r + 2) * st), 1) });
    let hv = |r: usize, c: usize| round(unsafe { raw(s.add(r * st + c + 2), st) });
    let g = |r: usize, c: usize| unsafe { ld(s.add((r + 2) * st + c + 2)) };
    if matches!((fx, fy), (2, 1..=3) | (1 | 3, 2)) {
        // The centre sample j: the vertical 6-tap over the unrounded
        // horizontal sums of window rows r..r + 6, kept as a sliding window.
        // Named values, not a rotated array (which would live in memory).
        let raw6 = |k: usize| unsafe { raw(s.add(k * st), 1) };
        let (mut w0, mut w1, mut w2, mut w3, mut w4, mut w5) =
            (raw6(0), raw6(1), raw6(2), raw6(3), raw6(4), raw6(5));
        for r in 0..h {
            if r > 0 {
                (w0, w1, w2, w3, w4) = (w1, w2, w3, w4, w5);
                w5 = raw6(r + 5);
            }
            let j = tap6_10(&[w0, w1, w2, w3, w4, w5]);
            let v = match (fx, fy) {
                (2, 2) => j,
                (2, 1) => avg(round(w2), j),
                (2, 3) => avg(j, round(w3)),
                (1, 2) => avg(hv(r, 0), j),
                _ => avg(j, hv(r, 1)), // (3, 2)
            };
            unsafe { store::<A>(d.add(r * ds), v) };
        }
        return;
    }
    for r in 0..h {
        let v = match (fx, fy) {
            (0, 0) => g(r, 0),
            (1, 0) => avg(g(r, 0), b(r)),
            (2, 0) => b(r),
            (3, 0) => avg(g(r, 1), b(r)),
            (0, 1) => avg(g(r, 0), hv(r, 0)),
            (0, 2) => hv(r, 0),
            (0, 3) => avg(g(r + 1, 0), hv(r, 0)),
            (1, 1) => avg(b(r), hv(r, 0)),
            (3, 1) => avg(b(r), hv(r, 1)),
            (1, 3) => avg(hv(r, 0), b(r + 1)),
            _ => avg(hv(r, 1), b(r + 1)), // (3, 3)
        };
        unsafe { store::<A>(d.add(r * ds), v) };
    }
}

/// Chroma interpolation (8.4.2.2.2) of the same W x h block (W 8 or 4) of
/// Cb and Cr with eighth-sample fraction (fx, fy): pairs of horizontally
/// adjacent samples weighted with byte multiply-adds, rounded by a
/// multiply with rounding ((v * 512 + 2^14) >> 15 is (v + 32) >> 6). `s`
/// are the (W + 1) x (h + 1) windows (row stride `st`), `d` the outputs
/// (stride `ds`).
///
/// # Safety
///
/// The CPU must have AVX2, and the windows and outputs hold those rows.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn chroma_pair<const W: usize, const A: bool>(
    d: [*mut u8; 2],
    ds: usize,
    s: [*const u8; 2],
    st: usize,
    fx: i32,
    fy: i32,
    h: usize,
) {
    // Weights of the first and second sample of a pair, top and bottom.
    let ab = _mm_set1_epi16((((fx * (8 - fy)) << 8) | ((8 - fx) * (8 - fy))) as i16);
    let cd = _mm_set1_epi16((((fx * fy) << 8) | ((8 - fx) * fy)) as i16);
    let round = _mm_set1_epi16(512);
    let blend = |t: __m128i, b: __m128i| {
        _mm_mulhrs_epi16(
            _mm_add_epi16(_mm_maddubs_epi16(t, ab), _mm_maddubs_epi16(b, cd)),
            round,
        )
    };
    // SAFETY: the caller guarantees the rows read and written.
    unsafe {
        if W == 8 {
            let pairs = |p: *const u8| {
                _mm_unpacklo_epi8(_mm_loadl_epi64(p.cast()), _mm_loadl_epi64(p.add(1).cast()))
            };
            let (mut t0, mut t1) = (pairs(s[0]), pairs(s[1]));
            for r in 0..h {
                let o = (r + 1) * st;
                let (b0, b1) = (pairs(s[0].add(o)), pairs(s[1].add(o)));
                let v = _mm_packus_epi16(blend(t0, b0), blend(t1, b1));
                let v = if A {
                    let o = _mm_loadl_epi64(d[0].add(r * ds).cast());
                    let o = _mm_castpd_si128(_mm_loadh_pd(
                        _mm_castsi128_pd(o),
                        d[1].add(r * ds).cast(),
                    ));
                    _mm_avg_epu8(o, v)
                } else {
                    v
                };
                _mm_storel_epi64(d[0].add(r * ds).cast(), v);
                _mm_storeh_pd(d[1].add(r * ds).cast(), _mm_castsi128_pd(v));
                (t0, t1) = (b0, b1);
            }
        } else {
            let ld = |p: *const u8| _mm_cvtsi32_si128(p.cast::<i32>().read_unaligned());
            let pairs = |o: usize| {
                let p = |s: *const u8| _mm_unpacklo_epi8(ld(s.add(o)), ld(s.add(o + 1)));
                _mm_unpacklo_epi64(p(s[0]), p(s[1]))
            };
            let mut t = pairs(0);
            for r in 0..h {
                let b = pairs((r + 1) * st);
                let v = blend(t, b);
                let v = _mm_packus_epi16(v, v);
                let v = if A {
                    let o = _mm_unpacklo_epi32(
                        ld(d[0].add(r * ds).cast_const()),
                        ld(d[1].add(r * ds).cast_const()),
                    );
                    _mm_avg_epu8(o, v)
                } else {
                    v
                };
                d[0].add(r * ds)
                    .cast::<i32>()
                    .write_unaligned(_mm_cvtsi128_si32(v));
                d[1].add(r * ds)
                    .cast::<i32>()
                    .write_unaligned(_mm_extract_epi32::<1>(v));
                t = b;
            }
        }
    }
}

/// Weighted prediction (8.4.2.3.2) of w x h samples (w 16, 8 or 4) in 32
/// bits: clip((a * wa + b * wb + k) >> s), with b and wb only for
/// bi-prediction; k is the rounding plus the offset shifted left by s, from pairs of 16-bit samples
/// multiplied and added into 32 bits. SSE2, so not gated on AVX2. `a` and
/// `b` have row stride [`crate::inter::MAX`].
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
    // SAFETY: SSE2 is part of the x86_64 baseline, and the caller
    // guarantees the rows read and written.
    unsafe {
        let z = _mm_setzero_si128();
        // Single prediction: the second sample of each pair is 1, weighted by
        // k (which fits 16 bits there), so the multiply-add adds it.
        let pair = |x: i32, y: i32| _mm_set1_epi32(y << 16 | (x & 0xffff));
        let (wts, kk, ones) = if BI {
            (pair(wa, wb), _mm_set1_epi32(k), z)
        } else {
            (pair(wa, k), z, _mm_set1_epi16(1))
        };
        let cnt = _mm_cvtsi32_si128(s as i32);
        let blend = |pa: __m128i, pb: __m128i| {
            let a = _mm_unpacklo_epi8(pa, z);
            let b = if BI { _mm_unpacklo_epi8(pb, z) } else { ones };
            let f = |v: __m128i| _mm_sra_epi32(_mm_add_epi32(_mm_madd_epi16(v, wts), kk), cnt);
            let v = _mm_packs_epi32(f(_mm_unpacklo_epi16(a, b)), f(_mm_unpackhi_epi16(a, b)));
            _mm_packus_epi16(v, v)
        };
        if w == 4 {
            let ld = |p: *const u8, r: usize| {
                let lane =
                    |r: usize| _mm_cvtsi32_si128(p.add(r * MAX).cast::<i32>().read_unaligned());
                _mm_unpacklo_epi32(lane(r), lane(r + 1))
            };
            for r in (0..h).step_by(2) {
                let v = blend(ld(a, r), if BI { ld(b, r) } else { z });
                out.add(r * stride)
                    .cast::<i32>()
                    .write_unaligned(_mm_cvtsi128_si32(v));
                let v = _mm_srli_epi64::<32>(v);
                out.add((r + 1) * stride)
                    .cast::<i32>()
                    .write_unaligned(_mm_cvtsi128_si32(v));
            }
            return;
        }
        for r in 0..h {
            for c in (0..w).step_by(8) {
                let o = r * MAX + c;
                let pb = if BI {
                    _mm_loadl_epi64(b.add(o).cast())
                } else {
                    z
                };
                let v = blend(_mm_loadl_epi64(a.add(o).cast()), pb);
                _mm_storel_epi64(out.add(r * stride + c).cast(), v);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn luma8_matches_the_8_lane_kernel() {
        if !super::ssse3() {
            return;
        }
        let mut x = 7u32;
        // A window exactly 13 samples wide, and a wider one.
        for st in [13, 24] {
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
                        // SAFETY: SSSE3 is available.
                        unsafe { super::luma8::<false>(&mut a, 20, &src, st, fx, fy, h) };
                        crate::inter::luma_vec8_for_test(&mut b, 20, &src, st, fx, fy, h);
                        assert_eq!(a, b, "st {st} h {h} fx {fx} fy {fy}");
                    }
                }
            }
        }
    }

    #[test]
    fn luma16_matches_the_8_lane_kernel() {
        if !super::available() {
            return;
        }
        let st = 24;
        let mut x = 99u32;
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
                    // SAFETY: AVX2 is available.
                    unsafe { super::luma16::<false>(&mut a, 20, &src, st, fx, fy, h) };
                    crate::inter::luma_vec_for_test(&mut b, 20, &src, st, fx, fy, h);
                    assert_eq!(a, b, "h {h} fx {fx} fy {fy}");
                }
            }
        }
    }
}
