//! Inter prediction samples: fractional sample interpolation (8.4.2.2) and
//! weighted sample prediction (8.4.2.3): fast 8-bit kernels, and 32-bit
//! ones for higher bit depths behind the generic `*_t` entry points.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use crate::picture::{Pixel, PlaneView};

/// Maximum partition size; prediction blocks are written with stride 16.
pub const MAX: usize = 16;

/// Window of reference samples around a block: (x0 - 2, y0 - 2) is
/// `data[0]`. Inside the picture it borrows the plane; at the edges it is a
/// copy with coordinates clamped into the picture (8-228, 8-229).
const WIN: usize = MAX + 5;

/// The window when it lies inside the picture.
#[inline(always)]
fn inner_window<'a>(p: &PlaneView<'a>, x0: i32, y0: i32, w: usize, h: usize) -> Option<&'a [u8]> {
    let (pw, ph) = (p.width as i32, p.height as i32);
    (x0 - 2 >= 0 && y0 - 2 >= 0 && x0 + w as i32 + 3 <= pw && y0 + h as i32 + 3 <= ph).then(|| {
        let o = (y0 - 2) as usize * p.stride + (x0 - 2) as usize;
        p.samples(o, (h + 4) * p.stride + w + 5)
    })
}

/// The window of a W-wide block as a copy into `edge` (row stride WIN),
/// for any position.
fn edge_window<const W: usize>(
    p: &PlaneView,
    x0: i32,
    y0: i32,
    h: usize,
    edge: &mut [u8; WIN * WIN],
) {
    let (pw, ph) = (p.width as i32, p.height as i32);
    let rows = (0..h + 5).map(|r| (y0 - 2 + r as i32).clamp(0, ph - 1) as usize);
    if x0 - 2 >= 0 && x0 + W as i32 + 3 <= pw {
        // Columns inside (only rows clamped): copies of a fixed size.
        let x = (x0 - 2) as usize;
        for (r, yy) in rows.enumerate() {
            edge[r * WIN..r * WIN + W + 5].copy_from_slice(p.samples(yy * p.stride + x, W + 5));
        }
        return;
    }
    for (r, yy) in rows.enumerate() {
        clamped_row(
            &mut edge[r * WIN..r * WIN + W + 5],
            p.samples(yy * p.stride, p.width),
            x0 - 2,
        );
    }
}

/// Samples `x..x + out.len()` of `row`, with coordinates clamped into it
/// (8-228): the edge sample repeats on either side.
#[inline]
fn clamped_row<T: Copy>(out: &mut [T], row: &[T], x: i32) {
    let (n, w) = (out.len() as i32, row.len() as i32);
    let left = (-x).clamp(0, n) as usize;
    let right = (w - x).clamp(0, n) as usize;
    if left < right {
        let from = (x + left as i32) as usize;
        out[left..right].copy_from_slice(&row[from..from + right - left]);
    }
    out[..left].fill(row[0]);
    out[right.max(left)..].fill(row[w as usize - 1]);
}

#[inline(always)]
fn tap(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32) -> i32 {
    a - 5 * b + 20 * c + 20 * d - 5 * e + f
}

#[inline(always)]
fn clip(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// Unrounded horizontal 6-tap over `s[c..c + 6]` for each output column; the
/// sum fits in i16 (-2550..=10710).
#[inline(always)]
fn h_taps<const W: usize>(s: &[u8]) -> [i16; W] {
    let s = &s[..W + 5];
    let mut o = [0i16; W];
    for c in 0..W {
        let t = |k: usize| s[c + k] as i16;
        o[c] = t(0) - 5 * t(1) + 20 * t(2) + 20 * t(3) - 5 * t(4) + t(5);
    }
    o
}

/// Copies `h` rows of W samples from `src` (row stride `st`) to `dst`
/// (stride `ds`), with one bounds check for the block.
#[inline(always)]
fn copy_rows<const W: usize>(dst: &mut [u8], ds: usize, src: &[u8], st: usize, h: usize) {
    copy_rows_a::<W, false>(dst, ds, src, st, h)
}

/// [`copy_rows`], or with `A` averaged into `dst` (the second prediction
/// of default bi-prediction, 8-273).
#[inline(always)]
fn copy_rows_a<const W: usize, const A: bool>(
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    st: usize,
    h: usize,
) {
    if h == 0 {
        return;
    }
    assert!(src.len() >= (h - 1) * st + W && dst.len() >= (h - 1) * ds + W);
    for r in 0..h {
        // SAFETY: rows 0..h of W samples are within both (asserted above).
        unsafe {
            let row = *src.as_ptr().add(r * st).cast::<[u8; W]>();
            put::<W, A>(&mut *dst.as_mut_ptr().add(r * ds).cast::<[u8; W]>(), row);
        }
    }
}

/// Writes `v` to `o`, or with `A` its average with `o` (8-273).
#[inline(always)]
fn put<const W: usize, const A: bool>(o: &mut [u8; W], v: [u8; W]) {
    *o = if A { avg(*o, v) } else { v };
}

/// Rounded half sample from an unrounded tap.
#[inline(always)]
fn half<const W: usize>(t: [i16; W]) -> [u8; W] {
    t.map(|v| ((v + 16) >> 5).clamp(0, 255) as u8)
}

/// Vertical half samples h at window column `col` for one block row
/// (window rows r..r + 6).
#[inline(always)]
fn v_half<const W: usize>(src: &[u8], stride: usize, r: usize, col: usize) -> [u8; W] {
    let row = |k: usize| &src[(r + k) * stride + col..(r + k) * stride + col + W];
    let (r0, r1, r2, r3, r4, r5) = (row(0), row(1), row(2), row(3), row(4), row(5));
    let mut o = [0u8; W];
    for c in 0..W {
        let v = r0[c] as i16 - 5 * r1[c] as i16 + 20 * r2[c] as i16 + 20 * r3[c] as i16
            - 5 * r4[c] as i16
            + r5[c] as i16;
        o[c] = ((v + 16) >> 5).clamp(0, 255) as u8;
    }
    o
}

/// Integer samples G of one block row starting at window column `col`.
#[inline(always)]
fn g_row<const W: usize>(src: &[u8], stride: usize, r: usize, col: usize) -> [u8; W] {
    src[(r + 2) * stride + col..(r + 2) * stride + col + W]
        .try_into()
        .unwrap()
}

#[inline(always)]
fn avg<const W: usize>(a: [u8; W], b: [u8; W]) -> [u8; W] {
    let mut o = [0u8; W];
    for c in 0..W {
        o[c] = ((a[c] as u16 + b[c] as u16 + 1) >> 1) as u8;
    }
    o
}

/// Luma sample interpolation (8.4.2.2.1) of a w x h block whose integer
/// position is (x, y) with quarter-sample fraction (fx, fy).
///
/// Quarter positions average two of the half-sample values b (horizontal),
/// h (vertical) and j (centre) or an integer sample, as in Table 8-12, with
/// H = G(+1, 0), M = G(0, +1), m = h(+1, 0) and s = b(0, +1).
/// The block is written to `dst` with row stride `ds`.
pub fn luma(
    dst: &mut [u8],
    ds: usize,
    p: &PlaneView,
    x: i32,
    y: i32,
    fx: usize,
    fy: usize,
    w: usize,
    h: usize,
) {
    luma_a::<false>(dst, ds, p, x, y, fx, fy, w, h)
}

/// [`luma`], or with `A` averaged into `dst` (default bi-prediction).
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(crate) fn luma_a<const A: bool>(
    dst: &mut [u8],
    ds: usize,
    p: &PlaneView,
    x: i32,
    y: i32,
    fx: usize,
    fy: usize,
    w: usize,
    h: usize,
) {
    match w {
        16 => luma_w::<16, A>(dst, ds, p, x, y, fx, fy, h),
        8 => luma_w::<8, A>(dst, ds, p, x, y, fx, fy, h),
        _ => luma_w::<4, A>(dst, ds, p, x, y, fx, fy, h),
    }
}

fn luma_w<const W: usize, const A: bool>(
    dst: &mut [u8],
    ds: usize,
    p: &PlaneView,
    x: i32,
    y: i32,
    fx: usize,
    fy: usize,
    h: usize,
) {
    if let Some(src) = inner_window(p, x, y, W, h) {
        return luma_src::<W, A>(dst, ds, src, p.stride, fx, fy, h);
    }
    luma_edge::<W, A>(dst, ds, p, x, y, fx, fy, h)
}

/// luma_w for a block whose window crosses the picture edge.
#[cold]
#[inline(never)]
fn luma_edge<const W: usize, const A: bool>(
    dst: &mut [u8],
    ds: usize,
    p: &PlaneView,
    x: i32,
    y: i32,
    fx: usize,
    fy: usize,
    h: usize,
) {
    let mut edge = [0u8; WIN * WIN];
    edge_window::<W>(p, x, y, h, &mut edge);
    luma_src::<W, A>(dst, ds, &edge, WIN, fx, fy, h)
}

/// luma_w from the window `src` with row stride `st`.
#[inline(always)]
fn luma_src<const W: usize, const A: bool>(
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    st: usize,
    fx: usize,
    fy: usize,
    h: usize,
) {
    // 8 x i16 vectors: faster everywhere for the positions with the centre
    // sample j; for the others faster on x86 and Cortex cores, slower than
    // the code below on Apple cores.
    if (fx, fy) == (0, 0) {
        // Integer positions are a copy of fixed-size rows.
        return copy_rows_a::<W, A>(dst, ds, &src[2 * st + 2..], st, h);
    }
    #[cfg(all(target_arch = "aarch64", not(altc_scalar)))]
    if W >= 8 {
        return crate::inter_neon::luma::<W, A>(dst, ds, src, st, fx, fy, h);
    }
    #[cfg(all(target_arch = "x86_64", not(altc_scalar)))]
    if W == 16 && crate::inter_avx2::available() {
        // SAFETY: the CPU has AVX2.
        return unsafe { crate::inter_avx2::luma16::<A>(dst, ds, src, st, fx, fy, h) };
    }
    #[cfg(all(target_arch = "x86_64", not(altc_scalar)))]
    if W == 8 && crate::inter_avx2::ssse3() {
        // SAFETY: the CPU has SSSE3.
        return unsafe { crate::inter_avx2::luma8::<A>(dst, ds, src, st, fx, fy, h) };
    }
    let j = matches!((fx, fy), (2, 1..=3) | (1 | 3, 2));
    if W == 4 || j || cfg!(not(target_vendor = "apple")) {
        return luma_vec::<W, A>(dst, ds, src, st, fx, fy, h);
    }
    let b = |r: usize| half::<W>(h_taps::<W>(&src[(r + 2) * st..]));
    let mut out =
        |r: usize, v: [u8; W]| put::<W, A>((&mut dst[r * ds..r * ds + W]).try_into().unwrap(), v);
    match (fx, fy) {
        (0, 0) => (0..h).for_each(|r| out(r, g_row::<W>(src, st, r, 2))),
        (1, 0) => (0..h).for_each(|r| out(r, avg(g_row(src, st, r, 2), b(r)))),
        (2, 0) => (0..h).for_each(|r| out(r, b(r))),
        (3, 0) => (0..h).for_each(|r| out(r, avg(g_row(src, st, r, 3), b(r)))),
        (0, 1) => (0..h).for_each(|r| out(r, avg(g_row(src, st, r, 2), v_half(src, st, r, 2)))),
        (0, 2) => (0..h).for_each(|r| out(r, v_half(src, st, r, 2))),
        (0, 3) => (0..h).for_each(|r| out(r, avg(g_row(src, st, r + 1, 2), v_half(src, st, r, 2)))),
        (1, 1) => (0..h).for_each(|r| out(r, avg(b(r), v_half(src, st, r, 2)))),
        (3, 1) => (0..h).for_each(|r| out(r, avg(b(r), v_half(src, st, r, 3)))),
        (1, 3) => (0..h).for_each(|r| out(r, avg(v_half(src, st, r, 2), b(r + 1)))),
        (3, 3) => (0..h).for_each(|r| out(r, avg(v_half(src, st, r, 3), b(r + 1)))),
        _ => {
            // Cases with j: unrounded horizontal taps of all h + 5 rows,
            // filtered vertically.
            let mut b1 = [[0i16; W]; MAX + 5];
            for (r, row) in b1.iter_mut().enumerate().take(h + 5) {
                *row = h_taps::<W>(&src[r * st..]);
            }
            for r in 0..h {
                let mut j = [0u8; W];
                for c in 0..W {
                    let t = |k: usize| b1[r + k][c] as i32;
                    j[c] = clip((tap(t(0), t(1), t(2), t(3), t(4), t(5)) + 512) >> 10);
                }
                let v = match (fx, fy) {
                    (2, 2) => j,
                    (2, 1) => avg(half(b1[r + 2]), j),
                    (2, 3) => avg(j, half(b1[r + 3])),
                    (1, 2) => avg(v_half(src, st, r, 2), j),
                    _ => avg(j, v_half(src, st, r, 3)), // (3, 2)
                };
                out(r, v);
            }
        }
    }
}

/// luma_vec for 16-wide blocks, for comparison in tests.
#[cfg(all(
    test,
    any(target_arch = "x86_64", target_arch = "aarch64"),
    not(altc_scalar)
))]
pub(crate) fn luma_vec_for_test(
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    st: usize,
    fx: usize,
    fy: usize,
    h: usize,
) {
    luma_vec::<16, false>(dst, ds, src, st, fx, fy, h)
}

/// luma_vec for 8-wide blocks, for comparison in tests.
#[cfg(all(
    test,
    any(target_arch = "x86_64", target_arch = "aarch64"),
    not(altc_scalar)
))]
pub(crate) fn luma_vec8_for_test(
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    st: usize,
    fx: usize,
    fy: usize,
    h: usize,
) {
    luma_vec::<8, false>(dst, ds, src, st, fx, fy, h)
}

/// luma_w for widths 8 and 16 at fractional positions (Table 8-12): 8
/// columns of a row per vector, or for width 4 rows r and r + 1 side by
/// side (h is even). The 6-tap sums fit in i16.
#[inline(always)]
fn luma_vec<const W: usize, const A: bool>(
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    st: usize,
    fx: usize,
    fy: usize,
    h: usize,
) {
    use crate::simd::I16x8 as V;
    // The window holds rows 0..h + 5 of W + 5 samples, checked once here.
    assert!(h <= MAX && src.len() >= (h + 4) * st + W + 5);
    assert!(dst.len() >= (h - 1) * ds + W);
    // Rows of a vector, and rows a step down the block moves.
    let pair = W == 4;
    let step = if pair { 2 } else { 1 };
    // SAFETY: every load below reads 8 samples (4 from each of two rows)
    // within the window asserted above (columns up to W + 5, rows up to
    // h + 4).
    let ld = |o: usize| unsafe {
        if pair {
            let row = |o: usize| src.as_ptr().add(o).cast::<u32>().read_unaligned() as u64;
            V::load_u8(&(row(o) | row(o + st) << 32).to_le_bytes())
        } else {
            V::load_u8(&*src.as_ptr().add(o).cast::<[u8; 8]>())
        }
    };
    let (c5, c20, c16, c1, zero, max) = (
        V::splat(5),
        V::splat(20),
        V::splat(16),
        V::splat(1),
        V::splat(0),
        V::splat(255),
    );
    // Unrounded 6-tap sum of six samples `o, o + step, ...`.
    let raw = |o: usize, step: usize| {
        let v = |k: usize| ld(o + k * step);
        v(0).add(v(5))
            .sub(v(1).add(v(4)).mul(c5))
            .add(v(2).add(v(3)).mul(c20))
    };
    let round = |t: V| t.add(c16).shr::<5>().max(zero).min(max);
    // Rounded, clipped half sample from six samples `o, o + step, ...`.
    let tap = |o: usize, step: usize| round(raw(o, step));
    // Half samples b (horizontal) and h (vertical) for block row r, columns
    // c.. of the block; G is the integer sample.
    let b = |r: usize, c: usize| tap((r + 2) * st + c, 1);
    let hv = |r: usize, c: usize| tap(r * st + c + 2, st);
    let g = |r: usize, c: usize| ld((r + 2) * st + c + 2);
    let avg = |a: V, b: V| a.add(b).add(c1).shr::<1>();
    let mut store = |r: usize, c: usize, v: V| {
        if pair {
            let mut b = [0; 8];
            v.store_u8(&mut b);
            let (b0, b1) = (b[..4].try_into().unwrap(), b[4..].try_into().unwrap());
            put::<4, A>((&mut dst[r * ds..r * ds + 4]).try_into().unwrap(), b0);
            put::<4, A>(
                (&mut dst[(r + 1) * ds..(r + 1) * ds + 4])
                    .try_into()
                    .unwrap(),
                b1,
            );
        } else {
            // SAFETY: within dst, asserted above.
            let o = unsafe { &mut *dst.as_mut_ptr().add(r * ds + c).cast::<[u8; 8]>() };
            let v = if A { avg(V::load_u8(o), v) } else { v };
            v.store_u8(o)
        }
    };
    if matches!((fx, fy), (2, 1..=3) | (1 | 3, 2)) {
        // Positions with the centre sample j: the vertical 6-tap over the
        // unrounded horizontal sums of window rows r..r + 6, kept as a
        // sliding window down the block.
        for c in (0..W).step_by(8) {
            // Named values, not a rotated array (which would live in memory).
            let (mut w0, mut w1, mut w2, mut w3, mut w4, mut w5) = (
                raw(c, 1),
                raw(st + c, 1),
                raw(2 * st + c, 1),
                raw(3 * st + c, 1),
                raw(4 * st + c, 1),
                raw(5 * st + c, 1),
            );
            for r in (0..h).step_by(step) {
                if r > 0 && pair {
                    (w0, w1, w2, w3) = (w2, w3, w4, w5);
                    w4 = raw((r + 4) * st + c, 1);
                    w5 = raw((r + 5) * st + c, 1);
                } else if r > 0 {
                    (w0, w1, w2, w3, w4) = (w1, w2, w3, w4, w5);
                    w5 = raw((r + 5) * st + c, 1);
                }
                let j = crate::simd::tap6_10([w0, w1, w2, w3, w4, w5])
                    .max(zero)
                    .min(max);
                let v = match (fx, fy) {
                    (2, 2) => j,
                    (2, 1) => avg(round(w2), j),
                    (2, 3) => avg(j, round(w3)),
                    (1, 2) => avg(hv(r, c), j),
                    _ => avg(j, hv(r, c + 1)), // (3, 2)
                };
                store(r, c, v);
            }
        }
        return;
    }
    for r in (0..h).step_by(step) {
        for c in (0..W).step_by(8) {
            let v = match (fx, fy) {
                (0, 0) => g(r, c),
                (1, 0) => avg(g(r, c), b(r, c)),
                (2, 0) => b(r, c),
                (3, 0) => avg(g(r, c + 1), b(r, c)),
                (0, 1) => avg(g(r, c), hv(r, c)),
                (0, 2) => hv(r, c),
                (0, 3) => avg(g(r + 1, c), hv(r, c)),
                (1, 1) => avg(b(r, c), hv(r, c)),
                (3, 1) => avg(b(r, c), hv(r, c + 1)),
                (1, 3) => avg(hv(r, c), b(r + 1, c)),
                _ => avg(hv(r, c + 1), b(r + 1, c)), // (3, 3)
            };
            store(r, c, v);
        }
    }
}

/// Chroma sample interpolation (8.4.2.2.2) of a w x h block at integer
/// position (x, y) with eighth-sample fraction (fx, fy), written to `dst`
/// with row stride `ds`.
pub fn chroma(
    dst: &mut [u8],
    ds: usize,
    p: &PlaneView,
    x: i32,
    y: i32,
    fx: i32,
    fy: i32,
    w: usize,
    h: usize,
) {
    chroma_a::<false>(dst, ds, p, x, y, fx, fy, w, h)
}

/// [`chroma`], or with `A` averaged into `dst` (default bi-prediction).
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(crate) fn chroma_a<const A: bool>(
    dst: &mut [u8],
    ds: usize,
    p: &PlaneView,
    x: i32,
    y: i32,
    fx: i32,
    fy: i32,
    w: usize,
    h: usize,
) {
    match w {
        16 => chroma_w::<16, A>(dst, ds, p, x, y, fx, fy, h),
        8 => chroma_w::<8, A>(dst, ds, p, x, y, fx, fy, h),
        4 => chroma_w::<4, A>(dst, ds, p, x, y, fx, fy, h),
        _ => chroma_w::<2, A>(dst, ds, p, x, y, fx, fy, h),
    }
}

fn chroma_w<const W: usize, const A: bool>(
    dst: &mut [u8],
    ds: usize,
    p: &PlaneView,
    x: i32,
    y: i32,
    fx: i32,
    fy: i32,
    h: usize,
) {
    // (W + 1) x (h + 1) window; clamped coordinates (8-266, 8-267) at the edges.
    let (pw, ph) = (p.width as i32, p.height as i32);
    if x >= 0 && y >= 0 && x + (W as i32) < pw && y + (h as i32) < ph {
        let o = y as usize * p.stride + x as usize;
        let src = p.samples(o, h * p.stride + W + 1);
        return chroma_src::<W, A>(dst, ds, src, p.stride, fx, fy, h);
    }
    chroma_edge::<W, A>(dst, ds, p, x, y, fx, fy, h)
}

/// chroma_w for a block whose window crosses the picture edge.
#[cold]
#[inline(never)]
fn chroma_edge<const W: usize, const A: bool>(
    dst: &mut [u8],
    ds: usize,
    p: &PlaneView,
    x: i32,
    y: i32,
    fx: i32,
    fy: i32,
    h: usize,
) {
    let ph = p.height as i32;
    let mut edge = [0u8; (MAX + 1) * (MAX + 1)];
    for r in 0..=h {
        let yy = (y + r as i32).clamp(0, ph - 1) as usize;
        let out = &mut edge[r * (MAX + 1)..r * (MAX + 1) + W + 1];
        clamped_row(out, p.samples(yy * p.stride, p.width), x);
    }
    chroma_src::<W, A>(dst, ds, &edge, MAX + 1, fx, fy, h)
}

/// chroma_w from the window `src` with row stride `st`.
#[inline(always)]
fn chroma_src<const W: usize, const A: bool>(
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    st: usize,
    fx: i32,
    fy: i32,
    h: usize,
) {
    if (fx, fy) == (0, 0) {
        // Integer positions are a copy of fixed-size rows.
        return copy_rows_a::<W, A>(dst, ds, src, st, h);
    }
    if W >= 4 {
        return chroma_vec::<W, A>(dst, ds, src, st, fx, fy, h);
    }
    let (wa, wb, wc, wd) = (
        ((8 - fx) * (8 - fy)) as u16,
        (fx * (8 - fy)) as u16,
        ((8 - fx) * fy) as u16,
        (fx * fy) as u16,
    );
    for r in 0..h {
        let top = &src[r * st..r * st + W + 1];
        let bot = &src[(r + 1) * st..(r + 1) * st + W + 1];
        let mut v = [0u8; W];
        for c in 0..W {
            v[c] = ((wa * top[c] as u16
                + wb * top[c + 1] as u16
                + wc * bot[c] as u16
                + wd * bot[c + 1] as u16
                + 32)
                >> 6) as u8;
        }
        put::<W, A>((&mut dst[r * ds..r * ds + W]).try_into().unwrap(), v);
    }
}

/// chroma_w for widths of 4 or more in 8 x i16 vectors: a row of 8 per
/// vector, or two rows of 4. The weights sum to 64, so the sums fit.
#[inline(always)]
fn chroma_vec<const W: usize, const A: bool>(
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    st: usize,
    fx: i32,
    fy: i32,
    h: usize,
) {
    use crate::simd::I16x8 as V;
    let w = |v: i32| V::splat(v as i16);
    let (wa, wb, wc, wd) = (
        w((8 - fx) * (8 - fy)),
        w(fx * (8 - fy)),
        w((8 - fx) * fy),
        w(fx * fy),
    );
    let c32 = V::splat(32);
    let px = |a: V, b: V, c: V, d: V| {
        wa.mul(a)
            .add(wb.mul(b))
            .add(wc.mul(c))
            .add(wd.mul(d))
            .add(c32)
            .shr::<6>()
    };
    if W == 4 {
        // Rows r and r + 1 side by side.
        let ld4 = |o0: usize, o1: usize| {
            let mut b = [0u8; 8];
            b[..4].copy_from_slice(&src[o0..o0 + 4]);
            b[4..].copy_from_slice(&src[o1..o1 + 4]);
            V::load_u8(&b)
        };
        let mut r = 0;
        while r < h {
            let (o0, o1, o2) = (r * st, (r + 1) * st, (r + 2) * st);
            let v = px(
                ld4(o0, o1),
                ld4(o0 + 1, o1 + 1),
                ld4(o1, o2),
                ld4(o1 + 1, o2 + 1),
            );
            let mut out = [0u8; 8];
            v.store_u8(&mut out);
            let (b0, b1) = (out[..4].try_into().unwrap(), out[4..].try_into().unwrap());
            put::<4, A>((&mut dst[r * ds..r * ds + 4]).try_into().unwrap(), b0);
            if r + 1 < h {
                put::<4, A>(
                    (&mut dst[(r + 1) * ds..(r + 1) * ds + 4])
                        .try_into()
                        .unwrap(),
                    b1,
                );
            }
            r += 2;
        }
        return;
    }
    // The weights are separable: each row is blended horizontally once and
    // serves as the bottom of one output row and the top of the next (the
    // sums are the same, at most 64 * 255).
    let (h0, h1, v0, v1) = (w(8 - fx), w(fx), w(8 - fy), w(fy));
    // The window holds rows 0..=h of W + 1 samples, checked once here.
    assert!(h >= 1 && src.len() > h * st + W && dst.len() >= (h - 1) * ds + W);
    // SAFETY: reads 8 samples within the window asserted above.
    let ldu = |o: usize| unsafe { V::load_u8(&*src.as_ptr().add(o).cast::<[u8; 8]>()) };
    let hrow = |o: usize| h0.mul(ldu(o)).add(h1.mul(ldu(o + 1)));
    for c in (0..W).step_by(8) {
        let mut top = hrow(c);
        for r in 0..h {
            let bot = hrow((r + 1) * st + c);
            let v = v0.mul(top).add(v1.mul(bot)).add(c32).shr::<6>();
            // SAFETY: within dst, asserted above.
            let o = unsafe { &mut *dst.as_mut_ptr().add(r * ds + c).cast::<[u8; 8]>() };
            let v = if A {
                v.add(V::load_u8(o)).add(V::splat(1)).shr::<1>()
            } else {
                v
            };
            v.store_u8(o);
            top = bot;
        }
    }
}

/// [`chroma`] of the same block of Cb and Cr (`p`, written to `dst`), both
/// planes in one vector where a kernel for the width does that.
pub fn chroma_pair(
    dst: [&mut [u8]; 2],
    ds: usize,
    p: [&PlaneView; 2],
    x: i32,
    y: i32,
    fx: i32,
    fy: i32,
    w: usize,
    h: usize,
) {
    chroma_pair_a::<false>(dst, ds, p, x, y, fx, fy, w, h)
}

/// [`chroma_pair`], or with `A` averaged into `dst` (default bi-prediction).
#[allow(clippy::too_many_arguments)]
#[inline(never)]
#[allow(clippy::needless_return)] // the SIMD path below is compiled out in scalar builds
pub(crate) fn chroma_pair_a<const A: bool>(
    dst: [&mut [u8]; 2],
    ds: usize,
    p: [&PlaneView; 2],
    x: i32,
    y: i32,
    fx: i32,
    fy: i32,
    w: usize,
    h: usize,
) {
    let (pw, ph, st) = (p[0].width as i32, p[0].height as i32, p[0].stride);
    let paired = (w == 8 || w == 4)
        && h >= 1
        && (fx, fy) != (0, 0)
        && x >= 0
        && y >= 0
        && x + (w as i32) < pw
        && y + (h as i32) < ph
        && (p[1].stride, p[1].width, p[1].height) == (st, p[0].width, p[0].height);
    #[cfg(all(target_arch = "x86_64", not(altc_scalar)))]
    let paired = paired && crate::inter_avx2::available();
    #[cfg(not(all(any(target_arch = "x86_64", target_arch = "aarch64"), not(altc_scalar))))]
    let paired = paired && false;
    let [d0, d1] = dst;
    if !paired {
        chroma_a::<A>(d0, ds, p[0], x, y, fx, fy, w, h);
        return chroma_a::<A>(d1, ds, p[1], x, y, fx, fy, w, h);
    }
    #[cfg(all(any(target_arch = "x86_64", target_arch = "aarch64"), not(altc_scalar)))]
    {
        #[cfg(all(target_arch = "x86_64", not(altc_scalar)))]
        use crate::inter_avx2::chroma_pair as kernel;
        #[cfg(target_arch = "aarch64")]
        use crate::inter_neon::chroma_pair as kernel;
        // The (w + 1) x (h + 1) windows of both planes.
        let (o, n) = (y as usize * st + x as usize, h * st + w + 1);
        let s = [
            p[0].samples::<u8>(o, n).as_ptr(),
            p[1].samples::<u8>(o, n).as_ptr(),
        ];
        let rows = (h - 1) * ds + w;
        assert!(d0.len() >= rows && d1.len() >= rows);
        let d = [d0.as_mut_ptr(), d1.as_mut_ptr()];
        // SAFETY: the windows and outputs hold the rows (checked above);
        // on x86 the CPU has AVX2.
        unsafe {
            match w {
                8 => kernel::<8, A>(d, ds, s, st, fx, fy, h),
                _ => kernel::<4, A>(d, ds, s, st, fx, fy, h),
            }
        }
    }
}

/// Explicit or implicit weights for one prediction (8.4.2.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Weight {
    pub log_wd: u32,
    pub w: i32,
    pub o: i32,
}

/// Writes the final prediction of one partition into `out` (stride `stride`).
/// `p1` is the list 1 prediction for bi-prediction. `wt` selects weighted
/// prediction (8.4.2.3.2); None means default (8.4.2.3.1).
pub fn combine(
    out: &mut [u8],
    stride: usize,
    p0: Option<&[u8; MAX * MAX]>,
    p1: Option<&[u8; MAX * MAX]>,
    wt: Option<(Weight, Weight)>,
    w: usize,
    h: usize,
) {
    #[inline(always)]
    fn rows(
        out: &mut [u8],
        stride: usize,
        w: usize,
        h: usize,
        mut f: impl FnMut(&mut [u8], usize),
    ) {
        for r in 0..h {
            f(&mut out[r * stride..r * stride + w], r * MAX);
        }
    }
    // Default prediction of the common widths with fixed-size rows.
    #[inline(always)]
    fn fixed<const W: usize>(
        out: &mut [u8],
        stride: usize,
        a: &[u8; MAX * MAX],
        b: Option<&[u8; MAX * MAX]>,
        h: usize,
    ) {
        let Some(b) = b else {
            return copy_rows::<W>(out, stride, a, MAX, h);
        };
        assert!(h <= MAX && (h == 0 || out.len() >= (h - 1) * stride + W));
        for r in 0..h {
            let x: &[u8; W] = a[r * MAX..r * MAX + W].try_into().unwrap();
            let y: &[u8; W] = b[r * MAX..r * MAX + W].try_into().unwrap();
            // SAFETY: rows 0..h of W samples are within out (asserted above).
            let o = unsafe { &mut *out.as_mut_ptr().add(r * stride).cast::<[u8; W]>() };
            for c in 0..W {
                o[c] = ((x[c] as u16 + y[c] as u16 + 1) >> 1) as u8;
            }
        }
    }
    if wt.is_none()
        && let Some(a) = p0.or(p1)
    {
        let b = p0.and(p1);
        match w {
            16 => return fixed::<16>(out, stride, a, b, h),
            8 => return fixed::<8>(out, stride, a, b, h),
            _ => {}
        }
    }
    // Loops over zipped rows so they vectorise (no bounds checks inside).
    match (p0, p1, wt) {
        (Some(a), Some(b), None) => rows(out, stride, w, h, |row, k| {
            let n = row.len();
            for (o, (&x, &y)) in row.iter_mut().zip(a[k..k + n].iter().zip(&b[k..k + n])) {
                *o = ((x as u16 + y as u16 + 1) >> 1) as u8;
            }
        }),
        (Some(a), None, None) | (None, Some(a), None) => rows(out, stride, w, h, |row, k| {
            row.copy_from_slice(&a[k..k + row.len()])
        }),
        (Some(a), Some(b), Some((w0, w1)))
            if weighted_simd::<true>(out, stride, a, b, w0, w1, w, h) => {}
        (Some(a), None, Some((wx, _))) | (None, Some(a), Some((_, wx)))
            if weighted_simd::<false>(out, stride, a, a, wx, wx, w, h) => {}
        (Some(a), Some(b), Some((w0, w1))) => {
            let (round, shift, off) = (1 << w0.log_wd, w0.log_wd + 1, (w0.o + w1.o + 1) >> 1);
            rows(out, stride, w, h, |row, k| {
                let n = row.len();
                for (o, (&x, &y)) in row.iter_mut().zip(a[k..k + n].iter().zip(&b[k..k + n])) {
                    *o = clip(((x as i32 * w0.w + y as i32 * w1.w + round) >> shift) + off);
                }
            })
        }
        (Some(a), None, Some((wx, _))) | (None, Some(a), Some((_, wx))) => {
            if wx.log_wd >= 1 {
                let round = 1 << (wx.log_wd - 1);
                rows(out, stride, w, h, |row, k| {
                    let n = row.len();
                    for (o, &x) in row.iter_mut().zip(&a[k..k + n]) {
                        *o = clip(((x as i32 * wx.w + round) >> wx.log_wd) + wx.o);
                    }
                })
            } else {
                rows(out, stride, w, h, |row, k| {
                    let n = row.len();
                    for (o, &x) in row.iter_mut().zip(&a[k..k + n]) {
                        *o = clip(x as i32 * wx.w + wx.o);
                    }
                })
            }
        }
        (None, None, _) => {}
    }
}

/// Weighted prediction of blocks 4 or more samples wide with a vector
/// kernel, where there is one; false when there is none.
#[inline(always)]
#[cfg_attr(
    not(all(any(target_arch = "x86_64", target_arch = "aarch64"), not(altc_scalar))),
    allow(unused_variables)
)]
fn weighted_simd<const BI: bool>(
    out: &mut [u8],
    stride: usize,
    a: &[u8; MAX * MAX],
    b: &[u8; MAX * MAX],
    w0: Weight,
    w1: Weight,
    w: usize,
    h: usize,
) -> bool {
    if w < 4
        || h == 0
        || cfg!(not(all(
            any(target_arch = "x86_64", target_arch = "aarch64"),
            not(altc_scalar)
        )))
    {
        return false;
    }
    #[cfg(all(any(target_arch = "x86_64", target_arch = "aarch64"), not(altc_scalar)))]
    {
        #[cfg(all(target_arch = "x86_64", not(altc_scalar)))]
        use crate::inter_avx2::weighted;
        #[cfg(target_arch = "aarch64")]
        use crate::inter_neon::weighted;
        // (8-301): ((a * w0 + b * w1 + 2^logWD) >> (logWD + 1)) + offset;
        // (8-297, 8-298): ((a * w + 2^(logWD - 1)) >> logWD) + o, or a * w + o.
        let (k, s) = if BI {
            let s = w0.log_wd + 1;
            ((1 << w0.log_wd) + (((w0.o + w1.o + 1) >> 1) << s), s)
        } else {
            let s = w0.log_wd;
            (((1 << s) >> 1) + (w0.o << s), s)
        };
        assert!(out.len() >= (h - 1) * stride + w && w <= MAX && h <= MAX && (w > 4 || h % 2 == 0));
        // SAFETY: the rows are within out (asserted above) and the buffers.
        unsafe {
            weighted::<BI>(
                out.as_mut_ptr(),
                stride,
                a.as_ptr(),
                b.as_ptr(),
                (w0.w, w1.w),
                k,
                s,
                w,
                h,
            )
        };
    }
    true
}

/// [`luma`] for any sample type; `max` is (1 << BitDepthY) - 1. With
/// `avg` (8-bit only), averaged into `dst` (default bi-prediction).
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn luma_t<T: Pixel>(
    dst: &mut [T],
    ds: usize,
    p: &PlaneView,
    x: i32,
    y: i32,
    fx: usize,
    fy: usize,
    w: usize,
    h: usize,
    max: i32,
    avg: bool,
) {
    if T::HIGH {
        debug_assert!(!avg);
        luma_wide(dst, ds, p, x, y, fx, fy, w, h, max);
    } else if avg {
        luma_a::<true>(T::u8s_mut(dst), ds, p, x, y, fx, fy, w, h);
    } else {
        luma_a::<false>(T::u8s_mut(dst), ds, p, x, y, fx, fy, w, h);
    }
}

/// [`chroma`] for any sample type, with `avg` as in [`luma_t`].
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn chroma_t<T: Pixel>(
    dst: &mut [T],
    ds: usize,
    p: &PlaneView,
    x: i32,
    y: i32,
    fx: i32,
    fy: i32,
    w: usize,
    h: usize,
    avg: bool,
) {
    if T::HIGH {
        debug_assert!(!avg);
        chroma_wide(dst, ds, p, x, y, fx, fy, w, h);
    } else if avg {
        chroma_a::<true>(T::u8s_mut(dst), ds, p, x, y, fx, fy, w, h);
    } else {
        chroma_a::<false>(T::u8s_mut(dst), ds, p, x, y, fx, fy, w, h);
    }
}

/// [`combine`] for any sample type; `max` is (1 << BitDepth) - 1.
#[inline]
pub fn combine_t<T: Pixel>(
    out: &mut [T],
    stride: usize,
    p0: Option<&[T; MAX * MAX]>,
    p1: Option<&[T; MAX * MAX]>,
    wt: Option<(Weight, Weight)>,
    w: usize,
    h: usize,
    max: i32,
) {
    if !T::HIGH {
        fn cast<T: Pixel>(b: &[T; MAX * MAX]) -> &[u8; MAX * MAX] {
            T::u8s(b).try_into().unwrap()
        }
        return combine(
            T::u8s_mut(out),
            stride,
            p0.map(cast),
            p1.map(cast),
            wt,
            w,
            h,
        );
    }
    let clip = |v: i32| T::clip(v, max);
    for r in 0..h {
        for c in 0..w {
            let k = r * MAX + c;
            let a = p0.map(|p| p[k].get());
            let b = p1.map(|p| p[k].get());
            out[r * stride + c] = match (a, b, wt) {
                (Some(a), Some(b), None) => T::new((a + b + 1) >> 1),
                (Some(a), None, None) | (None, Some(a), None) => T::new(a),
                (Some(a), Some(b), Some((w0, w1))) => clip(
                    ((a * w0.w + b * w1.w + (1 << w0.log_wd)) >> (w0.log_wd + 1))
                        + ((w0.o + w1.o + 1) >> 1),
                ),
                (Some(a), None, Some((wx, _))) | (None, Some(a), Some((_, wx))) => {
                    if wx.log_wd >= 1 {
                        clip(((a * wx.w + (1 << (wx.log_wd - 1))) >> wx.log_wd) + wx.o)
                    } else {
                        clip(a * wx.w + wx.o)
                    }
                }
                (None, None, _) => return,
            };
        }
    }
}

/// Luma sample interpolation (8.4.2.2.1, Table 8-12) in 32 bits.
fn luma_wide<T: Pixel>(
    dst: &mut [T],
    ds: usize,
    p: &PlaneView,
    x: i32,
    y: i32,
    fx: usize,
    fy: usize,
    w: usize,
    h: usize,
    max: i32,
) {
    let (pw, ph) = (p.width as i32, p.height as i32);
    // Integer sample G at (xx, yy) relative to the block, clamped (8-228, 8-229).
    let g = |xx: i32, yy: i32| {
        let (sx, sy) = ((x + xx).clamp(0, pw - 1), (y + yy).clamp(0, ph - 1));
        p.at::<T>(sy as usize * p.stride + sx as usize).get()
    };
    let clip = |v: i32| v.clamp(0, max);
    let b1 = |xx: i32, yy: i32| {
        tap(
            g(xx - 2, yy),
            g(xx - 1, yy),
            g(xx, yy),
            g(xx + 1, yy),
            g(xx + 2, yy),
            g(xx + 3, yy),
        )
    };
    let h1 = |xx: i32, yy: i32| {
        tap(
            g(xx, yy - 2),
            g(xx, yy - 1),
            g(xx, yy),
            g(xx, yy + 1),
            g(xx, yy + 2),
            g(xx, yy + 3),
        )
    };
    let b = |xx: i32, yy: i32| clip((b1(xx, yy) + 16) >> 5);
    let hh = |xx: i32, yy: i32| clip((h1(xx, yy) + 16) >> 5);
    let j = |xx: i32, yy: i32| {
        let t = |k: i32| b1(xx, yy + k);
        clip((tap(t(-2), t(-1), t(0), t(1), t(2), t(3)) + 512) >> 10)
    };
    let avg = |a: i32, c: i32| (a + c + 1) >> 1;
    for r in 0..h as i32 {
        for c in 0..w as i32 {
            let v = match (fx, fy) {
                (0, 0) => g(c, r),
                (1, 0) => avg(g(c, r), b(c, r)),
                (2, 0) => b(c, r),
                (3, 0) => avg(g(c + 1, r), b(c, r)),
                (0, 1) => avg(g(c, r), hh(c, r)),
                (0, 2) => hh(c, r),
                (0, 3) => avg(g(c, r + 1), hh(c, r)),
                (1, 1) => avg(b(c, r), hh(c, r)),
                (3, 1) => avg(b(c, r), hh(c + 1, r)),
                (1, 3) => avg(hh(c, r), b(c, r + 1)),
                (3, 3) => avg(hh(c + 1, r), b(c, r + 1)),
                (2, 1) => avg(b(c, r), j(c, r)),
                (2, 3) => avg(j(c, r), b(c, r + 1)),
                (1, 2) => avg(hh(c, r), j(c, r)),
                (3, 2) => avg(j(c, r), hh(c + 1, r)),
                _ => j(c, r),
            };
            dst[r as usize * ds + c as usize] = T::new(v);
        }
    }
}

/// Chroma sample interpolation (8.4.2.2.2) in 32 bits.
fn chroma_wide<T: Pixel>(
    dst: &mut [T],
    ds: usize,
    p: &PlaneView,
    x: i32,
    y: i32,
    fx: i32,
    fy: i32,
    w: usize,
    h: usize,
) {
    let (pw, ph) = (p.width as i32, p.height as i32);
    let g = |xx: i32, yy: i32| {
        let (sx, sy) = ((x + xx).clamp(0, pw - 1), (y + yy).clamp(0, ph - 1));
        p.at::<T>(sy as usize * p.stride + sx as usize).get()
    };
    for r in 0..h as i32 {
        for c in 0..w as i32 {
            let v = ((8 - fx) * (8 - fy) * g(c, r)
                + fx * (8 - fy) * g(c + 1, r)
                + (8 - fx) * fy * g(c, r + 1)
                + fx * fy * g(c + 1, r + 1)
                + 32)
                >> 6;
            dst[r as usize * ds + c as usize] = T::new(v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::picture::Plane;

    // The straightforward implementations, kept to check the fast ones.
    /// Chroma sample interpolation (8.4.2.2.2) of a w x h block at integer
    /// position (x, y) with eighth-sample fraction (fx, fy).
    fn chroma_reference(
        dst: &mut [u8; MAX * MAX],
        p: &PlaneView,
        x: i32,
        y: i32,
        fx: i32,
        fy: i32,
        w: usize,
        h: usize,
    ) {
        // (w + 1) x (h + 1) window with clamped coordinates (8-266, 8-267).
        let (pw, ph) = (p.width as i32, p.height as i32);
        let mut s = [[0u8; MAX + 1]; MAX + 1];
        let inside = x >= 0 && y >= 0 && x + (w as i32) < pw && y + (h as i32) < ph;
        for (r, row) in s.iter_mut().enumerate().take(h + 1) {
            let yy = (y + r as i32).clamp(0, ph - 1) as usize * p.stride;
            if inside {
                row[..w + 1].copy_from_slice(p.samples(yy + x as usize, w + 1));
            } else {
                for c in 0..=w {
                    row[c] = p.at(yy + (x + c as i32).clamp(0, pw - 1) as usize);
                }
            }
        }
        let (wa, wb, wc, wd) = ((8 - fx) * (8 - fy), fx * (8 - fy), (8 - fx) * fy, fx * fy);
        for r in 0..h {
            let (top, bot) = (&s[r], &s[r + 1]);
            let out = &mut dst[r * MAX..r * MAX + w];
            for c in 0..w {
                out[c] = ((wa * top[c] as i32
                    + wb * top[c + 1] as i32
                    + wc * bot[c] as i32
                    + wd * bot[c + 1] as i32
                    + 32)
                    >> 6) as u8;
            }
        }
    }

    /// Copies a (w + 5) x (h + 5) window of `p` starting at (x0 - 2, y0 - 2)
    /// with coordinates clamped into the picture (8-228, 8-229).
    fn fetch_ref(
        p: &PlaneView,
        x0: i32,
        y0: i32,
        w: usize,
        h: usize,
        out: &mut [[u8; MAX + 5]; MAX + 5],
    ) {
        let (pw, ph) = (p.width as i32, p.height as i32);
        let inside =
            x0 - 2 >= 0 && y0 - 2 >= 0 && x0 + w as i32 + 3 <= pw && y0 + h as i32 + 3 <= ph;
        for (r, row) in out.iter_mut().enumerate().take(h + 5) {
            let y = y0 - 2 + r as i32;
            if inside {
                let o = y as usize * p.stride + (x0 - 2) as usize;
                row[..w + 5].copy_from_slice(p.samples(o, w + 5));
            } else {
                let yy = y.clamp(0, ph - 1) as usize * p.stride;
                for c in 0..w + 5 {
                    row[c] = p.at(yy + (x0 - 2 + c as i32).clamp(0, pw - 1) as usize);
                }
            }
        }
    }

    /// Luma sample interpolation (8.4.2.2.1) of a w x h block whose integer
    /// position is (x, y) with quarter-sample fraction (fx, fy).
    ///
    /// The half-sample planes b (horizontal), h (vertical) and j (centre) are
    /// computed once per block as needed; quarter positions average two of
    /// them as in Table 8-12, with H = G(+1, 0), M = G(0, +1), m = h(+1, 0)
    /// and s = b(0, +1).
    fn luma_reference(
        dst: &mut [u8; MAX * MAX],
        p: &PlaneView,
        x: i32,
        y: i32,
        fx: usize,
        fy: usize,
        w: usize,
        h: usize,
    ) {
        let mut s = [[0u8; MAX + 5]; MAX + 5];
        fetch_ref(p, x, y, w, h, &mut s);
        // Integer sample G(c, r) is s[r + 2][c + 2].
        let g = |c: usize, r: usize| s[r + 2][c + 2] as i32;
        if (fx, fy) == (0, 0) {
            for r in 0..h {
                dst[r * MAX..r * MAX + w].copy_from_slice(&s[r + 2][2..2 + w]);
            }
            return;
        }
        // b1 over window rows 0..h+5 (only the rows each case needs are used).
        let need_b = fx != 0;
        let need_h = fy != 0 && fx != 2;
        let need_j = fx == 2 && fy != 0 || fy == 2 && fx != 0;
        let mut b1 = [[0i32; MAX]; MAX + 5];
        if need_b || need_j {
            let rows = if need_j { 0..h + 5 } else { 2..h + 3 };
            for r in rows {
                let row = &s[r];
                for c in 0..w {
                    b1[r][c] = tap(
                        row[c] as i32,
                        row[c + 1] as i32,
                        row[c + 2] as i32,
                        row[c + 3] as i32,
                        row[c + 4] as i32,
                        row[c + 5] as i32,
                    );
                }
            }
        }
        // b at block row r (window row r + 2), r = 0..=h.
        let bv = |c: usize, r: usize| clip((b1[r + 2][c] + 16) >> 5) as i32;
        // h at block column c, c = 0..=w (column c + 1 for m).
        let mut hv = [[0i32; MAX + 1]; MAX];
        if need_h {
            for r in 0..h {
                for c in 0..=w {
                    let cc = c + 2;
                    let v = tap(
                        s[r][cc] as i32,
                        s[r + 1][cc] as i32,
                        s[r + 2][cc] as i32,
                        s[r + 3][cc] as i32,
                        s[r + 4][cc] as i32,
                        s[r + 5][cc] as i32,
                    );
                    hv[r][c] = clip((v + 16) >> 5) as i32;
                }
            }
        }
        let mut jv = [[0i32; MAX]; MAX];
        if need_j {
            for r in 0..h {
                for c in 0..w {
                    let j1 = tap(
                        b1[r][c],
                        b1[r + 1][c],
                        b1[r + 2][c],
                        b1[r + 3][c],
                        b1[r + 4][c],
                        b1[r + 5][c],
                    );
                    jv[r][c] = clip((j1 + 512) >> 10) as i32;
                }
            }
        }
        // One monomorphised loop per position: no per-sample branching.
        #[inline(always)]
        fn fill(dst: &mut [u8; MAX * MAX], w: usize, h: usize, f: impl Fn(usize, usize) -> i32) {
            for r in 0..h {
                let row = &mut dst[r * MAX..r * MAX + w];
                for (c, d) in row.iter_mut().enumerate() {
                    *d = f(c, r) as u8;
                }
            }
        }
        let avg = |a: i32, b: i32| (a + b + 1) >> 1;
        match (fx, fy) {
            (0, 1) => fill(dst, w, h, |c, r| avg(g(c, r), hv[r][c])),
            (0, 2) => fill(dst, w, h, |c, r| hv[r][c]),
            (0, 3) => fill(dst, w, h, |c, r| avg(g(c, r + 1), hv[r][c])),
            (1, 0) => fill(dst, w, h, |c, r| avg(g(c, r), bv(c, r))),
            (2, 0) => fill(dst, w, h, bv),
            (3, 0) => fill(dst, w, h, |c, r| avg(g(c + 1, r), bv(c, r))),
            (2, 2) => fill(dst, w, h, |c, r| jv[r][c]),
            (2, 1) => fill(dst, w, h, |c, r| avg(bv(c, r), jv[r][c])),
            (2, 3) => fill(dst, w, h, |c, r| avg(jv[r][c], bv(c, r + 1))),
            (1, 2) => fill(dst, w, h, |c, r| avg(hv[r][c], jv[r][c])),
            (3, 2) => fill(dst, w, h, |c, r| avg(jv[r][c], hv[r][c + 1])),
            (1, 1) => fill(dst, w, h, |c, r| avg(bv(c, r), hv[r][c])),
            (3, 1) => fill(dst, w, h, |c, r| avg(bv(c, r), hv[r][c + 1])),
            (1, 3) => fill(dst, w, h, |c, r| avg(hv[r][c], bv(c, r + 1))),
            _ => fill(dst, w, h, |c, r| avg(hv[r][c + 1], bv(c, r + 1))), // (3, 3)
        }
    }

    fn ramp() -> Plane {
        let mut p = Plane::new(32, 32);
        for y in 0..32 {
            for x in 0..32 {
                p.data[y * p.stride + x] = (x * 4 + y) as u8;
            }
        }
        p
    }

    #[test]
    fn integer_and_half_positions_on_a_ramp() {
        let p = ramp();
        let mut d = [0u8; MAX * MAX];
        luma(&mut d, MAX, &p.view(), 8, 8, 0, 0, 4, 4);
        assert_eq!(d[0], 40);
        // On a linear ramp the 6-tap filter is exact: half-sample = average.
        luma(&mut d, MAX, &p.view(), 8, 8, 2, 0, 4, 4);
        assert_eq!(d[0], 42);
        luma(&mut d, MAX, &p.view(), 8, 8, 0, 2, 4, 4);
        assert_eq!(d[0], 41); // taps over 38..43 = 1296, (1296 + 16) >> 5
    }

    #[test]
    fn luma_matches_reference_everywhere() {
        let mut p = Plane::new(48, 40);
        let mut seed = 1u32;
        for v in p.data.iter_mut() {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            *v = (seed >> 16) as u8;
        }
        let (mut a, mut b) = ([0u8; MAX * MAX], [0u8; MAX * MAX]);
        for (w, h) in [(16, 16), (16, 8), (8, 16), (8, 8), (8, 4), (4, 8), (4, 4)] {
            for fy in 0..4 {
                for fx in 0..4 {
                    for (x, y) in [
                        (-20, -20),
                        (-3, 5),
                        (0, 0),
                        (2, 2),
                        (16, 12),
                        (30, 22),
                        (45, 38),
                        (60, 50),
                    ] {
                        luma(&mut a, MAX, &p.view(), x, y, fx, fy, w, h);
                        luma_reference(&mut b, &p.view(), x, y, fx, fy, w, h);
                        for r in 0..h {
                            assert_eq!(
                                a[r * MAX..r * MAX + w],
                                b[r * MAX..r * MAX + w],
                                "{w}x{h} ({fx},{fy}) at ({x},{y}) row {r}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn chroma_matches_reference_everywhere() {
        let mut p = Plane::new(24, 20);
        let mut seed = 7u32;
        for v in p.data.iter_mut() {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            *v = (seed >> 16) as u8;
        }
        let (mut a, mut b) = ([0u8; MAX * MAX], [0u8; MAX * MAX]);
        for (w, h) in [
            (8, 8),
            (8, 4),
            (4, 8),
            (4, 4),
            (4, 2),
            (2, 4),
            (2, 2),
            (8, 16),
            (4, 16),
            (16, 16),
        ] {
            for fy in 0..8 {
                for fx in 0..8 {
                    for (x, y) in [
                        (-12, -12),
                        (-1, 3),
                        (0, 0),
                        (5, 5),
                        (16, 4),
                        (22, 18),
                        (30, 30),
                    ] {
                        chroma(&mut a, MAX, &p.view(), x, y, fx, fy, w, h);
                        chroma_reference(&mut b, &p.view(), x, y, fx, fy, w, h);
                        for r in 0..h {
                            assert_eq!(
                                a[r * MAX..r * MAX + w],
                                b[r * MAX..r * MAX + w],
                                "{w}x{h} ({fx},{fy}) at ({x},{y})"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The 32-bit kernels on 16-bit planes holding 8-bit values give the
    /// 8-bit results, and keep values above 255 at 10 bits.
    #[test]
    fn wide_kernels_match_8_bit_ones() {
        let mut p = Plane::new(48, 40);
        let mut q = Plane::with_depth(48, 40, true);
        let mut seed = 3u32;
        for (a, b) in p.data.iter_mut().zip(q.data16.iter_mut()) {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            *a = (seed >> 16) as u8;
            *b = *a as u16;
        }
        let (mut a, mut b) = ([0u8; MAX * MAX], [0u16; MAX * MAX]);
        for (w, h) in [(16, 16), (8, 4), (4, 4)] {
            for (x, y) in [(-20, -20), (-3, 5), (16, 12), (45, 38)] {
                for fy in 0..4 {
                    for fx in 0..4 {
                        luma(&mut a, MAX, &p.view(), x, y, fx, fy, w, h);
                        luma_t(&mut b, MAX, &q.view(), x, y, fx, fy, w, h, 255, false);
                        for r in 0..h {
                            for c in 0..w {
                                assert_eq!(
                                    a[r * MAX + c] as u16,
                                    b[r * MAX + c],
                                    "luma ({fx},{fy})"
                                );
                            }
                        }
                    }
                }
                for (fx, fy) in [(0, 0), (3, 5), (7, 7)] {
                    chroma(&mut a, MAX, &p.view(), x, y, fx, fy, w, h);
                    chroma_t(&mut b, MAX, &q.view(), x, y, fx, fy, w, h, false);
                    for r in 0..h {
                        for c in 0..w {
                            assert_eq!(a[r * MAX + c] as u16, b[r * MAX + c], "chroma");
                        }
                    }
                }
            }
        }
        // 10-bit: a flat plane of 1000 stays 1000 at every fraction.
        let mut q = Plane::with_depth(32, 32, true);
        q.data16.fill(1000);
        for fy in 0..4 {
            for fx in 0..4 {
                luma_t(&mut b, MAX, &q.view(), 4, 4, fx, fy, 8, 8, 1023, false);
                assert!(b[..8].iter().all(|&v| v == 1000));
            }
        }
    }

    #[test]
    fn chroma_pairs_match_the_reference() {
        let mut planes = [Plane::new(24, 20), Plane::new(24, 20)];
        let mut seed = 11u32;
        for v in planes.iter_mut().flat_map(|p| p.data.iter_mut()) {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            *v = (seed >> 16) as u8;
        }
        let views = [planes[0].view(), planes[1].view()];
        let (mut a, mut b) = ([[0u8; MAX * MAX]; 2], [0u8; MAX * MAX]);
        for (w, h) in [(8, 8), (8, 4), (8, 16), (4, 8), (4, 4), (4, 2), (2, 2)] {
            for fy in 0..8 {
                for fx in 0..8 {
                    for (x, y) in [(-1, 3), (0, 0), (5, 5), (15, 11), (16, 4), (22, 18)] {
                        let [a0, a1] = &mut a;
                        chroma_pair([a0, a1], MAX, [&views[0], &views[1]], x, y, fx, fy, w, h);
                        for (out, view) in a.iter().zip(&views) {
                            chroma_reference(&mut b, view, x, y, fx, fy, w, h);
                            for r in 0..h {
                                assert_eq!(
                                    out[r * MAX..r * MAX + w],
                                    b[r * MAX..r * MAX + w],
                                    "{w}x{h} ({fx},{fy}) at ({x},{y})"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn weighted_prediction_matches_the_formulas() {
        let mut seed = 5u32;
        let mut rand = move |n: u32| {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            (seed >> 8) % n
        };
        let mut bufs = [[0u8; MAX * MAX]; 2];
        for _ in 0..4000 {
            for v in bufs.iter_mut().flatten() {
                *v = rand(256) as u8;
            }
            let log_wd = rand(8);
            let wt = |r: &mut dyn FnMut(u32) -> u32| Weight {
                log_wd,
                w: r(256) as i32 - 128,
                o: r(256) as i32 - 128,
            };
            let (w0, w1) = (wt(&mut rand), wt(&mut rand));
            let (w, h) = (
                [4, 8, 16][rand(3) as usize],
                [2, 4, 8, 16][rand(4) as usize],
            );
            let clip = |v: i32| v.clamp(0, 255) as u8;
            for bi in [false, true] {
                let mut out = [0u8; MAX * MAX];
                let [a, b] = &bufs;
                if bi {
                    combine(&mut out, MAX, Some(a), Some(b), Some((w0, w1)), w, h);
                } else {
                    combine(&mut out, MAX, Some(a), None, Some((w0, w1)), w, h);
                }
                for r in 0..h {
                    for c in 0..w {
                        let (x, y) = (a[r * MAX + c] as i32, b[r * MAX + c] as i32);
                        let want = if bi {
                            clip(
                                ((x * w0.w + y * w1.w + (1 << log_wd)) >> (log_wd + 1))
                                    + ((w0.o + w1.o + 1) >> 1),
                            )
                        } else if log_wd >= 1 {
                            clip(((x * w0.w + (1 << (log_wd - 1))) >> log_wd) + w0.o)
                        } else {
                            clip(x * w0.w + w0.o)
                        };
                        assert_eq!(out[r * MAX + c], want, "bi {bi} {w}x{h} {w0:?} {w1:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn chroma_bilinear_midpoint() {
        let p = ramp();
        let mut d = [0u8; MAX * MAX];
        chroma(&mut d, MAX, &p.view(), 4, 4, 4, 0, 2, 2);
        assert_eq!(d[0], 22); // between 20 and 24
    }
}
