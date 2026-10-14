//! Inverse scanning, scaling and transforms (8.5.6 to 8.5.13).
//!
//! Coefficient arrays are in raster order: index `i * N + j` holds c_ij with
//! i the row and j the column, as in the spec's equations.
#![allow(clippy::needless_range_loop)]

use crate::picture::Pixel;
use crate::ps::ScalingMatrices;
use crate::simd::{I16x8, transpose8};

/// 4x4 zig-zag and field scans (Table 8-13): scan index -> raster index.
pub const ZIGZAG_4X4: [u8; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];
pub const FIELD_4X4: [u8; 16] = [0, 4, 1, 8, 12, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15];

/// 8x8 scans (Table 8-14), as (row, column) -> raster index.
const fn raster8(t: [(u8, u8); 64]) -> [u8; 64] {
    let mut out = [0u8; 64];
    let mut k = 0;
    while k < 64 {
        out[k] = t[k].0 * 8 + t[k].1;
        k += 1;
    }
    out
}

#[rustfmt::skip]
pub const ZIGZAG_8X8: [u8; 64] = raster8([
    (0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),(2,1),(3,0),(4,0),(3,1),(2,2),(1,3),(0,4),(0,5),
    (1,4),(2,3),(3,2),(4,1),(5,0),(6,0),(5,1),(4,2),(3,3),(2,4),(1,5),(0,6),(0,7),(1,6),(2,5),(3,4),
    (4,3),(5,2),(6,1),(7,0),(7,1),(6,2),(5,3),(4,4),(3,5),(2,6),(1,7),(2,7),(3,6),(4,5),(5,4),(6,3),
    (7,2),(7,3),(6,4),(5,5),(4,6),(3,7),(4,7),(5,6),(6,5),(7,4),(7,5),(6,6),(5,7),(6,7),(7,6),(7,7),
]);

#[rustfmt::skip]
pub const FIELD_8X8: [u8; 64] = raster8([
    (0,0),(1,0),(2,0),(0,1),(1,1),(3,0),(4,0),(2,1),(0,2),(3,1),(5,0),(6,0),(7,0),(4,1),(1,2),(0,3),
    (2,2),(5,1),(6,1),(7,1),(3,2),(1,3),(0,4),(2,3),(4,2),(5,2),(6,2),(7,2),(3,3),(1,4),(0,5),(2,4),
    (4,3),(5,3),(6,3),(7,3),(3,4),(1,5),(0,6),(2,5),(4,4),(5,4),(6,4),(7,4),(3,5),(1,6),(2,6),(4,5),
    (5,5),(6,5),(7,5),(3,6),(0,7),(1,7),(4,6),(5,6),(6,6),(7,6),(2,7),(3,7),(4,7),(5,7),(6,7),(7,7),
]);

/// `scan` with raster positions transposed, (x, y) to (y, x), for N x N
/// blocks: where Dequant puts 8-bit coefficients (see add4x4_dq).
const fn transposed<const S: usize>(scan: [u8; S], n: u8) -> [u8; S] {
    let mut t = scan;
    let mut i = 0;
    while i < S {
        t[i] = scan[i] % n * n + scan[i] / n;
        i += 1;
    }
    t
}

pub const ZIGZAG_4X4_T: [u8; 16] = transposed(ZIGZAG_4X4, 4);
pub const FIELD_4X4_T: [u8; 16] = transposed(FIELD_4X4, 4);
pub const ZIGZAG_8X8_T: [u8; 64] = transposed(ZIGZAG_8X8, 8);
pub const FIELD_8X8_T: [u8; 64] = transposed(FIELD_8X8, 8);

/// v of equation 8-315.
const V4X4: [[i32; 3]; 6] = [
    [10, 16, 13],
    [11, 18, 14],
    [13, 20, 16],
    [14, 23, 18],
    [16, 25, 20],
    [18, 29, 23],
];
/// v of equation 8-318.
const V8X8: [[i32; 6]; 6] = [
    [20, 18, 32, 19, 25, 24],
    [22, 19, 35, 21, 28, 26],
    [26, 23, 42, 24, 33, 31],
    [28, 25, 45, 26, 35, 33],
    [32, 28, 51, 30, 40, 38],
    [36, 32, 58, 34, 46, 43],
];

fn norm_adjust_4x4(m: usize, i: usize, j: usize) -> i32 {
    match (i % 2, j % 2) {
        (0, 0) => V4X4[m][0],
        (1, 1) => V4X4[m][1],
        _ => V4X4[m][2],
    }
}

fn norm_adjust_8x8(m: usize, i: usize, j: usize) -> i32 {
    let k = if i % 4 == 0 && j % 4 == 0 {
        0
    } else if i % 2 == 1 && j % 2 == 1 {
        1
    } else if i % 4 == 2 && j % 4 == 2 {
        2
    } else if (i % 4 == 0 && j % 2 == 1) || (i % 2 == 1 && j % 4 == 0) {
        3
    } else if (i % 4 == 0 && j % 4 == 2) || (i % 4 == 2 && j % 4 == 0) {
        4
    } else {
        5
    };
    V8X8[m][k]
}

/// LevelScale4x4 and LevelScale8x8 (8.5.9) for every list and qP % 6, in
/// raster order. Lists: 4x4 [Y, Cb, Cr intra, Y, Cb, Cr inter]; 8x8
/// [Y intra, Y inter, Cb intra, Cb inter, Cr intra, Cr inter].
pub struct LevelScale {
    pub s4: [[[i32; 16]; 6]; 6],
    pub s8: [[[i32; 64]; 6]; 6],
}

impl LevelScale {
    pub fn new(m: &ScalingMatrices) -> Box<Self> {
        let mut ls = Box::new(Self {
            s4: [[[0; 16]; 6]; 6],
            s8: [[[0; 64]; 6]; 6],
        });
        for list in 0..6 {
            // The scaling lists are in zig-zag order (8.5.6, 8.5.7, frame scan).
            let mut w4 = [0i32; 16];
            for (k, &r) in ZIGZAG_4X4.iter().enumerate() {
                w4[r as usize] = m.list4x4[list][k] as i32;
            }
            let mut w8 = [0i32; 64];
            for (k, &r) in ZIGZAG_8X8.iter().enumerate() {
                w8[r as usize] = m.list8x8[list][k] as i32;
            }
            for q in 0..6 {
                for r in 0..16 {
                    ls.s4[list][q][r] = w4[r] * norm_adjust_4x4(q, r / 4, r % 4);
                }
                for r in 0..64 {
                    ls.s8[list][q][r] = w8[r] * norm_adjust_8x8(q, r / 8, r % 8);
                }
            }
        }
        ls
    }
}

/// Scales a 4x4 block in place (8.5.12.1). With `skip_dc` the DC value is
/// left as is (Intra_16x16 luma and chroma, whose DC comes from 8.5.10/11).
#[inline]
pub fn scale_4x4(c: &mut [i32; 16], ls: &[i32; 16], qp: i32, skip_dc: bool) {
    let q6 = (qp / 6) as u32;
    let start = skip_dc as usize;
    if qp >= 24 {
        let sh = q6 - 4;
        for k in start..16 {
            c[k] = (c[k] * ls[k]) << sh;
        }
    } else {
        let sh = 4 - q6;
        let round = 1 << (3 - q6);
        for k in start..16 {
            c[k] = (c[k] * ls[k] + round) >> sh;
        }
    }
}

/// Scales an 8x8 block in place (8.5.13.1).
#[inline]
pub fn scale_8x8(c: &mut [i32; 64], ls: &[i32; 64], qp: i32) {
    let q6 = (qp / 6) as u32;
    if qp >= 36 {
        let sh = q6 - 6;
        for k in 0..64 {
            c[k] = (c[k] * ls[k]) << sh;
        }
    } else {
        let sh = 6 - q6;
        let round = 1 << (5 - q6);
        for k in 0..64 {
            c[k] = (c[k] * ls[k] + round) >> sh;
        }
    }
}

/// Intra_16x16 luma DC: inverse Hadamard and scaling (8.5.10). `c` is the
/// 4x4 DC array in raster order; `ls00` is LevelScale4x4(qP % 6, 0, 0).
pub fn luma_dc(c: &mut [i32; 16], ls00: i32, qp: i32) {
    let mut f = [0i32; 16];
    // f = H c H with H rows [1 1 1 1], [1 1 -1 -1], [1 -1 -1 1], [1 -1 1 -1]
    // (equation 8-320): rows, then columns.
    let h4 = |r0: i32, r1: i32, r2: i32, r3: i32| {
        let (a, b) = (r0 + r1, r0 - r1);
        let (cc, d) = (r2 + r3, r2 - r3);
        [a + cc, a - cc, b - d, b + d]
    };
    for i in 0..4 {
        let o = h4(c[i * 4], c[i * 4 + 1], c[i * 4 + 2], c[i * 4 + 3]);
        f[i * 4..i * 4 + 4].copy_from_slice(&o);
    }
    for j in 0..4 {
        let o = h4(f[j], f[4 + j], f[8 + j], f[12 + j]);
        for (i, v) in o.into_iter().enumerate() {
            f[i * 4 + j] = v;
        }
    }
    let q6 = (qp / 6) as u32;
    for k in 0..16 {
        c[k] = if qp >= 36 {
            (f[k] * ls00) << (q6 - 6)
        } else {
            (f[k] * ls00 + (1 << (5 - q6))) >> (6 - q6)
        };
    }
}

/// 4:2:0 chroma DC: 2x2 transform and scaling (8.5.11, equation 8-326).
pub fn chroma_dc_420(c: &mut [i32; 4], ls00: i32, qp: i32) {
    let (a, b, cc, d) = (c[0], c[1], c[2], c[3]);
    let f = [
        a + b + cc + d,
        a - b + cc - d,
        a + b - cc - d,
        a - b - cc + d,
    ];
    for k in 0..4 {
        c[k] = ((f[k] * ls00) << (qp / 6)) >> 5;
    }
}

/// 4:2:2 chroma DC: 2x4 transform and scaling (8.5.11, equations 8-325,
/// 8-327 to 8-329). `c` is 4 rows by 2 columns in raster order; `ls` gives
/// LevelScale4x4(m, 0, 0) for m = 0..5.
pub fn chroma_dc_422(c: &mut [i32; 8], ls: impl Fn(usize) -> i32, qp: i32) {
    let mut t = [0i32; 8];
    // Left: 4x4 Hadamard over rows (columns of c). Right: 2x2 over columns.
    for j in 0..2 {
        let (c0, c1, c2, c3) = (c[j], c[2 + j], c[4 + j], c[6 + j]);
        t[j] = c0 + c1 + c2 + c3;
        t[2 + j] = c0 + c1 - c2 - c3;
        t[4 + j] = c0 - c1 - c2 + c3;
        t[6 + j] = c0 - c1 + c2 - c3;
    }
    let qp_dc = qp + 3;
    let s = ls((qp_dc % 6) as usize);
    for i in 0..4 {
        let (a, b) = (t[i * 2], t[i * 2 + 1]);
        for (k, f) in [a + b, a - b].into_iter().enumerate() {
            c[i * 2 + k] = if qp_dc >= 36 {
                (f * s) << (qp_dc / 6 - 6)
            } else {
                (f * s + (1 << (5 - qp_dc / 6))) >> (6 - qp_dc / 6)
            };
        }
    }
}

/// 4x4 inverse transform (8.5.12.2) added to the prediction in `dst`.
/// The spec bounds every intermediate value of conforming streams to 16
/// bits, so 16-bit vectors are exact: the row pass is scalar, the column
/// pass works on row pairs.
pub fn idct4x4_add(dst: &mut [u8], stride: usize, d: &[i32; 16]) {
    idct4x4_add_g::<false>(dst, stride, d, None)
}

/// [`idct4x4_add`] of a transposed block (as Dequant stores 8-bit blocks).
pub fn idct4x4_add_t(dst: &mut [u8], stride: usize, d: &[i32; 16]) {
    idct4x4_add_g::<true>(dst, stride, d, None)
}

/// Two horizontally adjacent 4x4 blocks, `a` left of `b`, both stored
/// transposed (as [`idct4x4_add_t`]), with DCs given apart when Some (as
/// [`idct4x4_add_t_dc`]): inverse transformed and added to the 8x4 samples
/// at `dst`, each vector holding a row (or column) of both blocks.
pub fn idct4x4x2_add_t(
    dst: &mut [u8],
    stride: usize,
    a: &[i32; 16],
    b: &[i32; 16],
    dc: Option<[i32; 2]>,
) {
    // [x0 | x1] and [x2 | x3] of each block, then x_i of both: the column
    // i of c (stored transposed), the lanes its rows.
    let a01 = I16x8::from_i32(a[..8].try_into().unwrap());
    let a23 = I16x8::from_i32(a[8..].try_into().unwrap());
    let b01 = I16x8::from_i32(b[..8].try_into().unwrap());
    let b23 = I16x8::from_i32(b[8..].try_into().unwrap());
    let mut x = [
        a01.combine_lo(b01),
        a01.combine_hi(b01),
        a23.combine_lo(b23),
        a23.combine_hi(b23),
    ];
    if let Some([da, db]) = dc {
        x[0] = x[0].with_lane0(da as i16).with_lane4(db as i16);
    }
    // The 1-D transform (8-338 to 8-345, 8-347 to 8-354), lane by lane.
    let pass = |x: [I16x8; 4]| {
        let (e0, e1) = (x[0].add(x[2]), x[0].sub(x[2]));
        let (e2, e3) = (x[1].shr::<1>().sub(x[3]), x[1].add(x[3].shr::<1>()));
        [e0.add(e3), e1.add(e2), e1.sub(e2), e0.sub(e3)]
    };
    // Rows first: y_i holds column i of f, lanes its rows; transposed in
    // each block's half, z_r holds row r of f (of both blocks).
    let y = pass(x);
    let (l01, h01) = (y[0].zip_lo(y[1]), y[0].zip_hi(y[1]));
    let (l23, h23) = (y[2].zip_lo(y[3]), y[2].zip_hi(y[3]));
    // Rows 0, 1 and 2, 3 of a, and of b.
    let (a01, a23) = (l01.zip_lo32(l23), l01.zip_hi32(l23));
    let (b01, b23) = (h01.zip_lo32(h23), h01.zip_hi32(h23));
    let z = [
        a01.combine_lo(b01),
        a01.combine_hi(b01),
        a23.combine_lo(b23),
        a23.combine_hi(b23),
    ];
    // Then the columns: w_k is row k of the residual.
    let w = pass(z);
    assert!(dst.len() >= 3 * stride + 8);
    for (k, v) in w.into_iter().enumerate() {
        let row: &mut [u8; 8] = (&mut dst[k * stride..k * stride + 8]).try_into().unwrap();
        v.rshr::<6>().add(I16x8::load_u8(row)).store_u8(row);
    }
}

/// [`idct4x4_add_t`] with coefficient 0 (zero in `d`) given apart: the DC
/// of an AC block, inserted after the load rather than stored before it.
pub fn idct4x4_add_t_dc(dst: &mut [u8], stride: usize, d: &[i32; 16], dc: i32) {
    idct4x4_add_g::<true>(dst, stride, d, Some(dc))
}

#[inline(always)]
fn idct4x4_add_g<const TRANSPOSED: bool>(
    dst: &mut [u8],
    stride: usize,
    d: &[i32; 16],
    dc: Option<i32>,
) {
    // Every intermediate value of a conforming stream fits in 16 bits
    // (8.5.12.1), so both passes work on 8 x i16 vectors holding two rows
    // or columns, [x0 | x1] and [x2 | x3].
    // Transposes the 4x4 matrix of rows [r0 | r1], [r2 | r3] to columns.
    let transpose = |a: I16x8, b: I16x8| {
        let (t0, t1) = (a.zip_lo(b), a.zip_hi(b));
        (t0.zip_lo(t1), t0.zip_hi(t1))
    };
    // The 1-D transform of x0..x3, lane by lane: [y0 | y1], [y3 | y2].
    let pass = |a: I16x8, b: I16x8| {
        // e0 = x0 + x2, e1 = x0 - x2, e2 = (x1 >> 1) - x3, e3 = x1 + (x3 >> 1).
        let e01 = a.add(b).combine_lo(a.sub(b));
        let e32 = a.add(b.shr::<1>()).combine_hi(a.shr::<1>().sub(b));
        (e01.add(e32), e01.sub(e32))
    };
    let mut rows = (
        I16x8::from_i32(d[..8].try_into().unwrap()),
        I16x8::from_i32(d[8..].try_into().unwrap()),
    );
    if let Some(dc) = dc {
        rows.0 = rows.0.with_lane0(dc as i16);
    }
    // Rows first (8-338 to 8-345): on the columns of d (the rows of a
    // transposed d).
    let (c01, c23) = if TRANSPOSED {
        rows
    } else {
        transpose(rows.0, rows.1)
    };
    let (f01, f32) = pass(c01, c23);
    // [f2 | f3] from [f3 | f2]; then the rows of f, for the columns.
    let f23 = f32.combine_hi(f32).combine_lo(f32);
    let (a, b) = transpose(f01, f23);
    let (h01, h32) = pass(a, b);
    let (h01, h32) = (h01.rshr::<6>(), h32.rshr::<6>()); // [h0 | h1], [h3 | h2]
    assert!(dst.len() >= 3 * stride + 4);
    let at = dst.as_mut_ptr();
    for (h, rows) in [(h01, [0, 1]), (h32, [3, 2])] {
        // SAFETY: rows 0..4 of 4 samples are within dst (asserted above).
        unsafe {
            let (r0, r1) = (at.add(rows[0] * stride), at.add(rows[1] * stride));
            let mut p = [0u8; 8];
            std::ptr::copy_nonoverlapping(r0, p.as_mut_ptr(), 4);
            std::ptr::copy_nonoverlapping(r1, p.as_mut_ptr().add(4), 4);
            h.add(I16x8::load_u8(&p)).store_u8(&mut p);
            std::ptr::copy_nonoverlapping(p.as_ptr(), r0, 4);
            std::ptr::copy_nonoverlapping(p.as_ptr().add(4), r1, 4);
        }
    }
}

/// One 8-point inverse transform pass (equations 8-358 to 8-381).
fn idct8_1d(d: [i32; 8]) -> [i32; 8] {
    let e0 = d[0] + d[4];
    let e1 = -d[3] + d[5] - d[7] - (d[7] >> 1);
    let e2 = d[0] - d[4];
    let e3 = d[1] + d[7] - d[3] - (d[3] >> 1);
    let e4 = (d[2] >> 1) - d[6];
    let e5 = -d[1] + d[7] + d[5] + (d[5] >> 1);
    let e6 = d[2] + (d[6] >> 1);
    let e7 = d[3] + d[5] + d[1] + (d[1] >> 1);
    let f0 = e0 + e6;
    let f1 = e1 + (e7 >> 2);
    let f2 = e2 + e4;
    let f3 = e3 + (e5 >> 2);
    let f4 = e2 - e4;
    let f5 = (e3 >> 2) - e5;
    let f6 = e0 - e6;
    let f7 = e7 - (e1 >> 2);
    [
        f0 + f7,
        f2 + f5,
        f4 + f3,
        f6 + f1,
        f6 - f1,
        f4 - f3,
        f2 - f5,
        f0 - f7,
    ]
}

/// idct8_1d on vectors: element k of every lane's 8-point input.
#[inline(always)]
fn idct8_v(d: [I16x8; 8]) -> [I16x8; 8] {
    let e0 = d[0].add(d[4]);
    let e1 = d[5].sub(d[3]).sub(d[7]).sub(d[7].shr::<1>());
    let e2 = d[0].sub(d[4]);
    let e3 = d[1].add(d[7]).sub(d[3]).sub(d[3].shr::<1>());
    let e4 = d[2].shr::<1>().sub(d[6]);
    let e5 = d[7].sub(d[1]).add(d[5]).add(d[5].shr::<1>());
    let e6 = d[2].add(d[6].shr::<1>());
    let e7 = d[3].add(d[5]).add(d[1]).add(d[1].shr::<1>());
    let f0 = e0.add(e6);
    let f1 = e1.add(e7.shr::<2>());
    let f2 = e2.add(e4);
    let f3 = e3.add(e5.shr::<2>());
    let f4 = e2.sub(e4);
    let f5 = e3.shr::<2>().sub(e5);
    let f6 = e0.sub(e6);
    let f7 = e7.sub(e1.shr::<2>());
    [
        f0.add(f7),
        f2.add(f5),
        f4.add(f3),
        f6.add(f1),
        f6.sub(f1),
        f4.sub(f3),
        f2.sub(f5),
        f0.sub(f7),
    ]
}

/// 8x8 inverse transform (8.5.13.2) added to the prediction in `dst`, in
/// 16-bit vectors (exact for conforming streams, as for 4x4): the columns
/// of the transposed block go through the row pass, then the transposed
/// result through the column pass.
pub fn idct8x8_add(dst: &mut [u8], stride: usize, d: &[i32; 64]) {
    idct8x8_add_g::<false>(dst, stride, d)
}

/// [`idct8x8_add`] of a transposed block (as Dequant stores 8-bit blocks).
pub fn idct8x8_add_t(dst: &mut [u8], stride: usize, d: &[i32; 64]) {
    idct8x8_add_g::<true>(dst, stride, d)
}

#[inline(always)]
fn idct8x8_add_g<const TRANSPOSED: bool>(dst: &mut [u8], stride: usize, d: &[i32; 64]) {
    let mut v: [I16x8; 8] =
        std::array::from_fn(|i| I16x8::from_i32(d[i * 8..i * 8 + 8].try_into().unwrap()));
    // The columns of d, for the row pass.
    if !TRANSPOSED {
        transpose8(&mut v);
    }
    let mut g = idct8_v(v);
    transpose8(&mut g);
    assert!(dst.len() >= 7 * stride + 8);
    for (i, h) in idct8_v(g).into_iter().enumerate() {
        // SAFETY: rows 0..8 of 8 samples are within dst (asserted above).
        let row = unsafe { &mut *dst.as_mut_ptr().add(i * stride).cast::<[u8; 8]>() };
        h.rshr::<6>().add(I16x8::load_u8(row)).store_u8(row);
    }
}

/// The nonzero levels of a transform block as decoded: their levelListIdx
/// (scan position within the block's list) and value.
pub struct Coeffs {
    pub n: usize,
    pub pos: [u8; 64],
    pub level: [i32; 64],
}

impl Default for Coeffs {
    fn default() -> Self {
        Self {
            n: 0,
            pos: [0; 64],
            level: [0; 64],
        }
    }
}

impl Coeffs {
    #[inline(always)]
    pub fn push(&mut self, pos: usize, level: i32) {
        self.pos[self.n] = pos as u8;
        self.level[self.n] = level;
        self.n += 1;
    }

    /// The levels at their levelListIdx in `list` (zeroed by the caller).
    pub fn to_list(&self, list: &mut [i32]) {
        for i in 0..self.n {
            list[self.pos[i] as usize] = self.level[i];
        }
    }
}

/// Scaling of one level of a 4x4 block (8-336, 8-337): `ls` is
/// LevelScale4x4(qP % 6, i, j).
#[inline(always)]
pub fn scale_level_4x4(c: i32, ls: i32, qp: i32) -> i32 {
    if qp >= 24 {
        (c * ls) << (qp / 6 - 4)
    } else {
        (c * ls + (1 << (3 - qp / 6))) >> (4 - qp / 6)
    }
}

/// Scaling of one level of an 8x8 block (8-368, 8-369).
#[inline(always)]
pub fn scale_level_8x8(c: i32, ls: i32, qp: i32) -> i32 {
    if qp >= 36 {
        (c * ls) << (qp / 6 - 6)
    } else {
        (c * ls + (1 << (5 - qp / 6))) >> (6 - qp / 6)
    }
}

/// 4x4 inverse transform added to `dst`, for any sample type: the SIMD
/// kernel for 8-bit samples, exact 32-bit arithmetic for higher bit depths
/// (whose intermediate values exceed 16 bits). `max` is (1 << BitDepth) - 1.
#[inline]
pub fn add4x4<T: Pixel>(dst: &mut [T], stride: usize, d: &[i32; 16], max: i32) {
    if T::HIGH {
        idct4x4_add_wide(dst, stride, d, max);
    } else {
        idct4x4_add(T::u8s_mut(dst), stride, d);
    }
}

/// [`add4x4`] of a block as Dequant stores it: transposed for 8-bit
/// samples (the vector transform then skips a transpose), in raster order
/// for higher bit depths. The block is zeroed once read, ready for the
/// next macroblock.
#[inline]
pub fn add4x4_dq<T: Pixel>(dst: &mut [T], stride: usize, d: &mut [i32; 16], max: i32) {
    if T::HIGH {
        idct4x4_add_wide(dst, stride, d, max);
    } else {
        idct4x4_add_t(T::u8s_mut(dst), stride, d);
    }
    *d = [0; 16];
}

/// Adds to the 8x4 samples at `dst` the residual of two horizontally
/// adjacent blocks with only DCs (scaled), `dc`: (dc + 32) >> 6 in each.
#[inline]
pub fn add_dc8x4(dst: &mut [u8], stride: usize, dc: [i32; 2]) {
    let [a, b] = dc.map(|v| ((v + 32) >> 6).clamp(-512, 512) as i16);
    let v = I16x8::from_array([a, a, a, a, b, b, b, b]);
    assert!(dst.len() >= 3 * stride + 8);
    for k in 0..4 {
        let row: &mut [u8; 8] = (&mut dst[k * stride..k * stride + 8]).try_into().unwrap();
        I16x8::load_u8(row).add(v).store_u8(row);
    }
}

/// [`add4x4_dq`] for two horizontally adjacent blocks at once (8-bit
/// samples), with DCs given apart when Some (see [`add4x4_dq_dc`]).
#[inline]
pub fn add4x4x2_dq(
    dst: &mut [u8],
    stride: usize,
    a: &mut [i32; 16],
    b: &mut [i32; 16],
    dc: Option<[i32; 2]>,
) {
    idct4x4x2_add_t(dst, stride, a, b, dc);
    *a = [0; 16];
    *b = [0; 16];
}

/// [`add4x4_dq`] of an AC block whose DC (coefficient 0, zero in `d`) is
/// given apart, so that it is not stored just before the block is loaded.
#[inline]
pub fn add4x4_dq_dc<T: Pixel>(dst: &mut [T], stride: usize, d: &mut [i32; 16], dc: i32, max: i32) {
    if T::HIGH {
        d[0] = dc;
        idct4x4_add_wide(dst, stride, d, max);
    } else {
        idct4x4_add_t_dc(T::u8s_mut(dst), stride, d, dc);
    }
    *d = [0; 16];
}

/// [`add8x8`] of a block as Dequant stores it (see [`add4x4_dq`]), zeroed
/// once read.
#[inline]
pub fn add8x8_dq<T: Pixel>(dst: &mut [T], stride: usize, d: &mut [i32; 64], max: i32) {
    if T::HIGH {
        idct8x8_add_wide(dst, stride, d, max);
    } else {
        idct8x8_add_t(T::u8s_mut(dst), stride, d);
    }
    *d = [0; 64];
}

/// As [`add4x4`] for the 8x8 transform.
#[inline]
pub fn add8x8<T: Pixel>(dst: &mut [T], stride: usize, d: &[i32; 64], max: i32) {
    if T::HIGH {
        idct8x8_add_wide(dst, stride, d, max);
    } else {
        idct8x8_add(T::u8s_mut(dst), stride, d);
    }
}

/// 4x4 inverse transform (8.5.12.2) in 32 bits, added to `dst`.
pub fn idct4x4_add_wide<T: Pixel>(dst: &mut [T], stride: usize, d: &[i32; 16], max: i32) {
    let mut f = [0i32; 16];
    for i in 0..4 {
        let r = &d[i * 4..i * 4 + 4];
        let e0 = r[0] + r[2];
        let e1 = r[0] - r[2];
        let e2 = (r[1] >> 1) - r[3];
        let e3 = r[1] + (r[3] >> 1);
        f[i * 4] = e0 + e3;
        f[i * 4 + 1] = e1 + e2;
        f[i * 4 + 2] = e1 - e2;
        f[i * 4 + 3] = e0 - e3;
    }
    for j in 0..4 {
        let g0 = f[j] + f[8 + j];
        let g1 = f[j] - f[8 + j];
        let g2 = (f[4 + j] >> 1) - f[12 + j];
        let g3 = f[4 + j] + (f[12 + j] >> 1);
        let h = [g0 + g3, g1 + g2, g1 - g2, g0 - g3];
        for (i, v) in h.into_iter().enumerate() {
            let p = &mut dst[i * stride + j];
            *p = T::clip(p.get() + ((v + 32) >> 6), max);
        }
    }
}

/// 8x8 inverse transform (8.5.13.2) in 32 bits, added to `dst`.
pub fn idct8x8_add_wide<T: Pixel>(dst: &mut [T], stride: usize, d: &[i32; 64], max: i32) {
    let mut g = [0i32; 64];
    for i in 0..8 {
        let row: [i32; 8] = d[i * 8..i * 8 + 8].try_into().unwrap();
        g[i * 8..i * 8 + 8].copy_from_slice(&idct8_1d(row));
    }
    for j in 0..8 {
        let col = std::array::from_fn(|i| g[i * 8 + j]);
        for (i, v) in idct8_1d(col).into_iter().enumerate() {
            let p = &mut dst[i * stride + j];
            *p = T::clip(p.get() + ((v + 32) >> 6), max);
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn two_blocks_match_one_at_a_time() {
        let mut x = 5u32;
        let mut rnd = |n: i32| {
            x = x.wrapping_mul(1_103_515_245).wrapping_add(12345);
            ((x >> 8) % (2 * n as u32 + 1)) as i32 - n
        };
        for round in 0..20_000 {
            // Small levels keep every intermediate in 16 bits, as in
            // conforming streams.
            let n = [4, 40, 400][round % 3];
            let a: [i32; 16] = std::array::from_fn(|_| if rnd(3) == 0 { rnd(n) } else { 0 });
            let b: [i32; 16] = std::array::from_fn(|_| if rnd(3) == 0 { rnd(n) } else { 0 });
            let px: Vec<u8> = (0..4 * 16).map(|_| (rnd(127) + 128) as u8).collect();
            let dc = (round % 2 == 1).then(|| [rnd(n), rnd(n)]);
            let mut want = px.clone();
            match dc {
                Some([da, db]) => {
                    idct4x4_add_t_dc(&mut want[..], 16, &a, da);
                    idct4x4_add_t_dc(&mut want[4..], 16, &b, db);
                }
                None => {
                    idct4x4_add_t(&mut want[..], 16, &a);
                    idct4x4_add_t(&mut want[4..], 16, &b);
                }
            }
            let mut got = px.clone();
            idct4x4x2_add_t(&mut got, 16, &a, &b, dc);
            assert_eq!(got, want, "round {round}");
        }
    }

    use super::*;

    // The wide implementations are the straightforward ones: they check
    // the fast ones.
    fn idct8x8_add_reference(dst: &mut [u8], stride: usize, d: &[i32; 64]) {
        idct8x8_add_wide(dst, stride, d, 255);
    }

    fn idct4x4_add_reference(dst: &mut [u8], stride: usize, d: &[i32; 16]) {
        idct4x4_add_wide(dst, stride, d, 255);
    }

    /// Whether every intermediate value of the 4x4 transform fits in 16
    /// bits, as conforming streams guarantee.
    fn fits_16_bits(d: &[i32; 16]) -> bool {
        let ok = |v: i32| (-32768..=32767).contains(&v);
        let mut f = [0i32; 16];
        for i in 0..4 {
            let r = &d[i * 4..i * 4 + 4];
            let e = [
                r[0] + r[2],
                r[0] - r[2],
                (r[1] >> 1) - r[3],
                r[1] + (r[3] >> 1),
            ];
            let v = [e[0] + e[3], e[1] + e[2], e[1] - e[2], e[0] - e[3]];
            if !e.iter().chain(&v).all(|&x| ok(x)) {
                return false;
            }
            f[i * 4..i * 4 + 4].copy_from_slice(&v);
        }
        (0..4).all(|j| {
            let c = [f[j], f[4 + j], f[8 + j], f[12 + j]];
            let g = [
                c[0] + c[2],
                c[0] - c[2],
                (c[1] >> 1) - c[3],
                c[1] + (c[3] >> 1),
            ];
            let h = [g[0] + g[3], g[1] + g[2], g[1] - g[2], g[0] - g[3]];
            g.iter().chain(&h).all(|&x| ok(x))
        })
    }

    #[test]
    fn idct8x8_matches_reference() {
        let mut seed = 5u32;
        let mut rnd = |n: u32| {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            seed >> 8 & (n - 1)
        };
        let mut tested = 0;
        for case in 0..20000 {
            let span = [8, 128, 1024, 2048][case % 4];
            let d: [i32; 64] = std::array::from_fn(|_| rnd(2 * span) as i32 - span as i32);
            // Skip blocks whose intermediate values a conforming stream
            // could not produce.
            let mut g = [0i32; 64];
            let ok = |v: &[i32]| v.iter().all(|&x| (-32768..=32767).contains(&x));
            for i in 0..8 {
                let row: [i32; 8] = d[i * 8..i * 8 + 8].try_into().unwrap();
                g[i * 8..i * 8 + 8].copy_from_slice(&idct8_1d(row));
            }
            let cols_ok = (0..8).all(|j| ok(&idct8_1d(std::array::from_fn(|i| g[i * 8 + j]))));
            if !ok(&g) || !cols_ok {
                continue;
            }
            tested += 1;
            let mut a = [0u8; 128];
            for v in a.iter_mut() {
                *v = rnd(256) as u8;
            }
            let mut b = a;
            idct8x8_add(&mut a, 16, &d);
            idct8x8_add_reference(&mut b, 16, &d);
            assert_eq!(a, b, "case {case}");
        }
        assert!(tested > 5000);
    }

    #[test]
    fn idct4x4_matches_reference() {
        let mut seed = 11u32;
        let mut rnd = |n: u32| {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            seed >> 8 & (n - 1)
        };
        for case in 0..20000 {
            // Coefficient magnitudes from tiny to the 16-bit limit.
            let span = [16, 256, 4096, 8192][case % 4];
            let d: [i32; 16] = std::array::from_fn(|_| rnd(2 * span) as i32 - span as i32);
            if !fits_16_bits(&d) {
                continue; // not a conforming block (8.5.12.2)
            }
            let mut a = [0u8; 64];
            for v in a.iter_mut() {
                *v = rnd(256) as u8;
            }
            let mut b = a;
            idct4x4_add(&mut a, 16, &d);
            idct4x4_add_reference(&mut b, 16, &d);
            assert_eq!(a, b, "case {case}");
        }
    }

    #[test]
    fn scans_are_permutations() {
        for s in [&ZIGZAG_8X8, &FIELD_8X8] {
            let mut seen = [false; 64];
            for &r in s.iter() {
                assert!(!seen[r as usize]);
                seen[r as usize] = true;
            }
        }
        assert_eq!(ZIGZAG_8X8[10], 32); // c40
        assert_eq!(FIELD_8X8[10], 40); // c50
    }

    #[test]
    fn flat_level_scale_matches_norm_adjust() {
        let ls = LevelScale::new(&ScalingMatrices::FLAT);
        assert_eq!(ls.s4[0][0][0], 16 * 10);
        assert_eq!(ls.s4[3][5][5], 16 * 29); // (1, 1)
        assert_eq!(ls.s4[0][2][1], 16 * 16); // (0, 1)
        assert_eq!(ls.s8[1][4][0], 16 * 32);
        assert_eq!(ls.s8[0][0][8 * 2 + 2], 16 * 32); // (2, 2)
        assert_eq!(ls.s8[0][0][1], 16 * 19); // (0, 1)
        assert_eq!(ls.s8[0][0][2], 16 * 25); // (0, 2)
        assert_eq!(ls.s8[0][0][3], 16 * 19); // (0, 3): i % 4 = 0, j % 2 = 1
        assert_eq!(ls.s8[0][0][8 + 2], 16 * 24); // (1, 2): otherwise
    }

    #[test]
    fn luma_dc_hadamard_matches_equation_8_320() {
        // A single coefficient at c01 gives f_ij = H_i0 * H_1j = row 1 of H.
        let mut c = [0i32; 16];
        c[1] = 1;
        // qP 36, LevelScale 64: dcY = f * 64 << 0, so the signs show directly.
        luma_dc(&mut c, 64, 36);
        assert_eq!(&c[..4], &[64, 64, -64, -64]);
        assert_eq!(&c[12..], &[64, 64, -64, -64]);
    }

    #[test]
    fn dc_only_4x4_adds_constant() {
        let mut dst = [100u8; 16];
        let mut d = [0i32; 16];
        d[0] = 64 * 5; // (h + 32) >> 6 = 5 everywhere
        idct4x4_add(&mut dst, 4, &d);
        assert!(dst.iter().all(|&v| v == 105));
        let mut dst = [10u8; 64];
        let mut d = [0i32; 64];
        d[0] = -64 * 20;
        idct8x8_add(&mut dst, 8, &d);
        assert!(dst.iter().all(|&v| v == 0));
    }
}
