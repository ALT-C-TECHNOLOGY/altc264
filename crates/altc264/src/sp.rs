//! Transform coefficient decoding for P macroblocks in SP slices and SI
//! macroblocks (8.6): the prediction is transformed, combined with the
//! residual levels and requantised with QS, so the picture is rebuilt from
//! coefficients alone. Blocks are 4x4 in raster order.

/// normAdjust4x4 (8-315) by qP % 6 and position class.
const NORM: [[i32; 3]; 6] = [
    [10, 16, 13],
    [11, 18, 14],
    [13, 20, 16],
    [14, 23, 18],
    [16, 25, 20],
    [18, 29, 23],
];

/// w of LevelScale2 (8-419).
const QUANT: [[i32; 3]; 6] = [
    [13107, 5243, 8066],
    [11916, 4660, 7490],
    [10082, 4194, 6554],
    [9362, 3647, 5825],
    [8192, 3355, 5243],
    [7282, 2893, 4559],
];

/// A (8-417).
const A: [i32; 3] = [16, 25, 20];

/// Position class of raster index k: (even, even), (odd, odd), mixed.
#[inline]
fn class(k: usize) -> usize {
    match (k / 4 % 2, k % 4 % 2) {
        (0, 0) => 0,
        (1, 1) => 1,
        _ => 2,
    }
}

/// The forward core transform of 8-415.
pub fn forward4x4(p: &[i32; 16]) -> [i32; 16] {
    let mut t = [0i32; 16];
    for i in 0..4 {
        let r = &p[4 * i..4 * i + 4];
        let (s03, d03, s12, d12) = (r[0] + r[3], r[0] - r[3], r[1] + r[2], r[1] - r[2]);
        t[4 * i] = s03 + s12;
        t[4 * i + 1] = 2 * d03 + d12;
        t[4 * i + 2] = s03 - s12;
        t[4 * i + 3] = d03 - 2 * d12;
    }
    let mut c = [0i32; 16];
    for j in 0..4 {
        let (s03, d03) = (t[j] + t[12 + j], t[j] - t[12 + j]);
        let (s12, d12) = (t[4 + j] + t[8 + j], t[4 + j] - t[8 + j]);
        c[j] = s03 + s12;
        c[4 + j] = 2 * d03 + d12;
        c[8 + j] = s03 - s12;
        c[12 + j] = d03 - 2 * d12;
    }
    c
}

/// Quantises `v` with QS: Sign(v) * ((Abs(v) * LevelScale2 + round) >> bits).
#[inline]
fn quant(v: i32, scale: i32, bits: i32) -> i32 {
    let q = ((v.unsigned_abs() as i64 * scale as i64 + (1i64 << (bits - 1))) >> bits) as i32;
    if v < 0 { -q } else { q }
}

/// (level * scale) << shl >> shr, saturated so damaged streams cannot overflow.
#[inline]
fn scaled(level: i32, scale: i32, shl: i32, shr: i32) -> i32 {
    ((level as i64 * scale as i64) << shl >> shr).clamp(-(1 << 28), 1 << 28) as i32
}

/// One 4x4 block: the levels to dequantise with QS (8.6.1.1 or 8.6.2.1).
/// With `skip_dc`, position 0 is left for the chroma DC process.
pub fn block(
    pred: &[i32; 16],
    levels: &[i32; 16],
    qp: i32,
    qs: i32,
    switching: bool,
    skip_dc: bool,
) -> [i32; 16] {
    let cp = forward4x4(pred);
    let (pm, pp) = ((qp % 6) as usize, qp / 6);
    let (sm, bits) = ((qs % 6) as usize, 15 + qs / 6);
    let mut c = [0i32; 16];
    for k in skip_dc as usize..16 {
        let cl = class(k);
        c[k] = if switching {
            quant(cp[k], QUANT[sm][cl], bits).saturating_add(levels[k])
        } else {
            let s = cp[k] + scaled(levels[k], NORM[pm][cl] * A[cl], pp, 6);
            quant(s, QUANT[sm][cl], bits)
        };
    }
    c
}

/// The chroma DC of the four 4x4 blocks of a 4:2:0 component (8.6.1.2,
/// 8.6.2.2): `pred_dc` holds the DC coefficients of the transformed
/// prediction blocks. Returns c00 of each block, already scaled.
pub fn chroma_dc(
    pred_dc: [i32; 4],
    levels: &[i32; 4],
    qpc: i32,
    qsc: i32,
    switching: bool,
) -> [i32; 4] {
    let [b0, b1, b2, b3] = pred_dc;
    // The 2x2 transform of 8-427, entry k = 2 * j + i holding dc[i][j].
    let mut m = [
        b0 + b1 + b2 + b3,
        b0 + b1 - b2 - b3,
        b0 - b1 + b2 - b3,
        b0 - b1 - b2 + b3,
    ];
    let (pm, pp) = ((qpc % 6) as usize, qpc / 6);
    let (sm, sp) = ((qsc % 6) as usize, qsc / 6);
    let bits = 16 + sp;
    // ChromaDCLevel[j * 2 + i] goes with dc[i][j] (8-428, 8-440).
    let levels = [levels[0], levels[2], levels[1], levels[3]];
    for k in 0..4 {
        let q = if switching {
            quant(m[k], QUANT[sm][0], bits).saturating_add(levels[k])
        } else {
            let s = m[k] + scaled(levels[k], NORM[pm][0] * A[0], pp, 5);
            quant(s, QUANT[sm][0], bits)
        };
        m[k] = scaled(q, NORM[sm][0], sp, 0);
    }
    [
        (m[0] + m[1] + m[2] + m[3]) >> 1,
        (m[0] + m[1] - m[2] - m[3]) >> 1,
        (m[0] - m[1] + m[2] - m[3]) >> 1,
        (m[0] - m[1] - m[2] + m[3]) >> 1,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_of_flat_block_is_dc() {
        let c = forward4x4(&[7; 16]);
        assert_eq!(c[0], 7 * 16);
        assert!(c[1..].iter().all(|&v| v == 0));
    }

    #[test]
    fn switching_without_residual_reproduces_flat_prediction() {
        // A flat block survives quantisation at a fine QS.
        let c = block(&[100; 16], &[0; 16], 28, 10, true, false);
        let mut coef = c;
        let ls: [i32; 16] = std::array::from_fn(|k| 16 * NORM[10 % 6][class(k)]);
        crate::transform::scale_4x4(&mut coef, &ls, 10, false);
        assert!((coef[0] + 32) >> 6 >= 95);
    }
}
