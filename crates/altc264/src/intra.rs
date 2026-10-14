//! Intra prediction (8.3.1.2, 8.3.2.2, 8.3.3, 8.3.4), for any sample bit
//! depth `bd`.
#![allow(clippy::needless_range_loop)]
//!
//! Predictors write into `dst` (stride `stride`); neighbouring samples are
//! passed in explicitly with their availability.

use crate::picture::Pixel;

/// Neighbouring samples of a 4x4 block: `top[0..8]` is p[0..7, -1] (the
/// last four already substituted per 8.3.1.2 when unavailable), `left` is
/// p[-1, 0..3] and `corner` is p[-1, -1].
pub struct Edge4<T = u8> {
    pub top: [T; 8],
    pub left: [T; 4],
    pub corner: T,
    pub has_top: bool,
    pub has_left: bool,
    pub has_corner: bool,
}

#[inline]
fn put<T: Pixel>(dst: &mut [T], stride: usize, x: usize, y: usize, v: u32) {
    dst[y * stride + x] = T::new(v as i32);
}

/// Intra_4x4 prediction (8.3.1.2.1 to 8.3.1.2.9). Returns false when the
/// mode needs samples that are not available (a corrupt stream).
pub fn pred4x4<T: Pixel>(dst: &mut [T], stride: usize, mode: u8, e: &Edge4<T>, bd: u32) -> bool {
    let t = |x: usize| e.top[x].get() as u32;
    let l = |y: usize| e.left[y].get() as u32;
    let c = e.corner.get() as u32;
    // p[x, -1] for x = -1..7 and p[-1, y] for y = -1..3, with p[-1, -1] shared.
    let p_top = |x: i32| if x < 0 { c } else { t(x as usize) };
    let p_left = |y: i32| if y < 0 { c } else { l(y as usize) };
    match mode {
        0 => {
            if !e.has_top {
                return false;
            }
            for y in 0..4 {
                for x in 0..4 {
                    put(dst, stride, x, y, t(x));
                }
            }
        }
        1 => {
            if !e.has_left {
                return false;
            }
            for y in 0..4 {
                for x in 0..4 {
                    put(dst, stride, x, y, l(y));
                }
            }
        }
        2 => {
            let v = match (e.has_top, e.has_left) {
                (true, true) => ((0..4).map(t).sum::<u32>() + (0..4).map(l).sum::<u32>() + 4) >> 3,
                (false, true) => ((0..4).map(l).sum::<u32>() + 2) >> 2,
                (true, false) => ((0..4).map(t).sum::<u32>() + 2) >> 2,
                (false, false) => 1 << (bd - 1),
            };
            for y in 0..4 {
                for x in 0..4 {
                    put(dst, stride, x, y, v);
                }
            }
        }
        3 => {
            // Diagonal_Down_Left
            if !e.has_top {
                return false;
            }
            for y in 0..4 {
                for x in 0..4 {
                    let v = if x == 3 && y == 3 {
                        (t(6) + 3 * t(7) + 2) >> 2
                    } else {
                        (t(x + y) + 2 * t(x + y + 1) + t(x + y + 2) + 2) >> 2
                    };
                    put(dst, stride, x, y, v);
                }
            }
        }
        4 => {
            // Diagonal_Down_Right
            if !(e.has_top && e.has_left && e.has_corner) {
                return false;
            }
            for y in 0..4i32 {
                for x in 0..4i32 {
                    let v = if x > y {
                        (p_top(x - y - 2) + 2 * p_top(x - y - 1) + p_top(x - y) + 2) >> 2
                    } else if x < y {
                        (p_left(y - x - 2) + 2 * p_left(y - x - 1) + p_left(y - x) + 2) >> 2
                    } else {
                        (p_top(0) + 2 * c + p_left(0) + 2) >> 2
                    };
                    put(dst, stride, x as usize, y as usize, v);
                }
            }
        }
        5 => {
            // Vertical_Right
            if !(e.has_top && e.has_left && e.has_corner) {
                return false;
            }
            for y in 0..4i32 {
                for x in 0..4i32 {
                    let z = 2 * x - y;
                    let v = if z >= 0 && z % 2 == 0 {
                        (p_top(x - (y >> 1) - 1) + p_top(x - (y >> 1)) + 1) >> 1
                    } else if z >= 0 {
                        (p_top(x - (y >> 1) - 2)
                            + 2 * p_top(x - (y >> 1) - 1)
                            + p_top(x - (y >> 1))
                            + 2)
                            >> 2
                    } else if z == -1 {
                        (p_left(0) + 2 * c + p_top(0) + 2) >> 2
                    } else {
                        (p_left(y - 1) + 2 * p_left(y - 2) + p_left(y - 3) + 2) >> 2
                    };
                    put(dst, stride, x as usize, y as usize, v);
                }
            }
        }
        6 => {
            // Horizontal_Down
            if !(e.has_top && e.has_left && e.has_corner) {
                return false;
            }
            for y in 0..4i32 {
                for x in 0..4i32 {
                    let z = 2 * y - x;
                    let v = if z >= 0 && z % 2 == 0 {
                        (p_left(y - (x >> 1) - 1) + p_left(y - (x >> 1)) + 1) >> 1
                    } else if z >= 0 {
                        (p_left(y - (x >> 1) - 2)
                            + 2 * p_left(y - (x >> 1) - 1)
                            + p_left(y - (x >> 1))
                            + 2)
                            >> 2
                    } else if z == -1 {
                        (p_left(0) + 2 * c + p_top(0) + 2) >> 2
                    } else {
                        (p_top(x - 1) + 2 * p_top(x - 2) + p_top(x - 3) + 2) >> 2
                    };
                    put(dst, stride, x as usize, y as usize, v);
                }
            }
        }
        7 => {
            // Vertical_Left
            if !e.has_top {
                return false;
            }
            for y in 0..4 {
                for x in 0..4 {
                    let k = x + (y >> 1);
                    let v = if y % 2 == 0 {
                        (t(k) + t(k + 1) + 1) >> 1
                    } else {
                        (t(k) + 2 * t(k + 1) + t(k + 2) + 2) >> 2
                    };
                    put(dst, stride, x, y, v);
                }
            }
        }
        8 => {
            // Horizontal_Up
            if !e.has_left {
                return false;
            }
            for y in 0..4 {
                for x in 0..4 {
                    let z = x + 2 * y;
                    let v = match z {
                        0 | 2 | 4 => (l(y + (x >> 1)) + l(y + (x >> 1) + 1) + 1) >> 1,
                        1 | 3 => {
                            (l(y + (x >> 1)) + 2 * l(y + (x >> 1) + 1) + l(y + (x >> 1) + 2) + 2)
                                >> 2
                        }
                        5 => (l(2) + 3 * l(3) + 2) >> 2,
                        _ => l(3),
                    };
                    put(dst, stride, x, y, v);
                }
            }
        }
        _ => return false,
    }
    true
}

/// Neighbouring samples of a 16x16 luma or chroma block: `top` is
/// p[0..w-1, -1], `left` is p[-1, 0..h-1].
pub struct EdgeMb<'a, T = u8> {
    pub top: &'a [T],
    pub left: &'a [T],
    pub corner: T,
    pub has_top: bool,
    /// The whole left column is available.
    pub has_left: bool,
    /// The upper and lower halves of the left column are available: they
    /// can differ in MBAFF frames with constrained intra prediction, which
    /// chroma DC prediction judges per 4x4 block (8.3.4).
    pub left_half: [bool; 2],
    pub has_corner: bool,
}

/// Intra_16x16 prediction (8.3.3).
pub fn pred16x16<T: Pixel>(dst: &mut [T], stride: usize, mode: u8, e: &EdgeMb<T>, bd: u32) -> bool {
    let max = (1 << bd) - 1;
    match mode {
        0 => {
            if !e.has_top {
                return false;
            }
            for y in 0..16 {
                dst[y * stride..y * stride + 16].copy_from_slice(&e.top[..16]);
            }
        }
        1 => {
            if !e.has_left {
                return false;
            }
            for y in 0..16 {
                dst[y * stride..y * stride + 16].fill(e.left[y]);
            }
        }
        2 => {
            let st: u32 = e.top[..16].iter().map(|&v| v.get() as u32).sum();
            let sl: u32 = e.left[..16].iter().map(|&v| v.get() as u32).sum();
            let v = T::new(match (e.has_top, e.has_left) {
                (true, true) => (st + sl + 16) >> 5,
                (false, true) => (sl + 8) >> 4,
                (true, false) => (st + 8) >> 4,
                (false, false) => 1 << (bd - 1),
            } as i32);
            for y in 0..16 {
                dst[y * stride..y * stride + 16].fill(v);
            }
        }
        3 => {
            if !(e.has_top && e.has_left && e.has_corner) {
                return false;
            }
            let p_top = |x: i32| {
                if x < 0 {
                    e.corner.get()
                } else {
                    e.top[x as usize].get()
                }
            };
            let p_left = |y: i32| {
                if y < 0 {
                    e.corner.get()
                } else {
                    e.left[y as usize].get()
                }
            };
            let h: i32 = (0..8)
                .map(|x| (x + 1) * (p_top(8 + x) - p_top(6 - x)))
                .sum();
            let v: i32 = (0..8)
                .map(|y| (y + 1) * (p_left(8 + y) - p_left(6 - y)))
                .sum();
            let a = 16 * (p_left(15) + p_top(15));
            let b = (5 * h + 32) >> 6;
            let c = (5 * v + 32) >> 6;
            for y in 0..16i32 {
                for x in 0..16i32 {
                    dst[y as usize * stride + x as usize] =
                        T::clip((a + b * (x - 7) + c * (y - 7) + 16) >> 5, max);
                }
            }
        }
        _ => return false,
    }
    true
}

/// Chroma intra prediction for one component (8.3.4), for MbWidthC x
/// MbHeightC = `w` x `h`. `chroma_format_idc` selects the plane constants.
#[allow(clippy::too_many_arguments)]
pub fn pred_chroma<T: Pixel>(
    dst: &mut [T],
    stride: usize,
    mode: u8,
    e: &EdgeMb<T>,
    w: usize,
    h: usize,
    chroma_format_idc: u8,
    bd: u32,
) -> bool {
    let (grey, max) = (1 << (bd - 1), (1 << bd) - 1);
    match mode {
        0 => {
            // DC per 4x4 chroma block (8.3.4.1 to 8.3.4.3).
            for yo in (0..h).step_by(4) {
                for xo in (0..w).step_by(4) {
                    let st = || {
                        e.top[xo..xo + 4]
                            .iter()
                            .map(|&v| v.get() as u32)
                            .sum::<u32>()
                    };
                    let sl = || {
                        e.left[yo..yo + 4]
                            .iter()
                            .map(|&v| v.get() as u32)
                            .sum::<u32>()
                    };
                    let both = || (st() + sl() + 4) >> 3;
                    let top = || (st() + 2) >> 2;
                    let left = || (sl() + 2) >> 2;
                    let has_left = e.left_half[(yo >= h / 2) as usize];
                    let v = if (xo == 0 && yo == 0) || (xo > 0 && yo > 0) {
                        match (e.has_top, has_left) {
                            (true, true) => both(),
                            (false, true) => left(),
                            (true, false) => top(),
                            _ => grey,
                        }
                    } else if xo > 0 && yo == 0 {
                        if e.has_top {
                            top()
                        } else if has_left {
                            left()
                        } else {
                            grey
                        }
                    } else if has_left {
                        left()
                    } else if e.has_top {
                        top()
                    } else {
                        grey
                    };
                    for y in yo..yo + 4 {
                        dst[y * stride + xo..y * stride + xo + 4].fill(T::new(v as i32));
                    }
                }
            }
        }
        1 => {
            if !e.has_left {
                return false;
            }
            for y in 0..h {
                dst[y * stride..y * stride + w].fill(e.left[y]);
            }
        }
        2 => {
            if !e.has_top {
                return false;
            }
            for y in 0..h {
                dst[y * stride..y * stride + w].copy_from_slice(&e.top[..w]);
            }
        }
        3 => {
            if !(e.has_top && e.has_left && e.has_corner) {
                return false;
            }
            let p_top = |x: i32| {
                if x < 0 {
                    e.corner.get()
                } else {
                    e.top[x as usize].get()
                }
            };
            let p_left = |y: i32| {
                if y < 0 {
                    e.corner.get()
                } else {
                    e.left[y as usize].get()
                }
            };
            let xcf = 4 * (chroma_format_idc == 3) as i32;
            let ycf = 4 * (chroma_format_idc != 1) as i32;
            let hh: i32 = (0..=3 + xcf)
                .map(|x| (x + 1) * (p_top(4 + xcf + x) - p_top(2 + xcf - x)))
                .sum();
            let vv: i32 = (0..=3 + ycf)
                .map(|y| (y + 1) * (p_left(4 + ycf + y) - p_left(2 + ycf - y)))
                .sum();
            let a = 16 * (p_left(h as i32 - 1) + p_top(w as i32 - 1));
            let b = ((34 - 29 * (chroma_format_idc == 3) as i32) * hh + 32) >> 6;
            let c = ((34 - 29 * (chroma_format_idc != 1) as i32) * vv + 32) >> 6;
            for y in 0..h as i32 {
                for x in 0..w as i32 {
                    dst[y as usize * stride + x as usize] =
                        T::clip((a + b * (x - 3 - xcf) + c * (y - 3 - ycf) + 16) >> 5, max);
                }
            }
        }
        _ => return false,
    }
    true
}

/// Unfiltered neighbouring samples of an 8x8 block: `top` is p[0..15, -1]
/// (p[8..15, -1] already substituted when unavailable), `left` is
/// p[-1, 0..7], `corner` is p[-1, -1].
pub struct Edge8<T = u8> {
    pub top: [T; 16],
    pub left: [T; 8],
    pub corner: T,
    pub has_top: bool,
    pub has_left: bool,
    pub has_corner: bool,
}

/// Intra_8x8 prediction (8.3.2.2) including reference sample filtering
/// (8.3.2.2.1). Returns false when the mode needs unavailable samples.
pub fn pred8x8<T: Pixel>(dst: &mut [T], stride: usize, mode: u8, e: &Edge8<T>, bd: u32) -> bool {
    let p = |v: T| v.get() as u32;
    let all = e.has_top && e.has_left && e.has_corner;
    let needs = match mode {
        0 | 3 | 7 => e.has_top,
        1 | 8 => e.has_left,
        4..=6 => all,
        2 => true,
        _ => false,
    };
    if !needs {
        return false;
    }
    // Filtered samples p', of the edges the mode reads: the top for all
    // but Horizontal and Horizontal_Up, the left for all but Vertical,
    // Diagonal_Down_Left and Vertical_Left.
    let (use_top, use_left) = (!matches!(mode, 1 | 8), !matches!(mode, 0 | 3 | 7));
    let mut top = [0u32; 16];
    let mut left = [0u32; 8];
    let mut corner = p(e.corner);
    if e.has_top && use_top {
        top[0] = if e.has_corner {
            (p(e.corner) + 2 * p(e.top[0]) + p(e.top[1]) + 2) >> 2
        } else {
            (3 * p(e.top[0]) + p(e.top[1]) + 2) >> 2
        };
        for x in 1..15 {
            top[x] = (p(e.top[x - 1]) + 2 * p(e.top[x]) + p(e.top[x + 1]) + 2) >> 2;
        }
        top[15] = (p(e.top[14]) + 3 * p(e.top[15]) + 2) >> 2;
    }
    if e.has_corner {
        corner = match (e.has_top, e.has_left) {
            (true, true) => (p(e.top[0]) + 2 * p(e.corner) + p(e.left[0]) + 2) >> 2,
            (true, false) => (3 * p(e.corner) + p(e.top[0]) + 2) >> 2,
            (false, true) => (3 * p(e.corner) + p(e.left[0]) + 2) >> 2,
            (false, false) => p(e.corner),
        };
    }
    if e.has_left && use_left {
        left[0] = if e.has_corner {
            (p(e.corner) + 2 * p(e.left[0]) + p(e.left[1]) + 2) >> 2
        } else {
            (3 * p(e.left[0]) + p(e.left[1]) + 2) >> 2
        };
        for y in 1..7 {
            left[y] = (p(e.left[y - 1]) + 2 * p(e.left[y]) + p(e.left[y + 1]) + 2) >> 2;
        }
        left[7] = (p(e.left[6]) + 3 * p(e.left[7]) + 2) >> 2;
    }
    match mode {
        0 => {
            let row: [T; 8] = std::array::from_fn(|x| T::new(top[x] as i32));
            for y in 0..8 {
                dst[y * stride..y * stride + 8].copy_from_slice(&row);
            }
            return true;
        }
        1 => {
            for y in 0..8 {
                dst[y * stride..y * stride + 8].fill(T::new(left[y] as i32));
            }
            return true;
        }
        2 => {
            let v = match (e.has_top, e.has_left) {
                (true, true) => (top[..8].iter().sum::<u32>() + left.iter().sum::<u32>() + 8) >> 4,
                (false, true) => (left.iter().sum::<u32>() + 4) >> 3,
                (true, false) => (top[..8].iter().sum::<u32>() + 4) >> 3,
                (false, false) => 1 << (bd - 1),
            };
            for y in 0..8 {
                dst[y * stride..y * stride + 8].fill(T::new(v as i32));
            }
            return true;
        }
        _ => {}
    }
    // The filtered edge as one line, left[7..0], corner, top[0..15], with
    // its 2-tap (h2) and 3-tap (g) averages: each directional mode reads
    // its samples from these (8.3.2.2.4 to 8.3.2.2.10). g[24] is the last
    // Diagonal_Down_Left sample, which repeats top[15].
    let mut ed = [0u32; 25];
    for (k, &v) in left.iter().enumerate() {
        ed[7 - k] = v;
    }
    ed[8] = corner;
    ed[9..].copy_from_slice(&top);
    let mut h2 = [0u32; 24];
    let mut g = [0u32; 25];
    for k in 0..24 {
        h2[k] = (ed[k] + ed[k + 1] + 1) >> 1;
    }
    for k in 1..24 {
        g[k] = (ed[k - 1] + 2 * ed[k] + ed[k + 1] + 2) >> 2;
    }
    g[24] = (top[14] + 3 * top[15] + 2) >> 2;
    #[inline(always)]
    fn rows8<T: Pixel>(dst: &mut [T], stride: usize, f: impl Fn(usize, usize) -> u32) {
        for y in 0..8 {
            let row = &mut dst[y * stride..y * stride + 8];
            for (x, d) in row.iter_mut().enumerate() {
                *d = T::new(f(x, y) as i32);
            }
        }
    }

    match mode {
        3 => rows8(dst, stride, |x, y| g[10 + x + y]),
        4 => rows8(dst, stride, |x, y| g[8 + x - y]),
        5 => rows8(dst, stride, |x, y| {
            let z = 2 * x as i32 - y as i32;
            if z < 0 {
                g[(9 + z) as usize]
            } else if z % 2 == 0 {
                h2[8 + x - (y >> 1)]
            } else {
                g[8 + x - (y >> 1)]
            }
        }),
        6 => rows8(dst, stride, |x, y| {
            let z = 2 * y as i32 - x as i32;
            if z < -1 {
                g[(7 - z) as usize]
            } else if z == -1 {
                g[8]
            } else if z % 2 == 0 {
                h2[7 - y + (x >> 1)]
            } else {
                g[8 - y + (x >> 1)]
            }
        }),
        7 => rows8(dst, stride, |x, y| {
            let k = x + (y >> 1);
            if y % 2 == 0 { h2[9 + k] } else { g[10 + k] }
        }),
        _ => rows8(dst, stride, |x, y| {
            let (z, k) = (x + 2 * y, y + (x >> 1));
            match z {
                0..=12 if z % 2 == 0 => h2[6 - k],
                0..=12 => g[6 - k],
                13 => (left[6] + 3 * left[7] + 2) >> 2,
                _ => left[7],
            }
        }),
    }
    true
}

#[cfg(test)]
fn pred8x8_reference<T: Pixel>(
    dst: &mut [T],
    stride: usize,
    mode: u8,
    e: &Edge8<T>,
    bd: u32,
) -> bool {
    let p = |v: T| v.get() as u32;
    // Filtered samples p'.
    let mut top = [0u32; 16];
    let mut left = [0u32; 8];
    let mut corner = p(e.corner);
    if e.has_top {
        top[0] = if e.has_corner {
            (p(e.corner) + 2 * p(e.top[0]) + p(e.top[1]) + 2) >> 2
        } else {
            (3 * p(e.top[0]) + p(e.top[1]) + 2) >> 2
        };
        for x in 1..15 {
            top[x] = (p(e.top[x - 1]) + 2 * p(e.top[x]) + p(e.top[x + 1]) + 2) >> 2;
        }
        top[15] = (p(e.top[14]) + 3 * p(e.top[15]) + 2) >> 2;
    }
    if e.has_corner {
        corner = match (e.has_top, e.has_left) {
            (true, true) => (p(e.top[0]) + 2 * p(e.corner) + p(e.left[0]) + 2) >> 2,
            (true, false) => (3 * p(e.corner) + p(e.top[0]) + 2) >> 2,
            (false, true) => (3 * p(e.corner) + p(e.left[0]) + 2) >> 2,
            (false, false) => p(e.corner),
        };
    }
    if e.has_left {
        left[0] = if e.has_corner {
            (p(e.corner) + 2 * p(e.left[0]) + p(e.left[1]) + 2) >> 2
        } else {
            (3 * p(e.left[0]) + p(e.left[1]) + 2) >> 2
        };
        for y in 1..7 {
            left[y] = (p(e.left[y - 1]) + 2 * p(e.left[y]) + p(e.left[y + 1]) + 2) >> 2;
        }
        left[7] = (p(e.left[6]) + 3 * p(e.left[7]) + 2) >> 2;
    }
    let t = |x: i32| if x < 0 { corner } else { top[x as usize] };
    let l = |y: i32| if y < 0 { corner } else { left[y as usize] };
    let all = e.has_top && e.has_left && e.has_corner;
    let needs = match mode {
        0 | 3 | 7 => e.has_top,
        1 | 8 => e.has_left,
        4..=6 => all,
        2 => true,
        _ => false,
    };
    if !needs {
        return false;
    }
    // One loop per mode, so the mode is not decided per sample.
    #[inline(always)]
    fn fill8<T: Pixel>(dst: &mut [T], stride: usize, f: impl Fn(i32, i32) -> u32) {
        for y in 0..8i32 {
            let row = &mut dst[y as usize * stride..y as usize * stride + 8];
            for (x, d) in row.iter_mut().enumerate() {
                *d = T::new(f(x as i32, y) as i32);
            }
        }
    }

    match mode {
        0 => {
            for y in 0..8 {
                for x in 0..8 {
                    dst[y * stride + x] = T::new(top[x] as i32);
                }
            }
        }
        1 => {
            for y in 0..8 {
                dst[y * stride..y * stride + 8].fill(T::new(left[y] as i32));
            }
        }
        2 => {
            let v = match (e.has_top, e.has_left) {
                (true, true) => (top[..8].iter().sum::<u32>() + left.iter().sum::<u32>() + 8) >> 4,
                (false, true) => (left.iter().sum::<u32>() + 4) >> 3,
                (true, false) => (top[..8].iter().sum::<u32>() + 4) >> 3,
                (false, false) => 1 << (bd - 1),
            };
            for y in 0..8 {
                dst[y * stride..y * stride + 8].fill(T::new(v as i32));
            }
        }
        3 => fill8(dst, stride, |x, y| {
            if x == 7 && y == 7 {
                (t(14) + 3 * t(15) + 2) >> 2
            } else {
                (t(x + y) + 2 * t(x + y + 1) + t(x + y + 2) + 2) >> 2
            }
        }),
        4 => fill8(dst, stride, |x, y| {
            if x > y {
                (t(x - y - 2) + 2 * t(x - y - 1) + t(x - y) + 2) >> 2
            } else if x < y {
                (l(y - x - 2) + 2 * l(y - x - 1) + l(y - x) + 2) >> 2
            } else {
                (t(0) + 2 * corner + l(0) + 2) >> 2
            }
        }),
        5 => fill8(dst, stride, |x, y| {
            let z = 2 * x - y;
            if z >= 0 && z % 2 == 0 {
                (t(x - (y >> 1) - 1) + t(x - (y >> 1)) + 1) >> 1
            } else if z >= 0 {
                (t(x - (y >> 1) - 2) + 2 * t(x - (y >> 1) - 1) + t(x - (y >> 1)) + 2) >> 2
            } else if z == -1 {
                (l(0) + 2 * corner + t(0) + 2) >> 2
            } else {
                (l(y - 2 * x - 1) + 2 * l(y - 2 * x - 2) + l(y - 2 * x - 3) + 2) >> 2
            }
        }),
        6 => fill8(dst, stride, |x, y| {
            let z = 2 * y - x;
            if z >= 0 && z % 2 == 0 {
                (l(y - (x >> 1) - 1) + l(y - (x >> 1)) + 1) >> 1
            } else if z >= 0 {
                (l(y - (x >> 1) - 2) + 2 * l(y - (x >> 1) - 1) + l(y - (x >> 1)) + 2) >> 2
            } else if z == -1 {
                (l(0) + 2 * corner + t(0) + 2) >> 2
            } else {
                (t(x - 2 * y - 1) + 2 * t(x - 2 * y - 2) + t(x - 2 * y - 3) + 2) >> 2
            }
        }),
        7 => fill8(dst, stride, |x, y| {
            let k = x + (y >> 1);
            if y % 2 == 0 {
                (t(k) + t(k + 1) + 1) >> 1
            } else {
                (t(k) + 2 * t(k + 1) + t(k + 2) + 2) >> 2
            }
        }),
        _ => fill8(dst, stride, |x, y| {
            let z = x + 2 * y;
            let k = y + (x >> 1);
            if z <= 12 && z % 2 == 0 {
                (l(k) + l(k + 1) + 1) >> 1
            } else if z < 13 {
                (l(k) + 2 * l(k + 1) + l(k + 2) + 2) >> 2
            } else if z == 13 {
                (l(6) + 3 * l(7) + 2) >> 2
            } else {
                l(7)
            }
        }),
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge() -> Edge4 {
        Edge4 {
            top: [10, 20, 30, 40, 50, 60, 70, 80],
            left: [15, 25, 35, 45],
            corner: 5,
            has_top: true,
            has_left: true,
            has_corner: true,
        }
    }

    #[test]
    fn dc_and_directional_4x4() {
        let mut d = [0u8; 16];
        assert!(pred4x4(&mut d, 4, 2, &edge(), 8));
        assert!(d.iter().all(|&v| v == 28)); // (100 + 120 + 4) >> 3
        assert!(pred4x4(&mut d, 4, 3, &edge(), 8));
        assert_eq!(d[0], 20); // (10 + 40 + 30 + 2) >> 2
        assert_eq!(d[15], 78); // (70 + 240 + 2) >> 2
        assert!(pred4x4(&mut d, 4, 8, &edge(), 8));
        assert_eq!(&d[12..16], &[45, 45, 45, 45]);
        let mut e = edge();
        e.has_top = false;
        assert!(!pred4x4(&mut d, 4, 0, &e, 8));
    }

    #[test]
    fn intra_8x8_matches_reference() {
        let mut seed = 5u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            (seed >> 16) as u8
        };
        for case in 0..2000 {
            let e = Edge8 {
                top: std::array::from_fn(|_| rnd()),
                left: std::array::from_fn(|_| rnd()),
                corner: rnd(),
                has_top: case % 4 != 1,
                has_left: case % 4 != 2,
                has_corner: case % 4 != 3,
            };
            for mode in 0..9 {
                let (mut a, mut b) = ([0u8; 64], [0u8; 64]);
                let ok = pred8x8(&mut a, 8, mode, &e, 8);
                assert_eq!(ok, pred8x8_reference(&mut b, 8, mode, &e, 8));
                if ok {
                    assert_eq!(a, b, "mode {mode} case {case}");
                }
            }
        }
    }

    #[test]
    fn plane_16x16_is_flat_for_flat_edges() {
        let top = [77u8; 16];
        let left = [77u8; 16];
        let e = EdgeMb {
            top: &top,
            left: &left,
            corner: 77,
            has_top: true,
            has_left: true,
            left_half: [true; 2],
            has_corner: true,
        };
        let mut d = [0u8; 256];
        assert!(pred16x16(&mut d, 16, 3, &e, 8));
        assert!(d.iter().all(|&v| v == 77));
    }
}
