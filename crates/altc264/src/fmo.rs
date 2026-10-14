//! Slice groups (flexible macroblock ordering, 8.2.2).
use crate::ps::{Pps, SliceGroupMap, Sps};
use crate::slice::SliceHeader;

/// mbToSliceGroupMap of the current picture (8.2.2.8), indexed by
/// macroblock address (pair-interleaved in MBAFF frames), or None without
/// slice groups.
pub fn mb_to_slice_group_map(sps: &Sps, pps: &Pps, h: &SliceHeader) -> Option<Vec<u8>> {
    let map = pps.slice_group_map.as_ref()?;
    let w = sps.pic_width_in_mbs as usize;
    let hm = sps.pic_height_in_map_units as usize;
    let size = w * hm;
    let groups = pps.num_slice_groups as usize;
    let units_in_group0 =
        |rate: u32| (h.slice_group_change_cycle as usize * rate as usize).min(size);
    let mut units = vec![0u8; size];
    match map {
        SliceGroupMap::Interleaved { run_length } => {
            // 8.2.2.1
            let mut i = 0;
            while i < size {
                for (g, &run) in run_length.iter().enumerate().take(groups) {
                    if i >= size {
                        break;
                    }
                    for j in 0..(run as usize).min(size - i) {
                        units[i + j] = g as u8;
                    }
                    i += run as usize;
                }
            }
        }
        SliceGroupMap::Dispersed => {
            // 8.2.2.2
            for (i, u) in units.iter_mut().enumerate() {
                *u = ((i % w + (i / w * groups) / 2) % groups) as u8;
            }
        }
        SliceGroupMap::Foreground {
            top_left,
            bottom_right,
        } => {
            // 8.2.2.3
            units.fill(groups as u8 - 1);
            for g in (0..groups - 1).rev() {
                let (tl, br) = (top_left[g] as usize, bottom_right[g] as usize);
                for y in tl / w..=(br / w).min(hm - 1) {
                    for x in tl % w..=br % w {
                        if let Some(u) = units.get_mut(y * w + x) {
                            *u = g as u8;
                        }
                    }
                }
            }
        }
        &SliceGroupMap::Changing {
            map_type,
            change_direction,
            change_rate,
        } => {
            let n0 = units_in_group0(change_rate);
            let dir = change_direction as i32;
            let upper_left = if change_direction { size - n0 } else { n0 };
            match map_type {
                3 => {
                    // 8.2.2.4, box-out
                    units.fill(1);
                    let (w, hm) = (w as i32, hm as i32);
                    let (mut x, mut y) = ((w - dir) / 2, (hm - dir) / 2);
                    let (mut left, mut top, mut right, mut bottom) = (x, y, x, y);
                    let (mut xd, mut yd) = (dir - 1, dir);
                    let mut k = 0;
                    while k < n0 {
                        let u = &mut units[(y * w + x) as usize];
                        let vacant = *u == 1;
                        if vacant {
                            *u = 0;
                        }
                        if xd == -1 && x == left {
                            left = (left - 1).max(0);
                            x = left;
                            (xd, yd) = (0, 2 * dir - 1);
                        } else if xd == 1 && x == right {
                            right = (right + 1).min(w - 1);
                            x = right;
                            (xd, yd) = (0, 1 - 2 * dir);
                        } else if yd == -1 && y == top {
                            top = (top - 1).max(0);
                            y = top;
                            (xd, yd) = (1 - 2 * dir, 0);
                        } else if yd == 1 && y == bottom {
                            bottom = (bottom + 1).min(hm - 1);
                            y = bottom;
                            (xd, yd) = (2 * dir - 1, 0);
                        } else {
                            (x, y) = (x + xd, y + yd);
                        }
                        k += vacant as usize;
                    }
                }
                4 => {
                    // 8.2.2.5, raster scan
                    for (i, u) in units.iter_mut().enumerate() {
                        *u = if i < upper_left { dir } else { 1 - dir } as u8;
                    }
                }
                _ => {
                    // 8.2.2.6, wipe
                    let mut k = 0;
                    for x in 0..w {
                        for y in 0..hm {
                            units[y * w + x] = if k < upper_left { dir } else { 1 - dir } as u8;
                            k += 1;
                        }
                    }
                }
            }
        }
        SliceGroupMap::Explicit { slice_group_id } => {
            // 8.2.2.7
            for (u, &g) in units.iter_mut().zip(slice_group_id) {
                *u = g;
            }
        }
    }
    // 8.2.2.8
    let mbaff = sps.mb_adaptive_frame_field && !h.field_pic;
    Some(if sps.frame_mbs_only || h.field_pic {
        units
    } else if mbaff {
        (0..2 * size).map(|i| units[i / 2]).collect()
    } else {
        (0..2 * size)
            .map(|i| units[i / (2 * w) * w + i % w])
            .collect()
    })
}

/// nextMbAddress (8-16): the next macroblock of the same slice group, or
/// the picture size when there is none.
#[inline]
pub fn next_mb_addr(map: Option<&[u8]>, addr: usize, total: usize) -> usize {
    let Some(map) = map else {
        return addr + 1;
    };
    let g = map.get(addr).copied();
    (addr + 1..total)
        .find(|&i| map.get(i).copied() == g)
        .unwrap_or(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_skips_other_groups() {
        let map = [0, 1, 0, 1, 1, 0];
        assert_eq!(next_mb_addr(Some(&map), 0, 6), 2);
        assert_eq!(next_mb_addr(Some(&map), 2, 6), 5);
        assert_eq!(next_mb_addr(Some(&map), 5, 6), 6);
        assert_eq!(next_mb_addr(Some(&map), 1, 6), 3);
        assert_eq!(next_mb_addr(None, 4, 6), 5);
    }
}
