//! Picture order count (8.2.1) and detection of the first slice of a new
//! picture (7.4.1.2.4). In MVC each view keeps its own state (H.8.1): the
//! "previous" pictures referred to below are those of the same view.

use crate::ps::{PicOrderCnt, Sps};
use crate::slice::{DecRefPicMarking, Mmco, SliceHeader};

/// Top/bottom field order counts of the current picture.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Poc {
    pub top: i32,
    pub bottom: i32,
}

/// What the picture is, as far as POC is concerned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Structure {
    Frame,
    TopField,
    BottomField,
}

impl Poc {
    /// PicOrderCnt(CurrPic) (8-1).
    pub fn pic(self, s: Structure) -> i32 {
        match s {
            Structure::Frame => self.top.min(self.bottom),
            Structure::TopField => self.top,
            Structure::BottomField => self.bottom,
        }
    }
}

pub fn structure(h: &SliceHeader) -> Structure {
    match (h.field_pic, h.bottom_field) {
        (false, _) => Structure::Frame,
        (true, false) => Structure::TopField,
        (true, true) => Structure::BottomField,
    }
}

fn has_mmco5(h: &SliceHeader) -> bool {
    matches!(&h.dec_ref_pic_marking, Some(DecRefPicMarking::Adaptive(ops)) if ops.contains(&Mmco::ForgetAll))
}

/// Per-view state carried between pictures.
#[derive(Clone, Debug, Default)]
pub struct PocState {
    // Type 0: of the previous reference picture.
    prev_msb: i32,
    prev_lsb: i32,
    // Types 1 and 2: of the previous picture.
    prev_frame_num_offset: i32,
    prev_frame_num: u32,
    prev_had_mmco5: bool,
}

impl PocState {
    /// Computes the POC of a picture from its first slice and updates the
    /// state as if the picture had been decoded.
    pub fn next(&mut self, sps: &Sps, h: &SliceHeader, nal_ref_idc: u8) -> Poc {
        let s = structure(h);
        let max_frame_num = 1i32 << sps.log2_max_frame_num;
        let is_ref = nal_ref_idc != 0;
        let mmco5 = has_mmco5(h);
        match &sps.poc {
            PicOrderCnt::Type0 { log2_max_lsb } => {
                let max_lsb = 1i32 << log2_max_lsb;
                let (prev_msb, prev_lsb) = if h.idr {
                    (0, 0)
                } else {
                    (self.prev_msb, self.prev_lsb)
                };
                let lsb = h.pic_order_cnt_lsb as i32;
                let msb = if lsb < prev_lsb && prev_lsb - lsb >= max_lsb / 2 {
                    prev_msb + max_lsb
                } else if lsb > prev_lsb && lsb - prev_lsb > max_lsb / 2 {
                    prev_msb - max_lsb
                } else {
                    prev_msb
                };
                let poc = match s {
                    Structure::Frame => Poc {
                        top: msb + lsb,
                        bottom: msb + lsb + h.delta_pic_order_cnt_bottom,
                    },
                    Structure::TopField => Poc {
                        top: msb + lsb,
                        bottom: 0,
                    },
                    Structure::BottomField => Poc {
                        top: 0,
                        bottom: msb + lsb,
                    },
                };
                if is_ref {
                    if mmco5 {
                        // 8.2.1.1: after mmco 5 the previous values are 0 and,
                        // unless this is a bottom field, the top field order
                        // count after the tempPicOrderCnt adjustment (8.2.1).
                        let adjusted = poc.top - poc.pic(s);
                        (self.prev_msb, self.prev_lsb) = (
                            0,
                            if s == Structure::BottomField {
                                0
                            } else {
                                adjusted
                            },
                        );
                    } else {
                        (self.prev_msb, self.prev_lsb) = (msb, lsb);
                    }
                }
                poc
            }
            PicOrderCnt::Type1 {
                offset_for_non_ref_pic,
                offset_for_top_to_bottom_field,
                offset_for_ref_frame,
                ..
            } => {
                let frame_num_offset = self.frame_num_offset(h, max_frame_num);
                let n = offset_for_ref_frame.len() as i32;
                let mut abs_frame_num = if n != 0 {
                    frame_num_offset + h.frame_num as i32
                } else {
                    0
                };
                if !is_ref && abs_frame_num > 0 {
                    abs_frame_num -= 1;
                }
                let mut expected = 0i32;
                if abs_frame_num > 0 {
                    let cycle = (abs_frame_num - 1) / n;
                    let in_cycle = ((abs_frame_num - 1) % n) as usize;
                    let delta_per_cycle: i32 = offset_for_ref_frame
                        .iter()
                        .fold(0i32, |a, &b| a.wrapping_add(b));
                    expected = cycle.wrapping_mul(delta_per_cycle);
                    for &o in &offset_for_ref_frame[..=in_cycle] {
                        expected = expected.wrapping_add(o);
                    }
                }
                if !is_ref {
                    expected = expected.wrapping_add(*offset_for_non_ref_pic);
                }
                let d = h.delta_pic_order_cnt;
                let poc = match s {
                    Structure::Frame => {
                        let top = expected + d[0];
                        Poc {
                            top,
                            bottom: top + offset_for_top_to_bottom_field + d[1],
                        }
                    }
                    Structure::TopField => Poc {
                        top: expected + d[0],
                        bottom: 0,
                    },
                    Structure::BottomField => Poc {
                        top: 0,
                        bottom: expected + offset_for_top_to_bottom_field + d[0],
                    },
                };
                self.finish_type12(h, frame_num_offset, mmco5);
                poc
            }
            PicOrderCnt::Type2 => {
                let frame_num_offset = self.frame_num_offset(h, max_frame_num);
                let t = if h.idr {
                    0
                } else if !is_ref {
                    2 * (frame_num_offset + h.frame_num as i32) - 1
                } else {
                    2 * (frame_num_offset + h.frame_num as i32)
                };
                self.finish_type12(h, frame_num_offset, mmco5);
                match s {
                    Structure::Frame => Poc { top: t, bottom: t },
                    Structure::TopField => Poc { top: t, bottom: 0 },
                    Structure::BottomField => Poc { top: 0, bottom: t },
                }
            }
        }
    }

    /// FrameNumOffset (8-6, 8-11).
    fn frame_num_offset(&self, h: &SliceHeader, max_frame_num: i32) -> i32 {
        if h.idr {
            0
        } else {
            let prev = if self.prev_had_mmco5 {
                0
            } else {
                self.prev_frame_num_offset
            };
            if self.prev_frame_num > h.frame_num {
                prev + max_frame_num
            } else {
                prev
            }
        }
    }

    fn finish_type12(&mut self, h: &SliceHeader, frame_num_offset: i32, mmco5: bool) {
        self.prev_frame_num_offset = frame_num_offset;
        // After mmco 5 the picture's frame_num is inferred to be 0 (7.4.3).
        self.prev_frame_num = if mmco5 { 0 } else { h.frame_num };
        self.prev_had_mmco5 = mmco5;
    }
}

/// Whether `cur` starts a new primary coded picture after `prev` (7.4.1.2.4),
/// both slices of the same view. `nal_ref_idc` values are passed separately
/// since they live in the NAL header.
pub fn first_slice_of_new_picture(
    sps: &Sps,
    prev: &SliceHeader,
    prev_ref_idc: u8,
    cur: &SliceHeader,
    cur_ref_idc: u8,
) -> bool {
    if prev.frame_num != cur.frame_num
        || prev.pps_id != cur.pps_id
        || prev.field_pic != cur.field_pic
        || (cur.field_pic && prev.bottom_field != cur.bottom_field)
        || ((prev_ref_idc == 0) != (cur_ref_idc == 0))
        || prev.idr != cur.idr
        || (cur.idr && prev.idr_pic_id != cur.idr_pic_id)
    {
        return true;
    }
    match sps.poc {
        PicOrderCnt::Type0 { .. } => {
            prev.pic_order_cnt_lsb != cur.pic_order_cnt_lsb
                || prev.delta_pic_order_cnt_bottom != cur.delta_pic_order_cnt_bottom
        }
        PicOrderCnt::Type1 { .. } => prev.delta_pic_order_cnt != cur.delta_pic_order_cnt,
        PicOrderCnt::Type2 => false,
    }
}
