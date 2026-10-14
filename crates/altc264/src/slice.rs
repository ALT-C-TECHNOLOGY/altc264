//! Slice headers (7.3.3), including the MVC reference list modification
//! (H.7.3.3.1.1) used by non-base view slices (NAL unit type 20).

use crate::Error;
use crate::bits::BitReader;
use crate::nal::{NalHeader, NalType};
use crate::ps::{PicOrderCnt, Pps, SliceGroupMap, Sps};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SliceType {
    P,
    B,
    I,
    Sp,
    Si,
}

impl SliceType {
    fn from_ue(v: u32) -> Self {
        match v % 5 {
            0 => Self::P,
            1 => Self::B,
            2 => Self::I,
            3 => Self::Sp,
            _ => Self::Si,
        }
    }

    pub fn is_intra(self) -> bool {
        matches!(self, Self::I | Self::Si)
    }

    /// Number of reference picture lists the slice uses.
    pub fn num_lists(self) -> usize {
        match self {
            Self::B => 2,
            Self::P | Self::Sp => 1,
            _ => 0,
        }
    }
}

/// One modification_of_pic_nums_idc operation (7.4.3.1, H.7.4.3.1.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefPicListModification {
    /// idc 0: picNumPred - (abs_diff_pic_num_minus1 + 1)
    ShortTermSub(u32),
    /// idc 1: picNumPred + (abs_diff_pic_num_minus1 + 1)
    ShortTermAdd(u32),
    /// idc 2
    LongTerm(u32),
    /// idc 4 (MVC): picViewIdxPred - (abs_diff_view_idx_minus1 + 1)
    InterViewSub(u32),
    /// idc 5 (MVC): picViewIdxPred + (abs_diff_view_idx_minus1 + 1)
    InterViewAdd(u32),
}

/// Memory management control operations (7.4.3.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mmco {
    /// 1: mark a short-term picture unused (difference_of_pic_nums_minus1).
    ForgetShort(u32),
    /// 2: mark a long-term picture unused (long_term_pic_num).
    ForgetLong(u32),
    /// 3: short-term to long-term (difference_of_pic_nums_minus1, long_term_frame_idx).
    ShortToLong(u32, u32),
    /// 4: max_long_term_frame_idx_plus1.
    MaxLongTermIdx(u32),
    /// 5: mark everything unused.
    ForgetAll,
    /// 6: current picture becomes long-term (long_term_frame_idx).
    CurrentToLong(u32),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecRefPicMarking {
    Idr {
        no_output_of_prior_pics: bool,
        long_term_reference: bool,
    },
    SlidingWindow,
    Adaptive(Vec<Mmco>),
}

/// Explicit weights for one reference: (weight, offset) for Y, Cb, Cr.
/// Absent flags take the defaults from 7.4.3.2 (2^denom, 0).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Weights {
    pub luma: (i16, i16),
    pub chroma: [(i16, i16); 2],
    pub luma_flag: bool,
    pub chroma_flag: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PredWeightTable {
    pub luma_log2_denom: u8,
    pub chroma_log2_denom: u8,
    pub lists: [Vec<Weights>; 2],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SliceHeader {
    pub first_mb_in_slice: u32,
    pub slice_type: SliceType,
    /// slice_type >= 5: all slices of the picture share this type.
    pub slice_type_fixed: bool,
    pub pps_id: u8,
    pub colour_plane_id: u8,
    pub frame_num: u32,
    pub field_pic: bool,
    pub bottom_field: bool,
    pub idr: bool,
    pub idr_pic_id: u32,
    pub pic_order_cnt_lsb: u32,
    pub delta_pic_order_cnt_bottom: i32,
    pub delta_pic_order_cnt: [i32; 2],
    pub redundant_pic_cnt: u32,
    pub direct_spatial_mv_pred: bool,
    pub num_ref_idx_active: [u32; 2],
    pub ref_pic_list_modification: [Vec<RefPicListModification>; 2],
    pub pred_weight_table: Option<PredWeightTable>,
    /// None for non-reference pictures (nal_ref_idc == 0).
    pub dec_ref_pic_marking: Option<DecRefPicMarking>,
    pub cabac_init_idc: u8,
    pub slice_qp_delta: i32,
    pub sp_for_switch: bool,
    pub slice_qs_delta: i32,
    pub disable_deblocking_filter_idc: u8,
    pub slice_alpha_c0_offset_div2: i8,
    pub slice_beta_offset_div2: i8,
    pub slice_group_change_cycle: u32,
    /// Bit offset of slice_data() within the RBSP.
    pub data_bit_offset: usize,
}

impl SliceHeader {
    /// Parses a slice header from the RBSP following the NAL header.
    /// `lookup` maps pic_parameter_set_id to the parameter sets that apply
    /// for this NAL unit (the subset SPS for non-base views).
    pub fn parse<'a>(
        rbsp: &[u8],
        nal: &NalHeader,
        lookup: impl FnOnce(u8) -> Option<(&'a Sps, &'a Pps)>,
    ) -> Result<Self, Error> {
        let mut r = BitReader::new(rbsp);
        let first_mb_in_slice = r.ue()?;
        let raw_type = r.ue_max("slice_type", 9)?;
        let slice_type = SliceType::from_ue(raw_type);
        let pps_id = r.ue_max("pic_parameter_set_id", 255)? as u8;
        let (sps, pps) = lookup(pps_id).ok_or(Error::Invalid("missing parameter set"))?;
        let mvc_ext = nal.nal_type.has_extension();
        let idr = match nal.mvc {
            Some(m) if mvc_ext => !m.non_idr,
            _ => nal.nal_type == NalType::SliceIdr,
        };
        if idr && slice_type.num_lists() > 0 && !mvc_ext {
            return Err(Error::Invalid("inter slice in IDR picture"));
        }

        let colour_plane_id = if sps.separate_colour_plane {
            r.u(2) as u8
        } else {
            0
        };
        if colour_plane_id > 2 {
            return Err(Error::OutOfRange {
                field: "colour_plane_id",
                value: 3,
            });
        }
        let frame_num = r.u(sps.log2_max_frame_num as u32);
        let (mut field_pic, mut bottom_field) = (false, false);
        if !sps.frame_mbs_only {
            field_pic = r.flag();
            if field_pic {
                bottom_field = r.flag();
            }
        }
        let mbaff = sps.mb_adaptive_frame_field && !field_pic;
        let pic_size_in_mbs =
            sps.pic_width_in_mbs * sps.frame_height_in_mbs() / (1 + field_pic as u32);
        if first_mb_in_slice * (1 + mbaff as u32) >= pic_size_in_mbs {
            return Err(Error::OutOfRange {
                field: "first_mb_in_slice",
                value: first_mb_in_slice as i64,
            });
        }
        let idr_pic_id = if idr {
            r.ue_max("idr_pic_id", 65535)?
        } else {
            0
        };

        let (mut pic_order_cnt_lsb, mut delta_pic_order_cnt_bottom, mut delta_pic_order_cnt) =
            (0, 0, [0; 2]);
        let bottom_present = pps.bottom_field_pic_order_in_frame_present && !field_pic;
        match &sps.poc {
            PicOrderCnt::Type0 { log2_max_lsb } => {
                pic_order_cnt_lsb = r.u(*log2_max_lsb as u32);
                if bottom_present {
                    delta_pic_order_cnt_bottom =
                        r.se_range("delta_pic_order_cnt_bottom", -i32::MAX, i32::MAX)?;
                }
            }
            PicOrderCnt::Type1 {
                delta_always_zero: false,
                ..
            } => {
                delta_pic_order_cnt[0] =
                    r.se_range("delta_pic_order_cnt[0]", -i32::MAX, i32::MAX)?;
                if bottom_present {
                    delta_pic_order_cnt[1] =
                        r.se_range("delta_pic_order_cnt[1]", -i32::MAX, i32::MAX)?;
                }
            }
            _ => {}
        }
        let redundant_pic_cnt = if pps.redundant_pic_cnt_present {
            r.ue_max("redundant_pic_cnt", 127)?
        } else {
            0
        };
        let direct_spatial_mv_pred = slice_type == SliceType::B && r.flag();

        let mut num_ref_idx_active = [0; 2];
        let lists = slice_type.num_lists();
        if lists > 0 {
            num_ref_idx_active = pps.num_ref_idx_default_active;
            if r.flag() {
                // 0..15 for frames, 0..31 for fields (7.4.3).
                let max = if field_pic { 31 } else { 15 };
                num_ref_idx_active[0] = r.ue_max("num_ref_idx_l0_active_minus1", max)? + 1;
                if lists == 2 {
                    num_ref_idx_active[1] = r.ue_max("num_ref_idx_l1_active_minus1", max)? + 1;
                }
            }
            if lists == 1 {
                num_ref_idx_active[1] = 0;
            }
        }

        let mut ref_pic_list_modification = [Vec::new(), Vec::new()];
        for list in ref_pic_list_modification.iter_mut().take(lists) {
            if r.flag() {
                loop {
                    let op = match r.ue()? {
                        0 => RefPicListModification::ShortTermSub(r.ue()?),
                        1 => RefPicListModification::ShortTermAdd(r.ue()?),
                        2 => RefPicListModification::LongTerm(r.ue()?),
                        3 => break,
                        4 if mvc_ext => RefPicListModification::InterViewSub(r.ue()?),
                        5 if mvc_ext => RefPicListModification::InterViewAdd(r.ue()?),
                        v => {
                            return Err(Error::OutOfRange {
                                field: "modification_of_pic_nums_idc",
                                value: v as i64,
                            });
                        }
                    };
                    // Each list can be reordered at most num_ref_idx_active + 1 times.
                    if list.len() > 32 {
                        return Err(Error::Invalid("ref_pic_list_modification length"));
                    }
                    list.push(op);
                }
            }
        }

        let weighted = (pps.weighted_pred && matches!(slice_type, SliceType::P | SliceType::Sp))
            || (pps.weighted_bipred_idc == 1 && slice_type == SliceType::B);
        let pred_weight_table = if weighted {
            Some(pred_weight_table(
                &mut r,
                sps,
                &num_ref_idx_active[..lists],
            )?)
        } else {
            None
        };

        let dec_ref_pic_marking = if nal.ref_idc != 0 {
            Some(if idr {
                DecRefPicMarking::Idr {
                    no_output_of_prior_pics: r.flag(),
                    long_term_reference: r.flag(),
                }
            } else if r.flag() {
                let mut ops = Vec::new();
                loop {
                    ops.push(match r.ue()? {
                        0 => break,
                        1 => Mmco::ForgetShort(r.ue()?),
                        2 => Mmco::ForgetLong(r.ue()?),
                        3 => Mmco::ShortToLong(r.ue()?, r.ue()?),
                        4 => Mmco::MaxLongTermIdx(r.ue()?),
                        5 => Mmco::ForgetAll,
                        6 => Mmco::CurrentToLong(r.ue()?),
                        v => {
                            return Err(Error::OutOfRange {
                                field: "memory_management_control_operation",
                                value: v as i64,
                            });
                        }
                    });
                    if ops.len() > 66 {
                        return Err(Error::Invalid(
                            "too many memory_management_control_operations",
                        ));
                    }
                }
                DecRefPicMarking::Adaptive(ops)
            } else {
                DecRefPicMarking::SlidingWindow
            })
        } else {
            None
        };

        let cabac_init_idc = if pps.entropy_coding_mode && !slice_type.is_intra() {
            r.ue_max("cabac_init_idc", 2)? as u8
        } else {
            0
        };
        let qp_bd_offset = 6 * (sps.bit_depth_luma as i32 - 8);
        let slice_qp_delta = r.se()?;
        let qp = 26 + pps.pic_init_qp_minus26 + slice_qp_delta;
        if !(-qp_bd_offset..=51).contains(&qp) {
            return Err(Error::OutOfRange {
                field: "SliceQPY",
                value: qp as i64,
            });
        }
        let (mut sp_for_switch, mut slice_qs_delta) = (false, 0);
        if matches!(slice_type, SliceType::Sp | SliceType::Si) {
            sp_for_switch = slice_type == SliceType::Sp && r.flag();
            slice_qs_delta = r.se()?;
            let qs = 26 + pps.pic_init_qs_minus26 + slice_qs_delta;
            if !(0..=51).contains(&qs) {
                return Err(Error::OutOfRange {
                    field: "QSY",
                    value: qs as i64,
                });
            }
        }
        let (mut disable_deblocking_filter_idc, mut alpha, mut beta) = (0, 0, 0);
        if pps.deblocking_filter_control_present {
            disable_deblocking_filter_idc = r.ue_max("disable_deblocking_filter_idc", 2)? as u8;
            if disable_deblocking_filter_idc != 1 {
                alpha = r.se_range("slice_alpha_c0_offset_div2", -6, 6)? as i8;
                beta = r.se_range("slice_beta_offset_div2", -6, 6)? as i8;
            }
        }
        let mut slice_group_change_cycle = 0;
        if let Some(SliceGroupMap::Changing { change_rate, .. }) = &pps.slice_group_map {
            let pic_size_in_map_units = sps.pic_width_in_mbs * sps.pic_height_in_map_units;
            let n = pic_size_in_map_units.div_ceil(*change_rate) + 1;
            let bits = 32 - (n - 1).leading_zeros();
            slice_group_change_cycle = r.u(bits);
        }
        if r.overrun() {
            return Err(Error::Truncated);
        }
        Ok(Self {
            first_mb_in_slice,
            slice_type,
            slice_type_fixed: raw_type >= 5,
            pps_id,
            colour_plane_id,
            frame_num,
            field_pic,
            bottom_field,
            idr,
            idr_pic_id,
            pic_order_cnt_lsb,
            delta_pic_order_cnt_bottom,
            delta_pic_order_cnt,
            redundant_pic_cnt,
            direct_spatial_mv_pred,
            num_ref_idx_active,
            ref_pic_list_modification,
            pred_weight_table,
            dec_ref_pic_marking,
            cabac_init_idc,
            slice_qp_delta,
            sp_for_switch,
            slice_qs_delta,
            disable_deblocking_filter_idc,
            slice_alpha_c0_offset_div2: alpha,
            slice_beta_offset_div2: beta,
            slice_group_change_cycle,
            data_bit_offset: r.pos(),
        })
    }
}

fn pred_weight_table(
    r: &mut BitReader,
    sps: &Sps,
    num_ref_idx_active: &[u32],
) -> Result<PredWeightTable, Error> {
    let luma_log2_denom = r.ue_max("luma_log2_weight_denom", 7)? as u8;
    let has_chroma = sps.chroma_array_type() != 0;
    let chroma_log2_denom = if has_chroma {
        r.ue_max("chroma_log2_weight_denom", 7)? as u8
    } else {
        0
    };
    // Offsets are in units of 1 << (BitDepth - 8) for high bit depth (7.4.3.2);
    // the coded range is -128..127 either way.
    let mut lists = [Vec::new(), Vec::new()];
    for (l, &n) in num_ref_idx_active.iter().enumerate() {
        for _ in 0..n {
            let mut w = Weights {
                luma: (1 << luma_log2_denom, 0),
                chroma: [(1 << chroma_log2_denom, 0); 2],
                luma_flag: false,
                chroma_flag: false,
            };
            if r.flag() {
                w.luma_flag = true;
                w.luma = (
                    r.se_range("luma_weight", -128, 127)? as i16,
                    r.se_range("luma_offset", -128, 127)? as i16,
                );
            }
            if has_chroma && r.flag() {
                w.chroma_flag = true;
                for c in &mut w.chroma {
                    *c = (
                        r.se_range("chroma_weight", -128, 127)? as i16,
                        r.se_range("chroma_offset", -128, 127)? as i16,
                    );
                }
            }
            lists[l].push(w);
        }
    }
    Ok(PredWeightTable {
        luma_log2_denom,
        chroma_log2_denom,
        lists,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::writer::BitWriter;
    use crate::nal::MvcExtension;

    fn sps_pps() -> (Sps, Pps) {
        let mut w = BitWriter::default();
        // High profile, 1920x1088, poc type 0 with 8-bit lsb, 4 ref frames.
        w.u(8, 100)
            .u(8, 0)
            .u(8, 41)
            .ue(0)
            .ue(1)
            .ue(0)
            .ue(0)
            .flag(false)
            .flag(false);
        w.ue(0)
            .ue(0)
            .ue(4)
            .ue(4)
            .flag(false)
            .ue(119)
            .ue(67)
            .flag(true)
            .flag(true)
            .flag(false)
            .flag(false)
            .trailing();
        let sps = Sps::parse(&w.bytes).unwrap();
        let mut w = BitWriter::default();
        // CABAC, bipred weights implicit, deblocking control present.
        w.ue(0)
            .ue(0)
            .flag(true)
            .flag(false)
            .ue(0)
            .ue(3)
            .ue(0)
            .flag(true)
            .u(2, 2);
        w.se(0)
            .se(0)
            .se(0)
            .flag(true)
            .flag(false)
            .flag(false)
            .trailing();
        (sps, Pps::parse(&w.bytes).unwrap())
    }

    fn parse(bytes: &[u8], nal: NalHeader) -> Result<SliceHeader, Error> {
        let (sps, pps) = sps_pps();
        SliceHeader::parse(bytes, &nal, |_| Some((&sps, &pps)))
    }

    fn nal(nal_type: NalType, ref_idc: u8) -> NalHeader {
        NalHeader {
            ref_idc,
            nal_type,
            mvc: None,
            len: 1,
        }
    }

    #[test]
    fn parses_p_slice_with_weights_and_mmco() {
        let mut w = BitWriter::default();
        w.ue(120).ue(5).ue(0).u(4, 3).u(8, 6); // first_mb, P (all), pps 0, frame_num 3, poc lsb 6
        w.flag(true).ue(1); // override: 2 refs
        w.flag(true).ue(0).ue(2).ue(3); // l0 modification: ShortTermSub(2)
        w.ue(5).ue(0); // weights: luma denom 5, chroma denom 0
        w.flag(true).se(40).se(-3).flag(false); // ref 0 luma weight
        w.flag(false).flag(true).se(1).se(2).se(-1).se(0); // ref 1 chroma weights
        w.flag(true).ue(1).ue(0).ue(0); // adaptive marking: ForgetShort(0)
        w.ue(1).se(-2).ue(0).se(1).se(-1); // cabac_init_idc, qp delta, deblocking
        let data_start = w.bytes.len();
        w.u(8, 0xaa);
        let h = parse(&w.bytes, nal(NalType::Slice, 2)).unwrap();
        assert_eq!(
            (
                h.first_mb_in_slice,
                h.slice_type,
                h.slice_type_fixed,
                h.frame_num,
                h.pic_order_cnt_lsb
            ),
            (120, SliceType::P, true, 3, 6)
        );
        assert_eq!(h.num_ref_idx_active, [2, 0]);
        assert_eq!(
            h.ref_pic_list_modification[0],
            [RefPicListModification::ShortTermSub(2)]
        );
        let wt = h.pred_weight_table.unwrap();
        assert_eq!(
            (wt.luma_log2_denom, wt.lists[0][0].luma, wt.lists[0][1].luma),
            (5, (40, -3), (32, 0))
        );
        assert_eq!(wt.lists[0][1].chroma, [(1, 2), (-1, 0)]);
        assert_eq!(
            h.dec_ref_pic_marking,
            Some(DecRefPicMarking::Adaptive(vec![Mmco::ForgetShort(0)]))
        );
        assert_eq!(
            (
                h.cabac_init_idc,
                h.slice_qp_delta,
                h.slice_alpha_c0_offset_div2,
                h.slice_beta_offset_div2
            ),
            (1, -2, 1, -1)
        );
        assert!(h.data_bit_offset <= data_start * 8);
    }

    #[test]
    fn parses_mvc_b_slice_with_inter_view_modification() {
        let mut w = BitWriter::default();
        w.ue(0).ue(1).ue(0).u(4, 0).ue(0).u(8, 0); // B, IDR in view 1 (idr_pic_id 0)
        w.flag(true).flag(false); // direct_spatial, no override
        w.flag(true).ue(5).ue(0).ue(3); // l0: InterViewAdd(0)
        w.flag(false);
        w.flag(false).flag(false); // IDR marking
        w.ue(0).se(0).ue(1);
        let mut n = nal(NalType::SliceExtension, 3);
        n.mvc = Some(MvcExtension {
            non_idr: false,
            priority_id: 0,
            view_id: 1,
            temporal_id: 0,
            anchor_pic: true,
            inter_view: false,
        });
        n.len = 4;
        let h = parse(&w.bytes, n).unwrap();
        assert!(h.idr && h.direct_spatial_mv_pred);
        assert_eq!(h.num_ref_idx_active, [4, 1]);
        assert_eq!(
            h.ref_pic_list_modification[0],
            [RefPicListModification::InterViewAdd(0)]
        );
        assert_eq!(h.disable_deblocking_filter_idc, 1);
        // idc 5 is only legal in MVC slices.
        let mut w2 = BitWriter::default();
        w2.ue(0)
            .ue(1)
            .ue(0)
            .u(4, 0)
            .u(8, 0)
            .flag(true)
            .flag(false)
            .flag(true)
            .ue(5)
            .ue(0)
            .ue(3)
            .flag(false)
            .ue(0)
            .se(0)
            .ue(1);
        assert!(matches!(
            parse(&w2.bytes, nal(NalType::Slice, 0)),
            Err(Error::OutOfRange {
                field: "modification_of_pic_nums_idc",
                ..
            })
        ));
    }

    #[test]
    fn rejects_out_of_range_first_mb() {
        let mut w = BitWriter::default();
        w.ue(120 * 68)
            .ue(7)
            .ue(0)
            .u(4, 0)
            .ue(0)
            .u(8, 0)
            .flag(false)
            .flag(false)
            .se(0)
            .ue(1);
        assert!(matches!(
            parse(&w.bytes, nal(NalType::SliceIdr, 3)),
            Err(Error::OutOfRange {
                field: "first_mb_in_slice",
                ..
            })
        ));
    }
}
