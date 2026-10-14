//! Parameter sets: SPS (7.3.2.1), subset SPS with the MVC extension
//! (7.3.2.1.3, H.7.3.2.1.4), PPS (7.3.2.2), VUI and HRD (E.1).

use crate::Error;
use crate::bits::BitReader;

/// Profiles whose SPS carries chroma format, bit depth and scaling matrices.
fn has_chroma_info(profile_idc: u8) -> bool {
    matches!(
        profile_idc,
        100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135
    )
}

pub const PROFILE_MULTIVIEW_HIGH: u8 = 118;
pub const PROFILE_STEREO_HIGH: u8 = 128;

// Table 7-3 and 7-4, in zigzag scan order.
pub const DEFAULT_4X4_INTRA: [u8; 16] = [
    6, 13, 13, 20, 20, 20, 28, 28, 28, 28, 32, 32, 32, 37, 37, 42,
];
pub const DEFAULT_4X4_INTER: [u8; 16] = [
    10, 14, 14, 20, 20, 20, 24, 24, 24, 24, 27, 27, 27, 30, 30, 34,
];
#[rustfmt::skip]
pub const DEFAULT_8X8_INTRA: [u8; 64] = [
    6, 10, 10, 13, 11, 13, 16, 16, 16, 16, 18, 18, 18, 18, 18, 23,
    23, 23, 23, 23, 23, 25, 25, 25, 25, 25, 25, 25, 27, 27, 27, 27,
    27, 27, 27, 27, 29, 29, 29, 29, 29, 29, 29, 31, 31, 31, 31, 31,
    31, 33, 33, 33, 33, 33, 36, 36, 36, 36, 38, 38, 38, 40, 40, 42,
];
#[rustfmt::skip]
pub const DEFAULT_8X8_INTER: [u8; 64] = [
    9, 13, 13, 15, 13, 15, 17, 17, 17, 17, 19, 19, 19, 19, 19, 21,
    21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 24, 24, 24, 24,
    24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 27, 27, 27, 27, 27,
    27, 28, 28, 28, 28, 28, 30, 30, 30, 30, 32, 32, 32, 33, 33, 35,
];

/// Resolved scaling matrices, zigzag order: 6 4x4 lists (Y/Cb/Cr intra,
/// Y/Cb/Cr inter) and 6 8x8 lists (Y intra, Y inter, Cb intra, Cb inter,
/// Cr intra, Cr inter). Only the first two 8x8 lists are used below 4:4:4.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScalingMatrices {
    pub list4x4: [[u8; 16]; 6],
    pub list8x8: [[u8; 64]; 6],
}

impl ScalingMatrices {
    pub const FLAT: Self = Self {
        list4x4: [[16; 16]; 6],
        list8x8: [[16; 64]; 6],
    };
}

/// Parsed but unresolved scaling_list() entries: None when not present,
/// Some(None) for "use default" (useDefaultScalingMatrixFlag), else the list.
type RawList<const N: usize> = Option<Option<[u8; N]>>;

// Indexed like the spec pseudocode; entries refer back to earlier ones.
#[allow(clippy::needless_range_loop)]
fn scaling_list<const N: usize>(r: &mut BitReader) -> Result<Option<[u8; N]>, Error> {
    let mut list = [0u8; N];
    let (mut last, mut next) = (8i32, 8i32);
    for j in 0..N {
        if next != 0 {
            let delta = r.se_range("delta_scale", -128, 127)?;
            next = (last + delta + 256) % 256;
            if j == 0 && next == 0 {
                return Ok(None);
            }
        }
        list[j] = if next == 0 { last } else { next } as u8;
        last = list[j] as i32;
    }
    Ok(Some(list))
}

/// Reads `count` scaling_list_present_flag/scaling_list() pairs.
fn raw_lists(
    r: &mut BitReader,
    count: usize,
) -> Result<([RawList<16>; 6], [RawList<64>; 6]), Error> {
    let mut l4: [RawList<16>; 6] = [None; 6];
    let mut l8: [RawList<64>; 6] = [None; 6];
    for i in 0..count {
        if r.flag() {
            if i < 6 {
                l4[i] = Some(scaling_list::<16>(r)?);
            } else {
                l8[i - 6] = Some(scaling_list::<64>(r)?);
            }
        }
    }
    Ok((l4, l8))
}

/// Applies fall-back rule A (`base` = None) or B (`base` = SPS lists), Table 7-2.
// Indexed like the spec pseudocode; entries refer back to earlier ones.
#[allow(clippy::needless_range_loop)]
fn resolve(
    l4: &[RawList<16>; 6],
    l8: &[RawList<64>; 6],
    base: Option<&ScalingMatrices>,
) -> ScalingMatrices {
    let mut m = ScalingMatrices::FLAT;
    for i in 0..6 {
        let default = if i < 3 {
            DEFAULT_4X4_INTRA
        } else {
            DEFAULT_4X4_INTER
        };
        m.list4x4[i] = match l4[i] {
            Some(Some(l)) => l,
            Some(None) => default,
            None if i == 0 || i == 3 => base.map_or(default, |b| b.list4x4[i]),
            None => m.list4x4[i - 1],
        };
    }
    for i in 0..6 {
        let default = if i % 2 == 0 {
            DEFAULT_8X8_INTRA
        } else {
            DEFAULT_8X8_INTER
        };
        m.list8x8[i] = match l8[i] {
            Some(Some(l)) => l,
            Some(None) => default,
            None if i < 2 => base.map_or(default, |b| b.list8x8[i]),
            None => m.list8x8[i - 2],
        };
    }
    m
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Hrd {
    pub cpb_cnt: u32,
    pub bit_rate_scale: u8,
    pub cpb_size_scale: u8,
    /// (bit_rate_value_minus1, cpb_size_value_minus1, cbr_flag) per CPB.
    pub cpbs: Vec<(u32, u32, bool)>,
    pub initial_cpb_removal_delay_length: u8,
    pub cpb_removal_delay_length: u8,
    pub dpb_output_delay_length: u8,
    pub time_offset_length: u8,
}

fn hrd(r: &mut BitReader) -> Result<Hrd, Error> {
    let cpb_cnt = r.ue_max("cpb_cnt_minus1", 31)? + 1;
    let bit_rate_scale = r.u(4) as u8;
    let cpb_size_scale = r.u(4) as u8;
    let mut cpbs = Vec::with_capacity(cpb_cnt as usize);
    for _ in 0..cpb_cnt {
        let br = r.ue_max("bit_rate_value_minus1", u32::MAX - 1)?;
        let cs = r.ue_max("cpb_size_value_minus1", u32::MAX - 1)?;
        cpbs.push((br, cs, r.flag()));
    }
    Ok(Hrd {
        cpb_cnt,
        bit_rate_scale,
        cpb_size_scale,
        cpbs,
        initial_cpb_removal_delay_length: r.u(5) as u8 + 1,
        cpb_removal_delay_length: r.u(5) as u8 + 1,
        dpb_output_delay_length: r.u(5) as u8 + 1,
        time_offset_length: r.u(5) as u8,
    })
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Timing {
    pub num_units_in_tick: u32,
    pub time_scale: u32,
    pub fixed_frame_rate: bool,
}

fn timing(r: &mut BitReader) -> Timing {
    Timing {
        num_units_in_tick: r.u(32),
        time_scale: r.u(32),
        fixed_frame_rate: r.flag(),
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BitstreamRestriction {
    pub motion_vectors_over_pic_boundaries: bool,
    pub max_bytes_per_pic_denom: u32,
    pub max_bits_per_mb_denom: u32,
    pub log2_max_mv_length_horizontal: u32,
    pub log2_max_mv_length_vertical: u32,
    pub max_num_reorder_frames: u32,
    pub max_dec_frame_buffering: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Vui {
    pub aspect_ratio_idc: Option<u8>,
    pub sar: Option<(u16, u16)>,
    pub overscan_appropriate: Option<bool>,
    pub video_format: u8,
    pub video_full_range: bool,
    /// (colour_primaries, transfer_characteristics, matrix_coefficients)
    pub colour_description: Option<(u8, u8, u8)>,
    pub chroma_sample_loc: Option<(u32, u32)>,
    pub timing: Option<Timing>,
    pub nal_hrd: Option<Hrd>,
    pub vcl_hrd: Option<Hrd>,
    pub low_delay_hrd: bool,
    pub pic_struct_present: bool,
    pub bitstream_restriction: Option<BitstreamRestriction>,
}

fn vui(r: &mut BitReader) -> Result<Vui, Error> {
    let mut v = Vui {
        video_format: 5,
        ..Default::default()
    };
    if r.flag() {
        let idc = r.u(8) as u8;
        v.aspect_ratio_idc = Some(idc);
        if idc == 255 {
            v.sar = Some((r.u(16) as u16, r.u(16) as u16));
        }
    }
    if r.flag() {
        v.overscan_appropriate = Some(r.flag());
    }
    if r.flag() {
        v.video_format = r.u(3) as u8;
        v.video_full_range = r.flag();
        if r.flag() {
            v.colour_description = Some((r.u(8) as u8, r.u(8) as u8, r.u(8) as u8));
        }
    }
    if r.flag() {
        v.chroma_sample_loc = Some((
            r.ue_max("chroma_sample_loc_type_top_field", 5)?,
            r.ue_max("chroma_sample_loc_type_bottom_field", 5)?,
        ));
    }
    if r.flag() {
        v.timing = Some(timing(r));
    }
    if r.flag() {
        v.nal_hrd = Some(hrd(r)?);
    }
    if r.flag() {
        v.vcl_hrd = Some(hrd(r)?);
    }
    if v.nal_hrd.is_some() || v.vcl_hrd.is_some() {
        v.low_delay_hrd = r.flag();
    }
    v.pic_struct_present = r.flag();
    if r.flag() {
        v.bitstream_restriction = Some(BitstreamRestriction {
            motion_vectors_over_pic_boundaries: r.flag(),
            max_bytes_per_pic_denom: r.ue_max("max_bytes_per_pic_denom", 16)?,
            max_bits_per_mb_denom: r.ue_max("max_bits_per_mb_denom", 16)?,
            // E.2.1 allows 0..15 here (16 only in the MVC/SVC VUI); accept 16
            // since it is informational and some encoders write it.
            log2_max_mv_length_horizontal: r.ue_max("log2_max_mv_length_horizontal", 16)?,
            log2_max_mv_length_vertical: r.ue_max("log2_max_mv_length_vertical", 16)?,
            max_num_reorder_frames: r.ue()?,
            max_dec_frame_buffering: r.ue()?,
        });
    }
    Ok(v)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PicOrderCnt {
    Type0 {
        log2_max_lsb: u8,
    },
    Type1 {
        delta_always_zero: bool,
        offset_for_non_ref_pic: i32,
        offset_for_top_to_bottom_field: i32,
        offset_for_ref_frame: Vec<i32>,
    },
    Type2,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sps {
    pub profile_idc: u8,
    /// constraint_set0..5 flags in bits 7..2, as coded.
    pub constraint_flags: u8,
    pub level_idc: u8,
    pub id: u8,
    pub chroma_format_idc: u8,
    pub separate_colour_plane: bool,
    pub bit_depth_luma: u8,
    pub bit_depth_chroma: u8,
    pub qpprime_y_zero_transform_bypass: bool,
    /// Present only when seq_scaling_matrix_present_flag is set; else Flat_4x4/Flat_8x8.
    pub scaling: Option<ScalingMatrices>,
    pub log2_max_frame_num: u8,
    pub poc: PicOrderCnt,
    pub max_num_ref_frames: u32,
    pub gaps_in_frame_num_allowed: bool,
    pub pic_width_in_mbs: u32,
    pub pic_height_in_map_units: u32,
    pub frame_mbs_only: bool,
    pub mb_adaptive_frame_field: bool,
    pub direct_8x8_inference: bool,
    /// (left, right, top, bottom) in crop units.
    pub crop: Option<[u32; 4]>,
    pub vui: Option<Vui>,
    /// Parsing ended exactly on rbsp_trailing_bits(). False suggests a parser
    /// bug or a malformed stream; decoding proceeds either way.
    pub trailing_ok: bool,
}

impl Sps {
    /// Parses seq_parameter_set_rbsp() from an RBSP (header byte removed).
    pub fn parse(rbsp: &[u8]) -> Result<Self, Error> {
        let mut r = BitReader::new(rbsp);
        let mut sps = Self::parse_data(&mut r)?;
        if r.overrun() {
            return Err(Error::Truncated);
        }
        sps.trailing_ok = r.at_trailing_bits();
        Ok(sps)
    }

    /// seq_parameter_set_data(), shared with the subset SPS.
    fn parse_data(r: &mut BitReader) -> Result<Self, Error> {
        let profile_idc = r.u(8) as u8;
        let constraint_flags = r.u(8) as u8 & 0xfc;
        let level_idc = r.u(8) as u8;
        let id = r.ue_max("seq_parameter_set_id", 31)? as u8;
        let (mut chroma_format_idc, mut separate_colour_plane) = (1, false);
        let (mut bit_depth_luma, mut bit_depth_chroma) = (8, 8);
        let mut qpprime_y_zero_transform_bypass = false;
        let mut scaling = None;
        if has_chroma_info(profile_idc) {
            chroma_format_idc = r.ue_max("chroma_format_idc", 3)? as u8;
            if chroma_format_idc == 3 {
                separate_colour_plane = r.flag();
            }
            bit_depth_luma = r.ue_max("bit_depth_luma_minus8", 6)? as u8 + 8;
            bit_depth_chroma = r.ue_max("bit_depth_chroma_minus8", 6)? as u8 + 8;
            qpprime_y_zero_transform_bypass = r.flag();
            if r.flag() {
                let count = if chroma_format_idc != 3 { 8 } else { 12 };
                let (l4, l8) = raw_lists(r, count)?;
                scaling = Some(resolve(&l4, &l8, None));
            }
        }
        let log2_max_frame_num = r.ue_max("log2_max_frame_num_minus4", 12)? as u8 + 4;
        let poc = match r.ue_max("pic_order_cnt_type", 2)? {
            0 => PicOrderCnt::Type0 {
                log2_max_lsb: r.ue_max("log2_max_pic_order_cnt_lsb_minus4", 12)? as u8 + 4,
            },
            1 => {
                let delta_always_zero = r.flag();
                let offset_for_non_ref_pic =
                    r.se_range("offset_for_non_ref_pic", -i32::MAX, i32::MAX)?;
                let offset_for_top_to_bottom_field =
                    r.se_range("offset_for_top_to_bottom_field", -i32::MAX, i32::MAX)?;
                let n = r.ue_max("num_ref_frames_in_pic_order_cnt_cycle", 255)?;
                let offset_for_ref_frame = (0..n)
                    .map(|_| r.se_range("offset_for_ref_frame", -i32::MAX, i32::MAX))
                    .collect::<Result<_, _>>()?;
                PicOrderCnt::Type1 {
                    delta_always_zero,
                    offset_for_non_ref_pic,
                    offset_for_top_to_bottom_field,
                    offset_for_ref_frame,
                }
            }
            _ => PicOrderCnt::Type2,
        };
        let max_num_ref_frames = r.ue_max("max_num_ref_frames", 16)?;
        let gaps_in_frame_num_allowed = r.flag();
        // Bounded far above any level limit; real limits are checked at activation.
        let pic_width_in_mbs = r.ue_max("pic_width_in_mbs_minus1", 1 << 16)? + 1;
        let pic_height_in_map_units = r.ue_max("pic_height_in_map_units_minus1", 1 << 16)? + 1;
        let frame_mbs_only = r.flag();
        let mb_adaptive_frame_field = !frame_mbs_only && r.flag();
        let direct_8x8_inference = r.flag();
        let crop = if r.flag() {
            Some([r.ue()?, r.ue()?, r.ue()?, r.ue()?])
        } else {
            None
        };
        let vui = if r.flag() { Some(vui(r)?) } else { None };
        Ok(Self {
            profile_idc,
            constraint_flags,
            level_idc,
            id,
            chroma_format_idc,
            separate_colour_plane,
            bit_depth_luma,
            bit_depth_chroma,
            qpprime_y_zero_transform_bypass,
            scaling,
            log2_max_frame_num,
            poc,
            max_num_ref_frames,
            gaps_in_frame_num_allowed,
            pic_width_in_mbs,
            pic_height_in_map_units,
            frame_mbs_only,
            mb_adaptive_frame_field,
            direct_8x8_inference,
            crop,
            vui,
            trailing_ok: false,
        })
    }

    /// ChromaArrayType (7.4.2.1.1).
    pub fn chroma_array_type(&self) -> u8 {
        if self.separate_colour_plane {
            0
        } else {
            self.chroma_format_idc
        }
    }

    /// The chroma format of decoded pictures: chroma_format_idc, which
    /// differs from ChromaArrayType with separate colour planes.
    pub fn picture_chroma_type(&self) -> u8 {
        if self.separate_colour_plane {
            3
        } else {
            self.chroma_array_type()
        }
    }

    /// Colour planes decoded as separate monochrome pictures.
    pub fn colour_planes(&self) -> usize {
        if self.separate_colour_plane { 3 } else { 1 }
    }

    /// BitDepthY and BitDepthC.
    pub fn bit_depths(&self) -> [u32; 2] {
        [self.bit_depth_luma as u32, self.bit_depth_chroma as u32]
    }

    pub fn frame_height_in_mbs(&self) -> u32 {
        (2 - self.frame_mbs_only as u32) * self.pic_height_in_map_units
    }

    /// Cropped output size in luma samples (7.4.2.1.1, equations 7-19 to 7-22).
    pub fn cropped_size(&self) -> (u32, u32) {
        let (w, h) = (self.pic_width_in_mbs * 16, self.frame_height_in_mbs() * 16);
        let Some([l, r, t, b]) = self.crop else {
            return (w, h);
        };
        let (sub_w, sub_h) = match self.chroma_array_type() {
            1 => (2, 2),
            2 => (2, 1),
            _ => (1, 1),
        };
        let (cx, cy) = (sub_w, sub_h * (2 - self.frame_mbs_only as u32));
        (
            w.saturating_sub(cx * (l + r)),
            h.saturating_sub(cy * (t + b)),
        )
    }
}

/// One view's references in seq_parameter_set_mvc_extension().
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ViewRefs {
    pub anchor_l0: Vec<u16>,
    pub anchor_l1: Vec<u16>,
    pub non_anchor_l0: Vec<u16>,
    pub non_anchor_l1: Vec<u16>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OperationPoint {
    pub temporal_id: u8,
    pub target_view_ids: Vec<u16>,
    pub num_views: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LevelValue {
    pub level_idc: u8,
    pub operation_points: Vec<OperationPoint>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MvcVuiOp {
    pub temporal_id: u8,
    pub target_output_view_ids: Vec<u16>,
    pub timing: Option<Timing>,
    pub nal_hrd: Option<Hrd>,
    pub vcl_hrd: Option<Hrd>,
    pub low_delay_hrd: bool,
    pub pic_struct_present: bool,
}

/// seq_parameter_set_mvc_extension() (H.7.3.2.1.4).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MvcExtension {
    /// view_id per view order index.
    pub view_ids: Vec<u16>,
    /// Inter-view references per view order index (index 0 is always empty).
    pub refs: Vec<ViewRefs>,
    pub levels: Vec<LevelValue>,
    pub vui: Option<Vec<MvcVuiOp>>,
    /// mvc_vui_parameters_extension() was present but did not end on the RBSP
    /// trailing bits, so it was dropped. Seen on some Blu-ray 3D encoders,
    /// which insert stray bits after vui_mvc_timing_info_present_flag. Nothing
    /// in it affects decoding.
    pub vui_discarded: bool,
}

fn view_id_list(r: &mut BitReader, field: &'static str) -> Result<Vec<u16>, Error> {
    let n = r.ue_max(field, 15)?;
    (0..n)
        .map(|_| r.ue_max("view_id", 1023).map(|v| v as u16))
        .collect()
}

fn mvc_extension(r: &mut BitReader) -> Result<MvcExtension, Error> {
    let num_views = r.ue_max("num_views_minus1", 1023)? as usize + 1;
    let view_ids = (0..num_views)
        .map(|_| r.ue_max("view_id", 1023).map(|v| v as u16))
        .collect::<Result<Vec<_>, _>>()?;
    let mut refs = vec![ViewRefs::default(); num_views];
    for v in refs.iter_mut().skip(1) {
        v.anchor_l0 = view_id_list(r, "num_anchor_refs_l0")?;
        v.anchor_l1 = view_id_list(r, "num_anchor_refs_l1")?;
    }
    for v in refs.iter_mut().skip(1) {
        v.non_anchor_l0 = view_id_list(r, "num_non_anchor_refs_l0")?;
        v.non_anchor_l1 = view_id_list(r, "num_non_anchor_refs_l1")?;
    }
    let num_levels = r.ue_max("num_level_values_signalled_minus1", 63)? + 1;
    let mut levels = Vec::with_capacity(num_levels as usize);
    for _ in 0..num_levels {
        let level_idc = r.u(8) as u8;
        let num_ops = r.ue_max("num_applicable_ops_minus1", 1023)? + 1;
        let mut operation_points = Vec::with_capacity(num_ops as usize);
        for _ in 0..num_ops {
            let temporal_id = r.u(3) as u8;
            let n = r.ue_max("applicable_op_num_target_views_minus1", 1023)? + 1;
            let target_view_ids = (0..n)
                .map(|_| {
                    r.ue_max("applicable_op_target_view_id", 1023)
                        .map(|v| v as u16)
                })
                .collect::<Result<_, _>>()?;
            let num_views = r.ue_max("applicable_op_num_views_minus1", 1023)? + 1;
            operation_points.push(OperationPoint {
                temporal_id,
                target_view_ids,
                num_views,
            });
        }
        levels.push(LevelValue {
            level_idc,
            operation_points,
        });
    }
    Ok(MvcExtension {
        view_ids,
        refs,
        levels,
        vui: None,
        vui_discarded: false,
    })
}

/// mvc_vui_parameters_extension() (H.14.1).
fn mvc_vui(r: &mut BitReader) -> Result<Vec<MvcVuiOp>, Error> {
    let n = r.ue_max("vui_mvc_num_ops_minus1", 1023)? + 1;
    let mut ops = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let temporal_id = r.u(3) as u8;
        let k = r.ue_max("vui_mvc_num_target_output_views_minus1", 1023)? + 1;
        let target_output_view_ids = (0..k)
            .map(|_| r.ue_max("vui_mvc_view_id", 1023).map(|v| v as u16))
            .collect::<Result<_, _>>()?;
        let timing = if r.flag() { Some(timing(r)) } else { None };
        let nal_hrd = if r.flag() { Some(hrd(r)?) } else { None };
        let vcl_hrd = if r.flag() { Some(hrd(r)?) } else { None };
        let low_delay_hrd = (nal_hrd.is_some() || vcl_hrd.is_some()) && r.flag();
        let pic_struct_present = r.flag();
        ops.push(MvcVuiOp {
            temporal_id,
            target_output_view_ids,
            timing,
            nal_hrd,
            vcl_hrd,
            low_delay_hrd,
            pic_struct_present,
        });
    }
    Ok(ops)
}

/// subset_seq_parameter_set_rbsp() (7.3.2.1.3). Only the MVC branch is decoded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubsetSps {
    pub sps: Sps,
    pub mvc: MvcExtension,
}

impl SubsetSps {
    pub fn parse(rbsp: &[u8]) -> Result<Self, Error> {
        let mut r = BitReader::new(rbsp);
        let mut sps = Sps::parse_data(&mut r)?;
        match sps.profile_idc {
            PROFILE_MULTIVIEW_HIGH | PROFILE_STEREO_HIGH | 134 => {}
            83 | 86 => return Err(Error::Unsupported("SVC subset SPS")),
            138 | 135 | 139 => return Err(Error::Unsupported("MVC+depth / 3D-AVC subset SPS")),
            _ => {
                return Err(Error::OutOfRange {
                    field: "subset profile_idc",
                    value: sps.profile_idc as i64,
                });
            }
        }
        if !r.flag() {
            return Err(Error::Invalid("bit_equal_to_one"));
        }
        let mut mvc = mvc_extension(&mut r)?;
        if r.overrun() {
            return Err(Error::Truncated);
        }
        if r.flag() {
            let vui = mvc_vui(&mut r);
            // additional_extension2_data is ignored, as decoders must.
            let ends_cleanly = !r.overrun() && (r.flag() || r.at_trailing_bits());
            match vui {
                Ok(v) if ends_cleanly => mvc.vui = Some(v),
                _ => mvc.vui_discarded = true,
            }
            sps.trailing_ok = ends_cleanly;
        } else {
            let extension2 = r.flag();
            if r.overrun() {
                return Err(Error::Truncated);
            }
            sps.trailing_ok = extension2 || r.at_trailing_bits();
        }
        Ok(Self { sps, mvc })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SliceGroupMap {
    Interleaved {
        run_length: Vec<u32>,
    },
    Dispersed,
    Foreground {
        top_left: Vec<u32>,
        bottom_right: Vec<u32>,
    },
    /// Types 3 (box-out), 4 (raster) and 5 (wipe).
    Changing {
        map_type: u8,
        change_direction: bool,
        change_rate: u32,
    },
    Explicit {
        slice_group_id: Vec<u8>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pps {
    pub id: u8,
    pub sps_id: u8,
    pub entropy_coding_mode: bool,
    pub bottom_field_pic_order_in_frame_present: bool,
    pub num_slice_groups: u32,
    pub slice_group_map: Option<SliceGroupMap>,
    pub num_ref_idx_default_active: [u32; 2],
    pub weighted_pred: bool,
    pub weighted_bipred_idc: u8,
    pub pic_init_qp_minus26: i32,
    pub pic_init_qs_minus26: i32,
    pub chroma_qp_index_offset: i32,
    pub deblocking_filter_control_present: bool,
    pub constrained_intra_pred: bool,
    pub redundant_pic_cnt_present: bool,
    /// As [`Sps::trailing_ok`]; for a PPS with a tail see [`PpsExt::trailing_ok`].
    pub trailing_ok: bool,
    /// RBSP and bit position of the optional High-profile tail
    /// (transform_8x8_mode_flag onwards). Its length depends on the
    /// referenced SPS's chroma_format_idc, which in MVC may differ between the
    /// SPS and subset SPS sharing an id, so it is resolved per activation.
    tail: Option<(Vec<u8>, usize)>,
}

/// PPS fields that depend on the active SPS.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PpsExt {
    pub transform_8x8_mode: bool,
    pub scaling: ScalingMatrices,
    pub second_chroma_qp_index_offset: i32,
    pub trailing_ok: bool,
}

impl Pps {
    pub fn parse(rbsp: &[u8]) -> Result<Self, Error> {
        let mut r = BitReader::new(rbsp);
        let id = r.ue_max("pic_parameter_set_id", 255)? as u8;
        let sps_id = r.ue_max("seq_parameter_set_id", 31)? as u8;
        let entropy_coding_mode = r.flag();
        let bottom_field_pic_order_in_frame_present = r.flag();
        let num_slice_groups = r.ue_max("num_slice_groups_minus1", 7)? + 1;
        let slice_group_map = if num_slice_groups > 1 {
            Some(match r.ue_max("slice_group_map_type", 6)? {
                0 => SliceGroupMap::Interleaved {
                    run_length: (0..num_slice_groups)
                        .map(|_| r.ue().map(|v| v + 1))
                        .collect::<Result<_, _>>()?,
                },
                1 => SliceGroupMap::Dispersed,
                2 => {
                    let (mut top_left, mut bottom_right) = (Vec::new(), Vec::new());
                    for _ in 0..num_slice_groups - 1 {
                        top_left.push(r.ue()?);
                        bottom_right.push(r.ue()?);
                    }
                    SliceGroupMap::Foreground {
                        top_left,
                        bottom_right,
                    }
                }
                t @ 3..=5 => SliceGroupMap::Changing {
                    map_type: t as u8,
                    change_direction: r.flag(),
                    change_rate: r.ue()? + 1,
                },
                _ => {
                    // Level 6.2 allows 139264 macroblocks.
                    let n = r.ue_max("pic_size_in_map_units_minus1", 1 << 18)? as usize + 1;
                    let bits = 32 - (num_slice_groups - 1).leading_zeros();
                    SliceGroupMap::Explicit {
                        slice_group_id: (0..n).map(|_| r.u(bits) as u8).collect(),
                    }
                }
            })
        } else {
            None
        };
        let num_ref_idx_default_active = [
            r.ue_max("num_ref_idx_l0_default_active_minus1", 31)? + 1,
            r.ue_max("num_ref_idx_l1_default_active_minus1", 31)? + 1,
        ];
        let weighted_pred = r.flag();
        let weighted_bipred_idc = r.u(2) as u8;
        if weighted_bipred_idc == 3 {
            return Err(Error::OutOfRange {
                field: "weighted_bipred_idc",
                value: 3,
            });
        }
        // Lower bounds depend on bit depth (-(26 + QpBdOffsetY)); 14-bit gives -62.
        let pic_init_qp_minus26 = r.se_range("pic_init_qp_minus26", -62, 25)?;
        let pic_init_qs_minus26 = r.se_range("pic_init_qs_minus26", -26, 25)?;
        let chroma_qp_index_offset = r.se_range("chroma_qp_index_offset", -12, 12)?;
        let deblocking_filter_control_present = r.flag();
        let constrained_intra_pred = r.flag();
        let redundant_pic_cnt_present = r.flag();
        let tail = r.more_rbsp_data().then(|| (rbsp.to_vec(), r.pos()));
        let trailing_ok = tail.is_some() || r.at_trailing_bits();
        if r.overrun() {
            return Err(Error::Truncated);
        }
        Ok(Self {
            id,
            sps_id,
            entropy_coding_mode,
            bottom_field_pic_order_in_frame_present,
            num_slice_groups,
            slice_group_map,
            num_ref_idx_default_active,
            weighted_pred,
            weighted_bipred_idc,
            pic_init_qp_minus26,
            pic_init_qs_minus26,
            chroma_qp_index_offset,
            deblocking_filter_control_present,
            constrained_intra_pred,
            redundant_pic_cnt_present,
            trailing_ok,
            tail,
        })
    }

    /// Resolves the High-profile tail against the SPS it is activated with.
    pub fn ext(&self, sps: &Sps) -> Result<PpsExt, Error> {
        let sps_scaling = sps.scaling.clone().unwrap_or(ScalingMatrices::FLAT);
        let Some((rbsp, pos)) = &self.tail else {
            return Ok(PpsExt {
                transform_8x8_mode: false,
                scaling: sps_scaling,
                second_chroma_qp_index_offset: self.chroma_qp_index_offset,
                trailing_ok: self.trailing_ok,
            });
        };
        let mut r = BitReader::new(rbsp);
        r.skip(*pos);
        let transform_8x8_mode = r.flag();
        let scaling = if r.flag() {
            let count =
                6 + if sps.chroma_format_idc != 3 { 2 } else { 6 } * transform_8x8_mode as usize;
            let (l4, l8) = raw_lists(&mut r, count)?;
            // Rule A when the SPS has no matrices, rule B otherwise (7.4.2.2).
            resolve(&l4, &l8, sps.scaling.as_ref())
        } else {
            sps_scaling
        };
        let second_chroma_qp_index_offset = r.se_range("second_chroma_qp_index_offset", -12, 12)?;
        if r.overrun() {
            return Err(Error::Truncated);
        }
        Ok(PpsExt {
            transform_8x8_mode,
            scaling,
            second_chroma_qp_index_offset,
            trailing_ok: r.at_trailing_bits(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::writer::BitWriter;

    /// A 1080p High-profile SPS with cropping, a scaling matrix and VUI.
    fn high_sps(w: &mut BitWriter, profile: u8, id: u32) {
        w.u(8, profile as u64).u(8, 0).u(8, 41).ue(id);
        w.ue(1).ue(0).ue(0).flag(false); // 4:2:0, 8-bit
        w.flag(true); // seq_scaling_matrix_present_flag
        w.flag(true).se(-2).se(-6); // list 0: 6, then nextScale 0 repeats 6
        for _ in 1..8 {
            w.flag(false);
        }
        w.ue(0).ue(0).ue(4); // log2_max_frame_num 4, poc type 0, lsb 8 bits
        w.ue(4).flag(false).ue(119).ue(67).flag(true).flag(true);
        w.flag(true).ue(0).ue(0).ue(0).ue(4); // crop bottom 8 lines
        w.flag(true); // vui
        w.flag(true)
            .u(8, 1)
            .flag(false)
            .flag(false)
            .flag(false)
            .flag(true)
            .u(32, 1001)
            .u(32, 48000)
            .flag(false);
        w.flag(false)
            .flag(false)
            .flag(false)
            .flag(true)
            .flag(true)
            .ue(0)
            .ue(0)
            .ue(15)
            .ue(15)
            .ue(1)
            .ue(4);
    }

    #[test]
    fn parses_high_sps() {
        let mut w = BitWriter::default();
        high_sps(&mut w, 100, 0);
        w.trailing();
        let sps = Sps::parse(&w.bytes).unwrap();
        assert_eq!((sps.pic_width_in_mbs, sps.frame_height_in_mbs()), (120, 68));
        assert_eq!(sps.cropped_size(), (1920, 1080));
        assert_eq!(sps.poc, PicOrderCnt::Type0 { log2_max_lsb: 8 });
        let s = sps.scaling.as_ref().unwrap();
        assert_eq!(&s.list4x4[0][..3], &[6, 6, 6]);
        assert_eq!(s.list4x4[1], s.list4x4[0]); // rule A: falls back to previous list
        assert_eq!(s.list4x4[3], DEFAULT_4X4_INTER);
        assert_eq!(s.list8x8[0], DEFAULT_8X8_INTRA);
        assert!(sps.trailing_ok);
        let vui = sps.vui.unwrap();
        assert_eq!(vui.timing.unwrap().time_scale, 48000);
        assert_eq!(
            vui.bitstream_restriction.unwrap().max_dec_frame_buffering,
            4
        );
    }

    #[test]
    fn parses_stereo_high_subset_sps() {
        let mut w = BitWriter::default();
        high_sps(&mut w, PROFILE_STEREO_HIGH, 1);
        w.flag(true); // bit_equal_to_one
        w.ue(1).ue(0).ue(1); // two views: 0, 1
        w.ue(1).ue(0).ue(0); // view 1 anchor refs: l0 = [0], l1 = []
        w.ue(1).ue(0).ue(0); // view 1 non-anchor refs: l0 = [0], l1 = []
        w.ue(0).u(8, 41).ue(0).u(3, 0).ue(1).ue(0).ue(1).ue(1); // one level, one op with views 0 and 1
        w.flag(false).flag(false).trailing();
        let s = SubsetSps::parse(&w.bytes).unwrap();
        assert_eq!(s.sps.id, 1);
        assert!(s.sps.trailing_ok);
        assert_eq!(s.mvc.view_ids, [0, 1]);
        assert_eq!(s.mvc.refs[1].anchor_l0, [0]);
        assert_eq!(s.mvc.refs[1].non_anchor_l0, [0]);
        assert_eq!(s.mvc.levels[0].operation_points[0].target_view_ids, [0, 1]);
    }

    #[test]
    fn discards_nonconforming_mvc_vui() {
        let mut w = BitWriter::default();
        high_sps(&mut w, PROFILE_STEREO_HIGH, 1);
        w.flag(true).ue(1).ue(0).ue(1).ue(1).ue(0).ue(0).ue(0).ue(0);
        w.ue(0).u(8, 41).ue(0).u(3, 0).ue(1).ue(0).ue(1).ue(1);
        // mvc_vui with two stray bits after the timing flag, as some encoders write.
        w.flag(true)
            .ue(0)
            .u(3, 0)
            .ue(0)
            .ue(0)
            .flag(true)
            .u(2, 0)
            .u(32, 5005)
            .u(32, 240000)
            .flag(true);
        w.flag(false).flag(false).flag(true).flag(false).trailing();
        let s = SubsetSps::parse(&w.bytes).unwrap();
        assert!(s.mvc.vui_discarded && s.mvc.vui.is_none() && !s.sps.trailing_ok);
        assert_eq!(s.mvc.refs[1].anchor_l0, [0]);
    }

    #[test]
    fn pps_tail_resolves_per_sps() {
        let mut w = BitWriter::default();
        high_sps(&mut w, 100, 0);
        w.trailing();
        let sps = Sps::parse(&w.bytes).unwrap();

        let mut w = BitWriter::default();
        w.ue(3)
            .ue(0)
            .flag(true)
            .flag(false)
            .ue(0)
            .ue(2)
            .ue(0)
            .flag(true)
            .u(2, 2);
        w.se(-4).se(0).se(2).flag(true).flag(false).flag(false);
        w.flag(true).flag(true); // transform_8x8_mode, pic_scaling_matrix_present
        w.flag(false).flag(false).flag(false); // 4x4 lists 0-2 absent
        w.flag(true).se(-8); // list 3 coded as "use default" (nextScale 0 at j = 0)
        w.flag(false).flag(false).flag(false).flag(false); // lists 4-5 and both 8x8 absent
        w.se(-1).trailing();
        let pps = Pps::parse(&w.bytes).unwrap();
        assert_eq!(
            (
                pps.id,
                pps.num_ref_idx_default_active,
                pps.weighted_bipred_idc
            ),
            (3, [3, 1], 2)
        );
        let ext = pps.ext(&sps).unwrap();
        assert!(ext.transform_8x8_mode && ext.trailing_ok);
        // Rule B: list 0 absent -> SPS list 0; list 3 present as default.
        assert_eq!(
            ext.scaling.list4x4[0],
            sps.scaling.as_ref().unwrap().list4x4[0]
        );
        assert_eq!(ext.scaling.list4x4[3], DEFAULT_4X4_INTER);
        assert_eq!(
            ext.scaling.list8x8[1],
            sps.scaling.as_ref().unwrap().list8x8[1]
        );
        assert_eq!(ext.second_chroma_qp_index_offset, -1);
    }

    #[test]
    fn baseline_pps_without_tail() {
        let mut w = BitWriter::default();
        w.ue(0)
            .ue(0)
            .flag(false)
            .flag(false)
            .ue(0)
            .ue(0)
            .ue(0)
            .flag(false)
            .u(2, 0);
        w.se(0)
            .se(0)
            .se(3)
            .flag(true)
            .flag(false)
            .flag(false)
            .trailing();
        let pps = Pps::parse(&w.bytes).unwrap();
        let mut w = BitWriter::default();
        w.u(8, 66)
            .u(8, 0)
            .u(8, 30)
            .ue(0)
            .ue(0)
            .ue(2)
            .ue(1)
            .flag(false)
            .ue(10)
            .ue(8)
            .flag(true)
            .flag(true)
            .flag(false)
            .flag(false)
            .trailing();
        let ext = pps.ext(&Sps::parse(&w.bytes).unwrap()).unwrap();
        assert_eq!(
            ext,
            PpsExt {
                transform_8x8_mode: false,
                scaling: ScalingMatrices::FLAT,
                second_chroma_qp_index_offset: 3,
                trailing_ok: true
            }
        );
    }
}
