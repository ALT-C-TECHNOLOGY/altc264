//! Decoded picture buffer: reference marking (8.2.5), reference picture
//! lists (8.2.4) and output order (C.4), for frames and fields.
//!
//! The DPB holds frame stores: a frame-coded picture, a complementary
//! field pair, or a single field. Marking and order counts are kept per
//! field; a frame is a reference only when both its fields are.

use std::collections::VecDeque;
use std::sync::Arc;

use crate::picture::{Parity, Picture, PlaneView};
use crate::progress::PicState;
use crate::slice::{DecRefPicMarking, Mmco, RefPicListModification};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Marking {
    Unused,
    Short,
    Long(u32),
}

/// What a coded picture is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PicStructure {
    Frame,
    Field(Parity),
}

impl PicStructure {
    pub fn parity(self) -> Option<Parity> {
        match self {
            PicStructure::Frame => None,
            PicStructure::Field(p) => Some(p),
        }
    }
}

/// A field decoded as a field picture, with its macroblocks.
#[derive(Clone)]
pub struct FieldPic {
    pub pic: Arc<Picture>,
    pub info: Arc<PicState>,
}

/// A frame store (C.4).
pub struct DpbPic {
    /// The frame's samples: the frame-coded picture or, once both fields
    /// are decoded, the interleaved fields.
    pub frame: Option<Arc<Picture>>,
    /// Macroblocks of a frame-coded picture (or of a non-existing frame).
    pub frame_info: Option<Arc<PicState>>,
    /// The frame was coded as an MBAFF frame.
    pub mbaff: bool,
    /// Field-coded pictures, top then bottom.
    pub fields: [Option<FieldPic>; 2],
    pub id: u32,
    pub frame_num: u32,
    /// TopFieldOrderCnt and BottomFieldOrderCnt.
    pub poc: [i32; 2],
    pub marking: [Marking; 2],
    /// Whether the first stored picture was a reference picture: a second
    /// field pairs with it only if it is one too (as JM pairs fields).
    pub orig_ref: bool,
    pub needed_for_output: bool,
    /// Inferred by the gaps-in-frame_num process (8.2.5.2): no samples.
    pub non_existing: bool,
    /// Output cropping and view, carried to the output queue.
    pub out: OutputInfo,
    /// 3D offset metadata of the GOP this picture belongs to.
    pub meta: Option<GopMeta>,
}

impl DpbPic {
    fn coded_frame(&self) -> bool {
        self.frame_info.is_some()
    }

    fn has(&self, p: Parity) -> bool {
        self.coded_frame() || self.fields[p as usize].is_some()
    }

    /// PicOrderCnt of the frame store: the smaller field order count of
    /// its decoded fields (8-1, and JM's frame store POC).
    fn poc(&self) -> i32 {
        match (self.has(Parity::Top), self.has(Parity::Bottom)) {
            (true, true) => self.poc[0].min(self.poc[1]),
            (true, false) => self.poc[0],
            _ => self.poc[1],
        }
    }

    fn is_ref(&self) -> bool {
        self.marking.iter().any(|&m| m != Marking::Unused)
    }

    /// Marking of the store as a frame: both fields must agree.
    fn frame_marking(&self) -> Marking {
        if self.has(Parity::Top) && self.has(Parity::Bottom) && self.marking[0] == self.marking[1] {
            self.marking[0]
        } else {
            Marking::Unused
        }
    }

    fn unmark(&mut self) {
        self.marking = [Marking::Unused; 2];
    }

    /// The progress of the frame's samples (fields are decoded whole).
    fn state(&self) -> Arc<PicState> {
        match (&self.frame_info, &self.fields) {
            (Some(s), _) => s.clone(),
            (None, [Some(f), _] | [None, Some(f)]) => f.info.clone(),
            (None, [None, None]) => PicState::done(Vec::new()),
        }
    }

    /// The frame for output: a missing field is grey.
    fn output_frame(&self) -> Arc<Picture> {
        if let Some(f) = &self.frame {
            return f.clone();
        }
        Arc::new(Picture::interleave(
            self.fields[0].as_ref().map(|f| &*f.pic),
            self.fields[1].as_ref().map(|f| &*f.pic),
        ))
    }
}

/// Offset metadata tagged with a GOP number (see sei.rs).
#[derive(Clone, Debug)]
pub struct GopMeta {
    pub gop: u64,
    pub data: Arc<crate::sei::OffsetMetadata>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct OutputInfo {
    /// Cropped size and top-left offset in luma samples.
    pub width: usize,
    pub height: usize,
    pub crop_x: usize,
    pub crop_y: usize,
    pub view_id: u16,
    /// Caller timestamp of the access unit (Decoder::set_timestamp).
    pub pts: Option<i64>,
    /// Access unit index, shared by the views of one access unit.
    pub au: u64,
    /// Not output: a leading picture of a stream that started at a non-IDR
    /// picture, whose references are missing.
    pub hidden: bool,
    /// VUI (colour_primaries, transfer_characteristics, matrix_coefficients).
    pub colour: Option<(u8, u8, u8)>,
    /// VUI video_full_range_flag.
    pub full_range: bool,
}

/// Co-located macroblocks of a reference picture, for direct prediction
/// (8.4.1.2.1): their arrangement depends on how the picture was coded.
#[derive(Clone)]
pub enum ColInfo {
    /// A frame-coded picture.
    Frame(Arc<PicState>),
    /// An MBAFF frame (macroblocks addressed in pairs).
    Mbaff(Arc<PicState>),
    /// A field-coded picture.
    Field(Arc<PicState>),
    /// A complementary field pair referenced as a frame: its fields and
    /// their order counts.
    Pair {
        fields: [Arc<PicState>; 2],
        poc: [i32; 2],
    },
}

/// One entry of RefPicList0/1: a frame, or a field when decoding a field.
#[derive(Clone)]
pub struct RefPic {
    /// The samples, or the frame that holds them (`field_of_frame`).
    pub pic: Arc<Picture>,
    /// The reference is this field of the frame `pic`.
    pub field_of_frame: Option<Parity>,
    /// The reference is a field of this parity.
    pub parity: Option<Parity>,
    pub col: ColInfo,
    /// Identifies the reference: frame store id << 2, plus 1 + parity for
    /// a field.
    pub id: u32,
    pub poc: i32,
    /// Top and bottom field order counts (a field: its own twice).
    pub field_poc: [i32; 2],
    pub long_term: bool,
    /// An inter-view prediction reference (G.8.4): neither short- nor long-term.
    pub inter_view: bool,
}

impl RefPic {
    /// Field `parity` of this frame reference, for field macroblocks of
    /// MBAFF frames (8.4.2.1).
    pub fn field(&self, parity: Parity) -> RefPic {
        RefPic {
            pic: self.pic.clone(),
            field_of_frame: Some(parity),
            parity: Some(parity),
            col: match &self.col {
                ColInfo::Pair { fields, .. } => ColInfo::Field(fields[parity as usize].clone()),
                c => c.clone(),
            },
            id: (self.id & !3) | (1 + parity as u32),
            poc: self.field_poc[parity as usize],
            field_poc: [self.field_poc[parity as usize]; 2],
            long_term: self.long_term,
            inter_view: self.inter_view,
        }
    }

    /// A frame-coded picture as a reference (inter-view prediction).
    pub fn frame(pic: Arc<Picture>, info: Arc<PicState>, id: u32, poc: i32) -> Self {
        RefPic {
            pic,
            field_of_frame: None,
            parity: None,
            col: ColInfo::Frame(info),
            id: id << 2,
            poc,
            field_poc: [poc, poc],
            long_term: false,
            inter_view: false,
        }
    }

    /// The progress of the picture holding the samples: rows must be
    /// final before they are read.
    pub fn state(&self) -> &Arc<PicState> {
        match &self.col {
            ColInfo::Frame(s) | ColInfo::Mbaff(s) | ColInfo::Field(s) => s,
            ColInfo::Pair { fields, .. } => &fields[0],
        }
    }

    /// Plane `i` of the reference's samples.
    pub fn plane(&self, i: usize) -> PlaneView<'_> {
        let p = &self.pic.planes[i];
        match self.field_of_frame {
            Some(parity) => p.field(parity),
            None => p.view(),
        }
    }
}

/// RefPicList0 and RefPicList1.
pub type RefLists = [Vec<Option<RefPic>>; 2];

/// A picture ready for output, in output order.
pub struct Output {
    pub pic: Arc<Picture>,
    /// Finished once the picture is decoded.
    pub state: Arc<PicState>,
    pub poc: i32,
    pub out: OutputInfo,
    pub meta: Option<GopMeta>,
}

/// The current decoded picture, for Dpb::store.
pub struct Decoded {
    pub pic: Arc<Picture>,
    pub info: Arc<PicState>,
    pub structure: PicStructure,
    /// An MBAFF frame.
    pub mbaff: bool,
    pub id: u32,
    pub frame_num: u32,
    /// TopFieldOrderCnt and BottomFieldOrderCnt (only the field's own entry
    /// matters for a field), after any mmco 5 adjustment.
    pub poc: [i32; 2],
    pub out: OutputInfo,
    pub meta: Option<GopMeta>,
}

/// An entry of an initial reference list: a frame store, or one of its fields.
pub type Entry = (usize, Option<Parity>);

pub struct Dpb {
    pics: Vec<DpbPic>,
    /// DPB size in frames (C.4: max_dec_frame_buffering or the level limit).
    pub size: usize,
    max_long_term_idx: Option<u32>,
    /// Id of the store holding the last decoded picture when it is a first
    /// field that the next field may complete.
    last_field: Option<u32>,
    pub output: VecDeque<Output>,
}

/// MaxDpbMbs per level_idc (Table A-1).
fn max_dpb_mbs(level_idc: u8) -> u32 {
    match level_idc {
        9 | 10 => 396,
        11 => 900,
        12 | 13 | 20 => 2376,
        21 => 4752,
        22 | 30 => 8100,
        31 => 18000,
        32 => 20480,
        40 | 41 => 32768,
        42 => 34816,
        50 => 110400,
        51 | 52 => 184320,
        _ => 696320,
    }
}

/// DPB size in frames for a sequence (A.3.1 h, E.2.1 max_dec_frame_buffering).
pub fn dpb_size(sps: &crate::ps::Sps) -> usize {
    let frame_mbs = sps.pic_width_in_mbs * sps.frame_height_in_mbs();
    let level = (max_dpb_mbs(sps.level_idc) / frame_mbs.max(1)).clamp(1, 16) as usize;
    match sps
        .vui
        .as_ref()
        .and_then(|v| v.bitstream_restriction.as_ref())
    {
        // Larger declared values are tolerated (JM rejects them).
        Some(r) => (r.max_dec_frame_buffering as usize)
            .max(sps.max_num_ref_frames as usize)
            .max(1),
        None => level,
    }
}

/// Orders the fields of frames for a field reference list (8.2.4.2.5):
/// alternating parity starting with `parity`, skipping fields that are
/// missing or do not satisfy `ok`, then the rest of the remaining parity.
fn alternate(frames: &[usize], parity: Parity, ok: impl Fn(usize, Parity) -> bool) -> Vec<Entry> {
    let mut out = Vec::new();
    let mut next = [0usize, 0usize]; // per parity: position in `frames`
    let mut want = parity;
    loop {
        let k = &mut next[want as usize];
        while *k < frames.len() && !ok(frames[*k], want) {
            *k += 1;
        }
        if *k < frames.len() {
            out.push((frames[*k], Some(want)));
            *k += 1;
            want = want.other();
        } else {
            let other = want.other();
            for &f in &frames[next[other as usize]..] {
                if ok(f, other) {
                    out.push((f, Some(other)));
                }
            }
            return out;
        }
    }
}

impl Dpb {
    pub fn new() -> Self {
        Self {
            pics: Vec::new(),
            size: 16,
            max_long_term_idx: None,
            last_field: None,
            output: VecDeque::new(),
        }
    }

    /// FrameNumWrap (8-27) relative to the current frame_num.
    fn frame_num_wrap(p: &DpbPic, frame_num: u32, max_frame_num: u32) -> i32 {
        if p.frame_num > frame_num {
            p.frame_num as i32 - max_frame_num as i32
        } else {
            p.frame_num as i32
        }
    }

    fn ref_pic(&self, i: usize, field: Option<Parity>) -> RefPic {
        let p = &self.pics[i];
        match field {
            None => RefPic {
                pic: p.frame.clone().expect("a frame reference has samples"),
                field_of_frame: None,
                parity: None,
                col: match &p.frame_info {
                    Some(info) if p.mbaff => ColInfo::Mbaff(info.clone()),
                    Some(info) => ColInfo::Frame(info.clone()),
                    None => ColInfo::Pair {
                        fields: [0, 1].map(|k| p.fields[k].as_ref().unwrap().info.clone()),
                        poc: p.poc,
                    },
                },
                id: p.id << 2,
                poc: p.poc[0].min(p.poc[1]),
                field_poc: p.poc,
                long_term: matches!(p.frame_marking(), Marking::Long(_)),
                inter_view: false,
            },
            Some(f) => {
                let (pic, field_of_frame, col) = match (&p.frame_info, &p.fields[f as usize]) {
                    (Some(info), _) => (
                        p.frame.clone().unwrap(),
                        Some(f),
                        if p.mbaff {
                            ColInfo::Mbaff(info.clone())
                        } else {
                            ColInfo::Frame(info.clone())
                        },
                    ),
                    (None, Some(fp)) => (fp.pic.clone(), None, ColInfo::Field(fp.info.clone())),
                    (None, None) => unreachable!("field references are decoded fields"),
                };
                RefPic {
                    pic,
                    field_of_frame,
                    parity: Some(f),
                    col,
                    id: p.id << 2 | (1 + f as u32),
                    poc: p.poc[f as usize],
                    field_poc: [p.poc[f as usize]; 2],
                    long_term: matches!(p.marking[f as usize], Marking::Long(_)),
                    inter_view: false,
                }
            }
        }
    }

    fn long_sorted(&self, pred: impl Fn(&DpbPic) -> Option<u32>) -> Vec<usize> {
        let mut long: Vec<(u32, usize)> = (0..self.pics.len())
            .filter_map(|i| pred(&self.pics[i]).map(|idx| (idx, i)))
            .collect();
        long.sort();
        long.into_iter().map(|(_, i)| i).collect()
    }

    /// LongTermFrameIdx of a store with a long-term field.
    fn any_long(p: &DpbPic) -> Option<u32> {
        p.marking.iter().find_map(|m| match m {
            Marking::Long(i) => Some(*i),
            _ => None,
        })
    }

    /// Initial RefPicList0 for P slices (8.2.4.2.1, 8.2.4.2.2).
    pub fn init_list_p(&self, cur: PicStructure, frame_num: u32, max_frame_num: u32) -> Vec<Entry> {
        let by_wrap = |pred: &dyn Fn(&DpbPic) -> bool| {
            let mut v: Vec<usize> = (0..self.pics.len())
                .filter(|&i| pred(&self.pics[i]))
                .collect();
            v.sort_by_key(|&i| {
                std::cmp::Reverse(Self::frame_num_wrap(
                    &self.pics[i],
                    frame_num,
                    max_frame_num,
                ))
            });
            v
        };
        match cur.parity() {
            None => {
                let mut l: Vec<Entry> = by_wrap(&|p| p.frame_marking() == Marking::Short)
                    .into_iter()
                    .map(|i| (i, None))
                    .collect();
                l.extend(
                    self.long_sorted(|p| match p.frame_marking() {
                        Marking::Long(idx) => Some(idx),
                        _ => None,
                    })
                    .into_iter()
                    .map(|i| (i, None)),
                );
                l
            }
            Some(parity) => {
                let short = by_wrap(&|p| p.marking.contains(&Marking::Short));
                let long = self.long_sorted(Self::any_long);
                let mut l = alternate(&short, parity, |i, f| {
                    self.pics[i].has(f) && self.pics[i].marking[f as usize] == Marking::Short
                });
                l.extend(alternate(&long, parity, |i, f| {
                    self.pics[i].has(f)
                        && matches!(self.pics[i].marking[f as usize], Marking::Long(_))
                }));
                l
            }
        }
    }

    /// Initial RefPicList0 and RefPicList1 for B slices (8.2.4.2.3, 8.2.4.2.4).
    pub fn init_lists_b(&self, cur: PicStructure, cur_poc: i32) -> [Vec<Entry>; 2] {
        // Frames: both fields short-term; fields: either. Non-existing
        // frames are left out.
        let short: Vec<usize> = (0..self.pics.len())
            .filter(|&i| {
                let p = &self.pics[i];
                !p.non_existing
                    && match cur {
                        PicStructure::Frame => p.frame_marking() == Marking::Short,
                        PicStructure::Field(_) => p.marking.contains(&Marking::Short),
                    }
            })
            .collect();
        let poc = |i: usize| self.pics[i].poc();
        // Frames: POC below the current one (8.2.4.2.3); fields: up to and
        // including it, which covers the first field of the current frame
        // (8.2.4.2.4).
        let first = |i: usize| match cur {
            PicStructure::Frame => poc(i) < cur_poc,
            PicStructure::Field(_) => poc(i) <= cur_poc,
        };
        let mut before: Vec<usize> = short.iter().copied().filter(|&i| first(i)).collect();
        let mut after: Vec<usize> = short.iter().copied().filter(|&i| !first(i)).collect();
        before.sort_by_key(|&i| std::cmp::Reverse(poc(i)));
        after.sort_by_key(|&i| poc(i));
        let mut f0 = before.clone();
        f0.extend(&after);
        let mut f1 = after;
        f1.extend(&before);
        let [l0, l1] = match cur.parity() {
            None => {
                let long = self.long_sorted(|p| match p.frame_marking() {
                    Marking::Long(idx) => Some(idx),
                    _ => None,
                });
                let frames = |v: Vec<usize>| -> Vec<Entry> {
                    v.into_iter()
                        .chain(long.iter().copied())
                        .map(|i| (i, None))
                        .collect()
                };
                [frames(f0), frames(f1)]
            }
            Some(parity) => {
                let long = self.long_sorted(Self::any_long);
                let short_ok = |i: usize, f: Parity| {
                    self.pics[i].has(f) && self.pics[i].marking[f as usize] == Marking::Short
                };
                let long_ok = |i: usize, f: Parity| {
                    self.pics[i].has(f)
                        && matches!(self.pics[i].marking[f as usize], Marking::Long(_))
                };
                let list = |v: Vec<usize>| {
                    let mut l = alternate(&v, parity, short_ok);
                    l.extend(alternate(&long, parity, long_ok));
                    l
                };
                [list(f0), list(f1)]
            }
        };
        let mut l1 = l1;
        if l1.len() > 1 && l1 == l0 {
            l1.swap(0, 1);
        }
        [l0, l1]
    }

    /// Builds a reference picture list (8.2.4, G.8.2): the initial list
    /// (`None` for an MVC IDR view component), then the inter-view
    /// references appended (G.8.2.1), then the modifications (8.2.4.3,
    /// G.8.2.2). `inter_view` holds the inter-view prediction references in
    /// reference view index order. Entries that resolve to nothing are "no
    /// reference picture".
    #[allow(clippy::too_many_arguments)]
    pub fn build_list(
        &self,
        init: Option<Vec<Entry>>,
        inter_view: &[RefPic],
        mods: &[RefPicListModification],
        num_active: usize,
        cur: PicStructure,
        frame_num: u32,
        max_frame_num: u32,
    ) -> Vec<Option<RefPic>> {
        #[derive(Clone, Copy, PartialEq, Eq)]
        enum E {
            Dpb(usize, Option<Parity>),
            View(usize),
        }
        let mut list: Vec<Option<E>> = init
            .unwrap_or_default()
            .into_iter()
            .map(|(i, f)| Some(E::Dpb(i, f)))
            .collect();
        list.truncate(num_active);
        // G.8.2.1: append from the first empty position (or the end).
        list.extend((0..inter_view.len()).map(|v| Some(E::View(v))));
        list.resize(list.len().max(num_active), None);
        list.truncate(num_active);

        // CurrPicNum and MaxPicNum (7.4.3); PicNum of fields (8-30, 8-31).
        let field = cur.parity();
        let (cur_pic_num, max_pic_num) = match field {
            None => (frame_num as i32, max_frame_num as i32),
            Some(_) => (2 * frame_num as i32 + 1, 2 * max_frame_num as i32),
        };
        let pic_num = |i: usize, f: Option<Parity>| {
            let w = Self::frame_num_wrap(&self.pics[i], frame_num, max_frame_num);
            match (field, f) {
                (Some(c), Some(f)) => 2 * w + (c == f) as i32,
                _ => w,
            }
        };
        let long_pic_num = |idx: u32, f: Option<Parity>| match (field, f) {
            (Some(c), Some(f)) => 2 * idx + (c == f) as u32,
            _ => idx,
        };
        // A frame store's references in this picture's terms.
        let refs = |i: usize| -> Vec<(Option<Parity>, Marking)> {
            let p = &self.pics[i];
            match field {
                None => vec![(None, p.frame_marking())],
                Some(_) => [Parity::Top, Parity::Bottom]
                    .into_iter()
                    .filter(|&f| p.has(f))
                    .map(|f| (Some(f), p.marking[f as usize]))
                    .collect(),
            }
        };
        let mut pred = cur_pic_num;
        let mut view_pred = -1i32;
        let max_view_idx = inter_view.len() as i32 - 1;
        let mut idx = 0usize;
        for m in mods {
            let target = match *m {
                RefPicListModification::ShortTermSub(d)
                | RefPicListModification::ShortTermAdd(d) => {
                    let delta = d as i32 + 1;
                    let no_wrap = if matches!(m, RefPicListModification::ShortTermSub(_)) {
                        let v = pred - delta;
                        if v < 0 { v + max_pic_num } else { v }
                    } else {
                        let v = pred + delta;
                        if v >= max_pic_num { v - max_pic_num } else { v }
                    };
                    pred = no_wrap;
                    let target = if no_wrap > cur_pic_num {
                        no_wrap - max_pic_num
                    } else {
                        no_wrap
                    };
                    (0..self.pics.len()).find_map(|i| {
                        refs(i).into_iter().find_map(|(f, mk)| {
                            (mk == Marking::Short && pic_num(i, f) == target)
                                .then_some(E::Dpb(i, f))
                        })
                    })
                }
                RefPicListModification::LongTerm(n) => (0..self.pics.len()).find_map(|i| {
                    refs(i).into_iter().find_map(|(f, mk)| match mk {
                        Marking::Long(idx) if long_pic_num(idx, f) == n => Some(E::Dpb(i, f)),
                        _ => None,
                    })
                }),
                RefPicListModification::InterViewSub(d)
                | RefPicListModification::InterViewAdd(d) => {
                    // G-6, G-7
                    let delta = d as i32 + 1;
                    let v = if matches!(m, RefPicListModification::InterViewSub(_)) {
                        let v = view_pred - delta;
                        if v < 0 { v + max_view_idx + 1 } else { v }
                    } else {
                        let v = view_pred + delta;
                        if v > max_view_idx {
                            v - (max_view_idx + 1)
                        } else {
                            v
                        }
                    };
                    view_pred = v;
                    (0..inter_view.len() as i32)
                        .contains(&v)
                        .then_some(E::View(v as usize))
                }
            };
            let Some(target) = target else { continue };
            if idx >= num_active {
                break;
            }
            // 8-37, 8-38, G-10: insert at idx, then drop later duplicates.
            list.insert(idx, Some(target));
            idx += 1;
            let mut n = idx;
            for c in idx..list.len() {
                if list[c] != Some(target) {
                    list[n] = list[c];
                    n += 1;
                }
            }
            list.truncate(n.max(num_active));
            list.resize(num_active.max(list.len()), None);
        }
        list.truncate(num_active);
        list.into_iter()
            .map(|e| match e? {
                E::Dpb(i, f) => Some(self.ref_pic(i, f)),
                E::View(v) => Some(inter_view[v].clone()),
            })
            .collect()
    }

    /// Decoding process for gaps in frame_num (8.2.5.2): inserts
    /// "non-existing" short-term frames for each skipped frame_num.
    pub fn fill_gaps(
        &mut self,
        prev_ref_frame_num: u32,
        frame_num: u32,
        max_frame_num: u32,
        max_num_ref_frames: u32,
        template: &Arc<Picture>,
        next_id: &mut u32,
    ) {
        self.last_field = None;
        let mut f = (prev_ref_frame_num + 1) % max_frame_num;
        while f != frame_num {
            self.sliding_window(f, max_frame_num, max_num_ref_frames);
            self.remove_unused();
            while self.pics.len() >= self.size {
                if !self.bump() {
                    break;
                }
            }
            *next_id += 1;
            self.pics.push(DpbPic {
                frame: Some(template.clone()),
                frame_info: Some(PicState::done(Vec::new())),
                mbaff: false,
                fields: [None, None],
                id: *next_id,
                frame_num: f,
                poc: [0, 0],
                marking: [Marking::Short; 2],
                orig_ref: true,
                needed_for_output: false,
                non_existing: true,
                out: OutputInfo::default(),
                meta: None,
            });
            f = (f + 1) % max_frame_num;
        }
    }

    /// Sliding window marking (8.2.5.3) for a frame or a first field.
    fn sliding_window(&mut self, frame_num: u32, max_frame_num: u32, max_num_ref_frames: u32) {
        let refs = self.pics.iter().filter(|p| p.is_ref()).count();
        if refs >= (max_num_ref_frames.max(1)) as usize
            && let Some(i) = (0..self.pics.len())
                .filter(|&i| self.pics[i].marking.contains(&Marking::Short))
                .min_by_key(|&i| Self::frame_num_wrap(&self.pics[i], frame_num, max_frame_num))
        {
            self.pics[i].unmark();
        }
    }

    fn remove_unused(&mut self) {
        self.pics.retain(|p| p.is_ref() || p.needed_for_output);
        if let Some(id) = self.last_field
            && !self.pics.iter().any(|p| p.id == id)
        {
            self.last_field = None;
        }
    }

    /// The "bumping" process (C.4.5.3). Returns false when nothing is waiting.
    fn bump(&mut self) -> bool {
        let Some(i) = (0..self.pics.len())
            .filter(|&i| self.pics[i].needed_for_output)
            .min_by_key(|&i| self.pics[i].poc())
        else {
            return false;
        };
        let p = &mut self.pics[i];
        p.needed_for_output = false;
        self.output.push_back(Output {
            pic: p.output_frame(),
            state: p.state(),
            poc: p.poc(),
            out: p.out,
            meta: p.meta.clone(),
        });
        if !self.pics[i].is_ref() {
            self.pics.remove(i);
            self.remove_unused();
        }
        true
    }

    /// Applies memory_management_control_operations (8.2.5.4) for the
    /// current picture. `cur` is the store of the first field when the
    /// current picture is the second field of a pair. Returns the current
    /// picture's marking when an mmco 6 assigns it, and whether an mmco 5
    /// was present.
    fn mmco(
        &mut self,
        ops: &[Mmco],
        structure: PicStructure,
        frame_num: u32,
        max_frame_num: u32,
        cur: Option<usize>,
    ) -> (Option<Marking>, bool) {
        let field = structure.parity();
        let cur_pic_num = match field {
            None => frame_num as i32,
            Some(_) => 2 * frame_num as i32 + 1,
        };
        // The picture (store, field) a picture number refers to.
        let find_short = |pics: &[DpbPic], pic_num_x: i32| -> Option<(usize, Option<Parity>)> {
            (0..pics.len()).find_map(|i| {
                let w = Self::frame_num_wrap(&pics[i], frame_num, max_frame_num);
                match field {
                    None => (pics[i].frame_marking() == Marking::Short && w == pic_num_x)
                        .then_some((i, None)),
                    Some(c) => [Parity::Top, Parity::Bottom].into_iter().find_map(|f| {
                        (pics[i].has(f)
                            && pics[i].marking[f as usize] == Marking::Short
                            && 2 * w + (c == f) as i32 == pic_num_x)
                            .then_some((i, Some(f)))
                    }),
                }
            })
        };
        let mut cur_marking = None;
        let mut mmco5 = false;
        for op in ops {
            match *op {
                Mmco::ForgetShort(d) => {
                    if let Some((i, f)) = find_short(&self.pics, cur_pic_num - (d as i32 + 1)) {
                        match f {
                            None => self.pics[i].unmark(),
                            Some(f) => self.pics[i].marking[f as usize] = Marking::Unused,
                        }
                    }
                }
                Mmco::ForgetLong(n) => {
                    for p in &mut self.pics {
                        match field {
                            None => {
                                if p.frame_marking() == Marking::Long(n) {
                                    p.unmark();
                                }
                            }
                            Some(c) => {
                                for f in [Parity::Top, Parity::Bottom] {
                                    if let Marking::Long(idx) = p.marking[f as usize]
                                        && 2 * idx + (c == f) as u32 == n
                                    {
                                        p.marking[f as usize] = Marking::Unused;
                                    }
                                }
                            }
                        }
                    }
                }
                Mmco::ShortToLong(d, idx) => {
                    if let Some((i, f)) = find_short(&self.pics, cur_pic_num - (d as i32 + 1)) {
                        // Other pictures holding the index lose it; within
                        // the same frame the other field keeps it.
                        for (k, p) in self.pics.iter_mut().enumerate() {
                            if k != i {
                                for m in &mut p.marking {
                                    if *m == Marking::Long(idx) {
                                        *m = Marking::Unused;
                                    }
                                }
                            }
                        }
                        match f {
                            None => self.pics[i].marking = [Marking::Long(idx); 2],
                            Some(f) => self.pics[i].marking[f as usize] = Marking::Long(idx),
                        }
                    }
                }
                Mmco::MaxLongTermIdx(plus1) => {
                    self.max_long_term_idx = plus1.checked_sub(1);
                    for p in &mut self.pics {
                        for m in &mut p.marking {
                            if let Marking::Long(idx) = *m
                                && self.max_long_term_idx.is_none_or(|max| idx > max)
                            {
                                *m = Marking::Unused;
                            }
                        }
                    }
                }
                Mmco::ForgetAll => {
                    for p in &mut self.pics {
                        p.unmark();
                    }
                    self.max_long_term_idx = None;
                    mmco5 = true;
                }
                Mmco::CurrentToLong(idx) => {
                    for (k, p) in self.pics.iter_mut().enumerate() {
                        if Some(k) != cur {
                            for m in &mut p.marking {
                                if *m == Marking::Long(idx) {
                                    *m = Marking::Unused;
                                }
                            }
                        }
                    }
                    cur_marking = Some(Marking::Long(idx));
                }
            }
        }
        (cur_marking, mmco5)
    }

    /// Marks and stores the current decoded picture (8.2.5, C.4.4, C.4.5).
    pub fn store(
        &mut self,
        mut cur: Decoded,
        ref_idc: u8,
        idr: bool,
        marking: Option<&DecRefPicMarking>,
        max_frame_num: u32,
        max_num_ref_frames: u32,
    ) {
        let is_ref = ref_idc != 0;
        let field = cur.structure.parity();
        // The first field this one completes (as JM pairs fields).
        let pair = match field {
            Some(f) if !idr => self.last_field.and_then(|id| {
                self.pics.iter().position(|p| {
                    p.id == id
                        && p.frame_num == cur.frame_num
                        && p.fields[f.other() as usize].is_some()
                        && p.fields[f as usize].is_none()
                        && p.orig_ref == is_ref
                })
            }),
            _ => None,
        };
        let mut cur_marking = Marking::Unused;
        let mut mmco5 = false;
        let mut adaptive = false;
        if is_ref {
            cur_marking = Marking::Short;
            match marking {
                Some(DecRefPicMarking::Idr {
                    no_output_of_prior_pics,
                    long_term_reference,
                }) => {
                    for p in &mut self.pics {
                        p.unmark();
                    }
                    if *no_output_of_prior_pics {
                        self.pics.clear();
                        self.last_field = None;
                    }
                    if *long_term_reference {
                        cur_marking = Marking::Long(0);
                        self.max_long_term_idx = Some(0);
                    } else {
                        self.max_long_term_idx = None;
                    }
                }
                Some(DecRefPicMarking::Adaptive(ops)) => {
                    adaptive = true;
                    let (m, five) =
                        self.mmco(ops, cur.structure, cur.frame_num, max_frame_num, pair);
                    if let Some(m) = m {
                        cur_marking = m;
                    }
                    if five {
                        mmco5 = true;
                        // 7.4.3: the picture is afterwards treated as frame_num 0.
                        cur.frame_num = 0;
                    }
                }
                _ => {}
            }
        }

        if let Some(i) = pair
            && !mmco5
        {
            // Second field: completes the frame store, no sliding window
            // (8.2.5.3) and no output step (C.4.5.1).
            let f = field.unwrap();
            let p = &mut self.pics[i];
            p.marking[f as usize] = cur_marking;
            p.poc[f as usize] = cur.poc[f as usize];
            p.fields[f as usize] = Some(FieldPic {
                pic: cur.pic,
                info: cur.info,
            });
            p.frame = Some(Arc::new(Picture::interleave(
                p.fields[0].as_ref().map(|x| &*x.pic),
                p.fields[1].as_ref().map(|x| &*x.pic),
            )));
            self.last_field = None;
            return;
        }

        if is_ref && !idr && !adaptive {
            self.sliding_window(cur.frame_num, max_frame_num, max_num_ref_frames);
        }
        self.remove_unused();
        let no_output = matches!(
            marking,
            Some(DecRefPicMarking::Idr {
                no_output_of_prior_pics: true,
                ..
            })
        );
        if (idr && !no_output) || mmco5 {
            while self.bump() {}
            self.remove_unused();
        }
        let cur_poc = match field {
            None => cur.poc[0].min(cur.poc[1]),
            Some(f) => cur.poc[f as usize],
        };
        let mut entry = DpbPic {
            frame: None,
            frame_info: None,
            mbaff: cur.mbaff,
            fields: [None, None],
            id: cur.id,
            frame_num: cur.frame_num,
            poc: cur.poc,
            marking: [Marking::Unused; 2],
            orig_ref: is_ref,
            needed_for_output: true,
            non_existing: false,
            out: cur.out,
            meta: cur.meta,
        };
        match field {
            None => {
                entry.marking = [cur_marking; 2];
                entry.frame = Some(cur.pic);
                entry.frame_info = Some(cur.info);
            }
            Some(f) => {
                entry.marking[f as usize] = cur_marking;
                entry.fields[f as usize] = Some(FieldPic {
                    pic: cur.pic,
                    info: cur.info,
                });
            }
        }
        if is_ref || field.is_some() {
            while self.pics.len() >= self.size && self.bump() {}
        } else {
            // C.4.5.2: output directly when it precedes everything waiting.
            loop {
                if self.pics.len() < self.size {
                    break;
                }
                let min_waiting = self
                    .pics
                    .iter()
                    .filter(|p| p.needed_for_output)
                    .map(|p| p.poc())
                    .min();
                if min_waiting.is_none_or(|m| cur_poc < m) {
                    self.output.push_back(Output {
                        pic: entry.frame.clone().unwrap(),
                        state: entry.state(),
                        poc: cur_poc,
                        out: entry.out,
                        meta: entry.meta.clone(),
                    });
                    return;
                }
                if !self.bump() {
                    break;
                }
            }
        }
        self.last_field = field.map(|_| entry.id);
        self.pics.push(entry);
    }

    /// Drops every picture, output or not (a seek).
    pub fn discard(&mut self) {
        self.pics.clear();
        self.output.clear();
        self.last_field = None;
    }

    /// Whether a picture of access unit `au` is stored.
    pub(crate) fn holds(&self, au: u64) -> bool {
        self.pics.iter().any(|p| p.out.au == au)
    }

    /// Outputs every picture still waiting (end of stream).
    pub fn flush(&mut self) {
        while self.bump() {}
        self.pics.clear();
        self.last_field = None;
    }
}

impl Default for Dpb {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_lists_alternate_parity() {
        // Frames in list order 0, 1, 2; frame 1 has only a bottom field.
        let has = |i: usize, f: Parity| !(i == 1 && f == Parity::Top);
        let l = alternate(&[0, 1, 2], Parity::Top, has);
        let want = [
            (0, Parity::Top),
            (0, Parity::Bottom),
            (2, Parity::Top),
            (1, Parity::Bottom),
            (2, Parity::Bottom),
        ];
        assert_eq!(l, want.map(|(i, f)| (i, Some(f))).to_vec());
    }
}
