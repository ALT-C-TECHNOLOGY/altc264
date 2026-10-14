//! Combining the two views of a stereo access unit into one frame.

use crate::decoder::Frame;

/// How the views of a stereo pair are combined into one output frame (see
/// Decoder::base_view_right for which view is which eye).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StereoLayout {
    /// Left | right at full resolution: twice the width.
    SideBySide,
    /// Left | right, each squeezed to half width: the width of one view.
    SideBySideHalf,
    /// Left above right at full resolution: twice the height.
    TopAndBottom,
    /// Left above right, each squeezed to half height.
    TopAndBottomHalf,
    /// HDMI 1.4 frame packing: left, a black gap of 45 lines (30 for
    /// 720-line views), then right; 1920x2205 for 1080p. With 4:2:0 the
    /// right view's chroma starts on the chroma row below the gap.
    FramePacking,
}

impl StereoLayout {
    fn gap(self, h: usize) -> usize {
        match self {
            StereoLayout::FramePacking if h == 720 => 30,
            StereoLayout::FramePacking => 45,
            _ => 0,
        }
    }
}

/// Packs `left` and `right` (same size and format) into one frame, which
/// carries the metadata (timestamp, plane offsets) of `base`.
pub(crate) fn pack(left: &Frame, right: &Frame, layout: StereoLayout, base: &Frame) -> Frame {
    use StereoLayout::*;
    let (w, h) = (left.width, left.height);
    let (cw, ch) = (left.chroma_width, left.chroma_height);
    let gap = layout.gap(h);
    let (ow, oh) = match layout {
        SideBySide => (2 * w, h),
        SideBySideHalf | TopAndBottomHalf => (w, h),
        TopAndBottom => (w, 2 * h),
        FramePacking => (w, 2 * h + gap),
    };
    let (ocw, och) = if cw == 0 {
        (0, 0)
    } else {
        // Chroma subsampling of the views, applied to the packed frame.
        (ow * cw / w, (oh * ch).div_ceil(h))
    };
    let mut planes = Vec::new();
    for i in 0..3 {
        let (Some(l), Some(r)) = (left.plane(i), right.plane(i)) else {
            break;
        };
        let (pw, ph, black) = if i == 0 {
            (ow, oh, 16)
        } else {
            (ocw, och, 128)
        };
        let mut out = vec![black; pw * ph];
        // Plane height of one view and the gap, in this plane's rows.
        let (vh, vg) = if i == 0 {
            (h, gap)
        } else {
            (ch, (h + gap) * ch / h - ch)
        };
        for (view, p) in [l, r].into_iter().enumerate() {
            for y in 0..p.height {
                let src = p.row(y);
                match layout {
                    SideBySide => {
                        let o = y * pw + view * p.width;
                        out[o..o + p.width].copy_from_slice(src);
                    }
                    TopAndBottom | FramePacking => {
                        let o = (y + view * (vh + vg)) * pw;
                        out[o..o + p.width].copy_from_slice(src);
                    }
                    SideBySideHalf => {
                        let half = p.width / 2;
                        let o = y * pw + view * half;
                        for (x, d) in out[o..o + half].iter_mut().enumerate() {
                            *d = ((src[2 * x] as u16 + src[2 * x + 1] as u16 + 1) >> 1) as u8;
                        }
                    }
                    TopAndBottomHalf => {
                        if y % 2 == 1 {
                            continue;
                        }
                        let next = p.row((y + 1).min(p.height - 1));
                        let o = (y / 2 + view * (vh / 2)) * pw;
                        for (x, d) in out[o..o + p.width].iter_mut().enumerate() {
                            *d = ((src[x] as u16 + next[x] as u16 + 1) >> 1) as u8;
                        }
                    }
                }
            }
        }
        planes.push(out);
    }
    base.with_planes(ow, oh, ocw, och, planes, Some(layout))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(w: usize, h: usize, y: u8, c: u8) -> Frame {
        Frame::owned(
            w,
            h,
            w / 2,
            h / 2,
            vec![vec![y; w * h], vec![c; w * h / 4], vec![c; w * h / 4]],
        )
    }

    #[test]
    fn layouts_place_both_views() {
        let (l, r) = (frame(8, 4, 10, 20), frame(8, 4, 30, 40));
        let sbs = pack(&l, &r, StereoLayout::SideBySide, &l);
        assert_eq!((sbs.width, sbs.height, sbs.chroma_width), (16, 4, 8));
        let y = sbs.plane(0).unwrap();
        assert_eq!((y.row(3)[7], y.row(3)[8]), (10, 30));
        let half = pack(&l, &r, StereoLayout::SideBySideHalf, &l);
        assert_eq!((half.width, half.plane(0).unwrap().row(0)[4]), (8, 30));
        let tab = pack(&l, &r, StereoLayout::TopAndBottomHalf, &l);
        let y = tab.plane(0).unwrap();
        assert_eq!((tab.height, y.row(1)[0], y.row(2)[0]), (4, 10, 30));
    }

    #[test]
    fn frame_packing_has_the_hdmi_gap() {
        let (l, r) = (frame(16, 1080, 50, 60), frame(16, 1080, 70, 80));
        let fp = pack(&l, &r, StereoLayout::FramePacking, &l);
        assert_eq!((fp.height, fp.chroma_height), (2205, 1103));
        let y = fp.plane(0).unwrap();
        assert_eq!(
            (
                y.row(1079)[0],
                y.row(1080)[0],
                y.row(1124)[0],
                y.row(1125)[0]
            ),
            (50, 16, 16, 70)
        );
        let u = fp.plane(1).unwrap();
        assert_eq!((u.row(539)[0], u.row(540)[0], u.row(562)[0]), (60, 128, 80));
    }
}
