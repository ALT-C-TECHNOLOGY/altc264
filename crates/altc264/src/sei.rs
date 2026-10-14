//! SEI messages (7.3.2.3, D.1) that the decoder uses: Blu-ray 3D offset
//! metadata, carried in user_data_unregistered inside an MVC scalable
//! nesting SEI of the dependent view.
//!
//! The offset metadata ("OFMD") layout comes from the Blu-ray format, which
//! is not public; this parser follows the layout observed consistently on
//! retail Blu-ray 3D streams:
//!
//! ```text
//! "OFMD"                                   32
//! reserved                                  4
//! frame_rate                                4  (1 = 23.976, 2 = 24, ...)
//! reserved                                  5
//! PTS[32..30], marker, PTS[29..15], marker,
//! PTS[14..0], marker                       36
//! reserved                                  1
//! number_of_offset_sequences                6
//! number_of_displayed_frames_in_GOP         8
//! reserved                                 16
//! for each sequence, for each frame:
//!     plane_offset_direction_flag           1
//!     plane_offset_value                    7
//! ```

use crate::bits::BitReader;

/// UUID of the Blu-ray offset metadata user_data_unregistered SEI.
pub const OFFSET_METADATA_UUID: [u8; 16] = [
    0x17, 0xee, 0x8c, 0x60, 0xf8, 0x4d, 0x11, 0xd9, 0x8c, 0xd6, 0x08, 0x00, 0x20, 0x0c, 0x9a, 0x66,
];

/// The offset of a graphics plane for one frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaneOffset {
    /// plane_offset_direction_flag as coded.
    pub direction: bool,
    /// plane_offset_value: horizontal shift in pixels.
    pub value: u8,
}

/// Offset metadata for one GOP.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OffsetMetadata {
    pub frame_rate: u8,
    /// PTS of the first displayed frame of the GOP (90 kHz).
    pub pts: u64,
    pub frames: usize,
    /// offsets[sequence][frame in display order].
    pub offsets: Vec<Vec<PlaneOffset>>,
}

fn parse_offset_metadata(b: &[u8]) -> Option<OffsetMetadata> {
    let b = b.strip_prefix(b"OFMD")?;
    let mut r = BitReader::new(b);
    r.u(4);
    let frame_rate = r.u(4) as u8;
    r.u(5);
    let mut pts = (r.u(3) as u64) << 30;
    r.u(1);
    pts |= (r.u(15) as u64) << 15;
    r.u(1);
    pts |= r.u(15) as u64;
    r.u(1);
    r.u(1);
    let sequences = r.u(6) as usize;
    let frames = r.u(8) as usize;
    r.u(16);
    let mut offsets = Vec::with_capacity(sequences);
    for _ in 0..sequences {
        let mut seq = Vec::with_capacity(frames);
        for _ in 0..frames {
            seq.push(PlaneOffset {
                direction: r.flag(),
                value: r.u(7) as u8,
            });
        }
        offsets.push(seq);
    }
    if r.overrun() {
        return None;
    }
    Some(OffsetMetadata {
        frame_rate,
        pts,
        frames,
        offsets,
    })
}

/// Iterates over the (payloadType, payload) pairs of an SEI RBSP (7.3.2.3.1).
fn messages(rbsp: &[u8]) -> impl Iterator<Item = (u32, &[u8])> {
    let mut pos = 0usize;
    std::iter::from_fn(move || {
        let read_vl = |pos: &mut usize| -> Option<u32> {
            let mut v = 0u32;
            loop {
                let b = *rbsp.get(*pos)?;
                *pos += 1;
                v += b as u32;
                if b != 0xff {
                    return Some(v);
                }
            }
        };
        // Stop at rbsp_trailing_bits.
        if pos >= rbsp.len() || (rbsp[pos] == 0x80 && pos + 1 == rbsp.len()) {
            return None;
        }
        let ty = read_vl(&mut pos)?;
        let size = read_vl(&mut pos)? as usize;
        let payload = rbsp.get(pos..pos + size)?;
        pos += size;
        Some((ty, payload))
    })
}

/// Skips the header of mvc_scalable_nesting() (H.13.1.1) and returns the
/// nested sei_message() data.
fn nested_messages(payload: &[u8]) -> Option<&[u8]> {
    let mut r = BitReader::new(payload);
    if r.flag() {
        // operation_point_flag
        let n = r.ue().ok()? + 1;
        for _ in 0..n {
            r.u(10); // sei_op_view_id
        }
        r.u(3); // sei_op_temporal_id
    } else if !r.flag() {
        // all_view_components_in_au_flag == 0
        let n = r.ue().ok()? + 1;
        for _ in 0..n {
            r.u(10); // sei_view_id
        }
    }
    let start = r.pos().div_ceil(8); // sei_nesting_zero_bits
    payload.get(start..)
}

/// Finds Blu-ray offset metadata in an SEI NAL unit's RBSP, top level or
/// inside MVC scalable nesting.
pub fn offset_metadata(rbsp: &[u8]) -> Option<OffsetMetadata> {
    let from_user_data = |p: &[u8]| {
        p.strip_prefix(&OFFSET_METADATA_UUID[..])
            .and_then(parse_offset_metadata)
    };
    for (ty, payload) in messages(rbsp) {
        match ty {
            5 => {
                if let Some(m) = from_user_data(payload) {
                    return Some(m);
                }
            }
            37 => {
                let nested = nested_messages(payload)?;
                for (nty, np) in messages(nested) {
                    if nty == 5
                        && let Some(m) = from_user_data(np)
                    {
                        return Some(m);
                    }
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_offset_metadata_layout() {
        // Header from a retail disc: 23.976 fps, 32 sequences, 24 frames.
        let mut b = b"OFMD".to_vec();
        b.extend([0x81, 0x00, 0xb3, 0x7b, 0xc2, 0x22, 0xa0, 0x18, 0x80, 0x00]);
        for seq in 0..32u8 {
            b.extend(std::iter::repeat_n(if seq == 0 { 0x03 } else { 0x80 }, 24));
        }
        let m = parse_offset_metadata(&b).unwrap();
        assert_eq!((m.frame_rate, m.frames, m.offsets.len()), (1, 24, 32));
        assert_eq!(
            m.offsets[0][5],
            PlaneOffset {
                direction: false,
                value: 3
            }
        );
        assert_eq!(
            m.offsets[1][0],
            PlaneOffset {
                direction: true,
                value: 0
            }
        );
        assert_eq!(m.pts, 431866402);
    }
}
