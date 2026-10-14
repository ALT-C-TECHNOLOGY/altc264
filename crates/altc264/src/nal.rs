//! NAL unit framing (Annex B) and headers (7.3.1, H.7.3.1.1).

/// NAL unit types used by this decoder (Table 7-1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum NalType {
    Unspecified = 0,
    Slice = 1,
    SliceDataA = 2,
    SliceDataB = 3,
    SliceDataC = 4,
    SliceIdr = 5,
    Sei = 6,
    Sps = 7,
    Pps = 8,
    AccessUnitDelimiter = 9,
    EndOfSequence = 10,
    EndOfStream = 11,
    Filler = 12,
    SpsExtension = 13,
    Prefix = 14,
    SubsetSps = 15,
    Dps = 16,
    Reserved17 = 17,
    Reserved18 = 18,
    SliceAux = 19,
    SliceExtension = 20,
    SliceExtensionDepth = 21,
    Reserved22 = 22,
    Reserved23 = 23,
    Unspecified24 = 24,
    Unspecified25 = 25,
    Unspecified26 = 26,
    Unspecified27 = 27,
    Unspecified28 = 28,
    Unspecified29 = 29,
    Unspecified30 = 30,
    Unspecified31 = 31,
}

impl NalType {
    #[inline]
    pub const fn from_bits(v: u8) -> Self {
        // SAFETY: NalType is repr(u8) with every value in 0..=31 defined.
        unsafe { core::mem::transmute(v & 31) }
    }

    /// Whether this unit carries an MVC/SVC header extension (7.3.1).
    #[inline]
    pub const fn has_extension(self) -> bool {
        matches!(
            self,
            Self::Prefix | Self::SliceExtension | Self::SliceExtensionDepth
        )
    }
}

/// `nal_unit_header_mvc_extension()` (H.7.3.1.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MvcExtension {
    pub non_idr: bool,
    pub priority_id: u8,
    pub view_id: u16,
    pub temporal_id: u8,
    pub anchor_pic: bool,
    pub inter_view: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NalHeader {
    pub ref_idc: u8,
    pub nal_type: NalType,
    /// Present for types 14/20 (and 21 when not 3D-AVC) with svc_extension_flag == 0.
    pub mvc: Option<MvcExtension>,
    /// Header length in bytes (1, or 4 with an extension).
    pub len: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeaderError {
    Empty,
    ForbiddenBit,
    Truncated,
    /// SVC (Annex G) or 3D-AVC (Annex J) extension; not decoded.
    UnsupportedExtension,
}

impl NalHeader {
    pub fn parse(nal: &[u8]) -> Result<Self, HeaderError> {
        let &b0 = nal.first().ok_or(HeaderError::Empty)?;
        if b0 & 0x80 != 0 {
            return Err(HeaderError::ForbiddenBit);
        }
        let nal_type = NalType::from_bits(b0);
        let ref_idc = (b0 >> 5) & 3;
        if !nal_type.has_extension() {
            return Ok(Self {
                ref_idc,
                nal_type,
                mvc: None,
                len: 1,
            });
        }
        let ext: [u8; 3] = nal
            .get(1..4)
            .ok_or(HeaderError::Truncated)?
            .try_into()
            .unwrap();
        // svc_extension_flag (or avc_3d_extension_flag for type 21).
        if ext[0] & 0x80 != 0 {
            return Err(HeaderError::UnsupportedExtension);
        }
        let bits = u32::from_be_bytes([0, ext[0], ext[1], ext[2]]);
        let mvc = MvcExtension {
            non_idr: bits >> 22 & 1 != 0,
            priority_id: (bits >> 16 & 0x3f) as u8,
            view_id: (bits >> 6 & 0x3ff) as u16,
            temporal_id: (bits >> 3 & 7) as u8,
            anchor_pic: bits >> 2 & 1 != 0,
            inter_view: bits >> 1 & 1 != 0,
        };
        Ok(Self {
            ref_idc,
            nal_type,
            mvc: Some(mvc),
            len: 4,
        })
    }
}

/// Iterates over NAL units in an Annex B byte stream, yielding each unit
/// without its start code or trailing zero bytes.
pub struct AnnexBIter<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> AnnexBIter<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        let pos = find_start_code(data, 0).map_or(data.len(), |p| p + 3);
        Self { data, pos }
    }
}

impl<'a> Iterator for AnnexBIter<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<&'a [u8]> {
        while self.pos < self.data.len() {
            let start = self.pos;
            let (end, next) = match find_start_code(self.data, start) {
                Some(p) => (p, p + 3),
                None => (self.data.len(), self.data.len()),
            };
            self.pos = next;
            let mut nal = &self.data[start..end];
            while let [rest @ .., 0] = nal {
                nal = rest;
            }
            if !nal.is_empty() {
                return Some(nal);
            }
        }
        None
    }
}

/// Position of the next `00 00 01` at or after `from`.
pub fn find_start_code(data: &[u8], from: usize) -> Option<usize> {
    use crate::simd8::U8x16;
    let mut i = from;
    loop {
        // Sixteen bytes without a zero hold no start of a start code:
        // skipped with one test (zeros are rare in coded data).
        while let Some(chunk) = data.get(i..i + 16) {
            if U8x16::load(chunk.try_into().unwrap())
                .lt(U8x16::splat(1))
                .any()
            {
                break;
            }
            i += 16;
        }
        // Byte by byte through those sixteen. Look at the third byte first:
        // anything above 1 rules out three positions.
        let end = i + 16;
        while i < end {
            if i + 2 >= data.len() {
                return None;
            }
            let c = data[i + 2];
            if c > 1 {
                i += 3;
            } else if c == 1 && data[i] == 0 && data[i + 1] == 0 {
                return Some(i);
            } else {
                i += 1;
            }
        }
    }
}

/// Strips emulation prevention bytes (7.4.1), appending the RBSP to `out`.
pub fn unescape_into(nal: &[u8], out: &mut Vec<u8>) {
    out.reserve(nal.len());
    let mut zeros = 0u32;
    let mut run_start = 0;
    let mut i = 0;
    while i < nal.len() {
        // Eight bytes without a zero hold no emulation_prevention_three_byte,
        // unless the first follows two zeros.
        if zeros < 2
            && let Some(w) = nal.get(i..i + 8)
        {
            let w = u64::from_le_bytes(w.try_into().unwrap());
            if w.wrapping_sub(0x0101_0101_0101_0101) & !w & 0x8080_8080_8080_8080 == 0 {
                zeros = 0;
                i += 8;
                continue;
            }
        }
        let b = nal[i];
        if zeros >= 2 && b == 3 {
            out.extend_from_slice(&nal[run_start..i]);
            run_start = i + 1;
            zeros = 0;
        } else {
            zeros = if b == 0 { zeros + 1 } else { 0 };
        }
        i += 1;
    }
    out.extend_from_slice(&nal[run_start..]);
}

/// AVCDecoderConfigurationRecord (ISO/IEC 14496-15, "avcC" extradata).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AvcConfig {
    /// Bytes per NAL unit length field in samples (1, 2 or 4).
    pub length_size: usize,
    /// SPS and PPS NAL units.
    pub nals: Vec<Vec<u8>>,
}

impl AvcConfig {
    pub fn parse(b: &[u8]) -> Option<Self> {
        if b.first() != Some(&1) || b.len() < 7 {
            return None;
        }
        let length_size = (b[4] & 3) as usize + 1;
        let mut pos = 5;
        let mut nals = Vec::new();
        let mut take = |pos: &mut usize, count: usize| -> Option<()> {
            for _ in 0..count {
                let n = u16::from_be_bytes(b.get(*pos..*pos + 2)?.try_into().ok()?) as usize;
                nals.push(b.get(*pos + 2..*pos + 2 + n)?.to_vec());
                *pos += 2 + n;
            }
            Some(())
        };
        let sps = (b[pos] & 31) as usize;
        pos += 1;
        take(&mut pos, sps)?;
        let pps = *b.get(pos)? as usize;
        pos += 1;
        take(&mut pos, pps)?;
        Some(Self { length_size, nals })
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn start_codes_match_a_byte_by_byte_scan() {
        let slow = |d: &[u8], from: usize| {
            (from..d.len().saturating_sub(2)).find(|&i| d[i..i + 3] == [0, 0, 1])
        };
        let mut x = 9u32;
        for round in 0..2000 {
            let len = (round * 7) % 300;
            let d: Vec<u8> = (0..len)
                .map(|_| {
                    x = x.wrapping_mul(1_103_515_245).wrapping_add(12345);
                    // Mostly nonzero, with runs of zeros and ones.
                    match (x >> 16) % 8 {
                        0 | 1 => 0,
                        2 => 1,
                        _ => (x >> 8) as u8 | 2,
                    }
                })
                .collect();
            for from in [0, 1, 5, 17] {
                assert_eq!(
                    find_start_code(&d, from),
                    slow(&d, from),
                    "round {round} from {from}"
                );
            }
        }
    }

    use super::*;

    #[test]
    fn splits_annex_b() {
        let s = [
            0, 0, 0, 1, 0x67, 1, 2, 0, 0, 1, 0x68, 3, 0, 0, 0, 0, 1, 0x65, 4,
        ];
        let nals: Vec<_> = AnnexBIter::new(&s).collect();
        assert_eq!(nals, [&[0x67, 1, 2][..], &[0x68, 3][..], &[0x65, 4][..]]);
    }

    #[test]
    fn handles_no_start_code_and_empty_units() {
        assert_eq!(AnnexBIter::new(&[1, 2, 3]).count(), 0);
        assert_eq!(
            AnnexBIter::new(&[0, 0, 1, 0, 0, 1, 0x09, 0xf0]).collect::<Vec<_>>(),
            [&[0x09, 0xf0][..]]
        );
    }

    #[test]
    fn removes_emulation_prevention() {
        let mut out = Vec::new();
        unescape_into(&[0x65, 0, 0, 3, 1, 0, 0, 3, 0, 0, 3, 3, 5], &mut out);
        assert_eq!(out, [0x65, 0, 0, 1, 0, 0, 0, 0, 3, 5]);
    }

    #[test]
    fn removes_emulation_prevention_like_a_byte_scan() {
        let reference = |nal: &[u8]| {
            let (mut out, mut zeros) = (Vec::new(), 0);
            for &b in nal {
                if zeros >= 2 && b == 3 {
                    zeros = 0;
                    continue;
                }
                zeros = if b == 0 { zeros + 1 } else { 0 };
                out.push(b);
            }
            out
        };
        // Bytes mostly 0, 3 and 1, so emulation prevention lands at every
        // offset relative to the 8-byte words.
        let mut x = 12345u32;
        for len in 0..200 {
            let nal: Vec<u8> = (0..len)
                .map(|_| {
                    x = x.wrapping_mul(1_103_515_245).wrapping_add(12345);
                    [0, 0, 3, 1, 0, 7][(x >> 16) as usize % 6]
                })
                .collect();
            let mut out = Vec::new();
            unescape_into(&nal, &mut out);
            assert_eq!(out, reference(&nal), "{nal:?}");
        }
    }

    #[test]
    fn parses_avcc_extradata() {
        let b = [
            1, 100, 0, 41, 0xff, 0xe1, 0, 3, 0x67, 1, 2, 1, 0, 2, 0x68, 9,
        ];
        let c = AvcConfig::parse(&b).unwrap();
        assert_eq!(c.length_size, 4);
        assert_eq!(c.nals, [vec![0x67, 1, 2], vec![0x68, 9]]);
    }

    #[test]
    fn parses_plain_header() {
        let h = NalHeader::parse(&[0x65]).unwrap();
        assert_eq!(
            (h.ref_idc, h.nal_type, h.mvc, h.len),
            (3, NalType::SliceIdr, None, 1)
        );
        assert_eq!(NalHeader::parse(&[0x85]), Err(HeaderError::ForbiddenBit));
    }

    #[test]
    fn parses_mvc_extension() {
        // ref_idc 1, type 20; non_idr=1 priority=0 view_id=1 temporal=0 anchor=0 inter_view=1 reserved=1
        let bits: u32 = 1 << 22 | 1 << 6 | 1 << 1 | 1;
        let b = bits.to_be_bytes();
        let h = NalHeader::parse(&[0x34, b[1], b[2], b[3]]).unwrap();
        let m = h.mvc.unwrap();
        assert_eq!(h.nal_type, NalType::SliceExtension);
        assert_eq!(h.len, 4);
        assert!(m.non_idr && m.inter_view && !m.anchor_pic);
        assert_eq!((m.view_id, m.priority_id, m.temporal_id), (1, 0, 0));
        assert_eq!(
            NalHeader::parse(&[0x34, 0x80, 0, 0]),
            Err(HeaderError::UnsupportedExtension)
        );
        assert_eq!(NalHeader::parse(&[0x34, 0]), Err(HeaderError::Truncated));
    }
}
