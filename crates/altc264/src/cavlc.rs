//! CAVLC residual decoding (7.3.5.3.2, 9.2).
#![allow(clippy::needless_range_loop)]

use std::sync::OnceLock;

use crate::Error;
use crate::bits::BitReader;
use crate::cavlc_tables::{
    COEFF_TOKEN, RUN_BEFORE, TOTAL_ZEROS_4X4, TOTAL_ZEROS_CHROMA_DC_420, TOTAL_ZEROS_CHROMA_DC_422,
    Vlc,
};
use crate::transform::Coeffs;

/// Direct lookup for a prefix-free code: indexed by the next `bits` bits,
/// each entry holds (value, length), length 0 for invalid codes.
struct Lut {
    bits: u32,
    entries: Vec<(u8, u8)>,
}

impl Lut {
    fn new(codes: &[Vlc]) -> Self {
        let bits = codes.iter().map(|c| c.len as u32).max().unwrap_or(1);
        let mut entries = vec![(0u8, 0u8); 1 << bits];
        for c in codes {
            let shift = bits - c.len as u32;
            let base = (c.code << shift) as usize;
            for e in &mut entries[base..base + (1 << shift)] {
                *e = (c.value, c.len);
            }
        }
        Self { bits, entries }
    }

    #[inline]
    fn read(&self, r: &mut BitReader) -> Result<u8, Error> {
        let (v, len) = self.entries[r.peek(self.bits) as usize];
        if len == 0 {
            return Err(Error::Invalid("VLC codeword"));
        }
        r.skip(len as usize);
        Ok(v)
    }
}

struct Tables {
    coeff_token: Vec<Lut>,
    tz_4x4: Vec<Lut>,
    tz_420: Vec<Lut>,
    tz_422: Vec<Lut>,
    run_before: Vec<Lut>,
}

fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| Tables {
        coeff_token: COEFF_TOKEN.iter().map(|t| Lut::new(t)).collect(),
        tz_4x4: TOTAL_ZEROS_4X4.iter().map(|t| Lut::new(t)).collect(),
        tz_420: TOTAL_ZEROS_CHROMA_DC_420
            .iter()
            .map(|t| Lut::new(t))
            .collect(),
        tz_422: TOTAL_ZEROS_CHROMA_DC_422
            .iter()
            .map(|t| Lut::new(t))
            .collect(),
        run_before: RUN_BEFORE.iter().map(|t| Lut::new(t)).collect(),
    })
}

/// residual_block_cavlc() with startIdx 0: decodes `max_coeff` levels
/// (16, 15 for AC, or 4 / 8 for chroma DC) into `coeff` (levelListIdx
/// order, zeroed by the caller) using the coeff_token table selected by `nc`
/// (-1 and -2 for chroma DC). Returns TotalCoeff.
pub fn residual_block(
    r: &mut BitReader,
    nc: i32,
    max_coeff: usize,
    coeff: &mut [i32],
) -> Result<u8, Error> {
    let mut list = Coeffs::default();
    let n = residual_list(r, nc, max_coeff, &mut list)?;
    list.to_list(coeff);
    Ok(n)
}

/// As [`residual_block`], giving the nonzero levels as a list.
pub fn residual_list(
    r: &mut BitReader,
    nc: i32,
    max_coeff: usize,
    coeff: &mut Coeffs,
) -> Result<u8, Error> {
    coeff.n = 0;
    let t = tables();
    let column = match nc {
        -1 => 4,
        -2 => 5,
        0..=1 => 0,
        2..=3 => 1,
        4..=7 => 2,
        _ => 3,
    };
    let token = t.coeff_token[column].read(r)?;
    let (trailing_ones, total) = ((token >> 5) as usize, (token & 31) as usize);
    if total == 0 {
        return Ok(0);
    }
    if total > max_coeff {
        return Err(Error::Invalid("TotalCoeff larger than the block"));
    }

    // Levels, highest frequency first (7.3.5.3.2, 9.2.2).
    let mut level = [0i32; 16];
    let mut suffix_length: u32 = if total > 10 && trailing_ones < 3 {
        1
    } else {
        0
    };
    // trailing_ones_sign_flag for each, read at once.
    let signs = r.u(trailing_ones as u32);
    for (i, l) in level.iter_mut().enumerate().take(trailing_ones) {
        *l = if signs >> (trailing_ones - 1 - i) & 1 != 0 {
            -1
        } else {
            1
        };
    }
    for i in trailing_ones..total {
        // level_prefix: leading zero bits before a one (9.2.2.1).
        let prefix = r.peek(32).leading_zeros();
        if prefix > 31 {
            return Err(Error::Invalid("level_prefix"));
        }
        r.skip(prefix as usize + 1);
        let mut code = (prefix.min(15) << suffix_length) as i32;
        if suffix_length > 0 || prefix >= 14 {
            let size = if prefix == 14 && suffix_length == 0 {
                4
            } else if prefix >= 15 {
                prefix - 3
            } else {
                suffix_length
            };
            code += r.u(size) as i32;
        }
        if prefix >= 15 && suffix_length == 0 {
            code += 15;
        }
        if prefix >= 16 {
            code += (1 << (prefix - 3)) - 4096;
        }
        if i == trailing_ones && trailing_ones < 3 {
            code += 2;
        }
        level[i] = if code % 2 == 0 {
            (code + 2) >> 1
        } else {
            (-code - 1) >> 1
        };
        if suffix_length == 0 {
            suffix_length = 1;
        }
        if level[i].abs() > (3 << (suffix_length - 1)) && suffix_length < 6 {
            suffix_length += 1;
        }
    }

    // total_zeros (9.2.3) and run_before (9.2.4).
    let mut zeros_left = if total < max_coeff {
        let lut = match max_coeff {
            4 => &t.tz_420[total - 1],
            8 => &t.tz_422[total - 1],
            _ => &t.tz_4x4[total - 1],
        };
        lut.read(r)? as usize
    } else {
        0
    };
    if total + zeros_left > max_coeff {
        return Err(Error::Invalid("total_zeros"));
    }
    let mut run = [0usize; 16];
    for i in 0..total - 1 {
        if zeros_left > 0 {
            let v = t.run_before[zeros_left.min(7) - 1].read(r)? as usize;
            if v > zeros_left {
                return Err(Error::Invalid("run_before"));
            }
            run[i] = v;
            zeros_left -= v;
        }
    }
    run[total - 1] = zeros_left;

    let mut pos: isize = -1;
    for i in (0..total).rev() {
        pos += run[i] as isize + 1;
        coeff.push(pos as usize, level[i]);
    }
    if r.overrun() {
        return Err(Error::Truncated);
    }
    Ok(total as u8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::writer::BitWriter;

    #[test]
    fn decodes_the_example_block() {
        // TotalCoeff 5 with levels (decoding order) +1 -1 -1 +1 +3 and runs 1 0 1 1 0.
        // TotalCoeff 5, TrailingOnes 3, nC 0: coeff_token 0000 100.
        let mut w = BitWriter::default();
        w.u(7, 0b0000100);
        // Trailing ones, highest frequency first: +1, -1, -1 -> signs 0 1 1.
        w.u(1, 0).u(1, 1).u(1, 1);
        // Level +1 (i = 3, first non-trailing with T1 = 3: no +2): levelCode 0 -> prefix 0: "1".
        w.u(1, 1);
        // Level +3 (suffixLength 1 after the first level): levelCode 4 -> prefix 2, suffix 0: "001" "0".
        w.u(3, 0b001).u(1, 0);
        // total_zeros = 3 with tzVlcIndex 5: codeword "111" (Table 9-7).
        w.u(3, 0b111);
        // run_before: zerosLeft 3 -> run 1 ("10"), zerosLeft 2 -> run 0 ("1"),
        // zerosLeft 2 -> run 1 ("01"), zerosLeft 1 -> run 1 ("0").
        w.u(2, 0b10).u(1, 1).u(2, 0b01).u(1, 0);
        w.trailing();
        let mut r = BitReader::new(&w.bytes);
        let mut c = [0i32; 16];
        assert_eq!(residual_block(&mut r, 0, 16, &mut c).unwrap(), 5);
        assert_eq!(&c[..9], &[3, 0, 1, 0, -1, -1, 0, 1, 0]);
    }
}
