//! Bit reader for RBSP data (7.2): u(n), ue(v), se(v) and RBSP framing.
//!
//! Reads past the end yield zero bits instead of failing, so fixed-length
//! fields stay branch-free; callers check [`BitReader::overrun`] (or use
//! `ue`/`se`, which fail on runaway prefixes) at the end of a syntax structure.

use crate::Error;

#[derive(Clone)]
pub struct BitReader<'a> {
    data: &'a [u8],
    pos: usize, // in bits
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Current position in bits.
    #[inline]
    pub fn pos(&self) -> usize {
        self.pos
    }

    #[inline]
    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    /// True once more bits were consumed than the buffer holds.
    #[inline]
    pub fn overrun(&self) -> bool {
        self.pos > self.data.len() * 8
    }

    #[inline]
    pub fn byte_aligned(&self) -> bool {
        self.pos & 7 == 0
    }

    /// The next 64 bits, MSB first, zero-filled past the end.
    #[inline]
    fn window(&self) -> u64 {
        let byte = self.pos >> 3;
        let raw = match self.data.get(byte..byte + 8) {
            Some(b) => u64::from_be_bytes(b.try_into().unwrap()),
            None => {
                let mut buf = [0u8; 8];
                let tail = self.data.get(byte..).unwrap_or(&[]);
                buf[..tail.len()].copy_from_slice(tail);
                u64::from_be_bytes(buf)
            }
        };
        raw << (self.pos & 7)
    }

    /// Peeks `n` bits (n <= 32) without consuming them.
    #[inline]
    pub fn peek(&self, n: u32) -> u32 {
        debug_assert!(n <= 32);
        if n == 0 {
            0
        } else {
            (self.window() >> (64 - n)) as u32
        }
    }

    #[inline]
    pub fn skip(&mut self, n: usize) {
        self.pos += n;
    }

    /// u(n) for n <= 32.
    #[inline]
    pub fn u(&mut self, n: u32) -> u32 {
        let v = self.peek(n);
        self.pos += n as usize;
        v
    }

    #[inline]
    pub fn flag(&mut self) -> bool {
        self.u(1) != 0
    }

    /// ue(v) (9.1). Values up to 2^32 - 2 are representable.
    #[inline]
    pub fn ue(&mut self) -> Result<u32, Error> {
        let lz = self.peek(32).leading_zeros();
        if lz >= 32 {
            return Err(Error::Truncated);
        }
        self.pos += lz as usize;
        Ok(((self.u(lz + 1) as u64) - 1) as u32)
    }

    /// se(v) (9.1.1).
    #[inline]
    pub fn se(&mut self) -> Result<i32, Error> {
        let k = self.ue()? as i64;
        Ok(if k & 1 != 0 { (k + 1) / 2 } else { -(k / 2) } as i32)
    }

    /// ue(v) checked against an inclusive maximum.
    pub fn ue_max(&mut self, field: &'static str, max: u32) -> Result<u32, Error> {
        let v = self.ue()?;
        if v > max {
            return Err(Error::OutOfRange {
                field,
                value: v as i64,
            });
        }
        Ok(v)
    }

    /// se(v) checked against an inclusive range.
    pub fn se_range(&mut self, field: &'static str, min: i32, max: i32) -> Result<i32, Error> {
        let v = self.se()?;
        if v < min || v > max {
            return Err(Error::OutOfRange {
                field,
                value: v as i64,
            });
        }
        Ok(v)
    }

    /// Bit position of the rbsp_stop_one_bit, if any.
    fn stop_bit(&self) -> Option<usize> {
        let last = self.data.iter().rposition(|&b| b != 0)?;
        Some(last * 8 + 7 - self.data[last].trailing_zeros() as usize)
    }

    /// more_rbsp_data() (7.2).
    pub fn more_rbsp_data(&self) -> bool {
        self.stop_bit().is_some_and(|s| self.pos < s)
    }

    /// Whether the reader sits exactly on rbsp_trailing_bits().
    pub fn at_trailing_bits(&self) -> bool {
        self.stop_bit() == Some(self.pos)
    }
}

#[cfg(test)]
pub(crate) mod writer {
    /// Bit writer used to build test bitstreams.
    #[derive(Default)]
    pub struct BitWriter {
        pub bytes: Vec<u8>,
        bits: usize,
    }

    impl BitWriter {
        pub fn u(&mut self, n: u32, v: u64) -> &mut Self {
            for i in (0..n).rev() {
                if self.bits % 8 == 0 {
                    self.bytes.push(0);
                }
                if v >> i & 1 != 0 {
                    *self.bytes.last_mut().unwrap() |= 0x80 >> (self.bits % 8);
                }
                self.bits += 1;
            }
            self
        }

        pub fn flag(&mut self, b: bool) -> &mut Self {
            self.u(1, b as u64)
        }

        pub fn ue(&mut self, v: u32) -> &mut Self {
            let x = v as u64 + 1;
            let len = 64 - x.leading_zeros();
            self.u(len - 1, 0).u(len, x)
        }

        pub fn se(&mut self, v: i32) -> &mut Self {
            let k = if v > 0 {
                2 * v as u32 - 1
            } else {
                (-2 * v as i64) as u32
            };
            self.ue(k)
        }

        pub fn trailing(&mut self) -> &mut Self {
            self.u(1, 1);
            while self.bits % 8 != 0 {
                self.u(1, 0);
            }
            self
        }
    }
}

#[cfg(test)]
mod tests {
    use super::writer::BitWriter;
    use super::*;

    #[test]
    fn reads_fixed_and_exp_golomb() {
        let mut w = BitWriter::default();
        w.u(3, 5)
            .ue(0)
            .ue(1)
            .ue(254)
            .se(-3)
            .se(7)
            .u(32, 0xdeadbeef)
            .ue(u32::MAX - 1)
            .trailing();
        let mut r = BitReader::new(&w.bytes);
        assert_eq!(r.u(3), 5);
        assert_eq!(r.ue(), Ok(0));
        assert_eq!(r.ue(), Ok(1));
        assert_eq!(r.ue(), Ok(254));
        assert_eq!(r.se(), Ok(-3));
        assert_eq!(r.se(), Ok(7));
        assert_eq!(r.u(32), 0xdeadbeef);
        assert_eq!(r.ue(), Ok(u32::MAX - 1));
        assert!(r.at_trailing_bits());
        assert!(!r.more_rbsp_data());
        assert!(!r.overrun());
    }

    #[test]
    fn fails_on_runaway_prefix_and_flags_overrun() {
        let mut r = BitReader::new(&[0, 0, 0, 0, 0]);
        assert_eq!(r.ue(), Err(Error::Truncated));
        let mut r = BitReader::new(&[0xff]);
        r.u(8);
        assert!(!r.overrun());
        assert_eq!(r.u(4), 0);
        assert!(r.overrun());
    }

    #[test]
    fn more_rbsp_data_respects_stop_bit() {
        // 1 data bit (1), then stop bit, then zero padding plus a cabac_zero_word-like zero byte.
        let data = [0b1100_0000, 0];
        let mut r = BitReader::new(&data);
        assert!(r.more_rbsp_data());
        r.u(1);
        assert!(!r.more_rbsp_data());
        assert!(r.at_trailing_bits());
    }
}
