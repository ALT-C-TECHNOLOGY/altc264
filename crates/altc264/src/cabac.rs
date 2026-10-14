//! CABAC arithmetic decoding engine (9.3.1, 9.3.3.2).
//!
//! The spec's 9-bit codIOffset and the next `cnt` bits of the slice data are
//! kept together in a 64-bit window: `value = codIOffset << cnt | lookahead`.
//! A decision then compares against `codIRange << cnt`, renormalization
//! becomes one shift computed with leading_zeros, and the window is refilled
//! a byte at a time only when the lookahead runs low.

use crate::cabac_tables::{CTX_INIT_I, CTX_INIT_PB, RANGE_TAB_LPS, TRANS_IDX_LPS, TRANS_IDX_MPS};

/// One context variable. Bits 57..63 hold (pStateIdx << 1) | valMPS and
/// bits 0..31 the four rangeTabLPS entries of pStateIdx, so a decision gets
/// its LPS range with a shift instead of a dependent table load, and its
/// next state's index with one more shift.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ctx(u64);

impl Default for Ctx {
    fn default() -> Self {
        Self::new(0, false)
    }
}

/// The packed context for (pStateIdx << 1) | valMPS.
const fn pack(s: usize) -> u64 {
    let r = RANGE_TAB_LPS[s >> 1];
    (s as u64) << 57
        | (r[0] as u64)
        | (r[1] as u64) << 8
        | (r[2] as u64) << 16
        | (r[3] as u64) << 24
}

impl Ctx {
    pub fn new(p_state_idx: u8, val_mps: bool) -> Self {
        Self(pack((p_state_idx << 1 | val_mps as u8) as usize))
    }

    /// (pStateIdx << 1) | valMPS.
    #[inline(always)]
    fn s(self) -> usize {
        (self.0 >> 57) as usize
    }

    pub fn state(self) -> u8 {
        (self.s() >> 1) as u8
    }

    pub fn mps(self) -> bool {
        self.s() & 1 != 0
    }

    /// Initialization (9.3.1.1, equation 9-5).
    pub fn init((m, n): (i8, i8), slice_qp: i32) -> Self {
        let pre = (((m as i32) * slice_qp.clamp(0, 51)) >> 4) + n as i32;
        let pre = pre.clamp(1, 126);
        if pre <= 63 {
            Self::new((63 - pre) as u8, false)
        } else {
            Self::new((pre - 64) as u8, true)
        }
    }
}

/// Context states after each outcome, indexed by the packed Ctx value
/// (9.3.3.2.1.1, Table 9-45). The LPS table also flips valMPS at state 0.
const NEXT_MPS: [u8; 128] = {
    let mut t = [0u8; 128];
    let mut i = 0;
    while i < 128 {
        t[i] = TRANS_IDX_MPS[i >> 1] << 1 | (i as u8 & 1);
        i += 1;
    }
    t
};

const NEXT_LPS: [u8; 128] = {
    let mut t = [0u8; 128];
    let mut i = 0;
    while i < 128 {
        let mps = i as u8 & 1;
        let flip = if i >> 1 == 0 { 1 } else { 0 };
        t[i] = TRANS_IDX_LPS[i >> 1] << 1 | (mps ^ flip);
        i += 1;
    }
    t
};

/// Initializes all 1024 context variables for a slice (9.3.1.1). `cabac_init_idc`
/// is ignored for I and SI slices.
pub fn init_contexts(ctx: &mut [Ctx; 1024], intra: bool, cabac_init_idc: u8, slice_qp: i32) {
    let table = if intra {
        &CTX_INIT_I
    } else {
        &CTX_INIT_PB[cabac_init_idc as usize]
    };
    for (c, &mn) in ctx.iter_mut().zip(table.iter()) {
        *c = Ctx::init(mn, slice_qp);
    }
}

/// Packed contexts after an MPS or LPS, indexed by (pStateIdx << 1) | valMPS.
#[cfg(test)]
static NEXT_MPS_CTX: [Ctx; 128] = {
    let mut t = [Ctx(0); 128];
    let mut i = 0;
    while i < 128 {
        t[i] = Ctx(pack(NEXT_MPS[i] as usize));
        i += 1;
    }
    t
};

#[cfg(test)]
static NEXT_LPS_CTX: [Ctx; 128] = {
    let mut t = [Ctx(0); 128];
    let mut i = 0;
    while i < 128 {
        t[i] = Ctx(pack(NEXT_LPS[i] as usize));
        i += 1;
    }
    t
};

/// Next packed context indexed by ((pStateIdx << 1) | valMPS) << 1 | bin is
/// the LPS, so the decision needs no branch on the outcome.
static NEXT_CTX: [Ctx; 256] = {
    let mut t = [Ctx(0); 256];
    let mut i = 0;
    while i < 128 {
        t[2 * i] = Ctx(pack(NEXT_MPS[i] as usize));
        t[2 * i + 1] = Ctx(pack(NEXT_LPS[i] as usize));
        i += 1;
    }
    t
};

/// Lookahead is topped up to more than this many bits before any operation.
const REFILL_BELOW: i32 = 16;

#[derive(Clone, Copy)]
pub struct Decoder<'a> {
    data: &'a [u8],
    /// Next byte of `data` to load into the window.
    pos: usize,
    value: u64,
    /// Number of lookahead bits held below the 9-bit offset.
    cnt: i32,
    range: u32,
}

/// Decoder::refill on x86 and Apple cores: out of line and by value, so the
/// decoding loops keep only the window in registers, which is faster there
/// (and slower on Cortex cores). The new (pos, value, cnt).
#[cold]
#[inline(never)]
fn refill(data: &[u8], mut pos: usize, mut value: u64, mut cnt: i32) -> (usize, u64, i32) {
    // value < 512 << cnt always holds, so up to 55 lookahead bits fit.
    if let Some(b) = data.get(pos..pos + 4)
        && cnt <= 23
    {
        let w = u32::from_be_bytes(b.try_into().unwrap());
        return (pos + 4, value << 32 | w as u64, cnt + 32);
    }
    while cnt <= 47 {
        let b = data.get(pos).copied().unwrap_or(0);
        value = value << 8 | b as u64;
        pos += 1;
        cnt += 8;
    }
    (pos, value, cnt)
}

impl<'a> Decoder<'a> {
    /// Starts decoding at a byte-aligned position (after cabac_alignment_one_bits),
    /// performing 9.3.1.2. Data past the end reads as zero bits.
    pub fn new(data: &'a [u8], byte_pos: usize) -> Self {
        let mut d = Self {
            data,
            pos: byte_pos,
            value: 0,
            cnt: -9,
            range: 510,
        };
        d.refill();
        d
    }

    #[inline(always)]
    fn refill(&mut self) {
        if cfg!(any(target_arch = "x86_64", target_vendor = "apple")) {
            (self.pos, self.value, self.cnt) = refill(self.data, self.pos, self.value, self.cnt);
            return;
        }
        // value < 512 << cnt always holds, so up to 55 lookahead bits fit.
        if let Some(b) = self.data.get(self.pos..self.pos + 4)
            && self.cnt <= 23
        {
            let w = u32::from_be_bytes(b.try_into().unwrap());
            self.value = self.value << 32 | w as u64;
            self.pos += 4;
            self.cnt += 32;
            return;
        }
        while self.cnt <= 47 {
            let b = self.data.get(self.pos).copied().unwrap_or(0);
            self.value = self.value << 8 | b as u64;
            self.pos += 1;
            self.cnt += 8;
        }
    }

    /// Tops up the lookahead: enough for two decisions or bypass bins
    /// (each takes at most 7 bits) with [`Self::decision_filled`].
    #[inline(always)]
    pub fn ensure(&mut self) {
        if self.cnt < REFILL_BELOW {
            self.refill();
        }
    }

    /// DecodeDecision (9.3.3.2.1).
    #[inline(always)]
    pub fn decision(&mut self, ctx: &mut Ctx) -> bool {
        self.ensure();
        self.decision_filled(ctx)
    }

    /// [`Self::decision`] for the syntax elements outside the residual
    /// loops, whose engine state is in memory anyway. On Cortex cores one
    /// shared copy is faster, saving their instruction caches a copy per
    /// call site; Apple and x86 cores prefer the copies.
    #[cfg_attr(
        all(target_arch = "aarch64", not(target_vendor = "apple")),
        inline(never)
    )]
    #[cfg_attr(
        not(all(target_arch = "aarch64", not(target_vendor = "apple"))),
        inline(always)
    )]
    pub fn decision_call(&mut self, ctx: &mut Ctx) -> bool {
        self.decision(ctx)
    }

    /// DecodeDecision without topping up the lookahead: at most two after
    /// [`Self::ensure`].
    #[inline(always)]
    pub fn decision_filled(&mut self, ctx: &mut Ctx) -> bool {
        let c = ctx.0;
        // qCodIRangeIdx * 8 selects the byte of the packed LPS row.
        let lps = (c >> (self.range >> 3 & 0x18)) as u32 & 0xff;
        let mps_range = self.range - lps;
        let scaled = (mps_range as u64) << self.cnt;
        // The outcome is often unpredictable (significance maps): select
        // rather than branch.
        let is_lps = self.value >= scaled;
        self.value -= std::hint::select_unpredictable(is_lps, scaled, 0);
        // c >> 56 is the state shifted left once.
        *ctx = NEXT_CTX[(c >> 56) as usize | is_lps as usize];
        // RenormD (9.3.3.2.2): range returns to 256..=510 in one shift.
        if cfg!(target_vendor = "apple") {
            // Both outcomes renormalized alongside the comparison rather
            // than after it (lps is 2..=240, mps_range 128..=508): faster on
            // Apple cores, slower on Cortex and no faster on x86.
            let lps_shift = lps.leading_zeros() as i32 - 23;
            let mps_shift = (mps_range < 256) as i32;
            self.range =
                std::hint::select_unpredictable(is_lps, lps << lps_shift, mps_range << mps_shift);
            self.cnt -= std::hint::select_unpredictable(is_lps, lps_shift, mps_shift);
        } else {
            self.range = std::hint::select_unpredictable(is_lps, lps, mps_range);
            let shift = self.range.leading_zeros() as i32 - 23;
            self.range <<= shift;
            self.cnt -= shift;
        }
        (c >> 57 & 1 != 0) ^ is_lps
    }

    /// DecodeBypass (9.3.3.2.3).
    #[inline(always)]
    pub fn bypass(&mut self) -> bool {
        self.ensure();
        self.bypass_filled()
    }

    /// DecodeBypass without topping up the lookahead: after [`Self::ensure`]
    /// and at most two decisions.
    #[inline(always)]
    pub fn bypass_filled(&mut self) -> bool {
        self.cnt -= 1;
        let scaled = (self.range as u64) << self.cnt;
        if self.value >= scaled {
            self.value -= scaled;
            true
        } else {
            false
        }
    }

    /// `n` bypass bins, first bin in the most significant position.
    #[inline]
    pub fn bypass_bits(&mut self, n: u32) -> u32 {
        (0..n).fold(0, |v, _| v << 1 | self.bypass() as u32)
    }

    /// DecodeTerminate (9.3.3.2.4). After it returns true, [`Self::bit_pos`]
    /// is the position of the next syntax element read with read_bits (the
    /// pcm alignment bits, or the bit after rbsp_stop_one_bit).
    #[inline]
    pub fn terminate(&mut self) -> bool {
        self.ensure();
        self.range -= 2;
        let scaled = (self.range as u64) << self.cnt;
        if self.value >= scaled {
            true
        } else {
            let shift = self.range.leading_zeros() as i32 - 23;
            self.range <<= shift;
            self.cnt -= shift;
            false
        }
    }

    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    /// Bit position in `data` of the next bit read_bits(1) would return.
    pub fn bit_pos(&self) -> usize {
        self.pos * 8 - self.cnt as usize
    }

    /// True once the engine has consumed bits beyond the end of the data.
    pub fn overrun(&self) -> bool {
        self.bit_pos() > self.data.len() * 8
    }
}

#[cfg(test)]
pub(crate) mod encoder {
    //! The informative CABAC encoder of 9.3.4, used to build test streams.
    use super::*;

    #[derive(Default)]
    pub struct Encoder {
        low: u32,
        range: u32,
        first_bit: bool,
        outstanding: u32,
        pub bits: Vec<bool>,
    }

    impl Encoder {
        pub fn new() -> Self {
            Self {
                low: 0,
                range: 510,
                first_bit: true,
                outstanding: 0,
                bits: Vec::new(),
            }
        }

        fn put_bit(&mut self, b: bool) {
            if self.first_bit {
                self.first_bit = false;
            } else {
                self.bits.push(b);
            }
            for _ in 0..self.outstanding {
                self.bits.push(!b);
            }
            self.outstanding = 0;
        }

        /// RenormE (9.3.4.3).
        fn renorm(&mut self) {
            while self.range < 256 {
                if self.low < 256 {
                    self.put_bit(false);
                } else if self.low >= 512 {
                    self.low -= 512;
                    self.put_bit(true);
                } else {
                    self.low -= 256;
                    self.outstanding += 1;
                }
                self.range <<= 1;
                self.low <<= 1;
            }
        }

        /// EncodeDecision (9.3.4.2).
        pub fn decision(&mut self, ctx: &mut Ctx, bin: bool) {
            let lps = RANGE_TAB_LPS[ctx.state() as usize][(self.range >> 6 & 3) as usize] as u32;
            self.range -= lps;
            if bin != ctx.mps() {
                self.low += self.range;
                self.range = lps;
                *ctx = NEXT_LPS_CTX[ctx.s()];
            } else {
                *ctx = NEXT_MPS_CTX[ctx.s()];
            }
            self.renorm();
        }

        /// EncodeBypass (9.3.4.4).
        pub fn bypass(&mut self, bin: bool) {
            self.low <<= 1;
            if bin {
                self.low += self.range;
            }
            if self.low >= 1024 {
                self.put_bit(true);
                self.low -= 1024;
            } else if self.low < 512 {
                self.put_bit(false);
            } else {
                self.low -= 512;
                self.outstanding += 1;
            }
        }

        /// EncodeTerminate (9.3.4.5), with EncodeFlush when `bin` is set.
        pub fn terminate(&mut self, bin: bool) {
            self.range -= 2;
            if bin {
                self.low += self.range;
                self.range = 2;
                self.renorm();
                self.put_bit(self.low >> 9 & 1 != 0);
                // WriteBits(((codILow >> 7) & 3) | 1, 2)
                let v = (self.low >> 7 & 3) | 1;
                self.bits.push(v & 2 != 0);
                self.bits.push(v & 1 != 0);
            } else {
                self.renorm();
            }
        }

        pub fn bytes(&self) -> Vec<u8> {
            let mut out = vec![0u8; self.bits.len().div_ceil(8)];
            for (i, &b) in self.bits.iter().enumerate() {
                if b {
                    out[i / 8] |= 0x80 >> (i % 8);
                }
            }
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::encoder::Encoder;
    use super::*;

    /// Small deterministic PRNG (xorshift64*) for test data.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545f4914f6cdd1d)
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum Op {
        Decision(usize, bool),
        Bypass(bool),
        Terminate0,
    }

    #[test]
    fn context_init_matches_equation_9_5() {
        assert_eq!(Ctx::init((0, 64), 26), Ctx::new(0, true));
        assert_eq!(Ctx::init((0, 63), 26), Ctx::new(0, false));
        assert_eq!(Ctx::init((-128, 127), 51), Ctx::new(62, false)); // clipped to 1
        assert_eq!(Ctx::init((20, -15), 40), Ctx::new(28, false)); // (20*40>>4)-15 = 35
        assert_eq!(Ctx::init((20, -15), 99), Ctx::init((20, -15), 51)); // QP clipped
    }

    #[test]
    fn state_transitions_flip_mps_only_at_state_zero() {
        assert!(!NEXT_LPS_CTX[Ctx::new(0, true).s()].mps());
        assert!(NEXT_LPS_CTX[Ctx::new(5, true).s()].mps());
        assert_eq!(NEXT_MPS_CTX[Ctx::new(62, false).s()], Ctx::new(62, false));
    }

    #[test]
    fn round_trips_random_bins_through_the_reference_encoder() {
        for seed in 1..40u64 {
            let mut rng = Rng(seed.wrapping_mul(0x9e3779b97f4a7c15));
            let n = 200 + (rng.next() % 5000) as usize;
            // Skewed bins so contexts move through many states.
            let ops: Vec<Op> = (0..n)
                .map(|_| match rng.next() % 10 {
                    0 => Op::Bypass(rng.next() & 1 != 0),
                    1 => Op::Terminate0,
                    _ => {
                        let c = (rng.next() % 8) as usize;
                        Op::Decision(c, rng.next() % (2 + c as u64) == 0)
                    }
                })
                .collect();
            let mut ctx_init = [Ctx::default(); 8];
            for (i, c) in ctx_init.iter_mut().enumerate() {
                *c = Ctx::init(((i as i8) * 7 - 20, 40 + i as i8 * 9), 30);
            }

            let mut enc = Encoder::new();
            let mut ctx = ctx_init;
            for &op in &ops {
                match op {
                    Op::Decision(c, b) => enc.decision(&mut ctx[c], b),
                    Op::Bypass(b) => enc.bypass(b),
                    Op::Terminate0 => enc.terminate(false),
                }
            }
            enc.terminate(true);
            let mut data = vec![0xffu8; 3]; // garbage before the slice data
            data.extend(enc.bytes());
            let end_bits = 24 + enc.bits.len();

            let mut dec = Decoder::new(&data, 3);
            let mut ctx = ctx_init;
            for (i, &op) in ops.iter().enumerate() {
                let ok = match op {
                    Op::Decision(c, b) => dec.decision(&mut ctx[c]) == b,
                    Op::Bypass(b) => dec.bypass() == b,
                    Op::Terminate0 => !dec.terminate(),
                };
                assert!(ok, "seed {seed}: op {i} {op:?} mismatched");
            }
            assert!(dec.terminate(), "seed {seed}: missing end");
            // The last bit read is the final bit the encoder wrote (rbsp_stop_one_bit).
            assert_eq!(
                dec.bit_pos(),
                end_bits,
                "seed {seed}: bit position after terminate"
            );
            assert!(!dec.overrun());
        }
    }
}
