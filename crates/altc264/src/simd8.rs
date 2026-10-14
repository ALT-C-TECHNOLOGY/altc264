//! A 16 x u8 vector type for 8-bit kernels that work on bytes (the
//! deblocking filter), with the backends of [`crate::simd`]: NEON on
//! aarch64, SSE2 on x86_64, SIMD128 on wasm32 built with it, and plain
//! arrays elsewhere (or with `--cfg altc_scalar`). Every backend gives
//! identical results.
//!
//! Masks are vectors whose lanes are all ones (true) or zero (false).

use crate::simd::I16x8;

#[cfg(all(target_arch = "aarch64", not(altc_scalar)))]
mod imp {
    use super::I16x8;
    use core::arch::aarch64::*;

    #[derive(Clone, Copy)]
    pub struct U8x16(uint8x16_t);

    // SAFETY (all functions): NEON is part of the aarch64 baseline.
    impl U8x16 {
        #[inline(always)]
        pub fn splat(v: u8) -> Self {
            unsafe { Self(vdupq_n_u8(v)) }
        }
        #[inline(always)]
        pub fn load(b: &[u8; 16]) -> Self {
            unsafe { Self(vld1q_u8(b.as_ptr())) }
        }
        #[inline(always)]
        pub fn store(self, b: &mut [u8; 16]) {
            unsafe { vst1q_u8(b.as_mut_ptr(), self.0) }
        }
        /// `lo` in lanes 0..8, `hi` in lanes 8..16.
        #[inline(always)]
        pub fn from_halves(lo: &[u8; 8], hi: &[u8; 8]) -> Self {
            unsafe { Self(vcombine_u8(vld1_u8(lo.as_ptr()), vld1_u8(hi.as_ptr()))) }
        }
        #[inline(always)]
        pub fn store_halves(self, lo: &mut [u8; 8], hi: &mut [u8; 8]) {
            unsafe {
                vst1_u8(lo.as_mut_ptr(), vget_low_u8(self.0));
                vst1_u8(hi.as_mut_ptr(), vget_high_u8(self.0));
            }
        }
        /// |self - o|.
        #[inline(always)]
        pub fn abd(self, o: Self) -> Self {
            unsafe { Self(vabdq_u8(self.0, o.0)) }
        }
        /// self < o, as a mask.
        #[inline(always)]
        pub fn lt(self, o: Self) -> Self {
            unsafe { Self(vcltq_u8(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn and(self, o: Self) -> Self {
            unsafe { Self(vandq_u8(self.0, o.0)) }
        }
        /// Lanes of `a` where `self` (a mask) is set, else of `b`.
        #[inline(always)]
        pub fn select(self, a: Self, b: Self) -> Self {
            unsafe { Self(vbslq_u8(self.0, a.0, b.0)) }
        }
        /// Whether any lane of the mask is set.
        #[inline(always)]
        pub fn any(self) -> bool {
            unsafe { vmaxvq_u8(self.0) != 0 }
        }
        /// (self + o + 1) >> 1.
        #[inline(always)]
        pub fn avg(self, o: Self) -> Self {
            unsafe { Self(vrhaddq_u8(self.0, o.0)) }
        }
        /// (self + o) >> 1.
        #[inline(always)]
        pub fn havg(self, o: Self) -> Self {
            unsafe { Self(vhaddq_u8(self.0, o.0)) }
        }
        /// Saturating.
        #[inline(always)]
        pub fn adds(self, o: Self) -> Self {
            unsafe { Self(vqaddq_u8(self.0, o.0)) }
        }
        /// Saturating.
        #[inline(always)]
        pub fn subs(self, o: Self) -> Self {
            unsafe { Self(vqsubq_u8(self.0, o.0)) }
        }
        /// Wrapping.
        #[inline(always)]
        pub fn sub(self, o: Self) -> Self {
            unsafe { Self(vsubq_u8(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn min(self, o: Self) -> Self {
            unsafe { Self(vminq_u8(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn max(self, o: Self) -> Self {
            unsafe { Self(vmaxq_u8(self.0, o.0)) }
        }
        // Signed byte arithmetic: lanes read as i8 (`_s`), or u8 with an
        // i8 operand (`_us`).
        /// (self - o) >> 1 of u8 lanes, as i8.
        #[inline(always)]
        pub fn hsub_s(self, o: Self) -> Self {
            unsafe { Self(vhsubq_u8(self.0, o.0)) }
        }
        /// self - o, saturated to i8.
        #[inline(always)]
        pub fn qsub_s(self, o: Self) -> Self {
            unsafe {
                let (a, b) = (vreinterpretq_s8_u8(self.0), vreinterpretq_s8_u8(o.0));
                Self(vreinterpretq_u8_s8(vqsubq_s8(a, b)))
            }
        }
        /// (self + o + 1) >> 1 of i8 lanes.
        #[inline(always)]
        pub fn avg_s(self, o: Self) -> Self {
            unsafe {
                let (a, b) = (vreinterpretq_s8_u8(self.0), vreinterpretq_s8_u8(o.0));
                Self(vreinterpretq_u8_s8(vrhaddq_s8(a, b)))
            }
        }
        /// self >> 1 of i8 lanes (arithmetic).
        #[inline(always)]
        pub fn shr1_s(self) -> Self {
            unsafe {
                Self(vreinterpretq_u8_s8(vshrq_n_s8::<1>(vreinterpretq_s8_u8(
                    self.0,
                ))))
            }
        }
        #[inline(always)]
        pub fn min_s(self, o: Self) -> Self {
            unsafe {
                let (a, b) = (vreinterpretq_s8_u8(self.0), vreinterpretq_s8_u8(o.0));
                Self(vreinterpretq_u8_s8(vminq_s8(a, b)))
            }
        }
        #[inline(always)]
        pub fn max_s(self, o: Self) -> Self {
            unsafe {
                let (a, b) = (vreinterpretq_s8_u8(self.0), vreinterpretq_s8_u8(o.0));
                Self(vreinterpretq_u8_s8(vmaxq_s8(a, b)))
            }
        }
        /// u8 lanes plus i8 lanes of `d`, saturated to 0..=255.
        #[inline(always)]
        pub fn adds_us(self, d: Self) -> Self {
            unsafe { Self(vsqaddq_u8(self.0, vreinterpretq_s8_u8(d.0))) }
        }
        /// Lanes 0..8, zero-extended.
        #[inline(always)]
        pub fn lo16(self) -> I16x8 {
            unsafe { I16x8(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(self.0)))) }
        }
        /// Lanes 8..16, zero-extended.
        #[inline(always)]
        pub fn hi16(self) -> I16x8 {
            unsafe { I16x8(vreinterpretq_s16_u16(vmovl_high_u8(self.0))) }
        }
        /// `lo` and `hi` saturated to 0..=255.
        #[inline(always)]
        pub fn pack(lo: I16x8, hi: I16x8) -> Self {
            unsafe { Self(vqmovun_high_s16(vqmovun_s16(lo.0), hi.0)) }
        }
        /// Lanes 0..8 from the bytes of `lo`, 8..16 from `hi`.
        #[inline(always)]
        pub fn from_u64(lo: u64, hi: u64) -> Self {
            unsafe { Self(vcombine_u8(vcreate_u8(lo), vcreate_u8(hi))) }
        }
        /// The same 128 bits as eight 16-bit lanes (byte 2i low in lane i).
        #[inline(always)]
        pub fn as_i16(self) -> I16x8 {
            unsafe { I16x8(vreinterpretq_s16_u8(self.0)) }
        }
    }

    /// See [`super::spread_bs`]: table lookups on bytes.
    #[inline(always)]
    pub fn spread_bs(bs: [u8; 4], quad: bool, tables: [[u8; 8]; 2]) -> (U8x16, U8x16) {
        unsafe {
            let b = vcombine_u8(vcreate_u8(u32::from_le_bytes(bs) as u64), vdup_n_u8(0));
            let idx: [u8; 16] = if quad {
                [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
            } else {
                [0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3]
            };
            let b16 = vqtbl1q_u8(b, vld1q_u8(idx.as_ptr()));
            let on = vcgtq_u8(b16, vdupq_n_u8(0));
            // Entries 0..4 for bS 0..4 (bS 0 and 4 have tC0 0), the second
            // half's table from entry 8.
            let t = |k: usize| u64::from_le_bytes(tables[k]);
            let tab = vcombine_u8(vcreate_u8(t(0)), vcreate_u8(t(if quad { 0 } else { 1 })));
            let off: [u8; 16] = [0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8];
            let tc = vqtbl1q_u8(tab, vaddq_u8(b16, vld1q_u8(off.as_ptr())));
            (U8x16(on), U8x16(tc))
        }
    }

    /// The 8 samples of 16 lines (rows `r[i]`) as 8 vectors of 16 lanes:
    /// vector k holds sample k of each line.
    #[inline(always)]
    pub fn transpose_in(r: &[[u8; 8]; 16]) -> [U8x16; 8] {
        unsafe {
            // Rows i and i + 8 in the halves of q[i], transposed as two 8x8
            // byte matrices at once.
            let q: [uint8x16_t; 8] = core::array::from_fn(|i| {
                vcombine_u8(vld1_u8(r[i].as_ptr()), vld1_u8(r[i + 8].as_ptr()))
            });
            transpose_halves(q).map(U8x16)
        }
    }

    /// The inverse of [`transpose_in`].
    #[inline(always)]
    pub fn transpose_out(v: [U8x16; 8], r: &mut [[u8; 8]; 16]) {
        unsafe {
            let q = transpose_halves(v.map(|x| x.0));
            for (i, x) in q.iter().enumerate() {
                vst1_u8(r[i].as_mut_ptr(), vget_low_u8(*x));
                vst1_u8(r[i + 8].as_mut_ptr(), vget_high_u8(*x));
            }
        }
    }

    /// Transposes the low halves of `q` as an 8x8 byte matrix, and the high
    /// halves as another.
    #[inline(always)]
    unsafe fn transpose_halves(q: [uint8x16_t; 8]) -> [uint8x16_t; 8] {
        unsafe {
            let t8 = |a, b| (vtrn1q_u8(a, b), vtrn2q_u8(a, b));
            let (a0, a1) = t8(q[0], q[1]);
            let (a2, a3) = t8(q[2], q[3]);
            let (a4, a5) = t8(q[4], q[5]);
            let (a6, a7) = t8(q[6], q[7]);
            let t16 = |a: uint8x16_t, b: uint8x16_t| {
                let (a, b) = (vreinterpretq_u16_u8(a), vreinterpretq_u16_u8(b));
                (vtrn1q_u16(a, b), vtrn2q_u16(a, b))
            };
            let (b0, b2) = t16(a0, a2);
            let (b1, b3) = t16(a1, a3);
            let (b4, b6) = t16(a4, a6);
            let (b5, b7) = t16(a5, a7);
            let t32 = |a: uint16x8_t, b: uint16x8_t| {
                let (a, b) = (vreinterpretq_u32_u16(a), vreinterpretq_u32_u16(b));
                (
                    vreinterpretq_u8_u32(vtrn1q_u32(a, b)),
                    vreinterpretq_u8_u32(vtrn2q_u32(a, b)),
                )
            };
            let (c0, c4) = t32(b0, b4);
            let (c1, c5) = t32(b1, b5);
            let (c2, c6) = t32(b2, b6);
            let (c3, c7) = t32(b3, b7);
            [c0, c1, c2, c3, c4, c5, c6, c7]
        }
    }
}

#[cfg(all(target_arch = "x86_64", not(altc_scalar)))]
mod imp {
    use super::I16x8;
    use core::arch::x86_64::*;

    #[derive(Clone, Copy)]
    pub struct U8x16(__m128i);

    // SAFETY (all functions): SSE2 is part of the x86_64 baseline; loads and
    // stores have their sizes from the array types.
    impl U8x16 {
        #[inline(always)]
        pub fn splat(v: u8) -> Self {
            unsafe { Self(_mm_set1_epi8(v as i8)) }
        }
        #[inline(always)]
        pub fn load(b: &[u8; 16]) -> Self {
            unsafe { Self(_mm_loadu_si128(b.as_ptr().cast())) }
        }
        #[inline(always)]
        pub fn from_u64(lo: u64, hi: u64) -> Self {
            unsafe { Self(_mm_set_epi64x(hi as i64, lo as i64)) }
        }
        #[inline(always)]
        pub fn store(self, b: &mut [u8; 16]) {
            unsafe { _mm_storeu_si128(b.as_mut_ptr().cast(), self.0) }
        }
        #[inline(always)]
        pub fn from_halves(lo: &[u8; 8], hi: &[u8; 8]) -> Self {
            unsafe {
                Self(_mm_unpacklo_epi64(
                    _mm_loadl_epi64(lo.as_ptr().cast()),
                    _mm_loadl_epi64(hi.as_ptr().cast()),
                ))
            }
        }
        #[inline(always)]
        pub fn store_halves(self, lo: &mut [u8; 8], hi: &mut [u8; 8]) {
            unsafe {
                _mm_storel_epi64(lo.as_mut_ptr().cast(), self.0);
                _mm_storel_epi64(hi.as_mut_ptr().cast(), _mm_unpackhi_epi64(self.0, self.0));
            }
        }
        #[inline(always)]
        pub fn abd(self, o: Self) -> Self {
            unsafe {
                Self(_mm_or_si128(
                    _mm_subs_epu8(self.0, o.0),
                    _mm_subs_epu8(o.0, self.0),
                ))
            }
        }
        #[inline(always)]
        pub fn lt(self, o: Self) -> Self {
            // self < o where o - self (saturating) is nonzero.
            unsafe {
                let ge = _mm_cmpeq_epi8(_mm_subs_epu8(o.0, self.0), _mm_setzero_si128());
                Self(_mm_xor_si128(ge, _mm_set1_epi8(-1)))
            }
        }
        #[inline(always)]
        pub fn and(self, o: Self) -> Self {
            unsafe { Self(_mm_and_si128(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn select(self, a: Self, b: Self) -> Self {
            unsafe {
                Self(_mm_or_si128(
                    _mm_and_si128(self.0, a.0),
                    _mm_andnot_si128(self.0, b.0),
                ))
            }
        }
        #[inline(always)]
        pub fn any(self) -> bool {
            unsafe { _mm_movemask_epi8(self.0) != 0 }
        }
        #[inline(always)]
        pub fn avg(self, o: Self) -> Self {
            unsafe { Self(_mm_avg_epu8(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn havg(self, o: Self) -> Self {
            // The rounded average, less one where the sum is odd.
            unsafe {
                let odd = _mm_and_si128(_mm_xor_si128(self.0, o.0), _mm_set1_epi8(1));
                Self(_mm_sub_epi8(_mm_avg_epu8(self.0, o.0), odd))
            }
        }
        #[inline(always)]
        pub fn adds(self, o: Self) -> Self {
            unsafe { Self(_mm_adds_epu8(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn subs(self, o: Self) -> Self {
            unsafe { Self(_mm_subs_epu8(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn sub(self, o: Self) -> Self {
            unsafe { Self(_mm_sub_epi8(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn min(self, o: Self) -> Self {
            unsafe { Self(_mm_min_epu8(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn max(self, o: Self) -> Self {
            unsafe { Self(_mm_max_epu8(self.0, o.0)) }
        }
        #[inline(always)]
        fn bias(self) -> __m128i {
            unsafe { _mm_xor_si128(self.0, _mm_set1_epi8(-128)) }
        }
        #[inline(always)]
        pub fn hsub_s(self, o: Self) -> Self {
            // (a + 255 - b + 1) >> 1 is 128 + floor((a - b) / 2).
            unsafe {
                let nb = _mm_xor_si128(o.0, _mm_set1_epi8(-1));
                Self(_mm_xor_si128(_mm_avg_epu8(self.0, nb), _mm_set1_epi8(-128)))
            }
        }
        #[inline(always)]
        pub fn qsub_s(self, o: Self) -> Self {
            unsafe { Self(_mm_subs_epi8(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn avg_s(self, o: Self) -> Self {
            // The unsigned average of the lanes offset by 128.
            unsafe {
                Self(_mm_xor_si128(
                    _mm_avg_epu8(self.bias(), o.bias()),
                    _mm_set1_epi8(-128),
                ))
            }
        }
        #[inline(always)]
        pub fn shr1_s(self) -> Self {
            // (x + 128) >> 1, less 64.
            unsafe {
                let h = _mm_and_si128(_mm_srli_epi16::<1>(self.bias()), _mm_set1_epi8(0x7f));
                Self(_mm_sub_epi8(h, _mm_set1_epi8(64)))
            }
        }
        #[inline(always)]
        pub fn min_s(self, o: Self) -> Self {
            unsafe {
                Self(_mm_xor_si128(
                    _mm_min_epu8(self.bias(), o.bias()),
                    _mm_set1_epi8(-128),
                ))
            }
        }
        #[inline(always)]
        pub fn max_s(self, o: Self) -> Self {
            unsafe {
                Self(_mm_xor_si128(
                    _mm_max_epu8(self.bias(), o.bias()),
                    _mm_set1_epi8(-128),
                ))
            }
        }
        #[inline(always)]
        pub fn adds_us(self, d: Self) -> Self {
            // The positive and negative parts of d, added and subtracted.
            unsafe {
                let zero = Self(_mm_setzero_si128());
                let pos = d.max_s(zero).0;
                let neg = zero.sub(d).max_s(zero).0;
                Self(_mm_subs_epu8(_mm_adds_epu8(self.0, pos), neg))
            }
        }
        #[inline(always)]
        pub fn lo16(self) -> I16x8 {
            unsafe { I16x8(_mm_unpacklo_epi8(self.0, _mm_setzero_si128())) }
        }
        #[inline(always)]
        pub fn hi16(self) -> I16x8 {
            unsafe { I16x8(_mm_unpackhi_epi8(self.0, _mm_setzero_si128())) }
        }
        #[inline(always)]
        pub fn pack(lo: I16x8, hi: I16x8) -> Self {
            unsafe { Self(_mm_packus_epi16(lo.0, hi.0)) }
        }
        #[inline(always)]
        pub fn as_i16(self) -> I16x8 {
            I16x8(self.0)
        }
    }

    #[inline(always)]
    pub fn transpose_in(r: &[[u8; 8]; 16]) -> [U8x16; 8] {
        unsafe {
            let ld = |i: usize| _mm_loadl_epi64(r[i].as_ptr().cast());
            // Bytes of row pairs, then 16-bit pairs of those, and so on: the
            // result interleaves the 16 rows sample by sample.
            let a: [__m128i; 8] =
                core::array::from_fn(|i| _mm_unpacklo_epi8(ld(2 * i), ld(2 * i + 1)));
            let b: [__m128i; 8] = core::array::from_fn(|i| {
                let (x, y) = (a[2 * (i / 2)], a[2 * (i / 2) + 1]);
                if i % 2 == 0 {
                    _mm_unpacklo_epi16(x, y)
                } else {
                    _mm_unpackhi_epi16(x, y)
                }
            });
            // b[2j] holds samples 0..4 of rows 4j..4j+4, b[2j+1] samples 4..8.
            let c: [__m128i; 8] = core::array::from_fn(|i| {
                // i: (half h of samples, pair g of row groups, lo/hi)
                let (h, g, hi) = (i / 4, i / 2 % 2, i % 2);
                let (x, y) = (b[4 * g + h], b[4 * g + 2 + h]);
                if hi == 0 {
                    _mm_unpacklo_epi32(x, y)
                } else {
                    _mm_unpackhi_epi32(x, y)
                }
            });
            // c[4h + 2g + s] holds samples 4h + 2s, 4h + 2s + 1 of rows
            // 8g..8g + 8 (8 bytes each).
            core::array::from_fn(|k| {
                let (h, s, odd) = (k / 4, k % 4 / 2, k % 2);
                let (x, y) = (c[4 * h + s], c[4 * h + 2 + s]);
                U8x16(if odd == 0 {
                    _mm_unpacklo_epi64(x, y)
                } else {
                    _mm_unpackhi_epi64(x, y)
                })
            })
        }
    }

    #[inline(always)]
    pub fn transpose_out(v: [U8x16; 8], r: &mut [[u8; 8]; 16]) {
        unsafe {
            // Row j is [v0[j], ..., v7[j]].
            let a: [__m128i; 8] = core::array::from_fn(|i| {
                let (x, y) = (v[2 * (i % 4)].0, v[2 * (i % 4) + 1].0);
                if i < 4 {
                    _mm_unpacklo_epi8(x, y)
                } else {
                    _mm_unpackhi_epi8(x, y)
                }
            });
            // a[i] (i < 4): pairs (v2i, v2i+1) for rows 0..8; a[4 + i] rows 8..16.
            let b: [__m128i; 8] = core::array::from_fn(|i| {
                let (half, pair, hi) = (i / 4, i / 2 % 2, i % 2);
                let (x, y) = (a[4 * half + 2 * pair], a[4 * half + 2 * pair + 1]);
                if hi == 0 {
                    _mm_unpacklo_epi16(x, y)
                } else {
                    _mm_unpackhi_epi16(x, y)
                }
            });
            // b[4h + 2p + s]: vectors 4p..4p + 4 for rows 8h + 4s..8h + 4s + 4.
            for h in 0..2 {
                for s in 0..2 {
                    let (x, y) = (b[4 * h + s], b[4 * h + 2 + s]);
                    let (lo, hi) = (_mm_unpacklo_epi32(x, y), _mm_unpackhi_epi32(x, y));
                    let row = 8 * h + 4 * s;
                    _mm_storel_epi64(r[row].as_mut_ptr().cast(), lo);
                    _mm_storel_epi64(r[row + 1].as_mut_ptr().cast(), _mm_unpackhi_epi64(lo, lo));
                    _mm_storel_epi64(r[row + 2].as_mut_ptr().cast(), hi);
                    _mm_storel_epi64(r[row + 3].as_mut_ptr().cast(), _mm_unpackhi_epi64(hi, hi));
                }
            }
        }
    }
}

#[cfg(any(altc_scalar, not(any(target_arch = "aarch64", target_arch = "x86_64"))))]
mod imp {
    use super::I16x8;

    #[derive(Clone, Copy)]
    pub struct U8x16([u8; 16]);

    impl U8x16 {
        #[inline(always)]
        fn map2(self, o: Self, f: impl Fn(u8, u8) -> u8) -> Self {
            Self(core::array::from_fn(|i| f(self.0[i], o.0[i])))
        }
        pub fn splat(v: u8) -> Self {
            Self([v; 16])
        }
        pub fn load(b: &[u8; 16]) -> Self {
            Self(*b)
        }
        pub fn from_u64(lo: u64, hi: u64) -> Self {
            Self::from_halves(&lo.to_le_bytes(), &hi.to_le_bytes())
        }
        pub fn store(self, b: &mut [u8; 16]) {
            *b = self.0;
        }
        pub fn from_halves(lo: &[u8; 8], hi: &[u8; 8]) -> Self {
            Self(core::array::from_fn(
                |i| if i < 8 { lo[i] } else { hi[i - 8] },
            ))
        }
        pub fn store_halves(self, lo: &mut [u8; 8], hi: &mut [u8; 8]) {
            lo.copy_from_slice(&self.0[..8]);
            hi.copy_from_slice(&self.0[8..]);
        }
        pub fn abd(self, o: Self) -> Self {
            self.map2(o, |a, b| a.abs_diff(b))
        }
        pub fn lt(self, o: Self) -> Self {
            self.map2(o, |a, b| if a < b { 0xff } else { 0 })
        }
        pub fn and(self, o: Self) -> Self {
            self.map2(o, |a, b| a & b)
        }
        pub fn select(self, a: Self, b: Self) -> Self {
            Self(core::array::from_fn(|i| {
                (self.0[i] & a.0[i]) | (!self.0[i] & b.0[i])
            }))
        }
        pub fn any(self) -> bool {
            self.0.iter().any(|&m| m != 0)
        }
        pub fn avg(self, o: Self) -> Self {
            self.map2(o, |a, b| ((a as u16 + b as u16 + 1) >> 1) as u8)
        }
        pub fn havg(self, o: Self) -> Self {
            self.map2(o, |a, b| ((a as u16 + b as u16) >> 1) as u8)
        }
        pub fn adds(self, o: Self) -> Self {
            self.map2(o, u8::saturating_add)
        }
        pub fn subs(self, o: Self) -> Self {
            self.map2(o, u8::saturating_sub)
        }
        pub fn sub(self, o: Self) -> Self {
            self.map2(o, u8::wrapping_sub)
        }
        pub fn min(self, o: Self) -> Self {
            self.map2(o, u8::min)
        }
        pub fn max(self, o: Self) -> Self {
            self.map2(o, u8::max)
        }
        fn map2s(self, o: Self, f: impl Fn(i32, i32) -> i32) -> Self {
            Self(core::array::from_fn(|i| {
                f(self.0[i] as i8 as i32, o.0[i] as i8 as i32) as i8 as u8
            }))
        }
        pub fn hsub_s(self, o: Self) -> Self {
            self.map2(o, |a, b| ((a as i32 - b as i32) >> 1) as i8 as u8)
        }
        pub fn qsub_s(self, o: Self) -> Self {
            self.map2s(o, |a, b| (a - b).clamp(-128, 127))
        }
        pub fn avg_s(self, o: Self) -> Self {
            self.map2s(o, |a, b| (a + b + 1) >> 1)
        }
        pub fn shr1_s(self) -> Self {
            self.map2s(self, |a, _| a >> 1)
        }
        pub fn min_s(self, o: Self) -> Self {
            self.map2s(o, i32::min)
        }
        pub fn max_s(self, o: Self) -> Self {
            self.map2s(o, i32::max)
        }
        pub fn adds_us(self, d: Self) -> Self {
            Self(core::array::from_fn(|i| {
                (self.0[i] as i32 + d.0[i] as i8 as i32).clamp(0, 255) as u8
            }))
        }
        pub fn lo16(self) -> I16x8 {
            I16x8::from_array(core::array::from_fn(|i| self.0[i] as i16))
        }
        pub fn as_i16(self) -> I16x8 {
            I16x8::from_array(core::array::from_fn(|i| {
                i16::from_le_bytes([self.0[2 * i], self.0[2 * i + 1]])
            }))
        }
        pub fn hi16(self) -> I16x8 {
            I16x8::from_array(core::array::from_fn(|i| self.0[i + 8] as i16))
        }
        pub fn pack(lo: I16x8, hi: I16x8) -> Self {
            let (mut a, mut b) = ([0u8; 8], [0u8; 8]);
            lo.store_u8(&mut a);
            hi.store_u8(&mut b);
            Self::from_halves(&a, &b)
        }
    }

    pub fn transpose_in(r: &[[u8; 8]; 16]) -> [U8x16; 8] {
        core::array::from_fn(|k| U8x16(core::array::from_fn(|i| r[i][k])))
    }

    pub fn transpose_out(v: [U8x16; 8], r: &mut [[u8; 8]; 16]) {
        for (i, row) in r.iter_mut().enumerate() {
            *row = core::array::from_fn(|k| v[k].0[i]);
        }
    }
}

pub use imp::{U8x16, transpose_in, transpose_out};

/// NEON only (see deblock::lanes): per lane, for four segments of bS `bs`
/// over 16 lanes (`quad`: four lanes each; else two each, in both halves),
/// whether the segment's bS is above 0 (a mask) and the entry for its bS of
/// a table of tC0 by bS, `tables[0]` (entries 1 to 3, the others 0), or for
/// the upper half without `quad` `tables[1]`.
#[cfg(all(target_arch = "aarch64", not(altc_scalar)))]
pub use imp::spread_bs;
