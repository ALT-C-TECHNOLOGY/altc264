//! A minimal 8 x i16 vector type for the pixel kernels: NEON on aarch64,
//! SSE2 on x86_64, SIMD128 on wasm32 built with it, and plain arrays
//! elsewhere (or anywhere with
//! `--cfg altc_scalar`, to test that path). Kernels are written once
//! against [`I16x8`]; every backend gives identical results.
//!
//! Masks are vectors whose lanes are all ones (true) or zero (false).

#[cfg(all(target_arch = "aarch64", not(altc_scalar)))]
mod imp {
    use core::arch::aarch64::*;

    #[derive(Clone, Copy)]
    pub struct I16x8(pub int16x8_t);

    // SAFETY (all functions): NEON is part of the aarch64 baseline.
    impl I16x8 {
        #[inline(always)]
        pub fn splat(v: i16) -> Self {
            unsafe { Self(vdupq_n_s16(v)) }
        }
        /// The low 16 bits of each of eight i32s.
        #[inline(always)]
        pub fn from_i32(a: &[i32; 8]) -> Self {
            unsafe {
                let lo = vmovn_s32(vld1q_s32(a.as_ptr()));
                let hi = vmovn_s32(vld1q_s32(a.as_ptr().add(4)));
                Self(vcombine_s16(lo, hi))
            }
        }
        /// `self` with lane 0 replaced by `v`.
        #[inline(always)]
        pub fn with_lane0(self, v: i16) -> Self {
            unsafe { Self(vsetq_lane_s16::<0>(v, self.0)) }
        }
        /// `self` with lane 4 replaced by `v`.
        #[inline(always)]
        pub fn with_lane4(self, v: i16) -> Self {
            unsafe { Self(vsetq_lane_s16::<4>(v, self.0)) }
        }
        #[inline(always)]
        pub fn from_array(a: [i16; 8]) -> Self {
            unsafe { Self(vld1q_s16(a.as_ptr())) }
        }
        /// Zero-extends 8 bytes.
        #[inline(always)]
        pub fn load_u8(b: &[u8; 8]) -> Self {
            unsafe { Self(vreinterpretq_s16_u16(vmovl_u8(vld1_u8(b.as_ptr())))) }
        }
        /// Saturates each lane to 0..=255.
        #[inline(always)]
        pub fn store_u8(self, b: &mut [u8; 8]) {
            unsafe { vst1_u8(b.as_mut_ptr(), vqmovun_s16(self.0)) }
        }
        #[inline(always)]
        pub fn add(self, o: Self) -> Self {
            unsafe { Self(vaddq_s16(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn sub(self, o: Self) -> Self {
            unsafe { Self(vsubq_s16(self.0, o.0)) }
        }
        /// Low 16 bits of the products.
        #[inline(always)]
        pub fn mul(self, o: Self) -> Self {
            unsafe { Self(vmulq_s16(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn shl<const N: i32>(self) -> Self {
            unsafe { Self(vshlq_n_s16::<N>(self.0)) }
        }
        /// Arithmetic shift right.
        #[inline(always)]
        pub fn shr<const N: i32>(self) -> Self {
            unsafe { Self(vshrq_n_s16::<N>(self.0)) }
        }
        #[inline(always)]
        pub fn abs(self) -> Self {
            unsafe { Self(vabsq_s16(self.0)) }
        }
        #[inline(always)]
        pub fn min(self, o: Self) -> Self {
            unsafe { Self(vminq_s16(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn max(self, o: Self) -> Self {
            unsafe { Self(vmaxq_s16(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn lt(self, o: Self) -> Self {
            unsafe { Self(vreinterpretq_s16_u16(vcltq_s16(self.0, o.0))) }
        }
        #[inline(always)]
        pub fn and(self, o: Self) -> Self {
            unsafe { Self(vandq_s16(self.0, o.0)) }
        }
        /// Lanes of `a` where `self` (a mask) is set, else of `b`.
        #[inline(always)]
        pub fn select(self, a: Self, b: Self) -> Self {
            unsafe { Self(vbslq_s16(vreinterpretq_u16_s16(self.0), a.0, b.0)) }
        }
        /// Whether any lane of the mask is set.
        #[inline(always)]
        pub fn any(self) -> bool {
            unsafe { vmaxvq_u16(vreinterpretq_u16_s16(self.0)) != 0 }
        }
        /// The low halves of `self` and `o`.
        /// Lanes 0..4 of `self` and `o` interleaved: s0 o0 s1 o1 s2 o2 s3 o3.
        #[inline(always)]
        pub fn zip_lo(self, o: Self) -> Self {
            unsafe { Self(vzip1q_s16(self.0, o.0)) }
        }
        /// Lanes 4..8 of `self` and `o` interleaved.
        #[inline(always)]
        pub fn zip_hi(self, o: Self) -> Self {
            unsafe { Self(vzip2q_s16(self.0, o.0)) }
        }
        /// The 32-bit elements 0, 1 of `self` and `o` interleaved.
        #[inline(always)]
        pub fn zip_lo32(self, o: Self) -> Self {
            unsafe {
                let (a, b) = (vreinterpretq_s32_s16(self.0), vreinterpretq_s32_s16(o.0));
                Self(vreinterpretq_s16_s32(vzip1q_s32(a, b)))
            }
        }
        /// The 32-bit elements 2, 3 of `self` and `o` interleaved.
        #[inline(always)]
        pub fn zip_hi32(self, o: Self) -> Self {
            unsafe {
                let (a, b) = (vreinterpretq_s32_s16(self.0), vreinterpretq_s32_s16(o.0));
                Self(vreinterpretq_s16_s32(vzip2q_s32(a, b)))
            }
        }
        #[inline(always)]
        pub fn combine_lo(self, o: Self) -> Self {
            unsafe { Self(vcombine_s16(vget_low_s16(self.0), vget_low_s16(o.0))) }
        }
        /// The high halves of `self` and `o`.
        #[inline(always)]
        pub fn combine_hi(self, o: Self) -> Self {
            unsafe { Self(vcombine_s16(vget_high_s16(self.0), vget_high_s16(o.0))) }
        }
        /// (v + (1 << (N - 1))) >> N without overflow.
        #[inline(always)]
        pub fn rshr<const N: i32>(self) -> Self {
            unsafe { Self(vrshrq_n_s16::<N>(self.0)) }
        }
    }

    /// ((r0 + r5) - 5 (r1 + r4) + 20 (r2 + r3) + 512) >> 10 in 32 bits,
    /// saturated to i16: the 6-tap filter over unrounded 6-tap sums (luma
    /// sample j). The pairwise sums must fit in i16.
    #[inline(always)]
    pub fn tap6_10(r: [I16x8; 6]) -> I16x8 {
        unsafe {
            let a = vaddq_s16(r[0].0, r[5].0);
            let b = vaddq_s16(r[1].0, r[4].0);
            let c = vaddq_s16(r[2].0, r[3].0);
            let lo = vmlal_n_s16(
                vmlal_n_s16(vmovl_s16(vget_low_s16(a)), vget_low_s16(b), -5),
                vget_low_s16(c),
                20,
            );
            let hi = vmlal_high_n_s16(vmlal_high_n_s16(vmovl_high_s16(a), b, -5), c, 20);
            I16x8(vcombine_s16(
                vqrshrn_n_s32::<10>(lo),
                vqrshrn_n_s32::<10>(hi),
            ))
        }
    }

    /// Transposes an 8x8 matrix of rows.
    #[inline(always)]
    pub fn transpose8(r: &mut [I16x8; 8]) {
        // SAFETY: NEON is part of the aarch64 baseline.
        unsafe {
            let t = |a: int16x8_t, b: int16x8_t| (vtrn1q_s16(a, b), vtrn2q_s16(a, b));
            let (t0, t1) = t(r[0].0, r[1].0);
            let (t2, t3) = t(r[2].0, r[3].0);
            let (t4, t5) = t(r[4].0, r[5].0);
            let (t6, t7) = t(r[6].0, r[7].0);
            let w = |a: int16x8_t, b: int16x8_t| {
                let (a, b) = (vreinterpretq_s32_s16(a), vreinterpretq_s32_s16(b));
                (vtrn1q_s32(a, b), vtrn2q_s32(a, b))
            };
            let (u0, u2) = w(t0, t2);
            let (u1, u3) = w(t1, t3);
            let (u4, u6) = w(t4, t6);
            let (u5, u7) = w(t5, t7);
            let d = |a: int32x4_t, b: int32x4_t| {
                let (a, b) = (vreinterpretq_s64_s32(a), vreinterpretq_s64_s32(b));
                (
                    I16x8(vreinterpretq_s16_s64(vtrn1q_s64(a, b))),
                    I16x8(vreinterpretq_s16_s64(vtrn2q_s64(a, b))),
                )
            };
            (r[0], r[4]) = d(u0, u4);
            (r[1], r[5]) = d(u1, u5);
            (r[2], r[6]) = d(u2, u6);
            (r[3], r[7]) = d(u3, u7);
        }
    }
}

#[cfg(all(target_arch = "x86_64", not(altc_scalar)))]
mod imp {
    use core::arch::x86_64::*;

    #[derive(Clone, Copy)]
    pub struct I16x8(pub __m128i);

    // SAFETY (all functions): SSE2 is part of the x86_64 baseline.
    impl I16x8 {
        #[inline(always)]
        pub fn splat(v: i16) -> Self {
            unsafe { Self(_mm_set1_epi16(v)) }
        }
        /// The low 16 bits of each of eight i32s.
        #[inline(always)]
        pub fn from_i32(a: &[i32; 8]) -> Self {
            unsafe {
                // Sign-extending the low halves first makes the saturating
                // pack a truncation.
                let low = |v: __m128i| _mm_srai_epi32::<16>(_mm_slli_epi32::<16>(v));
                let lo = _mm_loadu_si128(a.as_ptr().cast());
                let hi = _mm_loadu_si128(a.as_ptr().add(4).cast());
                Self(_mm_packs_epi32(low(lo), low(hi)))
            }
        }
        /// `self` with lane 0 replaced by `v`.
        #[inline(always)]
        pub fn with_lane0(self, v: i16) -> Self {
            unsafe { Self(_mm_insert_epi16::<0>(self.0, v as i32)) }
        }
        #[inline(always)]
        pub fn with_lane4(self, v: i16) -> Self {
            unsafe { Self(_mm_insert_epi16::<4>(self.0, v as i32)) }
        }
        #[inline(always)]
        pub fn from_array(a: [i16; 8]) -> Self {
            unsafe { Self(_mm_loadu_si128(a.as_ptr().cast())) }
        }
        #[inline(always)]
        pub fn load_u8(b: &[u8; 8]) -> Self {
            unsafe {
                let v = _mm_loadl_epi64(b.as_ptr().cast());
                Self(_mm_unpacklo_epi8(v, _mm_setzero_si128()))
            }
        }
        #[inline(always)]
        pub fn store_u8(self, b: &mut [u8; 8]) {
            unsafe { _mm_storel_epi64(b.as_mut_ptr().cast(), _mm_packus_epi16(self.0, self.0)) }
        }
        #[inline(always)]
        pub fn add(self, o: Self) -> Self {
            unsafe { Self(_mm_add_epi16(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn sub(self, o: Self) -> Self {
            unsafe { Self(_mm_sub_epi16(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn mul(self, o: Self) -> Self {
            unsafe { Self(_mm_mullo_epi16(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn shl<const N: i32>(self) -> Self {
            unsafe { Self(_mm_slli_epi16::<N>(self.0)) }
        }
        #[inline(always)]
        pub fn shr<const N: i32>(self) -> Self {
            unsafe { Self(_mm_srai_epi16::<N>(self.0)) }
        }
        #[inline(always)]
        pub fn abs(self) -> Self {
            unsafe {
                Self(_mm_max_epi16(
                    self.0,
                    _mm_sub_epi16(_mm_setzero_si128(), self.0),
                ))
            }
        }
        #[inline(always)]
        pub fn min(self, o: Self) -> Self {
            unsafe { Self(_mm_min_epi16(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn max(self, o: Self) -> Self {
            unsafe { Self(_mm_max_epi16(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn lt(self, o: Self) -> Self {
            unsafe { Self(_mm_cmplt_epi16(self.0, o.0)) }
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
        /// Lanes 0..4 of `self` and `o` interleaved: s0 o0 s1 o1 s2 o2 s3 o3.
        #[inline(always)]
        pub fn zip_lo(self, o: Self) -> Self {
            unsafe { Self(_mm_unpacklo_epi16(self.0, o.0)) }
        }
        /// Lanes 4..8 of `self` and `o` interleaved.
        #[inline(always)]
        pub fn zip_hi(self, o: Self) -> Self {
            unsafe { Self(_mm_unpackhi_epi16(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn zip_lo32(self, o: Self) -> Self {
            unsafe { Self(_mm_unpacklo_epi32(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn zip_hi32(self, o: Self) -> Self {
            unsafe { Self(_mm_unpackhi_epi32(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn combine_lo(self, o: Self) -> Self {
            unsafe { Self(_mm_unpacklo_epi64(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn combine_hi(self, o: Self) -> Self {
            unsafe { Self(_mm_unpackhi_epi64(self.0, o.0)) }
        }
        #[inline(always)]
        pub fn rshr<const N: i32>(self) -> Self {
            // (v >> N) + bit N - 1 of v: the rounding without the addition.
            let below = unsafe { Self(_mm_sra_epi16(self.0, _mm_cvtsi32_si128(N - 1))) };
            self.shr::<N>().add(below.and(Self::splat(1)))
        }
    }

    /// ((r0 + r5) - 5 (r1 + r4) + 20 (r2 + r3) + 512) >> 10 in 32 bits,
    /// saturated to i16: the 6-tap filter over unrounded 6-tap sums (luma
    /// sample j). The pairwise sums must fit in i16.
    #[inline(always)]
    pub fn tap6_10(r: [I16x8; 6]) -> I16x8 {
        unsafe {
            let a = _mm_add_epi16(r[0].0, r[5].0);
            let b = _mm_add_epi16(r[1].0, r[4].0);
            let c = _mm_add_epi16(r[2].0, r[3].0);
            // (a, b) pairs times (1, -5) and (c, 0) pairs times (20, 0).
            let k_ab = _mm_set1_epi32((-5i32 << 16) | 1);
            let k_c = _mm_set1_epi32(20);
            let zero = _mm_setzero_si128();
            let round = _mm_set1_epi32(512);
            let half = |ab: __m128i, c0: __m128i| {
                let t = _mm_add_epi32(_mm_madd_epi16(ab, k_ab), _mm_madd_epi16(c0, k_c));
                _mm_srai_epi32::<10>(_mm_add_epi32(t, round))
            };
            let lo = half(_mm_unpacklo_epi16(a, b), _mm_unpacklo_epi16(c, zero));
            let hi = half(_mm_unpackhi_epi16(a, b), _mm_unpackhi_epi16(c, zero));
            I16x8(_mm_packs_epi32(lo, hi))
        }
    }

    /// Transposes an 8x8 matrix of rows.
    #[inline(always)]
    pub fn transpose8(r: &mut [I16x8; 8]) {
        // SAFETY: SSE2 is part of the x86_64 baseline.
        unsafe {
            let lo16 = |a: I16x8, b: I16x8| _mm_unpacklo_epi16(a.0, b.0);
            let hi16 = |a: I16x8, b: I16x8| _mm_unpackhi_epi16(a.0, b.0);
            let (a0, a1) = (lo16(r[0], r[1]), hi16(r[0], r[1]));
            let (a2, a3) = (lo16(r[2], r[3]), hi16(r[2], r[3]));
            let (a4, a5) = (lo16(r[4], r[5]), hi16(r[4], r[5]));
            let (a6, a7) = (lo16(r[6], r[7]), hi16(r[6], r[7]));
            let (b0, b1) = (_mm_unpacklo_epi32(a0, a2), _mm_unpackhi_epi32(a0, a2));
            let (b2, b3) = (_mm_unpacklo_epi32(a1, a3), _mm_unpackhi_epi32(a1, a3));
            let (b4, b5) = (_mm_unpacklo_epi32(a4, a6), _mm_unpackhi_epi32(a4, a6));
            let (b6, b7) = (_mm_unpacklo_epi32(a5, a7), _mm_unpackhi_epi32(a5, a7));
            r[0] = I16x8(_mm_unpacklo_epi64(b0, b4));
            r[1] = I16x8(_mm_unpackhi_epi64(b0, b4));
            r[2] = I16x8(_mm_unpacklo_epi64(b1, b5));
            r[3] = I16x8(_mm_unpackhi_epi64(b1, b5));
            r[4] = I16x8(_mm_unpacklo_epi64(b2, b6));
            r[5] = I16x8(_mm_unpackhi_epi64(b2, b6));
            r[6] = I16x8(_mm_unpacklo_epi64(b3, b7));
            r[7] = I16x8(_mm_unpackhi_epi64(b3, b7));
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(altc_scalar)))]
mod imp {
    use core::arch::wasm32::*;

    #[derive(Clone, Copy)]
    pub struct I16x8(pub v128);

    impl I16x8 {
        #[inline(always)]
        pub fn splat(v: i16) -> Self {
            Self(i16x8_splat(v))
        }
        /// The low 16 bits of each of eight i32s.
        #[inline(always)]
        pub fn from_i32(a: &[i32; 8]) -> Self {
            // SAFETY: reads the 32 bytes of `a`.
            let (lo, hi) = unsafe {
                (
                    v128_load(a.as_ptr().cast()),
                    v128_load(a.as_ptr().add(4).cast()),
                )
            };
            Self(i16x8_shuffle::<0, 2, 4, 6, 8, 10, 12, 14>(lo, hi))
        }
        /// `self` with lane 0 replaced by `v`.
        #[inline(always)]
        pub fn with_lane0(self, v: i16) -> Self {
            Self(i16x8_replace_lane::<0>(self.0, v))
        }
        #[inline(always)]
        pub fn with_lane4(self, v: i16) -> Self {
            Self(i16x8_replace_lane::<4>(self.0, v))
        }
        #[inline(always)]
        pub fn from_array(a: [i16; 8]) -> Self {
            // SAFETY: reads the 16 bytes of `a`.
            unsafe { Self(v128_load(a.as_ptr().cast())) }
        }
        /// Zero-extends 8 bytes.
        #[inline(always)]
        pub fn load_u8(b: &[u8; 8]) -> Self {
            // SAFETY: reads the 8 bytes of `b`.
            unsafe { Self(u16x8_load_extend_u8x8(b.as_ptr())) }
        }
        /// Saturates each lane to 0..=255.
        #[inline(always)]
        pub fn store_u8(self, b: &mut [u8; 8]) {
            let v = u8x16_narrow_i16x8(self.0, self.0);
            // SAFETY: writes the 8 bytes of `b`.
            unsafe { v128_store64_lane::<0>(v, b.as_mut_ptr().cast()) }
        }
        #[inline(always)]
        pub fn add(self, o: Self) -> Self {
            Self(i16x8_add(self.0, o.0))
        }
        #[inline(always)]
        pub fn sub(self, o: Self) -> Self {
            Self(i16x8_sub(self.0, o.0))
        }
        #[inline(always)]
        pub fn mul(self, o: Self) -> Self {
            Self(i16x8_mul(self.0, o.0))
        }
        #[inline(always)]
        pub fn shl<const N: i32>(self) -> Self {
            Self(i16x8_shl(self.0, N as u32))
        }
        /// Arithmetic shift right.
        #[inline(always)]
        pub fn shr<const N: i32>(self) -> Self {
            Self(i16x8_shr(self.0, N as u32))
        }
        #[inline(always)]
        pub fn abs(self) -> Self {
            Self(i16x8_abs(self.0))
        }
        #[inline(always)]
        pub fn min(self, o: Self) -> Self {
            Self(i16x8_min(self.0, o.0))
        }
        #[inline(always)]
        pub fn max(self, o: Self) -> Self {
            Self(i16x8_max(self.0, o.0))
        }
        #[inline(always)]
        pub fn lt(self, o: Self) -> Self {
            Self(i16x8_lt(self.0, o.0))
        }
        #[inline(always)]
        pub fn and(self, o: Self) -> Self {
            Self(v128_and(self.0, o.0))
        }
        #[inline(always)]
        pub fn select(self, a: Self, b: Self) -> Self {
            Self(v128_bitselect(a.0, b.0, self.0))
        }
        #[inline(always)]
        pub fn any(self) -> bool {
            v128_any_true(self.0)
        }
        /// Lanes 0..4 of `self` and `o` interleaved: s0 o0 s1 o1 s2 o2 s3 o3.
        #[inline(always)]
        pub fn zip_lo(self, o: Self) -> Self {
            Self(i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(self.0, o.0))
        }
        /// Lanes 4..8 of `self` and `o` interleaved.
        #[inline(always)]
        pub fn zip_hi(self, o: Self) -> Self {
            Self(i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(self.0, o.0))
        }
        #[inline(always)]
        pub fn zip_lo32(self, o: Self) -> Self {
            Self(i16x8_shuffle::<0, 1, 8, 9, 2, 3, 10, 11>(self.0, o.0))
        }
        #[inline(always)]
        pub fn zip_hi32(self, o: Self) -> Self {
            Self(i16x8_shuffle::<4, 5, 12, 13, 6, 7, 14, 15>(self.0, o.0))
        }
        #[inline(always)]
        pub fn combine_lo(self, o: Self) -> Self {
            Self(i64x2_shuffle::<0, 2>(self.0, o.0))
        }
        #[inline(always)]
        pub fn combine_hi(self, o: Self) -> Self {
            Self(i64x2_shuffle::<1, 3>(self.0, o.0))
        }
        #[inline(always)]
        pub fn rshr<const N: i32>(self) -> Self {
            // (v >> N) + bit N - 1 of v, as the other backends.
            let below = Self(i16x8_shr(self.0, (N - 1) as u32));
            self.shr::<N>().add(below.and(Self::splat(1)))
        }
    }

    /// ((r0 + r5) - 5 (r1 + r4) + 20 (r2 + r3) + 512) >> 10 in 32 bits,
    /// saturated to i16: the 6-tap filter over unrounded 6-tap sums (luma
    /// sample j). The pairwise sums must fit in i16.
    #[inline(always)]
    pub fn tap6_10(r: [I16x8; 6]) -> I16x8 {
        let a = i16x8_add(r[0].0, r[5].0);
        let b = i16x8_add(r[1].0, r[4].0);
        let c = i16x8_add(r[2].0, r[3].0);
        let (m5, m20, round) = (i16x8_splat(-5), i16x8_splat(20), i32x4_splat(512));
        let lo = i32x4_add(
            i32x4_add(i32x4_extend_low_i16x8(a), i32x4_extmul_low_i16x8(b, m5)),
            i32x4_add(i32x4_extmul_low_i16x8(c, m20), round),
        );
        let hi = i32x4_add(
            i32x4_add(i32x4_extend_high_i16x8(a), i32x4_extmul_high_i16x8(b, m5)),
            i32x4_add(i32x4_extmul_high_i16x8(c, m20), round),
        );
        I16x8(i16x8_narrow_i32x4(i32x4_shr(lo, 10), i32x4_shr(hi, 10)))
    }

    /// Transposes an 8x8 matrix of rows.
    #[inline(always)]
    pub fn transpose8(r: &mut [I16x8; 8]) {
        let lo16 = |a: I16x8, b: I16x8| i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(a.0, b.0);
        let hi16 = |a: I16x8, b: I16x8| i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(a.0, b.0);
        let lo32 = |a: v128, b: v128| i32x4_shuffle::<0, 4, 1, 5>(a, b);
        let hi32 = |a: v128, b: v128| i32x4_shuffle::<2, 6, 3, 7>(a, b);
        let lo64 = |a: v128, b: v128| I16x8(i64x2_shuffle::<0, 2>(a, b));
        let hi64 = |a: v128, b: v128| I16x8(i64x2_shuffle::<1, 3>(a, b));
        let (a0, a1) = (lo16(r[0], r[1]), hi16(r[0], r[1]));
        let (a2, a3) = (lo16(r[2], r[3]), hi16(r[2], r[3]));
        let (a4, a5) = (lo16(r[4], r[5]), hi16(r[4], r[5]));
        let (a6, a7) = (lo16(r[6], r[7]), hi16(r[6], r[7]));
        let (b0, b1) = (lo32(a0, a2), hi32(a0, a2));
        let (b2, b3) = (lo32(a1, a3), hi32(a1, a3));
        let (b4, b5) = (lo32(a4, a6), hi32(a4, a6));
        let (b6, b7) = (lo32(a5, a7), hi32(a5, a7));
        r[0] = lo64(b0, b4);
        r[1] = hi64(b0, b4);
        r[2] = lo64(b1, b5);
        r[3] = hi64(b1, b5);
        r[4] = lo64(b2, b6);
        r[5] = hi64(b2, b6);
        r[6] = lo64(b3, b7);
        r[7] = hi64(b3, b7);
    }
}

#[cfg(any(
    altc_scalar,
    not(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))
))]
mod imp {
    #[derive(Clone, Copy)]
    pub struct I16x8(pub [i16; 8]);

    impl I16x8 {
        #[inline(always)]
        fn map2(self, o: Self, f: impl Fn(i16, i16) -> i16) -> Self {
            Self(core::array::from_fn(|i| f(self.0[i], o.0[i])))
        }
        pub fn splat(v: i16) -> Self {
            Self([v; 8])
        }
        /// The low 16 bits of each of eight i32s.
        pub fn from_i32(a: &[i32; 8]) -> Self {
            Self(a.map(|v| v as i16))
        }
        /// `self` with lane 0 replaced by `v`.
        pub fn with_lane0(self, v: i16) -> Self {
            let mut a = self.0;
            a[0] = v;
            Self(a)
        }
        pub fn with_lane4(self, v: i16) -> Self {
            let mut a = self.0;
            a[4] = v;
            Self(a)
        }
        pub fn zip_lo32(self, o: Self) -> Self {
            let (a, b) = (self.0, o.0);
            Self([a[0], a[1], b[0], b[1], a[2], a[3], b[2], b[3]])
        }
        pub fn zip_hi32(self, o: Self) -> Self {
            let (a, b) = (self.0, o.0);
            Self([a[4], a[5], b[4], b[5], a[6], a[7], b[6], b[7]])
        }
        pub fn from_array(a: [i16; 8]) -> Self {
            Self(a)
        }
        pub fn load_u8(b: &[u8; 8]) -> Self {
            Self(b.map(|v| v as i16))
        }
        pub fn store_u8(self, b: &mut [u8; 8]) {
            *b = self.0.map(|v| v.clamp(0, 255) as u8);
        }
        pub fn add(self, o: Self) -> Self {
            self.map2(o, i16::wrapping_add)
        }
        pub fn sub(self, o: Self) -> Self {
            self.map2(o, i16::wrapping_sub)
        }
        pub fn mul(self, o: Self) -> Self {
            self.map2(o, i16::wrapping_mul)
        }
        pub fn shl<const N: i32>(self) -> Self {
            Self(self.0.map(|v| v << N))
        }
        pub fn shr<const N: i32>(self) -> Self {
            Self(self.0.map(|v| v >> N))
        }
        pub fn abs(self) -> Self {
            Self(self.0.map(i16::wrapping_abs))
        }
        pub fn min(self, o: Self) -> Self {
            self.map2(o, i16::min)
        }
        pub fn max(self, o: Self) -> Self {
            self.map2(o, i16::max)
        }
        pub fn lt(self, o: Self) -> Self {
            self.map2(o, |a, b| -((a < b) as i16))
        }
        pub fn and(self, o: Self) -> Self {
            self.map2(o, |a, b| a & b)
        }
        pub fn select(self, a: Self, b: Self) -> Self {
            Self(core::array::from_fn(|i| {
                if self.0[i] != 0 { a.0[i] } else { b.0[i] }
            }))
        }
        pub fn any(self) -> bool {
            self.0.iter().any(|&v| v != 0)
        }
        /// Lanes 0..4 of `self` and `o` interleaved: s0 o0 s1 o1 s2 o2 s3 o3.
        pub fn zip_lo(self, o: Self) -> Self {
            Self(core::array::from_fn(|i| {
                if i % 2 == 0 {
                    self.0[i / 2]
                } else {
                    o.0[i / 2]
                }
            }))
        }
        /// Lanes 4..8 of `self` and `o` interleaved.
        pub fn zip_hi(self, o: Self) -> Self {
            Self(core::array::from_fn(|i| {
                if i % 2 == 0 {
                    self.0[4 + i / 2]
                } else {
                    o.0[4 + i / 2]
                }
            }))
        }
        pub fn combine_lo(self, o: Self) -> Self {
            Self(core::array::from_fn(|i| {
                if i < 4 { self.0[i] } else { o.0[i - 4] }
            }))
        }
        pub fn combine_hi(self, o: Self) -> Self {
            Self(core::array::from_fn(|i| {
                if i < 4 { self.0[i + 4] } else { o.0[i] }
            }))
        }
        pub fn rshr<const N: i32>(self) -> Self {
            Self(self.0.map(|v| ((v as i32 + (1 << (N - 1))) >> N) as i16))
        }
    }

    /// ((r0 + r5) - 5 (r1 + r4) + 20 (r2 + r3) + 512) >> 10 in 32 bits,
    /// saturated to i16: the 6-tap filter over unrounded 6-tap sums (luma
    /// sample j). The pairwise sums must fit in i16.
    pub fn tap6_10(r: [I16x8; 6]) -> I16x8 {
        I16x8(core::array::from_fn(|i| {
            let t = |k: usize| r[k].0[i] as i32;
            let v = (t(0) + t(5) - 5 * (t(1) + t(4)) + 20 * (t(2) + t(3)) + 512) >> 10;
            v.clamp(i16::MIN as i32, i16::MAX as i32) as i16
        }))
    }

    /// Transposes an 8x8 matrix of rows.
    pub fn transpose8(r: &mut [I16x8; 8]) {
        let m = r.map(|v| v.0);
        *r = core::array::from_fn(|j| I16x8(core::array::from_fn(|i| m[i][j])));
    }
}

pub use imp::{I16x8, tap6_10, transpose8};

impl I16x8 {
    /// Clamps each lane to -t..=t.
    #[inline(always)]
    pub fn clamp_sym(self, t: Self) -> Self {
        self.min(t).max(I16x8::splat(0).sub(t))
    }
}
