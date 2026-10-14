//! altc264: an H.264/AVC decoder with MVC (Stereo High) support.

#![forbid(unsafe_op_in_unsafe_fn)]

/// The CPU has BMI1, BMI2, LZCNT and POPCNT (x86-64-v3's scalar bit
/// operations), checked once.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) fn bit_ops() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering::Relaxed};
    // 0: not checked yet, 1: no, 2: yes.
    static HAVE: AtomicU8 = AtomicU8::new(0);
    match HAVE.load(Relaxed) {
        0 => {
            let yes = std::is_x86_feature_detected!("bmi1")
                && std::is_x86_feature_detected!("bmi2")
                && std::is_x86_feature_detected!("lzcnt")
                && std::is_x86_feature_detected!("popcnt");
            HAVE.store(1 + yes as u8, Relaxed);
            yes
        }
        h => h == 2,
    }
}

/// A function whose body is also compiled, on x86_64, with the bit
/// operations of [`bit_ops`] (one-instruction shifts by a register count,
/// leading zeros and population counts) and used when the CPU has them,
/// as decoders built for x86-64-v3 do throughout. Generic parameters come
/// in brackets: their declarations, then the arguments.
macro_rules! bit_ops_fn {
    (
        $(#[$m:meta])*
        $vis:vis fn $name:ident [$($g:tt)*] [$($ga:tt)*] ($($a:ident: $t:ty),* $(,)?) $(-> $r:ty)? $body:block
    ) => {
        $(#[$m])*
        $vis fn $name<$($g)*>($($a: $t),*) $(-> $r)? {
            #[cfg(target_arch = "x86_64")]
            if $crate::bit_ops() {
                #[target_feature(enable = "bmi1,bmi2,lzcnt,popcnt")]
                fn fast<$($g)*>($($a: $t),*) $(-> $r)? $body
                // SAFETY: the CPU has the features (bit_ops).
                return unsafe { fast::<$($ga)*>($($a),*) };
            }
            $body
        }
    };
}

pub mod bits;
pub mod cabac;
pub mod cabac_se;
pub mod cabac_tables;
pub mod cavlc;
pub mod cavlc_tables;
pub mod deblock;
pub mod decoder;
pub mod dpb;
pub mod fmo;
pub mod inter;
#[cfg(all(target_arch = "x86_64", not(altc_scalar)))]
mod inter_avx2;
#[cfg(all(target_arch = "aarch64", not(altc_scalar)))]
mod inter_neon;
pub mod intra;
pub mod mb;
pub mod nal;
pub mod picture;
pub mod poc;
mod pool;
pub mod progress;
pub mod ps;
pub mod sei;
mod simd;
mod simd8;
pub mod slice;
pub mod slice_decoder;
pub mod sp;
pub mod stereo;
pub mod transform;

use core::fmt;

/// Bitstream errors. Field names follow the spec's syntax element names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// The syntax structure ran past the end of its NAL unit.
    Truncated,
    OutOfRange {
        field: &'static str,
        value: i64,
    },
    Invalid(&'static str),
    Unsupported(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => f.write_str("truncated syntax structure"),
            Self::OutOfRange { field, value } => write!(f, "{field} out of range: {value}"),
            Self::Invalid(what) => write!(f, "invalid {what}"),
            Self::Unsupported(what) => write!(f, "unsupported: {what}"),
        }
    }
}

impl std::error::Error for Error {}
