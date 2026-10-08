//! Type-safe wrappers that simulate native floating-point environment in Rust.
//!
//! These types shouldn't exist in the C port, since C provides native capabilities for changing
//! rounding modes and accessing exception flags. (Though YMMV if Wasm adds instructions with static
//! rounding modes.)

use crate::{Exceptions, RoundingMode};
use core::arch::asm;
use core::cell::Cell;
use core::cmp::Ordering;
use core::fmt::{self, Debug, Display, Formatter, LowerHex};
use core::ops::{Add, BitAnd, BitOr, BitXorAssign, Div, Mul, Neg, Not, Shr, Sub};

/// Type-safe wrapper around `f32`.
#[derive(Clone, Copy, Debug)]
pub struct F32(f32);

/// Type-safe wrapper around `f64`.
#[derive(Clone, Copy, Debug)]
pub struct F64(f64);

/// Common interface for [`F32`] and [`F64`].
pub trait Native:
    Copy
    + Debug
    + Neg<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Compare
{
    type Bits: Bits;
    const ZERO: Self;
    const HALF: Self;
    const ONE: Self;
    const TWO: Self;
    const INFINITY: Self;
    const MIN_EXPONENT: i32; // smallest normal exponent, differs from std's `MIN_EXP`!
    const TWOP_MAXE: Self; // precomputed constant to avoid this arithmetic setting flags in runtime
    const TWOP_NEG_MAXE_SPLIT: Self; // `2^(-(maxe + 1) / 2)`, used in multiplication
    const NAN_QUIETNESS_BIT: Self::Bits; // machine-independent, just 2^k
    const EXPONENT_MASK: Self::Bits;
    const MANTISSA_MASK: Self::Bits;
    const MANTISSA_LEN: u32; // the number of digits in the mantissa, *excluding* the hidden bit
    const MIN_EXP: i32;
    fn is_nan(self) -> bool;
    /// Checks whether the value represents a signaling NaN on the current machine.
    ///
    /// This can differ from whether the same bit sequence represents a signaling NaN in the
    /// emulated RISC-V machine! Specifically, some platforms, like MIPS, have an inverted quietness
    /// bit in the NaN representation. `is_native_signaling_nan` is intended to be used on native
    /// operation outputs, [`Float::is_emulated_signaling_nan`] is intended for high-level inputs.
    fn is_native_signaling_nan(self) -> bool;
    fn is_finite(self) -> bool;
    fn is_infinite(self) -> bool;
    fn is_sign_negative(self) -> bool;
    fn abs(self) -> Self;
    fn to_bits(self) -> Self::Bits;
    fn from_bits(bits: Self::Bits) -> Self;
    /// Add a value to the bitwise representation.
    fn nudge(self, offset: i8) -> Self;
    fn bottom_bit(self) -> bool;
    /// Extract the exponent field of the bit value. Valid even for non-finite numbers.
    ///
    /// Note that the unbiased exponent is `biased_exponent - 1 - MIN_EXP`, not
    /// `biased_exponent - MIN_EXP`, due to the presence of subnormal values.
    fn biased_exponent(self) -> u32;
    fn mul_add(self, b: Self, c: Self) -> Self;
    /// Compute `self^n` without setting flags. Useful only for constants.
    fn powi(self, n: i32) -> Self;
}

macro_rules! define_methods {
    (
        $ty:ident => $native:ident, $bits:ident,
        twop_neg_maxe_split = $twop_neg_maxe_split:literal,
        nan_quietness_bit = $nan_quietness_bit:literal,
        exponent_mask = $exponent_mask:literal,
        mantissa_mask = $mantissa_mask:literal
    ) => {
        impl Native for $ty {
            type Bits = $bits;
            const ZERO: Self = Self(0.0);
            const HALF: Self = Self(0.5);
            const ONE: Self = Self(1.0);
            const TWO: Self = Self(2.0);
            const INFINITY: Self = Self($native::INFINITY);
            const MIN_EXPONENT: i32 = $native::MIN_EXP - 1;
            const TWOP_MAXE: Self = Self(($native::MAX / 2.0).next_up());
            const TWOP_NEG_MAXE_SPLIT: Self = Self($twop_neg_maxe_split);
            const NAN_QUIETNESS_BIT: Self::Bits = $nan_quietness_bit;
            const EXPONENT_MASK: Self::Bits = $exponent_mask; // f32::EXPONENT_MASK is unstable
            const MANTISSA_MASK: Self::Bits = $mantissa_mask; // f32::MANTISSA_MASK is unstable
            const MANTISSA_LEN: u32 = $native::MANTISSA_DIGITS - 1;
            const MIN_EXP: i32 = $native::MIN_EXP;
            fn is_nan(self) -> bool {
                self.0.is_nan()
            }
            fn is_native_signaling_nan(self) -> bool {
                // This bit check needs to be inverted on MIPS and such!
                self.is_nan() && self.to_bits() & Self::NAN_QUIETNESS_BIT == 0
            }
            fn is_finite(self) -> bool {
                self.0.is_finite()
            }
            fn is_infinite(self) -> bool {
                self.0.is_infinite()
            }
            fn is_sign_negative(self) -> bool {
                self.0.is_sign_negative()
            }
            fn abs(self) -> Self {
                // IEEE-754 says abs is defined on bit values and doesn't affect exceptions.
                Self(self.0.abs())
            }
            fn to_bits(self) -> $bits {
                self.0.to_bits()
            }
            fn from_bits(bits: $bits) -> Self {
                Self($native::from_bits(bits))
            }
            fn nudge(self, offset: i8) -> Self {
                Self::from_bits(self.to_bits().wrapping_add(offset as Self::Bits))
            }
            fn bottom_bit(self) -> bool {
                self.to_bits() & 1 != 0
            }
            fn biased_exponent(self) -> u32 {
                ((self.to_bits() & Self::EXPONENT_MASK) >> Self::MANTISSA_LEN) as u32
            }
            fn mul_add(self, b: Self, c: Self) -> Self {
                Self(self.0.mul_add(b.0, c.0))
            }
            fn powi(self, n: i32) -> Self {
                Self(self.0.powi(n))
            }
        }

        impl Neg for $ty {
            type Output = Self;
            fn neg(self) -> Self {
                // IEEE-754 says negation is defined on bit values and doesn't affect exceptions.
                Self(-self.0)
            }
        }

        impl Display for $ty {
            fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.0)
            }
        }
    };
}

define_methods!(
    F32 => f32, u32,
    twop_neg_maxe_split = 5.421011e-20,
    nan_quietness_bit = 0x400000,
    exponent_mask = 0x7f800000,
    mantissa_mask = 0x7fffff
);
define_methods!(
    F64 => f64, u64,
    twop_neg_maxe_split = 7.458340731200207e-155,
    nan_quietness_bit = 0x8000000000000,
    exponent_mask = 0x7ff0000000000000,
    mantissa_mask = 0xfffffffffffff
);

/// Common interface for `u32` and `u64`.
pub trait Bits:
    Copy
    + LowerHex
    + BitXorAssign
    + Add<Output = Self>
    + BitAnd<Output = Self>
    + BitOr<Output = Self>
    + Shr<u32, Output = Self>
    + Not<Output = Self>
    + Eq
    + Ord
{
    const ZERO: Self;
    const ONE: Self;
    fn trailing_zeros(self) -> u32;
}

macro_rules! define_bits {
    ($ty:ident) => {
        impl Bits for $ty {
            const ZERO: Self = 0;
            const ONE: Self = 1;
            fn trailing_zeros(self) -> u32 {
                self.trailing_zeros()
            }
        }
    };
}

define_bits!(u32);
define_bits!(u64);

/// Native floating-point environment.
#[derive(Clone, Copy, Default)]
pub struct Env {
    pub exceptions: Exceptions,
    pub round: RoundingMode,
}

impl Env {
    fn to_mxcsr(self) -> u32 {
        self.exceptions.bits()
            | (0b111111 << 7)
            | (match self.round {
                RoundingMode::ToNearest => 0,
                RoundingMode::Floor => 1,
                RoundingMode::Ceil => 2,
                RoundingMode::Trunc => 3,
                RoundingMode::ToNearestTiesToMaxMagnitude => panic!("not supported by x86"),
            } << 13)
    }

    fn from_mxcsr(mask: u32) -> Self {
        Self {
            exceptions: Exceptions::from_bits(mask & 0b111101).unwrap(),
            round: match (mask >> 13) & 3 {
                0 => RoundingMode::ToNearest,
                1 => RoundingMode::Floor,
                2 => RoundingMode::Ceil,
                3 => RoundingMode::Trunc,
                _ => unreachable!(),
            },
        }
    }
}

thread_local! {
    static ENV: Cell<Env> = Cell::new(Env::default());
}

/// Read the native floating-point environment.
pub fn get_env() -> Env {
    ENV.get()
}

/// Write the native floating-point environment.
pub fn set_env(env: Env) {
    ENV.set(env)
}

macro_rules! define_binop {
    ($trait:ident($method:ident) for $ty:ident => $insn:ident) => {
        impl $trait for $ty {
            type Output = Self;

            fn $method(self, rhs: Self) -> Self {
                #[cfg(feature = "explore")]
                return Self($trait::$method(self.0, rhs.0));

                let mut out = Self(0.0);
                let mut env = get_env().to_mxcsr();
                unsafe {
                    asm!(
                        "ldmxcsr [{2}]",
                        concat!(stringify!($insn), " {0}, {1}"),
                        "stmxcsr [{2}]",
                        "ldmxcsr [{3}]", // don't forget to restore default environment to avoid UB
                        inout(xmm_reg) self.0 => out.0,
                        in(xmm_reg) rhs.0,
                        in(reg) &mut env,
                        in(reg) &0x1f80u32,
                    );
                }
                set_env(Env::from_mxcsr(env));
                out
            }
        }
    };
}

define_binop!(Add(add) for F32 => addss);
define_binop!(Sub(sub) for F32 => subss);
define_binop!(Mul(mul) for F32 => mulss);
define_binop!(Div(div) for F32 => divss);
define_binop!(Add(add) for F64 => addsd);
define_binop!(Sub(sub) for F64 => subsd);
define_binop!(Mul(mul) for F64 => mulsd);
define_binop!(Div(div) for F64 => divsd);

pub trait Compare {
    fn signaling_cmp(&self, rhs: &Self) -> Option<Ordering>;
    fn quiet_cmp(&self, rhs: &Self) -> Option<Ordering>;
    fn fast_cmp(&self, rhs: &Self) -> Ordering;
    fn inner(&self) -> impl PartialOrd;
}

macro_rules! define_comparison_type {
    ($(#[$meta:meta])* $ty:ident => $method:ident$(, $wrap:ident)?) => {
        $(#[$meta])*
        pub struct $ty<T>(pub T);

        impl<T: Compare> PartialOrd for $ty<T> {
            fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
                #[cfg(feature = "explore")]
                return self.0.inner().partial_cmp(&rhs.0.inner());

                $($wrap)?(self.0.$method(&rhs.0))
            }

            #[cfg(feature = "explore")]
            fn lt(&self, rhs: &Self) -> bool {
                self.0.inner() < rhs.0.inner()
            }

            #[cfg(feature = "explore")]
            fn le(&self, rhs: &Self) -> bool {
                self.0.inner() <= rhs.0.inner()
            }

            #[cfg(feature = "explore")]
            fn gt(&self, rhs: &Self) -> bool {
                self.0.inner() > rhs.0.inner()
            }

            #[cfg(feature = "explore")]
            fn ge(&self, rhs: &Self) -> bool {
                self.0.inner() >= rhs.0.inner()
            }
        }

        impl<T: Compare> PartialEq for $ty<T> {
            fn eq(&self, rhs: &Self) -> bool {
                #[cfg(feature = "explore")]
                return self.0.inner() == rhs.0.inner();

                self.partial_cmp(rhs) == Some(Ordering::Equal)
            }
        }
    };
}

define_comparison_type!(
    /// Comparisons that raise exceptions when qNaN is involved.
    Signaling => signaling_cmp
);
define_comparison_type!(
    /// Comparisons that don't raise exceptions when qNaN is involved.
    Quiet => quiet_cmp
);
define_comparison_type!(
    /// Comparisons that assume inputs are not NaN.
    ///
    /// Intended to be implemented either with quiet or signaling comparisons depending on which is
    /// more efficient. The Rust emulation panics for testing purposes.
    Fast => fast_cmp, Some
);

impl<T: Compare> Ord for Fast<T> {
    fn cmp(&self, rhs: &Self) -> Ordering {
        self.0.fast_cmp(&rhs.0)
    }
}

impl<T: Compare> Eq for Fast<T> {}

macro_rules! define_comparison {
    ($name:ident => $signaling:ident, $quiet:ident) => {
        impl Compare for $name {
            define_comparison_method!(signaling_cmp => $signaling);
            define_comparison_method!(quiet_cmp => $quiet);
            fn fast_cmp(&self, rhs: &Self) -> Ordering {
                self.inner().partial_cmp(&rhs.inner()).expect("NaN input to fast comparison")
            }
            fn inner(&self) -> impl PartialOrd {
                self.0
            }
        }
    };
}

macro_rules! define_comparison_method {
    ($name:ident => $insn:ident) => {
        fn $name(&self, rhs: &Self) -> Option<Ordering> {
            let mut out = Some(Ordering::Greater);
            let mut env = get_env().to_mxcsr();
            unsafe {
                asm!(
                    "ldmxcsr [{2}]",
                    concat!(stringify!($insn), " {0}, {1}"),
                    "stmxcsr [{2}]",
                    "ldmxcsr [{3}]", // don't forget to restore default environment to avoid UB
                    "jp {4}",
                    "je {5}",
                    "jb {6}",
                    in(xmm_reg) self.0,
                    in(xmm_reg) rhs.0,
                    in(reg) &mut env,
                    in(reg) &0x1f80u32,
                    label { out = None },
                    label { out = Some(Ordering::Equal) },
                    label { out = Some(Ordering::Less) },
                );
            }
            set_env(Env::from_mxcsr(env));
            out
        }
    };
}

define_comparison!(F32 => comiss, ucomiss);
define_comparison!(F64 => comisd, ucomisd);
