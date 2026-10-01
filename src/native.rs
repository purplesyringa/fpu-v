//! Type-safe wrappers that simulate native floating-point environment in Rust.
//!
//! These types shouldn't exist in the C port, since C provides native capabilities for changing
//! rounding modes and accessing exception flags. (Though YMMV if Wasm adds instructions with static
//! rounding modes.)

use crate::{Exceptions, RoundingMode};
use core::arch::asm;
use core::cell::Cell;
use core::cmp::Ordering;
use core::fmt::{self, Display, Formatter};
use core::ops::{Add, BitXorAssign, Div, Mul, Neg, Sub};

/// Type-safe wrapper around `f32`.
#[derive(Clone, Copy, Debug)]
pub struct F32(f32);

/// Type-safe wrapper around `f64`.
#[derive(Clone, Copy, Debug)]
pub struct F64(f64);

/// Common interface for [`F32`] and [`F64`].
pub trait Native:
    Copy
    + Neg<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + PartialOrd
    + PartialEq
{
    type Bits: Copy + BitXorAssign + Eq;
    const ZERO: Self;
    const HALF: Self;
    const TWO: Self;
    const INFINITY: Self;
    const MAX: Self;
    const TWOP_MAXE: Self; // precomputed constant to avoid this arithmetic setting flags in runtime
    fn is_nan(self) -> bool;
    // Avoid defining `is_signaling_nan` or alike, because that's non-deterministic based on the
    // platform (since MIPS uses an inverted representation). Use the method on `types::*` instead.
    fn is_finite(self) -> bool;
    fn is_infinite(self) -> bool;
    fn is_sign_negative(self) -> bool;
    fn abs(self) -> Self;
    fn to_bits(self) -> Self::Bits;
    fn from_bits(bits: Self::Bits) -> Self;
    /// Add a value to the bitwise representation.
    fn nudge(self, offset: i8) -> Self;
}

macro_rules! define_methods {
    ($ty:ident => $native:ident, $bits:ident) => {
        impl Native for $ty {
            type Bits = $bits;
            const ZERO: Self = Self(0.0);
            const HALF: Self = Self(0.5);
            const TWO: Self = Self(2.0);
            const INFINITY: Self = Self($native::INFINITY);
            const MAX: Self = Self($native::MAX);
            const TWOP_MAXE: Self = Self(($native::MAX / 2.0).next_up());
            fn is_nan(self) -> bool {
                self.0.is_nan()
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

define_methods!(F32 => f32, u32);
define_methods!(F64 => f64, u64);

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

macro_rules! define_comparison {
    ($ty:ident => $insn:ident) => {
        impl PartialOrd for $ty {
            fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
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
        }

        impl PartialEq for $ty {
            fn eq(&self, rhs: &Self) -> bool {
                self.partial_cmp(rhs) == Some(Ordering::Equal)
            }
        }
    };
}

define_comparison!(F32 => ucomiss);
define_comparison!(F64 => ucomisd);
