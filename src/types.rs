use crate::native::{self, Native};
use bitflags::bitflags;
use core::fmt::{self, Debug, Formatter};

/// Type-safe wrapper around `f32`.
#[derive(Clone, Copy)]
pub struct F32(pub(crate) u32);

/// Type-safe wrapper around `f64`.
#[derive(Clone, Copy)]
pub struct F64(pub(crate) u64);

/// Type-safe [`F32`] literal, guaranteed to be non-NaN.
#[macro_export]
macro_rules! F32 {
    ($x:literal) => {
        <$crate::F32 as $crate::Float>::from_bits(f32::to_bits($x))
    };
}

/// Type-safe [`F64`] literal, guaranteed to be non-NaN.
#[macro_export]
macro_rules! F64 {
    ($x:literal) => {
        <$crate::F64 as $crate::Float>::from_bits(f64::to_bits($x))
    };
}

/// Common interface for [`F32`] and [`F64`].
pub trait Float: Copy + Debug {
    type Native: Native;

    const CANONICAL_NAN: Self;
    const CANONICAL_SIGNALING_NAN: Self;

    /// Bitcast from an integer value.
    ///
    /// Can represent `NaN`, in which case the payload is preserved.
    fn from_bits(x: <Self::Native as Native>::Bits) -> Self;

    /// Bitcast to an integer value.
    fn to_bits(self) -> <Self::Native as Native>::Bits;

    /// Convert from a native value, preserving `NaN` payload.
    ///
    /// Note that on hosts with inverted `NaN` quietness, like MIPS, this will preserve the bitwise
    /// value of the bit, not its meaning.
    fn from_native_transmuting_nan(x: Self::Native) -> Self;

    // Avoid defining `to_native` -- use `Env::to_native` instead.

    // /// Convert to a native value.
    // ///
    // /// If the value is `NaN`, the resulting `NaN` may be non-deterministically signaling or quiet
    // /// compared to the original `NaN` (because on MIPS and PA-RISC the signaling bit is inverted).
    // fn to_native_transmuting_nan(self) -> Self::Native;

    /// Check whether the value is a signaling `NaN`.
    ///
    /// This is valid for values obtained from [`Float::from_native_transmuting_nan`], but not for
    /// those obtained by bitwise cast after an FP operation, because NaN representation can differ
    /// between RISC-V and the host. See [`Native::is_is_native_signaling_nan`] for more info.
    fn is_emulated_signaling_nan(self) -> bool;
}

impl F32 {
    /// Bitcast from a value boxed in `f64`.
    ///
    /// Can represent `NaN`, in which case the payload is preserved.
    pub const fn from_boxed(x: u64) -> Self {
        if x >= 0xffffffff00000000 {
            Self(x as u32)
        } else {
            Self::CANONICAL_NAN
        }
    }
}

macro_rules! define_methods {
    ($ty:ident => nan = $nan:literal, signaling = $signaling:expr) => {
        impl Float for $ty {
            type Native = native::$ty;

            const CANONICAL_NAN: Self = Self($nan);
            const CANONICAL_SIGNALING_NAN: Self = Self($nan ^ Self::Native::NAN_QUIETNESS_BIT);

            fn from_bits(x: <Self::Native as Native>::Bits) -> Self {
                Self(x)
            }

            fn to_bits(self) -> <Self::Native as Native>::Bits {
                self.0
            }

            fn from_native_transmuting_nan(x: Self::Native) -> Self {
                Self(x.to_bits())
            }

            // fn to_native_transmuting_nan(self) -> Self::Native {
            //     Self::Native::from_bits(self.0)
            // }

            fn is_emulated_signaling_nan(self) -> bool {
                $signaling.contains(&((self.to_bits() << 1) >> 1))
            }
        }

        impl Debug for $ty {
            fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                write!(
                    f,
                    "{}({} as {:#x})",
                    stringify!($ty),
                    <Self as Float>::Native::from_bits(self.0),
                    self.0
                )
            }
        }
    };
}

define_methods!(F32 => nan = 0x7fc00000, signaling = 0x7f800001..0x7fc00000);
define_methods!(F64 => nan = 0x7ff8000000000000, signaling = 0x7ff0000000000001..0x7ff8000000000000);

/// Floating-point environment.
///
/// If the host supports floating-point exceptions, the source of truth is the native floating-point
/// environment. Otherwise, they are stored here.
///
/// If the host supports rounding modes, the native rounding mode must always be in sync with the
/// emulated rounding mode, with the exception of RMM, which may be represented as NE in native
/// flags if the host doesn't support RMM.
///
/// Due to the need for this synchronization, this type should not generally be mutated directly.
pub struct Env {
    features: HostFeatures,
    exceptions: Exceptions,
    round: RoundingMode,
}

/// Features supported by the host.
#[derive(Clone, Copy, Debug)]
pub struct HostFeatures {
    /// Host supports exceptions.
    pub exceptions: bool,
    /// Host supports basic IEEE-754 rounding modes.
    pub round: bool,
    /// Host supports RMM rounding mode. Implies `round`.
    pub rmm: bool,
    /// Host has an inverted definition of qNaN vs sNaN.
    pub inverted_nan_quietness: bool,
    /// Host automatically canonicalizes NaN on FP operations. This means that it produces a bitwise
    /// value equivalent to the RISC-V NaN, not that it produces something with similar semantics!
    pub nan_canonicalization: bool,
    /// Host detects underflow "after rounding", not "before rounding". Implies `exceptions`.
    pub underflow_after_rounding: bool,
    /// Host supports FMA.
    pub fma: bool,
}

bitflags! {
    /// Floating-point exceptions.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct Exceptions: u32 {
        // Bit values chosen for similarity with x86 to make `native` easier to implement, they
        // don't have to look this way in the C port.
        const INVALID = 1;
        const DIVIDE_BY_ZERO = 4;
        const OVERFLOW = 8;
        const UNDERFLOW = 0x10;
        const INEXACT = 0x20;
    }
}

/// Emulated rounding mode.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum RoundingMode {
    #[default]
    ToNearest,
    Floor,
    Ceil,
    Trunc,
    ToNearestTiesToMaxMagnitude,
}

impl Env {
    /// Initialize from host features.
    pub fn new(features: HostFeatures) -> Self {
        Self {
            features,
            exceptions: Exceptions::default(),
            round: RoundingMode::default(),
        }
    }

    /// Get features supported by the host.
    pub fn features(&self) -> &HostFeatures {
        &self.features
    }

    /// Raise all exceptions in the mask.
    pub fn raise(&mut self, mask: Exceptions) {
        if self.features.exceptions {
            let mut env = native::get_env();
            env.exceptions |= mask;
            native::set_env(env);
        } else {
            self.exceptions |= mask;
        }
    }

    /// Get active rounding mode.
    pub fn get_rounding_mode(&self) -> RoundingMode {
        self.round
    }

    /// Change active rounding mode.
    pub fn set_rounding_mode(&mut self, round: RoundingMode) {
        self.round = round;
        if !self.features.round {
            return;
        }
        let mut env = native::get_env();
        env.round = if round == RoundingMode::ToNearestTiesToMaxMagnitude && !self.features.rmm {
            RoundingMode::ToNearest
        } else {
            round
        };
        native::set_env(env);
    }

    /// If the active rounding mode needs to be emulated, returns it. In this case, the host mode is
    /// guaranteed to be round-to-nearest.
    pub fn emulate_rounding_mode(&self) -> Option<RoundingMode> {
        if (self.round == RoundingMode::ToNearestTiesToMaxMagnitude && !self.features.rmm)
            || (self.round != RoundingMode::ToNearest && !self.features.round)
        {
            Some(self.round)
        } else {
            None
        }
    }

    /// Get raised exceptions.
    ///
    /// This function shouldn't be used except for testing. Since exceptions accumulate, the
    /// returned flags denote exceptions raised since the beginning of the execution, not during the
    /// last operation, so using it in calculations can effectively trigger false positives.
    pub fn get_exceptions(&self) -> Exceptions {
        if self.features.exceptions && !cfg!(feature = "explore") {
            native::get_env().exceptions
        } else {
            self.exceptions
        }
    }

    /// Load host exception flags, if available.
    pub fn save_host_exceptions(&self) -> Option<Exceptions> {
        if self.features.exceptions {
            Some(native::get_env().exceptions)
        } else {
            None
        }
    }

    /// Restore host exception flags, if saved.
    pub fn restore_host_exceptions(&self, ex: Option<Exceptions>) {
        if self.features.exceptions {
            let mut env = native::get_env();
            env.exceptions = ex.unwrap();
            native::set_env(env);
        }
    }

    /// Convert a native value to a float, canonicalizing `NaN` payload if the host doesn't
    /// canonicalize NaNs on FP operations automatically.
    ///
    /// Note that this function shouldn't be used if an input is passed to the output directly
    /// without involving an FP operation, since that can cause a non-canonical NaN (with the sign
    /// bit set, or with the quiet bit off, or with a non-zero payload) to propagate. There is no
    /// helper method for such a scenario because that isn't expected to arise.
    pub fn from_native_canonicalizing_nan_after_op<F: Float>(&self, x: F::Native) -> F {
        #[cfg(feature = "explore")]
        return F::from_native_transmuting_nan(x);

        // This check both validates that this function is not directly applied to function inputs
        // without passing through an FP op (bruteforce tests should eventually trigger such
        // a condition with an sNaN input) and ensures that the NaN can be canonicalized
        // efficiently.
        assert!(
            !x.is_native_signaling_nan(),
            "unexpected signaling NaN after FP operation",
        );
        if self.features.nan_canonicalization {
            F::from_native_transmuting_nan(x)
        } else {
            if x.is_nan() {
                F::CANONICAL_NAN
            } else {
                F::from_bits(x.to_bits())
            }
            // As written, this test compiles to a comparison and a (hopefully rarely taken) jump.
            // This has ideal throughput and doesn't affect latency when the jump is well-predicted,
            // but can cause trouble if branch prediction doesn't work as well.
            //
            // Depending on the architecture, there can be ways to improve this case at the cost of
            // latency. How useful that is depends on the data, so I'm not sure which way is better:
            // as always, benchmarking is your friend. The branchless implementations follow, mostly
            // because it's an interesting puzzle if I'm being honest.
            //
            // AVX-512 provides a direct instruction for doing this:
            //     vrangess x, CANONICAL_NAN, x, 4
            // With AVX only, we can use:
            //     vcmpunordss tmp, x, x
            //     vblendvps x, x, CANONICAL_NAN, tmp
            // ...and similarly for SSE 4.1. There is also a clever implementation for SSE,
            // borrowing the approach V8 uses in vectorized fmin/fmax [1]:
            //     xorps tmp, tmp
            //     cmpunordss tmp, x
            //     andps tmp, !CANONICAL_NAN
            //     andnps x, tmp
            // For non-NaN values, this is a no-op. For NaNs, `vcmpunordss` returns a full mask,
            // `andps` reduces it to `!CANONICAL_NAN`, and `andnps` resolves to `x & CANONICAL_NAN`.
            // This ends up producing the right value because `CANONICAL_NAN` is a submask of all
            // quiet NaNs, and this function isn't invoked with signaling NaNs. This is essentially
            // a pre-SSE 4.1 implementation of `blend`, but without having to mask twice by using
            // the structure of the problem.
            //
            // On ARM, this is generally unnecessary, as VFPv3 has a "default NaN" mode where all
            // NaN-producing arithmetic operations return a canonical NaN compatible with RISC-V,
            // regardless of whether they are forwarding a qNaN input, handling sNaN input, or
            // generating a new qNaN due to an invalid operation exception. [2] If that's not
            // possible, a single-instruction implementation exists:
            //     fminnm x, CANONICAL_NAN, x
            //
            // [1]: https://github.com/v8/v8/blob/19be4913881bb02c5d9b4f1c7547ee2d1273120b/src/compiler/backend/x64/code-generator-x64.cc#L2542
            // [2]: https://support.arm.com/documentation/ddi0406/c/Application-Level-Architecture/Application-Level-Programmers--Model/Floating-point-data-types-and-arithmetic/NaN-handling-and-the-Default-NaN
        }
    }

    /// Convert a float to a native value, possibly naturalizing `NaN`.
    ///
    /// Some environments (MIPS and PA-RISC) have a flipped definition of the quietness NaN bit,
    /// which can cause exceptions to be raised inadvertently if the float is simply transmuted.
    ///
    /// This function guarantees that the returned value behaves the same way as the correct value
    /// in *arithmetic*, but it doesn't actually invert the quietness bit if the host doesn't
    /// support exceptions, so the quietness of the resulting number may still be incorrect. Don't
    /// check the quietness of the native value -- use [`Float::is_emulated_signaling_nan`] instead.
    pub fn to_native<T: Float>(&self, x: T) -> T::Native {
        let mut bits = x.to_bits();
        if self.features.exceptions
            && self.features.inverted_nan_quietness
            && T::Native::from_bits(bits).is_nan()
        {
            bits ^= T::Native::NAN_QUIETNESS_BIT;
        }
        T::Native::from_bits(bits)
    }

    /// Compute `a * b + c`, rounding once.
    ///
    /// Panics if the FMA feature is disabled.
    pub fn mul_add<T: Native>(&self, a: T, b: T, c: T) -> T {
        assert!(self.features.fma, "FMA is disabled");
        a.mul_add(b, c)
    }
}

impl HostFeatures {
    pub const X86: Self = Self {
        exceptions: true,
        round: true,
        rmm: false,
        inverted_nan_quietness: false,
        nan_canonicalization: false,
        underflow_after_rounding: true,
        fma: true,
    };

    pub const WASM: Self = Self {
        exceptions: false,
        round: false,
        rmm: false,
        inverted_nan_quietness: false,
        nan_canonicalization: false,
        underflow_after_rounding: false,
        fma: false,
    };

    pub const ONLY_ROUNDING: Self = Self {
        exceptions: false,
        round: true,
        rmm: false,
        inverted_nan_quietness: false,
        nan_canonicalization: false,
        underflow_after_rounding: false,
        fma: false,
    };

    pub const ONLY_ROUNDING_FMA: Self = Self {
        exceptions: false,
        round: true,
        rmm: false,
        inverted_nan_quietness: false,
        nan_canonicalization: false,
        underflow_after_rounding: false,
        fma: true,
    };

    pub const ONLY_EXCEPTIONS_X86: Self = Self {
        exceptions: true,
        round: false,
        rmm: false,
        inverted_nan_quietness: false,
        nan_canonicalization: false,
        underflow_after_rounding: true,
        fma: false,
    };
}
