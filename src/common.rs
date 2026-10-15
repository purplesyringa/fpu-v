use crate::{
    Env, Exceptions, RoundingMode,
    native::{Fast, Native, Quiet},
};
use core::cmp::Ordering;

/// A floating-point calculation whose properties can be queried.
///
/// Logically, implementors of this trait should contain the inputs to the calculation.
///
/// This trait defines a lot of methods that are necessary for emulating rounding and exceptions.
pub trait Operation {
    /// The floating-point type in question.
    type T: Native;

    /// Execute the operation.
    ///
    /// This function can be invoked with any active *native* rounding mode and needs to behave
    /// accordingly, and raise exceptions according to that mode if they are supported natively. It
    /// doesn't need to worry about emulating anything. For example, `a + b` is a valid
    /// implementation.
    fn compute(&self) -> Self::T;

    /// Get the mathematical error from `compute()` in NE rounding.
    ///
    /// The `result` argument is obtained from `compute()` and is guaranteed to be finite. This
    /// function must return a value `error` such that `result + error = exact result` holds
    /// mathematically.
    ///
    /// The native rounding mode is always NE. This function must not raise exceptions.
    fn error(&self, result: Self::T) -> Self::T;

    /// Check if `compute()` may overflow with NE rounding.
    ///
    /// This function should define a cheap, preferably good heuristic that returns `true` if the
    /// calculation may NE-overflow, i.e. `|exact result| >= 2^maxe * (2 - 2^(-mantissa_len-1))`. It
    /// may have false positives, but not false negatives. Propagating infinity from an input is not
    /// considered an overflow (though returning `true` is still allowed as per the false positive
    /// rule).
    ///
    /// The native rounding mode is always NE. This function should try to avoid raising exceptions,
    /// but it's technically allowed to raise overflow if the calculation would overflow regardless
    /// of the rounding mode (i.e. the mathematical result is `>= 2^(maxe+1)` by absolute value).
    fn may_ne_overflow(&self) -> bool;

    /// Execute the operation with the result multiplied by `2^-k` before rounding.
    ///
    /// "Before rounding means" that the returned value must be equal to
    /// `round(exact result * 2^-k)`. This affects the behavior on underflows and overflows, for
    /// example: if `compute()` returns infinity, `compute_reduced()` might still be a finite value.
    ///
    /// The implementor chooses `k` such that `exact result * 2^-k` never overflows, while still
    /// being possible to compute without introducing intermediate underflows. To do so, it may
    /// assume that `may_ne_overflow()` has returned `true`. (For example, if `may_ne_overflow()`
    /// only returns `true` is the inputs are large, it allows this function to multiply them by
    /// some `2^-k` without worrying about underflows.)
    ///
    /// The native rounding mode is always NE. This function must raise the same exceptions as
    /// `compute()` would, had the exponent range been unbounded.
    fn compute_reduced(&self) -> Self::T;

    /// Get the mathematical error from `compute_reduced()` in NE rounding.
    ///
    /// Just like `error`, but for `compute_reduced` instead of `compute`.
    fn error_reduced(&self, result_reduced: Self::T) -> Self::T;

    /// Compute `result_reduced * 2^k`, where `k` is chosen consistently with `compute_reduced`.
    ///
    /// The native rounding mode is always NE. This function must raise exceptions consistently with
    /// the formula `result_reduced * 2^k`, i.e. its body should be equivalent to
    /// `result_reduced * 2^k`.
    fn lift(&self, result_reduced: Self::T) -> Self::T;

    /// Check whether there is a `NaN` input to the operation.
    fn has_nan_input(&self) -> bool;

    /// Check whether there is an *emulated* signaling `NaN` input to the operation.
    fn has_signaling_nan_input(&self) -> bool;

    /// Check whether there is an infinite input to the operation.
    fn has_infinite_input(&self) -> bool;

    /// Check whether the calculation overflows to a finite value in the emulated rounding mode.
    ///
    /// Unlike `may_ne_overflow`, this check must be exact and covers the emulated mode, not NE.
    /// This function is only invoked for floor, ceil, and trunc modes, so it must check whether
    /// `exact result >= 2^(maxe+1)` holds (for floor; similarly for other modes).
    ///
    /// To do so efficiently, it may assume that all inputs are finite, and that the NE rounding of
    /// the result is `+inf`, `-inf`, or `+-inf`, respectively for each rounding mode.
    ///
    /// The native rounding mode is always NE. This function must not raise exceptions.
    fn is_finite_overflow(&self, round: RoundingMode) -> bool;

    /// Check whether `compute()` is exact in a native rounding.
    ///
    /// This is similar to `error()`, but only returns a boolean value, and the native rounding mode
    /// can be arbitrary. `result` is guaranteed to have been produced from `compute()` with the
    /// active rounding mode.
    ///
    /// `result` can be non-finite, in which case this function should return `false`.
    ///
    /// This function may raise any exceptions.
    fn is_exact_native(&self, result: Self::T, env: &mut Env) -> bool;

    /// Check whether the calculation overflows to a finite value in a native rounding mode.
    ///
    /// This function must check whether `exact result >= 2^(maxe+1)` holds (for floor; similarly
    /// for other modes), but there are major differences from `is_finite_overflow`:
    /// - The *native* rounding mode is floor, ceil, or trunc.
    /// - While `is_finite_overflow` checks for overflow assuming NE overflow produced infinity,
    ///   `is_overflow_finite_native` checks for overflow assuming the correctly rounded result is
    ///   finite, i.e. finiteness is a precondition and not a postcondition.
    ///
    /// This function may assume that the computation is inexact. It may raise any exceptions.
    fn is_overflow_finite_native(&self, env: &mut Env) -> bool;

    /// Check whether the calculation underflows in a native rounding mode.
    ///
    /// This function must check underflow "after rounding", i.e. whether
    /// `round(exact result) < 2^mine`, where `round` limits the mantissa length, but not the
    /// exponent range.
    ///
    /// This function may assume that the computation is inexact. It may raise any exceptions.
    fn is_underflow_native(&self) -> bool;
}

/// Compute the value of an [`Operation`], emulating rounding and exceptions if necessary.
///
/// This function never returns an input without going through `op.compute()` or
/// `op.compute_reduced()`, so `NaN`s are autocanonicalized if the host supports that.
#[inline(always)]
pub fn compute_operation<Op: Operation>(op: &Op, env: &mut Env) -> Op::T {
    let Some(round) = env.emulate_rounding_mode() else {
        let result = op.compute();
        if !env.features().exceptions && !op.is_exact_native(result, env) {
            if result.is_finite() {
                env.raise(Exceptions::INEXACT);

                // Only floor/ceil/trunc can overflow to a finite value.
                if matches!(
                    env.get_rounding_mode(),
                    RoundingMode::Floor | RoundingMode::Ceil | RoundingMode::Trunc
                ) && op.is_overflow_finite_native(env)
                {
                    env.raise(Exceptions::OVERFLOW);
                }

                if op.is_underflow_native() {
                    env.raise(Exceptions::UNDERFLOW);
                }
            } else {
                set_exceptions_non_finite(result, op, env);
            }
        }
        return result;
    };

    if round == RoundingMode::ToNearestTiesToMaxMagnitude {
        // NE operations set the exact same flags as RMM operations. `OVERFLOW` and `UNDERFLOW`
        // specifically, the only flags affected by the rounding mode, work out because infinity and
        // the smallest normal number are even for the purposes of NE, so RMM and NE agree to round
        // towards them.
        let result = op.compute();
        return adjust_rounding(result, || op.error(result), op, round, env);
    }

    // For every mode but NE and RMM, it's possible that the correctly rounded result is finite, but
    // rounded to nearest it's infinite. For example, `2^(maxe+1) - eps` rounds to a finite value
    // with floor, but to infinity with NE. That's a difference of 1 ulp, but it can't be corrected
    // like other errors because the error cannot be computed when infinities are introduced. We
    // have to detect it in a different manner.

    if !env.features().exceptions {
        // In absence of exceptions, we're allowed to just compute `op()` and look at it, even
        // though it may result in a spurious overflow.
        let result = op.compute();
        // Finite values can be adjusted with a generic algorithm. `adjust_rounding` checks if the
        // result is finite first thing, so this check doesn't impede the fast path.
        if result.is_finite() {
            return adjust_rounding(result, || op.error(result), op, round, env);
        }

        // Flooring saturates a very positive output to the largest representable float, so a value
        // of `+inf` as a result of NE rounding needs to be corrected, unless it's forwarding
        // an infinity from the input. `-inf`, however, is always correct: obtaining `-inf` from NE
        // means that the exact value is below the most negative finite float, so it'll remain
        // `-inf` when floored. Hence there is a notion of "safe", non-corrected values.
        let is_safe_infinity = match round {
            RoundingMode::ToNearest => unreachable!(),
            RoundingMode::Floor => Quiet(result) == Quiet(-Op::T::INFINITY),
            RoundingMode::Ceil => Quiet(result) == Quiet(Op::T::INFINITY),
            RoundingMode::Trunc => false,
            RoundingMode::ToNearestTiesToMaxMagnitude => unreachable!(),
        };
        // If the result is `NaN`, one of the inputs must be infinite or NaN, so this check passes.
        if is_safe_infinity || op.has_infinite_input() || op.has_nan_input() {
            set_exceptions_non_finite(result, op, env);
            return result;
        }

        // If all inputs are finite, but the result is infinite, the infinity must come from NE
        // overflow and should be corrected. The correctly rounded output is the same regardless of
        // whether the mathematical result is slightly below `2^(maxe+1)` or above it; only the
        // overflow condition depends on it. We set the flag here instead of letting the caller deal
        // with it so that `is_overflow` can skip checking for finite inputs, exactness, etc.
        env.raise(Exceptions::INEXACT);
        if op.is_finite_overflow(round) {
            env.raise(Exceptions::OVERFLOW);
        }
        return result.nudge(-1);
    }

    // With exceptions, `compute()` can erroneously raise the overflow flag, so we need to check for
    // overflow indirectly. XXX: what about underflow?
    if !op.may_ne_overflow() {
        // This is our fast path.
        let result = op.compute();
        return adjust_rounding(result, || op.error(result), op, round, env);
    }

    // Sets the same flags as `compute()`, except for overflow (and underflow? XXX).
    let result_reduced = op.compute_reduced();
    if !result_reduced.is_finite() {
        // The actual result is `NaN` or an infinity was propagated from `a` or `b`. We must have
        // already set the right flags, and we don't need to worry about adjusting infinity to
        // a finite value because such infinities are from propagation and not overflow.
        return result_reduced;
    }

    // Shouldn't overflow or return non-finite values.
    let result_reduced = adjust_rounding(
        result_reduced,
        || op.error_reduced(result_reduced),
        op,
        round,
        env,
    );

    // Almost a correct result, except for saturating to `+-inf` instead of finite values. This
    // line raises overflow if `result_reduced * 2^k >= 2^maxe`, which is notably not the same
    // thing as the incorrect `exact result >= 2^maxe`, but rather matches the correct
    // `round_unbounded(exact result) >= 2^maxe`.
    let result = op.lift(result_reduced);

    let is_overflow = match round {
        RoundingMode::ToNearest => unreachable!(),
        RoundingMode::Floor => Quiet(result) == Quiet(Op::T::INFINITY),
        RoundingMode::Ceil => Quiet(result) == Quiet(-Op::T::INFINITY),
        RoundingMode::Trunc => Quiet(result.abs()) == Quiet(Op::T::INFINITY),
        RoundingMode::ToNearestTiesToMaxMagnitude => unreachable!(),
    };
    result.nudge(if is_overflow { -1 } else { 0 })
}

#[inline(always)]
fn adjust_rounding<Op: Operation>(
    result: Op::T,
    error: impl FnOnce() -> Op::T,
    op: &Op,
    round: RoundingMode,
    env: &mut Env,
) -> Op::T {
    if !result.is_finite() {
        // Infinities are already handled by `compute_emulating_rounding` gracefully for rounding
        // modes where it matters.
        set_exceptions_non_finite(result, op, env);
        return result;
    }

    let error_value = error();

    // If the expression for `error_value` ends with computing a difference, this can be merged into
    // its calculation for all modes except RMM, which requires computing the exact error.
    let error = Fast(error_value).cmp(&Fast(Op::T::ZERO));

    // +1 means away from zero, -1 means towards zero.
    let nudge = match round {
        RoundingMode::ToNearest => unreachable!(),
        RoundingMode::Floor => {
            if error == Ordering::Less {
                // A zero with an error towards -inf must be -0, so this nudges zero correctly.
                if Fast(result) > Fast(Op::T::ZERO) {
                    -1
                } else {
                    1
                }
            } else if error == Ordering::Equal && Fast(result) == Fast(Op::T::ZERO) {
                // XXX!!!!!
                // Special case: IEEE-754 requires floor(x - x) to return -0, not +0 like other
                // rounding modes. We can't just nudge by `-1` here to cross the sign boundary
                // because that'd convert `+0` to `NaN` and not `-0`.
                return -Op::T::ZERO;
            } else {
                0
            }
        }
        RoundingMode::Ceil => {
            if error == Ordering::Greater {
                // A zero with an error towards +inf must be +0, so this nudges zero correctly.
                if Fast(result) >= Fast(Op::T::ZERO) {
                    1
                } else {
                    -1
                }
            } else {
                0
            }
        }
        RoundingMode::Trunc => {
            // These comparisons can be written either by comparing `result` to `0`, or by checking
            // its sign bit, because a value rounded to zero cannot have an error towards zero (as
            // long as rounding retains the sign). Don't optimize this with multiplication, it'll
            // cause precision issues and spurious flags.
            if (error == Ordering::Less && Fast(result) > Fast(Op::T::ZERO))
                || (error == Ordering::Greater && Fast(result) < Fast(Op::T::ZERO))
            {
                -1
            } else {
                0
            }
        }
        RoundingMode::ToNearestTiesToMaxMagnitude => {
            if error == Ordering::Equal
                || (error == Ordering::Less && Fast(result) > Fast(Op::T::ZERO))
                || (error == Ordering::Greater && Fast(result) < Fast(Op::T::ZERO))
            {
                // Exact results and bias away from zero are already compliant with RMM.
                0
            } else {
                // All remaining ties are rounded towards zero and should be nudged away from zero,
                // we just need to detect them.
                //
                // Since `result` was rounded towards zero, the real value lies between `result` and
                // `result.nudge(1)`. The latter can be infinite, but since infinity is even, this
                // is guaranteed not to be a tie, and the comparison ends up working correctly. And
                // since no floating-point operation here produces infinity from finite inputs, we
                // don't need to avoid a spurious overflow.
                if Fast(error_value * Op::T::TWO) == Fast(result.nudge(1) - result) {
                    1
                } else {
                    0
                }
            }
        }
    };

    let result = result.nudge(nudge);

    // `result` can become infinite during adjustment in floor, ceil, and trunc modes. For example,
    // `floor` can adjust a large negative value to `-inf`. Notably, the direction of adjustment is
    // always away from the direction in which values are saturated to finite floats, so the
    // produced infinity is correct. We do need to set flags though.
    if result.is_infinite() {
        env.raise(Exceptions::OVERFLOW | Exceptions::INEXACT);
    } else if !env.features().exceptions {
        // Whether an error occurs doesn't depend on the rounding mode, so we can reuse the NE error
        // for this check.
        if error != Ordering::Equal {
            env.raise(Exceptions::INEXACT);

            // XXX: check underflow

            // We've already checked for overflow-to-infinite, and overflow-to-finite is handled in
            // `compute_operation`.
        }
    }

    result
}

/// For a host that doesn't support exception flags, raise flags arising from an operation resulting
/// in a non-finite output.
///
/// This sets `INVALID` for newly arising NaNs and `OVERFLOW` for newly arising infinities.
fn set_exceptions_non_finite<Op: Operation>(result: Op::T, op: &Op, env: &mut Env) {
    #[cfg(not(feature = "explore"))]
    assert!(
        !result.is_finite(),
        "set_exceptions_non_finite was called with a finite result",
    );
    if result.is_nan() {
        // IEEE-754 says qNaN propagation doesn't raise invalid operation, only sNaN or newly
        // arising NaNs do.
        if !op.has_nan_input() || op.has_signaling_nan_input() {
            env.raise(Exceptions::INVALID);
        }
    } else {
        // IEEE-754 says infinity propagation doesn't raise invalid operation, only newly arising
        // infinities do.
        if !op.has_infinite_input() {
            env.raise(Exceptions::OVERFLOW | Exceptions::INEXACT);
        }
    }
}
