use crate::{
    Env, Exceptions, Float, RoundingMode,
    native::{Fast, Native, Quiet},
};
use core::cmp::Ordering;

/// Compute the value of an operation with emulated rounding.
///
/// This function implements an algorithm that raises the correct exceptions and handles rounding
/// near the `2^(maxe+1)` boundary correctly. It's tricky because a calculation can NE-overflow even
/// if it wouldn't overflow with the correct rounding mode, resulting in a spurious exception. We
/// avoid this with the following algorithm:
///
/// 1. If the inputs are small enough that the operation can't NE-overflow, compute it and then
///    adjust rounding.
/// 2. Otherwise, scale down the inputs by `2^-k` such that the operation can no longer NE-overflow,
///    compute and adjust the result, and scale it back.
///
/// The second step is complex because no cheap predicate can predict a spurious overflow without
/// a bunch of false positives, and we want it cheap because (1) is the common case.
///
/// This function takes quite a few specific primitives as input:
///
/// - `op()` defines the operation: it returns the NE-rounded result, i.e. `round(exact result)`.
///
/// - `error(result)` defines the error from NE rounding: it returns a value `error` such that
///   `round(exact result) + error = exact result` holds mathematically. `result` is the value
///   returned by `op()`. `error` is only invoked if `result` is finite.
///
/// - `may_overflow()` provides a heuristic for whether the result may overflow in NE mode (not the
///   emulated mode!). This heuristic should be high-quality, but may not have false negatives.
///   Propagating infinity from an infinite input is not considered an overflow, so `false` is
///   preferable, but `true` works too. `may_overflow` should generally not raise exceptions, though
///   it's technically allowed to raise overflow if the calculation would overflow regardless of the
///   rounding mode (which in practice means the mathematical result is above `2^(maxe+1)`).
///
/// - `reduced_op()` defines the operation with the mathematical result multiplied by some `2^-k`
///   *prior to* NE-rounding, i.e. `round(exact result * 2^-k)`. This detail affects overflows: if
///   `op()` returns infinity, `reduced_op()` might still have to return a finite value.
///   `reduced_op()` must raise the same flags as `op()` would had the exponent range been
///   unbounded; specifically, this means that the multiplication by `2^-k` as implemented by
///   `reduced_op` must not cause underflows, and the computation itself must not overflow. To
///   make this possible, `reduced_op()` may assume that `may_overflow()` has returned `true`.
///
/// - `reduced_error(reduced_result)` defines the error from `reduced_op` with the same meaning as
///   `error`.
///
/// - `lift(reduced_result)` computes `reduced_result * 2^k`.
///
/// - `are_inputs_finite()` checks whether all input to the operation are finite.
///
/// On hosts with exceptions, this function raising all exceptions assuming that `op` raises the
/// correct exceptions for NE mode. On hosts without exceptions, this function doesn't raise any
/// emulated exceptions.
#[inline(always)]
pub fn compute_emulating_rounding<T: Native>(
    op: impl FnOnce() -> T,
    error: impl FnOnce(T) -> T,
    may_overflow: impl FnOnce() -> bool,
    reduced_op: impl FnOnce() -> T,
    reduced_error: impl FnOnce(T) -> T,
    lift: impl FnOnce(T) -> T,
    are_inputs_finite: impl FnOnce() -> bool,
    round: RoundingMode,
    env: &mut Env,
) -> T {
    if round == RoundingMode::ToNearestTiesToMaxMagnitude {
        // NE operations set the exact same flags as RMM operations. `OVERFLOW` and `UNDERFLOW`
        // specifically, the only flags affected by the rounding mode, work out because infinity and
        // the smallest normal number are even for the purposes of NE, so RMM and NE agree to round
        // towards them.
        return adjust_rounding_finite(op(), error, round, env);
    }

    // For every mode but NE and RMM, it's possible that the correctly rounded result is finite, but
    // rounded to nearest it's infinite. That's a difference of 1 ulp, but it can't be corrected
    // with the same generic algorithm because computing the error fails when infinities are
    // introduced. We have to detect it in a different manner.

    if !env.features().exceptions {
        // In absence exceptions, we're allowed to just compute `op()`, even though it may result in
        // a spurious overflow.
        let result = op();
        // Finite values are necessarily safe. `adjust_rounding_finite` checks if the result is
        // finite anyway, so this check doesn't impede the fast path.
        if result.is_finite() {
            return adjust_rounding_finite(result, error, round, env);
        }

        // Flooring saturates a very positive output to the largest representable float, so a value
        // of `+inf` as a result of NE rounding needs to be corrected, unless it's forwarding
        // an infinity from the input. `-inf`, however, is always correct: obtaining `-inf` from NE
        // means that the exact value is below the most negative finite float, so it'll remain
        // `-inf` when floored.
        let is_safe_infinity = match round {
            RoundingMode::ToNearest => unreachable!(),
            RoundingMode::Floor => Quiet(result) == Quiet(-T::INFINITY),
            RoundingMode::Ceil => Quiet(result) == Quiet(T::INFINITY),
            RoundingMode::Trunc => false,
            RoundingMode::ToNearestTiesToMaxMagnitude => unreachable!(),
        };
        // If the result is `NaN`, one of the inputs must be non-finite, so this check passes.
        if is_safe_infinity || !are_inputs_finite() {
            return result;
        }

        // If all inputs are finite, but the result is infinite, the infinity must come from NE
        // overflow and should be corrected. The correctly rounded output is the same regardless of
        // whether the mathematical result is slightly below `2^(maxe+1)` or above it. The caller is
        // responsible for raising an emulated overflow exception if necessary -- we can't do it
        // here unconditionally, and detecting it here is no less expensive than afterwards.
        return result.nudge(-1);
    }

    // With exceptions, `op()` can erroneously raise the overflow flag, so we need to check for
    // overflow indirectly. XXX: what about underflow?
    if !may_overflow() {
        return adjust_rounding_finite(op(), error, round, env);
    }

    // Sets the same flags as `op()`, except for overflow (and underflow? XXX).
    let reduced_result = reduced_op();
    if !reduced_result.is_finite() {
        // The actual result is `NaN` or an infinity was propagated from `a` or `b`. `reduced_op()`
        // must have already set the right flags, and we don't need to worry about adjusting
        // infinity to a finite value because `reduced_op` cannot overflow.
        return reduced_result;
    }

    // Shouldn't overflow or return non-finite values.
    let reduced_result = adjust_rounding_finite(reduced_result, reduced_error, round, env);

    // Almost a correct result, except for saturating to `+-inf` instead of finite values. This
    // line raises overflow if `reduced_result * 2^k >= 2^maxe`, which is notably not the same
    // thing as the incorrect `exact result >= 2^maxe`, but rather matches the correct
    // `round_unbounded(exact result) >= 2^maxe`.
    let result = lift(reduced_result);

    let is_matching_infinity = match round {
        RoundingMode::ToNearest => unreachable!(),
        RoundingMode::Floor => Quiet(result) == Quiet(T::INFINITY),
        RoundingMode::Ceil => Quiet(result) == Quiet(-T::INFINITY),
        RoundingMode::Trunc => Quiet(result.abs()) == Quiet(T::INFINITY),
        RoundingMode::ToNearestTiesToMaxMagnitude => unreachable!(),
    };
    result.nudge(if is_matching_infinity { -1 } else { 0 })
}

/// Given a value rounded in NE mode as well as the error, return a correctly rounded value.
///
/// The exact value is assumed to be mathematically equal to `result + error(result)`.
///
/// On hosts with exceptions, this function raises `OVERFLOW` if rounding adjusts `result` from
/// finite to infinite (e.g. largest finite float plus a tiny value under ceil). On hosts without
/// exceptions, the caller is responsible for this instead.
#[inline(always)]
fn adjust_rounding_finite<T: Native>(
    result: T,
    error: impl FnOnce(T) -> T,
    round: RoundingMode,
    env: &mut Env,
) -> T {
    if !result.is_finite() {
        // Infinities are already handled by `compute_emulating_rounding` gracefully for rounding
        // modes where it matters.
        return result;
    }

    let error_value = error(result);

    // If the expression for `error_value` ends with computing a difference, this can be merged into
    // its calculation for all modes except RMM, which requires computing the exact error.
    let error = Fast(error_value).cmp(&Fast(T::ZERO));

    // +1 means away from zero, -1 means towards zero.
    let nudge = match round {
        RoundingMode::ToNearest => unreachable!(),
        RoundingMode::Floor => {
            if error == Ordering::Less {
                // A zero with an error towards -inf must be -0, so this nudges zero correctly.
                if Fast(result) > Fast(T::ZERO) { -1 } else { 1 }
            } else if error == Ordering::Equal && Fast(result) == Fast(T::ZERO) {
                // XXX!!!!!
                // Special case: IEEE-754 requires floor(x - x) to return -0, not +0 like other
                // rounding modes. We can't just nudge by `-1` here to cross the sign boundary
                // because that'd convert `+0` to `NaN` and not `-0`.
                return -T::ZERO;
            } else {
                0
            }
        }
        RoundingMode::Ceil => {
            if error == Ordering::Greater {
                // A zero with an error towards +inf must be +0, so this nudges zero correctly.
                if Fast(result) >= Fast(T::ZERO) { 1 } else { -1 }
            } else {
                0
            }
        }
        RoundingMode::Trunc => {
            // These comparisons can be written either by comparing `result` to `0`, or by checking
            // its sign bit, because a value rounded to zero cannot have an error towards zero (as
            // long as rounding retains the sign). Don't optimize this with multiplication, it'll
            // cause precision issues and spurious flags.
            if (error == Ordering::Less && Fast(result) > Fast(T::ZERO))
                || (error == Ordering::Greater && Fast(result) < Fast(T::ZERO))
            {
                -1
            } else {
                0
            }
        }
        RoundingMode::ToNearestTiesToMaxMagnitude => {
            if error == Ordering::Equal
                || (error == Ordering::Less && Fast(result) > Fast(T::ZERO))
                || (error == Ordering::Greater && Fast(result) < Fast(T::ZERO))
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
                if Fast(error_value * T::TWO) == Fast(result.nudge(1) - result) {
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
    if result.is_infinite() && env.features().exceptions {
        env.raise(Exceptions::OVERFLOW | Exceptions::INEXACT);
    }

    result
}

/// For a host that doesn't support exception flags, raise simple flags arising from `a op b = out`,
/// where `out` is non-finite.
///
/// This sets `INVALID` for newly arising NaNs and `OVERFLOW` for newly arising infinities.
///
/// `float1` and `float2` must correspond to the original emulated values of `a` and `b`
/// respectively.
pub fn set_exceptions_non_finite<T: Native, F: Float>(
    a: T,
    b: T,
    out: T,
    float1: F,
    float2: F,
    env: &mut Env,
) {
    assert!(
        !out.is_finite(),
        "set_exceptions_non_finite was called with a finite output",
    );
    if out.is_nan() {
        // IEEE-754 says qNaN propagation doesn't raise invalid operation, only sNaN or newly
        // arising NaNs do.
        if !(a.is_nan() || b.is_nan())
            || (float1.is_emulated_signaling_nan() || float2.is_emulated_signaling_nan())
        {
            env.raise(Exceptions::INVALID);
        }
    } else {
        // IEEE-754 says infinity propagation doesn't raise invalid operation, only newly arising
        // infinities do.
        if !(a.is_infinite() || b.is_infinite()) {
            env.raise(Exceptions::OVERFLOW | Exceptions::INEXACT);
        }
    }
}
