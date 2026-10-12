use crate::{
    Env, Exceptions, Float, RoundingMode,
    native::{Fast, Native},
};
use core::cmp::Ordering;

/// Compute the value of an operation with emulated rounding.
///
/// - `op()` defines the operation: it returns the NE-rounded result, i.e. `round(exact result)`.
///
/// - `error(result)` defines the error from NE rounding: it returns a value `error` such that
///   `round(exact result) + error = exact result` holds mathematically. `result` is the value
///   returned by `op()`. `error` is only invoked if `result` is finite.
///
/// - `may_overflow()` provides a heuristic for whether the result may overflow in NE mode (not the
///   emulated mode!). This heuristic should be high-quality, but may not have false negatives.
///   Propagating infinity from an infinite input is not considered an overflow, so it doesn't
///   matter if the caller decides to return `false` or `true` in this case.
///
/// - `half_op()` defines the operation with the result halved *prior to* rounding, i.e.
///   `round(exact result * 0.5)`. This detail affects e.g. overflows: if `op()` returns infinity,
///   `half_op()` might still have to return a finite value. To make this computation efficient,
///   `half_op` may assume that one of the following conditions hold:
///   - If the host supports exceptions: `may_overflow()` has returned `true`.
///   - If the host doesn't support exceptions: `op()` has returned infinity.
///
/// - `half_error(half_result)` defines the error from `half_op` with the same meaning as `error`.
///
/// On hosts with exceptions, this function raising all exceptions assuming that `op` raises the
/// correct exceptions for NE mode. On hosts without exceptions, this function raises overflow if
/// the rounded result is finite, as well as underflow, because these conditions are difficult to
/// detect afterwards.
#[inline(always)]
pub fn compute_emulating_rounding<T: Native>(
    op: impl FnOnce() -> T,
    error: impl FnOnce(T) -> T,
    may_overflow: impl FnOnce() -> bool,
    half_op: impl FnOnce() -> T,
    half_error: impl FnOnce(T) -> T,
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
    // rounded to nearest it's infinite. For example, a result slightly below `2^(maxe+1)` rounds to
    // `+inf` with NE, but floors to the largest finite value. This can inadvertently raise the
    // overflow flag when it shouldn't be raised. Compare this to a situation where the overflow
    // does happen, e.g. for a result `>= 2^(maxe+1)`: both NE and floor return the same result,
    // i.e. the largest finite value, but the overflow flag is different.
    //
    // This condition is expensive to detect post-factum with emulated rounding modes, which is why
    // we detect it here. To avoid performance issues, we only do so when the result actually might
    // have overflowed.
    if env.features().exceptions {
        // With exceptions, `op()` can erroneously raise the overflow flag, so we need an indirect
        // check. XXX: what about underflow?
        if !may_overflow() {
            return adjust_rounding_finite(op(), error, round, env);
        }
    } else {
        // Without exceptions, checking `op()` for infinity is the most obviously correct test.
        let result = op();
        if !result.is_infinite() {
            return adjust_rounding_finite(result, error, round, env);
        }
    }

    core::hint::cold_path();

    // Sets the same flags as `a + b`, except for overflow (and underflow? XXX).
    let half_result = half_op();
    if !half_result.is_finite() {
        // The actual result is `NaN`, or `a` or `b` must have been infinite, quit immediately so
        // that we don't raise more exceptions or try to adjust it to a finite value.
        return half_result;
    }

    // Shouldn't overflow or return non-finite values.
    let half_result = adjust_rounding_finite(half_result, half_error, round, env);

    // Almost a correct result, except for saturating to `+-inf` instead of finite values. This
    // raises overflow if `|half_result| >= 2^maxe`, which is correct because it applies to
    // a *rounded* version of `half_result` as opposed to the exact one, so it's equivalent to
    // `|result| >= 2^(maxe+1)` as if the exponent was unbounded.
    let result = half_result * T::TWO;
    if result.is_infinite() && !env.features().exceptions {
        // Handled here because it's difficult to detect post-factum in `set_exceptions`.
        env.raise(Exceptions::OVERFLOW | Exceptions::INEXACT);
    }

    let limit = T::TWO.powi(T::MAX_EXPONENT);
    let saturate_to_finite = match round {
        RoundingMode::ToNearest => unreachable!(),
        // Flooring saturates a very positive output to the largest representable float, but a very
        // negative output to `-inf`, because `-inf` is <= every finite value. So we can't just
        // compare absolute values here and need per-mode logic.
        RoundingMode::Floor => Fast(half_result) >= Fast(limit),
        RoundingMode::Ceil => Fast(half_result) <= Fast(-limit),
        RoundingMode::Trunc => Fast(half_result.abs()) >= Fast(limit),
        RoundingMode::ToNearestTiesToMaxMagnitude => unreachable!(),
    };

    result.nudge(if saturate_to_finite { -1 } else { 0 })
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
