use crate::{
    Env, Exceptions, Float, RoundingMode,
    common::set_exceptions_non_finite,
    native::{Fast, Native, Quiet},
};
use core::cmp::Ordering;

#[inline(always)]
pub fn add<T: Float>(a: T, b: T, env: &mut Env) -> T {
    let out = do_add(
        env.to_native(a),
        env.to_native(b),
        // Pass the original emulated floats so that they can be tested for sNaN without involving
        // platform-specific shenanigans -- see various comments in `native` and `types` for the
        // differences between `is_emulated_signaling_nan` and `is_native_signaling_nan`.
        a,
        b,
        env,
    );
    // `do_add` never returns inputs directly without applying some kind of FP operation to them, so
    // NaNs are autocanonicalized on platforms that support that.
    env.from_native_canonicalizing_nan_after_op(out)
}

pub fn sub<T: Float>(a: T, b: T, env: &mut Env) -> T {
    // IEEE-754 says `a - b` is equivalent to `a + (-b)`. The compiler should be able to rewrite
    // the fast path of `do_add` to optimize out the negation. `-b` also doesn't set exceptions.
    let out = do_add(env.to_native(a), -env.to_native(b), a, b, env);
    env.from_native_canonicalizing_nan_after_op(out)
}

#[inline(always)]
fn do_add<T: Native, F: Float>(a: T, b: T, float1: F, float2: F, env: &mut Env) -> T {
    let sum = if let Some(round) = env.emulate_rounding_mode() {
        add_with_rounding(a, b, round, env)
    } else {
        a + b
    };
    if !env.features().exceptions {
        set_exceptions(a, b, sum, float1, float2, env);
    }
    sum
}

// On hosts with exceptions, this function is responsible for raising all exceptions. On hosts
// without exceptions, it's responsible for raising overflow if the rounded sum is finite, because
// this condition is difficult to detect afterwards.
#[inline(always)]
fn add_with_rounding<T: Native>(a: T, b: T, round: RoundingMode, env: &mut Env) -> T {
    if round == RoundingMode::ToNearestTiesToMaxMagnitude {
        // NE addition sets the exact same flags as RMM addition. `OVERFLOW` specifically, the only
        // flag affected by the rounding mode during addition, works out because infinity is even
        // for the purposes of NE, so RMM and NE agree to round towards it.
        return adjust_rounding_finite(a, b, a + b, round, env);
    }

    // For every mode but NE and RMM, it's possible that correctly rounded `a + b` is finite, but
    // rounded to nearest it's infinite. For example, a sum slightly below `2^(maxe+1)` rounds to
    // +inf with NE, but floors to the largest finite value. This can inadvertently raise the
    // overflow flag when it shouldn't be raised. Compare this to a situation where the overflow
    // does happen, e.g. for a sum `>= 2^(maxe+1)`: both NE and floor return the same result, i.e.
    // the largest finite value, but the overflow flag is different.
    //
    // This condition is expensive to detect post-factum with emulated rounding modes, which is why
    // we detect it here. To avoid performance issues, we only do so when the sum actually might
    // have overflowed.
    if !env.features().exceptions {
        // Without exceptions, checking `a + b` for infinity is the most obviously correct test.
        let sum = a + b;
        if !sum.is_infinite() {
            return adjust_rounding_finite(a, b, sum, round, env);
        }
        // Since adding a very small value to a finite value never overflows in NE mode, if we get
        // to this point, `a` and `b` are guaranteed to be large.
    } else {
        // With exceptions, though, `a + b` can erroneously raise the overflow flag, so we need
        // an indirect check. We use two conditions to test that.
        //
        // First, `max(|a|, |b|) < 2^maxe` recognizes the most common case by magnitude.
        //
        // For reasons explained below, we need to ensure that this arm also catches subnormal
        // inputs (adding a subnormal never causes an overflow). But we can do slightly better than
        // looking for subnormals exclusively. Notice that as long as
        //     min(|a|, |b|) < 2^(maxe-manissa_len-1),
        // combining with
        //     |a|, |b| <= 2^maxe * (2 - 2^-mantissa_len)
        // gives:
        //     |a + b| < 2^maxe * (2 - 2^-mantissa_len) + 2^(maxe-manissa_len-1)
        //         = 2^maxe * (2 - 2^-(mantissa_len+1)),
        // ...which implies no NE overflow.
        let a_bits = a.abs().to_bits();
        let b_bits = b.abs().to_bits();
        // Make sure not to trigger exceptions on NaN, but still capture NaN inputs
        if !(Quiet(T::from_bits(a_bits.max(b_bits))) >= Quiet(T::TWO.powi(T::MAX_EXPONENT)))
            || Fast(T::from_bits(a_bits.min(b_bits)))
                < Fast(T::TWO.powi(T::MAX_EXPONENT - T::MANTISSA_LEN as i32 - 1))
        {
            return adjust_rounding_finite(a, b, a + b, round, env);
        }
    }

    core::hint::cold_path();

    // By this point, we know that `a` and `b` are not subnormals (though possibly infinite or NaN),
    // so we can compute `a/2 + b/2`, be certain that it doesn't overflow, round it correctly, and
    // multiply by `2` while taking overflow-to-finite semantics into account.
    let half_a = a * T::HALF;
    let half_b = b * T::HALF;

    // Sets the same flags as `a + b`, except for overflow.
    let half_sum = half_a + half_b;

    if !half_sum.is_finite() {
        // The addition returned `NaN`, or `a` or `b` must have been infinite, quit immediately so
        // that we don't raise more exceptions or try to adjust it to a finite value.
        return half_sum;
    }

    // Shouldn't overflow or return non-finite values.
    let half_sum = adjust_rounding_finite(half_a, half_b, half_sum, round, env);

    // Almost a correct sum, except for saturating to `+-inf` instead of finite values. This raises
    // overflow if `|half_sum| >= 2^maxe`, which is correct because it applies to a *rounded*
    // version of `a/2 + b/2` as opposed to the exact one.
    let sum = half_sum * T::TWO;
    if sum.is_infinite() && !env.features().exceptions {
        // Handled here because it's difficult to detect post-factum in `set_exceptions`.
        env.raise(Exceptions::OVERFLOW | Exceptions::INEXACT);
    }

    let limit = T::TWO.powi(T::MAX_EXPONENT);
    let saturate_to_finite = match round {
        RoundingMode::ToNearest => unreachable!(),
        // Flooring saturates a very positive output to the largest representable float, but a very
        // negative output to `-inf`, because `-inf` is <= every finite value. So we can't just
        // compare absolute values here and need per-mode logic.
        RoundingMode::Floor => Fast(half_sum) >= Fast(limit),
        RoundingMode::Ceil => Fast(half_sum) <= Fast(-limit),
        RoundingMode::Trunc => Fast(half_sum.abs()) >= Fast(limit),
        RoundingMode::ToNearestTiesToMaxMagnitude => unreachable!(),
    };

    sum.nudge(if saturate_to_finite { -1 } else { 0 })
}

// On hosts with exceptions, this function raises `OVERFLOW` if rounding adjusts `sum` from finite
// to infinite (e.g. largest finite float plus a tiny value under ceil). On hosts without
// exceptions, `set_exceptions` is responsible for this instead.
#[inline(always)]
fn adjust_rounding_finite<T: Native>(a: T, b: T, sum: T, round: RoundingMode, env: &mut Env) -> T {
    if !sum.is_finite() {
        // Infinities are already handled by `add_with_rounding` gracefully for rounding modes where
        // it matters.
        return sum;
    }

    // Avoid using the branchless variant of 2Sum to compute the error: as [1] points out, it can
    // return a non-finite error even for finite inputs in edge cases.
    //
    // Note that the sign of the error indicates the direction that cancels the rounding, not the
    // direction rounding was performed towards.
    //
    // Computing this does not raise any exceptions because everything is exact.
    //
    // [1]: https://uwplse.org/2025/08/04/two-sum.html
    let error_value = if Fast(a.abs()) >= Fast(b.abs()) {
        b - (sum - a)
    } else {
        a - (sum - b)
    };

    // This can be merged into the calculation of `error`, removing one subtraction, for all modes
    // except RMM, which requires computing the exact error.
    let error = Fast(error_value).cmp(&Fast(T::ZERO));

    // +1 means away from zero, -1 means towards zero.
    let nudge = match round {
        RoundingMode::ToNearest => unreachable!(),
        RoundingMode::Floor => {
            if error == Ordering::Less {
                // A zero with an error towards -inf must be -0, so this nudges zero correctly.
                if Fast(sum) > Fast(T::ZERO) { -1 } else { 1 }
            } else if error == Ordering::Equal && Fast(sum) == Fast(T::ZERO) {
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
                if Fast(sum) >= Fast(T::ZERO) { 1 } else { -1 }
            } else {
                0
            }
        }
        RoundingMode::Trunc => {
            // These comparisons can be written either by comparing `sum` to `0`, or by checking its
            // sign bit, because a value rounded to zero cannot have an error towards zero (as long
            // as rounding retains the sign). Don't optimize this with multiplication, it'll cause
            // precision issues and spurious flags.
            if (error == Ordering::Less && Fast(sum) > Fast(T::ZERO))
                || (error == Ordering::Greater && Fast(sum) < Fast(T::ZERO))
            {
                -1
            } else {
                0
            }
        }
        RoundingMode::ToNearestTiesToMaxMagnitude => {
            if error == Ordering::Equal
                || (error == Ordering::Less && Fast(sum) > Fast(T::ZERO))
                || (error == Ordering::Greater && Fast(sum) < Fast(T::ZERO))
            {
                // Exact results and bias away from zero are already compliant with RMM.
                0
            } else {
                // All remaining ties are rounded towards zero and should be nudged away from zero,
                // we just need to detect them.
                //
                // Since `sum` was rounded towards zero, the real value lies between `sum` and
                // `sum.nudge(1)`. The latter can be infinite, but since infinity is even, this is
                // guaranteed not to be a tie, and the comparison ends up working correctly. And
                // since no floating-point operation here produces infinity from finite inputs, we
                // don't need to avoid a spurious overflow.
                if Fast(error_value * T::TWO) == Fast(sum.nudge(1) - sum) {
                    1
                } else {
                    0
                }
            }
        }
    };

    let sum = sum.nudge(nudge);

    // `sum` can become infinite during adjustment in floor, ceil, and trunc modes. For example,
    // `floor` can adjust a large negative value to `-inf`. Notably, the direction of adjustment is
    // always away from the direction in which values are saturated to finite floats, so the
    // produced infinity is correct. We do need to set flags though.
    if sum.is_infinite() && env.features().exceptions {
        env.raise(Exceptions::OVERFLOW | Exceptions::INEXACT);
    }

    sum
}

#[inline(always)]
fn set_exceptions<T: Native, F: Float>(a: T, b: T, sum: T, float1: F, float2: F, env: &mut Env) {
    if !sum.is_finite() {
        set_exceptions_non_finite(a, b, sum, float1, float2, env);
        return;
    }

    // We avoid using 2Sum/Fast2Sum to compute the error because it assumes round-to-nearest, while
    // the rounding mode here can be arbitrary, so we use another approach.
    let is_exact = Fast(sum - a) == Fast(b) && Fast(sum - b) == Fast(a);

    // If the sum is exact, both subtractions are exact as well and comparisons return true.
    //
    // If the sum is inexact, we use a lemma from [1] to obtain the sign of the error:
    //
    //     Lemma 2.5. Let a and b be two binary FP numbers, with e_a>=e_b. Let s \in {RD(a+b),
    //     RU(a+b)}. The number s-a is a floating-point number (which implies that it will be
    //     computed exactly, with any rounding function).
    //
    // In a nutshell, this means that if `|a| >= |b|`, `sum - a` is exact, so `sum - a == b` fails,
    // and symmetrically for `|a| <= |b|`, so at least one condition returns `false`.
    //
    // [1]: Sylvie Boldo, Stef Graillat, and Jean-Michel Muller. 2017. On the Robustness of the 2Sum
    //      and Fast2Sum Algorithms. ACM Trans. Math. Softw. 44, 1, Article 4 (March 2018),
    //      14 pages. https://doi.org/10.1145/3054947
    if is_exact {
        return;
    }

    env.raise(Exceptions::INEXACT);

    // Addition can never underflow, so we don't need to test for it.

    if env.features().round
        && matches!(
            env.get_rounding_mode(),
            RoundingMode::Floor | RoundingMode::Ceil | RoundingMode::Trunc
        )
    {
        // Floor, ceil, and trunc can overflow without returning `+-inf`, e.g. floor overflows in
        // such a way if the true sum is `>= 2^(maxe+1)`. Generally speaking, any sum with
        // an absolute value `>= 2^(maxe+1)` overflows, but whenever it returns infinity it's easier
        // to handle. IEEE-754 says overflow happens if the *rounded* value, assuming an unlimited
        // exponent, doesn't fit, so what we actually want to check here is
        // `|round(a + b)| >= 2^(maxe+1)`. For hosts without rounding mode support, this is already
        // handled by `add_with_rounding`, but for powerful hosts we need some extra wiring.
        //
        // We can use the same approach as in `add_with_rounding`, and luckily it's quite cheap if
        // rounding modes are native. As a reminder, as long as we don't have subnormals, `a/2` and
        // `b/2` are exact, `a/2 + b/2` can't overflow and is thus rounded just like `a + b` would
        // be with unlimited exponent range, and so `|a/2 + b/2| >= 2^maxe` is equivalent to the
        // overflow condition for `a + b`.
        //
        // The only new issue is that we can no longer assume that `a` and `b` are normal. Luckily,
        // things still work out. For example, for floor:
        // - Large finite value + positive subnormal returns the same value regardless of the
        //   subnormal, so it doesn't matter that it effectively becomes a bit smaller.
        // - Large finite value + negative subnormal returns the same value as long as the subnormal
        //   doesn't fall to zero, but halving a negative subnormal under floor mode retains this
        //   property.
        let limit = T::TWO.powi(T::MAX_EXPONENT);
        if Fast((a * T::HALF + b * T::HALF).abs()) >= Fast(limit) {
            env.raise(Exceptions::OVERFLOW);
        }
        // This calculation (`a * 0.5 + b * 0.5`) can be optimized a little further. We can manually
        // decrease the exponent with integer arithmetic, which is clearly correct except when
        // `|a| < 2^(mine + 1)` (or similarly with `b`). The intended result in this case is always
        // `false`, and it turns out that we always get there:
        // - For `2^mine <= |a| < 2^(mine + 1)`, it ends up producing a subnormal that is slightly
        //   different, but still small enough not to affect anything.
        // - For `0 < |a| < 2^mine`, it ends up producing NaN, which propagates to the sum. It can
        //   be an sNaN, which would raise an unexpected exception, but this function is only used
        //   when there are no exceptions.
        // - `a = 0` results in infinity, but adding a zero is always exact, so we don't end up in
        //   this branch at all.
        // I avoided doing this because this is a slow path, so investing time in this is not very
        // fruitful. But we can improve this if it ends up being useful elsewhere.
    }
}
