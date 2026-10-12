use crate::{
    Env, Exceptions, Float, RoundingMode,
    common::{compute_emulating_rounding, set_exceptions_non_finite},
    native::{Fast, Native, Quiet},
};

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
    compute_emulating_rounding(
        || a + b,
        |sum| sum_error(a, b, sum),
        || {
            // We use two conditions to check if the addition may overflow in NE mode.
            //
            // First, `max(|a|, |b|) < 2^maxe` recognizes the most common case by magnitude.
            //
            // To make `half_op` fast, we also need to ensure that this arm catches subnormal inputs
            // (since adding a subnormal never causes an overflow in NE mode). But we can do
            // slightly better than looking for subnormals exclusively. Notice that as long as:
            //     min(|a|, |b|) < 2^(maxe-manissa_len-1),
            // ...combining it with:
            //     |a|, |b| <= 2^maxe * (2 - 2^-mantissa_len)
            // ...gives:
            //     |a + b| < 2^maxe * (2 - 2^-mantissa_len) + 2^(maxe-manissa_len-1)
            //         = 2^maxe * (2 - 2^-(mantissa_len+1)),
            // ...which implies no NE overflow.
            let a_bits = a.abs().to_bits();
            let b_bits = b.abs().to_bits();
            // Make sure not to trigger exceptions on NaN, but still capture NaN inputs
            Quiet(T::from_bits(a_bits.max(b_bits))) >= Quiet(T::TWO.powi(T::MAX_EXPONENT))
                && Fast(T::from_bits(a_bits.min(b_bits)))
                    >= Fast(T::TWO.powi(T::MAX_EXPONENT - T::MANTISSA_LEN as i32 - 1))
        },
        || {
            // Computes `(a + b) * 0.5` without overflows. The reasoning behind this logic depends
            // on whether we have exceptions.
            //
            // If we do, `compute_emulating_rounding` promises that `may_overflow` has returned
            // `true`, so `a` and `b` aren't subnormals (though they may be infinite or NaN), and
            // thus we can compute `a/2 + b/2` without risking underflow.
            //
            // If we don't, the sum must be infinite, which can happen either due to benevolent
            // overflow (which implies lack of subnormal inputs), or because one of the inputs was
            // infinite, in which case we'll just forward infinity -- and possibly cause
            // an underflow when handling the other argument, but that doesn't matter because we
            // don't have exceptions.
            a * T::HALF + b * T::HALF
        },
        |reduced_sum| sum_error(a * T::HALF, b * T::HALF, reduced_sum),
        |reduced_sum| reduced_sum * T::TWO,
        round,
        env,
    )
}

#[inline(always)]
fn sum_error<T: Native>(a: T, b: T, sum: T) -> T {
    // Avoid using the branchless variant of 2Sum to compute the error: as [1] points out, it can
    // return a non-finite error even for finite inputs in edge cases.
    //
    // Note that the sign of the error indicates the direction that cancels the rounding, not the
    // direction rounding was performed towards.
    //
    // Computing this does not raise any exceptions because everything is exact.
    //
    // [1]: https://uwplse.org/2025/08/04/two-sum.html
    if Fast(a.abs()) >= Fast(b.abs()) {
        b - (sum - a)
    } else {
        a - (sum - b)
    }
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
        // handled by `compute_emulating_rounding`, but for powerful hosts we need extra wiring.
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
