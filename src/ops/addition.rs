use crate::{Env, Exceptions, Float, RoundingMode, native::Native};

pub fn add<T: Float>(a: T, b: T, env: &mut Env) -> T {
    // Pass the original emulated floats so that they can be tested for sNaN without involving
    // platform-specific shenanigans -- see various comments in `native` and `types` for why
    // `is_signaling_nan` cannot be defined on `Native` directly.
    T::from_native_canonicalizing_nan(do_add(env.to_native(a), env.to_native(b), a, b, env))
}

pub fn sub<T: Float>(a: T, b: T, env: &mut Env) -> T {
    // IEEE-754 says `a - b` is equivalent to `a + (-b)`. The compiler should be able to rewrite
    // the fast path of `do_add` to optimize out the negation. `-b` also doesn't set exceptions.
    T::from_native_canonicalizing_nan(do_add(env.to_native(a), -env.to_native(b), a, b, env))
}

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
fn add_with_rounding<T: Native>(a: T, b: T, round: RoundingMode, env: &mut Env) -> T {
    if round == RoundingMode::ToNearestTiesToMaxMagnitude {
        // NE addition sets the exact same flags as RMM addition. `OVERFLOW` specifically, the only
        // flag affected by the rounding mode during addition, works out because infinity is even
        // for the purposes of NE, so RMM and NE agree to round towards it.
        return adjust_rounding_finite(a, b, a + b, round, env);
    }

    // For every rounding mode but NE and RMM, it's possible that correctly rounded `a + b` is
    // finite, but rounded to nearest it's infinite. For example, a sum slightly below +inf rounds
    // to +inf with NE, but floors to a finite value. This can erroneously raise the overflow flag,
    // so save exceptions beforehand.
    let ex = env.save_host_exceptions();

    let sum = a + b;
    if !sum.is_infinite() {
        // Keep the exceptions from `a + b`, because everything except `OVERFLOW` is unaffected by
        // rounding mode, and the remaining rounding modes never introduce new infinities.
        // XXX: is underflow affected by rounding mode? IEEE-754 seems to imply "yes", but I
        // couldn't figure out how it's affected
        return adjust_rounding_finite(a, b, sum, round, env);
    }

    // Flooring is defined as taking the largest representable float less than or equal to the exact
    // value. Notably, this means that if `a` and `b` are large finite numbers, `floor(a + b)` is
    // equal to the largest representable float, *not* `+inf`, because `+inf` is not <= a finite
    // number. Despite that, IEEE-754 says that the overflow flag should be raised in this case.
    //
    // The presence of saturation to a finite value makes it difficult to separate the case where
    // `a + b` is a little below `2^(maxe+1)` and gets rounded to infinity by NE, but should be
    // rounded to the largest representable float and not raise overflow, from the case where it's
    // above `2^(maxe+1)` and the infinity is real and the sum is saturated to the same output.
    //
    // To avoid complex logic, we compute `floor(a/2 + b/2)` instead of `floor(a + b)`, moving the
    // boundary to `2^maxe`. `a/2 + b/2` never overflows, since it caps at the largest representable
    // float, allowing adjustment logic to work with finite values. Post-processing is solely
    // responsible for handling overflows and saturation based on a correctly floored value (modulo
    // limited exponent range).
    //
    // The only issue with using `a/2` and `b/2` are subnormals. We usually don't get subnormals
    // on this path, because `finite + subnormal` never overflows for NE, but `a` can be subnormal
    // if `b` is infinite or vice versa. In this case, the end result is still correct, but we need
    // to make sure to perform the halving before restoring exceptions, so that the underflow flag
    // is not raised spuriously.
    let half_a = a * T::HALF;
    let half_b = b * T::HALF;

    env.restore_host_exceptions(ex);

    // This line can only raise INEXACT, and we do want it to raise INEXACT:
    // - `INVALID`: if we didn't get any `NaN`s the first time, we shouldn't get them now.
    // - `OVERFLOW`: `half_a + half_b` never overflows, and if any input is infinite `OVERFLOW` is
    //   not set either.
    // - `UNDERFLOW`: addition never underflows.
    // - `INEXACT`: can only be caused by precision loss, always matches between `a/2 + b/2` and
    //   `a + b` in absence of subnormals, and if subnormals are present we must have an infinite
    //   input so it still matches.
    let half_sum = half_a + half_b;

    if half_sum.is_infinite() {
        // `a` or `b` must have been infinite.
        return half_sum;
    }

    // Shouldn't overflow or return infinity.
    let half_sum = adjust_rounding_finite(half_a, half_b, half_sum, round, env);

    // Sum saturating to `+-inf`. Raises overflow if `|half_sum| >= 2^maxe`, which is correct since
    // we want to raise overflow regardless of whether we saturate to a finite value or infinity.
    let sum = half_sum * T::TWO;
    if sum.is_infinite() && !env.features().exceptions {
        // Handled here because it's difficult to detect post-factum in `set_exceptions`.
        env.raise(Exceptions::OVERFLOW | Exceptions::INEXACT);
    }

    // Flooring saturates a very positive output to the largest representable float, but a very
    // negative output to `-inf`, because `-inf` is <= every finite value. So we can't just compare
    // absolute values here.
    let limit = (T::MAX / T::TWO).nudge(1);
    let saturate_to_finite = match round {
        RoundingMode::ToNearest => unreachable!(),
        RoundingMode::Floor => half_sum >= limit,
        RoundingMode::Ceil => half_sum <= -limit,
        RoundingMode::Trunc => true,
        RoundingMode::ToNearestTiesToMaxMagnitude => unreachable!(),
    };

    if saturate_to_finite {
        sum.nudge(-1)
    } else {
        sum
    }
}

// On hosts with exceptions, this function raises `OVERFLOW` if rounding adjusts `sum` from finite
// to infinite (e.g. largest finite float plus a tiny value under ceil). On hosts without
// exceptions, `set_exceptions` is responsible for this instead.
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
    let error = if a.abs() >= b.abs() {
        b - (sum - a)
    } else {
        a - (sum - b)
    };

    // +1 means away from zero, -1 means towards zero.
    let nudge = match round {
        RoundingMode::ToNearest => unreachable!(),
        RoundingMode::Floor => {
            if error < T::ZERO {
                // A zero with an error towards -inf must be -0, so this nudges zero correctly.
                if sum > T::ZERO { -1 } else { 1 }
            } else {
                0
            }
        }
        RoundingMode::Ceil => {
            if error > T::ZERO {
                // A zero with an error towards +inf must be +0, so this nudges zero correctly.
                if sum >= T::ZERO { 1 } else { -1 }
            } else {
                0
            }
        }
        RoundingMode::Trunc => {
            // These comparisons can be written either by comparing `sum` to `0`, or by checking its
            // sign bit, because a value rounded to zero cannot have an error towards zero (as long
            // as rounding retains the sign). Don't optimize this with multiplication, it'll cause
            // precision issues and spurious flags.
            if (error < T::ZERO && sum > T::ZERO) || (error > T::ZERO && sum < T::ZERO) {
                -1
            } else {
                0
            }
        }
        RoundingMode::ToNearestTiesToMaxMagnitude => {
            if error == T::ZERO
                || (error < T::ZERO && sum > T::ZERO)
                || (error > T::ZERO && sum < T::ZERO)
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
                if error * T::TWO == sum.nudge(1) - sum {
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

fn set_exceptions<T: Native, F: Float>(a: T, b: T, sum: T, float1: F, float2: F, env: &mut Env) {
    if sum.is_finite() {
        // We avoid using 2Sum/Fast2Sum to compute the error because it assumes round-to-nearest,
        // while the rounding mode here can be arbitrary, so we use another approach.
        let is_exact = sum - a == b && sum - b == a;

        // If the sum is exact, both subtractions are exact as well and comparisons return true.
        //
        // If the sum is inexact, we use a lemma from [1] to obtain the sign of the error:
        //
        //     Lemma 2.5. Let a and b be two binary FP numbers, with e_a>=e_b. Let s \in {RD(a+b),
        //     RU(a+b)}. The number s-a is a floating-point number (which implies that it will be
        //     computed exactly, with any rounding function).
        //
        // In a nutshell, this means that if `|a| >= |b|`, `sum - a` is exact, so `sum - a == b`
        // fails, and symmetrically for `|a| <= |b|`, so at least one condition returns `false`.
        //
        // [1]: Sylvie Boldo, Stef Graillat, and Jean-Michel Muller. 2017. On the Robustness of the
        //      2Sum and Fast2Sum Algorithms. ACM Trans. Math. Softw. 44, 1, Article 4 (March 2018),
        //      14 pages. https://doi.org/10.1145/3054947
        if !is_exact {
            // Addition can never underflow, so we don't need to test for it.
            env.raise(Exceptions::INEXACT);

            if env.features().round
                && matches!(
                    env.get_rounding_mode(),
                    RoundingMode::Floor | RoundingMode::Ceil | RoundingMode::Trunc
                )
            {
                // Floor, ceil, and trunc can overflow without returning `+-inf`, e.g. floor
                // overflows in such a way if the true sum is `>= 2^(maxe+1)`. For hosts without
                // rounding mode support, this is already handled by `add_with_rounding`, but for
                // powerful hosts we need some extra wiring.
                //
                // We can use the same approach as in `add_with_rounding`, and luckily it's quite
                // cheap if rounding modes are native. As a reminder, as long as we don't have
                // subnormals, `a/2` and `b/2` are exact, `a/2 + b/2` can't overflow and is thus
                // rounded just like `a + b` would be with unlimited exponent range, and so
                // `|a/2 + b/2| >= 2^maxe` is equivalent to the overflow condition for `a + b`.
                //
                // The only new issue is that we can no longer assume that `a` and `b` are normal.
                // Luckily, things still work out. For example, for floor:
                // - Large finite value + positive subnormal returns the same value regardless of
                //   the subnormal, so it doesn't matter that it effectively becomes a bit smaller.
                // - Large finite value + negative subnormal returns the same value as long as the
                //   subnormal doesn't fall to zero, but halving a negative subnormal under floor
                //   mode retains this property.
                let limit = (T::MAX / T::TWO).nudge(1);
                if (a * T::HALF + b * T::HALF).abs() >= limit {
                    env.raise(Exceptions::OVERFLOW);
                }
            }
        }
    } else {
        if sum.is_nan() {
            // IEEE-754 says qNaN propagation doesn't raise invalid operation, only sNaN or newly
            // arising NaNs do.
            if !(a.is_nan() || b.is_nan())
                || (float1.is_signaling_nan() || float2.is_signaling_nan())
            {
                env.raise(Exceptions::INVALID);
            }
        } else {
            // IEEE-754 says overflow raises inexact as well.
            env.raise(Exceptions::OVERFLOW | Exceptions::INEXACT);
        }
    }
}
