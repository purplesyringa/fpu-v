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

fn add_with_rounding<T: Native>(a: T, b: T, round: RoundingMode, env: &mut Env) -> T {
    if round == RoundingMode::ToNearestTiesToMaxMagnitude {
        // NE addition sets the exact same flags as RMM addition. `OVERFLOW` specifically, the only
        // flag affected by the rounding mode during addition, works out because infinity is even
        // for the purposes of NE, so RMM and NE agree to round towards it.
        return adjust_rounding_finite(a, b, a + b, round, env);
    }

    // For every rounding mode but RMM, it's possible that correctly rounded `a + b` is finite, but
    // rounded to nearest it's infinite. The first thing this implies is that we need to save
    // exception flags, because we may inadvertently raise overflow.
    let ex = env.save_host_exceptions();

    let sum = a + b;
    if !sum.is_infinite() {
        // Keep the exceptions from `a + b`, because everything except `OVERFLOW` is unaffected by
        // rounding mode. `adjust_rounding_finite` makes sure to raise `OVERFLOW` if the correctly
        // rounded sum turns out to be infinite, if necessary.
        return adjust_rounding_finite(a, b, sum, round, env);
    }

    // Separate real infinity from rounding to infinity. As long as `a` and `b` are normal numbers,
    // `round(a + b) = round(a/2 + b/2) * 2`, and if `a + b` is ambiguously close to infinity,
    // `a/2 + b/2` is guaranteed to be comfortably far from it to not cause trouble. We *usually*
    // shouldn't get subnormals here, because `finite + subnormal` never overflows for NE, but `a`
    // can be subnormal if `b` is `+inf` or vice versa. In this case, the end result is still
    // correct, but we need to make sure to perform the halving before restoring exceptions, so that
    // the underflow flag is not raised spuriously.
    let half_a = a * T::HALF;
    let half_b = b * T::HALF;

    env.restore_host_exceptions(ex);

    // Going over affected flags:
    // - `INVALID`: if we didn't get any `NaN`s the first time, we shouldn't get them now.
    // - `OVERFLOW`: may be set either by `a/2 + b/2` or `*2` on real infinities, which is correct.
    // - `UNDERFLOW`: addition and `*2` never underflow.
    // - `INEXACT`: when caused by precision loss, matches between `a/2 + b/2` and `a + b` in
    //   absence of subnormals. When caused by overflow, gets set either by addition or `*2`; this
    //   subsumes the case where subnormals are halved with loss of precision.
    let half_sum = half_a + half_b;
    if half_sum.is_infinite() {
        return half_sum;
    }
    adjust_rounding_finite(half_a, half_b, half_sum, round, env) * T::TWO
}

// On hosts with exceptions, this function raises `OVERFLOW` if rounding adjusts `sum` from finite
// to infinite. On hosts without exceptions, `set_exceptions` is responsible for this instead.
fn adjust_rounding_finite<T: Native>(a: T, b: T, sum: T, round: RoundingMode, env: &mut Env) -> T {
    if !sum.is_finite() {
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

    // Nudging the maximum value away from zero produces infinity, which is correct numerically, but
    // doesn't automatically raise the overflow and inexact flags.
    if env.features().exceptions && sum.is_infinite() {
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
