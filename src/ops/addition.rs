use crate::{
    Env, Float, RoundingMode,
    common::{Operation, compute_operation},
    native::{Fast, Native, Quiet},
};

struct Add<F: Float> {
    a: F::Native,
    b: F::Native,
    a_float: F,
    b_float: F,
}

#[inline(always)]
pub fn add<T: Float>(a: T, b: T, env: &mut Env) -> T {
    let result = compute_operation(
        &Add {
            a: env.to_native(a),
            b: env.to_native(b),
            // Store the original emulated floats so that they can be tested for sNaN without
            // involving platform-specific shenanigans -- see various comments in `native` and
            // `types` for the differences between `is_emulated_signaling_nan` and
            // `is_native_signaling_nan`.
            a_float: a,
            b_float: b,
        },
        env,
    );
    // `compute_operation` never returns inputs directly without applying some kind of FP operation
    // to them, so NaNs are autocanonicalized on platforms that support that.
    env.from_native_canonicalizing_nan_after_op(result)
}

pub fn sub<T: Float>(a: T, b: T, env: &mut Env) -> T {
    // IEEE-754 says `a - b` is equivalent to `a + (-b)`. The compiler should be able to rewrite
    // the fast path of `do_add` to optimize out the negation. `-b` also doesn't set exceptions.
    let result = compute_operation(
        &Add {
            a: env.to_native(a),
            b: -env.to_native(b),
            a_float: a,
            b_float: b,
        },
        env,
    );
    env.from_native_canonicalizing_nan_after_op(result)
}

impl<F: Float> Operation for Add<F> {
    type T = F::Native;

    fn compute(&self) -> Self::T {
        self.a + self.b
    }

    fn error(&self, result: Self::T) -> Self::T {
        sum_error(self.a, self.b, result)
    }

    fn may_ne_overflow(&self) -> bool {
        // We use two conditions.
        //
        // First, `max(|a|, |b|) < 2^maxe` recognizes the most common case by magnitude.
        //
        // But to make `compute_reduced` fast, we also need to catch subnormal inputs (which is
        // correct because adding a subnormal never causes an overflow in NE mode). But we can do
        // slightly better than looking for subnormals exclusively. Notice that as long as:
        //     min(|a|, |b|) < 2^(maxe-manissa_len-1),
        // ...combining it with:
        //     |a|, |b| <= 2^maxe * (2 - 2^-mantissa_len)
        // ...gives:
        //     |a + b| < 2^maxe * (2 - 2^-mantissa_len) + 2^(maxe-manissa_len-1)
        //         = 2^maxe * (2 - 2^-(mantissa_len+1)),
        // ...which implies no NE overflow.
        let a_bits = self.a.abs().to_bits();
        let b_bits = self.b.abs().to_bits();
        // Make sure not to trigger exceptions on NaN; arithmetic with NaN cannot overflow.
        Quiet(Self::T::from_bits(a_bits.max(b_bits)))
            >= Quiet(Self::T::TWO.powi(Self::T::MAX_EXPONENT))
            && Fast(Self::T::from_bits(a_bits.min(b_bits)))
                >= Fast(Self::T::TWO.powi(Self::T::MAX_EXPONENT - Self::T::MANTISSA_LEN as i32 - 1))
    }

    fn compute_reduced(&self) -> Self::T {
        // Computes `(a + b) * 0.5` without overflows. We know that `may_overflow` returned `true`,
        // so `a` and `b` aren't subnormals, and thus we can compute `a/2`, `b/2` without underflow.
        self.a * Self::T::HALF + self.b * Self::T::HALF
    }

    fn error_reduced(&self, result_reduced: Self::T) -> Self::T {
        sum_error(
            self.a * Self::T::HALF,
            self.b * Self::T::HALF,
            result_reduced,
        )
    }

    fn lift(&self, result_reduced: Self::T) -> Self::T {
        result_reduced * Self::T::TWO
    }

    fn has_nan_input(&self) -> bool {
        self.a.is_nan() || self.b.is_nan()
    }

    fn has_signaling_nan_input(&self) -> bool {
        self.a_float.is_emulated_signaling_nan() || self.b_float.is_emulated_signaling_nan()
    }

    fn has_infinite_input(&self) -> bool {
        self.a.is_infinite() || self.b.is_infinite()
    }

    fn is_finite_overflow(&self, round: RoundingMode) -> bool {
        // Checks `a + b >= 2^(maxe+1)`.
        //
        // Let's say `a >= b`, then we test `b >= (2^maxe - a) + 2^maxe`. If `a >= 2^maxe`, the RHS
        // is exact and we get a correct result. If `a < 2^maxe`, LHS is also `< 2^maxe` while RHS
        // is `>= 2^maxe`, so we correctly get `false`. Similarly if `a < b`.
        let limit = Self::T::TWO.powi(Self::T::MAX_EXPONENT);
        let (max, min) = if Fast(self.a) >= Fast(self.b) {
            (self.a, self.b)
        } else {
            (self.b, self.a)
        };
        let is_too_high = Fast(min) >= Fast((limit - max) + limit);
        let is_too_low = Fast(max) <= Fast((-limit - min) - limit);
        match round {
            RoundingMode::ToNearest => unreachable!(),
            RoundingMode::Floor => is_too_high,
            RoundingMode::Ceil => is_too_low,
            RoundingMode::Trunc => is_too_high || is_too_low,
            RoundingMode::ToNearestTiesToMaxMagnitude => unreachable!(),
        }
    }

    fn is_exact_native(&self, result: Self::T) -> bool {
        // We avoid using 2Sum/Fast2Sum to compute the error because it assumes round-to-nearest,
        // while the rounding mode here can be arbitrary, so we use another approach.
        // XXX: this should be `Fast`
        Quiet(result - self.a) == Quiet(self.b) && Quiet(result - self.b) == Quiet(self.a)
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
        // This condition fails for `NaN` for obvious reasons, and for infinity because either both
        // inputs are finite and the comparisons fail trivially, or one of the inputs is the same
        // infinity and the corresponding subtraction produces `NaN`. This improves the fast path.
        //
        // [1]: Sylvie Boldo, Stef Graillat, and Jean-Michel Muller. 2017. On the Robustness of the
        //      2Sum and Fast2Sum Algorithms. ACM Trans. Math. Softw. 44, 1, Article 4 (March 2018),
        //      14 pages. https://doi.org/10.1145/3054947
    }

    fn is_overflow_finite_native(&self, round: RoundingMode) -> bool {
        // We want to check `a + b >= 2^(maxe+1)`. As long as we don't have subnormals, `a/2` and
        // `b/2` are exact, `a/2 + b/2` can't overflow and is thus rounded just like `a + b` would
        // be with unlimited exponent range, and so `a/2 + b/2 >= 2^maxe` is equivalent to the
        // overflow condition for `a + b`.
        //
        // For performance, we'd like to divide by `2` by adjusting the exponent instead of writing
        // `* 0.5`. Luckily, things work out in this case even if `a` or `b` are subnormal. The
        // intended result in this case is `false`, and we always get there:
        // - For `2^mine <= |a| < 2^(mine + 1)`, decrementing the exponent produces a subnormal that
        //   is slightly different from the true value, but is still small enough that adding it to
        //   any fintie value can't cause overflow.
        // - For `0 < |a| < 2^mine`, it ends up producing NaN, which propagates to the sum. It can
        //   be an sNaN, which would raise an unexpected exception, but this function is allowed to
        //   raise exceptions per the trait contract.
        // - `a = 0` results in infinity, but adding a zero is always exact, so we don't end up in
        //   this function at all.
        //
        // The same general idea applies to ceil and trunc. Technically we're allowed to use
        // `|a + b|` in all scenarios, because an overflow in the wrong direction results in
        // an infinite output and we're guaranteed the output was finite, but dropping `|...|` is
        // faster if the mode is known.
        let half = |x: Self::T| {
            Self::T::from_bits(x.to_bits() - Self::T::with_biased_exponent(1).to_bits())
        };
        let half_sum = half(self.a) + half(self.b);
        let limit = Self::T::TWO.powi(Self::T::MAX_EXPONENT);
        match round {
            RoundingMode::ToNearest => unreachable!(),
            // XXX: This can be `Fast`, but it can get `NaN` inputs.
            RoundingMode::Floor => Quiet(half_sum) >= Quiet(limit),
            RoundingMode::Ceil => Quiet(half_sum) <= Quiet(-limit),
            RoundingMode::Trunc => Quiet(half_sum.abs()) >= Quiet(limit),
            RoundingMode::ToNearestTiesToMaxMagnitude => unreachable!(),
        }
    }
}

#[inline(always)]
fn sum_error<T: Native>(a: T, b: T, sum: T) -> T {
    // Avoid using the branchless variant of 2Sum to compute the error: as [1] points out, it can
    // return a non-finite error even for finite inputs in edge cases.
    //
    // Note that the sign of the error indicates the direction that cancels the rounding, not the
    // direction rounding was performed towards. In other words, `a + b = sum + error`.
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
