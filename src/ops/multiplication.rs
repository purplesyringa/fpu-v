use crate::{
    Env, Float, RoundingMode,
    common::{Operation, compute_operation},
    native::{Bits, Fast, Native, Quiet},
};

struct Mul<F: Float> {
    a: F::Native,
    b: F::Native,
    a_float: F,
    b_float: F,
}

#[inline(always)]
pub fn mul<T: Float>(a: T, b: T, env: &mut Env) -> T {
    let result = compute_operation(
        &Mul {
            a: env.to_native(a),
            b: env.to_native(b),
            // See the explanation in `addition.rs` for why we need to store emulated floats.
            a_float: a,
            b_float: b,
        },
        env,
    );
    // `compute_operation` never returns inputs directly without applying some kind of FP operation
    // to them, so NaNs are autocanonicalized on platforms that support that.
    env.from_native_canonicalizing_nan_after_op(result)

    // XXX underflow
    // let product = if let Some(round) = env.emulate_rounding_mode() {
    //     mul_with_rounding(a, b, round, env)
    // } else if env.features().exceptions && !env.features().underflow_after_rounding {
    //     mul_with_underflow(a, b)
    // } else {
    //     a * b
    // };
}

impl<F: Float> Operation for Mul<F> {
    type T = F::Native;

    fn compute(&self) -> Self::T {
        self.a * self.b
    }

    fn error(&self, result: Self::T) -> Self::T {
        product_error(self.a, self.b, result)
    }

    fn may_ne_overflow(&self) -> bool {
        // We'd like to write a condition without FP operations that implies
        //     |a * b| < 2^maxe * (2 - 2^-(mantissa_len+1))
        // and holds as often as possible. We can do that by computing a bitwise sum of `a` and `b`
        // and inspecting the resulting exponent, which is roughly correlated with `log2 |a * b|`.
        //
        // Let's formalize this. For a number `a` with an unbiased exponent `k`, we have:
        //     |a| <= 2^k * (2 - 2^-mantissa_len).
        // Note that this holds even for subnormals. Therefore,
        //     |a * b| <= 2^(k_a + k_b) * (2 - 2^-mantissa_len)^2
        //              < 2^(k_a + k_b + 1) * (2 - 2^-(mantissa_len+1)),
        // so as long as the exponent field in the sum of `a` and `b` is at most `maxe - 1`
        // (adjusting for bias), we're golden.
        let limit = Self::T::with_biased_exponent(
            (-(Self::T::MIN_EXPONENT - 1) * 2 + Self::T::MAX_EXPONENT) as u32,
        );
        // We'd need an `&` for `abs` anyway, so might as well filter out the mantissa to avoid
        // dealing with the mantissa overflowing into the exponent.
        (self.a.to_bits() & Self::T::EXPONENT_MASK) + (self.b.to_bits() & Self::T::EXPONENT_MASK)
            >= limit.to_bits()
    }

    fn compute_reduced(&self) -> Self::T {
        // `may_overflow` returning `true` implies `k_a + k_b >= maxe`. While `|a| >= 2^k_a` doesn't
        // hold in general due to subnormals, we still have:
        //     |a| >= 2^(k_a - mantissa_len + 1)
        // ...as long as `a` is non-zero. Since at most one of `a` and `b` can be subnormal while
        // still triggering `may_overflow`, this implies that:
        //     |a * b| >= 2^(k_a + k_b - mantissa_len + 1) >= 2^(maxe - mantissa_len + 1).
        //
        // From this and `|a|, |b| < 2^(maxe + 1)` (assuming there are no infinities), we get
        // `|a|, |b| >= 2^-mantissa_len`.
        //
        // Let `k = ceil(maxe / 2) + 1`. Then
        //     |a * 2^-k|, |b * 2^-k| >= 2^(-mantissa_len + k) > 2^mine,
        // which means underflow doesn't occur during multiplication by `2^-k`, and
        //     |(a * 2^-k) * (b * 2^-k)| < 2^(2 * (maxe + 1 - k)) <= 2^maxe
        //     |(a * 2^-k) * (b * 2^-k)| >= 2^(maxe - mantissa_len + 1 - 2*k) >= 2^mine,
        // which means no overflow or underflow occurs during the multiplication of the two reduced
        // numbers.
        //
        // Note that we can't just multiply `a` or `b` by `2^-2k` individually, since that can
        // underflow.
        //
        // Special cases:
        // - If `a` is zero, `may_overflow` only returns `true` if `k_b >= maxe`, which implies that
        //   `b * coeff` can't overflow.
        // - If `a` is infinite, `k_a + k_b >= maxe` implies `k_b >= -1`, so multiplying by `2^-k`
        //   still can't underflow.
        let k = (Self::T::MAX_EXPONENT as u32).div_ceil(2) as i32 + 1;
        let coeff = Self::T::TWO.powi(-k);
        (self.a * coeff) * (self.b * coeff)
    }

    fn error_reduced(&self, result_reduced: Self::T) -> Self::T {
        let k = (Self::T::MAX_EXPONENT as u32).div_ceil(2) as i32 + 1;
        let coeff = Self::T::TWO.powi(-k); // XXX: deduplicate this constant
        product_error(self.a * coeff, self.b * coeff, result_reduced)
    }

    fn lift(&self, result_reduced: Self::T) -> Self::T {
        // `2^2k` doesn't fit in a float because `2k > maxe`. We could technically use `/ 2^-2k`
        // because subnormals are a thing, but that's much slower than two multiplications.
        let k = (Self::T::MAX_EXPONENT as u32).div_ceil(2) as i32 + 1;
        let coeff = Self::T::TWO.powi(k);
        (result_reduced * coeff) * coeff
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
        // Checks `a * b >= 2^(maxe+1)`.
        todo!()
    }

    #[inline(always)]
    fn is_exact_native(&self, result: Self::T, env: &mut Env) -> bool {
        let round = env.get_rounding_mode();

        if env.features().fma {
            // `a * b - result` can't be expressed as a float in general, because it may be below
            // the subnormal range. But we can still use FMA.
            //
            // For floor, `result <= a * b`, so `-a * b + result <= 0`, and any error will be
            // floored to a non-zero negative value. Similarly, for ceil, `-a * b + result >= 0`,
            // and any error will be ceiled to a non-zero positive value.
            if matches!(round, RoundingMode::Floor | RoundingMode::Ceil) {
                return Quiet(env.mul_add(-self.a, self.b, result)) == Quiet(Self::T::ZERO);
            }

            // For other modes this approach fails. For example, for trunc, if `a * b` was positive
            // and rounded down, i.e. `result < a * b`, `-a * b + result` will be negative and will
            // be rounded up, so it may still be zeroed. But we can still make decisions based on
            // the sign of the FMA.
            //
            // If `result = a * b` is exact, `-a * b + result` is a sum of values of different signs
            // (possibly zeros), for which IEEE-754 says the return value is `+0`. Similarly, for
            // `a * b - result`, the return value is also `+0`.
            //
            // Yet if `result < a * b`, `-a * b + result` will be `-0` or less, and if
            // `result > a * b`, `a * b - result` will be `-0` or less.
            return env.mul_add(-self.a, self.b, result).to_bits() == Self::T::ZERO.to_bits()
                && env.mul_add(self.a, self.b, -result).to_bits() == Self::T::ZERO.to_bits();
        }

        if !result.is_finite() {
            return false;
        }

        // For some rounding modes, exactness can be checked with division. For example, for floor,
        // if the product is inexact:
        //     round(a * b) < a * b
        //     => round(a * b) / |a| < sign(a) * b
        //     => round(round(a * b) / |a|) < sign(a) * b
        // Note that we need to divide by the absolute value so that the errors from multiplication
        // and division can't cancel out. We don't need this for trunc.
        //
        // If `a = 0`, this check will compare with `NaN`, so we use `< !` instead of `==`. Since
        // the result is finite, we don't need to worry about `0 * inf` and `0 * NaN`.
        match round {
            RoundingMode::Floor => {
                return !(Quiet(result / self.a.abs())
                    < Quiet(if self.a.is_sign_negative() {
                        -self.b
                    } else {
                        self.b
                    }));
            }
            RoundingMode::Ceil => {
                return !(Quiet(result / self.a.abs())
                    > Quiet(if self.a.is_sign_negative() {
                        -self.b
                    } else {
                        self.b
                    }));
            }
            RoundingMode::Trunc => {
                let trunc_b = result / self.a;
                return !(Quiet(trunc_b) < Quiet(self.b) || Quiet(trunc_b) > Quiet(self.b));
            }
            RoundingMode::ToNearest | RoundingMode::ToNearestTiesToMaxMagnitude => {}
        }

        // For NE and RMM, we use a more general algorithm. We check whether the lowest bit set in
        // `a * b` (when treated mathematically as a sum of `2^k_i`, so the index can be negative)
        // fits in `result`. This validates that no bits were lost due to rounding.
        //
        // It's cheaper than computing the exact error with the Veltkamp algorithm, which fails for
        // subnormals anyway.
        //
        // Let `k` be the index of the lowest bit set in the exact product. To check exactness, we
        // test `round(result - 2^k) != result`. Here's why that works.
        //
        // - If all bits of the answer fit in the mantissa, the subtraction is exact and this test
        //   passes, signifying an exact computation.
        // - If two or more bits are rounded away, `2^k` is below 0.5 ulp and the test fails.
        // - The most interesting scenario is when exactly one bit gets rounded away. That is a tie,
        //   so tie resolution matters. Depending on the rounding mode, we need different proofs:
        //   - For NE, the exact product `...?|1` is rounded to `...0` (possibly incrementing the
        //     exponent), which attracts nearby ties, so `round(result - 2^k) = result`.
        //   - For RMM, the initial computation of `a * b` rounds the tie up, so `result - 2^k` is
        //     the same mathematical value and rounds to `result` again (assuming that
        //     `result >= 0`). This is the only place where it matters that we test
        //     `result - 2^k` and not `result + 2^k`.

        // `lowest_bit_set` produces a zero for zero inputs, which our logic can't distinguish from
        // an underflown `2^k` (which must be inexact), so get this out of the way first.
        if Fast(self.a) == Fast(Self::T::ZERO) || Fast(self.b) == Fast(Self::T::ZERO) {
            return true;
        }
        // This computation should be vectorized over `a` and `b`. Rust refuses to do so, probably
        // due to the number of nested trivial functions. The C port should make sure this works.
        let bit_a = lowest_bit_set(self.a).abs();
        let bit_b = lowest_bit_set(self.b).abs();

        // Underflows can occur during the multiplication. That's fine if they are rounded down to
        // `0` (because, if the correct product has a bit that must underflow, the entire product
        // must be inexact, so `false` is the correct answer). This is the case for NE, which
        // priortizes `0` over the smallest subnormal, but not for RMM. For now we use an ugly
        // check, but there might be a better way to do it.
        if round == RoundingMode::ToNearestTiesToMaxMagnitude
            && Fast(bit_a * (bit_b * Self::T::TWO))
                < Fast(Self::T::TWO.powi(Self::T::MIN_EXPONENT + 1))
        {
            return false;
        }

        Fast(result.abs() - bit_a * bit_b) != Fast(result.abs())
    }

    #[inline(always)]
    fn is_overflow_finite_native(&self, env: &mut Env) -> bool {
        // We need to check if `|a * b| >= 2^(maxe+1)`.
        let round = env.get_rounding_mode();
        let limit = Self::T::TWO.powi(Self::T::MAX_EXPONENT);

        if env.features().fma {
            // With FMA available, we can rewrite `a * b >= 2^(maxe+1)` as
            //     a * b - 2^maxe >= 2^maxe,
            // which for flooring is equivalent to
            //     round(a * b - 2^maxe) >= 2^maxe.
            // Ceiling and truncation use similar conditions.
            return match round {
                RoundingMode::ToNearest => unreachable!(),
                RoundingMode::Floor => Fast(env.mul_add(self.a, self.b, -limit)) >= Fast(limit),
                RoundingMode::Ceil => Fast(env.mul_add(self.a, self.b, limit)) <= Fast(-limit),
                RoundingMode::Trunc => {
                    Fast(env.mul_add(self.a.abs(), self.b.abs(), -limit)) >= Fast(limit)
                }
                RoundingMode::ToNearestTiesToMaxMagnitude => unreachable!(),
            };
        }

        // Rewrite `a * b >= 2^(maxe+1)` as `a * (b * 0.5) >= 2^maxe`. For flooring, this is
        // equivalent to
        //     round(a * (b * 0.5)) >= 2^maxe
        // ...as long as `b * 0.5` doesn't underflow. But if it underflows, it must be small enough
        // that `a * b` can't be close to `2^maxe` anyway. Other rounding modes work similarly.
        let half_product = self.a * (self.b * Self::T::HALF);
        match round {
            RoundingMode::ToNearest => unreachable!(),
            RoundingMode::Floor => Fast(half_product) >= Fast(limit),
            RoundingMode::Ceil => Fast(half_product) <= Fast(-limit),
            RoundingMode::Trunc => Fast(half_product.abs()) >= Fast(limit),
            RoundingMode::ToNearestTiesToMaxMagnitude => unreachable!(),
        }
    }

    fn is_underflow_native(&self) -> bool {
        // We need to check if `|round(a * b)| < 2^mine`, where `round` limits the mantissa length,
        // but not the exponent range.
        //
        // If `result` is a subnormal, we know that an underflow has occurred: if the subnormal
        // value is preferred to a normal value during the full rounding, adding more precision
        // (which is what making the exponent unlimited does) isn't going to change that, so the
        // normal cannot be rounded to.
        //
        // But the converse claim fails. Consider `2^mine - eps`, with `eps` chosen to be slightly
        // under 1 ulp for the would-be exponent `mine - 1` (point shown as `*` on the diagram
        // below). With an unlimited exponent range, it NE-rounds to `2^mine - ulp`, so the
        // underflow occurs. But with a limited exponent range, it's located between `2^mine` and
        // `2^mine - 2 ulp`, and the former is closer, so `result` is a normal number.
        //
        //     ..|..|..|..|*.|.....|.....|.....| with unlimited range
        //                |<
        //
        //     ..|.....|...*.|.....|.....|.....| with limited range
        //                 ->|
        //
        // This condition can only occur if the rounded product is exactly `+-2^mine`. To
        // disambiguate it, we can increase the exponent range just a little by computing
        // `a * (b * 2)` instead of `a * b`, and checking its absolute value for `< 2^(mine+1)`
        // instead of (effectively) `< 2^mine`. This covers both the subnormal rounded product and
        // borderline rounded product cases, as long as no overflow occurs during the calculation of
        // `b * 2` -- and it can't occur for small products, because `|a * b| < 2^mine` implies
        //     |b| < 2^mine / |a| <= 2^mine / 2^(mine - mantissa_len) = 2^mantissa_len
        // ...as long as `a != 0`, which holds because the product is inexact. For
        // `|a * b| >= 2^mine` this can overflow, but it'll return `false`, which is correct.
        //
        // Note that `a * (b * 2)` needs to be computed with the same rounding mode as `product` was
        // computed originally, so this logic is only valid for native modes. It also cannot in
        // general be simplified with FMA, because for asymmetric rounding modes, `a * b` rounds
        // differently than `-a * b`, so bounds checking becomes tricky and slower than this logic.
        Fast((self.a * (self.b * Self::T::TWO)).abs())
            < Fast(Self::T::TWO.powi(Self::T::MIN_EXPONENT + 1))
    }
}

fn product_error<T: Native>(x: T, y: T, z: T) -> T {
    // Computes a value `error` such that `a * b = product + error` mathematically, where `product`
    // is the NE rounding of the mathematical value `a * b`.

    // https://ir.cwi.nl/pub/9159/9159D.pdf
    // "disregarding overflow and underflow" -- oopsie daisy?
    // > Moreover, we assume that floating-point addition and subtraction are optimal [NE with
    // > arbitrary ties] and multiplication is faithful [any rounding].

    // // Veltkamp's algorithm (necessary because we need `product` to be close to `a * b` for a very
    // // specific definition of "close"):
    // let a_high = T::from_bits(a.to_bits() & !T::HALF_MANTISSA_MASK);
    // let a_low = a - a_high;
    // let b_high = T::from_bits(b.to_bits() & !T::HALF_MANTISSA_MASK);
    // let b_low = b - b_high;
    // (((a_high * b_high - product) + a_high * b_low) + a_low * b_high) + a_low * b_low
    panic!()
}

// XXX: this function is untested because my hardware (x86) never uses it
fn mul_with_underflow<T: Native>(a: T, b: T) -> T {
    // RISC-V detects underflow "after rounding", a condition which is defined as the mathematical
    // result having an absolute value below `2^mine` when rounded to a limited mantissa length, but
    // without limiting the exponent range.
    //
    // On the other hand, many hosts detect underflow "before rounding", when the mathematical
    // result has an absolute value below `2^mine` without any rounding.
    //
    // Note that the fully rounded result being a subnormal is yet another different condition. The
    // differences between the three conditions are:
    // 1. "Before rounding": mantissa is unlimited, exponent is unlimited.
    // 2. "After rounding": mantissa is limited, exponent is unlimited.
    // 3. Subnormal: mantissa is limited, exponent is limited.
    //
    // We need to implement (2) based on access to (1) and (3), which is quite tricky to get right.
    // Getting a subnormal implies underflow "after rounding", and that one in turn implies
    // underflow "before rounding", but there are no converse implications.
    //
    // The differences manifest at `2^mine - eps` for different epsilons. Specifically, products
    // that are closer to `2^mine` than to the largest subnormal round to normals under NE, but
    // underflow "before rounding":
    //
    //                            prev       2^mine
    //     |           |           |       #-->|           |           |
    //     _______________________/       a*b  \________________________
    //            subnormals                            normals
    //
    // Among them, products that are closer to the midpoint between `2^mine` and the previous
    // subnormal (`mid`), than to `2^mine`, round to that midpoint with an unlimited exponent and
    // thus underflow "after rounding", but round to normals with a limited exponent:
    //
    //                            prev  mid  2^mine
    //     |     |     |     |     |     |<#-->|           |           |
    //     _______________________/       a*b  \________________________
    //        unbounded normals                         normals
    //
    // The native underflow is spurious specifically for products that round up to `2^mine` instead
    // of `mid` when the exponent range is increased by 1 (which, in this case, is equivalent to
    // making it unlimited):
    //
    //                            prev  mid  2^mine
    //     |     |     |     |     |     |   #>|           |           |
    //     _______________________/        a*b \________________________
    //        unbounded normals                         normals
    //
    // We can detect that by computing `a * (b * 2) == 2^(mine+1)` (though that also includes
    // rounding *down* to `2^mine`). But we need to be careful about not accidentally raising the
    // overflow flag when computing the product. Since we can't compute `a * b` immediately either
    // due to spurious underflows, we have to estimate the orders of magnitude beforehand.

    // We'd like to write a condition without FP operations that implies `|a * b| >= 2^mine` and
    // holds as often as possible. `|a| >= 2^mantissa_len || |b| >= 2^mantissa_len` suffices, since
    // either implies
    //     |a * b| >= 2^(mantissa_len + (mine - mantissa_len)) = 2^mine,
    // ...but we can find a better condition by adding `|a|` and `|b|` bitwise. This adds the
    // exponents of `|a|` and `|b|`, so it's actually a reasonable approximation of the product as
    // long as we adjust for the exponent bias.
    //
    // Subnormals introduce some trouble: we'd love to say that `exp_a + exp_b >= mine` implies
    // `|a * b| >= 2^mine`, but that fails, because having a stored exponent `exp_a` doesn't imply
    // `|a| >= 2^exp_a` if `exp_a = mine - 1`. We actually need
    //     exp_a + exp_b >= mine + mantissa_len - 1
    // ...to claim anything useful: if only `a` is denormal, this corresponds to
    //     mine - 1 + exp_b >= mine + mantissa_len - 1
    //     => exp_b >= mantissa_len
    //     => |b| >= 2^mantissa_len
    //     => |a * b| >= 2^(mine - mantissa_len) * 2^mantissa_len = 2^mine,
    // ...and if both `a` and `b` are denormal, the condition always fails. (This also sometimes
    // matches zeros by accident, but that's fine.)
    //
    // The pseudo-product has the biased "exponent"
    //     (exp_a - (mine - 1)) + (exp_b - (mine - 1)) = exp_a + exp_b - 2 * (mine - 1),
    // or one greater if the mantissa overflows into the exponent. As long as it's at least
    // `2 - mine + mantissa_len`, we have:
    //     exp_a + exp_b - 2 * (mine - 1) + 1 >= 2 - mine + mantissa_len
    //     =>  exp_a + exp_b >= mine + mantissa_len - 1,
    // which implies `|a * b| >= 2^mine` as discussed above.
    //
    // Note that as long as `exp_a >= mantissa_len + 1`, the condition always holds, which means
    // that a) we're not losing much compared to independent checks, b) it also captures `NaN`s and
    // infinities.
    let limit = 2 + (-T::MIN_EXPONENT) as u32 + T::MANTISSA_LEN;
    if a.abs().to_bits() + b.abs().to_bits() >= T::with_biased_exponent(limit).to_bits() {
        return a * b;
    }

    // `a` and `b` both being below `2^(mantissa_len + 1)` implies that neither `b * 2`, nor
    // `a * (b * 2)` overflow.
    //
    // This computation raises underflow if `a * b < 2^(mine-1)`, but that implies
    // `round(a * b) < 2^mine`, so it's fine. Similarly, it can raise inexact, though it's not
    // guaranteed to do so if `a * b` is on the subnormal boundary.
    let double_product = a * (b * T::TWO);

    // `a` and `b` are finite, so there can be no `NaN` here.
    if Fast(double_product) != Fast(T::TWO.powi(T::MIN_EXPONENT + 1)) {
        // No risk of spurious underflow. Note that returning `double_product * 0.5` here would be
        // incorrect, because that introduces double rounding for subnormal `double_product`s.
        return a * b;
    }

    core::hint::cold_path();

    // Lemma: given a set of valid FP numbers A and its subset B, if `x` rounded wrt. A rounds to
    // a value in B, it rounds to the same value wrt. B, regardless of the rounding mode.
    //
    // Given `double_product = 2^(mine+1)`, this implies that `a * b` rounds to `2^mine` and we know
    // which value to return.
    //
    // This lemma covers the inexact flag by implying `round(a * b) = round(a * (b * 2)) * 0.5`:
    // `* 2` and `* 0.5` are exact, `a * (b * 2)` is the only source of inexactness on the right, so
    // it raises the inexact flag iff `a * b` raises it. We can thus avoid evaluating `a * b` on
    // this path (which is the entire point, because it underflows spuriously).
    T::TWO.powi(T::MIN_EXPONENT)
}

/// Returns the smallest value `+-2^k` such that the binary expansion of the finite non-zero input
/// `x` contains the addend `2^k`.
///
/// The sign of the resulting value matches the sign of `x`.
fn lowest_bit_set<T: Native>(x: T) -> T {
    #[cfg(not(feature = "explore"))]
    assert!(x.is_finite(), "non-finite input to lowest_bit_set");

    // The formulas below are broken for zeros.
    #[cfg(not(feature = "explore"))]
    assert!(Fast(x) != Fast(T::ZERO), "zero input to lowest_bit_set");

    // We can find the lowest bit either by adding the exponent to the ctz of the mantissa manually,
    // or by mixing bit twiddling with FP operations. The latter is preferable because it can be
    // vectorized, while the former requires ctz, and this code path is most commonly reached on
    // targets like Wasm, where vectorization is available.

    // This is essentially just `x fp- (x & !(x - 1))`, but with extra logic to handle zeroed
    // mantissa without overflowing into the exponent. Don't ask me how I fit this in 4 operations,
    // this formula was found by brute-force, but you can make sense of it.
    let a = x.to_bits() & !T::MANTISSA_MASK;
    T::from_bits(x.to_bits() & (a - x.to_bits())) - T::from_bits(a)

    // Bit twidding-only implementation returning the index `k`:
    //     let exp = x.biased_exponent().max(1) as i32 - 1 - T::MIN_EXP; // max(1) for subnormals
    //     let ctz = (x.to_bits() | (T::MANTISSA_MASK + T::Bits::ONE)).trailing_zeros();
    //     exp + ctz as i32 - T::MANTISSA_LEN as i32
}
