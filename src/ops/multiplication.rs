use crate::{
    Env, Exceptions, Float, RoundingMode,
    common::set_exceptions_non_finite,
    native::{Bits, Fast, Native, Quiet},
};

#[inline(always)]
pub fn mul<T: Float>(a: T, b: T, env: &mut Env) -> T {
    let (float1, float2) = (a, b);
    let a = env.to_native(a);
    let b = env.to_native(b);

    let product = if let Some(round) = env.emulate_rounding_mode() {
        // TODO
        unimplemented!()
    } else if env.features().exceptions && !env.features().underflow_after_rounding {
        mul_with_underflow(a, b)
    } else {
        a * b
    };
    if !env.features().exceptions {
        // See the explanation in `addition.rs` for why we need to pass the emulated floats here.
        set_exceptions(a, b, product, float1, float2, env);
    }
    // TODO: verify this claim:
    // `do_mul` never returns inputs directly without applying some kind of FP operation to them, so
    // NaNs are autocanonicalized on platforms that support that.
    env.from_native_canonicalizing_nan_after_op(product)
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
    let limit = T::TWO.powi(T::MIN_EXPONENT + 1);
    if Fast(double_product) != Fast(limit) {
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

// fn error_ne<T: Native>(a: T, b: T, product: T) -> T {
//     // Computes a value `error` such that `a * b = product + error` mathematically, where `product`
//     // is the NE rounding of the mathematical value `a * b`.

//     // https://ir.cwi.nl/pub/9159/9159D.pdf
//     // "disregarding overflow and underflow" -- oopsie daisy?
//     // > Moreover, we assume that floating-point addition and subtraction are optimal [NE with
//     // > arbitrary ties] and multiplication is faithful [any rounding].

//     // Veltkamp's algorithm (necessary because we need `product` to be close to `a * b` for a very
//     // specific definition of "close"):
//     let a_high = T::from_bits(a.to_bits() & !T::HALF_MANTISSA_MASK);
//     let a_low = a - a_high;
//     let b_high = T::from_bits(b.to_bits() & !T::HALF_MANTISSA_MASK);
//     let b_low = b - b_high;
//     (((a_high * b_high - product) + a_high * b_low) + a_low * b_high) + a_low * b_low
// }

#[inline(always)]
fn set_exceptions<T: Native, F: Float>(
    a: T,
    b: T,
    product: T,
    float1: F,
    float2: F,
    env: &mut Env,
) {
    if !product.is_finite() {
        set_exceptions_non_finite(a, b, product, float1, float2, env);
        return;
    }

    if env.features().fma {
        if is_exact_with_fma(a, b, product, env) {
            return;
        }
    } else {
        if is_exact_without_fma(a, b, product, env) {
            // Inexactness arising from overflows to finite values, such as the ones that happen in
            // flooring mode, are not detected by this logic, so they have to be checked for
            // explicitly. See `set_overflow` for more info.
            set_overflow(a, b, env);
            return;
        }
    }

    env.raise(Exceptions::INEXACT);

    // RISC-V detects underflow "after rounding", a condition which is defined as the mathematical
    // result having an absolute value below `2^mine` when rounded to a limited mantissa length, but
    // without limiting the exponent range.
    //
    // `product` being a subnormal implies that an underflow has occurred, because if the subnormal
    // value is preferred to a normal value during rounding, adding more precision isn't going to
    // change that, so the normal cannot be rounded to.
    //
    // However, the converse claim fails. Consider `2^mine - eps`, with `eps` chosen to be slightly
    // under 1 ulp for the would-be exponent `mine - 1` (point shown as `*` on the diagram below).
    // With an unlimited exponent range, it NE-rounds to `2^mine - ulp`, so the underflow occurs.
    // But with a limited exponent range, it's located between `2^mine` and `2^mine - 2 ulp`, and
    // the former is closer, so `product` is a normal number.
    //
    //     ..|..|..|..|*.|.....|.....|.....| with unlimited range
    //                |<
    //
    //     ..|.....|...*.|.....|.....|.....| with limited range
    //                 ->|
    //
    // This condition can only occur if the rounded product is exactly `+-2^mine`. To disambiguate
    // it, we can increase the exponent range just a little by computing `a * (b * 2)` instead of
    // `a * b`, and checking its absolute value for `< 2^(mine+1)` instead of (effectively)
    // `< 2^mine`. This covers both the subnormal rounded product and borderline rounded product
    // cases, as long as no overflow occurs during the calculation of `b * 2` -- and it can't occur
    // for small products, because `|a * b| < 2^mine` implies
    //     |b| < 2^mine / |a| <= 2^mine / 2^(mine - mantissa_len) = 2^mantissa_len
    // ...as long as `a != 0`, which holds because the product is inexact. For `|a * b| >= 2^mine`
    // this can overflow, but it'll just result in `false`, which is correct.
    //
    // Note that `a * (b * 2)` needs to be computed with the same rounding mode as `product` was
    // computed originally, so this logic only runs for native modes, and emulated ones are handled
    // by `mul_with_rounding` instead (XXX).
    if env.emulate_rounding_mode().is_none() {
        let limit = T::TWO.powi(T::MIN_EXPONENT + 1);
        let is_underflow = Fast((a * (b * T::TWO)).abs()) < Fast(limit);
        if is_underflow {
            env.raise(Exceptions::UNDERFLOW);
        }
    }

    set_overflow(a, b, env);
}

fn is_exact_with_fma<T: Native>(a: T, b: T, product: T, env: &mut Env) -> bool {
    assert!(
        product.is_finite(),
        "non-finite product passed to is_exact_with_fma"
    );

    // The error `a * b - product` can't be expressed as a float in general, because it may be below
    // the subnormal range, causing us to erroneously claim an exact result. In the worst case, for
    // `a` and `b` equal to the smallest subnormal, `2^(mine - mantissa_len)`, the error may be
    // `2^(mine - mantissa_len)` times smaller than the smallest positive float!
    //
    // We want to keep `fma(a, b, -product) == 0` as a fast path, though. This is correct regardless
    // of rounding mode as long as the mathematical error is `>= 2^(mine - mantissa_len)`. Giving
    // a good lower bound on the error is difficult in presence of subnormals that have a shortened
    // mantissa, but we can at least say that:
    // - `a` has no bits below `floor(log2 |a|) - mantissa_len`.
    // - `b` has no bits below `floor(log2 |b|) - mantissa_len`.
    // - `a * b` thus has no bits below `floor(log2 |a|) + floor(log2 |b|) - 2 * mantissa_len`,
    //   so the error, if present, must be at least 2 to the power of that value.
    //
    // This gives us
    //     floor(log2 |a|) + floor(log2 |b|) - 2 * mantissa_len >= mine - mantissa_len
    //     <=> floor(log2 |a|) + floor(log2 |b|) >= mine + mantissa_len
    // ...as a sufficient condition for an error to not be rounded away (it can still be rounded,
    // but not straight to zero). Since:
    //     floor(log2 |a|) + floor(log2 |b|) >= floor(log2 |a| + log2 |b|) - 1
    //         = floor(log2 |a * b|) - 1,
    // the condition
    //     floor(log2 |a * b|) >= mine + mantissa_len + 1
    // is also sufficient, and that is implied by
    //     |a * b| >= 2^(mine + mantissa_len + 1),
    // which in turn is implied by
    //     round(|a * b|) > 2^(mine + mantissa_len + 1).
    let limit = T::TWO.powi(T::MIN_EXPONENT + T::MANTISSA_LEN as i32 + 1);
    if Fast(product.abs()) > Fast(limit) {
        return Fast(env.mul_add(a, b, -product)) == Fast(T::ZERO);
    }

    // For the remaining values, we have
    //     round(|a * b|) <= 2^(mine + mantissa_len + 1),
    // which is sufficient to guarantee that multiplying `product` by `2^-(mine - mantissa_len)`
    // to make the error visible won't overflow:
    //     (mine + mantissa_len + 1) - (mine - mantissa_len) = 2 * mantissa_len + 1 < maxe
    // It also gives bounds on `a` and `b`, as long as the inputs are non-zero:
    //     |a * b| < round(|a * b|) * (1 + 2^-mantissa_len) < round(|a * b|) * 2
    //         <= 2^(mine + mantissa_len + 2)
    //     |a| = |a * b| / |b| <= |a * b| / 2^(mine - mantissa_len)
    //         < 2^(mine + mantissa_len + 2) / 2^(mine - mantissa_len)
    //         = 2^(2 * mantissa_len + 2)
    // ...and same for `|b|`. This gives `exp_a <= 2 * mantissa_len + 1`, which doesn't allocate as
    // much space for upscaling as `product`, but `2^(-(mine - mantissa_len) / 2)` still fits:
    //     2 * mantissa_len + 1 - (mine - mantissa_len) / 2
    //     = 2.5 * mantissa_len + 1 + (-mine) / 2
    //     = 2.5 * mantissa_len + 0.5 + maxe / 2 < maxe.
    // In practice we ceil the `/ 2` to load fewer constants.
    //
    // The above reasoning fails for zero inputs. For example, if `a = 0`, this can overflow when
    // computing `b * coeff`. This can imply one of two things:
    // 1. It overflows to a finite value, we get `fma(0, finite, 0) = 0`, and everything works fine.
    // 2. It overflows to infinity, we get `fma(0, inf, 0) = NaN`.
    // To handle (2), we treat NaN errors as exact, because NaNs don't arise otherwise.
    //
    // Note that we have to multiply `product` by `2^2k` in two steps because `2k > maxe`. After the
    // first multiplication, the number is guaranteed to be normal, so we could just increase the
    // exponent by `k` instead of doing another multiplication, but we avoid doing that to load
    // fewer constants.
    let coeff =
        T::TWO.powi((-(T::MIN_EXPONENT - T::MANTISSA_LEN as i32) as u32).div_ceil(2) as i32);
    let error = env.mul_add(a * coeff, b * coeff, -(product * coeff) * coeff);
    error.is_nan() || Fast(error) == Fast(T::ZERO)
}

// Checks for exactness, assuming there was no finite overflow.
fn is_exact_without_fma<T: Native>(a: T, b: T, product: T, env: &mut Env) -> bool {
    assert!(
        product.is_finite(),
        "non-finite product passed to is_exact_without_fma"
    );

    // On hosts without FMA, we can use a trick to perform an exactness check without computing the
    // error. It suffices to check if the lowest bit set in `a * b` (when treated mathematically as
    // a sum of `2^k_i`, so the index can be negative) is no lower than the lowest bit that *can*
    // fit in `product` (which is determined only by its exponent). Essentially this checks that no
    // bits were lost due to rounding. This works in all cases except when the product overflows to
    // a finite value, which is why the caller of this function also checks for overflow.
    //
    // This is cheaper than running Veltkamp's algorithm to compute the error, and that algorithm
    // doesn't work in presence of underflows anyway, *and* it requires the active rounding mode to
    // be round-to-nearest, which we can't guarantee.
    //
    // What we want to check here is essentially
    //     |lowest_bit_set(a) * lowest_bit_set(b)| >= |lowest_bit_fit(product)|,
    // but there are a ton of nuances.

    // Let's start with the functions themselves. `lowest_bit_set` only works for non-zero inputs,
    // and while it can be adjusted to return `+inf` for zeros, it's cheaper to just perform this
    // check at the beginning.
    if Fast(a) == Fast(T::ZERO) || Fast(b) == Fast(T::ZERO) {
        return true;
    }

    // This should be vectorized over `a` and `b`. Rust refuses to do that for some reason, probably
    // due to the number of nested trivial functions.
    let bit_a = lowest_bit_set(a);
    let bit_b = lowest_bit_set(b);
    let bit_product = lowest_bit_fit_abs(product);

    // Underflows can occur during the multiplication, which is fine if they are rounded down to `0`
    // (because, if the correct product has a bit that must underflow, the entire product must be
    // inexact, so `0 >= non-zero` -> `false` is the correct answer).
    if !env.features().round {
        return Fast((bit_a * bit_b).abs()) >= Fast(bit_product);
    }

    // But in rounding modes other than NE `bit_a * bit_b` can round up to the smallest subnormal,
    // causing false positives. We can fix the underflow by computing
    //     |lowest_bit_set(a) * (lowest_bit_set(b) * 2)| >= |lowest_bit_fit(product) * 2|
    // ...instead, which makes the multiplication underflow to a number smaller than the one on
    // the right, thus consistently returning `false`.
    //
    // That trades it for an overflow *only* for `b = +-2^maxe`, but such a multiplication is
    // always exact (assuming the product is finite), so getting `+inf >= ...` -> `true` here works
    // out fine. But we aren't guaranteed to get `+inf`, instead we might get the largest finite
    // number due to the rounding mode being weird, which is *slightly* below `bit_b * 2`
    // mathematically, but that's enough for the check to fail.
    //
    // So we need a small adjustment there as well. Note that we can't just nudge the value up by 1
    // unconditionally: while it seems like `(bit_b * 2).nudge(1)` can be larger than `bit_b * 2`
    // only by 50% at worst (when `bit_b` is the smallest subnormal), this stops being true once
    // it's multiplied by `bit_a`, e.g. under ceiling we can get a 100% increase compared to
    // `bit_a * (bit_b * 2)`.
    //
    // More notes:
    //
    // Multiplying by 2 and not `2^mantissa_len`, which would simplify some logic, is necessary,
    // because even 4 allows `b = 2^maxe * (1 + 0.5)` to cause an inexact product while overflowing
    // to `+inf` on the `* 4`.
    //
    // We can invoke `lowest_bit_fit` on a zero input if the product is below the range of
    // subnormals. In this case, treating a zero `product` as a subnormal is correct from the
    // precision perspective, so there is no special logic for it. But it does force us to handle
    // the "`a = 0` or `b = 0`" case manually, otherwise we'll get `0 >= non-zero`.
    //
    // `bit_product * 2` doesn't overflow because it's `lowest_bit_fit` and not `lowest_bit_set`, so
    // its upper bound is determined by the lowest bit of the mantissa, not its hidden bit.
    let bit_b2 = bit_b * T::TWO;
    let bit_b2 = bit_b2.nudge(bit_b2.bottom_bit() as i8); // largest finite value -> infinity
    !(Quiet((bit_a * bit_b2).abs()) < Quiet(bit_product * T::TWO))
}

fn set_overflow<T: Native>(a: T, b: T, env: &mut Env) {
    if !env.features().round
        || !matches!(
            env.get_rounding_mode(),
            RoundingMode::Floor | RoundingMode::Ceil | RoundingMode::Trunc
        )
    {
        return;
    }

    // Floor, ceil, and trunc can overflow without returning `+-inf`, e.g. floor overflows in such
    // a way if the true product is `>= 2^(maxe+1)`. See the corresponding comment in `addition.rs`
    // for more info; formally, we need to check if `|round(a * b)| >= 2^(maxe+1)` would hold if the
    // exponent range was unbounded.

    if env.features().fma {
        // With FMA available, we can rewrite `a * b >= 2^(maxe+1)` (which is different from
        // `round(a * b)`, but not for flooring) as
        //     a * b - 2^maxe >= 2^maxe,
        // which for flooring is equivalent to
        //     round(a * b - 2^maxe) >= 2^maxe.
        // Notably, this last equivalence fails for ceiling, which means we can't use
        //     round(|a * b| - 2^maxe) >= 2^maxe
        // for all rounding modes and have to use mode-specific conditions.
        let coeff = T::TWO.powi(T::MAX_EXPONENT);
        let is_overflow = match env.get_rounding_mode() {
            RoundingMode::ToNearest => unreachable!(),
            RoundingMode::Floor => Fast(env.mul_add(a, b, -coeff)) >= Fast(coeff),
            RoundingMode::Ceil => Fast(env.mul_add(a, b, coeff)) <= Fast(-coeff),
            RoundingMode::Trunc => Fast(env.mul_add(a.abs(), b.abs(), -coeff)) >= Fast(coeff),
            RoundingMode::ToNearestTiesToMaxMagnitude => unreachable!(),
        };
        if is_overflow {
            env.raise(Exceptions::OVERFLOW | Exceptions::INEXACT);
        }
        return;
    }

    // Lemma: for normal numbers `x` and `y`, `round(x * y)` has a maximum exponent of
    // `exp_x + exp_y + 1`, because in worst-case scenario:
    //     x = 2^exp_x * (2 - 2^-mantissa_len)
    //     y = 2^exp_y * (2 - 2^-mantissa_len)
    // ...we have:
    //     x * y = 2^(exp_x + expy) * (2 - 2^-mantissa_len)^2
    //           <= 2^(exp_x + expy + 1) * (2 - 2^-mantissa_len)
    // ...so the product fits in `exp_x + exp_y + 1` regardless of the rounding mode.
    //
    // Let `k = (maxe + 1) / 2`, always a whole number for standard floats. By the lemma,
    // `round((a * 2^-k) * (b * 2^-k))` has an upper bound on the exponent of `maxe`:
    //     (exp_a - k) + (exp_b - k) + 1 = exp_a + exp_b - maxe <= maxe
    // ...so it doesn't overflow and rounds just like if `round(a * b)` had an unlimited exponent
    // from the above. We can thus write the check as:
    //     |round((a * 2^-k) * (b * 2^-k))| >= 2^(maxe+1) * 2^-2k = 1
    // ...but only as long as no underflow occurs in computing `a * 2^-k` and `b * 2^-k`.
    //
    // Let's see what an underflow can affect.
    // - Can a false negative happen? `a * 2^-k` underflows if `exp_a - k < mine`, which implies
    //       exp_a + exp_b + 1 <= mine + k + maxe < maxe
    //   ...which indicates no overflow, so no.
    // - Can a false positive happen? If `a * 2^-k` underflows, it rounds either to zero or to the
    //   smallest subnormal depending on the rounding mode. In the worst-case scenario, we get:
    //       2^(mine-mantissa_len) * |b * 2^-k| < 2^(mine-mantissa_len) * 2^(maxe+1-k)
    //           = 2^((mine + maxe) - mantissa_len + 1 - k)
    //           = 2^(1 - mantissa_len + 1 - k)
    //           < 1,
    //   ...so the comparison fails, correctly.
    let coeff = T::TWO.powi(-(T::MAX_EXPONENT + 1) / 2);
    if Fast(((a * coeff) * (b * coeff)).abs()) >= Fast(T::ONE) {
        env.raise(Exceptions::OVERFLOW | Exceptions::INEXACT);
    }
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
    // this formula was found by brute-force.
    let a = x.to_bits() & !T::MANTISSA_MASK;
    T::from_bits(x.to_bits() & (a - x.to_bits())) - T::from_bits(a)

    // Bit twidding-only implementation returning the index `k`:
    //     let exp = x.biased_exponent().max(1) as i32 - 1 - T::MIN_EXP; // max(1) for subnormals
    //     let ctz = (x.to_bits() | (T::MANTISSA_MASK + T::Bits::ONE)).trailing_zeros();
    //     exp + ctz as i32 - T::MANTISSA_LEN as i32
}

/// Returns the smallest value `2^k` such that `2^k` can be set in the finite input `x` given its
/// exponent. A zero input is treated as a subnormal value.
fn lowest_bit_fit_abs<T: Native>(x: T) -> T {
    assert!(x.is_finite(), "non-finite input to lowest_bit_fit");

    let factor = T::TWO.powi(-(T::MANTISSA_LEN as i32));

    let exp_bits = x.to_bits() & T::EXPONENT_MASK;
    if exp_bits == T::Bits::ZERO {
        // This almost always means `x` is a subnormal. `x = 0` happens only if the product `a * b`
        // underflows to `0` while `a != 0` and `b != 0`, which shouldn't commonly occur.
        T::TWO.powi(T::MIN_EXPONENT) * factor
    } else {
        T::from_bits(exp_bits) * factor
    }

    // Bit twiddling-only implementation returning the index `k`:
    //     x.biased_exponent().max(1) as i32 - 1 - T::MIN_EXPONENT - T::MANTISSA_LEN as i32
}
