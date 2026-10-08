use crate::{
    Env, Exceptions, Float, RoundingMode,
    common::set_exceptions_non_finite,
    native::{Bits, Fast, Native, Quiet},
};

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
    // denormal (`mid`), than to `2^mine`, round to that midpoint with an unlimited exponent and
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
    // overflow flag. Since we can't compute `a * b` immediately either, this forces branching on
    // `a` and `b` before doing anything.
    //
    // [1]: https://stackoverflow.com/questions/79864400/unexpected-underflow-state-after-floating-point-multiplication-has-a-subnormal-r#comment140939988_79866158

    if a.abs().to_bits().max(b.abs().to_bits()) >= T::TWOP_MANTISSA_DIGITS.to_bits() {
        // `|a| >= 2^mantissa_digits` implies
        //     |a * b| >= 2^(mantissa_digits + (mine - mantissa_digits)) = 2^mine,
        // i.e. no underflow of any kind, as long as `b != 0`. `b = 0` doesn't introduce any
        // underflows either. Checking both `a` and `b` ensures we get onto this fast path as often
        // as possible. The bitwise check instead of a floating-point one is a) faster, b) also
        // captures NaN and infinities in this branch.
        return a * b;
    }

    // `a` and `b` both being small implies that neither `b * 2`, nor `a * (b * 2)` overflow. This
    // computation raises underflow for `a * b < 2^(mine-1)`, but that implies
    // `round(a * b) < 2^mine`, so it's fine. Similarly, it can raise inexact, though it's not
    // guaranteed to be so if `a * b` is on the subnormal boundary.
    let double_product = a * (b * T::TWO);

    // `a` and `b` are finite, so there can be no `NaN` here.
    if Fast(double_product) != Fast(T::MIN_POSITIVE * T::TWO) {
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
    T::MIN_POSITIVE
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

    // env.features().fma
    if false {
        // TODO: handle machines with FMA.
        //
        // We cannot compute the error `a * b - product` as a float generally, because the error may
        // be smaller than the subnormal range, causing us to erroneously claim an exact result.
        // Maybe try FMA multiple times with different factors?
        unimplemented!()
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
        let limit = T::MIN_POSITIVE * T::TWO;
        let is_underflow = Fast((a * (b * T::TWO)).abs()) < Fast(limit);
        if is_underflow {
            env.raise(Exceptions::UNDERFLOW);
        }
    }

    set_overflow(a, b, env);
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
    // So we also nudge `bit_b * 2` up by 1:
    // - For `bit_b * 2` rounded up to the largest finite value, this nudges it to infinity.
    // - For `bit_b * 2` rounded up to infinity, this nudges it to a (likely signaling) NaN, which
    //   forces us to write the comparison carefully. We use a quiet comparison despite being on
    //   a path without exceptions, because fast comparisons assert lack of NaNs for more thorough
    //   testing; the C port should use a fast comparison.
    // - In the remaining cases, this slightly increases the value, up to 50% in case where `bit_b`
    //   is the smallest subnormal, which causes `bit_a * (bit_b * 2)` to increase by the same
    //   percentage, and since we're only dealing with powers of two, any increase below 100%
    //   doesn't break anything despite looking scary.
    //
    // Notes:
    //
    // Multiplying by 2 and not `2^mantissa_digits`, which would simplify some logic, is necessary,
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
    !(Quiet((bit_a * (bit_b * T::TWO).nudge(1)).abs()) < Quiet(bit_product * T::TWO))
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
    // for more info; in a nutshell, here we just need to check if `|round(a * b)| >= 2^(maxe+1)`
    // would hold if the exponent range was unbounded.
    //
    // Lemma: `round(x * y)` has a maximum exponent of `exp_x + exp_y + 1`, because in worst-case
    // scenario:
    //     x = 2^exp_x * (2 - 2^-mantissa_digits)
    //     y = 2^exp_y * (2 - 2^-mantissa_digits)
    // ...we have:
    //     x * y = 2^(exp_x + expy) * (2 - 2^-mantissa_digits)^2
    //           <= 2^(exp_x + expy + 1) * (2 - 2^-mantissa_digits)
    // ...and that implies it fits in `exp_x + exp_y + 1` regardless of the rounding mode.
    //
    // Now let `k = (maxe + 1) / 2`; this is always a whole number for standard floats. By the
    // lemma, `round((a * 2^-k) * (b * 2^-k))` has a maximum possible exponent of:
    //     (exp_a - k) + (exp_b - k) + 1 = exp_a + exp_b - maxe <= maxe
    // ...so it doesn't overflow and thus rounds just like if `round(a * b)` had an unlimited
    // exponent from the above. We can thus safely rewrite the check as:
    //     |round((a * 2^-k) * (b * 2^-k))| >= 2^(maxe+1) * 2^-2k = 1
    // ...but only as long as no underflow occurs in computing `a * 2^-k` and `b * 2^-k`.
    //
    // Let's see when an underflow can happen and what it can affect.
    // - Can we get a false negative? `a * 2^-k` underflows if `exp_a - k < mine`, but then
    //       exp_a + exp_b + 1 <= mine + k + maxe < maxe
    //   ...which indicates no overflow, so the answer is no.
    // - Can we get a false positive? If `a * 2^-k` underflows, it rounds either to zero or to the
    //   smallest subnormal depending on the rounding mode. In the former case, the condition
    //   simplifies to `0 >= 1` and fails. In the latter case, we get:
    //       2^(mine-mantissa_digits) * |b * 2^-k| < 2^(mine-mantissa_digits) * 2^(maxe+1-k)
    //           = 2^((mine + maxe) - mantissa_digits + 1 - k)
    //           = 2^(1 - mantissa_digits + 1 - k)
    //           < 1,
    //   so the comparison fails again and the answer is no.
    if Fast(((a * T::TWOP_NEG_MAXE_SPLIT) * (b * T::TWOP_NEG_MAXE_SPLIT)).abs()) >= Fast(T::ONE) {
        env.raise(Exceptions::OVERFLOW | Exceptions::INEXACT);
    }
}

/// Returns the smallest value `+-2^k` such that the binary expansion of the finite non-zero input
/// `x` contains the addend `2^k`.
///
/// The sign of the resulting value matches the sign of `x`.
fn lowest_bit_set<T: Native>(x: T) -> T {
    assert!(x.is_finite(), "non-finite input to lowest_bit_set");

    // The formulas below are completely broken for zeros (they either return the wrong values or
    // mess up something else), which is why this function shouldn't be invoked on zeros. We could
    // fix that with special-casing, but that's slow and unnecessary.
    assert!(Fast(x) != Fast(T::ZERO), "zero input to lowest_bit_set");

    // There are two ways to find the lowest bit:
    // 1. With bit twiddling alone, we can essentially replicate a small part of a soft FPU. That
    //    mainly requires adding the exponent to the ctz of the mantissa. It's not slow on any
    //    modern CPUs, but since we're on the "no host exceptions" path, chances are we're not on
    //    a modern CPU and we don't have native ctz (or we're running on Wasm).
    // 2. By combining bit twiddling with FP operations, we get somewhat simpler logic, but, what's
    //    more important, it can be *vectorized* between `a` and `b`. This provides a large speedup
    //    over bit twiddling even with native ctz available, as long as SIMD is present, and for
    //    Wasm it feels fair to assume that it is.
    // A major difference between these two approaches is that the former returns the *index* of the
    // bit, while the latter returns `2^k` itself. This function implements the latter approach, but
    // an untested snippet for the former is provided below in case we need it for some platform.

    // As long as the mantissa is non-zero, `x & (x - 1)` unsets its lowest bit, and then
    // `x - (x & (x - 1))` returns a float corresponding to that bit. This can't underflow because
    // subnormals can always represent 1 ulp of any float, and this is at least 1 ulp.
    //
    // For a zero mantissa, we have a power of two, so `x` itself is the correct answer. We mask out
    // the subtrahend instead of blending as an optimization (this retains the correct sign because
    // `x` is non-zero). We also use `x & !MANTISSA_MASK == x` instead of `x & MANTISSA_MASK == 0`
    // to check for a zero mantissa so that the comparison runs on FP numbers and not integers,
    // because SSE2 doesn't support 64-bit integer SIMD comparisons, but supports `double` SIMD
    // comparisons.
    x - if Fast(T::from_bits(x.to_bits() & !T::MANTISSA_MASK)) == Fast(x) {
        T::ZERO
    } else {
        T::from_bits(x.to_bits() & x.nudge(-1).to_bits())
    }

    // Bit twidding-only implementation returning the index `k`:
    //     let exp = x.biased_exponent().max(1) as i32 - 1 - T::MIN_EXP; // max(1) for subnormals
    //     let ctz = (x.to_bits() | (T::MANTISSA_MASK + T::Bits::ONE)).trailing_zeros();
    //     exp + ctz as i32 - T::MANTISSA_DIGITS as i32
}

/// Returns the smallest value `2^k` such that `2^k` can be set in the finite input `x` given its
/// exponent. A zero input is treated as a subnormal value.
fn lowest_bit_fit_abs<T: Native>(x: T) -> T {
    assert!(x.is_finite(), "non-finite input to lowest_bit_fit");

    let factor = T::ONE / T::TWOP_MANTISSA_DIGITS;

    let exp_bits = x.to_bits() & T::EXPONENT_MASK;
    if exp_bits == T::Bits::ZERO {
        // This almost always means `x` is a subnormal. If `x` is a zero, it means that the product
        // `a * b` underflowed to `0` while `a != 0` and `b != 0`, which shouldn't commonly occur.
        // This evaluation is constant, and it's only consumed by a comparison, so this doesn't
        // enter microcode and we don't have to worry about zeros getitng slowed down by anything
        // other than branch prediction.
        T::MIN_POSITIVE * factor
    } else {
        T::from_bits(exp_bits) * factor
    }

    // Bit twiddling-only implementation returning the index `k`:
    //     x.biased_exponent().max(1) as i32 - 1 - T::MIN_EXP - T::MANTISSA_DIGITS as i32
}
