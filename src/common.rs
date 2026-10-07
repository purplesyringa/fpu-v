use crate::{Env, Exceptions, Float, native::Native};

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
