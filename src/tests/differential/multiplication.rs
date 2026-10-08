use super::{BitwiseCmp, diff_test};
use crate::{
    F32, Float, RoundingMode,
    native::{self, Native},
    ops::mul,
};

#[test]
fn multiplication() {
    let check = |a: F32, b: F32| {
        for round in [
            RoundingMode::ToNearest,
            RoundingMode::Floor,
            RoundingMode::Ceil,
            RoundingMode::Trunc,
            // RoundingMode::ToNearestTiesToMaxMagnitude,
        ] {
            diff_test(
                round,
                |env| BitwiseCmp(mul(a, b, env)),
                || format!("{a:?} * {b:?} in {round:?}"),
            );
        }
    };

    let mut rng = fastrand::Rng::with_seed(1);
    for _ in 0..1000000 {
        let a = F32::from_bits(rng.u32(..));
        let b = F32::from_bits(rng.u32(..));
        check(a, b);
    }
    // for _ in 0..100 {
    //     let a = native::F32::from_bits(rng.u32(..));
    //     for sum_exp in -150..130 {
    //         for sum_nudge in -5..5 {
    //             let sum = native::F32::from_bits(2.0f32.powi(sum_exp).to_bits()).nudge(sum_nudge);
    //             let b_base = sum - a;
    //             for b_nudge in -5..5 {
    //                 let b = b_base.nudge(b_nudge);
    //                 check(F32::from_bits(a.to_bits()), F32::from_bits(b.to_bits()));
    //             }
    //         }
    //     }
    // }
    // for _ in 0..1000 {
    //     let a = F32::from_bits(rng.u32(..));
    //     for b in [
    //         F32::from_native_transmuting_nan(native::F32::INFINITY),
    //         F32::from_native_transmuting_nan(-native::F32::INFINITY),
    //         F32::CANONICAL_NAN,
    //         F32::CANONICAL_SIGNALING_NAN,
    //     ] {
    //         check(a, b);
    //         check(b, a);
    //     }
    // }
}
