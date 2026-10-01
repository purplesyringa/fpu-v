use super::{BitwiseCmp, diff_test};
use crate::{F32, Float, RoundingMode, ops::add};

#[test]
fn addition() {
    let mut rng = fastrand::Rng::with_seed(1);
    for round in [
        RoundingMode::ToNearest,
        RoundingMode::Floor,
        RoundingMode::Ceil,
        RoundingMode::Trunc,
        RoundingMode::ToNearestTiesToMaxMagnitude,
    ] {
        for _ in 0..1000000 {
            let a = F32::from_bits(rng.u32(..));
            let b = F32::from_bits(rng.u32(..));
            diff_test(
                round,
                |env| BitwiseCmp(add(a, b, env)),
                || format!("{a:?} + {b:?} in {round:?}"),
            );
        }
    }
}
