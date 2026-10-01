mod addition;

use crate::{Env, Float, HostFeatures, RoundingMode, native};
use core::fmt::Debug;

fn diff_test<R: Copy + Eq + Debug>(
    round: RoundingMode,
    f: impl Fn(&mut Env) -> R,
    context: impl FnOnce() -> String,
) {
    let results = [
        HostFeatures::X86,
        HostFeatures::WASM,
        HostFeatures::ONLY_ROUNDING,
        HostFeatures::ONLY_EXCEPTIONS,
    ]
    .map(|features| {
        // Clear exceptions and/or rounding mode set by previous runs
        native::set_env(native::Env::default());

        let mut env = Env::new(features);
        env.set_rounding_mode(round);
        (features, (f(&mut env), env.get_exceptions()))
    });

    let (features1, out1) = results[0];
    for &(features2, out2) in &results[1..] {
        if out1 != out2 {
            assert_eq!(
                out1,
                out2,
                "comparing {features1:?} vs {features2:?}, {}",
                context()
            );
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct BitwiseCmp<F>(F);

impl<F: Float> PartialEq for BitwiseCmp<F> {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl<F: Float> Eq for BitwiseCmp<F> {}
