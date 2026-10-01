mod addition;

use crate::{Env, HostFeatures, RoundingMode, native};

fn diff_test<R>(
    round: RoundingMode,
    f: impl Fn(&mut Env) -> R,
    compare: impl FnOnce(R, R),
    context: impl FnOnce() -> String,
) {
    // Clear exceptions set by previous runs
    native::set_env(native::Env::default());

    let mut x86_env = Env::new(HostFeatures::X86);
    x86_env.set_rounding_mode(round);
    let x86_out = f(&mut x86_env);
    let x86_ex = x86_env.get_exceptions();

    // Reset host rounding mode to NE
    native::set_env(native::Env::default());

    let mut wasm_env = Env::new(HostFeatures::WASM);
    wasm_env.set_rounding_mode(round);
    let wasm_out = f(&mut wasm_env);
    let wasm_ex = wasm_env.get_exceptions();

    compare(x86_out, wasm_out);
    if x86_ex != wasm_ex {
        assert_eq!(x86_ex, wasm_ex, "{}", context());
    }
}
