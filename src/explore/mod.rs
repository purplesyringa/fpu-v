//! Use `cargo asm -F explore <function-name>` to explore the approximate lowering of each function.

use crate::{Env, F32, HostFeatures, RoundingMode, ops};

#[inline(never)]
pub fn add32(a: F32, b: F32) -> F32 {
    let mut env = Env::new(HostFeatures::WASM);
    env.set_rounding_mode(RoundingMode::Floor);
    ops::add(a, b, &mut env)
}
