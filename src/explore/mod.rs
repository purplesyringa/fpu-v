//! Use `cargo asm -F explore <function-name>` to explore the approximate lowering of each function.

use crate::{Env, F32, HostFeatures, RoundingMode, ops};

#[inline(never)]
pub fn mul32(a: F32, b: F32) -> F32 {
    let mut env = Env::new(HostFeatures::X86);
    // env.set_rounding_mode(RoundingMode::Floor);
    ops::mul(a, b, &mut env)
}
