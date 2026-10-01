# fpu-v

This is in-progress test ground for [RVVM](https://github.com/LekKit/RVVM), which currently has a buggy FPU and is overrun with LLM slop doing anything but reducing the messiness.

This collection of snippets provides a reference optimized RISC-V-compliant FPU implementation, including FMA, flags, and rounding mods, only requiring the host to support basic flag-less round-to-nearest arithmetic.

This is not a library, in the sense that it's not something you want to use directly: it uses x86-specific inline assembly as a polyfill for tests because Rust doesn't support floating-point environments. The intention is that these calls are replaced with native C operators and `#pragma STDC FENV_ACCESS` in the C port, and this repo stays as a reference implementation.
