# fpu-v

This is in-progress test ground for [RVVM](https://github.com/LekKit/RVVM), which currently has a buggy FPU that is overrun with hotfixes doing anything but reducing the messiness.

This collection of snippets provides a reference optimized RISC-V-compliant FPU implementation, including FMA, flags, and rounding modes, only requiring the host to support basic flagless round-to-nearest arithmetic.

This is not a library, in the sense that it's not something you want to use directly: it uses x86-specific inline assembly as a polyfill for tests because Rust doesn't support floating-point environments, etc. The intention is that these calls are replaced with native C operators and `#pragma STDC FENV_ACCESS` in the C port, and this repo stays as a reference implementation where exhaustive tests can be run and commentary can be present.


## LLM policy

LLM usage in RVVM has historically caused a great deal of trouble and is partially what got us into this mess. While LLM models -- even frontier ones -- can uncover new bugs by testing, they empirically turn out to be terrible at reasoning about their causes, and have a habit of preferring quick overconfident fixes and ignoring subtleties without a trace. For us, validating code and writing human-accessible proofs is already difficult; having to deal with LLM output just makes it more tiring, and the amount of spam and back-and-forth involved that no one learns anything from adds insult to injury.

For this reason among others, LLM-assisted pull requests are disallowed for this repo. If you want to fix a bug or introduce a feature, you need to work through it yourself, not let a model convince you of the appropriate approach. It's fine if you get it wrong the first time if you're willing to learn.

Using LLMs to find issues is allowed, if only because we can't distringuish the source of reports, but the issues have to be human-written for all the same reasons. If you're a human, making mistakes is understandable and discussing math-based bugs like this makes for a refreshing experience; but if you didn't work through the bug without LLM assistance, please don't add root cause analysis to the report, just keep the reproducer and a short comment if there are nuances.
