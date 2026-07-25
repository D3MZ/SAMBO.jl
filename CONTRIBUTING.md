# Contributing

Run `julia --project=. -e 'using Pkg; Pkg.test()'` and the benchmark scripts before submitting changes. Follow the Julia [style guide](https://docs.julialang.org/en/v1/manual/style-guide/) and [performance tips](https://docs.julialang.org/en/v1/manual/performance-tips/): function arguments first, `!` for mutation, concrete parametric fields, column-major traversal, preallocation, and function barriers around dynamic setup.

## Provenance and licensing

Sambo.jl is a source-guided port of AGPL-licensed Python SAMBO and is therefore distributed under AGPL-3.0-or-later, not MIT. Preserve attribution and document upstream source files consulted in pull requests. Do not copy third-party code carrying an incompatible license. New algorithm implementations should cite either their upstream AGPL source or the relevant paper/specification.

Changes affecting parity must update `spec/parity.md`; graphing changes should add structural and CairoMakie reference-image tests.
