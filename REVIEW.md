# Independent review

Reviewed against the Julia style guide and performance tips on Julia 1.12.6. Existing source files were not changed.

## Findings

### High

- **The algorithm names overstate the implementations** (`src/algorithms.jl:233-296`). `SCEUA` never uses `complexes`, `complex_size`, or `contraction` and does not evolve shuffled complexes; `SHGO` is random sampling followed by Gaussian perturbations, with no simplicial-homology machinery. README's unqualified “supports SCE-UA” is therefore misleading, although “SHGO-style,” the experimental warning, benchmark labels, and parity document partly correct this.
- **Graph API is advertised but effectively untested** (`ext/SamboMakieExt.jl`, `test/runtests.jl`). No test loads Makie or calls any exported plot function, despite parity rows saying the plots are implemented. Extension loading, supported Makie versions, plot structure, embedding, categorical axes, and image output can all regress undetected.

### Medium

- **Graph constructors do not follow normal Makie return conventions**: they return only `Figure`, while mutating forms return the supplied axis/grid position rather than created plot/axis objects. This makes composition and later customization awkward. Fixed keywords precede `axis...`, so attempts to override keys such as `xlabel`/`ylabel` can also produce duplicate-keyword errors (`ext/SamboMakieExt.jl:17-20, 75-81, 127-133`).
- **Graphs show latent rather than decoded coordinates** (`diagnostics.jl:22-31`, `ext/SamboMakieExt.jl:69-97, 114-157`). Continuous axes are labelled with parameter names but remain 0–1 and receive no bound ticks; `evaluationsdata`'s “decoded observations” docstring is inaccurate because no decoded observations are returned. Partial-dependence queries also ignore constraints. These can make plots materially misleading.
- **The public space API validates too late** (`src/spaces.jl:35-52`). `SearchSpace` accepts unsupported dimension objects and fails later with internal `MethodError`s. Validate dimensions at construction and document/export the dimension protocol if custom dimensions are intended.
- **The optimizer is unnecessarily fixed to `Float64`** (`src/algorithms.jl:78-80, 94`; `src/spaces.jl:106`). This conflicts with the style guide's generic-code advice and prevents preserving `Float32`/other suitable numeric inputs even though algorithm parameter types are generic.
- **SMBO is allocation-heavy**. The supplied run measured about 7.8 MB and 144,694 allocations for only 40 two-dimensional evaluations, versus 47.6 KB/491 for SCE-UA-style and 34.3 KB/293 for SHGO-style. Candidate generation, full prediction arrays, repeated decoding, and copied/indexed candidate matrices are obvious optimization targets (`src/algorithms.jl:94-210`). This falls short of the performance tips' emphasis on measuring allocations and reusing storage.

### Low

- Compact one-line struct/function bodies (`src/core.jl:29-36, 72`) and exported names without docstrings make the API harder to read and document; the generated docs are correspondingly sparse.
- Tests mainly assert budgets, finiteness, and shapes. They do not establish optimization quality, algorithm semantics, threaded behavior, callbacks, multiple outstanding ask/tell batches, table schema, inference, or allocation regressions.

## Tests and benchmarks

- `julia --project=. -e 'using Pkg; Pkg.test()'`: **53/53 passed**.
- `julia --project=benchmark benchmark/benchmarks.jl`: completed on Julia 1.12.6; all methods made exactly 40 objective calls. Seeded minima were 0.7728 (SCE-UA-style), 0.7797 (SMBO-RBF), and 2.0436 (SHGO-style), which is weak quality evidence at this budget, not a parity result.

## Benchmark-claim verdict

**The explicit benchmark claims are honest.** README avoids a speedup claim, discloses non-equivalent algorithms and budget/memory/timing differences, and the script reports actual call counts and quality. Compilation is excluded by BenchmarkTools as stated. However, the benchmark is only one tiny objective, one seed, one budget, and 20 timing samples; it supports reproducibility, not comparative speed, quality, or equivalence. The separate Python script was not runnable in this environment, so no cross-language result was verified.
