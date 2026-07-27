# Plotting

Pure functions `convergencedata`, `regretdata`, `partialdependence`, and
`evaluationsdata` do not load a plotting package.

Loading Makie activates the four figure constructors and their mutating counterparts:
`convergenceplot`, `regretplot`, `objectiveplot`, and `evaluationsplot`.
`objectiveplot` uses the fitted SMBO model or an explicitly supplied compatible model;
it does not silently fit a model for non-model-based algorithms.
`evaluationsdata` and `evaluationsplot` show evaluated observations by default;
`IncludeKnownObservations()` explicitly includes seeded observations at evaluation zero.
Surrogate predictions are loss-oriented internally and partial-dependence values are
converted back to the result's objective orientation for maximization.

`reference/generate_python_plots.py` and `reference/generate_julia_plots.jl`
render the same saved optimization trace. The paired output in
`reference/plots/comparison.png` is the visual-style parity fixture.
`reference/compare_plots.jl` rerenders the Julia figures and checks their aspect,
foreground density, luminance, and edge density against the Python images in CI.
