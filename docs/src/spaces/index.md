# Spaces

`Box(lower, upper)` is the homogeneous continuous fast path. `SearchSpace`
combines `Continuous`, native ranges, and `Choices`, and decodes keyword
dimensions to a `NamedTuple`.

The default latent coordinate type is `Float64`. Use
`SearchSpace(Float32; ...)` or `SearchSpace(BigFloat; ...)` to select another
floating-point coordinate type.

Solvers store latent points as `dimensions × candidates` matrices in `[0, 1]`.
Discrete coordinates are canonicalized before evaluation, duplicate checks, and
tracing. Fixed dimensions use latent coordinate zero.

`Choices` uses ordinal latent geometry in the order supplied. Use an explicit
alternative encoding when nominal categories require Hamming geometry.

`encode` and `decode` form the public conversion boundary. Objectives and
constraints receive decoded native Julia values.
