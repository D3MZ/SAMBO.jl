# Spaces

`Box(lower, upper)` is the homogeneous continuous fast path. `SearchSpace` combines
`Continuous`, native ranges, and `Choices`, and decodes keyword dimensions to a
`NamedTuple`.

All solvers store latent points as `dimensions × candidates` floating-point matrices
in `[0, 1]`. Discrete coordinates are canonicalized to their exact range/category
encoding before evaluation, duplicate checks, and tracing. Discrete decoding uses
equal-width bins, and fixed dimensions always use latent coordinate zero.

`encode` and `decode` form the public conversion boundary. Objectives and constraints
receive decoded native Julia values.
