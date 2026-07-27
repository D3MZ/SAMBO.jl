# Interoperability

Weak dependencies activate package extensions for Makie, SurrogatesBase,
OptimizationBase, and MLJTuning. Core SAMBO can be loaded and tested without
these dependencies.

SurrogatesBase adapters convert column-major point matrices at the extension
boundary. Native Gaussian-process prediction remains matrix-oriented and
workspace-backed.

The OptimizationBase adapter translates common limits, callbacks, constraints,
optimization sense, and native return codes. `SAMBOTuning()` integrates SMBO
ask/tell batches with MLJTuning and respects measure orientation and numeric
range scales.

Third-party extensions use the public protocols documented on the
[API page](../api/index.md), not internal solver or workspace fields.
