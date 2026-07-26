from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sambo import minimize
from sambo.plot import (
    plot_convergence,
    plot_evaluations,
    plot_objective,
    plot_regret,
)


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "plots" / "python"
OUTPUT.mkdir(parents=True, exist_ok=True)


def objective(point):
    x, y, z = point
    return (
        (x - 0.2) ** 2
        + 5 * (y - x**2) ** 2
        + (z + 0.35) ** 2
        + 0.25 * (x - z) ** 2
    )


result = minimize(
    objective,
    bounds=[(-1.0, 1.0)] * 3,
    method="smbo",
    max_iter=60,
    rng=42,
)

plots = {
    "convergence": plot_convergence(result, true_minimum=0.0),
    "regret": plot_regret(result, true_minimum=0.0),
    "objective": plot_objective(
        result,
        resolution=18,
        n_samples=128,
        names=["x", "y", "z"],
    ),
    "evaluations": plot_evaluations(result, names=["x", "y", "z"]),
}

for name, figure in plots.items():
    figure.savefig(OUTPUT / f"{name}.png", dpi=120, facecolor="white")
    plt.close(figure)

np.savetxt(
    ROOT / "plots" / "trace.csv",
    np.column_stack([np.asarray(result.xv, dtype=float), result.funv]),
    delimiter=",",
    header="x,y,z,objective",
    comments="",
)
