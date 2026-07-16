"""Python port of ``KMRCD/halfkernel.m``: a synthetic two-half-moons dataset."""

import numpy as np


def halfkernel(n1, n2, minx, r1, r2, noise, ratio, rng=None):
    rng = rng if rng is not None else np.random.default_rng()

    phi1 = rng.random(n1) * np.pi
    inner = np.column_stack([
        minx + r1 * np.sin(phi1) - 0.5 * noise + noise * rng.random(n1),
        r1 * ratio * np.cos(phi1) - 0.5 * noise + noise * rng.random(n1),
        np.ones(n1),
    ])

    phi2 = rng.random(n2) * np.pi
    outer = np.column_stack([
        minx + r2 * np.sin(phi2) - 0.5 * noise + noise * rng.random(n2),
        r2 * ratio * np.cos(phi2) - 0.5 * noise + noise * rng.random(n2),
        np.zeros(n2),
    ])

    return np.vstack([inner, outer])
