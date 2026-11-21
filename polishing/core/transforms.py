"""Migrated numerical service: TransformMixin; legacy numerical conventions retained."""
from __future__ import annotations
import numpy as np

class TransformMixin:
    @staticmethod
    def rotate_vectors(vectors, axis, cosine, sine):
        """Rodrigues rotation without invoking a BLAS matrix product."""
        vectors = np.asarray(vectors, dtype=float)
        axis = np.asarray(axis, dtype=float)
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        return (vectors * cosine + np.cross(axis, vectors) * sine +
                np.sum(vectors * axis, axis=-1, keepdims=True) * axis * (1.0 - cosine))
