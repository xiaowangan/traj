"""Migrated numerical service: SimulationMixin; legacy numerical conventions retained."""
from __future__ import annotations
import numpy as np

class SimulationMixin:
    @staticmethod
    def fft_convolve_same(data, kernel):
        data = np.asarray(data, dtype=float)
        kernel = np.asarray(kernel, dtype=float)
        shape = (data.shape[0] + kernel.shape[0] - 1,
                 data.shape[1] + kernel.shape[1] - 1)
        spectrum = np.fft.rfftn(data, shape, axes=(0, 1)) * np.fft.rfftn(kernel, shape, axes=(0, 1))
        full = np.fft.irfftn(spectrum, shape, axes=(0, 1))
        y0 = kernel.shape[0] // 2
        x0 = kernel.shape[1] // 2
        return full[y0:y0 + data.shape[0], x0:x0 + data.shape[1]]
