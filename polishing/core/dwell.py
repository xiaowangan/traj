"""Migrated numerical service: DwellSolverMixin; legacy numerical conventions retained."""
from __future__ import annotations
import numpy as np
import math

class DwellSolverMixin:
    @classmethod
    def solve_bounded_least_squares(cls, surface, kernel, dx, dy,
                                    uniform=0.0, min_dwell=0.0, max_dwell=np.inf,
                                    trajectory_trim=0.0, evaluation_trim=0.0,
                                    iterations=300, tolerance=1e-6):
        surface = np.asarray(surface, dtype=float)
        kernel = np.asarray(kernel, dtype=float)
        if surface.ndim != 2 or kernel.ndim != 2:
            raise ValueError("面形和抛光斑必须是二维矩阵")
        if not np.isfinite(surface).any() or not np.any(kernel > 0):
            raise ValueError("面形或抛光斑没有有效数据")
        hy = kernel.shape[0] // 2; hx = kernel.shape[1] // 2
        work = np.pad(surface, ((hy, hy), (hx, hx)), constant_values=np.nan)
        surface_mask = np.isfinite(work)
        trajectory_mask = cls.erode_mask(surface_mask, max(trajectory_trim, 0.0), dx, dy)
        evaluation_mask = cls.erode_mask(surface_mask, max(evaluation_trim, 0.0), dx, dy)
        if not trajectory_mask.any() or not evaluation_mask.any():
            raise ValueError("轨迹区或检测区为空，请减小裁边量")
        target = np.zeros_like(work)
        base = float(np.nanmin(work))
        target[surface_mask] = work[surface_mask] - base + float(uniform)
        lower = max(float(min_dwell), 0.0)
        upper = float(max_dwell)
        if upper < lower:
            raise ValueError("最大驻留时间不能小于最小驻留时间")
        dwell = np.zeros_like(work)
        dwell[trajectory_mask] = lower
        estimate = dwell.copy()
        momentum = 1.0
        lipschitz = max(float(np.sum(np.abs(kernel))) ** 2, 1e-12)
        objective = []
        for _ in range(max(1, int(iterations))):
            removal_est = cls.fft_convolve_same(estimate, kernel)
            error = np.zeros_like(work)
            error[evaluation_mask] = removal_est[evaluation_mask] - target[evaluation_mask]
            gradient = cls.fft_convolve_same(error, np.flip(kernel, axis=(0, 1)))
            candidate = estimate - gradient / lipschitz
            candidate[trajectory_mask] = np.clip(candidate[trajectory_mask], lower, upper)
            candidate[~trajectory_mask] = 0.0
            new_momentum = (1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum)) / 2.0
            accelerated = candidate + ((momentum - 1.0) / new_momentum) * (candidate - dwell)
            accelerated[~trajectory_mask] = 0.0
            # 不用 np.linalg.norm：其底层 BLAS 在部分环境会崩溃，逐元素范数更稳
            diff = candidate - dwell
            delta = float(np.sqrt(np.sum(diff * diff)))
            scale = max(float(np.sqrt(np.sum(dwell * dwell))), 1.0)
            dwell = candidate; estimate = accelerated; momentum = new_momentum
            if len(objective) < 20 or len(objective) % 10 == 0:
                residual_now = removal_est[evaluation_mask] - target[evaluation_mask]
                objective.append(float(np.sum(residual_now * residual_now)))
            if delta / scale < tolerance:
                break
        removal = cls.fft_convolve_same(dwell, kernel)
        residual = np.full_like(work, np.nan)
        residual[surface_mask] = work[surface_mask] - removal[surface_mask]
        before = work[evaluation_mask]
        after = residual[evaluation_mask]
        before_rms = float(np.sqrt(np.mean((before - np.mean(before)) ** 2)))
        after_rms = float(np.sqrt(np.mean((after - np.mean(after)) ** 2)))
        display_dwell = dwell.copy(); display_dwell[~trajectory_mask] = np.nan
        x = (np.arange(work.shape[1]) - (work.shape[1] - 1) / 2.0) * dx
        y = (np.arange(work.shape[0]) - (work.shape[0] - 1) / 2.0) * dy
        return {
            "dwell": display_dwell, "removal": removal, "residual": residual,
            "target": target, "surface_mask": surface_mask,
            "trajectory_mask": trajectory_mask, "evaluation_mask": evaluation_mask,
            "x": x, "y": y, "before_rms": before_rms, "after_rms": after_rms,
            "iterations": len(objective), "objective": objective,
        }
