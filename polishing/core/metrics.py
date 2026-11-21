"""Migrated numerical service: MetricsMixin; legacy numerical conventions retained."""
from __future__ import annotations
import numpy as np

class MetricsMixin:
    @staticmethod
    def stats(data):
        clean = np.asarray(data, dtype=float)
        clean = clean[np.isfinite(clean)]
        if clean.size == 0:
            return {"rms": float("nan"), "pv": float("nan"), "count": 0}
        centered = clean - np.mean(clean)
        return {
            "rms": float(np.sqrt(np.mean(centered * centered))),
            "pv": float(np.max(centered) - np.min(centered)),
            "count": int(clean.size),
        }

    @staticmethod
    def _largest_valid_rectangle(mask):
        """Return the largest all-valid axis-aligned rectangle as (y0,y1,x0,x1)."""
        rows, cols = mask.shape
        heights = np.zeros(cols, dtype=int)
        best_area = 0
        best = None
        for y in range(rows):
            heights = np.where(mask[y], heights + 1, 0)
            stack = []
            for x in range(cols + 1):
                h = int(heights[x]) if x < cols else 0
                start = x
                while stack and stack[-1][1] > h:
                    left, old_h = stack.pop()
                    area = old_h * (x - left)
                    if area > best_area:
                        best_area = area
                        best = (y - old_h + 1, y + 1, left, x)
                    start = left
                if not stack or stack[-1][1] < h:
                    stack.append((start, h))
        if best is None:
            raise ValueError("面形中没有足够大的连续有效矩形区域")
        return best

    @classmethod
    def evaluate_surface(cls, data, dx_mm, dy_mm, band_min, band_max,
                         detrend="plane", window="hann", roi=None):
        """Estimate a normalized two-sided 2-D PSD on the largest complete ROI.

        The complete rectangular ROI avoids zero-filling circular apertures and
        internal holes. PSD integrates to the window-weighted variance. ``roi``
        (y0, y1, x0, x1) pins the region so that before/after states are
        compared on exactly the same samples.
        """
        z = np.asarray(data, dtype=float)
        if z.ndim != 2 or min(z.shape) < 8:
            raise ValueError("面形至少需要 8×8 的二维网格")
        if not np.isfinite(dx_mm) or not np.isfinite(dy_mm) or dx_mm <= 0 or dy_mm <= 0:
            raise ValueError("X/Y 采样间距必须是正数，单位为 mm")
        valid = np.isfinite(z)
        if roi is None:
            y0, y1, x0, x1 = cls._largest_valid_rectangle(valid)
        else:
            y0, y1, x0, x1 = (int(value) for value in roi)
            if not (0 <= y0 < y1 <= z.shape[0] and 0 <= x0 < x1 <= z.shape[1]):
                raise ValueError("指定的检测区域超出面形网格")
            if not valid[y0:y1, x0:x1].all():
                raise ValueError("指定的检测区域包含无效点")
        roi = z[y0:y1, x0:x1].copy()
        ny, nx = roi.shape
        if min(ny, nx) < 8:
            raise ValueError(f"最大连续有效矩形仅 {nx}×{ny} 点，至少需要 8×8 点")

        nyquist = min(0.5 / dx_mm, 0.5 / dy_mm)
        min_resolvable = max(1.0 / (nx * dx_mm), 1.0 / (ny * dy_mm))
        if not np.isfinite(band_min) or not np.isfinite(band_max) or band_min < 0 or band_max <= band_min:
            raise ValueError("空间频段必须满足 0 ≤ 下限 < 上限，单位 mm⁻¹")
        if band_max > nyquist:
            raise ValueError(f"频段上限 {band_max:g} mm⁻¹ 超过保守 Nyquist 上限 {nyquist:g} mm⁻¹")

        yy, xx = np.indices(roi.shape, dtype=float)
        xx = (xx - (nx - 1) / 2.0) * dx_mm
        yy = (yy - (ny - 1) / 2.0) * dy_mm
        if detrend == "plane":
            design = np.column_stack((np.ones(roi.size), xx.ravel(), yy.ravel()))
            coefficients, *_ = np.linalg.lstsq(design, roi.ravel(), rcond=None)
            residual = roi - (design @ coefficients).reshape(roi.shape)
        elif detrend == "piston":
            residual = roi - float(np.mean(roi))
        elif detrend == "none":
            residual = roi.copy()
        else:
            raise ValueError("未知低阶去除方式")

        if window == "hann":
            win = np.outer(np.hanning(ny), np.hanning(nx))
        elif window == "none":
            win = np.ones_like(roi)
        else:
            raise ValueError("未知窗函数")
        win_power = float(np.mean(win * win))
        if win_power <= 0:
            raise ValueError("窗函数有效功率为零")
        spectrum = np.fft.fftshift(np.fft.fft2(residual * win))
        psd = (dx_mm * dy_mm / (nx * ny * win_power)) * np.abs(spectrum) ** 2
        qx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx_mm))
        qy = np.fft.fftshift(np.fft.fftfreq(ny, d=dy_mm))
        qxx, qyy = np.meshgrid(qx, qy)
        radial = np.hypot(qxx, qyy)
        dq = (1.0 / (nx * dx_mm)) * (1.0 / (ny * dy_mm))
        band_mask = (radial >= band_min) & (radial <= band_max)
        if not np.any(band_mask):
            raise ValueError("所选频段内没有 FFT 频点；请调整频段或扩大有效测量区域")
        band_rms = float(np.sqrt(np.sum(psd[band_mask]) * dq))
        low_mask = (radial > 0) & (radial < band_min)
        high_mask = (radial > band_max) & (radial <= nyquist)
        low_rms = float(np.sqrt(np.sum(psd[low_mask]) * dq))
        high_rms = float(np.sqrt(np.sum(psd[high_mask]) * dq))
        spectral_rms = float(np.sqrt(np.sum(psd[radial <= nyquist]) * dq))

        frequency_step = min(1.0 / (nx * dx_mm), 1.0 / (ny * dy_mm))
        edges = np.arange(0.0, nyquist + frequency_step, frequency_step)
        if len(edges) < 2:
            edges = np.array([0.0, nyquist])
        radial_frequency, radial_psd = [], []
        for low, high in zip(edges[:-1], edges[1:]):
            select = (radial >= low) & (radial < high)
            radial_frequency.append((low + high) / 2)
            radial_psd.append(float(np.mean(psd[select])) if np.any(select) else np.nan)

        angle = np.mod(np.arctan2(qyy, qxx), np.pi)
        sectors = 18
        sector_edges = np.linspace(0.0, np.pi, sectors + 1)
        direction_rms = []
        for index in range(sectors):
            selected = band_mask & (angle >= sector_edges[index]) & (angle < sector_edges[index + 1])
            direction_rms.append(float(np.sqrt(np.sum(psd[selected]) * dq)) if np.any(selected) else 0.0)
        peak_index = int(np.argmax(direction_rms))
        return {
            "roi": (int(y0), int(y1), int(x0), int(x1)),
            "roi_shape": (int(ny), int(nx)),
            "valid_fraction": float(np.count_nonzero(valid) / valid.size),
            "grid_shape": tuple(map(int, z.shape)),
            "dx_mm": float(dx_mm),
            "dy_mm": float(dy_mm),
            "nyquist_mm_inv": float(nyquist),
            "min_resolvable_mm_inv": float(min_resolvable),
            "band_below_resolution": bool(band_min < min_resolvable),
            "resolution_mm_inv": (float(1.0 / (nx * dx_mm)), float(1.0 / (ny * dy_mm))),
            "stats": cls.stats(roi),
            "detrended_stats": cls.stats(residual),
            "band_min_mm_inv": float(band_min),
            "band_max_mm_inv": float(band_max),
            "band_rms": band_rms,
            "low_frequency_rms": low_rms,
            "high_frequency_rms": high_rms,
            "spectral_rms": spectral_rms,
            "radial_frequency": np.asarray(radial_frequency),
            "radial_psd": np.asarray(radial_psd),
            "direction_centers_deg": np.degrees((sector_edges[:-1] + sector_edges[1:]) / 2),
            "direction_rms": np.asarray(direction_rms),
            "dominant_direction_deg": float((peak_index + 0.5) * 180 / sectors),
            "psd": psd,
            "qx": qx,
            "qy": qy,
            "detrend": detrend,
            "window": window,
            "psd_unit": "height_unit²·mm²",
        }
