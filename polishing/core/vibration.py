"""共形振动中频演化预测骨架：手动参数、未标定，接口按实验辨识的需要预留。

采用 docs/第一性原理展开.md §11.3 的谱演化形式

    P_t(q, θ) = exp[-2β(q, θ)·t] · P_0(q, θ) + P_new(t)

β 的函数形式（幂律 × 方向各向异性）与 P_new（白噪声线性累积）都是**占位假设**，不是已验证的物理结论；
等实验数据到位后用 ``fit_beta`` 辨识，再替换这两处函数形式。在此之前所有预测都标记为“未标定、外推”。
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import numpy as np

from polishing.core.assessment import check_target, verdict_of

DIRECTION_SECTORS = 18


@dataclass(frozen=True)
class VibrationModel:
    beta0: float                    # 参考空间频率处的平滑速率 (1/s)
    q_ref: float = 0.1              # 参考空间频率 (周期/mm)
    exponent: float = 0.0           # β ∝ (q/q_ref)^exponent
    anisotropy: float = 0.0         # 0–1：垂直于振动方向的波纹平滑变弱的程度
    direction_deg: float = 0.0      # 振动方向（空间频率矢量角，0°=X）
    p_new_rate: float = 0.0         # 工艺引入的新误差，PSD 单位/s（白噪声线性累积）
    calibration: str = "uncalibrated"

    def validate(self):
        if not self.beta0 >= 0:
            raise ValueError("平滑速率 β0 不能为负")
        if not self.q_ref > 0:
            raise ValueError("参考空间频率必须大于 0")
        if not 0.0 <= self.anisotropy <= 1.0:
            raise ValueError("各向异性必须在 0–1 之间")
        if not self.p_new_rate >= 0:
            raise ValueError("新误差率不能为负")
        return self

    def beta(self, q, theta_rad):
        q = np.asarray(q, dtype=float)
        radial = np.where(q > 0, (np.maximum(q, 1e-300) / self.q_ref) ** self.exponent, 0.0)
        angle = np.sin(np.asarray(theta_rad, dtype=float) - np.radians(self.direction_deg))
        return self.beta0 * radial * (1.0 - self.anisotropy * angle ** 2)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**{key: data[key] for key in cls.__dataclass_fields__ if key in data})


def _frequency_grid(qx, qy):
    qxx, qyy = np.meshgrid(qx, qy)
    return np.hypot(qxx, qyy), np.arctan2(qyy, qxx)


def evolve_psd(psd, qx, qy, model, t):
    """把加工前 PSD 演化 t 秒；直流分量不参与。"""
    radial, theta = _frequency_grid(qx, qy)
    decay = np.exp(-2.0 * model.beta(radial, theta) * float(t))
    evolved = decay * np.asarray(psd, dtype=float) + model.p_new_rate * float(t)
    evolved[radial == 0] = psd[radial == 0]
    return evolved


def psd_metrics(psd, qx, qy, band_min, band_max, nyquist):
    """与 evaluate_surface 同一分带约定的 PSD 指标（低/中/高频 RMS、方向 RMS、径向谱）。"""
    radial, theta = _frequency_grid(qx, qy)
    dq = float(qx[1] - qx[0]) * float(qy[1] - qy[0])
    band = (radial >= band_min) & (radial <= band_max)
    low = (radial > 0) & (radial < band_min)
    high = (radial > band_max) & (radial <= nyquist)

    def rms(mask):
        return float(np.sqrt(np.sum(psd[mask]) * dq))

    angle = np.mod(theta, np.pi)
    edges = np.linspace(0.0, np.pi, DIRECTION_SECTORS + 1)
    direction = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sector = band & (angle >= lo) & (angle < hi)
        direction.append(rms(sector) if sector.any() else 0.0)
    step = min(float(qx[1] - qx[0]), float(qy[1] - qy[0]))
    bins = np.arange(0.0, nyquist + step, step)
    centers, radial_psd = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        select = (radial >= lo) & (radial < hi)
        centers.append(float((lo + hi) / 2))
        radial_psd.append(float(np.mean(psd[select])) if np.any(select) else None)
    return {
        "pv": None,
        "rms": rms(radial <= nyquist),
        "low_rms": rms(low), "msf_rms": rms(band), "high_rms": rms(high),
        "dominant_direction_deg": float((int(np.argmax(direction)) + 0.5) * 180 / DIRECTION_SECTORS),
        "band_below_resolution": False,
        "radial_frequency": centers, "radial_psd": radial_psd,
        "direction_centers_deg": [float(v) for v in np.degrees((edges[:-1] + edges[1:]) / 2)],
        "direction_rms": direction,
    }


def predict_evolution(evaluation, model, duration_s, target=None, steps=121):
    """由加工前面形检测结果预测中频 RMS 随时间的演化，并给出最佳时长和达标时间。"""
    model.validate()
    if not duration_s > 0:
        raise ValueError("计划加工时长必须大于 0")
    psd, qx, qy = evaluation["psd"], evaluation["qx"], evaluation["qy"]
    band_min, band_max = evaluation["band_min_mm_inv"], evaluation["band_max_mm_inv"]
    nyquist = evaluation["nyquist_mm_inv"]
    times = np.linspace(0.0, float(duration_s), int(steps))
    curve = []
    for t in times:
        metrics = psd_metrics(evolve_psd(psd, qx, qy, model, t), qx, qy, band_min, band_max, nyquist)
        curve.append((metrics["msf_rms"], metrics["low_rms"], metrics["high_rms"]))
    curve = np.asarray(curve)
    before = psd_metrics(psd, qx, qy, band_min, band_max, nyquist)
    final = psd_metrics(evolve_psd(psd, qx, qy, model, duration_s), qx, qy, band_min, band_max, nyquist)
    # PSD 含窗的全谱 RMS 与扫描分支的去低阶 RMS 定义不同，放进同一张表会误导；
    # 共形振动骨架只给出分带指标，总 RMS 和 PV 不预测。
    before["rms"] = final["rms"] = None
    best = int(np.argmin(curve[:, 0]))
    limit = None if target is None else target.msf_rms_max
    reach = _first_crossing(times, curve[:, 0], limit) if limit is not None else None

    checks = check_target(final, target)
    if target is not None and target.time_max_s is not None:
        checks.append({"name": "计划加工时长 (s)", "value": float(duration_s),
                       "limit": target.time_max_s, "ok": float(duration_s) <= target.time_max_s})
    flags = ["手动参数、未经实验标定：预测只用于流程演示和接口联调，不是已验证工艺（外推）。"]
    if evaluation["height_unit"] == "未知高度单位":
        flags.append("高度单位未确认：指标和目标都按导入数据的原数值比较，不是 nm。")
    if target is None or target.is_empty():
        flags.append("尚未设定加工目标，只比较、不判定是否达标。")
    elif target.rms_max is not None:
        flags.append("共形振动骨架只预测分带（低/中/高频）RMS，“去低阶 RMS 上限”这一项未判定。")
    if 0 < best < len(times) - 1 or (best == 0 and curve[-1, 0] > curve[0, 0]):
        flags.append(f"新误差使中频 RMS 在 t≈{times[best]:.0f} s 后回升，继续加工反而变差。")
    if limit is not None and reach is None:
        flags.append("在计划时长内中频 RMS 未达到目标。")
    return {
        "kind": "conformal_vibration",
        "status": "predicted",
        "calibration": model.calibration,
        "extrapolated": True,
        "model": model.to_dict(),
        "duration_s": float(duration_s),
        "unit": evaluation["height_unit"],
        "protocol": {"band_min": float(band_min), "band_max": float(band_max)},
        "target": target.to_dict() if target else None,
        "times_s": [float(v) for v in times],
        "msf_rms_curve": [float(v) for v in curve[:, 0]],
        "states": {"before": before, "predicted": final},
        "best_time_s": float(times[best]), "best_msf_rms": float(curve[best, 0]),
        "time_to_target_s": reach,
        "checks": checks, "verdict": verdict_of(checks), "flags": flags,
    }


def _first_crossing(times, values, limit):
    """第一次降到 limit 以下的时间（线性插值）；始终高于 limit 则返回 None。"""
    below = np.nonzero(values <= limit)[0]
    if below.size == 0:
        return None
    index = int(below[0])
    if index == 0:
        return 0.0
    t0, t1, v0, v1 = times[index - 1], times[index], values[index - 1], values[index]
    return float(t0 + (v0 - limit) / (v0 - v1) * (t1 - t0)) if v0 != v1 else float(t1)


def fit_beta(times, amplitudes, a0=None, noise_floor=0.0):
    """由单一波纹的幅值（或中频 RMS）随时间的衰减辨识 β。

    拟合 ln(a/a0) = -β·t（过原点最小二乘）。低于噪声底、非正或 t=0 的点不参与；
    ``a0`` 缺省取 t=0 的测量值。返回 β、标准误差、拟合点数和 R²。
    """
    t = np.asarray(times, dtype=float)
    a = np.asarray(amplitudes, dtype=float)
    if t.shape != a.shape:
        raise ValueError("时间与幅值数量必须一致")
    if a0 is None:
        zero = np.nonzero(t == 0)[0]
        if zero.size == 0:
            raise ValueError("需要 t=0 的初始幅值，或显式给出 a0")
        a0 = float(a[zero[0]])
    if not a0 > 0:
        raise ValueError("初始幅值必须大于 0")
    use = np.isfinite(t) & np.isfinite(a) & (t > 0) & (a > max(float(noise_floor), 0.0)) & (a > 0)
    if int(np.count_nonzero(use)) < 1:
        raise ValueError("没有高于噪声底的有效数据点，无法辨识 β")
    t, y = t[use], np.log(a[use] / a0)
    beta = float(-np.sum(t * y) / np.sum(t * t))
    residual = y + beta * t
    n = int(t.size)
    stderr = float(np.sqrt(np.sum(residual ** 2) / (n - 1) / np.sum(t * t))) if n > 1 else None
    spread = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - np.sum(residual ** 2) / spread) if n > 1 and spread > 0 else None
    return {"beta": beta, "stderr": stderr, "n_points": n, "r2": r2}
