"""同一检测标准下的前/预测后/实测后比较、验收判定与重复测量噪声底（纯数值，不依赖 Qt）。"""
from __future__ import annotations
import numpy as np

from polishing.core.alignment import region_areas
from polishing.core.metrics import MetricsMixin
from polishing.core.units import UNKNOWN_UNIT_LABEL

STATE_LABELS = (("before", "加工前"), ("predicted", "预测后"), ("measured", "实测后"))
SCANNING_MODEL_NOTE = ("扫描驻留预测只是固定 TIF 的卷积模型，尚未经实验验证；"
                       "其中频改善不能当作共形振动的中频预测结论。")


def evaluate_with_protocol(data, dx_mm, dy_mm, protocol, roi=None):
    """按检测标准计算 PSD 与分带指标；高度先换算到 nm（未确认单位时保持原值）。"""
    protocol.validate()
    scale, unit = protocol.scale_to_nm()
    result = MetricsMixin.evaluate_surface(
        np.asarray(data, dtype=float) * scale, dx_mm, dy_mm,
        protocol.band_min, protocol.band_max,
        detrend=protocol.detrend, window=protocol.window, roi=roi)
    result["height_unit"] = unit
    result["height_unit_input"] = protocol.unit_text
    return result


def common_roi(*arrays):
    """几个面形都有效的最大连续矩形，用来保证前后状态在同一批采样点上比较。"""
    valid = np.ones(np.shape(arrays[0]), dtype=bool)
    for array in arrays:
        valid &= np.isfinite(array)
    return MetricsMixin._largest_valid_rectangle(valid)


def summarize(result):
    """把完整检测结果压缩成可保存、可比较的指标（PV/RMS 均为按协议去低阶后）。"""
    return {
        "pv": float(result["detrended_stats"]["pv"]),
        "rms": float(result["detrended_stats"]["rms"]),
        "low_rms": float(result["low_frequency_rms"]),
        "msf_rms": float(result["band_rms"]),
        "high_rms": float(result["high_frequency_rms"]),
        "dominant_direction_deg": float(result["dominant_direction_deg"]),
        "band_below_resolution": bool(result["band_below_resolution"]),
        "roi_shape": list(result["roi_shape"]),
        "radial_frequency": [float(v) for v in result["radial_frequency"]],
        "radial_psd": [None if not np.isfinite(v) else float(v) for v in result["radial_psd"]],
        "direction_centers_deg": [float(v) for v in result["direction_centers_deg"]],
        "direction_rms": [float(v) for v in result["direction_rms"]],
    }


def check_target(metrics, target, dwell_total_s=None):
    """逐项对照加工目标；返回 [{name, value, limit, ok}]，未设限的项不出现。"""
    checks = []
    if target is None:
        return checks
    if target.msf_rms_max is not None:
        checks.append({"name": "中频 RMS", "value": metrics["msf_rms"],
                       "limit": target.msf_rms_max, "ok": metrics["msf_rms"] <= target.msf_rms_max})
    if target.rms_max is not None and metrics.get("rms") is not None:
        checks.append({"name": "去低阶 RMS", "value": metrics["rms"],
                       "limit": target.rms_max, "ok": metrics["rms"] <= target.rms_max})
    if target.time_max_s is not None and dwell_total_s is not None:
        checks.append({"name": "累计驻留时间 (s)", "value": dwell_total_s,
                       "limit": target.time_max_s, "ok": dwell_total_s <= target.time_max_s})
    return checks


def verdict_of(checks):
    if not checks:
        return "no_target"
    return "pass" if all(item["ok"] for item in checks) else "fail"


def protocol_flags(protocol, target, before_metrics):
    flags = []
    if protocol.height_unit == "raw":
        flags.append("高度单位未确认：指标和目标都按导入数据的原数值比较，不是 nm。")
    if before_metrics.get("band_below_resolution"):
        flags.append("中频下限低于检测区域的最低可分辨频率，低端频段结论不可靠。")
    if target is None or target.is_empty():
        flags.append("尚未设定加工目标，只比较、不判定是否达标。")
    return flags


def assess_solution(solution, protocol, dx_mm, dy_mm, target=None, measured_after=None):
    """用检测标准重新检测驻留预测：加工前、预测加工后（以及可选的实测加工后）。

    三个状态取同一检测区、同一 ROI、同一协议，因此指标之差只来自加工本身。
    ``measured_after`` 是与求解网格同形状、已配准的加工后面形。
    """
    evaluation_mask = np.asarray(solution["evaluation_mask"], dtype=bool)
    residual = np.asarray(solution["residual"], dtype=float)
    work = residual + np.asarray(solution["removal"], dtype=float)
    arrays = {"before": np.where(evaluation_mask, work, np.nan),
              "predicted": np.where(evaluation_mask, residual, np.nan)}
    if measured_after is not None:
        measured = np.asarray(measured_after, dtype=float)
        if measured.shape != residual.shape:
            raise ValueError("实测加工后面形必须与求解网格形状一致（需先配准）")
        arrays["measured"] = np.where(evaluation_mask, measured, np.nan)
    roi = common_roi(*arrays.values())
    results = {name: evaluate_with_protocol(data, dx_mm, dy_mm, protocol, roi=roi)
               for name, data in arrays.items()}
    states = {name: summarize(result) for name, result in results.items()}
    dwell_total = float(np.nansum(solution["dwell"]))
    basis = "measured" if "measured" in states else "predicted"
    checks = check_target(states[basis], target, dwell_total)
    return {
        "kind": "scanning",
        "status": "measured" if basis == "measured" else "predicted",
        "protocol": protocol.to_dict(),
        "target": target.to_dict() if target else None,
        "unit": results["before"]["height_unit"],
        "grid_pitch_mm": [float(dx_mm), float(dy_mm)],
        "roi": [int(v) for v in roi],
        "states": states,
        "dwell_total_s": dwell_total,
        "regions_mm2": region_areas(solution, dx_mm, dy_mm),
        "checks": checks,
        "verdict": verdict_of(checks),
        "flags": protocol_flags(protocol, target, states["before"]) + [SCANNING_MODEL_NOTE],
    }


def noise_floor(repeat_a, repeat_b, dx_mm, dy_mm, protocol):
    """由同一表面的两次重复测量估计噪声底。

    差值的方差是单次噪声的两倍，所以先除以 √2。结果包含重复装夹带来的变化，
    只有在两次测量之间表面没有被加工时才可当作仪器噪声底。
    """
    a = np.asarray(repeat_a, dtype=float)
    b = np.asarray(repeat_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("两次重复测量必须在同一网格上")
    roi = common_roi(a, b)
    result = evaluate_with_protocol((a - b) / np.sqrt(2.0), dx_mm, dy_mm, protocol, roi=roi)
    return {"msf_rms": float(result["band_rms"]), "rms": float(result["detrended_stats"]["rms"]),
            "low_rms": float(result["low_frequency_rms"]),
            "high_rms": float(result["high_frequency_rms"]), "roi": [int(v) for v in roi],
            "unit": result["height_unit"]}


def format_comparison(assessment):
    """把比较结果排成文字表，界面和报告共用。"""
    unit = assessment.get("unit", UNKNOWN_UNIT_LABEL)
    states = assessment["states"]
    columns = [(key, label) for key, label in STATE_LABELS if states.get(key)]
    band = assessment["protocol"]
    rows = (("PV", "pv"), ("RMS", "rms"), ("低频 RMS", "low_rms"),
            (f"中频 RMS（{band['band_min']:g}–{band['band_max']:g} 周期/mm）", "msf_rms"),
            ("高频 RMS", "high_rms"))
    lines = ["指标（去低阶后，单位 " + unit + "）｜" + "｜".join(label for _, label in columns)
             + ("｜预测变化" if "predicted" in states else "")]
    for caption, key in rows:
        values = [states[name].get(key) for name, _ in columns]
        text = ["—" if value is None else f"{value:.4g}" for value in values]
        change = ""
        before = states["before"].get(key)
        predicted = states.get("predicted", {}).get(key)
        if before not in (None, 0) and predicted is not None:
            change = f"｜{(predicted - before) / before:+.1%}"
        lines.append(f"{caption}｜" + "｜".join(text) + change)
    verdict = {"pass": "达标", "fail": "未达标", "no_target": "未设目标"}[assessment["verdict"]]
    lines.append(f"判定（依据{'实测' if assessment['status'] == 'measured' else '预测'}）：{verdict}")
    for item in assessment["checks"]:
        lines.append(f"  {'✔' if item['ok'] else '✘'} {item['name']}：{item['value']:.4g}（目标 ≤ {item['limit']:g}）")
    lines.extend("⚠ " + flag for flag in assessment["flags"])
    return "\n".join(lines)


def short_summary(assessment):
    """给窄面板用的几行摘要。"""
    unit = assessment.get("unit", UNKNOWN_UNIT_LABEL)
    before = assessment["states"]["before"]
    after = assessment["states"].get("measured") or assessment["states"]["predicted"]
    verdict = {"pass": "达标", "fail": "未达标", "no_target": "未设加工目标"}[assessment["verdict"]]
    label = "实测后" if assessment["status"] == "measured" else "预测后"
    lines = [f"中频 RMS：{before['msf_rms']:.4g} → {after['msf_rms']:.4g} {unit}（{label}）；判定：{verdict}。"]
    if assessment["kind"] == "conformal_vibration":
        reach = assessment.get("time_to_target_s")
        lines.append(f"最佳时长约 {assessment['best_time_s']:.0f} s；"
                     + (f"预计 {reach:.0f} s 达到目标。" if reach is not None else "计划时长内未达到目标。"))
    lines.extend("⚠ " + flag for flag in assessment["flags"])
    return "\n".join(lines)
