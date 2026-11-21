"""由案例状态驱动的检测与对齐检查；界面和案例恢复共用，避免两处各算一遍。"""
from __future__ import annotations

from polishing.core.alignment import check_alignment
from polishing.core.assessment import evaluate_with_protocol


def select_evaluation_source(state):
    """有原始测量网格和采样间距时优先使用它们，否则用案例计算网格。"""
    measurement = state.get("surface_raw")
    pitch = state.get("surface_pixel")
    if measurement is not None and pitch is not None:
        return measurement, float(pitch), float(pitch), "原始测量网格"
    return (state.get("surface"), float(state.get("dx", 1.0)), float(state.get("dy", 1.0)),
            "案例计算网格（文件未提供原始采样间距）")


def evaluate_current_surface(state, protocol):
    surface, dx, dy, source = select_evaluation_source(state)
    if surface is None:
        raise ValueError("请先导入加工前面形")
    result = evaluate_with_protocol(surface, dx, dy, protocol)
    result["protocol"] = {
        **protocol.to_dict(),
        "band_min_mm_inv": protocol.band_min, "band_max_mm_inv": protocol.band_max,
        "height_unit_input": protocol.unit_text,
        "sample_pitch_mm": (dx, dy), "evaluation_source": source,
    }
    result["evaluation_source"] = source
    return result


def check_current_alignment(state):
    """期望曲面与案例网格上的实测面形都已就绪时才检查，否则返回 None。"""
    model, surface = state.get("model"), state.get("surface")
    if model is None or surface is None:
        return None
    return check_alignment(model, surface, float(state.get("dx", 1.0)), float(state.get("dy", 1.0)))
