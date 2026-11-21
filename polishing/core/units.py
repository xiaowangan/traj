"""高度单位约定：检测与预测内部统一用 nm，换算必须显式、可追溯。"""
from __future__ import annotations

HEIGHT_UNIT_CHOICES = (
    ("raw", "按导入数据原数值（不换算）"), ("nm", "nm"), ("um", "μm"),
    ("mm", "mm"), ("waves", "Zygo 波长数（需波长）"),
)
UNKNOWN_UNIT_LABEL = "未知高度单位"
_TO_NM = {"nm": 1.0, "um": 1.0e3, "mm": 1.0e6}


def height_scale_to_nm(unit, wavelength_nm=None):
    """返回 (换算到 nm 的系数, 显示用单位名)。

    ``raw`` 表示用户尚未确认数据的物理单位：数值原样使用，单位名为“未知高度单位”，
    此时任何以 nm 书写的阈值都没有物理意义。
    """
    if unit == "raw":
        return 1.0, UNKNOWN_UNIT_LABEL
    if unit == "waves":
        if wavelength_nm is None or not float(wavelength_nm) > 0:
            raise ValueError("波长数需要大于 0 的测量波长 (nm)")
        return float(wavelength_nm), "nm"
    if unit in _TO_NM:
        return _TO_NM[unit], "nm"
    raise ValueError(f"未知高度单位：{unit}")


def describe_height_unit(unit, wavelength_nm=None):
    if unit == "waves":
        return f"波长数 × {float(wavelength_nm):g} nm"
    return {"raw": "导入数据原单位", "nm": "nm", "um": "μm", "mm": "mm"}[unit]
