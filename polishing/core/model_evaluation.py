"""检测标准与加工目标的数据定义（不含计算）。"""
from __future__ import annotations
from dataclasses import asdict, dataclass

from polishing.core.units import describe_height_unit, height_scale_to_nm

_DETRENDS = ("plane", "piston", "none")
_WINDOWS = ("hann", "none")


@dataclass(frozen=True)
class EvaluationProtocol:
    """加工前后必须共用的同一套检测规则；任何字段变化都应视为新协议版本。"""
    band_min: float
    band_max: float
    detrend: str = "plane"
    window: str = "hann"
    height_unit: str = "raw"
    wavelength_nm: float | None = None
    version: str = "1"

    def validate(self):
        if not self.band_min >= 0 or not self.band_max > self.band_min:
            raise ValueError("空间频段必须满足 0 ≤ 下限 < 上限，单位 mm⁻¹")
        if self.detrend not in _DETRENDS:
            raise ValueError("未知低阶去除方式")
        if self.window not in _WINDOWS:
            raise ValueError("未知窗函数")
        self.scale_to_nm()
        return self

    def scale_to_nm(self):
        return height_scale_to_nm(self.height_unit, self.wavelength_nm)

    @property
    def unit_text(self):
        return describe_height_unit(self.height_unit, self.wavelength_nm)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**{key: data[key] for key in cls.__dataclass_fields__ if key in data})


@dataclass(frozen=True)
class AcceptanceTarget:
    """本次加工的验收目标；数值与检测标准使用相同的高度单位，空值表示不设限。"""
    msf_rms_max: float | None = None
    rms_max: float | None = None
    time_max_s: float | None = None

    def is_empty(self):
        return self.msf_rms_max is None and self.rms_max is None and self.time_max_s is None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**{key: data.get(key) for key in cls.__dataclass_fields__})
