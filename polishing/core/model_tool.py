"""工具状态和去除函数的数据定义草案。"""
from dataclasses import dataclass, field

@dataclass(frozen=True)
class ToolState:
    id: str
    tool_id: str
    material: str
    diameter_mm: float
    condition: str
    recorded_parameters: dict[str, object] = field(default_factory=dict)

@dataclass(frozen=True)
class RemovalFunction:
    id: str
    tool_state_id: str
    kernel: object
    dx_mm: float
    dy_mm: float
    duration_s: float
    calibration_id: str
    kernel_origin: tuple[int, int]
    rate_unit: str = "nm/s"
    applicability: str = "unvalidated"
