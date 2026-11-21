"""曲面、口径与坐标系的数据定义，几何求值由 geometry 层实现。"""
from dataclasses import dataclass, field

@dataclass(frozen=True)
class SurfaceGeometry:
    id: str
    kind: str
    parameters: dict[str, float] = field(default_factory=dict)
    aperture: dict[str, object] = field(default_factory=dict)
    coordinate_system: str = "workpiece"
    version: str = "1"
