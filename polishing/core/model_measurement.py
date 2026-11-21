"""测量和检测契约草案；数组校验与配准尚待实现。"""
from dataclasses import dataclass, field

@dataclass(frozen=True)
class SurfaceMap:
    id: str
    z: object
    x: object
    y: object
    valid_mask: object
    aperture_mask: object
    source_id: str
    height_unit: str = "nm"
    coordinate_unit: str = "mm"
    coordinate_system: str = "workpiece"
    processing_history: tuple[str, ...] = ()
    registration_transform: object = None

@dataclass(frozen=True)
class EvaluationProtocol:
    id: str
    version: str
    msf_min_mm_inv: float
    msf_max_mm_inv: float
    detrending: str
    window: str
    psd_definition: str
    missing_data_policy: str
    evaluation_region_id: str

@dataclass(frozen=True)
class Prediction:
    id: str
    initial_state_id: str
    plan_version_id: str
    model_version_id: str
    protocol_id: str
    metrics: dict[str, float] = field(default_factory=dict)
    applicability: str = "unvalidated"
    status: str = "draft"
