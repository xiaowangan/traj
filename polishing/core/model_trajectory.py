"""分段轨迹与驻留结果；单位为 mm、s、mm/s、rad。"""
from dataclasses import dataclass, field
from enum import Enum

class SegmentType(str, Enum):
    PROCESSING = "processing"
    TRANSFER = "transfer"
    APPROACH = "approach"
    RETRACT = "retract"

@dataclass(frozen=True)
class TrajectorySegment:
    kind: SegmentType
    xyz: object
    normals: object
    feed_mm_s: object = None
    sampling_weights_mm2: object = None

@dataclass(frozen=True)
class Trajectory:
    id: str
    segments: tuple[TrajectorySegment, ...]
    coordinate_system: str = "workpiece"
    input_fingerprint: str = ""

@dataclass(frozen=True)
class DwellSolution:
    id: str
    dwell_s: object
    removal_nm: object
    residual_nm: object
    masks: dict[str, object] = field(default_factory=dict)
    solver_config: dict[str, object] = field(default_factory=dict)
    input_fingerprint: str = ""
    status: str = "unvalidated"
