"""工件、表面状态、工艺方案与加工执行的数据定义草案。"""
from dataclasses import dataclass, field
from enum import Enum

class ProcessType(str, Enum):
    SCANNING = "scanning"
    CONFORMAL_VIBRATION = "conformal_vibration"

class WorkflowStage(str, Enum):
    IMPORT = "import"
    EVALUATE = "evaluate"
    PLAN = "plan"
    PREDICT = "predict"
    EXECUTE = "execute"
    REMEASURE = "remeasure"
    VALIDATE = "validate"
    ARCHIVE = "archive"

@dataclass(frozen=True)
class Workpiece:
    id: str
    material: str
    geometry_id: str

@dataclass(frozen=True)
class SurfaceState:
    id: str
    workpiece_id: str
    previous_state_id: str | None = None
    produced_by_run_id: str | None = None
    measurement_ids: tuple[str, ...] = ()

@dataclass(frozen=True)
class ProcessPlan:
    id: str
    version: str
    process_type: ProcessType
    initial_state_id: str
    tool_state_id: str
    parameters: dict[str, object] = field(default_factory=dict)
    model_version_id: str | None = None

@dataclass(frozen=True)
class ProcessRun:
    id: str
    plan_version_id: str
    initial_state_id: str
    final_state_id: str | None = None
    actual_parameters: dict[str, object] = field(default_factory=dict)
    anomalies: tuple[str, ...] = ()

@dataclass(frozen=True)
class Project:
    id: str
    name: str
    workpiece_ids: tuple[str, ...] = ()
    run_ids: tuple[str, ...] = ()

@dataclass(frozen=True)
class ProcessCase:
    """One traceable pass through the eight-stage processing workflow."""
    id: str
    project_id: str
    workpiece_id: str
    initial_state_id: str | None = None
    plan_version_id: str | None = None
    prediction_id: str | None = None
    process_run_id: str | None = None
    final_state_id: str | None = None
    validation_id: str | None = None
    archived: bool = False

@dataclass(frozen=True)
class WorkflowSnapshot:
    case_id: str
    active_stage: WorkflowStage
    revision: int
    completed_stages: tuple[WorkflowStage, ...] = ()
    invalidated_products: tuple[str, ...] = ()
