"""Shared case state and dependency invalidation for the eight-stage workflow."""
from __future__ import annotations

from dataclasses import dataclass
from polishing.core.model_project import WorkflowStage

WORKFLOW_STAGES = tuple(stage.value for stage in WorkflowStage)
STAGE_TITLES = {
    "import": "导入工件", "evaluate": "面形初测", "plan": "制定方案", "predict": "效果预估",
    "execute": "生成程序", "remeasure": "加工复测", "validate": "结果对比", "archive": "结果存档",
}
# 每个阶段产出的状态键：用于判断“已失效”
STAGE_PRODUCTS = {
    "import": {"model", "surface"}, "evaluate": {"evaluation"}, "plan": {"process_plan"},
    "predict": {"frozen_plan", "solution", "assessment"}, "execute": {"trajectory", "cnc"},
}
_DOWNSTREAM_RESULTS = ("solution", "trajectory", "cnc", "assessment", "frozen_plan", "prediction",
                       "validation", "archive_record")
_DEPENDENTS = {
    "dx": ("model", "surface", "spot", "evaluation", "alignment", "process_plan") + _DOWNSTREAM_RESULTS,
    "dy": ("model", "surface", "spot", "evaluation", "alignment", "process_plan") + _DOWNSTREAM_RESULTS,
    "model": ("evaluation", "alignment", "coverage_preview", "disk_fit", "process_plan") + _DOWNSTREAM_RESULTS,
    "surface_raw": ("evaluation", "alignment", "process_plan") + _DOWNSTREAM_RESULTS,
    "surface": ("evaluation", "alignment", "process_plan") + _DOWNSTREAM_RESULTS,
    "evaluation": ("process_plan",) + _DOWNSTREAM_RESULTS,
    "protocol": ("assessment", "frozen_plan", "validation", "archive_record"),
    "target": ("assessment", "frozen_plan", "validation", "archive_record"),
    "spot_raw": ("spot", "process_plan") + _DOWNSTREAM_RESULTS,
    "spot": ("process_plan",) + _DOWNSTREAM_RESULTS,
    "process_records": ("process_plan",) + _DOWNSTREAM_RESULTS,
    "process_plan": _DOWNSTREAM_RESULTS,
    "coverage_preview": ("process_plan",),
    "solution": ("trajectory", "cnc", "assessment", "frozen_plan", "prediction", "validation", "archive_record"),
    "trajectory": ("cnc", "validation", "archive_record"),
    "cnc": ("validation", "archive_record"),
    "assessment": ("frozen_plan", "validation", "archive_record"),
    "frozen_plan": ("trajectory", "cnc", "validation", "archive_record"),
    "post_surface": ("validation", "archive_record"),
    "validation": ("archive_record",),
}


class WorkflowState(dict):
    """Compatibility dictionary that invalidates stale downstream products."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.revision = 0
        self.invalidated: set[str] = set()

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.invalidated.discard(key)
        self.revision += 1
        # 依赖关系是传递的：方案解冻后，由它生成的轨迹和 CNC 也一并失效
        pending, seen = list(_DEPENDENTS.get(key, ())), set()
        while pending:
            dependent = pending.pop(0)
            if dependent in seen:
                continue
            seen.add(dependent)
            if super().get(dependent) is not None:
                super().__setitem__(dependent, None)
                self.invalidated.add(dependent)
                pending.extend(_DEPENDENTS.get(dependent, ()))

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def mark_current(self, key):
        self.invalidated.discard(key)


def create_dwell_state():
    return WorkflowState({
        "dx": 1.0, "dy": 1.0, "model": None, "surface_raw": None,
        "surface": None, "surface_pixel": None, "spot_raw": None,
        "spot_pixel": None, "spot_before": None, "spot": None,
        "solution": None, "trajectory": None, "cnc": None, "disk_fit": None,
        "evaluation": None, "prediction": None, "execution": None,
        "process_records": None, "process_plan": None,
        "post_surface": None, "validation": None, "archive_record": None,
        "coverage_preview": None, "surface_meta": None, "protocol": None,
        "target": None, "alignment": None, "assessment": None, "frozen_plan": None,
    })


@dataclass
class WorkflowContext:
    state: WorkflowState
    active_stage: str = "import"

    def set_stage(self, stage: str):
        if stage not in WORKFLOW_STAGES:
            raise ValueError(f"未知工艺阶段：{stage}")
        self.active_stage = stage

    def stage_status(self, stage: str) -> str:
        s = self.state
        completed = {
            "import": s.get("model") is not None and s.get("surface") is not None,
            "evaluate": s.get("evaluation") is not None,
            "plan": s.get("process_plan") is not None,
            # 效果预估阶段以“用户确认并冻结方案”为完成，而不是算出一个结果就算完成
            "predict": s.get("frozen_plan") is not None,
            "execute": s.get("cnc") is not None or s.get("execution") is not None,
            "remeasure": s.get("post_surface") is not None,
            "validate": s.get("validation") is not None,
            "archive": s.get("archive_record") is not None,
        }[stage]
        return "已完成" if completed else "待完成"

    def stage_is_stale(self, stage: str) -> bool:
        """本阶段的结果曾经有过、但因上游输入改动被清掉了（需要重做）。"""
        return self.stage_status(stage) != "已完成" and bool(
            STAGE_PRODUCTS.get(stage, set()) & self.state.invalidated)
