"""工具与设备：资源页维护工具参数，制定方案页选用工具并对当前期望曲面做贴合检查。

按“每个参数只在一处填写”的原则：
- 抛光盘口径、材料、转速、压力、进给属于工具/设备本身 → 工艺资源 ·「工具与设备」页；
- 本次用哪种工具、哪个抛光阶段 → 制定方案第 2 步；
- 贴合检查需要的 R/K/A 与工件口径直接取导入工件生成的期望曲面，不再手填。
"""
from __future__ import annotations
import math
import numpy as np
from PyQt5.QtWidgets import (QComboBox, QGroupBox, QHBoxLayout, QLabel, QPushButton,
                             QVBoxLayout, QWidget)
from polishing.ui.widgets.inputs import combox_input
from polishing.workflow.engine import DwellTimeEngine

_NOTE = "color:#52657a; font-size:11px;"


class ToolResourceMixin:
    # ── 工艺资源：工具与设备参数 ─────────────────────────────────────
    def _build_dwell_initial_page(self):
        scroll, layout = self._dwell_scroll_page(
            "工艺资源 · 工具与设备", "维护抛光盘与设备参数，供各个加工案例选用；贴合检查在制定方案第 2 步进行。")
        disk_group = QGroupBox("抛光盘"); disk_form = QVBoxLayout(disk_group)
        self._dwell_add_fields(disk_form, [
            ("dw_disk_diameter", "抛光盘口径 (mm)：", ""),
            ("dw_disk_material", "抛光盘材料：", "聚氨酯"),
        ])
        layout.addWidget(disk_group)

        device_group = QGroupBox("设备工况"); device_form = QVBoxLayout(device_group)
        self._dwell_add_fields(device_form, [
            ("dw_disk_speed", "抛光盘转速 (rpm)：", ""),
            ("dw_pressure", "抛光压力：", ""),
            ("dw_feed", "进给速度 (mm/min)：", ""),
        ])
        suggest = QPushButton("自动推荐压力和进给")
        suggest.clicked.connect(self._dwell_suggest_process)
        device_form.addWidget(suggest)
        note = QLabel("推荐值按抛光盘口径和转速估算，压力单位需按设备确认；推荐值可手动修改。"
                      "这些参数随方案一起记录，不参与驻留求解，也不替代生成程序阶段的限速。")
        note.setWordWrap(True); note.setStyleSheet(_NOTE)
        device_form.addWidget(note)
        layout.addWidget(device_group)
        self.dw_initial_info = QLabel(""); self.dw_initial_info.setWordWrap(True)
        layout.addWidget(self.dw_initial_info)
        # 上一步 / 下一步只在从制定方案进入时显示；从“工艺资源”单独打开时隐藏
        flow_bar = QWidget()
        flow_row = QHBoxLayout(flow_bar)
        flow_row.setContentsMargins(0, 0, 0, 0)
        flow_row.addStretch()
        self._dwell_add_next_button(
            flow_row, "dw_tool_back", "上一步：返回制定方案", "idx_workflow_plan",
            tooltip="回到制定方案，继续做贴合检查。")
        self._dwell_add_next_button(
            flow_row, "dw_tool_next", "下一步：设置抛光斑（TIF）", "idx_dwell_spot",
            tooltip="继续到制定方案的第 3 步，导入或标定抛光斑。")
        layout.addWidget(flow_bar)
        self._resource_flow_widgets.append(flow_bar)
        layout.addStretch()
        self.dw_disk_diameter.textChanged.connect(self._dwell_invalidate_fit)
        return scroll

    def _open_resource_from_plan(self, index):
        """从制定方案进入工艺资源页：这样的进入才显示上一步 / 下一步按钮。"""
        self._resource_entry_from_plan = True
        self.setCurrentIndex(index)

    def _update_resource_flow(self, index):
        """按进入方式显示或隐藏工艺资源页里的流程按钮（页面切换时调用）。"""
        resource_pages = (self.idx_dwell_initial, self.idx_dwell_spot)
        if index in resource_pages:
            self._resource_in_flow = (
                self._resource_entry_from_plan
                or (self._last_index in resource_pages and self._resource_in_flow))
            self._resource_entry_from_plan = False
            for widget in self._resource_flow_widgets:
                widget.setVisible(self._resource_in_flow)
        else:
            self._resource_in_flow = False
        self._last_index = index

    def _dwell_suggest_process(self):
        try:
            diameter = self._dwell_float(self.dw_disk_diameter, "抛光盘口径", 0, False)
            speed = self._dwell_float(self.dw_disk_speed, "抛光盘转速", 0)
            if not np.isfinite([diameter, speed]).all():
                raise ValueError("口径和转速必须是有限数值")
            self.dw_pressure.setText(f"{3.14 * (diameter / 2) ** 2 * 0.028:.6g}")
            self.dw_feed.setText(f"{speed * diameter / 2 * 0.1:.6g}")
            self._dwell_report("已填入推荐的压力和进给，可按实际工况修改。", self.dw_initial_info)
        except ValueError as exc:
            self._dwell_error("工艺建议", exc, self.dw_initial_info)

    def _dwell_collect_process_records(self):
        """本次方案使用的工具与工况记录（随方案保存，不参与计算）。"""
        records = {"tool": self.workflow_plan_tool.currentText(),
                   "stage": self.dw_stage.currentText(),
                   "material": self.dw_disk_material.text().strip()}
        for key, attr, name in (("disk_diameter", "dw_disk_diameter", "抛光盘口径"),
                                ("disk_speed", "dw_disk_speed", "抛光盘转速"),
                                ("pressure", "dw_pressure", "抛光压力"),
                                ("feed", "dw_feed", "进给速度")):
            edit = getattr(self, attr)
            value = self._dwell_float(edit, name, 0) if edit.text().strip() else None
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name}必须是有限数值")
            records[key] = value
        return records

    # ── 制定方案第 2 步：工具与贴合 ─────────────────────────────────────
    def _build_plan_tool_section(self, layout):
        group = QGroupBox("2  工具与贴合")
        box = QVBoxLayout(group)
        self.workflow_plan_tool = QComboBox()
        self.workflow_plan_tool.addItems(["小工具", "气囊"])
        if self._dwell_state.get("tool"):
            self.workflow_plan_tool.setCurrentText(self._dwell_state["tool"])
        self._add_inline_combo(box, "加工工具：", self.workflow_plan_tool)
        self.dw_stage = QComboBox(); self.dw_stage.addItems(["粗抛", "平滑"])
        self._add_inline_combo(box, "抛光阶段：", self.dw_stage)
        self.dw_stage.currentTextChanged.connect(self._dwell_stage_changed)

        self.dw_tool_summary = QLabel()
        self.dw_tool_summary.setWordWrap(True)
        self.dw_tool_summary.setStyleSheet(_NOTE)
        box.addWidget(self.dw_tool_summary)
        edit_tool = QPushButton("修改工具与设备参数（工艺资源）")
        edit_tool.clicked.connect(lambda: self._open_resource_from_plan(self.idx_dwell_initial))
        box.addWidget(edit_tool)

        check = QPushButton("贴合检查")
        check.setObjectName("primaryBtn")
        check.clicked.connect(self._dwell_check_disk_fit)
        box.addWidget(check)
        self.dw_fit_info = QLabel("用期望曲面的面形参数和抛光盘口径估算接触间隙；结果在左侧显示。")
        self.dw_fit_info.setWordWrap(True)
        box.addWidget(self.dw_fit_info)
        hint = QLabel("间隙 ≤0.05 mm、空间变化率 ≤5×10⁻⁵ 为参考标准；只支持平面、球面、非球面，"
                      "不含弹性变形与边缘悬伸。工具类型目前只记录到方案，不参与数值求解。")
        hint.setWordWrap(True); hint.setStyleSheet(_NOTE)
        box.addWidget(hint)
        layout.addWidget(group)
        self.workflow_tool_group = group

    def _dwell_stage_changed(self, stage):
        self.dw_disk_material.setText("聚氨酯" if stage == "粗抛" else "沥青")

    def _refresh_tool_summary(self):
        if not hasattr(self, "dw_tool_summary"):
            return
        values = []
        for caption, attr, unit in (("抛光盘口径", "dw_disk_diameter", " mm"),
                                    ("材料", "dw_disk_material", ""),
                                    ("转速", "dw_disk_speed", " rpm"),
                                    ("压力", "dw_pressure", ""),
                                    ("进给", "dw_feed", " mm/min")):
            text = getattr(self, attr).text().strip()
            values.append(f"{caption} {text}{unit}" if text else f"{caption} 未填")
        try:
            r, k, _a, d = self._dwell_fit_surface()
            surface = f"贴合口径 D={d:.4g} mm，R={r:g}，K={k:g}"
        except ValueError as exc:
            surface = str(exc)
        self.dw_tool_summary.setText(
            "工具参数（来自工艺资源）：" + "，".join(values) + "。\n工件（来自导入工件）：" + surface + "。")

    def _dwell_fit_surface(self):
        """从导入工件的期望曲面取贴合检查需要的 R、K、A4/A6/A8 和工件最大口径 D。"""
        model = self._dwell_state.get("model")
        if model is None:
            raise ValueError("尚未生成期望曲面")
        config = model.get("config") if isinstance(model, dict) else None
        if not config:
            raise ValueError("期望曲面缺少面形参数，请重新生成")
        kind = config.get("type")
        if kind == "平面":
            r, k, a = 0.0, 0.0, [0.0, 0.0, 0.0]
        elif kind == "球面":
            r, k, a = config.get("r", 0.0), 0.0, [0.0, 0.0, 0.0]
        elif kind == "非球面":
            r, k = config.get("r", 0.0), config.get("k", 0.0)
            a = (list(config.get("a", [])) + [0.0, 0.0, 0.0])[:3]
        else:
            raise ValueError(f"{kind}暂不支持贴合检查（只支持平面、球面、非球面）")
        if config.get("aperture") == "矩形":
            extent = math.hypot(config.get("lx", 0.0), config.get("ly", 0.0))
        else:
            extent = config.get("diameter", 0.0)
        # 离轴件：母镜上最远点到光轴的距离决定需要检查的半径
        d = 2.0 * math.hypot(config.get("x_offset", 0.0), config.get("y_offset", 0.0)) + extent
        return r, k, a, d

    def _dwell_invalidate_fit(self, *_):
        if self._dwell_state.get("disk_fit") is not None:
            self._dwell_state["disk_fit"] = None
        if hasattr(self, "dw_fit_info"):
            self.dw_fit_info.setText("抛光盘口径已变化，请重新做贴合检查。")
            self.dw_fit_info.setStyleSheet("color:#52657a;")
        self._refresh_tool_summary()

    def _dwell_check_disk_fit(self):
        try:
            r, k, a, d = self._dwell_fit_surface()
            disk = self._dwell_float(self.dw_disk_diameter, "抛光盘口径", 0, False)
            result = DwellTimeEngine.disk_fit(r, k, a, d, disk)
            self._dwell_state["disk_fit"] = result
            self._dwell_refresh_fit_preview()
            verdict = "满足参考标准" if result["passed"] else "超出参考标准，建议减小抛光盘口径"
            self._dwell_report(f"贴合检查：最大间隙 {result['max_gap']:.6g} mm；"
                               f"最大空间变化率 {result['max_rate']:.6g}。{verdict}。",
                               self.dw_fit_info, error=not result["passed"])
        except ValueError as exc:
            self._dwell_error("贴合检查", exc, self.dw_fit_info)

    def _dwell_refresh_fit_preview(self):
        self._main.preview.plot_dwell_fit(self._dwell_state.get("disk_fit"))
