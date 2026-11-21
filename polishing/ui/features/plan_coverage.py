"""Stage three: process selection, modeled-surface coverage, and candidate plans."""
import numpy as np
from PyQt5.QtWidgets import QComboBox, QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget
from polishing.core.vibration import VibrationModel
from polishing.ui.widgets.inputs import lineedit_input

class PlanCoverageMixin:
    def _build_plan_controls(self, box):
        # ── 第 1 步：加工方式（检测给出建议，由用户决定）──
        method_group = QGroupBox("1  加工方式")
        method_box = QVBoxLayout(method_group)
        self.workflow_process_suggestion = QLabel()
        self.workflow_process_suggestion.setWordWrap(True)
        self.workflow_process_suggestion.setStyleSheet(
            "background:#e8f0fa; color:#10243f; padding:6px; border:1px solid #b8c7d8;")
        method_box.addWidget(self.workflow_process_suggestion)
        process_row = QHBoxLayout()
        process_label = QLabel("加工方式：")
        process_label.setMinimumWidth(84)
        process_row.addWidget(process_label)
        self.workflow_process_type = QComboBox()
        self.workflow_process_type.addItem("扫描驻留", "scanning")
        self.workflow_process_type.addItem("共形振动", "conformal_vibration")
        process_row.addWidget(self.workflow_process_type, 1)
        self.workflow_confirm_process = QPushButton("确定加工方式")
        self.workflow_confirm_process.clicked.connect(self._workflow_confirm_process_type)
        process_row.addWidget(self.workflow_confirm_process)
        method_box.addLayout(process_row)
        self._workflow_process_confirmed = False

        self.workflow_vibration_note = QLabel(
            "共形振动分支目前是骨架：按“初始 PSD × exp(−2βt) + 新引入误差”预测中频演化。"
            "下面的参数需要手动填写，未经实验标定，预测只用于流程演示，结果会标注为外推。")
        self.workflow_vibration_note.setWordWrap(True)
        self.workflow_vibration_note.setStyleSheet(
            "background:#fff8e8; color:#73510d; padding:8px; border:1px solid #e4c276;")
        method_box.addWidget(self.workflow_vibration_note)
        self.workflow_vibration_note.setVisible(False)
        box.addWidget(method_group)

        # ── 第 2 步：工具与贴合（两种方式共用），确认加工方式后出现 ──
        self._build_plan_tool_section(box)
        self.workflow_tool_group.setVisible(False)

        self.workflow_vibration_controls = QWidget()
        vibration_layout = QVBoxLayout(self.workflow_vibration_controls)
        vibration_layout.setContentsMargins(0, 0, 0, 0)
        vibration_group = QGroupBox("3  共形振动参数（手动，待标定）")
        vibration_box = QVBoxLayout(vibration_group)
        self._dwell_add_fields(vibration_box, [
            ("workflow_vib_beta0", "平滑速率 β0 (1/s)：", "0.002"),
            ("workflow_vib_qref", "参考空间频率 q_ref (周期/mm)：", "0.1"),
            ("workflow_vib_exponent", "频率指数（β ∝ (q/q_ref)^n）：", "1"),
            ("workflow_vib_aniso", "方向各向异性 (0–1)：", "0"),
            ("workflow_vib_direction", "振动方向 (°，0=X)：", "0"),
            ("workflow_vib_pnew", "新误差率 P_new (PSD 单位/s)：", "0"),
            ("workflow_vib_duration", "计划加工时长 (s)：", "600"),
        ])
        vibration_hint = QLabel(
            "默认值只是演示值。β 的频率/方向依赖形式和 P_new 都是占位假设，"
            "待第一轮实验后用“幅值随时间衰减”辨识。")
        vibration_hint.setWordWrap(True)
        vibration_hint.setStyleSheet("color:#52657a; font-size:11px;")
        vibration_box.addWidget(vibration_hint)
        vibration_layout.addWidget(vibration_group)
        self.workflow_vibration_controls.setVisible(False)
        box.addWidget(self.workflow_vibration_controls)
        for edit in (self.workflow_vib_beta0, self.workflow_vib_qref, self.workflow_vib_exponent,
                     self.workflow_vib_aniso, self.workflow_vib_direction,
                     self.workflow_vib_pnew, self.workflow_vib_duration):
            edit.textChanged.connect(self._workflow_invalidate_plan)

        self.workflow_scan_controls = QWidget()
        scan_layout = QVBoxLayout(self.workflow_scan_controls)
        scan_layout.setContentsMargins(0, 0, 0, 0)
        self.workflow_scan_controls.setVisible(False)

        self._build_plan_tif_section(scan_layout)

        coverage_group = QGroupBox("4  覆盖路径并预览")
        coverage_box = QVBoxLayout(coverage_group)
        self.workflow_coverage_spacing, spacing_row = lineedit_input("行间距 (mm)：", "")
        self.workflow_coverage_spacing_label = spacing_row.itemAt(0).widget()
        coverage_box.addLayout(spacing_row)
        self.workflow_coverage_step, step_row = lineedit_input("轨迹点间距 (mm)：", "")
        self.workflow_coverage_step_label = step_row.itemAt(0).widget()
        coverage_box.addLayout(step_row)
        self.workflow_coverage_pattern = QComboBox()
        self.workflow_coverage_pattern.addItem("栅形", "raster")
        self.workflow_coverage_pattern.addItem("螺旋形", "spiral")
        self._add_inline_combo(coverage_box, "轨迹形式：", self.workflow_coverage_pattern)
        self.workflow_coverage_direction = QComboBox()
        self.workflow_coverage_direction.addItem("沿 X 方向走刀", "X")
        self.workflow_coverage_direction.addItem("沿 Y 方向走刀", "Y")
        self.workflow_coverage_direction_row = QWidget()
        direction_layout = QHBoxLayout(self.workflow_coverage_direction_row)
        direction_layout.setContentsMargins(0, 0, 0, 0)
        direction_label = QLabel("走刀方向：")
        direction_label.setMinimumWidth(84)
        direction_layout.addWidget(direction_label)
        direction_layout.addWidget(self.workflow_coverage_direction, 1)
        coverage_box.addWidget(self.workflow_coverage_direction_row)
        self.workflow_coverage_hint = QLabel()
        self.workflow_coverage_hint.setWordWrap(True)
        self.workflow_coverage_hint.setStyleSheet("color:#52657a; font-size:11px;")
        coverage_box.addWidget(self.workflow_coverage_hint)
        preview_row = QHBoxLayout()
        self.workflow_coverage_preview = QPushButton("轨迹预览")
        self.workflow_coverage_preview.clicked.connect(self._workflow_preview_coverage)
        preview_row.addWidget(self.workflow_coverage_preview)
        self.workflow_coverage_status = QLabel("尚未预览覆盖轨迹。")
        self.workflow_coverage_status.setWordWrap(True)
        preview_row.addWidget(self.workflow_coverage_status, 1)
        coverage_box.addLayout(preview_row)
        scan_layout.addWidget(coverage_group)

        self.workflow_plan_tool.currentIndexChanged.connect(self._workflow_tool_changed)
        self.workflow_coverage_spacing.textChanged.connect(self._workflow_invalidate_coverage)
        self.workflow_coverage_step.textChanged.connect(self._workflow_invalidate_coverage)
        self.workflow_coverage_pattern.currentIndexChanged.connect(
            self._workflow_coverage_pattern_changed)
        self.workflow_coverage_direction.currentIndexChanged.connect(
            self._workflow_invalidate_coverage)
        self.workflow_process_type.currentIndexChanged.connect(
            self._workflow_process_selection_changed)
        box.addWidget(self.workflow_scan_controls)

        # ── 最后一步：生成候选方案（编号随加工方式变化）──
        generate_group = QGroupBox("5  生成候选方案")
        self.workflow_generate_group = generate_group
        generate_box = QVBoxLayout(generate_group)
        self.workflow_plan_readiness = QLabel()
        self.workflow_plan_readiness.setWordWrap(True)
        self.workflow_plan_readiness.setStyleSheet(
            "background:#e8f0fa; color:#10243f; padding:8px; border:1px solid #b8c7d8;")
        generate_box.addWidget(self.workflow_plan_readiness)
        confirm = QPushButton("生成候选方案")
        confirm.setObjectName("primaryBtn")
        confirm.clicked.connect(self._workflow_confirm_plan)
        self.workflow_candidate_button = confirm
        generate_box.addWidget(confirm)
        self._dwell_add_next_button(
            generate_box, "workflow_plan_next", "下一步：效果预估",
            "idx_workflow_predict", required=("process_plan",),
            tooltip="候选方案生成后进入效果预估阶段。")
        box.addWidget(generate_group)
        self._workflow_coverage_pattern_changed()
        self._refresh_plan_readiness()

    def _workflow_read_vibration(self):
        """读取共形振动手动参数，返回 (VibrationModel, 计划时长 s)。"""
        model = VibrationModel(
            beta0=self._dwell_float(self.workflow_vib_beta0, "β0", 0.0),
            q_ref=self._dwell_float(self.workflow_vib_qref, "参考空间频率", 0.0, False),
            exponent=self._dwell_float(self.workflow_vib_exponent, "频率指数"),
            anisotropy=self._dwell_float(self.workflow_vib_aniso, "各向异性", 0.0),
            direction_deg=self._dwell_float(self.workflow_vib_direction, "振动方向"),
            p_new_rate=self._dwell_float(self.workflow_vib_pnew, "新误差率", 0.0)).validate()
        return model, self._dwell_float(self.workflow_vib_duration, "计划加工时长", 0.0, False)

    def _workflow_plan_missing(self):
        if (getattr(self, "_workflow_process_confirmed", False)
                and self.workflow_process_type.currentData() == "conformal_vibration"):
            missing = [label for key, label in (("model", "期望曲面"), ("evaluation", "加工前面形检测"))
                       if self._dwell_state.get(key) is None]
            try:
                self._workflow_read_vibration()
            except ValueError as exc:
                missing.append(f"共形振动参数（{exc}）")
            return missing
        checks = (
            ("model", "期望曲面"),
            ("evaluation", "加工前面形检测"),
            ("spot", "抛光斑/TIF"),
            ("coverage_preview", "曲面覆盖轨迹预览"),
        )
        missing = [label for key, label in checks if self._dwell_state.get(key) is None]
        if not getattr(self, "_workflow_process_confirmed", False):
            missing.insert(0, "确认加工方式")
        elif self.workflow_coverage_pattern.currentData() == "spiral":
            missing.append("螺旋驻留执行路径（当前只支持曲面预览）")
        if not self.workflow_plan_tool.currentText().strip():
            missing.append("加工工具")
        return missing

    def _refresh_plan_readiness(self):
        label = getattr(self, "workflow_plan_readiness", None)
        if label is None:
            return
        self._refresh_plan_tif_info()
        self._refresh_tool_summary()
        self.workflow_process_suggestion.setText(self._workflow_process_suggestion())
        vibration = self.workflow_process_type.currentData() == "conformal_vibration"
        self.workflow_generate_group.setTitle(f"{4 if vibration else 5}  生成候选方案")
        missing = self._workflow_plan_missing()
        candidate_button = getattr(self, "workflow_candidate_button", None)
        if candidate_button is not None:
            candidate_button.setEnabled(not missing and self._workflow_process_confirmed)
        if missing:
            label.setText("候选方案尚未就绪，缺少：" + "、".join(missing) + "。")
            label.setStyleSheet(
                "background:#fff8e8; color:#73510d; padding:8px; border:1px solid #e4c276;")
        else:
            if self._dwell_state.get("process_plan") is None:
                label.setText(
                    "加工前面形检测和共形振动参数已就绪，可以生成 1 个当前候选方案（未标定）。"
                    if self.workflow_process_type.currentData() == "conformal_vibration" else
                    "工具、TIF 和曲面覆盖预览已就绪，可以生成 1 个当前候选方案。")
            else:
                label.setText("当前候选方案已生成（1 个）；调整工具或覆盖参数后需重新生成。")
            label.setStyleSheet(
                "background:#e8f5ec; color:#1a6f3a; padding:8px; border:1px solid #9bc7aa;")

    def _workflow_confirm_plan(self):
        missing = self._workflow_plan_missing()
        if missing:
            self._dwell_error(
                "制定方案尚未完成",
                ValueError("请先完成：" + "、".join(missing)))
            self._refresh_plan_readiness()
            return
        try:
            records = self._dwell_collect_process_records()
        except ValueError as exc:
            self._dwell_error("工具与设备参数", exc)
            return
        self._dwell_state["process_records"] = records
        self._dwell_state["tool"] = records["tool"]
        if self.workflow_process_type.currentData() == "conformal_vibration":
            model, duration = self._workflow_read_vibration()
            self._dwell_state["process_plan"] = {
                "process_type": "conformal_vibration",
                "tool": records["tool"], "stage": records["stage"], "process_records": records,
                "model": model.to_dict(), "duration_s": duration,
                "calibration_status": model.calibration, "validated": False,
            }
            self._dwell_state.mark_current("process_plan")
            self._refresh_plan_readiness()
            self._refresh_workflow_statuses()
            self._dwell_report("已生成 1 个共形振动候选方案（手动参数，未标定），可以进入效果预估阶段。")
            return
        spot = self._dwell_state["spot"]
        coverage = self._dwell_state["coverage_preview"]
        plan = {
            "process_type": "scanning",
            "tool": records["tool"],
            "stage": records["stage"],
            "process_records": records,
            "tif_shape": tuple(spot["kernel"].shape),
            "tif_impulse": float(spot["impulse"]),
            "trajectory_strategy": "dwell_driven_raster",
            "coverage": dict(coverage),
            "model_required_for_curved_cnc": True,
        }
        self._dwell_state["process_plan"] = plan
        self._dwell_state.mark_current("process_plan")
        if coverage["direction"] == "X":
            self.dw_cnc_dx.setText(f"{coverage['point_step_mm']:g}")
            self.dw_cnc_dy.setText(f"{coverage['line_spacing_mm']:g}")
        else:
            self.dw_cnc_dx.setText(f"{coverage['line_spacing_mm']:g}")
            self.dw_cnc_dy.setText(f"{coverage['point_step_mm']:g}")
        self.dw_cnc_line.setCurrentText(coverage["direction"])
        self._refresh_plan_readiness()
        self._refresh_workflow_statuses()
        self._dwell_report("已生成 1 个扫描驻留候选方案，可以进入效果预估阶段。")

    def _workflow_process_selection_changed(self, *_args):
        self._workflow_process_confirmed = False
        self.workflow_scan_controls.setVisible(False)
        self.workflow_tool_group.setVisible(False)
        self.workflow_vibration_note.setVisible(False)
        self.workflow_vibration_controls.setVisible(False)
        self.workflow_candidate_button.setEnabled(False)
        self._workflow_invalidate_plan()

    def _workflow_confirm_process_type(self):
        self._workflow_process_confirmed = True
        scanning = self.workflow_process_type.currentData() == "scanning"
        self.workflow_scan_controls.setVisible(scanning)
        self.workflow_tool_group.setVisible(True)
        self.workflow_vibration_note.setVisible(not scanning)
        self.workflow_vibration_controls.setVisible(not scanning)
        self._refresh_plan_readiness()

    def _workflow_tool_changed(self, *_args):
        self._dwell_state["tool"] = self.workflow_plan_tool.currentText()
        self._workflow_invalidate_plan()

    def _workflow_invalidate_plan(self, *_args):
        if self._dwell_state.get("process_plan") is not None:
            self._dwell_state["process_plan"] = None
        self._refresh_workflow_statuses()

    def _workflow_invalidate_coverage(self, *_args):
        self._dwell_state["coverage_preview"] = None
        self.workflow_coverage_status.setText("参数已更改，请重新预览覆盖轨迹。")
        self.workflow_coverage_status.setStyleSheet("color:#73510d;")
        self._refresh_workflow_statuses()

    def _workflow_coverage_pattern_changed(self, *_args):
        raster = self.workflow_coverage_pattern.currentData() == "raster"
        self.workflow_coverage_direction_row.setVisible(raster)
        self.workflow_coverage_spacing_label.setText(
            "行间距 (mm)：" if raster else "螺距 (mm)：")
        self.workflow_coverage_step_label.setText(
            "轨迹点间距 (mm)：" if raster else "弧长取点间距 (mm)：")
        if raster:
            self.workflow_coverage_hint.setText(
                "蓝线表示工件表面上的加工覆盖线段。边缘跨线转移未画成加工线；"
                "当前驻留求解仍使用案例网格和 TIF，方向及间距会带入生成程序页。")
        else:
            self.workflow_coverage_hint.setText(
                "螺旋形会贴合导入工件曲面显示覆盖预览。当前驻留求解与“生成程序”的刀路仅支持栅形，"
                "因此螺旋形暂不能生成可执行的扫描驻留候选方案。")
        self._workflow_invalidate_coverage()

    def _workflow_preview_coverage(self):
        try:
            model = self._dwell_state.get("model")
            if model is None:
                raise ValueError("请先在导入工件阶段生成期望曲面模型")
            spacing = self._dwell_float(
                self.workflow_coverage_spacing, "行间距或螺距", 0.0, False)
            step = self._dwell_float(
                self.workflow_coverage_step, "轨迹点间距", 0.0, False)
            pattern = self.workflow_coverage_pattern.currentData()
            if pattern == "raster":
                paths = self._workflow_surface_raster(
                    model, spacing, step, self.workflow_coverage_direction.currentData())
            else:
                paths = self._workflow_surface_spiral(model, spacing, step)
            point_count = sum(len(path) for path in paths)
            segment_count = len(paths)
            if point_count < 2:
                raise ValueError("当前口径和间距下没有足够的有效轨迹点")
            if point_count > 60000:
                raise ValueError("预览点超过 60000 个；请增大点间距或行间距后再预览")
            preview = {
                "pattern": pattern,
                "direction": self.workflow_coverage_direction.currentData() if pattern == "raster" else None,
                "line_spacing_mm": spacing if pattern == "raster" else None,
                "pitch_mm": spacing if pattern == "spiral" else None,
                "point_step_mm": step,
                "point_count": point_count,
                "segment_count": segment_count,
            }
            self._dwell_state["coverage_preview"] = preview
            self._dwell_state.mark_current("coverage_preview")
            self._coverage_view = (model, paths, pattern)
            self._main.preview.plot_dwell_coverage(model, paths, pattern=pattern)
            self.workflow_coverage_status.setText(
                f"预览完成：{segment_count} 条有效线段，{point_count} 个采样点。")
            self.workflow_coverage_status.setStyleSheet("color:#1a6f3a;")
            self._refresh_workflow_statuses()
        except Exception as exc:
            self.workflow_coverage_status.setText(str(exc))
            self.workflow_coverage_status.setStyleSheet("color:#c0392b;")

    @staticmethod
    def _workflow_surface_raster(model, spacing, step, direction):
        points = np.asarray(model["points"], dtype=float)
        normals = np.asarray(model["normals"], dtype=float)
        mask = np.asarray(model["mask"], dtype=bool)
        ny, nx = mask.shape
        dx = float(model["dx"])
        dy = float(model["dy"])
        if direction == "X":
            along = list(range(0, nx, max(1, int(round(step / dx)))))
            cross = list(range(0, ny, max(1, int(round(spacing / dy)))))
            if along[-1] != nx - 1: along.append(nx - 1)
            if cross[-1] != ny - 1: cross.append(ny - 1)
        else:
            along = list(range(0, ny, max(1, int(round(step / dy)))))
            cross = list(range(0, nx, max(1, int(round(spacing / dx)))))
            if along[-1] != ny - 1: along.append(ny - 1)
            if cross[-1] != nx - 1: cross.append(nx - 1)
        if len(along) * len(cross) > 60000:
            raise ValueError("预览估算点超过 60000 个；请增大点间距或行间距")
        scan_lines = []
        if direction == "X":
            scan_lines = [[(r, c) for c in along] for r in cross]
        else:
            scan_lines = [[(r, c) for r in along] for c in cross]
        paths = []
        clearance = max(min(dx, dy) * 0.025, 0.01)
        for line_index, line in enumerate(scan_lines):
            segments = []
            current = []
            for r, c in line:
                if (mask[r, c] and np.isfinite(points[r, c]).all()
                        and np.isfinite(normals[r, c]).all()):
                    point = points[r, c] + normals[r, c] * clearance
                    current.append(np.concatenate((point, normals[r, c])).tolist())
                elif current:
                    if len(current) > 1: segments.append(current)
                    current = []
            if len(current) > 1: segments.append(current)
            if line_index % 2:
                segments = [list(reversed(segment)) for segment in reversed(segments)]
            paths.extend(segments)
        return paths

    @staticmethod
    def _workflow_surface_spiral(model, pitch, arc_step):
        """Sample an Archimedean spiral on the modeled surface for preview only."""
        points = np.asarray(model["points"], dtype=float)
        normals = np.asarray(model["normals"], dtype=float)
        mask = np.asarray(model["mask"], dtype=bool)
        ny, nx = mask.shape
        dx, dy = float(model["dx"]), float(model["dy"])
        if not np.any(mask):
            raise ValueError("期望曲面没有有效口径点")
        xs = (np.arange(nx) - (nx - 1) / 2.0) * dx
        ys = (np.arange(ny) - (ny - 1) / 2.0) * dy
        local_x, local_y = np.meshgrid(xs, ys)
        max_radius = float(np.max(np.hypot(local_x[mask], local_y[mask])))
        b = pitch / (2.0 * np.pi)
        clearance = max(min(dx, dy) * 0.025, 0.01)
        theta = 0.0
        paths, current = [], []
        samples = 0
        while b * theta <= max_radius + 1e-9:
            radius = b * theta
            fx = radius * np.cos(theta) / dx + (nx - 1) / 2.0
            fy = radius * np.sin(theta) / dy + (ny - 1) / 2.0
            valid = 0 <= fx <= nx - 1 and 0 <= fy <= ny - 1
            if valid:
                c0 = int(np.floor(fx)); r0 = int(np.floor(fy))
                c1 = min(c0 + 1, nx - 1); r1 = min(r0 + 1, ny - 1)
                corners = ((r0, c0), (r0, c1), (r1, c0), (r1, c1))
                valid = all(mask[r, c] and np.isfinite(points[r, c]).all()
                            and np.isfinite(normals[r, c]).all() for r, c in corners)
            if valid:
                u, v = fx - c0, fy - r0
                weights = ((1 - u) * (1 - v), u * (1 - v),
                           (1 - u) * v, u * v)
                point = sum(weight * points[r, c]
                            for weight, (r, c) in zip(weights, corners))
                normal = sum(weight * normals[r, c]
                             for weight, (r, c) in zip(weights, corners))
                normal /= max(float(np.linalg.norm(normal)), 1e-12)
                current.append(np.concatenate((point + normal * clearance, normal)).tolist())
            elif current:
                if len(current) > 1: paths.append(current)
                current = []
            samples += 1
            if samples > 60000:
                raise ValueError("螺旋预览点超过 60000 个，请增大螺距或取点间距")
            theta += min(0.22, arc_step / max(np.hypot(radius, b), 1e-12))
        if len(current) > 1: paths.append(current)
        return paths

