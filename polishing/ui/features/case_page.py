"""案例文件：保存与重新打开（由窗口“工具”栏调用）。"""
import os
from PyQt5.QtWidgets import QFileDialog
from polishing.workflow.case_io import load_case, restore_case, save_case
from polishing.workflow.state import create_dwell_state

# 上游到下游；恢复后若界面控件的信号清空了某项，按此顺序补回。
_REAPPLY_ORDER = ("dx", "dy", "model", "surface_raw", "surface_pixel", "surface", "surface_meta",
                  "alignment", "spot_raw", "spot_pixel", "spot_before", "spot", "tool",
                  "process_records", "protocol", "target", "evaluation", "coverage_preview",
                  "process_plan", "solution", "assessment", "frozen_plan")
_GRID_FIELDS = ("dw_model_dx", "dw_model_dy", "dw_dx", "dw_dy", "dw_surface_dx", "dw_surface_dy",
                "dw_spot_dx", "dw_spot_dy", "dw_solve_dx", "dw_solve_dy")


class CaseFileMixin:
    @staticmethod
    def _case_set_text(edit, value):
        edit.blockSignals(True)
        edit.setText("" if value is None else f"{value:g}" if isinstance(value, (int, float)) else str(value))
        edit.blockSignals(False)

    @staticmethod
    def _case_set_combo(combo, data):
        index = combo.findData(data)
        if index >= 0:
            combo.blockSignals(True)
            combo.setCurrentIndex(index)
            combo.blockSignals(False)

    def _case_save(self):
        path, _ = QFileDialog.getSaveFileName(
            self._main, "保存加工案例", "case.polcase", "加工案例 (*.polcase)")
        if path:
            self._case_save_to(path)

    def _case_save_to(self, path):
        try:
            manifest = save_case(self._dwell_state, path,
                                 name=os.path.splitext(os.path.basename(path))[0])
        except Exception as exc:
            self._dwell_error("案例保存失败", exc)
            return
        self._dwell_report(
            f"案例已保存：{os.path.basename(path)}（输入指纹 {manifest['input_fingerprint'][:12]}…）。"
            "曲面网格、检测、轨迹和 CNC 不保存：打开时自动重算前两项，轨迹和 CNC 需重新生成。")

    def _case_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self._main, "打开加工案例", "", "加工案例 (*.polcase)")
        if path:
            self._case_open_from(path)

    def _case_open_from(self, path):
        try:
            inputs, records, manifest = load_case(path)
        except Exception as exc:
            self._dwell_error("案例打开失败", exc)
            return
        state = self._dwell_state
        for key, value in create_dwell_state().items():
            state[key] = value
        report = restore_case(state, inputs, records, manifest)
        snapshot = dict(state)
        self._case_populate_widgets()
        for key in _REAPPLY_ORDER:
            if snapshot.get(key) is not None and state.get(key) is not snapshot[key]:
                state[key] = snapshot[key]
        state.invalidated.clear()
        self.setCurrentIndex(self.idx_workflow_import)
        self._refresh_workflow_statuses()
        self._case_refresh_previews()
        lines = [f"{'✔' if ok else '✘'} {name}：{text}" for name, ok, text in report]
        self._dwell_report(
            f"案例已打开：{os.path.basename(path)}。" + " ".join(lines)
            + "（轨迹和 CNC 未保存，需在生成程序阶段重新生成。）",
            error=not all(ok for _, ok, _ in report))

    def _case_populate_widgets(self):
        """把恢复出来的状态写回各页输入框；写入时屏蔽信号，避免控件反过来清空状态。"""
        state = self._dwell_state
        dx, dy = state.get("dx", 1.0), state.get("dy", 1.0)
        for attr in _GRID_FIELDS:
            if hasattr(self, attr):
                self._case_set_text(getattr(self, attr), dx if attr.endswith("dx") else dy)
        model = state.get("model")
        if model is not None:
            self._dwell_apply_model_config(model["config"])
        meta = state.get("surface_meta") or {}
        if meta.get("file"):
            self._case_set_text(self.dw_surface_file, meta["file"])
        self._workflow_apply_surface_meta()
        self._workflow_refresh_alignment()
        protocol = state.get("protocol")
        if protocol:
            self._case_set_text(self.workflow_eval_min, protocol["band_min"])
            self._case_set_text(self.workflow_eval_max, protocol["band_max"])
            self._workflow_sync_band_fields()
            self._case_set_combo(self.workflow_eval_detrend, protocol["detrend"])
            self.workflow_evaluation_result.setText("面形检测已按保存的检测标准重新计算，详见左侧图表。")
        target = state.get("target") or {}
        self._case_set_text(self.workflow_target_msf, target.get("msf_rms_max"))
        self._case_set_text(self.workflow_target_rms, target.get("rms_max"))
        time_s = target.get("time_max_s")
        self._case_set_text(self.workflow_target_time, None if time_s is None else time_s / 3600.0)
        plan = state.get("process_plan")
        if plan:
            kind = plan.get("process_type", "scanning")
            self._case_set_combo(self.workflow_process_type, kind)
            self._workflow_process_confirmed = True
            self.workflow_scan_controls.setVisible(kind == "scanning")
            self.workflow_tool_group.setVisible(True)
            self.workflow_vibration_note.setVisible(kind != "scanning")
            self.workflow_vibration_controls.setVisible(kind != "scanning")
            if kind == "conformal_vibration":
                model_values = plan["model"]
                for attr, key in (("workflow_vib_beta0", "beta0"), ("workflow_vib_qref", "q_ref"),
                                  ("workflow_vib_exponent", "exponent"),
                                  ("workflow_vib_aniso", "anisotropy"),
                                  ("workflow_vib_direction", "direction_deg"),
                                  ("workflow_vib_pnew", "p_new_rate")):
                    self._case_set_text(getattr(self, attr), model_values[key])
                self._case_set_text(self.workflow_vib_duration, plan["duration_s"])
        coverage = state.get("coverage_preview")
        if coverage:
            raster = coverage["pattern"] == "raster"
            self._case_set_text(self.workflow_coverage_spacing,
                                coverage["line_spacing_mm"] if raster else coverage["pitch_mm"])
            self._case_set_text(self.workflow_coverage_step, coverage["point_step_mm"])
            self._case_set_combo(self.workflow_coverage_pattern, coverage["pattern"])
            if raster:
                self._case_set_combo(self.workflow_coverage_direction, coverage["direction"])
            self._workflow_coverage_pattern_changed()
        records = state.get("process_records") or {}
        for attr, key in (("dw_disk_diameter", "disk_diameter"), ("dw_disk_speed", "disk_speed"),
                          ("dw_pressure", "pressure"), ("dw_feed", "feed"), ("dw_disk_material", "material")):
            if records.get(key) is not None:
                self._case_set_text(getattr(self, attr), records[key])
        if records.get("stage"):
            self.dw_stage.blockSignals(True)
            self.dw_stage.setCurrentText(records["stage"])
            self.dw_stage.blockSignals(False)
        if state.get("tool"):
            self.workflow_plan_tool.blockSignals(True)
            self.workflow_plan_tool.setCurrentText(state["tool"])
            self.workflow_plan_tool.blockSignals(False)

    def _case_refresh_previews(self):
        """恢复后尽力刷新左侧预览；三维预览不可用时只影响显示，不影响案例数据。"""
        state = self._dwell_state
        preview = self._main.preview
        try:
            if state.get("model") is not None:
                preview.plot_dwell_model(state["model"], "期望曲面（来自案例）")
            if state.get("surface") is not None:
                self._dwell_refresh_surface_preview()
        except Exception as exc:
            self._main.terminal_output.appendPlainText(f"[案例] 预览未刷新：{exc}")
