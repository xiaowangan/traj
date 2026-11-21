"""Migrated dwell_solve page behavior; existing layout preserved."""
from __future__ import annotations
import numpy as np
from PyQt5.QtWidgets import QComboBox, QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget
from polishing.core.assessment import assess_solution, short_summary
from polishing.core.model_evaluation import AcceptanceTarget, EvaluationProtocol
from polishing.core.vibration import VibrationModel, predict_evolution
from polishing.ui.widgets.inputs import combox_input
from polishing.workflow.case_io import input_fingerprint
from polishing.workflow.engine import DwellTimeEngine

class PredictionPageMixin:
    def _build_dwell_solve_page(self):
        scroll, layout = self._dwell_scroll_page(
            "效果预估", "按方案中选定的加工方式试算加工结果，判定能否达标，确认后冻结方案。")

        # ── 扫描驻留：1 区域 → 2 求解 → 3 修饰（可选）→ 4 达标判定 ──
        self.dw_scan_predict = QWidget()
        scan = QVBoxLayout(self.dw_scan_predict)
        scan.setContentsMargins(0, 0, 0, 0)
        trim_group = QGroupBox("1  求解区域"); trim_form = QVBoxLayout(trim_group)
        self._dwell_add_fields(trim_form, [
            ("dw_traj_margin", "轨迹区裁边 (mm)：", "0"),
            ("dw_eval_trim", "检测区裁边 (mm)：", "0"),
        ])
        scan.addWidget(trim_group)

        method_group = QGroupBox("2  求解驻留时间"); method_form = QVBoxLayout(method_group)
        self.dw_solver_method = QComboBox()
        self.dw_solver_method.addItems(["—— 请选择算法类型 ——", "最小二乘法", "脉冲迭代法"])
        combox_input(method_form, "算法类型：", self.dw_solver_method)
        # 共用参数：两种算法都需要（与 MATLAB 一致）
        self.dw_box_uniform = self._dwell_field_box(
            method_form, "dw_uniform", "均抛厚度：", "0.5")
        self.dw_box_max_dwell = self._dwell_field_box(
            method_form, "dw_max_dwell", "最大驻留时间 (s)：", "20")
        self.dw_box_min_dwell = self._dwell_field_box(
            method_form, "dw_min_dwell", "最小驻留时间 (s)：", "0.02")
        # 最小二乘法专用参数
        self.dw_box_ls_iterations = self._dwell_field_box(
            method_form, "dw_solve_iterations", "最大迭代次数：", "300")
        self.dw_box_ls_tolerance = self._dwell_field_box(
            method_form, "dw_solve_tolerance", "收敛阈值：", "1e-6")
        # 脉冲迭代法专用参数与备注（算法暂不可用）
        self.dw_box_pulse_iterations = self._dwell_field_box(
            method_form, "dw_pulse_iterations", "迭代次数：", "300")
        self.dw_box_pulse_hint = QWidget()
        pulse_hint_layout = QVBoxLayout(self.dw_box_pulse_hint)
        pulse_hint_layout.setContentsMargins(0, 0, 0, 0); pulse_hint_layout.setSpacing(0)
        pulse_hint = QLabel("备注：脉冲迭代法暂不可用，请选择最小二乘法求解。")
        pulse_hint.setStyleSheet("color:#50647a; font-size:11px;")
        pulse_hint.setWordWrap(True)
        pulse_hint_layout.addWidget(pulse_hint)
        method_form.addWidget(self.dw_box_pulse_hint)
        self._solve_box_attrs = (
            "dw_box_uniform", "dw_box_max_dwell", "dw_box_min_dwell",
            "dw_box_ls_iterations", "dw_box_ls_tolerance",
            "dw_box_pulse_iterations", "dw_box_pulse_hint",
        )
        self.dw_solver_method.currentIndexChanged.connect(self._dwell_solve_sync_visibility)
        self._dwell_solve_sync_visibility()
        solve = QPushButton("求解驻留时间")
        solve.setObjectName("primaryBtn")
        method_form.addWidget(solve)
        self.dw_solve_info = QLabel(""); self.dw_solve_info.setWordWrap(True)
        method_form.addWidget(self.dw_solve_info)
        scan.addWidget(method_group)

        modify_group = QGroupBox("3  边缘修饰（可选）"); modify_form = QVBoxLayout(modify_group)
        self._dwell_add_fields(modify_form, [
            ("dw_modify_width", "边缘修饰宽度 (mm)：", "0"),
            ("dw_edge_adjust", "边缘驻留调整量：", "0"),
        ])
        modify = QPushButton("驻留时间修饰")
        modify_form.addWidget(modify)
        scan.addWidget(modify_group)

        # 计算网格只在导入工件设定；这里保留只读值供计算使用，不再显示
        grid_box = QWidget(); grid_form = QVBoxLayout(grid_box)
        self._dwell_add_fields(grid_form, [
            ("dw_solve_dx", "X 间距 (mm)：", "1"), ("dw_solve_dy", "Y 间距 (mm)：", "1"),
        ], readonly=("dw_solve_dx", "dw_solve_dy"))
        grid_box.setVisible(False)
        scan.addWidget(grid_box)

        # ── 预测后检测：预测残差按面形初测阶段冻结的协议重新检测，判定能否达标 ──
        assess_group = QGroupBox("4  达标判定（与加工前同一检测标准）")
        assess_form = QVBoxLayout(assess_group)
        self.dw_assess_info = QLabel("求解后自动检测；需要先在“面形初测”阶段确定检测标准。")
        self.dw_assess_info.setWordWrap(True)
        self.dw_assess_info.setStyleSheet("color:#52657a; font-size:11px;")
        assess_form.addWidget(self.dw_assess_info)
        view_assess = QPushButton("查看前后检测对比")
        view_assess.clicked.connect(self._dwell_show_assessment)
        assess_form.addWidget(view_assess)
        scan.addWidget(assess_group)
        layout.addWidget(self.dw_scan_predict)

        # ── 共形振动：1 参数（来自制定方案）→ 2 预测中频演化 ──
        self.dw_vib_predict = QWidget()
        vib = QVBoxLayout(self.dw_vib_predict)
        vib.setContentsMargins(0, 0, 0, 0)
        params_group = QGroupBox("1  振动参数（来自制定方案）")
        params_form = QVBoxLayout(params_group)
        self.dw_vib_plan_summary = QLabel()
        self.dw_vib_plan_summary.setWordWrap(True)
        self.dw_vib_plan_summary.setStyleSheet("color:#10243f; font-size:11px;")
        params_form.addWidget(self.dw_vib_plan_summary)
        edit_plan = QPushButton("去制定方案修改参数")
        edit_plan.clicked.connect(lambda: self.setCurrentIndex(self.idx_workflow_plan))
        params_form.addWidget(edit_plan)
        vib.addWidget(params_group)
        vibration_group = QGroupBox("2  预测中频演化（骨架，未标定）")
        vibration_form = QVBoxLayout(vibration_group)
        vibration_hint = QLabel(
            "由加工前 PSD 预测中频 RMS 随时间的变化、最佳加工时长和达标时间。"
            "结果为外推，不能当作已验证工艺。")
        vibration_hint.setWordWrap(True)
        vibration_hint.setStyleSheet("color:#52657a; font-size:11px;")
        vibration_form.addWidget(vibration_hint)
        vibration_run = QPushButton("预测中频演化")
        vibration_run.setObjectName("primaryBtn")
        vibration_run.clicked.connect(self._dwell_predict_vibration)
        vibration_form.addWidget(vibration_run)
        self.dw_vib_info = QLabel("")
        self.dw_vib_info.setWordWrap(True)
        vibration_form.addWidget(self.dw_vib_info)
        vib.addWidget(vibration_group)
        layout.addWidget(self.dw_vib_predict)

        # ── 最后一步：确认方案（冻结）→ 解锁执行 ──
        self.dw_freeze_group = QGroupBox("5  确认方案")
        freeze_form = QVBoxLayout(self.dw_freeze_group)
        self.dw_freeze_info = QLabel()
        self.dw_freeze_info.setWordWrap(True)
        freeze_form.addWidget(self.dw_freeze_info)
        row = QHBoxLayout()
        self.dw_confirm_button = QPushButton("确认并冻结方案")
        self.dw_confirm_button.setObjectName("primaryBtn")
        self.dw_confirm_button.clicked.connect(lambda: self._dwell_confirm_plan())
        back = QPushButton("返回制定方案调整")
        back.clicked.connect(lambda: self.setCurrentIndex(self.idx_workflow_plan))
        row.addWidget(self.dw_confirm_button, 1); row.addWidget(back)
        freeze_form.addLayout(row)
        self._dwell_add_next_button(
            freeze_form, "dw_solve_next", "下一步：生成程序",
            "idx_dwell_cnc", required=("frozen_plan",),
            tooltip="确认并冻结方案后进入生成程序阶段，生成变速轨迹与 CNC/PATH。")
        layout.addWidget(self.dw_freeze_group)
        layout.addStretch()
        solve.clicked.connect(self._do_dwell_least_squares)
        modify.clicked.connect(self._do_dwell_modify)
        return scroll

    def _refresh_predict_view(self):
        """效果预估页只显示方案中选定的那种加工方式，并刷新“确认方案”一步。"""
        if not hasattr(self, "dw_freeze_group"):
            return
        state = self._dwell_state
        plan = state.get("process_plan") or {}
        kind = plan.get("process_type")
        vibration = kind == "conformal_vibration"
        self.dw_scan_predict.setVisible(not vibration)
        self.dw_vib_predict.setVisible(vibration)
        self.dw_freeze_group.setTitle(f"{3 if vibration else 5}  确认方案")
        if vibration:
            model = plan.get("model", {})
            self.dw_vib_plan_summary.setText(
                f"β0={model.get('beta0', '—')} 1/s，q_ref={model.get('q_ref', '—')} 周期/mm，"
                f"新误差率 {model.get('p_new_rate', '—')}，计划时长 {plan.get('duration_s', '—')} s。")
        frozen = state.get("frozen_plan")
        assessment = state.get("assessment")
        has_result = (state.get("assessment") is not None if vibration
                      else state.get("solution") is not None)
        self.dw_confirm_button.setEnabled(bool(kind) and has_result and frozen is None)
        if frozen is not None:
            note = f"（理由：{frozen['reason']}）" if frozen.get("reason") else ""
            text, style = (f"方案已确认并冻结：{frozen.get('label', '当前方案')}，"
                           f"判定 {frozen.get('verdict_text', '—')}{note}。"
                           "修改上游参数会自动解除冻结。"), "color:#1a6f3a;"
        elif not has_result:
            text, style = ("先完成上面的预测，再确认方案。", "color:#52657a;")
        elif assessment is not None and assessment.get("verdict") == "fail":
            text, style = ("预测未达标。建议返回制定方案调整参数；如仍要按此方案加工，确认时需填写理由。",
                           "color:#73510d;")
        else:
            text, style = ("预测结果已出。确认后方案冻结，生成程序阶段才能生成加工文件。", "color:#10243f;")
        self.dw_freeze_info.setStyleSheet(style + " font-size:11px;")
        self.dw_freeze_info.setText(text)

    def _dwell_confirm_plan(self, reason=None):
        """冻结当前方案：预测先于加工；之后上游任何改动都会自动解除冻结。"""
        state = self._dwell_state
        plan = state.get("process_plan")
        try:
            if plan is None:
                raise ValueError("请先在制定方案阶段生成候选方案")
            vibration = plan.get("process_type") == "conformal_vibration"
            if vibration and state.get("assessment") is None:
                raise ValueError("请先预测中频演化")
            if not vibration and state.get("solution") is None:
                raise ValueError("请先求解驻留时间")
        except ValueError as exc:
            self._dwell_error("确认方案", exc, self.dw_freeze_info)
            return False
        assessment = state.get("assessment")
        verdict = assessment.get("verdict") if assessment else "no_target"
        if verdict == "fail" and not reason:
            from PyQt5.QtWidgets import QInputDialog
            reason, ok = QInputDialog.getText(
                self._main, "预测未达标", "预测未达标。如仍要按此方案加工，请写明理由：")
            reason = reason.strip()
            if not ok or not reason:
                return False
        state["frozen_plan"] = {
            "process_type": plan.get("process_type"),
            "label": "共形振动" if vibration else "扫描驻留",
            "verdict": verdict,
            "verdict_text": {"pass": "达标", "fail": "未达标", "no_target": "未设目标"}.get(verdict, verdict),
            "reason": reason or "",
            "input_fingerprint": input_fingerprint(state),
        }
        state.mark_current("frozen_plan")
        self._dwell_report("方案已确认并冻结，可以进入生成程序阶段。")
        self._refresh_workflow_statuses()
        return True

    def _dwell_solve_sync_visibility(self, *_args):
        """按算法下拉框的当前选择，只显示该算法需要填写的参数行（仿建模页）。"""
        shared = ("dw_box_uniform", "dw_box_max_dwell", "dw_box_min_dwell")
        shown = {
            "最小二乘法": shared + ("dw_box_ls_iterations", "dw_box_ls_tolerance"),
            "脉冲迭代法": shared + ("dw_box_pulse_iterations", "dw_box_pulse_hint"),
        }.get(self.dw_solver_method.currentText(), ())
        for attr in self._solve_box_attrs:
            getattr(self, attr).setVisible(attr in shown)

    def _do_dwell_least_squares(self):
        try:
            if self._dwell_state.get("process_plan") is None:
                raise ValueError("请先在制定方案阶段确认当前候选方案")
            method = self.dw_solver_method.currentText()
            if method == "—— 请选择算法类型 ——":
                raise ValueError("请先选择算法类型")
            if method != "最小二乘法":
                raise ValueError("脉冲迭代法暂不可用，请选择最小二乘法求解")
            surface = self._dwell_state.get("surface"); spot = self._dwell_state.get("spot")
            if surface is None: raise ValueError("请先完成面形数据导入与处理")
            if spot is None: raise ValueError("请先计算抛光斑去除函数")
            dx = self._dwell_float(self.dw_solve_dx, "X间距", 0.0, False)
            dy = self._dwell_float(self.dw_solve_dy, "Y间距", 0.0, False)
            solution = DwellTimeEngine.solve_bounded_least_squares(
                surface, spot["kernel"], dx, dy,
                uniform=self._dwell_float(self.dw_uniform, "均抛厚度", 0.0),
                min_dwell=self._dwell_float(self.dw_min_dwell, "最小驻留时间", 0.0),
                max_dwell=self._dwell_float(self.dw_max_dwell, "最大驻留时间", 0.0),
                trajectory_trim=self._dwell_float(self.dw_traj_margin, "轨迹区裁边", 0.0),
                evaluation_trim=self._dwell_float(self.dw_eval_trim, "检测区裁边", 0.0),
                iterations=int(self._dwell_float(self.dw_solve_iterations, "最大迭代次数", 1.0)),
                tolerance=self._dwell_float(self.dw_solve_tolerance, "收敛阈值", 0.0, False))
            self._dwell_state["solution"] = solution
            self._dwell_report(
                f"最小二乘求解完成：RMS {solution['before_rms']:.6g} → {solution['after_rms']:.6g}，"
                f"记录 {len(solution['objective'])} 个收敛检查点。", self.dw_solve_info)
            self._dwell_refresh_solve_preview()
            self._dwell_assess_prediction()
        except Exception as exc: self._dwell_error("驻留时间求解失败", exc, self.dw_solve_info)

    def _dwell_assess_message(self, text, warn=False):
        self.dw_assess_info.setText(text)
        self.dw_assess_info.setStyleSheet(
            "color:#73510d; font-size:11px;" if warn else "color:#10243f; font-size:11px;")

    def _dwell_assess_prediction(self):
        """用面形初测阶段冻结的协议重新检测预测残差，写入 assessment（含输入指纹）。"""
        state = self._dwell_state
        solution = state.get("solution")
        if solution is None:
            return
        if state.get("protocol") is None:
            self._dwell_assess_message(
                "预测后检测未做：请先在“面形初测”阶段计算检测结果，以确定中频频段和高度单位。", warn=True)
            return
        try:
            protocol = EvaluationProtocol.from_dict(state["protocol"])
            target = AcceptanceTarget.from_dict(state["target"]) if state.get("target") else None
            assessment = assess_solution(
                solution, protocol, float(state.get("dx", 1.0)), float(state.get("dy", 1.0)), target)
        except Exception as exc:
            self._dwell_assess_message(f"预测后检测未完成：{exc}", warn=True)
            return
        assessment["input_fingerprint"] = input_fingerprint(state)
        state["assessment"] = assessment
        state.mark_current("assessment")
        self._dwell_assess_message(short_summary(assessment), warn=assessment["verdict"] == "fail")
        self._dwell_refresh_flow_buttons()
        if hasattr(self, "_refresh_workflow_statuses"):
            self._refresh_workflow_statuses()

    def _dwell_show_assessment(self):
        assessment = self._dwell_state.get("assessment")
        if assessment is None:
            self._dwell_assess_message("还没有检测结果：请先求解驻留时间或预测中频演化。", warn=True)
            return
        try:
            self._main.preview.plot_dwell_assessment(assessment)
        except Exception as exc:
            self._dwell_assess_message(f"二维预览失败：{exc}", warn=True)

    def _dwell_predict_vibration(self):
        state = self._dwell_state
        try:
            plan = state.get("process_plan")
            if not plan or plan.get("process_type") != "conformal_vibration":
                raise ValueError("请先在制定方案阶段选择共形振动并生成候选方案")
            evaluation = state.get("evaluation")
            if evaluation is None:
                raise ValueError("请先完成加工前面形检测")
            target = AcceptanceTarget.from_dict(state["target"]) if state.get("target") else None
            assessment = predict_evolution(
                evaluation, VibrationModel.from_dict(plan["model"]), plan["duration_s"], target)
        except Exception as exc:
            self._dwell_error("共形振动预测失败", exc, self.dw_vib_info)
            return
        assessment["input_fingerprint"] = input_fingerprint(state)
        state["assessment"] = assessment
        state.mark_current("assessment")
        self._dwell_report(short_summary(assessment), self.dw_vib_info)
        self._dwell_show_assessment()
        if hasattr(self, "_refresh_workflow_statuses"):
            self._refresh_workflow_statuses()

    def _dwell_refresh_solve_preview(self):
        """把求解结果同步到左侧 2×2 四面板：
        上排=加工前面形误差（检测区）/ 驻留时间（单位面积），
        下排=仿真去除量 / 加工后残差（检测区）。"""
        solution = self._dwell_state.get("solution")
        if solution is None:
            return
        eva = solution["evaluation_mask"]
        surf = solution["surface_mask"]
        work = solution["residual"] + solution["removal"]
        before = np.full_like(work, np.nan); before[eva] = work[eva]
        removal = np.full_like(work, np.nan); removal[surf] = solution["removal"][surf]
        residual = np.full_like(work, np.nan); residual[eva] = solution["residual"][eva]
        x = solution["x"]; y = solution["y"]
        dx = float(x[1] - x[0]) if len(x) > 1 else 1.0
        dy = float(y[1] - y[0]) if len(y) > 1 else 1.0
        dwell = solution["dwell"] / max(dx * dy, 1e-12)
        try:
            self._main.preview.plot_dwell_solution(
                [before, dwell, removal, residual],
                ["加工前面形误差（检测区）", "驻留时间（单位面积）",
                 "仿真去除量", "加工后残差（检测区）"])
        except Exception as exc:
            self._dwell_report(f"二维预览失败：{exc}", self.dw_solve_info, error=True)

    def _do_dwell_modify(self):
        try:
            solution = self._dwell_state.get("solution")
            if solution is None: raise ValueError("请先完成最小二乘法求解")
            width = self._dwell_float(self.dw_modify_width, "边缘修饰宽度", 0.0)
            adjustment = self._dwell_float(self.dw_edge_adjust, "边缘驻留调整量")
            if width <= 0 or adjustment == 0:
                self._dwell_report("修饰参数为零，驻留时间保持不变。", self.dw_solve_info); return
            dx = self._dwell_float(self.dw_solve_dx, "X间距", 0.0, False)
            dy = self._dwell_float(self.dw_solve_dy, "Y间距", 0.0, False)
            mask = solution["trajectory_mask"]
            inner = DwellTimeEngine.erode_mask(mask, width, dx, dy)
            edge = mask & ~inner
            dwell = solution["dwell"].copy()
            dwell[edge] += adjustment * dx * dy
            minimum = self._dwell_float(self.dw_min_dwell, "最小驻留时间", 0.0)
            maximum = self._dwell_float(self.dw_max_dwell, "最大驻留时间", 0.0)
            dwell[mask] = np.clip(dwell[mask], minimum, maximum)
            solution["dwell"] = dwell
            # MATLAB 版修饰后只重画驻留时间一幅图；这里重算正仿真与残差，四幅图全部更新
            self._dwell_resimulate(solution)
            # 同一 solution 对象已原位修改，重新赋值以统一触发下游轨迹/CNC 失效。
            self._dwell_state["solution"] = solution
            self._dwell_assess_prediction()
            self._dwell_report(
                f"边缘驻留时间修饰完成：调整 {np.count_nonzero(edge)} 个节点，"
                f"修饰后 RMS {solution['before_rms']:.6g} → {solution['after_rms']:.6g}。",
                self.dw_solve_info)
            self._dwell_refresh_solve_preview()
        except Exception as exc: self._dwell_error("驻留时间修饰失败", exc, self.dw_solve_info)

    def _dwell_resimulate(self, solution):
        """驻留时间被修饰后，用当前抛光斑重新正仿真：去除量=驻留⊛去除函数，
        残差=面形−去除量，并更新检测区前后 RMS。"""
        spot = self._dwell_state.get("spot")
        if spot is None:
            return
        dwell = np.where(np.isfinite(solution["dwell"]), solution["dwell"], 0.0)
        removal = DwellTimeEngine.fft_convolve_same(dwell, spot["kernel"])
        work = solution["residual"] + solution["removal"]
        surf = solution["surface_mask"]
        residual = np.full_like(work, np.nan)
        residual[surf] = work[surf] - removal[surf]
        eva = solution["evaluation_mask"]
        before = work[eva]
        after = residual[eva]
        solution.update(
            removal=removal, residual=residual,
            before_rms=float(np.sqrt(np.mean((before - np.mean(before)) ** 2))),
            after_rms=float(np.sqrt(np.mean((after - np.mean(after)) ** 2))))
