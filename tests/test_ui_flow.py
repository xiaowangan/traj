"""按 docs/前五阶段完整演示案例.md 的步骤，通过真实界面方法走完整条链，并验证新增的评价/预测/案例功能。"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from polishing.io.resources import EXAMPLE_ROOT
from polishing.workflow import case_io
from polishing.workflow.engine import DwellTimeEngine

SURFACE_FILE = str(Path(EXAMPLE_ROOT) / "6S-320-250-FQZM-R2289.10-4#-0818-cgh-296-230.dat")
SPOT_FILE = str(Path(EXAMPLE_ROOT) / "spot_20_disk.dat")


class DemoFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtGui import QFont
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setStyle("Fusion")
        cls.app.setFont(QFont("Microsoft YaHei", 9))
        # 测试结束时 Qt 对象先于 matplotlib 的延迟重绘被销毁，会打印无害的报错；渲染由 test 内显式 draw() 覆盖
        cls._draw_idle = FigureCanvasQTAgg.draw_idle
        FigureCanvasQTAgg.draw_idle = lambda self, *args, **kwargs: None

    @classmethod
    def tearDownClass(cls):
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        FigureCanvasQTAgg.draw_idle = cls._draw_idle

    def new_window(self):
        from polishing.ui.viewers.occ import PreviewCanvas
        from polishing.ui.window import MainWindow
        stack = []
        for target, replacement in (("_init_occ_widgets", lambda self: None),
                                    ("plot_dwell_model", lambda *a, **k: None),
                                    ("plot_dwell_coverage", lambda *a, **k: None)):
            patcher = patch.object(PreviewCanvas, target, replacement)
            patcher.start()
            stack.append(patcher)
        self.addCleanup(lambda: [patcher.stop() for patcher in stack])
        window = MainWindow()
        self.addCleanup(window.close)
        panel = window.stacked_widget
        panel._dwell_error = lambda title, error, label=None: self.fail(f"{title}: {error}")
        return window, panel

    def build_model_and_surface(self, panel, lx="296", ly="230"):
        import polishing.ui.features.import_measurement as measurement_page
        panel.dw_model_dx.setText("2")
        panel.dw_model_dy.setText("2")
        panel.dw_aperture.setCurrentText("矩形")
        panel.dw_model_lx.setText(lx)
        panel.dw_model_ly.setText(ly)
        panel.dw_model_corner.setText("0")
        panel.dw_model_type.setCurrentText("球面")
        panel.dw_model_r.setText("2289.10")
        panel.dw_model_k.setText("0")
        panel._dwell_generate_model()
        with patch.object(measurement_page.QFileDialog, "getOpenFileName",
                          lambda *args, **kwargs: (SURFACE_FILE, "")):
            panel._dwell_select_surface()

    def evaluate(self, panel):
        panel.workflow_eval_min.setText("0.02")
        panel.workflow_eval_max.setText("0.20")
        panel._workflow_evaluate_surface()

    def build_scanning_plan(self, panel):
        state = panel._dwell_state
        raw, pixel = DwellTimeEngine.read_grid(SPOT_FILE)
        state.update(spot_raw=raw, spot_pixel=pixel, spot_before=-raw, spot=None)
        panel.dw_spot_duration.setText("60")
        panel._dwell_calculate_spot()
        panel._workflow_confirm_process_type()
        panel.workflow_coverage_spacing.setText("4")
        panel.workflow_coverage_step.setText("4")
        panel._workflow_preview_coverage()
        panel._workflow_confirm_plan()

    def solve(self, panel):
        panel.dw_solver_method.setCurrentText("最小二乘法")
        for attr, value in (("dw_traj_margin", "0"), ("dw_eval_trim", "6"), ("dw_uniform", "0.05"),
                            ("dw_max_dwell", "120"), ("dw_min_dwell", "0.02"),
                            ("dw_solve_iterations", "300"), ("dw_solve_tolerance", "1e-6")):
            getattr(panel, attr).setText(value)
        panel._do_dwell_least_squares()

    def render_all(self, window):
        """强制真正绘制一遍四个面板，保证新增绘图代码不会在渲染时报错。"""
        for canvas in window.preview._surface_data_preview._canvases:
            canvas.draw()

    def test_import_numbers_header_wavelength_and_alignment(self):
        window, panel = self.new_window()
        self.build_model_and_surface(panel)
        state = panel._dwell_state
        stats = DwellTimeEngine.stats(state["surface"])
        self.assertEqual(state["surface"].shape, (117, 149))
        self.assertEqual(stats["count"], 16848)
        self.assertAlmostEqual(stats["pv"], 0.658420, places=6)
        self.assertAlmostEqual(stats["rms"], 0.0638267, places=7)
        self.assertAlmostEqual(state["surface_meta"]["wavelength_nm_header"], 632.8, places=1)
        # Zygo 数据按文件头波长自动换算为 nm，界面不再有单位选项
        self.assertFalse(hasattr(panel, "workflow_eval_unit"))
        self.assertIn("自动换算", panel.workflow_eval_unit_hint.text())
        self.assertTrue(state["alignment"]["ok"], state["alignment"]["warnings"])
        self.assertIn("对齐检查通过", panel.workflow_alignment_label.text())

    def test_band_frequency_and_wavelength_are_linked_both_ways(self):
        window, panel = self.new_window()
        panel.workflow_eval_min.setText("0.02")
        panel.workflow_eval_max.setText("0.2")
        self.assertEqual(panel.workflow_eval_wl_max.text(), "50")
        self.assertEqual(panel.workflow_eval_wl_min.text(), "5")
        panel.workflow_eval_wl_min.setText("2")      # 波长下限 2 mm → 频率上限 0.5 周期/mm
        panel.workflow_eval_wl_max.setText("20")     # 波长上限 20 mm → 频率下限 0.05 周期/mm
        self.assertEqual(panel.workflow_eval_max.text(), "0.5")
        self.assertEqual(panel.workflow_eval_min.text(), "0.05")
        panel.workflow_eval_min.setText("0")         # 频率 0 对应无穷大波长
        self.assertEqual(panel.workflow_eval_wl_max.text(), "")
        panel.workflow_eval_min.setText("")
        self.assertEqual(panel.workflow_eval_wl_max.text(), "")

    def test_resource_page_buttons_show_only_inside_the_plan_flow(self):
        window, panel = self.new_window()
        tool_bar_buttons = (panel.dw_tool_back, panel.dw_tool_next, panel.dw_spot_next)
        # 从工艺资源单独打开：没有流程按钮
        panel.setCurrentIndex(panel.idx_dwell_initial)
        self.assertTrue(all(button.isHidden() or button.parent().isHidden() for button in tool_bar_buttons[:2]))
        self.assertTrue(panel.dw_spot_next.isHidden())
        # 从制定方案第 2 步进入：显示上一步 / 下一步，并能走回方案
        panel.setCurrentIndex(panel.idx_workflow_plan)
        panel._open_resource_from_plan(panel.idx_dwell_initial)
        self.assertFalse(panel.dw_tool_back.parent().isHidden())
        panel.dw_tool_next.click()
        self.assertEqual(panel.currentIndex(), panel.idx_dwell_spot)
        self.assertFalse(panel.dw_spot_next.isHidden())   # 流程内继续显示
        panel.setCurrentIndex(panel.idx_dwell_initial)
        panel.dw_tool_back.click()
        self.assertEqual(panel.currentIndex(), panel.idx_workflow_plan)
        # 再从顶部“工艺资源”单独打开，按钮重新隐藏
        panel.setCurrentIndex(panel.idx_dwell_initial)
        self.assertTrue(panel.dw_tool_back.parent().isHidden())
        panel.setCurrentIndex(panel.idx_dwell_spot)
        self.assertTrue(panel.dw_spot_next.isHidden())

    def test_misaligned_aperture_is_reported_on_import(self):
        window, panel = self.new_window()
        self.build_model_and_surface(panel, lx="100", ly="100")
        report = panel._dwell_state["alignment"]
        self.assertFalse(report["ok"])
        self.assertIn("对齐检查有警告", panel.workflow_alignment_label.text())

    def test_scanning_prediction_is_reevaluated_with_the_evaluation_protocol(self):
        window, panel = self.new_window()
        state = panel._dwell_state
        self.build_model_and_surface(panel)
        # 评价之前不做预测后评价，并明确提示
        self.evaluate(panel)
        self.assertEqual(state["evaluation"]["height_unit"], "nm")
        self.assertEqual(state["protocol"]["height_unit"], "waves")
        self.assertAlmostEqual(state["protocol"]["wavelength_nm"], 632.8)
        self.build_scanning_plan(panel)
        self.assertIsNotNone(state["process_plan"])
        self.solve(panel)

        self.assertAlmostEqual(state["solution"]["before_rms"], 0.0561041, places=6)
        self.assertAlmostEqual(state["solution"]["after_rms"], 0.00133346, places=7)
        assessment = state["assessment"]
        self.assertEqual(assessment["kind"], "scanning")
        before, predicted = assessment["states"]["before"], assessment["states"]["predicted"]
        self.assertLess(predicted["msf_rms"], before["msf_rms"])
        # 预测页评价区（案例网格）与评价页（原始网格）的中频 RMS 应是同一量级
        self.assertAlmostEqual(before["msf_rms"] / state["evaluation"]["band_rms"], 1.0, delta=0.05)
        self.assertEqual(assessment["unit"], "nm")
        self.assertTrue(case_io.is_current(assessment, state))
        self.assertTrue(any("卷积模型" in flag for flag in assessment["flags"]))
        # 预测以“确认方案”为完成
        self.assertEqual(panel._workflow_context.stage_status("predict"), "待完成")
        self.assertTrue(panel._dwell_confirm_plan())
        self.assertEqual(panel._workflow_context.stage_status("predict"), "已完成")

        # 设定加工目标 → 旧预测评价失效；重新求解后给出达标/未达标判定
        panel.workflow_target_msf.setText("1")
        self.assertIsNone(state["assessment"])
        self.solve(panel)
        self.assertEqual(state["assessment"]["verdict"], "pass")
        panel.workflow_target_msf.setText("0.1")
        self.solve(panel)
        self.assertEqual(state["assessment"]["verdict"], "fail")
        self.assertIn("未达标", panel.dw_assess_info.text())

        panel._dwell_show_assessment()
        self.render_all(window)
        preview = window.preview
        self.assertEqual(preview._label_dwell_sub_left.text().strip(), "径向平均 PSD 对比")
        self.assertEqual(preview._label_dwell_sub_right.text().strip(), "分带 RMS 对比")

    def test_prediction_without_protocol_is_refused_politely(self):
        window, panel = self.new_window()
        state = panel._dwell_state
        yy, xx = np.mgrid[-5:6, -5:6]
        state["surface"] = 0.01 * (xx * xx + yy * yy)
        state["spot"] = DwellTimeEngine.build_spot(-np.exp(-(xx * xx + yy * yy) / 4), 1, 1, 1, 10)
        state["process_plan"] = {"process_type": "scanning"}
        panel.dw_solver_method.setCurrentText("最小二乘法")
        panel._do_dwell_least_squares()
        self.assertIsNotNone(state["solution"])
        self.assertIsNone(state["assessment"])
        self.assertIn("请先在“面形初测”阶段", panel.dw_assess_info.text())

    def test_case_save_and_reopen_restores_everything_checkable(self):
        window, panel = self.new_window()
        state = panel._dwell_state
        self.build_model_and_surface(panel)
        self.evaluate(panel)
        panel.workflow_target_msf.setText("1")
        self.build_scanning_plan(panel)
        self.solve(panel)
        self.assertTrue(panel._dwell_confirm_plan())
        saved_fingerprint = case_io.input_fingerprint(state)
        saved_assessment = state["assessment"]
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            path = str(Path(folder) / "demo.polcase")
            panel._case_save_to(path)

            window2, panel2 = self.new_window()
            state2 = panel2._dwell_state
            panel2._case_open_from(path)

        self.assertEqual(case_io.input_fingerprint(state2), saved_fingerprint)
        np.testing.assert_array_equal(state2["surface"], state["surface"])
        self.assertEqual(state2["protocol"], state["protocol"])
        self.assertIsNotNone(state2["evaluation"])
        self.assertAlmostEqual(state2["evaluation"]["band_rms"], state["evaluation"]["band_rms"], places=9)
        self.assertEqual(state2["process_plan"]["process_type"], "scanning")
        self.assertIsNotNone(state2["solution"])
        self.assertEqual(state2["assessment"]["verdict"], saved_assessment["verdict"])
        self.assertTrue(case_io.is_current(state2["assessment"], state2))
        # 界面输入框也被写回，且没有被控件信号清空
        self.assertEqual(panel2.workflow_eval_min.text(), "0.02")
        self.assertEqual(panel2.workflow_eval_wl_max.text(), "50")  # 频率下限 0.02 → 波长上限 50 mm
        self.assertEqual(panel2.workflow_target_msf.text(), "1")
        self.assertEqual(panel2.workflow_coverage_spacing.text(), "4")
        self.assertEqual(float(panel2.dw_model_lx.text()), 296.0)
        self.assertIsNone(state2["trajectory"])  # 轨迹/CNC 不保存
        self.assertEqual(panel2._workflow_context.stage_status("predict"), "已完成")

        # 重新打开后改动任一输入，旧预测按依赖关系失效
        state2["spot"] = dict(state2["spot"])
        self.assertIsNone(state2["assessment"])

    def test_vibration_branch_end_to_end(self):
        window, panel = self.new_window()
        state = panel._dwell_state
        self.build_model_and_surface(panel)
        self.evaluate(panel)
        panel.workflow_process_type.setCurrentIndex(panel.workflow_process_type.findData("conformal_vibration"))
        panel._workflow_confirm_process_type()
        self.assertFalse(panel.workflow_vibration_controls.isHidden())
        self.assertTrue(panel.workflow_scan_controls.isHidden())
        self.assertTrue(panel.workflow_candidate_button.isEnabled())

        # 非法参数不能生成候选方案
        panel.workflow_vib_aniso.setText("2")
        self.assertFalse(panel.workflow_candidate_button.isEnabled())
        self.assertIn("共形振动参数", panel.workflow_plan_readiness.text())
        panel.workflow_vib_aniso.setText("0")

        panel.workflow_vib_beta0.setText("0.004")
        panel.workflow_vib_pnew.setText("0")
        panel.workflow_vib_duration.setText("900")
        panel.workflow_target_msf.setText(f"{state['evaluation']['band_rms'] * 0.25:.6g}")
        panel._workflow_confirm_plan()
        plan = state["process_plan"]
        self.assertEqual(plan["process_type"], "conformal_vibration")
        self.assertFalse(plan["validated"])
        self.assertEqual(plan["calibration_status"], "uncalibrated")

        panel._dwell_predict_vibration()
        assessment = state["assessment"]
        self.assertEqual(assessment["kind"], "conformal_vibration")
        self.assertTrue(assessment["extrapolated"])
        # β 为常数（指数默认 1，在带内不同频点略有差异），达标时间应落在 ln4/β 附近的合理范围
        self.assertIsNotNone(assessment["time_to_target_s"])
        self.assertGreater(assessment["time_to_target_s"], 0)
        self.assertEqual(assessment["verdict"], "pass")
        self.assertTrue(any("未经实验标定" in flag for flag in assessment["flags"]))
        self.assertTrue(panel._dwell_confirm_plan())
        self.assertEqual(panel._workflow_context.stage_status("predict"), "已完成")
        self.render_all(window)

        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            path = str(Path(folder) / "vibration.polcase")
            panel._case_save_to(path)
            window2, panel2 = self.new_window()
            panel2._case_open_from(path)
        state2 = panel2._dwell_state
        self.assertEqual(state2["process_plan"]["process_type"], "conformal_vibration")
        self.assertEqual(panel2.workflow_vib_beta0.text(), "0.004")
        self.assertEqual(panel2.workflow_process_type.currentData(), "conformal_vibration")
        self.assertTrue(case_io.is_current(state2["assessment"], state2))
        self.assertEqual(state2["assessment"]["best_time_s"], assessment["best_time_s"])

    def test_tampered_case_file_is_flagged_on_open(self):
        window, panel = self.new_window()
        self.build_model_and_surface(panel)
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            path = Path(folder) / "demo.polcase"
            panel._case_save_to(str(path))
            inputs, records, manifest = case_io.load_case(str(path))
            inputs["surface"] = inputs["surface"] + 1e-6
            report = case_io.restore_case(panel._dwell_state, inputs, records, manifest)
        self.assertFalse(dict((name, ok) for name, ok, _ in report)["输入完整性"])


if __name__ == "__main__":
    unittest.main()
