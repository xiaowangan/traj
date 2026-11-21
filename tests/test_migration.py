"""Migration regression tests, using self-contained frozen original outputs."""
import os
import subprocess
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
from migration_cases import numerical_results
from polishing.workflow.engine import DwellTimeEngine
from polishing.workflow import trajectory
from polishing.io.resources import EXAMPLE_ROOT, RESOURCE_ROOT


class NumericalMigrationTests(unittest.TestCase):
    def test_original_numerical_baseline(self):
        actual = numerical_results(DwellTimeEngine, trajectory, EXAMPLE_ROOT)
        with np.load(ROOT / "tests/fixtures/original_outputs.npz") as expected:
            self.assertEqual(set(expected.files), set(actual))
            for name, values in actual.items():
                with self.subTest(case=name):
                    np.testing.assert_allclose(values, expected[name], rtol=1e-11,
                                               atol=1e-12, equal_nan=True)

    def test_pure_services_do_not_load_gui(self):
        code = ("import sys; from polishing.workflow.engine import DwellTimeEngine; "
                "from polishing.workflow import trajectory; "
                "assert not any(k.startswith(('PyQt5', 'OCC')) for k in sys.modules)")
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_desktop_entry_event_loop(self):
        code = (
            "import runpy; from pathlib import Path; "
            "from PyQt5.QtWidgets import QApplication; from PyQt5.QtCore import QTimer; "
            "from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg; "
            "FigureCanvasQTAgg.draw = lambda self: None; "
            "FigureCanvasQTAgg.draw_idle = lambda self: None; "
            "from polishing.ui import bootstrap; "
            "from polishing.ui.viewers.occ import PreviewCanvas; "
            "PreviewCanvas._init_occ_widgets = lambda self: None; "
            "bootstrap.verify_license = lambda: (True, 'test'); "
            "app = QApplication([]); QTimer.singleShot(300, app.quit); "
            "runpy.run_path(str(Path('main.py').resolve()), run_name='__main__')"
        )
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT,
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_local_resources_and_license(self):
        from polishing.io import license as verification
        self.assertTrue(Path(verification.LICENSE_FILE).is_relative_to(ROOT))
        self.assertTrue(RESOURCE_ROOT.is_relative_to(ROOT))
        self.assertTrue((RESOURCE_ROOT / "stylesheets/main.css").is_file())

    def test_text_exports_and_xyz_reader(self):
        from polishing.io.trajectory_export import (
            save_trajectory_txt, save_surface_trajectory_txt,
        )
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            folder = Path(folder)
            points = trajectory.generate_planar_raster("R", rect_A=4, rect_B=4)
            for name, exporter in [("planar.txt", save_trajectory_txt),
                                   ("surface.txt", save_surface_trajectory_txt)]:
                path = folder / name
                exporter(points, str(path))
                np.testing.assert_allclose(np.loadtxt(path), points)
            xyz = np.array([[x, y, x+y] for y in [-1, 0, 1] for x in [-1, 0, 1]])
            np.savetxt(folder / "grid.xyz", xyz)
            grid, pixel = DwellTimeEngine.read_grid(str(folder / "grid.xyz"))
            self.assertEqual(grid.shape, (3, 3))
            self.assertEqual(pixel, 1)
            self.assertEqual(grid[2, 2], 2)


class UiMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtGui import QFont
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setStyle("Fusion")
        cls.app.setFont(QFont("Microsoft YaHei", 9))

    def test_occ_backend_bindings(self):
        from polishing.ui.viewers.occ import PreviewCanvas
        bindings = PreviewCanvas._occ_imports()
        self.assertIn("qtViewer3d", bindings)
        self.assertIn("AIS_Triangulation", bindings)
        self.assertIn("gp_Cylinder", bindings)

    def test_initial_settings_and_page_restore(self):
        from polishing.ui.window import MainWindow
        from polishing.ui.viewers.occ import PreviewCanvas
        from polishing.ui.widgets.Icons import get_icon
        from polishing.ui.widgets.StyleSheets import get_stylesheet
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as external_cwd:
            os.chdir(external_cwd)
            try:
                with patch.object(PreviewCanvas, "_init_occ_widgets", lambda self: None):
                    window = MainWindow()
                    window.show()
                    panel = window.stacked_widget
                    # 贴合检查直接取建档的期望曲面（球面 R1000，口径 100），不再手填 R/K/D
                    panel.dw_aperture.setCurrentText("圆口径")
                    panel.dw_model_diameter.setText("100")
                    panel.dw_model_type.setCurrentText("球面")
                    panel.dw_model_r.setText("1000")
                    panel._dwell_generate_model()
                    self.assertEqual(panel._dwell_fit_surface()[3], 100.0)
                    panel.setCurrentIndex(panel.idx_dwell_initial)
                    for attr, value in [("dw_disk_diameter", "10"), ("dw_disk_speed", "60")]:
                        getattr(panel, attr).setText(value)
                    panel._dwell_check_disk_fit()
                    self.assertIsNotNone(panel._dwell_state["disk_fit"])
                    panel.dw_stage.setCurrentText("平滑")
                    panel._dwell_suggest_process()
                    records = panel._dwell_collect_process_records()
                    self.assertEqual(records["material"], "沥青")
                    self.assertEqual(records["feed"], 30)
                    preview = window.preview._surface_data_preview
                    self.assertEqual(preview._axes[0].get_aspect(), "auto")
                    self.assertIn("空间变化率", window.preview._label_dwell_sub_right.text())
                    window.preview.plot_dwell_surface(np.ones((4, 4)), np.ones((4, 4)))
                    self.assertFalse(preview._line_mode)
                    panel.setCurrentIndex(panel.idx_dwell_model)
                    panel.setCurrentIndex(panel.idx_workflow_plan)  # 规划页显示贴合曲线
                    self.assertTrue(preview._line_mode)
                    self.assertEqual(len(preview._colorbars), 0)
                    self.assertFalse(get_icon("icon").isNull())
                    self.assertTrue(get_stylesheet("main"))
                    self.assertEqual(Path.cwd(), Path(external_cwd))
                    self.app.processEvents()
                    panel.dw_disk_diameter.setText("12")
                    self.assertIsNone(panel._dwell_state["disk_fit"])
                    window.close()
            finally:
                os.chdir(original_cwd)

    def test_dwell_pipeline_in_migrated_window(self):
        from polishing.ui.window import MainWindow
        from polishing.ui.viewers.occ import PreviewCanvas
        with patch.object(PreviewCanvas, "_init_occ_widgets", lambda self: None), \
             patch.object(PreviewCanvas, "plot_surface", lambda *args, **kwargs: None):
            window = MainWindow()
            panel = window.stacked_widget
            panel._dwell_error = lambda title, error, label=None: self.fail(f"{title}: {error}")
            yy, xx = np.mgrid[-5:6, -5:6]
            state = panel._dwell_state

            state["surface"] = 0.01 * (xx * xx + yy * yy)
            state["spot"] = DwellTimeEngine.build_spot(
                -np.exp(-(xx * xx + yy * yy) / 4), 1, 1, 1, 10)
            state["process_plan"] = {"process_type": "scanning"}
            panel.dw_solver_method.setCurrentText("最小二乘法")
            panel._do_dwell_least_squares()
            self.assertIsNotNone(state["solution"])
            panel._dwell_generate_trajectory()
            self.assertIsNotNone(state["trajectory"])
            panel._dwell_generate_cnc()
            self.assertIsNotNone(state["cnc"])
            self.assertEqual(state["cnc"]["data"].shape[1], 6)
            self.assertTrue(np.isfinite(state["cnc"]["data"]).all())
            window.close()

    def test_eight_stage_workflow_and_invalidation(self):
        from polishing.workflow.state import WORKFLOW_STAGES
        from polishing.ui.window import MainWindow
        from polishing.ui.viewers.occ import PreviewCanvas
        with patch.object(PreviewCanvas, "_init_occ_widgets", lambda self: None), \
             patch.object(PreviewCanvas, "plot_dwell_evaluation", lambda *args, **kwargs: None):
            window = MainWindow()
            panel = window.stacked_widget
            self.assertEqual(panel.currentIndex(), panel.idx_workflow_import)
            self.assertFalse(window.preview._label_dwell.isHidden())
            self.assertEqual(tuple(window._workflow_actions), WORKFLOW_STAGES)
            self.assertEqual(len(panel._workflow_pages), 8)
            for stage in WORKFLOW_STAGES:
                action = window._workflow_actions[stage]
                self.assertFalse(action.icon().isNull(), stage)
                action.trigger()
                self.assertEqual(panel.currentIndex(), getattr(panel, f"idx_workflow_{stage}"))
                self.assertEqual(panel._workflow_context.active_stage, stage)
                self.assertFalse(window.preview._label_dwell.isHidden())
            for stage in ("remeasure", "validate", "archive"):
                page = panel._workflow_pages[stage]
                labels = [item for item in page.findChildren(type(panel.workflow_evaluation_result))]
                self.assertTrue(any("暂不执行" in label.text() for label in labels))

            state = panel._dwell_state
            state["solution"] = {"old": True}
            state["trajectory"] = {"old": True}
            state["cnc"] = {"old": True}
            state["surface"] = np.ones((8, 8))
            self.assertIsNone(state["solution"])
            self.assertIsNone(state["trajectory"])
            self.assertIsNone(state["cnc"])
            panel.workflow_eval_min.setText("0.125")
            panel.workflow_eval_max.setText("0.5")
            panel._workflow_evaluate_surface()
            self.assertEqual(state["evaluation"]["stats"]["count"], 64)
            self.assertIn("psd", state["evaluation"])
            self.assertEqual(panel._workflow_context.stage_status("evaluate"), "已完成")
            self.app.processEvents()
            window.close()

    def test_first_five_stage_internal_navigation(self):
        from polishing.ui.window import MainWindow
        from polishing.ui.viewers.occ import PreviewCanvas
        with patch.object(PreviewCanvas, "_init_occ_widgets", lambda self: None), \
             patch.object(PreviewCanvas, "plot_dwell_coverage", lambda *args, **kwargs: None):
            window = MainWindow()
            panel = window.stacked_widget
            state = panel._dwell_state
            axis = np.arange(16, dtype=float)
            yy, xx = np.meshgrid(axis, axis, indexing="ij")
            points = np.stack((xx, yy, np.zeros_like(xx)), axis=-1)
            normals = np.zeros_like(points); normals[..., 2] = 1.0
            model = {"points": points, "normals": normals,
                     "mask": np.ones((16, 16), dtype=bool), "dx": 1.0, "dy": 1.0}

            self.assertEqual(window._workflow_actions["import"].text(), "导入工件")
            self.assertEqual(window._workflow_actions["plan"].text(), "制定方案")

            panel.setCurrentIndex(panel.idx_dwell_model)
            self.assertFalse(panel.dw_model_next.isEnabled())
            state["model"] = model
            panel._dwell_refresh_flow_buttons()
            self.assertTrue(panel.dw_model_next.isEnabled())
            panel.dw_model_next.click()
            self.assertEqual(panel.currentIndex(), panel.idx_dwell_surface)

            surface = np.ones((16, 16))
            state.update(surface_raw=surface, surface=np.array(surface, copy=True))
            panel._dwell_refresh_flow_buttons()
            self.assertTrue(panel.dw_surface_next.isEnabled())
            panel.dw_surface_next.click()
            self.assertEqual(panel.currentIndex(), panel.idx_workflow_evaluate)

            panel.workflow_eval_min.setText("0.0625")
            panel.workflow_eval_max.setText("0.5")
            panel._workflow_evaluate_surface()
            self.assertTrue(panel.workflow_evaluate_next.isEnabled())
            panel.workflow_evaluate_next.click()
            self.assertEqual(panel.currentIndex(), panel.idx_workflow_plan)

            panel._workflow_confirm_process_type()
            state["spot"] = {"kernel": np.ones((3, 3)), "impulse": 9.0}
            panel.workflow_coverage_spacing.setText("4")
            panel.workflow_coverage_step.setText("1")
            panel.workflow_coverage_pattern.setCurrentText("螺旋形")
            self.assertTrue(panel.workflow_coverage_direction_row.isHidden())
            panel._workflow_preview_coverage()
            self.assertEqual(state["coverage_preview"]["pattern"], "spiral")
            self.assertGreater(state["coverage_preview"]["point_count"], 2)
            self.assertFalse(panel.workflow_candidate_button.isEnabled())
            panel.workflow_coverage_pattern.setCurrentText("栅形")
            self.assertFalse(panel.workflow_coverage_direction_row.isHidden())
            panel._workflow_preview_coverage()
            self.assertIsNotNone(state["coverage_preview"])
            panel._workflow_confirm_plan()
            self.assertIsNotNone(state["process_plan"])
            self.assertEqual(state["process_plan"]["coverage"]["line_spacing_mm"], 4.0)
            self.assertTrue(panel.workflow_plan_next.isEnabled())
            panel.workflow_plan_next.click()
            self.assertEqual(panel.currentIndex(), panel.idx_workflow_predict)

            panel.setCurrentIndex(panel.idx_dwell_solve)
            state["solution"] = {"ready": True}
            panel._dwell_refresh_flow_buttons()
            # 执行必须先确认方案
            self.assertFalse(panel.dw_solve_next.isEnabled())
            state["frozen_plan"] = {"process_type": "scanning"}
            panel._dwell_refresh_flow_buttons()
            self.assertTrue(panel.dw_solve_next.isEnabled())
            panel.dw_solve_next.click()
            self.assertEqual(panel.currentIndex(), panel.idx_dwell_cnc)

            state["surface"] = np.zeros((16, 16))
            self.assertIsNone(state["process_plan"])
            window.close()

    def test_startup_first_bar_uses_final_width(self):
        from PyQt5.QtWidgets import QWidget
        from polishing.ui.window import MainWindow
        from polishing.ui.viewers.occ import PreviewCanvas
        from polishing.ui.viewers.surface import SurfaceDataPreview

        def lightweight_surface(widget, parent=None):
            QWidget.__init__(widget, parent)
            widget._panel_hosts = []

        with patch.object(PreviewCanvas, "_init_occ_widgets", lambda self: None), \
             patch.object(SurfaceDataPreview, "__init__", lightweight_surface):
            window = MainWindow()
            window.show()
            self.app.processEvents()
            self.app.processEvents()
            banner = window.preview._label_dwell
            self.assertFalse(banner.isHidden())
            self.assertEqual(banner.x(), 0)
            self.assertEqual(banner.width(), window.preview.width())
            self.assertGreater(banner.width(), 500)
            window.close()


if __name__ == "__main__":
    unittest.main()
