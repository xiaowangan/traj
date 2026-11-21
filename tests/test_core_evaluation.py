"""评价可信化、预测—评价打通、共形振动骨架与案例存取的数值验证（合成数据，不依赖界面）。"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from polishing.core import vibration
from polishing.core.alignment import check_alignment
from polishing.core.assessment import (
    assess_solution, check_target, common_roi, evaluate_with_protocol,
    format_comparison, noise_floor)
from polishing.core.model_evaluation import AcceptanceTarget, EvaluationProtocol
from polishing.core.units import height_scale_to_nm
from polishing.workflow import case_io
from polishing.workflow.engine import DwellTimeEngine
from polishing.workflow.state import create_dwell_state


def sine_surface(n, pitch, amplitude, wavelength, angle_deg=0.0):
    """已知幅值、波长和方向的正弦面；波长取整数个周期，避免谱泄漏影响验收。"""
    axis = (np.arange(n) - (n - 1) / 2.0) * pitch
    xx, yy = np.meshgrid(axis, axis)
    angle = np.radians(angle_deg)
    return amplitude * np.cos(2 * np.pi * (xx * np.cos(angle) + yy * np.sin(angle)) / wavelength)


def protocol(band=(0.05, 0.2), **overrides):
    values = dict(band_min=band[0], band_max=band[1], detrend="none", window="none",
                  height_unit="nm")
    values.update(overrides)
    return EvaluationProtocol(**values)


class EvaluationAccuracyTests(unittest.TestCase):
    def test_sine_rms_is_amplitude_over_root_two(self):
        # 128 点 × 1 mm，波长 8 mm → 整 16 个周期落在网格内
        surface = sine_surface(128, 1.0, 10.0, 8.0)
        result = evaluate_with_protocol(surface, 1.0, 1.0, protocol(band=(0.1, 0.15)))
        self.assertAlmostEqual(result["band_rms"], 10.0 / np.sqrt(2), delta=1e-6)

    def test_hann_window_rms_close_to_truth(self):
        surface = sine_surface(256, 1.0, 10.0, 8.0)
        result = evaluate_with_protocol(
            surface, 1.0, 1.0, protocol(band=(0.1, 0.15), window="hann"))
        self.assertAlmostEqual(result["band_rms"] / (10.0 / np.sqrt(2)), 1.0, delta=0.02)

    def test_spectral_peak_frequency_and_direction(self):
        for angle in (0.0, 40.0, 90.0, 130.0):
            with self.subTest(angle=angle):
                surface = sine_surface(128, 1.0, 5.0, 8.0, angle)
                result = evaluate_with_protocol(surface, 1.0, 1.0, protocol(band=(0.05, 0.25)))
                radial = np.asarray(result["radial_psd"])
                peak = result["radial_frequency"][int(np.nanargmax(radial))]
                self.assertAlmostEqual(peak, 1 / 8.0, delta=2 * result["resolution_mm_inv"][0])
                circular = abs(result["dominant_direction_deg"] - angle)
                self.assertLessEqual(min(circular, 180 - circular), 10.0)

    def test_band_decomposition_adds_up(self):
        surface = sine_surface(128, 1.0, 5.0, 8.0) + sine_surface(128, 1.0, 3.0, 40.0, 30.0)
        # 去均值后直流分量为零，低/中/高三带的能量之和才恒等于全谱能量
        result = evaluate_with_protocol(surface, 1.0, 1.0, protocol(band=(0.1, 0.15), detrend="piston"))
        total = result["low_frequency_rms"] ** 2 + result["band_rms"] ** 2 + result["high_frequency_rms"] ** 2
        self.assertAlmostEqual(np.sqrt(total), result["spectral_rms"], places=9)
        # Parseval：全谱能量≈样本标准差（非整数周期的有限样本方差不等于 a²/2；
        # 谱泄漏尾巴有极小一部分落在 Nyquist 圆外的方形角落，故取相对 1e-4）
        self.assertAlmostEqual(result["spectral_rms"] / float(np.std(surface)), 1.0, delta=1e-4)

    def test_unequal_pitch(self):
        axis_x = (np.arange(128) - 63.5) * 1.0
        axis_y = (np.arange(64) - 31.5) * 2.0
        xx, yy = np.meshgrid(axis_x, axis_y)
        surface = 4.0 * np.cos(2 * np.pi * xx / 8.0)
        result = evaluate_with_protocol(surface, 1.0, 2.0, protocol(band=(0.1, 0.15)))
        self.assertAlmostEqual(result["band_rms"], 4.0 / np.sqrt(2), delta=1e-6)

    def test_circular_aperture_uses_inscribed_rectangle(self):
        axis = (np.arange(160) - 79.5) * 1.0
        xx, yy = np.meshgrid(axis, axis)
        surface = 6.0 * np.cos(2 * np.pi * xx / 8.0)
        surface[np.hypot(xx, yy) > 70] = np.nan
        # 内接矩形约 99 点，不是整数个周期；矩形窗会有谱泄漏，加 Hann 窗后应接近真值
        result = evaluate_with_protocol(
            surface, 1.0, 1.0, protocol(band=(0.1, 0.15), window="hann"))
        self.assertLess(result["valid_fraction"], 1.0)
        self.assertLess(result["roi_shape"][0], 160)
        self.assertAlmostEqual(result["band_rms"] / (6.0 / np.sqrt(2)), 1.0, delta=0.02)


class UnitContractTests(unittest.TestCase):
    def test_same_physical_surface_gives_same_metrics_in_any_unit(self):
        nanometres = sine_surface(96, 1.0, 12.0, 8.0)
        reference = evaluate_with_protocol(nanometres, 1.0, 1.0, protocol(band=(0.1, 0.15)))
        cases = (("um", nanometres / 1e3, None), ("mm", nanometres / 1e6, None),
                 ("waves", nanometres / 632.8, 632.8))
        for unit, data, wavelength in cases:
            with self.subTest(unit=unit):
                result = evaluate_with_protocol(
                    data, 1.0, 1.0, protocol(band=(0.1, 0.15), height_unit=unit, wavelength_nm=wavelength))
                self.assertAlmostEqual(result["band_rms"], reference["band_rms"], places=9)
                self.assertEqual(result["height_unit"], "nm")

    def test_unconfirmed_unit_is_reported_and_waves_need_wavelength(self):
        scale, label = height_scale_to_nm("raw")
        self.assertEqual((scale, label), (1.0, "未知高度单位"))
        with self.assertRaises(ValueError):
            height_scale_to_nm("waves")
        with self.assertRaises(ValueError):
            protocol(height_unit="waves").validate()

    def test_zygo_header_reports_wavelength(self):
        sample = ROOT / "data/examples"
        files = sorted(sample.glob("*.dat"))
        self.assertTrue(files)
        for path in files:
            self.assertAlmostEqual(
                DwellTimeEngine.read_zygo_header(str(path))["wavelength_nm"], 632.8, places=2)


class NoiseFloorAndRoiTests(unittest.TestCase):
    def test_noise_floor_from_repeat_measurements(self):
        rng = np.random.default_rng(7)
        truth = sine_surface(128, 1.0, 20.0, 8.0)
        sigma = 2.0
        a = truth + rng.normal(0, sigma, truth.shape)
        b = truth + rng.normal(0, sigma, truth.shape)
        band = (0.1, 0.2)
        floor = noise_floor(a, b, 1.0, 1.0, protocol(band=band))
        # 白噪声的带内方差 = σ² × 频带面积 / 总频域面积（正方形 Nyquist 域，面积 1/(dx·dy)）
        expected = sigma * np.sqrt(np.pi * (band[1] ** 2 - band[0] ** 2))
        self.assertAlmostEqual(floor["msf_rms"] / expected, 1.0, delta=0.05)
        self.assertAlmostEqual(floor["rms"] / sigma, 1.0, delta=0.05)

    def test_pinned_roi_gives_identical_samples(self):
        before = sine_surface(64, 1.0, 5.0, 8.0)
        after = before * 0.5
        after[:10] = np.nan
        roi = common_roi(before, after)
        res_a = evaluate_with_protocol(before, 1.0, 1.0, protocol(), roi=roi)
        res_b = evaluate_with_protocol(after, 1.0, 1.0, protocol(), roi=roi)
        self.assertEqual(res_a["roi"], res_b["roi"])
        self.assertAlmostEqual(res_b["band_rms"], 0.5 * res_a["band_rms"], places=9)
        with self.assertRaises(ValueError):
            evaluate_with_protocol(after, 1.0, 1.0, protocol(), roi=(0, 64, 0, 64))


def solved_case(ripple_amplitude=3.0):
    """带中频波纹的碗形面 + 高斯 TIF，求出真实的驻留解。"""
    yy, xx = np.mgrid[-40:41, -40:41].astype(float)
    surface = 0.002 * (xx ** 2 + yy ** 2) + ripple_amplitude * np.cos(2 * np.pi * xx / 8.0)
    tif = DwellTimeEngine.build_spot(
        -np.exp(-(xx[34:47, 34:47] ** 2 + yy[34:47, 34:47] ** 2) / 8.0), 1, 1, 1, 10)
    solution = DwellTimeEngine.solve_bounded_least_squares(
        surface, tif["kernel"], 1.0, 1.0, uniform=0.5, min_dwell=0.0, max_dwell=60.0,
        trajectory_trim=0.0, evaluation_trim=6.0, iterations=200, tolerance=1e-7)
    return surface, solution


class PredictionAssessmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.surface, cls.solution = solved_case()

    def test_assessment_uses_same_protocol_roi_and_matches_direct_evaluation(self):
        proto = protocol(band=(0.08, 0.2), detrend="plane", window="hann")
        assessment = assess_solution(self.solution, proto, 1.0, 1.0)
        mask = self.solution["evaluation_mask"]
        work = np.where(mask, self.solution["residual"] + self.solution["removal"], np.nan)
        direct = evaluate_with_protocol(work, 1.0, 1.0, proto, roi=assessment["roi"])
        self.assertAlmostEqual(assessment["states"]["before"]["msf_rms"], direct["band_rms"], places=10)
        self.assertLess(assessment["states"]["predicted"]["msf_rms"],
                        assessment["states"]["before"]["msf_rms"])
        self.assertEqual(assessment["status"], "predicted")
        self.assertGreater(assessment["dwell_total_s"], 0)
        self.assertIn("evaluation", assessment["regions_mm2"])
        self.assertLessEqual(assessment["regions_mm2"]["evaluation"], assessment["regions_mm2"]["solver_surface"])

    def test_target_checks_and_verdict(self):
        proto = protocol(band=(0.08, 0.2), detrend="plane")
        loose = assess_solution(self.solution, proto, 1.0, 1.0, AcceptanceTarget(msf_rms_max=1e6))
        tight = assess_solution(self.solution, proto, 1.0, 1.0, AcceptanceTarget(msf_rms_max=1e-9))
        none = assess_solution(self.solution, proto, 1.0, 1.0, None)
        self.assertEqual((loose["verdict"], tight["verdict"], none["verdict"]),
                         ("pass", "fail", "no_target"))
        slow = assess_solution(self.solution, proto, 1.0, 1.0, AcceptanceTarget(time_max_s=1e-3))
        self.assertEqual(slow["verdict"], "fail")
        self.assertTrue(any("时间" in item["name"] for item in slow["checks"]))
        self.assertTrue(any("未设定加工目标" in flag for flag in none["flags"]))

    def test_three_way_comparison_with_measured_state(self):
        proto = protocol(band=(0.08, 0.2), detrend="plane")
        measured = np.where(self.solution["evaluation_mask"], self.solution["residual"] * 1.5, np.nan)
        measured = np.nan_to_num(measured, nan=0.0) + np.where(self.solution["evaluation_mask"], 0, np.nan)
        assessment = assess_solution(self.solution, proto, 1.0, 1.0,
                                     AcceptanceTarget(msf_rms_max=1e6), measured_after=measured)
        self.assertEqual(assessment["status"], "measured")
        states = assessment["states"]
        self.assertGreater(states["measured"]["msf_rms"], states["predicted"]["msf_rms"])
        text = format_comparison(assessment)
        for label in ("加工前", "预测后", "实测后", "判定（依据实测）"):
            self.assertIn(label, text)
        with self.assertRaises(ValueError):
            assess_solution(self.solution, proto, 1.0, 1.0, measured_after=np.zeros((3, 3)))

    def test_band_above_nyquist_is_refused_not_faked(self):
        with self.assertRaises(ValueError):
            assess_solution(self.solution, protocol(band=(0.1, 0.9)), 1.0, 1.0)

    def test_unknown_unit_is_flagged(self):
        assessment = assess_solution(self.solution, protocol(band=(0.08, 0.2), height_unit="raw"), 1.0, 1.0)
        self.assertTrue(any("高度单位未确认" in flag for flag in assessment["flags"]))
        self.assertEqual(assessment["unit"], "未知高度单位")

    def test_check_target_ignores_unset_limits(self):
        metrics = {"msf_rms": 1.0, "rms": 2.0}
        self.assertEqual(check_target(metrics, AcceptanceTarget(rms_max=3.0)),
                         [{"name": "去低阶 RMS", "value": 2.0, "limit": 3.0, "ok": True}])
        self.assertEqual(check_target(metrics, None), [])


class AlignmentTests(unittest.TestCase):
    def make_model(self, aperture="矩形", lx=40.0, ly=30.0):
        return DwellTimeEngine.generate_model({
            "type": "平面", "aperture": aperture, "dx": 1.0, "dy": 1.0, "lx": lx, "ly": ly,
            "diameter": lx, "r": 0.0, "k": 0.0, "a": ()})

    def test_matching_surface_is_ok(self):
        model = self.make_model()
        surface = np.where(model["mask"], 1.0, np.nan)
        report = check_alignment(model, surface, 1.0, 1.0)
        self.assertTrue(report["ok"], report["warnings"])
        self.assertAlmostEqual(report["aperture_coverage"], 1.0)

    def test_shifted_undersized_and_outside_data_are_reported(self):
        model = self.make_model()
        small = np.full((20, 20), 1.0)
        report = check_alignment(model, small, 1.0, 1.0)
        self.assertFalse(report["ok"])
        self.assertTrue(any("覆盖" in text or "相差" in text for text in report["warnings"]))
        large = np.ones((90, 90))
        report = check_alignment(model, large, 1.0, 1.0)
        self.assertTrue(any("落在期望口径之外" in text for text in report["warnings"]))
        shifted = np.full(model["mask"].shape, np.nan)
        shifted[:, 12:] = 1.0
        shifted = np.where(model["mask"], shifted, np.nan)
        report = check_alignment(model, shifted, 1.0, 1.0)
        self.assertTrue(any("偏移" in text for text in report["warnings"]))


class VibrationSkeletonTests(unittest.TestCase):
    def evaluation(self, amplitude=10.0, wavelength=8.0):
        surface = sine_surface(128, 1.0, amplitude, wavelength)
        return evaluate_with_protocol(surface, 1.0, 1.0, protocol(band=(0.1, 0.15)))

    def test_zero_beta_and_no_new_error_changes_nothing(self):
        evaluation = self.evaluation()
        model = vibration.VibrationModel(beta0=0.0)
        result = vibration.predict_evolution(evaluation, model, 100.0)
        self.assertAlmostEqual(result["states"]["predicted"]["msf_rms"], evaluation["band_rms"], places=9)

    def test_psd_metrics_match_evaluate_surface(self):
        evaluation = self.evaluation()
        metrics = vibration.psd_metrics(
            evaluation["psd"], evaluation["qx"], evaluation["qy"],
            evaluation["band_min_mm_inv"], evaluation["band_max_mm_inv"], evaluation["nyquist_mm_inv"])
        self.assertAlmostEqual(metrics["msf_rms"], evaluation["band_rms"], places=10)
        self.assertAlmostEqual(metrics["low_rms"], evaluation["low_frequency_rms"], places=10)
        self.assertAlmostEqual(metrics["high_rms"], evaluation["high_frequency_rms"], places=10)
        self.assertAlmostEqual(metrics["rms"], evaluation["spectral_rms"], places=10)

    def test_single_ripple_decays_as_exp_minus_beta_t(self):
        evaluation = self.evaluation(10.0)
        beta = 0.004
        model = vibration.VibrationModel(beta0=beta, exponent=0.0)
        for t in (50.0, 200.0, 600.0):
            decayed = vibration.psd_metrics(
                vibration.evolve_psd(evaluation["psd"], evaluation["qx"], evaluation["qy"], model, t),
                evaluation["qx"], evaluation["qy"], 0.1, 0.15, evaluation["nyquist_mm_inv"])
            self.assertAlmostEqual(decayed["msf_rms"] / evaluation["band_rms"], np.exp(-beta * t), places=9)

    def test_new_error_makes_curve_turn_up_and_is_reported(self):
        evaluation = self.evaluation(10.0)
        model = vibration.VibrationModel(beta0=0.01, exponent=0.0, p_new_rate=0.5)
        result = vibration.predict_evolution(evaluation, model, 2000.0)
        self.assertGreater(result["best_time_s"], 0)
        self.assertLess(result["best_time_s"], 2000.0)
        self.assertGreater(result["msf_rms_curve"][-1], result["best_msf_rms"])
        self.assertTrue(any("回升" in flag for flag in result["flags"]))
        self.assertTrue(result["extrapolated"])
        self.assertEqual(result["calibration"], "uncalibrated")

    def test_time_to_target_interpolation_and_unreachable_target(self):
        evaluation = self.evaluation(10.0)
        beta = 0.004
        model = vibration.VibrationModel(beta0=beta, exponent=0.0)
        goal = evaluation["band_rms"] * 0.25
        result = vibration.predict_evolution(
            evaluation, model, 1000.0, AcceptanceTarget(msf_rms_max=goal), steps=401)
        self.assertAlmostEqual(result["time_to_target_s"], np.log(4) / beta, delta=2.0)
        self.assertEqual(result["verdict"], "pass")
        impossible = vibration.predict_evolution(
            evaluation, model, 1000.0, AcceptanceTarget(msf_rms_max=1e-9))
        self.assertIsNone(impossible["time_to_target_s"])
        self.assertEqual(impossible["verdict"], "fail")

    def test_vibration_does_not_pretend_to_predict_total_rms(self):
        evaluation = self.evaluation()
        model = vibration.VibrationModel(beta0=0.004)
        result = vibration.predict_evolution(
            evaluation, model, 300.0, AcceptanceTarget(rms_max=1.0, msf_rms_max=1e6))
        self.assertIsNone(result["states"]["predicted"]["rms"])
        self.assertEqual([item["name"] for item in result["checks"]], ["中频 RMS"])
        self.assertTrue(any("未判定" in flag for flag in result["flags"]))
        self.assertIn("—", format_comparison(result))

    def test_direction_dependent_beta(self):
        model = vibration.VibrationModel(beta0=0.01, anisotropy=1.0, direction_deg=0.0)
        along = model.beta(0.1, 0.0)
        across = model.beta(0.1, np.pi / 2)
        self.assertAlmostEqual(float(along), 0.01)
        self.assertAlmostEqual(float(across), 0.0)

    def test_fit_beta_recovers_truth_and_respects_noise_floor(self):
        rng = np.random.default_rng(3)
        beta, a0 = 0.003, 10.0
        times = np.array([0, 100, 200, 400, 800, 1600], dtype=float)
        amplitude = a0 * np.exp(-beta * times) * (1 + rng.normal(0, 0.01, times.size))
        fit = vibration.fit_beta(times, amplitude)
        self.assertAlmostEqual(fit["beta"], beta, delta=0.0002)
        self.assertGreater(fit["r2"], 0.99)
        floor = float(np.exp(-beta * 1600) * a0 * 1.05)
        trimmed = vibration.fit_beta(times, amplitude, noise_floor=floor)
        self.assertLess(trimmed["n_points"], fit["n_points"])
        with self.assertRaises(ValueError):
            vibration.fit_beta(times, amplitude, noise_floor=1e9)

    def test_fit_beta_from_evolved_surface_round_trip(self):
        # 空间域真实衰减 → 评价 → 辨识，闭环应还原 β
        beta = 0.002
        times = np.array([0, 150, 300, 600, 1200], dtype=float)
        measured = []
        for t in times:
            surface = sine_surface(128, 1.0, 10.0 * np.exp(-beta * t), 8.0)
            measured.append(evaluate_with_protocol(
                surface, 1.0, 1.0, protocol(band=(0.1, 0.15)))["band_rms"])
        self.assertAlmostEqual(vibration.fit_beta(times, measured)["beta"], beta, places=9)


class CaseIoTests(unittest.TestCase):
    def build_state(self):
        state = create_dwell_state()
        state.update(dx=2.0, dy=2.0)
        config = {"type": "球面", "aperture": "矩形", "dx": 2.0, "dy": 2.0, "lx": 60.0, "ly": 40.0,
                  "diameter": 0.0, "r": 2289.1, "k": 0.0, "a": [0.0, 0.0, 0.0]}
        state["model"] = DwellTimeEngine.generate_model(config)
        axis = np.arange(-15, 16, dtype=float)
        yy, xx = np.meshgrid(axis * 2, axis * 2, indexing="ij")
        surface = 0.01 * (xx ** 2 + yy ** 2) + 3 * np.cos(2 * np.pi * xx / 16)
        state.update(surface_raw=surface, surface_pixel=2.0, surface=surface.copy(),
                     surface_meta={"file": "demo.dat", "wavelength_nm_header": 632.8})
        state["spot"] = {"kernel": np.exp(-(np.mgrid[-4:5, -4:5] ** 2).sum(axis=0) / 8.0),
                         "impulse": 1.0, "dx": 2.0, "dy": 2.0, "volume_efficiency": 0.1}
        proto = protocol(band=(0.02, 0.2), detrend="plane", window="hann", height_unit="nm")
        state["protocol"] = proto.to_dict()
        state["target"] = AcceptanceTarget(msf_rms_max=0.5).to_dict()
        return state

    def test_round_trip_restores_inputs_and_flags_stale_predictions(self):
        state = self.build_state()
        state["process_plan"] = {"process_type": "scanning", "tif_shape": (9, 9)}
        record = {"kind": "scanning", "verdict": "pass", "input_fingerprint": case_io.input_fingerprint(state)}
        state["assessment"] = record
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            path = Path(folder) / "case.polcase"
            case_io.save_case(state, str(path), name="demo")
            inputs, records, manifest = case_io.load_case(str(path))
            fresh = create_dwell_state()
            report = case_io.restore_case(fresh, inputs, records, manifest)
        self.assertTrue(all(ok for _, ok, _ in report), report)
        np.testing.assert_array_equal(fresh["surface"], state["surface"])
        np.testing.assert_array_equal(fresh["spot"]["kernel"], state["spot"]["kernel"])
        self.assertEqual(fresh["protocol"], state["protocol"])
        self.assertEqual(fresh["model"]["mask"].shape, state["model"]["mask"].shape)
        self.assertIsNotNone(fresh["evaluation"])
        self.assertIsNotNone(fresh["alignment"])
        self.assertEqual(fresh["process_plan"]["process_type"], "scanning")
        self.assertTrue(case_io.is_current(fresh["assessment"], fresh))
        # 输入改变后，旧预测被依赖关系清空，且指纹不再匹配
        stale = dict(record)
        fresh["surface"] = fresh["surface"] + 1.0
        self.assertIsNone(fresh["assessment"])
        self.assertFalse(case_io.is_current(stale, fresh))

    def test_tampered_input_is_detected(self):
        state = self.build_state()
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            path = Path(folder) / "case.polcase"
            manifest = case_io.save_case(state, str(path))
            inputs, records, manifest = case_io.load_case(str(path))
            inputs["surface"] = inputs["surface"] + 1e-6
            report = case_io.restore_case(create_dwell_state(), inputs, records, manifest)
        self.assertFalse(dict((name, ok) for name, ok, _ in report)["输入完整性"])

    def test_non_finite_values_survive_round_trip(self):
        state = self.build_state()
        state["surface"] = state["surface"].copy()
        state["surface"][0, 0] = np.nan
        state["process_plan"] = {"x": float("inf"), "n": np.int64(3)}
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            path = Path(folder) / "case.polcase"
            case_io.save_case(state, str(path))
            inputs, records, manifest = case_io.load_case(str(path))
        self.assertTrue(np.isnan(inputs["surface"][0, 0]))
        self.assertEqual(records["process_plan"], {"x": float("inf"), "n": 3})

    def test_foreign_file_is_rejected(self):
        import zipfile
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as folder:
            path = Path(folder) / "bad.polcase"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("case.json", '{"format": "other"}')
                archive.writestr("arrays.npz", b"")
            with self.assertRaises(Exception):
                case_io.load_case(str(path))


if __name__ == "__main__":
    unittest.main()
