"""Deterministic cases for comparison against the pre-migration numerical baseline."""
import numpy as np


def numerical_results(engine, trajectory, example_root):
    results = {}

    def record(prefix, value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key != "config":
                    record(f"{prefix}.{key}", item)
        elif isinstance(value, (tuple, list)) and value and isinstance(value[0], np.ndarray):
            for index, item in enumerate(value):
                record(f"{prefix}.{index}", item)
        elif value is not None:
            array = np.asarray(value)
            if array.dtype.kind in "biufc":
                results[prefix] = array

    data = np.arange(81, dtype=float).reshape(9, 9) / 100
    data[0, :] = np.nan
    data[3, 3] = np.nan
    record("stats", engine.stats(data))
    for name, args in [
        ("trim_surface", (data, 1, 1, 1)),
        ("mean_filter", (data, 3, 1, 1)),
        ("rms_filter", (data, 1.5)),
        ("nearest_fill", (data,)),
        ("extend_surface", (data, 2, 1, 1)),
        ("resample_grid", (data, 1, 1, 0.5, 0.5)),
        ("rotate_vectors", (np.array([[0., 0., 1.], [1., 0., 0.]]),
                            np.array([0., 1., 0.]), 0.5, np.sqrt(0.75))),
    ]:
        record(name, getattr(engine, name)(*args))
    for radius in [0, 500, -500]:
        record(f"fit.{radius}", engine.disk_fit(radius, -0.5, [1e-9, 0, 0], 20, 5))

    model = None
    for index, config in enumerate([
        {"type": "平面", "aperture": "圆口径", "diameter": 10},
        {"type": "非球面", "aperture": "圆口径", "diameter": 10,
         "r": 500, "k": -0.5, "a": [1e-8], "x_offset": 2, "tilt": 5},
        {"type": "柱面", "aperture": "矩形", "lx": 10, "ly": 8,
         "rx": 500, "ry": 0, "corner": 1},
        {"type": "自由曲面", "aperture": "正多边形", "diameter": 10,
         "rx": 500, "ry": 600, "axy": [1e-3, 2e-3, 1e-5]},
    ]):
        model = engine.generate_model(dict(config, dx=1, dy=1))
        record(f"model.{index}", model)
    record("sample_nearest", engine.sample_nearest(model, [0., 1.], [0., 1.]))

    yy, xx = np.mgrid[-4:5, -4:5]
    raw_spot = -np.exp(-(xx * xx + yy * yy) / 4)
    spot = engine.build_spot(raw_spot, 1, 1, 1, 10, 0.5, -0.5)
    record("spot", spot)
    surface = 0.05 * (xx * xx + yy * yy) + 0.01 * xx
    surface[0, :] = np.nan
    kernel = np.array([[0., 0.1, 0.], [0.2, 0.6, 0.1], [0., 0.05, 0.]])
    solution = engine.solve_bounded_least_squares(
        surface, kernel, 1, 1, uniform=0.1, max_dwell=20,
        trajectory_trim=1, evaluation_trim=1, iterations=40)
    record("solution", solution)
    samples = engine.raster_from_dwell(solution, 1, 2, "X", "左下")
    record("dwell_raster", samples)
    flat_xyz = np.column_stack((samples[:, :2], np.zeros(len(samples))))
    normals = np.tile([0., 0., 1.], (len(samples), 1))
    record("cnc", engine.generate_cnc(samples, flat_xyz, normals,
                                     {"min_speed": 0.01, "max_speed": 20}, 1, 1))
    for filename in ["1300-145-0.3斑1.dat", "spot_20_disk.dat"]:
        record(f"dat.{filename}", engine.read_grid(str(example_root / filename)))

    cases = [
        ("generate_planar_raster", {"shape": "R", "rect_A": 10, "rect_B": 8}),
        ("generate_planar_spiral", {"shape": "C", "circle_R": 10,
                                    "spiral_R_max": 10, "pitch": 2}),
        ("generate_aspherical", {"R": 500, "full_width": 10, "full_length": 8}),
        ("generate_spherical", {"R": 20, "h": 5}),
        ("generate_spherical", {"R": 20, "h": 5, "surf_type": "concave",
                                 "traj_type": "S", "pitch": 2}),
        ("generate_cylindrical", {"R": 20, "axis_min": -5, "axis_max": 5}),
        ("generate_cylindrical", {"R": 20, "axis_min": -5, "axis_max": 5,
                                   "surf_type": "V", "traj_type": "S", "pitch": 2}),
        ("generate_conical", {"H": 5, "alpha_deg": 30}),
    ]
    for index, (name, config) in enumerate(cases):
        record(f"trajectory.{index}.{name}", getattr(trajectory, name)(**config))
    return results
