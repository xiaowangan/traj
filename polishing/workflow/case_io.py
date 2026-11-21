"""加工案例的保存与重新打开（单个 .polcase 压缩文件，不依赖 Qt）。

保存的是“输入 + 冻结的预测记录”，不是所有中间产物：
- 输入：网格、期望曲面参数、实测面形、TIF、检测标准、加工目标、制定方案选择；
- 记录：候选方案、预测检测（assessment）及其对应的输入指纹；
- 不保存可由输入重新算出的东西（曲面网格、检测 PSD、轨迹、CNC），打开后按需重算。

每条记录带“输入指纹”。打开案例或使用旧预测时，只要当前输入的指纹与记录不一致，就说明预测已经过期。
"""
from __future__ import annotations
import hashlib
import io
import json
import zipfile
import numpy as np

CASE_FORMAT = "polishing-case"
CASE_VERSION = 1
# 严格按上游到下游排序：先写上游，下游产物才不会被依赖关系清空。
_INPUT_KEYS = ("dx", "dy", "surface_raw", "surface_pixel", "surface", "surface_meta",
               "spot_raw", "spot_pixel", "spot_before", "spot", "tool", "process_records",
               "protocol", "target")
_RECORD_KEYS = ("coverage_preview", "process_plan", "solution", "assessment", "frozen_plan")
_FINGERPRINT_KEYS = ("dx", "dy", "model_config", "surface_raw", "surface_pixel", "surface",
                     "spot_raw", "spot", "protocol", "target")


def _to_plain(value, arrays, path):
    """把状态值转成可写入 JSON 的结构，大数组放进 arrays 并留下引用。"""
    if isinstance(value, np.ndarray):
        arrays[path] = value
        return {"__array__": path}
    if isinstance(value, dict):
        return {str(key): _to_plain(item, arrays, f"{path}/{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item, arrays, f"{path}/{index}") for index, item in enumerate(value)]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return {"__float__": repr(value)}
    return value


def _from_plain(value, arrays):
    if isinstance(value, dict):
        if set(value) == {"__array__"}:
            return arrays[value["__array__"]]
        if set(value) == {"__float__"}:
            return float(value["__float__"])
        return {key: _from_plain(item, arrays) for key, item in value.items()}
    if isinstance(value, list):
        return [_from_plain(item, arrays) for item in value]
    return value


def _digest_update(digest, value):
    if isinstance(value, np.ndarray):
        digest.update(str(value.shape).encode() + str(value.dtype).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, dict):
        for key in sorted(value):
            digest.update(str(key).encode())
            _digest_update(digest, value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            _digest_update(digest, item)
    else:
        digest.update(json.dumps(value, sort_keys=True, default=repr).encode())


def input_fingerprint(state):
    """当前案例输入的 SHA-256 指纹；任何输入变化都会改变它。"""
    digest = hashlib.sha256()
    model = state.get("model")
    parts = dict((key, state.get(key)) for key in _FINGERPRINT_KEYS if key != "model_config")
    parts["model_config"] = None if model is None else model.get("config")
    _digest_update(digest, parts)
    return digest.hexdigest()


def is_current(record, state):
    """记录（如 assessment）是否仍对应当前输入。"""
    return bool(record) and record.get("input_fingerprint") == input_fingerprint(state)


def save_case(state, path, name=""):
    """把案例写成 .polcase；返回写入的清单。"""
    arrays, inputs, records = {}, {}, {}
    for key in _INPUT_KEYS:
        if state.get(key) is not None:
            inputs[key] = _to_plain(state[key], arrays, f"inputs/{key}")
    model = state.get("model")
    if model is not None:
        inputs["model_config"] = _to_plain(model["config"], arrays, "inputs/model_config")
    for key in _RECORD_KEYS:
        if state.get(key) is not None:
            records[key] = _to_plain(state[key], arrays, f"records/{key}")
    manifest = {
        "format": CASE_FORMAT, "version": CASE_VERSION, "name": name,
        "input_fingerprint": input_fingerprint(state),
        "inputs": inputs, "records": records,
        "not_saved": ["model", "evaluation", "alignment", "trajectory", "cnc"],
    }
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **{key.replace("/", "|"): value for key, value in arrays.items()})
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("case.json", json.dumps(manifest, ensure_ascii=False, indent=1))
        archive.writestr("arrays.npz", buffer.getvalue())
    return manifest


def load_case(path):
    """读取 .polcase，返回 (inputs, records, manifest)；指纹不符时抛出 ValueError。"""
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("case.json").decode("utf-8"))
        with np.load(io.BytesIO(archive.read("arrays.npz"))) as stored:
            arrays = {key.replace("|", "/"): stored[key] for key in stored.files}
    if manifest.get("format") != CASE_FORMAT:
        raise ValueError("不是本软件的案例文件")
    if manifest.get("version", 0) > CASE_VERSION:
        raise ValueError("案例文件版本高于当前软件，请升级后再打开")
    inputs = {key: _from_plain(value, arrays) for key, value in manifest["inputs"].items()}
    records = {key: _from_plain(value, arrays) for key, value in manifest["records"].items()}
    return inputs, records, manifest


def restore_case(state, inputs, records, manifest):
    """按上游到下游的顺序写回状态；返回逐条检查结果 [(名称, 是否通过, 说明)]。"""
    from polishing.workflow.engine import DwellTimeEngine
    from polishing.workflow.evaluation_service import (
        check_current_alignment, evaluate_current_surface)
    from polishing.core.model_evaluation import EvaluationProtocol

    report = []
    for key in _INPUT_KEYS:
        if key in inputs:
            state[key] = inputs[key]
    if "model_config" in inputs:
        state["model"] = DwellTimeEngine.generate_model(inputs["model_config"])
    # 恢复后重新计算指纹：与保存时一致才说明数据完整、未被改动。
    restored = input_fingerprint(state)
    matches = restored == manifest.get("input_fingerprint")
    report.append(("输入完整性", matches, "输入数据与保存时一致" if matches
                   else "输入指纹与保存时不一致，文件可能损坏或被改动"))
    # 检测和对齐检查可由输入重算；必须先于方案/预测写入，否则会按依赖关系清空它们。
    try:
        state["alignment"] = check_current_alignment(state)
        if state.get("protocol") is not None and state.get("surface") is not None:
            state["evaluation"] = evaluate_current_surface(
                state, EvaluationProtocol.from_dict(state["protocol"]))
    except ValueError as exc:
        report.append(("重新检测", False, f"按保存的检测标准重新检测失败：{exc}"))
    for key in _RECORD_KEYS:
        if key in records:
            state[key] = records[key]
    record = records.get("assessment")
    if record is not None:
        current = is_current(record, state)
        report.append(("预测记录", current, "预测记录与当前输入一致" if current
                       else "预测记录对应的输入已变化，视为过期"))
    return report
