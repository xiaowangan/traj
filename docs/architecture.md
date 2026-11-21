# 代码结构、模块地图与修改入口

本项目采用四个代码目录，保留八阶段业务入口并把底层计算放到同一处，减少跨目录跳转。

```text
main.py
polishing/
  core/       曲面建模、测量、工艺、轨迹规划、驻留与仿真
  workflow/   八阶段状态、计算门面、流程使用的轨迹入口
  io/         数据读取、PATH/TXT 输出、授权、资源路径
  ui/         features/ 八阶段功能页；widgets/ 控件；viewers/ 预览；窗口
resources/   图标和样式
data/        样例数据
tests/       迁移基线与检查（89 组旧版输出）
docs/        用户流程和技术说明
```

## 从哪里修改

| 要修改的功能 | 位置 |
|---|---|
| 建档、曲面和加工前测量 | `polishing/ui/features/import_geometry.py`、`import_measurement.py` |
| 面形评价 | `polishing/ui/features/evaluation.py`；指标和处理在 `polishing/core/` |
| 规划与工具/TIF 资源 | `polishing/ui/features/plan_coverage.py`、`plan_tif.py`、`resource_tool.py`、`resource_tif.py` |
| 几何曲面轨迹算法 | `polishing/core/trajectory_*.py` |
| 驻留预测 | `polishing/core/dwell.py`、`simulation.py`；界面在 `ui/features/prediction.py` |
| 执行与文件输出 | `ui/features/execution.py`、`polishing/io/path_export.py` |
| 流程状态与计算门面 | `polishing/workflow/` |
| 窗口、预览与控件 | `polishing/ui/window.py`、`viewers/`、`widgets/` |

`workflow/engine.py` 将纯计算服务组合成界面兼容门面；数值实现不加载 Qt/OCC。`core/` 中 `model_*.py` 是状态与数据契约，`sampling.py`、`surfaces.py`、`transforms.py` 是曲面基础能力。`io/` 负责外部文件、许可证及机器文件格式。

## 当前边界与未完成项

八阶段中建档、评价、规划、预测和执行有可用功能；复测、验证和归档仍是占位入口。当前数据仍以字典和数组为主，项目数据库、后测配准、完整口径 PSD 估计及共形振动预测尚未完成。

## 资源与测试

`resources/icons/` 中的 PNG 由图标加载器按文件名动态登记，Ribbon 和菜单会按图标名取用；`resources/stylesheets/` 的五个 CSS 文件由样式加载器启动时读取。`data/examples/` 是应用内的数据选择样例。删除这些资源会导致界面图标、样式或样例入口缺失。

`tests/` 不是应用运行依赖，但用于修改算法后检查输出兼容性。`fixtures/original_outputs.npz` 是旧版冻结结果，`migration_cases.py` 是其固定输入；不应把基线当作算法正确性的证明。

## 验证

从项目根目录运行 `python -B tests/test_migration.py -v`。`tests/fixtures/original_outputs.npz` 保存迁移基线，`tests/migration_cases.py` 定义固定输入。数值基线用于确认重构没有改变既有输出；界面检查需要 PyQt5 环境。它不验证真实设备执行安全性或完整三维视觉表现。
