# traj：分层后的抛光工艺软件

现有 Python 软件功能已迁入此目录，并按“建档 → 评价 → 规划 → 预测 → 执行 → 复测 → 验证 → 归档”重组顶层流程。前五阶段已经通过页面内部的“下一步”按钮连成实际操作链，复测、验证和归档暂保留入口。轨迹规划、驻留流程、工具与工艺条件、抛光斑/TIF、OCC/二维预览、TXT/PATH 输出和授权继续可用。启动及计算均使用 traj 内的代码与资源，不导入父目录旧程序。

## 启动

在原工程目录的终端执行：

```powershell
conda activate proj
cd traj
python -B main.py
```

也可从原工程根目录执行 `conda run --no-capture-output -n proj python -B traj/main.py`。三维显示使用 pythonocc-core；二维显示使用 matplotlib。可选 XLSX 导出函数才需要 openpyxl。

## 当前组织

代码按四个目录整理，八阶段功能入口集中在 `ui/features/`：

- `polishing/core/`：曲面建模与采样、测量处理与指标、接触/去除模型、轨迹、驻留和仿真。文件以 `trajectory_`、`measurement_`、`model_` 等前缀标明用途。
- `polishing/workflow/`：桌面业务使用的计算门面、八阶段状态及轨迹导出入口。
- `polishing/io/`：DAT/网格读取、TXT/PATH 导出、授权和资源路径。
- `polishing/ui/`：八阶段功能页、Ribbon 控件、二维/OCC 预览及窗口启动。
- `main.py`：启动入口；`resources/`、`data/`、`tests/`、`docs/` 分别存放资源、样例、检查和说明。

几何曲面与各类轨迹算法集中在 `polishing/core/`，可从 `trajectory_*.py` 找到对应实现；桌面功能从 `polishing/ui/features/` 找到。源码中不再保留旧 `pages/`、`tasks/` 空目录；Python 字节码缓存不纳入项目内容。

## 验证

```powershell
python -B tests/test_migration.py -v
```

迁移基线包含 89 组旧版输出，覆盖面形处理、模型、DAT、去除函数、驻留、刀位及多类轨迹。测试包含数值、读写与界面流程检查；界面检查需要 PyQt5。基线用于检查重构前后的兼容性，不代表完整三维视觉或机床执行验证已经通过。

## 尚待实现

加工前评价现已支持矩形有效区域内的二维 PSD、中频带 RMS 与方向分析；任意掩膜口径谱估计、共形振动中频预测、配准、案例数据库、后台任务和统一对象模型尚未贯穿界面全流程。本次是现有功能迁移与评价能力的增量实现，不代表研究计划全部完成；兼容数值 API 仍使用现有字典/数组形式，统一对象模型尚未贯穿界面全流程。

- [架构与迁移详情](docs/architecture.md)
- [八阶段业务逻辑链](docs/八阶段流程逻辑链.md)
- [评价协议与当前实现边界](docs/evaluation_protocol.md)
- [前五阶段完整演示案例](docs/前五阶段完整演示案例.md)
- [研究设计快照](docs/第一性原理展开.md)
- [小白导读](docs/小白导读.md)
- [本轮改动记录](docs/改动记录.md)
