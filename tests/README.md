# 回归检查

在项目根目录运行：

```powershell
python -B tests/test_migration.py -v
```

`migration_cases.py` 定义固定输入，`fixtures/original_outputs.npz` 保存迁移前算法生成的 89 组数组和指标。测试对比当前计算输出与这些冻结结果，并检查资源路径、文件读写及界面流程状态。

界面检查需要安装项目使用的 PyQt5 和相关绘图库；在缺少 GUI 依赖的环境中，数值检查可单独运行，但界面检查不能据此判为通过。冻结结果只证明迁移前后数值兼容，不证明原算法满足物理模型或设备执行要求。
