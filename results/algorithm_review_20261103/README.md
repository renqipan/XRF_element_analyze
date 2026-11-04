# 算法复核证据（2026-11-04完成）

- `before.json`：原始Git版本的34次复算，完整密度列及支持/待定集合。
- `after.json`：最终算法34次复算，包括明确硬件采集域、失败状态及警告。
- `comparison.json`：相同参数的逐谱前后对照。
- `verification.json`、`tests.txt`：210项测试、构建及源码哈希。
- `pure_pb_acquisition_domain.png`、`pure_pb_final_fit.json`：纯Pb诊断图与最终模型。
- `pure_pb_before_cutoff_review.json`：修复TLLD之前的诊断，不作最终结果。
- `declared_cutoff_review.json`：硬件配置和原算法漏判采集域的样本。
- `input_and_artifact_inventory.json`：原始数据和历史产物逐文件盘点。
- `existing_*artifact_audit.json`：两个历史批次的产物一致性检查。

`with_background`表示将用户确认的纯Pb谱作为参考的敏感性试验，不是独立空白背景或可靠定量验证。无参考谱结果也仍是给定S/薄样品近似下的表观量。原始数据及历史结果未覆盖。
