# 当前批次与历史报告

`batch_manifest.json`、`batch_summary.txt`和`relative_areal_density_all.txt`定义本次运行的输入、状态和数值结果。当前输入目录含17份MCA；本次计算不恢复已从工作区移除的原始谱。

结果目录可以保留以往运行但不属于当前输入集合的报告。它们不进入本次TXT汇总，查看历史报告时请核对其`fit_summary.json`中的版本和输入哈希。0.3的全局模板门控结果与0.4的独立主线面积定量不能混用。

0.4中的全谱`fitted_lines.csv`仍表示全局模板诊断；面密度使用`quantification_peaks.csv`的净面积与灵敏度，拟合分量和审计见`quantification_fit.csv`、`quantification_regions.png`及`quantification_audit.json`。
