# 第二轮优化证据（2026-11-04）

人工判读是参考假设，不是真值，且不进入识别程序。完整原理与限制见项目根目录 `ALGORITHM_REVIEW_ROUND2_20261104.md`。

- `PLAN.md`：修改前的原则与方案，包含用户后续补充的第18族报告要求。
- `before.json`、`before/`：用户实际输出的基线汇总、16份谱的精简逐谱证据；源码基线为`7ff68d6`。完整旧输出另备份在本机`/tmp/xrf-round2-original-output`，不依赖该临时目录复现本目录对照图。
- `manual_reference.json`：用户人工判读原意；`manual_comparison.md/.csv/.json`：读取真实四行TXT得到的前后对照，不计算实物准确率。
- `comparison.json/.csv/_table.md`：全部16份谱的支持状态、输入哈希及A/S比较；`comparison.png`：原始计数与前后拟合。
- 最终完整批量产物位于项目根目录 `XRF_Analyze_results/`，以`batch_manifest.json`列出的样品为准；历史目录不属于当前批次。旧输出未处理样品的历史删除状态没有用来删除原始数据。
- `parameter_training.json`：全字典、5组门槛、9个合成案例（噪声偏移5000）；`holdout_validation.json`：新噪声偏移6000；`holdout_baseline.json`：同一留出噪声、上一版源码与参数。报告显示策略不进入似然或已支持集合，新增第18族规则由最终完整测试及输出审计另外验证。
- `parameter_selection.json`：最终参数及选择依据；`pb_reference_control.json`：默认未扣背景条件下，Pb参考谱只支持Pb。
- `possible_reporting_parameters.json`：在同一组已保存拟合上比较普通待定显示门槛0/6/9/12/16；默认9将记录数36降到30。`before_possible_tightening.json`及`possible_reporting_invariance.json`验证此显示规则不改变确定集合与密度数学文件。
- `verification_complete.log`：最终完整测试和CLI检查；`batch_run.log`：最终默认Shell全量执行日志；`source_manifest.json`：当前源码/输入哈希；`before_hashes.json`：修改前的源码与输入哈希。
- `XRF_Analyze_results/artifact_audit.json`：逐谱与总表、结论过滤、A/S、分母门控、分量、图片与哈希审计。日志中的依赖警告为xraylib SWIG弃用警告。

中间算法试验JSON可能仍留在本地，不作为最终结论，未全部纳入提交。

```bash
./analyze_all_spectrums.sh
.venv/bin/python scripts/audit_batch_results.py XRF_Analyze_results
.venv/bin/python scripts/summarize_optimization.py \
  --root results/optimization_round2_20261104 --batch XRF_Analyze_results
.venv/bin/python scripts/compare_manual_review.py \
  --before results/optimization_round2_20261104/before/relative_areal_density_all.txt \
  --reference results/optimization_round2_20261104/manual_reference.json \
  --after XRF_Analyze_results --output-dir results/optimization_round2_20261104
.venv/bin/python scripts/plot_manual_review.py \
  --before results/optimization_round2_20261104/before \
  --after XRF_Analyze_results/spectra \
  --output results/optimization_round2_20261104/comparison.png
.venv/bin/python scripts/tune_possible_reporting.py \
  --batch XRF_Analyze_results --output /tmp/xrf-possible-floors.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/tune_identification.py \
  --all-elements --seed-offset 5000 \
  --variants current support_16 support_36 threshold_3 threshold_4 \
  --output /tmp/xrf-round2-training.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/tune_identification.py \
  --all-elements --seed-offset 6000 --variants current \
  --output /tmp/xrf-round2-holdout.json
PYTHON_BIN=.venv/bin/python bash scripts/verify.sh
```
