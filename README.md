# XRF Spectrum Analyzer 0.4

对SDD测量的XRF原始计数谱执行：自动寻峰、亚道峰心拟合、全谱多分支元素识别、多元素泊松联合拟合，以及**仅对证据支持且通过定量质量检查的元素**计算表观相对质量面密度。

本轮将**元素识别的全谱模板**与**面密度计算的局部净峰面积**分开：默认从已支持元素的主线独立测量面积，按共同尺度灵敏度计算比值。全谱的散射、背景或远处分支比失配保留诊断，不再自动清空所有可测主线的比例。已有Z≤24的Kα主导识别、K/L/M同空穴层分支扩展、场景先验、用户中心 `find_peaks`和TXT批处理继续使用。当前流程以本README和 `XRF_analyze_strategy.md` 为准；历史结果见`VALIDATION.md`。代码仓库：[renqipan/XRF_element_analyze](https://github.com/renqipan/XRF_element_analyze)（私有）。

## 安装与测试

建议Python 3.8–3.12。当前验证环境Python 3.8.10、NumPy 1.24.3、SciPy 1.10.1、xraylib 4.2.0；实际依赖见 `requirements.lock.txt`。该文件是环境版本记录，不是带wheel哈希的跨平台锁。

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q
```

本机缺少ensurepip时可用已有系统pip在项目内引导，不改系统包：

```bash
python3 -m venv --without-pip --system-site-packages .venv
python3 -m pip install --target .venv/lib/python3.8/site-packages \
  'numpy>=1.23,<2' 'xraylib>=4.1,<5' 'pytest>=7,<9' 'setuptools>=61,<76' wheel
.venv/bin/python -m pip install --no-build-isolation -e '.[test]'
PYTHON_BIN=.venv/bin/python scripts/verify.sh
```

回归测试包含真实谱、已知真值的泊松合成谱、手动峰心、元素先验和批处理失败隔离。当前测试与真实数据的复验结果见 `VALIDATION.md`。

## 自动分析真实谱

```bash
.venv/bin/python -m xrf_analyzer analyze \
  --spectrum SDD_Spectrum/shafa_50kv_20uA.mca \
  --calibration data/calibration.csv \
  --sensitivity data/xrf_relative_sensitivity.csv \
  --quantification-method local-peaks \
  --config configs/shafa.json --reference-element Fe --output-dir results/shafa_current
```

默认 `find_peaks=None`，自动发现观测峰。参数JSON中同名值由CLI覆盖。`--elements Fe,Ca,Cu,Hg,Pb`可限制候选，`--exclude-elements`可排除。文件名注明50 kV，配置记录该电压；没有依据不自动猜测靶材。若仪器确为Ag靶，可加 `--tube-target Ag`；靶元素的瑞利峰不作为样品含Ag的确证。

默认 `--quantification-method local-peaks` 使用独立主线面积定量。需要复现旧版严格的全谱共同q方法时，使用 `--quantification-method global-template`；此模式继续受全局模型和固定分支比质量检查约束。

### 用户指定峰心

`find_peaks`默认值为Python `None`（JSON为`null`）。提供时**替代自动寻峰**，不是往自动发现列表再加峰。

```bash
.venv/bin/python -m xrf_analyzer analyze \
  --spectrum SDD_Spectrum/shafa_50kv_20uA.mca \
  --calibration data/calibration.csv \
  --sensitivity data/xrf_relative_sensitivity.csv \
  --config configs/shafa.json --reference-element Fe --output-dir results/shafa_manual \
  --find_peaks '[3.69, 3.688, 6.4, 7.06, 10.5, 12.6, 14.7, 9.98, 11.9]'
```

`--find-peaks`与`--find_peaks`等价，也接受空格分开的数值，例如 `--find-peaks 6.4039 7.058`。推荐给方括号列表加引号，避免shell展开。精确重复、空列表、负值、NaN和无穷值报错；接近但不同的中心会保留。

默认 `--find-peaks-axis reference`：数值是用户指定的参考中心，模型固定这些相对位置，仅拟合全谱共享的能量偏移/增益。**所有被约束的分支中心都取自用户列表**；未列出的其它已知分支仍按物理模型参与检验，未列出不等于不存在。

如果输入的是已经在当前标定谱轴上读出的实测中心，使用 `--find-peaks-axis observed`，这些中心直接固定，不再施加共享平移或增益。`fitted_lines.csv`同时保存数据库参考值、`Specified_Center_keV`、最终观测轴上的中心和`Center_Source`，不混淆两种坐标。

3.69与3.688 keV远小于SDD分辨率，保留两个输入约束但只作为一份独立观测证据。全谱识别模板的同一元素幅度仍通过S/分支比关联；局部定量在不可分主组内部保留S比例，不将双线当作两个独立面积测量。输入不存在的峰不会自动确证元素；输入粗略四舍五入的中心可能增加模型失配。参数不完整时仍可能得到待定或空定量结果。

等价配置：`configs/shafa_manual_reference.json`。

### 批量与合成谱

逐个处理 `XRF_SDD/` 中所有MCA或CSV（扩展名不区分大小写）：

```bash
./analyze_all_spectrums.sh
```

默认输出到 `XRF_Analyze_results/`。脚本可从任意工作目录执行；优先使用项目 `.venv/bin/python`，也可通过 `PYTHON_BIN`选择Python。输入目录、输出目录、标定、灵敏度和配置分别可由 `XRF_INPUT_DIR`、`XRF_OUTPUT_DIR`、`CALIBRATION_FILE`、`SENSITIVITY_FILE`、`XRF_CONFIG`覆盖，CLI分析选项可直接追加：

```bash
./analyze_all_spectrums.sh --input-dir XRF_SDD --output-dir XRF_Analyze_results \
  --reference-element Fe --element-prior heritage
```

每个输入的完整文件名都会保留，避免 `sample.mca` 与 `sample.csv`相互覆盖：

- `<完整文件名>.spectrum_fit.png`：输出目录中的逐样品拟合图片。
- `spectra/<完整文件名>/`：完整的单样品报告，包括原有逐样品CSV。
- **`relative_areal_density_all.txt`**：所有样品的面密度比值汇总，UTF-8 BOM编码、制表符分隔的TXT；每个证据支持的元素一行，包含参考元素、相对比值、定量状态、无法估计的原因以及该样品的待定元素。没有支持元素或处理失败的样品也保留状态行；待定元素不进入比值分母。
- `batch_summary.txt`：每样品一行的支持/可能元素、拟合状态和图片路径。
- `batch_manifest.json`：输入哈希、参数和逐文件处理进度。

汇总的**空比值代表暂不可估计**，不能解释为零含量或强行填1。每处理完一份谱立即保存汇总；单文件输入或拟合失败会继续下一个样品，最终返回非零。模型失配与程序失败分开记录。默认不从文件名猜测40/50 kV、靶材或灵敏度匹配关系；各样品的定量解释仍需核对实验条件。批处理仅遍历输入目录第一层，避免误读子目录中的派生结果。

批处理沿用所选定量方法，TXT中保留`q_Fit`、`q_Area`、所选分支、方法和质量状态。可用 `.venv/bin/python scripts/audit_batch_results.py XRF_Analyze_results`复核模型分量、元素筛选、TXT、图片和哈希；实际执行记录见`VALIDATION.md`。产物完整、局部比例可估计和全谱模型有效是不同判断。

生成与分析固定真值的合成谱：

```bash
.venv/bin/python examples/generate_examples.py
.venv/bin/python -m xrf_analyzer analyze \
  --spectrum examples/mixture_Fe_Cu_Pb.csv examples/overlap_Pb_As.csv \
  --calibration examples/calibration.csv --sensitivity data/xrf_relative_sensitivity.csv \
  --output-dir results/demo_batch --energy-range 1 35 --elements Fe,Cu,Pb,As,Co
```

每个样品输出到独立子目录。示例是固定随机种子的泊松**样品谱**，不是入射激发谱；真值、偏移、峰宽和辅助L1幅度保存在`.truth.json`。批量不自动将不同live time的q作为计数率比较。

## 真实谱的身份与定量诊断

0.4自动分析`SDD_Spectrum/shafa_50kv_20uA.mca`，以Fe为参考，得到：

| 元素 | 主组净面积 counts | 对应S合计 | 表观相对面密度（Fe=1） | 比值相对不确定度 |
|---|---:|---:|---:|---:|
| Ca | 499.49 | 6.25939 | 0.154585 | 11.10% |
| Fe | 6421.61 | 12.4399 | 1 | 参考归一化 |
| Hg | 1955.68 | 5.159645 | 0.734261 | 15.50% |
| Pb | 53230.07 | 5.104659 | 20.200559 | 1.94% |

比值误差包含局部计数、背景/响应参数、共享道相关、过散度和窗口差异；不包含未知S标定、自吸收及几何系统误差。这些是给定S/薄样品假设下的表观值，Cr/Ag/Pm待定，未进入分母。全局deviance/dof仍约7.018，局部主线可测与全谱模型有效分开报告。

本轮当前目录17份谱已全部复算，43条支持元素记录均有数值比值，输出到`XRF_Analyze_results/relative_areal_density_all.txt`；121项回归通过。高窗口敏感性及局部响应边界等标志保留在质量列，不能把所有数值视为同等精度。详细验证见`VALIDATION.md`。

`results/shafa_current/`是最新版自动结果；`results/shafa_manual/`对应上述用户指定中心示例。旧结果目录保留作历史对照，不作为当前算法输出。

先检查`element_candidates.csv`中的身份状态，再检查`quantification_peaks.csv`中该支持元素的主线是否可测。Ca等低Z元素主要依赖Kα，弱Kβ不必独立形成最大值；原始计数、共享标定和竞争元素仍参与识别。Cr/Pm这类低于探测器分辨率的重叠解释需要保留待定，不能因某个反事实模型中的面积为正就进入面密度分母。

旧版将全谱`deviance/dof`超限、固定分支比失配或全局响应问题统一传递成“所有比例不可估计”。这混淆了远处模型误差与局部峰面积能否测量；全谱强制分支共享q还可能拉偏主线面积。0.4在支持身份之后，单独对主线区域做局部原始计数拟合，用净面积与对应S计算相对面密度。全谱失配仍显示在`fit_summary.json`和全谱图中，局部测量与误差则见`quantification_audit.json`、`quantification_fit.csv`和`quantification_regions.png`。

## 算法

1. **原始计数和标定。** 校准CSV做线性回归。两点标定不产生虚假的零协方差；含完整误差时加权。所有输入保留SHA256。
2. **有效采集域。** 长精确零前缀且随后骤增时，保守推断低能截断。当前文件推断约2.9908 keV，不把未采集区当成“没有Pb M线”。推断需仪器确认，`--no-infer-cutoff`可关闭。
3. **峰心。** SNIP背景、分辨率相关平滑发现凸起；相邻峰对原始泊松计数联合拟合中心、面积、宽度和正背景。没有独立极大值的肩峰不编造新观测证据。
4. **更多分支。** 原九分支CSV保留。只对正S建立锚点；同空穴层满足`S_line=C_shell*RadRate_line`时补全遗漏K/L/M跃迁。已有锚点`S/RadRate`相对不一致超过5%则停止该层补全。原S不再乘产额，不改写、不归一化。缺少L1锚点时只加独立非定量辅助壳层幅度，不跨壳层猜S。独立辅助壳层须有其主q模板的可检测贡献；主模板信息分数低于候选阈值（默认3.5）时禁用其辅助壳层并重拟合，避免无主线证据的自由壳层抢占其它元素信号。
5. **全谱配对。** 结合峰心误差、未分辨组中心、缺失强线的可见性、分支比例及少元素原则构造候选。最终共享增益和偏移回传配对，避免低能钙被旧偏移初值误排。局部K分支检验能寻找弱Kβ贡献，不要求它形成独立最大值。
6. **场景先验。** 仅在证据接近/无法区分时偏好常见元素；强数据支持的非常见元素照常保留。偏好不制造`supported`，歧义仍排除定量。
7. **联合拟合。** `lambda=positive_background+sum(q_i*S_il*integrated_Gaussian)+auxiliary_shells+instrument`，q非负。全谱共享受限偏移、增益、分辨率，不逐线任意漂移。数值参数缩放不改变S的物理尺度。
8. **残差与证据。** 删除元素时一起删除其辅助壳层并重拟合；残差新候选还需至少一处未被已有元素解释的原始观测特征，避免用稀有元素填已有强峰形状错误。
9. **低Z与统计。** 对Z≤24且具有Kα分支的元素增加Kα主导路径：峰心吻合、标定独立受约束、Kα窄/宽窗口局部ΔD均≥25、全元素删除ΔD≥25，且其它元素对该区域的竞争受控时，弱Kβ不必独立检出。独立标定锚点须分别有可检测的主模板贡献，不使用辅助壳层、共享被测Kα或宽散射结构。合并为一个分辨率组的低能K系也可接受局部检验。若由Kα强度预计Kβ应明显可见，而原始计数和分支比检验显著矛盾，仍拒绝该路径。被未确证且比例失配的重叠候选压制的Kα可做反事实复核，但最多恢复待定，不提升确证或定量。其它多分支证据规则继续使用；单个自由漂移的Kα不能同时标定和确证自己。
10. **独立面积定量。** 默认对身份支持元素选择可测Kα/Lα/Mα主组，重新拟合净面积、局部正背景和邻峰干扰；可分辨的远处分支不再强制共享q。未分辨主组内部保留已有S比例。自动峰模式允许有限的局部位移/宽度修正，用户指定中心时遵守固定中心约束。窄/宽窗口分别检验可检测性、面积覆盖、可辨识性和背景稳定性，严重失败的元素仍不计算比例。
11. **误差与状态。** 从局部拟合的逐道影响函数求协方差，保留共享通道造成的跨元素相关项，再计入局部过散度及窗口差异；比值传播使用该协方差。全谱失配作为独立诊断，显式源条件冲突仍撤下比例。`global-template`模式保留旧版严格定量规则。局部ΔD、模式分数和先验均不是已校准概率。

可关闭/调整：`--no-low-z-kalpha`、`--no-atomic-completion`、`--no-estimate-resolution`、`--no-fit-gain`、`--fixed-response`、`--alignment-max-shift-kev`、`--centroid-tolerance-kev`、`--max-deviance-per-dof`。低Z阈值也可通过配置JSON调整，全部请求参数和有效参数写入摘要。

## 常见与非常见元素

默认`general`分类覆盖全部118种元素：**51种常见、67种非常见**。这是文物/考古/常规矿物分析的可审查工程先验，不是普适地壳丰度表，也不是可检测性、浓度或存在概率。

- `data/common_elements.csv`：默认常见列表。
- `data/uncommon_elements.csv`：默认非常见列表。
- `data/element_classes.csv`：全部元素及不同场景分类。

默认常见：H Li Be B C N O F Na Mg Al Si P S Cl K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn As Se Br Rb Sr Y Zr Nb Mo Ag Cd In Sn Sb Cs Ba La Ce Nd W Au Hg Pb Bi。

`--element-prior general|heritage|archaeology|minerals|none`可切换场景/关闭先验。例如矿物场景把更多稀土及Th/U纳入常规考虑；文物场景侧重颜料、陶瓷、玻璃和金属。规则源在`element_prior.py`，CSV是导出清单，修改CSV不会静默改变模型。`--prior-tie-margin`默认0.5，限定先验能介入的接近分数范围。

有歧义时输出`Preferred_By_Prior`和`Prior_Alternatives`，逐峰表给出`Preferred_By_Context_Prior`。例如Cr/Pm均未确证且能量解释接近时，优先展示Cr，但二者不参与质量比例。真正无法辨识的共线模型继续标为`unidentifiable`。

## 面密度原理、计算与输出

`relative_areal_density.csv` **仅包含`Status=supported`的元素**。待定、歧义、不可辨识和未支持元素均不出现在这张表；所有候选q/状态在`element_diagnostics.csv`，待定清单在`elements_need_confirmation.csv`，日志和JSON明确提示需要进一步确定。

每次分析结束，终端打印证据支持存在的元素、各自相对参考元素的面密度比值，以及可能存在、需要进一步确定的元素。身份支持但未通过定量检查时明确显示“暂不可估计”及原因，不把待定q当成有效比值。批处理对每份输入同样输出摘要；输入失败也明确报告失败状态。

在给定共同尺度S及表观薄样品近似下，某个未分辨主线组g满足：

`A_ig = C * rho_i * S_ig`，`S_ig = sum(S_il, l in group g)`。

其中A为扣除连续背景及邻峰后的净面积，C为同一份谱共有的通量、几何和曝光尺度，rho为元素质量面密度。因此：

`q_Area_i = A_ig / S_ig`

`Relative_to_Reference = q_Area_i / q_Area_ref`

`Fraction_among_Reported_Elements = q_Area_i / sum(q_Area_eligible)`。

C在同一谱内比值中抵消；S不再重复乘荧光产额或分支率。面积和S必须对应同一组谱线，不能用Kα双线总面积除以单条Kα1灵敏度。局部测量仍使用原始计数的Poisson似然，不把扣背景后的差谱当Poisson数据。

默认`local-peaks`仅对身份支持、主线可检测、可辨识、窗口稳定且有正`q_Area`与可估计误差的元素计算比例。邻峰可作为干扰参数参与拟合，但待定元素不因此获得定量身份。窄/宽两种窗口对同一批计数高度相关，不视为独立重复测量；不同元素定量窗口共享通道时，逐道影响函数保留统计协方差。局部过散度扩张该协方差，窄/宽窗口点估计差另作为模型敏感性项。该误差仍不覆盖所有实验系统偏差。

`q_Fit`始终保留全谱固定分支模板的共同幅度，可能受全局失配影响；`q_Area`是主线独立净面积/S。局部模式的比值使用后者，二者不能混用。面密度表中的`q_Std`、`q_Stat_Std`对应面积估计及其统计分量；`Global_Quantification_Status`保留旧全谱诊断。需要复现旧方法时选择`--quantification-method global-template`，其比值使用`q_Fit`并保留严格全局门控。

显式`--reference-element Fe`不可测时不会悄悄换参考。无合格元素时不计算比值；局部主线不可分、缺少正S、窗口严重不稳定或显式源条件冲突时保留具体原因。仅缺少S来源元数据会标记未验证，并在假定用户提供S适用的条件下报告表观结果。份额是**合格报告元素之间的归一化量**，不是样品全部元素质量百分比，更不是绝对g/cm²。

| 文件 | 内容 |
|---|---|
| `calibration_result.json`, `calibration.png` | 标定、残差和采集域 |
| `detected_peaks.csv` | 观测/指定中心、宽度、面积、误差和来源 |
| `peak_line_matches.csv`, `peak_assignments.csv` | 全候选匹配、主归属、少量重叠和先验偏好 |
| `element_candidates.csv`, `element_pattern_scores.csv` | 身份、模式、比较和类别 |
| `element_diagnostics.csv`, `elements_need_confirmation.csv` | 所有候选及待定元素，不能混入定量分母 |
| `element_classes.csv` | 本次运行使用的118元素分类 |
| `fitted_lines.csv`, `fitted_groups.csv` | 有定量S的逐线/组合面积，S来源和手动中心约束 |
| `auxiliary_fitted_lines.csv` | 无定量S的独立壳层面积 |
| `branch_ratio_evidence.csv` | 窄/宽窗口的原始计数分支检验及自由/理论比例 |
| `calcium_analysis.json`, `calcium_region.png` | 钙的详细证据与局部图（有相关候选时） |
| `relative_areal_density.csv` | 仅已支持元素；通过定量质量门控时才有比例 |
| `quantification_peaks.csv` | 独立主线的净面积、S合计、q_Area、统计/合并误差、所选分支与失败原因 |
| `quantification_audit.json` | 窄/宽窗口、邻峰干扰、局部响应、协方差及估计条件的审计 |
| `quantification_fit.csv`, `quantification_regions.png` | 局部原始计数、拟合分量、残差及定量区域图 |
| `fit_spectrum.csv`, `fit_components.csv` | 原始计数、总模型、背景、散射、元素分量和残差 |
| `fit_summary.json` | 完整配置、输入哈希、版本、推断和警告 |
| `spectrum_fit.png`, `residuals.png`, `unassigned_peaks.csv` | 谱图、残差和未解释特征 |

全谱`fitted_lines.csv`中的`Net_Area=q_Fit*S`为全谱模型线面积；它与`quantification_peaks.csv`中独立测量的主组面积分开。`Gross_Area`为中心±3σ内原始计数（边界道按长度分摊），包括背景和其它峰。重叠ROI的Gross不能相加；同范围比较用`Net_Area_in_ROI`，不能将Gross−全峰Net视作精确背景。不可分双线共享组面积及相关误差。

摘要区分数值`fit_status`、全谱`analysis_status`/`global_model_valid`和当前方法的`quantitative_interpretation_valid`。局部主线可测时，后者可为true而`global_model_valid`为false；它表示在给定S和局部模型条件下存在可报告比值，不是全谱模型或实物成分已验证。输入/优化失败返回非零并清除同目录旧成功输出，避免误用历史结果。

## 输入与科学限制

- 标定CSV：`channel,energy_keV`，兼容`Channel`/`Energy`/`energy`/实际文件的`energy(keV)`。至少两道，拒绝重复/非正斜率/非有限数；兼容现有仪器`(0,0)`原点并明确记录。
- 样品CSV：`counts`，可选`channel`、`energy_keV`及采集时间/电压。原始非负整数计数；道连续递增。单列默认从0起，`--channel-start 1`可改；补缺道需显式开启。
- MCA：支持真实文件所用的Amptek PMCA ASCII标记、MCAC道数、时间和嵌入标定。Windows头部字符允许明确的latin-1回退，数据区严格数值校验。`.mac`按内容识别，未知/二进制明确报错。
- 灵敏度CSV保留原九列契约；0、缺失、负数/无穷含义分别处理。新增线只从已有同层正S推导，补线不改写用户CSV。无锚点壳层不臆造S。
- 能量只从`xraylib.LineEnergy`读取，单位keV；`data/characteristic_lines_energy.csv`保留为原始参考，不参与分析。
- `data/backgroud_spectrum.mca`的采集条件与样品不同或不确定，当前仅作参考资料，不直接缩放扣除。连续背景由样品原始计数局部估计；需要未来建立匹配的背景测量、曝光归一及计数联合模型后才能作定量背景输入。

当前默认薄样品、无自吸收、全能峰效率1.0。局部误差计入计数统计、拟合参数、过散度和窗口敏感性，仍不包含S、效率、自吸收/分层、截断推断、元素选择及所有背景形状的系统误差。原S缺少来源元数据，源条件匹配未经验证；文物与矿物的基体可能改变分支比。元素未支持不表示不存在，元素存在也不能单独确定化合物、年代或真伪。常见性不会弥补这些信息缺失。

## 研究依据与发布

同层补线使用xraylib公开的`CS_FluorLine=CS_FluorShell*RadRate`关系：[xraylib API](https://github.com/tschoonj/xraylib/wiki/The-xraylib-API-list-of-all-functions#x-ray-fluorescence-cross-sections)。校准、重叠和实验条件对谱线比的影响参见[PyMca教程](https://www.silx.org/pub/doc/PyMca/latest/training/quantification/index.html)。场景分类是工程启发式，参考[Getty文化遗产XRF材料](https://www.getty.edu/conservation/publications_resources/pdf_publications/pdf/handheld-xrf-cultural-heritage.pdf)和[Texas A&M岩矿pXRF说明](https://gcr.tamu.edu/instruments/pXRF.html)，不声称这些机构发布过本项目的51/67分类。

上传包含代码、配置、原始数据、示例、测试、文档及结果；`.venv`、Python缓存、构建产物和环境密钥不上传。原始MCA/CSV通过Git属性保留字节，方便SHA256审计。

补充原始资料位于`XRF_SDD/`。批处理按运行时目录中实际存在的MCA/CSV逐一分析，不按历史文件清单恢复或补造样品；各采集条件需分别匹配标定与灵敏度。实际运行统计以 `VALIDATION.md`和批处理清单为准。
