# XRF Spectrum Analyzer 0.2

对SDD测量的XRF原始计数谱执行：自动寻峰、亚道峰心拟合、全谱多分支元素识别、多元素泊松联合拟合，以及**仅对证据支持且通过定量质量检查的元素**计算表观相对质量面密度。

本轮新增低能钙检验、K/L/M同空穴层分支扩展、场景常见元素平局先验，以及用户指定峰心 `find_peaks`。当前流程以本README和 `XRF_analyze_strategy.md` 为准；`ANALYSIS_IMPROVEMENTS.md`保留上一轮推导和对照。代码仓库：[renqipan/XRF_element_analyze](https://github.com/renqipan/XRF_element_analyze)（私有）。

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

当前 **80项测试通过**，包含真实谱、已知真值的泊松合成谱、手动峰心和元素先验回归。

## 自动分析真实谱

```bash
.venv/bin/python -m xrf_analyzer analyze \
  --spectrum SDD_Spectrum/shafa_50kv_20uA.mca \
  --calibration data/calibration.csv \
  --sensitivity data/xrf_relative_sensitivity.csv \
  --config configs/shafa.json --output-dir results/shafa_current
```

默认 `find_peaks=None`，自动发现观测峰。参数JSON中同名值由CLI覆盖。`--elements Fe,Ca,Cu,Hg,Pb`可限制候选，`--exclude-elements`可排除。文件名注明50 kV，配置记录该电压；没有依据不自动猜测靶材。若仪器确为Ag靶，可加 `--tube-target Ag`；靶元素的瑞利峰不作为样品含Ag的确证。

### 用户指定峰心

`find_peaks`默认值为Python `None`（JSON为`null`）。提供时**替代自动寻峰**，不是往自动发现列表再加峰。

```bash
.venv/bin/python -m xrf_analyzer analyze \
  --spectrum SDD_Spectrum/shafa_50kv_20uA.mca \
  --calibration data/calibration.csv \
  --sensitivity data/xrf_relative_sensitivity.csv \
  --config configs/shafa.json --output-dir results/shafa_manual \
  --find_peaks '[3.69, 3.688, 6.4, 7.06, 10.5, 12.6, 14.7, 9.98, 11.9]'
```

`--find-peaks`与`--find_peaks`等价，也接受空格分开的数值，例如 `--find-peaks 6.4039 7.058`。推荐给方括号列表加引号，避免shell展开。精确重复、空列表、负值、NaN和无穷值报错；接近但不同的中心会保留。

默认 `--find-peaks-axis reference`：数值是用户指定的参考中心，模型固定这些相对位置，仅拟合全谱共享的能量偏移/增益。**所有被约束的分支中心都取自用户列表**；未列出的其它已知分支仍按物理模型参与检验，未列出不等于不存在。

如果输入的是已经在当前标定谱轴上读出的实测中心，使用 `--find-peaks-axis observed`，这些中心直接固定，不再施加共享平移或增益。`fitted_lines.csv`同时保存数据库参考值、`Specified_Center_keV`、最终观测轴上的中心和`Center_Source`，不混淆两种坐标。

3.69与3.688 keV远小于SDD分辨率，保留两个输入约束但只作为一份独立观测证据。同一元素的幅度仍通过S/分支比关联，不将不可分双线当作独立面积测量。输入不存在的峰不会自动确证元素；输入粗略四舍五入的中心可能增加模型失配。参数不完整时仍可能得到待定或空定量结果。

等价配置：`configs/shafa_manual_reference.json`。

### 批量与合成谱

```bash
.venv/bin/python examples/generate_examples.py
.venv/bin/python -m xrf_analyzer analyze \
  --spectrum examples/mixture_Fe_Cu_Pb.csv examples/overlap_Pb_As.csv \
  --calibration examples/calibration.csv --sensitivity data/xrf_relative_sensitivity.csv \
  --output-dir results/demo_batch --energy-range 1 35 --elements Fe,Cu,Pb,As,Co
```

每个样品输出到独立子目录。示例是固定随机种子的泊松**样品谱**，不是入射激发谱；真值、偏移、峰宽和辅助L1幅度保存在`.truth.json`。批量不自动将不同live time的q作为计数率比较。

## 当前真实谱与钙

`results/shafa_current/`是最新版自动结果；`results/shafa_manual/`对应上述用户指定中心示例。旧结果目录保留作历史对照，不作为当前算法输出。

自动结果中Ca、Fe、Hg、Pb获得默认规则下的`Status=supported`。Ca Kα的实测复合峰约3.82 keV；使用全谱最终偏移和增益后，与Ca理论Kα位置相符。程序在原始计数上比较Kα/Kβ固定比例、自由比例及逐组移除模型，并用两种背景窗口复验：

- Ca Kα局部删除增益ΔD约315，证据强。
- Ca Kβ（含同能分支）ΔD约6.1，宽窗口约6.4；证据偏弱、接近默认阈值，适合复测确认。
- 自由拟合的Kβ/Kα面积比约0.095，模型比约0.126；固定比与自由比的差异ΔD约0.60，没有显示明显矛盾。

因此自动流程在联合证据下支持Ca，但不把它说成无条件确证。使用用户给出的粗略手动中心时，Ca仍待定，不能因输入3.69 keV就强行确认。详见`calcium_analysis.json`、`branch_ratio_evidence.csv`和`calcium_region.png`。

自动全谱deviance/dof约7.02，仍有背景、散射、分支比或基体方面的失配，故**真实谱的比例继续留空**。`Status`表示元素身份，`Quantification_Status`表示模型是否允许计算比例，两者独立。

## 算法

1. **原始计数和标定。** 校准CSV做线性回归。两点标定不产生虚假的零协方差；含完整误差时加权。所有输入保留SHA256。
2. **有效采集域。** 长精确零前缀且随后骤增时，保守推断低能截断。当前文件推断约2.9908 keV，不把未采集区当成“没有Pb M线”。推断需仪器确认，`--no-infer-cutoff`可关闭。
3. **峰心。** SNIP背景、分辨率相关平滑发现凸起；相邻峰对原始泊松计数联合拟合中心、面积、宽度和正背景。没有独立极大值的肩峰不编造新观测证据。
4. **更多分支。** 原九分支CSV保留。只对正S建立锚点；同空穴层满足`S_line=C_shell*RadRate_line`时补全遗漏K/L/M跃迁。已有锚点`S/RadRate`相对不一致超过5%则停止该层补全。原S不再乘产额，不改写、不归一化。缺少L1锚点时只加独立非定量辅助壳层幅度，不跨壳层猜S。
5. **全谱配对。** 结合峰心误差、未分辨组中心、缺失强线的可见性、分支比例及少元素原则构造候选。最终共享增益和偏移回传配对，避免低能钙被旧偏移初值误排。局部K分支检验能寻找弱Kβ贡献，不要求它形成独立最大值。
6. **场景先验。** 仅在证据接近/无法区分时偏好常见元素；强数据支持的非常见元素照常保留。偏好不制造`supported`，歧义仍排除定量。
7. **联合拟合。** `lambda=positive_background+sum(q_i*S_il*integrated_Gaussian)+auxiliary_shells+instrument`，q非负。全谱共享受限偏移、增益、分辨率，不逐线任意漂移。数值参数缩放不改变S的物理尺度。
8. **残差与证据。** 删除元素时一起删除其辅助壳层并重拟合；残差新候选还需至少一处未被已有元素解释的原始观测特征，避免用稀有元素填已有强峰形状错误。
9. **统计与定量。** 独立分支证据、模型比较和局部比例一致性共同判别。Fisher信息包含背景/响应/辅助幅度，比值误差包含协方差。边界/奇异/模型失配不报伪精确区间。局部ΔD、模式分数和先验均不是已校准概率。

可关闭/调整：`--no-atomic-completion`、`--no-estimate-resolution`、`--no-fit-gain`、`--fixed-response`、`--alignment-max-shift-kev`、`--centroid-tolerance-kev`、`--max-deviance-per-dof`。全部请求参数和有效参数写入摘要。

## 常见与非常见元素

默认`general`分类覆盖全部118种元素：**51种常见、67种非常见**。这是文物/考古/常规矿物分析的可审查工程先验，不是普适地壳丰度表，也不是可检测性、浓度或存在概率。

- `data/common_elements.csv`：默认常见列表。
- `data/uncommon_elements.csv`：默认非常见列表。
- `data/element_classes.csv`：全部元素及不同场景分类。

默认常见：H Li Be B C N O F Na Mg Al Si P S Cl K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn As Se Br Rb Sr Y Zr Nb Mo Ag Cd In Sn Sb Cs Ba La Ce Nd W Au Hg Pb Bi。

`--element-prior general|heritage|archaeology|minerals|none`可切换场景/关闭先验。例如矿物场景把更多稀土及Th/U纳入常规考虑；文物场景侧重颜料、陶瓷、玻璃和金属。规则源在`element_prior.py`，CSV是导出清单，修改CSV不会静默改变模型。`--prior-tie-margin`默认0.5，限定先验能介入的接近分数范围。

有歧义时输出`Preferred_By_Prior`和`Prior_Alternatives`，逐峰表给出`Preferred_By_Context_Prior`。例如Cr/Pm均未确证且能量解释接近时，优先展示Cr，但二者不参与质量比例。真正无法辨识的共线模型继续标为`unidentifiable`。

## 面密度计算与输出

`relative_areal_density.csv` **仅包含`Status=supported`的元素**。待定、歧义、不可辨识和未支持元素均不出现在这张表；所有候选q/状态在`element_diagnostics.csv`，待定清单在`elements_need_confirmation.csv`，日志和JSON明确提示需要进一步确定。

只有识别支持、`Quantification_Status=supported`、正q和可估计误差同时满足，才计算：

`Relative_to_Reference=q_i/q_ref`

`Fraction_among_Reported_Elements=q_i/sum(q_eligible)`。

显式`--reference-element Fe`不可靠时不会悄悄换参考。没有合格元素时输出合法空表；身份支持但强度模型失配时保留诊断q，比例为空。份额是**报告元素之间的归一化量**，不是样品全部元素质量百分比，更不是绝对g/cm²。

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
| `fit_spectrum.csv`, `fit_components.csv` | 原始计数、总模型、背景、散射、元素分量和残差 |
| `fit_summary.json` | 完整配置、输入哈希、版本、推断和警告 |
| `spectrum_fit.png`, `residuals.png`, `unassigned_peaks.csv` | 谱图、残差和未解释特征 |

`Net_Area=q*S`为完整线面积；`Gross_Area`为中心±3σ内原始计数（边界道按长度分摊），包括背景和其它峰。重叠ROI的Gross不能相加；同范围比较用`Net_Area_in_ROI`，不能将Gross−全峰Net视作精确背景。不可分双线共用q和相关误差。

摘要区分数值`fit_status`、模型`analysis_status`、`quantitative_interpretation_valid`。模型失配仍完整保存诊断；不能仅凭命令退出码认定科学模型有效。输入/优化失败返回非零并清除同目录旧成功输出，避免误用历史结果。

## 输入与科学限制

- 标定CSV：`channel,energy_keV`，兼容`Channel`/`Energy`/`energy`/实际文件的`energy(keV)`。至少两道，拒绝重复/非正斜率/非有限数；兼容现有仪器`(0,0)`原点并明确记录。
- 样品CSV：`counts`，可选`channel`、`energy_keV`及采集时间/电压。原始非负整数计数；道连续递增。单列默认从0起，`--channel-start 1`可改；补缺道需显式开启。
- MCA：支持真实文件所用的Amptek PMCA ASCII标记、MCAC道数、时间和嵌入标定。Windows头部字符允许明确的latin-1回退，数据区严格数值校验。`.mac`按内容识别，未知/二进制明确报错。
- 灵敏度CSV保留原九列契约；0、缺失、负数/无穷含义分别处理。新增线只从已有同层正S推导，补线不改写用户CSV。无锚点壳层不臆造S。
- 能量只从`xraylib.LineEnergy`读取，单位keV；`data/characteristic_lines_energy.csv`保留为原始参考，不参与分析。

当前默认薄样品、无自吸收、全能峰效率1.0。统计区间不包括源S、效率、自吸收/分层、背景、截断推断、元素选择等系统误差。原S缺少来源元数据，源条件匹配未经验证；文物与矿物的基体可能改变分支比。元素未支持不表示不存在，元素存在也不能单独确定化合物、年代或真伪。常见性不会弥补这些信息缺失。

## 研究依据与发布

同层补线使用xraylib公开的`CS_FluorLine=CS_FluorShell*RadRate`关系：[xraylib API](https://github.com/tschoonj/xraylib/wiki/The-xraylib-API-list-of-all-functions#x-ray-fluorescence-cross-sections)。校准、重叠和实验条件对谱线比的影响参见[PyMca教程](https://www.silx.org/pub/doc/PyMca/latest/training/quantification/index.html)。场景分类是工程启发式，参考[Getty文化遗产XRF材料](https://www.getty.edu/conservation/publications_resources/pdf_publications/pdf/handheld-xrf-cultural-heritage.pdf)和[Texas A&M岩矿pXRF说明](https://gcr.tamu.edu/instruments/pXRF.html)，不声称这些机构发布过本项目的51/67分类。

上传包含代码、配置、原始数据、示例、测试、文档及结果；`.venv`、Python缓存、构建产物和环境密钥不上传。原始MCA/CSV通过Git属性保留字节，方便SHA256审计。

补充原始资料位于`XRF_SDD/`（20份MCA，发布期间加入），均保留原始字节。当前验证范围为`SDD_Spectrum/shafa_50kv_20uA.mca`和合成示例；新增资料需逐采集条件匹配标定与灵敏度，见该目录说明。
