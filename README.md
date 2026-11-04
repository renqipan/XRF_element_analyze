# XRF Spectrum Analyzer

本程序分析SDD测量的XRF原始计数谱，完成自动寻峰、元素识别、谱峰拟合，并对身份获得支持且通过定量检查的元素计算**表观相对质量面密度**。支持单谱、多个指定谱和整个目录的批量分析。

当前版本为0.5.0。单谱命令默认分析参数与`configs/shafa.json`一致：**Ag靶、50kV、1–40keV、自动寻峰并保存中间文件**，不固定Fe参考，参考元素自动选择。参数仍可覆盖。批处理默认Ag靶；额外背景参考默认关闭。

## 1. 安装和查看帮助

以下命令在项目根目录执行。项目要求Python ≥3.8；依赖NumPy、SciPy、pandas、Matplotlib和xraylib，版本范围见[pyproject.toml](pyproject.toml)。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
```

如果项目的`.venv`已经安装完成，直接使用后文命令即可，无需重复安装或激活环境。安装后也可使用`xrf-analyze analyze ...`入口，本文统一采用模块命令。

```bash
# 单谱分析的全部可用选项
.venv/bin/python -m xrf_analyzer analyze --help

# 目录批处理选项；同时支持转发单谱分析选项
.venv/bin/python -m xrf_analyzer.batch --help
```

运行项目测试：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q
# 或使用验证脚本：
PYTHON_BIN=.venv/bin/python bash scripts/verify.sh
```

## 2. 准备输入文件

每次分析至少需要样品谱、能量标定和相对灵敏度三类文件。必须确认标定和灵敏度适用于测量条件；程序不从文件名推测管压，也不自动验证S表适用性。

| 输入 | 要求 | 项目示例 |
|---|---|---|
| 样品谱 | Amptek PMCA ASCII MCA，或含`counts`列的CSV；原始计数为非负整数 | `SDD_Spectrum/shafa_50kv_20uA.mca` |
| 标定CSV | 至少两个不同道号的线性标定点，能量单位keV | `data/calibration.csv` |
| 灵敏度CSV | 跨元素共同尺度的九分支S表，不逐行归一化 | `data/xrf_relative_sensitivity.csv` |
| 独立背景，可选 | 原始计数；通道和标定匹配，覆盖分析域；默认不使用 | 自行提供条件适配的空白谱 |

样品CSV示例：

```csv
channel,counts
0,0
1,3
2,12
```

省略`channel`时默认从0起，可用`--channel-start 1`改为从1起。显式道号须连续递增；缺道默认报错，只有`--fill-missing-channels`才补零并记录。重复CSV列名、负数、非整数和非有限计数会报错。可选`live_time_s`、`real_time_s`、`tube_voltage_kV`列须在整份谱中保持一致。

标定CSV示例：

```csv
channel,energy_keV
0,0
299,22.08
```

兼容项目现有`energy(keV)`等列名；斜率必须为正。显式标定用于分析，MCA内嵌标定用于一致性检查。

灵敏度CSV必须包含：

```text
Element,Kalpha1,Kalpha2,Kbeta1,Lalpha1,Lalpha2,Lbeta1,Lbeta2,Lgamma1,Malpha1
```

`Element`写为`26 Fe`这类“原子序数+元素符号”。正S参与分析，0与缺失分别记录，负值和无穷值拒绝。谱线能量来自xraylib，`data/characteristic_lines_energy.csv`仅保留为历史参考。

## 3. 单谱分析

使用项目自带标定和灵敏度时，最简命令为：

```bash
.venv/bin/python -m xrf_analyzer analyze \
  --spectrum SDD_Spectrum/shafa_50kv_20uA.mca \
  --output-dir results/shafa_default
```

自定义文件时，追加`--calibration 文件路径`和/或`--sensitivity 文件路径`。

显式使用shafa配置：

```bash
.venv/bin/python -m xrf_analyzer analyze \
  --spectrum SDD_Spectrum/shafa_50kv_20uA.mca \
  --calibration data/calibration.csv \
  --sensitivity data/xrf_relative_sensitivity.csv \
  --config configs/shafa.json \
  --output-dir results/shafa_ag
```

必填参数只有`--spectrum`和`--output-dir`。`--calibration`默认使用`data/calibration.csv`，`--sensitivity`默认使用`data/xrf_relative_sensitivity.csv`，仍可显式覆盖；两个默认路径均相对于运行命令的工作目录，因此建议从项目根目录运行。`--config`可选。`configs/shafa.json`设置1–40keV能区、50kV管压、Ag靶和中间文件输出，适用于对应采集条件。

默认自动寻峰。结束时终端显示支持元素、可报告的比值和待定元素。输出目录可自选；复用目录会更新报告，要保留旧结果请使用新目录。

一次指定多个谱时，各输入stem必须不同，报告写入输出目录下以stem命名的子目录：

```bash
.venv/bin/python -m xrf_analyzer analyze \
  --spectrum examples/mixture_Fe_Cu_Pb.csv examples/overlap_Pb_As.csv \
  --calibration examples/calibration.csv \
  --sensitivity data/xrf_relative_sensitivity.csv \
  --energy-range 1 35 \
  --elements Fe,Cu,Pb,As,Co \
  --output-dir results/examples_ag
```

## 4. 靶材和默认配置

指定靶材使用`--tube-target`加规范元素符号，例如向完整单谱命令追加`--tube-target Cu`。靶材与定量参考元素`--reference-element Fe`是两个不同参数。

JSON使用下划线字段名，例如：

```json
{
  "tube_target": "Ag",
  "tube_voltage_kv": 50.0,
  "energy_range": [1.0, 40.0],
  "quantification_method": "local-peaks",
  "background_spectrum": null
}
```

**优先级：命令行参数 > `--config`指定的JSON > 命令入口及基础默认值。** JSON只需填写要覆盖的字段。

| 配置位置 | 作用 |
|---|---|
| [src/xrf_analyzer/config.py](src/xrf_analyzer/config.py) | `AnalysisConfig`定义基础参数；`single_spectrum_defaults()`定义单谱命令预设 |
| [src/xrf_analyzer/cli.py](src/xrf_analyzer/cli.py) | 应用单谱预设，设置标定/灵敏度默认路径并合并JSON及CLI覆盖 |
| [src/xrf_analyzer/batch.py](src/xrf_analyzer/batch.py) | 批处理入口默认Ag靶、1–40keV能区 |
| [configs/default.json](configs/default.json) | Shell批处理脚本默认加载的JSON配置 |
| [configs/shafa.json](configs/shafa.json) | shafa自动寻峰配置，Ag靶、50kV |
| [configs/shafa_manual_reference.json](configs/shafa_manual_reference.json) | shafa手动参考峰心配置，Ag靶、50kV |

单谱命令通过`single_spectrum_defaults()`提供与当前`configs/shafa.json`一致的内置分析参数，不依赖运行时读取该JSON。`--config`文件在此预设上覆盖，CLI再覆盖JSON；例如`--tube-voltage-kv 40`可指定其它管压。批处理继续使用自己的入口默认值和配置文件，未因此默认指定50kV。

基础`AnalysisConfig()`保留通用默认值；Python API需要相同单谱预设时，可调用`single_spectrum_defaults()`取得配置。单谱预设的`reference_element=None`，不会固定Fe。需要Fe=1时显式添加`--reference-element Fe`；它会改变比值归一化基准，所以与自动参考的数值不必相同，但元素身份和主线面积应一致。

查看当前单谱命令完整预设（输入文件路径默认值另外见`--help`中的`--calibration`和`--sensitivity`）：

```bash
.venv/bin/python - <<'PY'
import json
from xrf_analyzer.config import single_spectrum_defaults
print(json.dumps(single_spectrum_defaults().to_dict(), indent=2, ensure_ascii=False))
PY
```

查看某个JSON文件：

```bash
.venv/bin/python -m json.tool configs/shafa.json
```

运行后，输出目录`fit_summary.json`中的`parameters`记录合并后的配置；`effective_parameters`、`acquisition_cutoff`和`response_initialization`记录实际采用的采集域及自适应参数。`--help`列出可用选项，但不会逐项展示所有默认值。

### 常用选项

| 参数 | 用途及默认行为 |
|---|---|
| `--tube-target Ag` | 靶材；两个命令入口默认Ag |
| `--tube-voltage-kv 50` | 单谱默认50kV；批处理按其配置，不从文件名猜测 |
| `--scatter-angle-deg 90` | 已知散射角，单位度；默认不指定 |
| `--energy-range 1 40` | 分析能区keV，与有效采集域相交 |
| `--reference-element Fe` | 显式指定参考；默认从合格元素中选相对误差较小者 |
| `--elements Fe,Ca,Cu,Hg,Pb` | 限制候选元素；默认无白名单 |
| `--exclude-elements Pm,Tc` | 排除指定元素 |
| `--quantification-method local-peaks` | 默认局部净面积/S定量；`global-template`使用严格全谱模板模式 |
| `--element-prior heritage` | 场景先验：`general`（默认）、`heritage`、`archaeology`、`minerals`、`none` |
| `--save-intermediate` | 保存灵敏度状态和初始背景；单谱默认开启，JSON中`save_intermediate: false`可关闭 |
| `--fixed-response` | 固定共享响应，默认允许拟合 |
| `--no-infer-cutoff` | 关闭形状截断猜测；明确硬件阈值仍生效 |
| `--bootstrap 100` | 条件Poisson重采样诊断；默认0，开启后耗时增加 |

其余选项见`analyze --help`。只有配置字段而没有CLI选项的参数，通过JSON设置，例如`max_iterations`。

## 5. 用户指定峰心

`find_peaks`默认`None`（JSON为`null`），表示自动寻峰。非空列表会**替代自动寻峰**。shafa手动参考峰心示例：

```bash
.venv/bin/python -m xrf_analyzer analyze \
  --spectrum SDD_Spectrum/shafa_50kv_20uA.mca \
  --calibration data/calibration.csv \
  --sensitivity data/xrf_relative_sensitivity.csv \
  --config configs/shafa_manual_reference.json \
  --reference-element Fe \
  --output-dir results/shafa_manual_ag
```

也可向完整命令追加`--find-peaks '[6.4039, 7.058]'`覆盖JSON列表；`--find_peaks`为同义拼写，也接受空格分隔的数字。

- `--find-peaks-axis reference`（默认）：输入参考中心，固定相对位置，允许共享增益/偏移。
- `--find-peaks-axis observed`：输入当前标定轴上的实测中心，受约束中心不再施加共享增益/偏移。

接近但不同的中心可保留，未分辨中心仍只算一份独立观测证据。精确重复、空列表、非正值、NaN和无穷值拒绝。输入峰心不会自动确证元素；未列出的其它已知分支仍参与检验。

## 6. 整个目录批量分析

```bash
./analyze_all_spectrums.sh
```

脚本默认遍历`XRF_SDD/`第一层的MCA/CSV（扩展名大小写不敏感），加载`configs/default.json`，使用`data/calibration.csv`与`data/xrf_relative_sensitivity.csv`，输出至`XRF_Analyze_results/`。

自定义输入、输出和靶材：

```bash
./analyze_all_spectrums.sh \
  --input-dir XRF_SDD \
  --output-dir XRF_Analyze_results_ag \
  --tube-target Ag \
  --reference-element Fe

# 改用其它靶材：
./analyze_all_spectrums.sh --tube-target Cu
```

只有各样品确实共享采集条件时，才为整批指定`--tube-voltage-kv 50`或同一专用配置；不同条件应分组运行。

脚本转发单谱分析选项，可从其它工作目录调用，默认项目路径根据脚本位置确定；用户传入的相对路径按运行命令时的目录解析。也支持环境变量：

| 环境变量 | 用途 |
|---|---|
| `PYTHON_BIN` | Python解释器，默认优先使用项目`.venv/bin/python` |
| `XRF_INPUT_DIR` / `XRF_OUTPUT_DIR` | 输入/输出目录 |
| `CALIBRATION_FILE` / `SENSITIVITY_FILE` | 标定/灵敏度文件 |
| `XRF_CONFIG` | 配置JSON |
| `XRF_BLAS_THREADS` | BLAS/OpenMP线程数，默认1 |

单个输入或优化失败会记录原因并继续，汇总在每个样品完成后保存。全部成功返回0，样品失败返回1，批处理配置/目录等错误返回2。模型失配与优化失败分别报告。

## 7. 可选独立背景

`background_spectrum`默认`None`，不使用额外参考谱，但仍拟合连续背景。有独立测得、条件适配的空白谱时，可向单谱命令或批处理脚本追加背景参数：

```bash
# 替换为实际独立空白文件：
./analyze_all_spectrums.sh \
  --background-spectrum path/to/independent_blank.mca \
  --output-dir XRF_Analyze_results_with_blank

# 即使JSON配置了背景，也可显式关闭：
./analyze_all_spectrums.sh --background None
```

`--background-spectrum`、`--background_spectrum`、兼容拼写`--backgroud_spectrum`、`--backgroud-spectrum`及`--background`均为同一参数。JSON使用路径字符串或`null`。

模型联合拟合样品/参考的原始Poisson计数、非负全局scale及逐道背景均值，保留参考计数噪声，不把扣背景差谱当Poisson数据。要求精确通道/标定匹配并覆盖分析域，不隐式插值。单一scale无法修复几何、遮挡和谱形变化；scale误差不可估时相关比值暂不可估计。

**`data/backgroud_spectrum.mca`经用户确认为纯Pb样品XRF参考谱，不是通用空白背景。** 它与`XRF_SDD/Pb_50keV_20uA_back3mm.mca`字节相同，程序拒绝样品以自身或相同副本充当独立背景。含Pb基底的测量另需基底贡献和样品衰减模型。详见[data/README.md](data/README.md)。

## 8. 输出文件和查看顺序

先检查`fit_summary.json`的运行状态，再看`element_candidates.csv`的身份状态，最后看定量状态、比值及误差。

### 单谱输出

| 文件 | 内容 |
|---|---|
| `fit_summary.json` | 输入哈希、请求/有效配置、拟合状态、参考元素、质量标志与警告 |
| `calibration_result.json`、`calibration.png` | 标定及有效采集域 |
| `detected_peaks.csv` | 观测/指定峰心、宽度、面积及误差 |
| `element_candidates.csv`、`element_diagnostics.csv` | 候选身份及定量状态 |
| `elements_need_confirmation.csv` | 待定/歧义元素，排除定量 |
| `peak_line_matches.csv`、`peak_assignments.csv` | 峰与谱线候选配对、归属 |
| `relative_areal_density.csv` | 仅身份支持元素；通过定量检查才填写比值 |
| `quantification_peaks.csv`、`quantification_audit.json` | 局部净面积、对应S、误差及失败原因（局部模式） |
| `quantification_fit.csv`、`quantification_regions.png` | 局部面积拟合与图 |
| `fitted_lines.csv`、`fitted_groups.csv` | 全谱模板逐线/组面积，区别于局部测量面积 |
| `auxiliary_fitted_lines.csv` | 无定量S的辅助壳层贡献 |
| `branch_ratio_evidence.csv`、`branch_diagnostics.csv` | 分支证据及比例失配 |
| `fit_spectrum.csv`、`fit_components.csv` | 原始计数、全谱模型、分量与残差 |
| `spectrum_fit.png`、`residuals.png` | 全谱拟合与残差图 |
| `<谱文件stem>_elements.png` | 原始曲线与支持元素标签 |
| `sensitivity_states.csv`、`background_initial.csv` | 指定`--save-intermediate`后额外保存 |

Ca诊断另有`calcium_analysis.json`及适用时的`calcium_region.png`。其它辅助文件以当前运行输出为准。

### 批量输出

- `spectra/<完整谱文件名>/`：各样品完整报告。
- 根目录`<stem>_elements.png`：简洁元素图；同stem输入使用独立子目录避免覆盖。
- `relative_areal_density_all.txt`：UTF-8 BOM文本，每个输入严格四行：文件名、支持元素及相对面密度、待定元素、空白行。失败输入也占四行。
- `density_details.json`：机器可读数值、误差、方法与失败原因。
- `batch_summary.txt`：每样品一行的摘要。
- `batch_manifest.json`：输入哈希、参数、进度及逐样品结果。

复核已有成功批次的产物一致性：

```bash
.venv/bin/python scripts/audit_batch_results.py XRF_Analyze_results_ag
```

该工具检查表格、模型分量、图片、哈希等一致性，不能验证未知样品成分真值。

## 9. 定量含义和常见问题

局部定量默认使用：

```text
q_Area_i = 净主线组面积 A_i / 该组灵敏度合计 S_i
相对面密度(i/ref) = q_Area_i / q_Area_ref
```

双线总面积须除以同一双线的S合计。`q_Fit`是全谱固定分支模板幅度，`q_Area`是局部面积/S，默认比值使用后者。归一化份额只包含合格报告元素，不是样品全元素质量百分比，也不是绝对g/cm²。

- **“暂不可估计”或空值**：查看`Quantification_Status`及`Measurement_Failure_Reason`；可能由弱主线、不可分重叠、窗口不稳定、背景误差未知或源条件冲突造成。空值不能当0。
- **指定Fe参考后没有比值**：Fe必须身份支持且定量可用；不可测时不会自动换参考。可核对条件，或不指定参考以自动选择。
- **Ag没有作为样品元素报告**：已指定Ag靶时，靶荧光/散射与样品Ag存在来源混淆，不能仅凭这些峰确证样品含Ag。
- **数值收敛但全谱质量不好**：`fit_status=success`表示数值收敛；`global_model_valid`表示全谱质量；`quantitative_interpretation_valid`表示给定模型下存在可报告比值，三者含义不同。
- **部分低能道不分析**：对已验证的Amptek正常谱及固件/道映射，优先采用文件记录的硬件TLLD，否则才保守推断。`--no-infer-cutoff`不能把硬件未采集区当真实零计数，原始文件不会改写。
- **输入或背景不匹配**：检查道起点、道数、能量单位、标定及背景覆盖域，不应修改原始计数以绕过检查。

结果依赖共同尺度S、实际激发条件及薄样品近似。自吸收、分层、几何、效率、仪器信号和元素选择可能影响识别及比例；当前误差不涵盖全部系统误差。软件`supported`表示谱线证据支持，不等于材料成分已独立确证。

## 10. 文件导航

- [src/xrf_analyzer/](src/xrf_analyzer/)：程序实现和默认配置类。
- [configs/](configs/)：JSON配置；[data/](data/)：标定、灵敏度和输入说明。
- `SDD_Spectrum/`、`XRF_SDD/`：原始谱，批处理以实际存在的文件为准。
- [examples/](examples/)：合成谱和真值；[tests/](tests/)：回归测试。
- [XRF_analyze_strategy.md](XRF_analyze_strategy.md)：算法原理与实现。
- [VALIDATION.md](VALIDATION.md)：验证记录。历史结果反映当时参数，不代表已按当前Ag默认配置重新计算。
