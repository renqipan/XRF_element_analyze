# 原始输入说明

2026-11-03，用户确认：`backgroud_spectrum.mca` 是**纯Pb样品激发的XRF谱**，不是通用空白/环境背景谱。

该文件与 `../XRF_SDD/Pb_50keV_20uA_back3mm.mca` 完全相同：

`SHA256 = 82ed0e0cfcd047ac64d9ee5ae9ab7fb8e6fdb5f4df198ee86db3480dd96c1f64`

默认 `background_spectrum=None`。不使用该参考谱的结果作为当前主分析；旧有带该谱的结果只表示假设“参考谱按一个自由scale叠加”时的敏感性，不能视作环境背景已被正确消除。程序拒绝将同一样品文件或其字节相同的副本用作独立Poisson背景。

若纯Pb谱代表测量装置的Pb基底，需要额外描述样品/基底几何和样品对基底荧光的能量相关衰减；一个共同scale不能自动区分样品Pb与基底Pb。纯Pb标准也不能在缺少曝光、通量、几何、质量面密度和自吸收信息时直接给出其它样品的绝对质量面密度。

`calibration.csv` 与根目录同名文件字节一致；`xrf_relative_sensitivity.csv` 与根目录同名文件字节一致。原始计数、标定点和灵敏度数值未因本次审查而修改。`characteristic_lines_energy.csv`保留为历史参考，分析中的谱线能量和吸收边取自xraylib。
