# 0.2验证记录

日期：2026-10-30（Asia/Shanghai）。Python 3.8.10、NumPy 1.24.3、SciPy 1.10.1、xraylib 4.2.0。验证环境版本见requirements.lock.txt。

- 完整回归：**80 passed**，2条xraylib SWIG依赖弃用警告；命令`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q`。
- 可安装：editable安装0.2.0成功；构建wheel并验证manual_peaks、element_prior、branch_evidence和atomic_completion模块齐全，wheel内CLI帮助可运行。
- 最新自动输出：results/shafa_current，schema 3.0，501道，deviance/dof约7.024。Ca/Fe/Hg/Pb达到默认身份支持；Cr/Ag/Pm需进一步确定。Ca Kα ΔD约315，Kβ窄/宽窗口约6.12/6.39，比例诊断相容但Kβ偏弱。
- 用户完整find_peaks示例：results/shafa_manual，9个指定中心合为8份独立证据；Fe/Hg/Pb支持，Ca待定。所有受约束中心均来自用户列表，无自动峰发现。deviance/dof约12.748。
- 两份真实谱结果仍全局失配，支持元素的比例为空；不能将诊断q当可靠面密度。未确证元素未进入主面密度文件，亦未进入分母。
- 默认分类：118元素，51 common+67 uncommon，无遗漏/重复。常见优先仅限接近证据；强罕见证据保留；严格共线模型仍不可辨识并排除定量。
- 合成Fe:Cu:Pb真值1:0.5:2，更新补线后的回收约1:0.49855:2.00792，比例份额只含Fe/Cu/Pb。
- Pb/As合成谱q真值1600/400，回收约1603.20/387.60；二者均通过支持与定量质量门控。
- 输出审计：模型严格正且等于背景+散射+元素分量；线面积非负；定量S为正；所有输入哈希匹配；原始标定、S和MCA未修改。钙局部图及主图已目视检查。
- 数据、示例、文档及历史对照结果随仓库保存；虚拟环境、缓存、构建产物和凭据由Git忽略。

统计区间不包含源/效率/自吸收/分层/背景选择/截断推断/先验与元素选择系统误差。身份supported是软件证据状态，不是经标准样品校准的概率或成分确证。

发布期间新增的`XRF_SDD/`中20份原始MCA亦随私有仓库保存；只检查PMCA文件标记及上传字节完整性，本轮没有对其余采集条件声称定量验证。
