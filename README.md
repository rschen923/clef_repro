# CLEF 复现与扩展研究：标注稀缺条件下的跨模态对比学习

**课程作业声明**：本项目为**华东理工大学「人工智能技术与应用」课程作业**，
由陈润声（`rschen923`，Y17260029@mail.ecust.edu.cn）与吴思琪完成。

配套报告：《基于跨模态对比学习提升生物学优质标注稀缺条件下下游数据的效率与泛化》。

---

## 一、本项目与 CLEF 的关系

**CLEF**（**C**ontrastive-learning of **L**anguage **E**mbedding and Biological **F**eatures）
是已发表的方法，非本项目提出。其原始出处为：

> Peng Y, Wu J, Sun Y, Zhang Y, Wang Q, Shao S.
> *Contrastive-learning of language embedding and biological features for cross modality
> encoding and effector prediction*. **Nature Communications**, 2025.
> DOI: `10.1038/s41467-025-56526-1`
> 官方代码仓库：<https://github.com/PengYueECUST1997/CLEF>（MIT 许可）
> Zenodo 归档：DOI `10.5281/zenodo.14587661`

CLEF 的核心思路是：以对比学习把**蛋白语言模型（PLM）表征**与**多源生物学特征**
对齐到同一表征空间，从而提升 III / IV / VI 型分泌效应蛋白（T3SE / T4SE / T6SE）的预测性能。

**本仓库不是 CLEF 的官方实现，而是课程作业性质的独立复现与扩展研究**：

- 沿用 CLEF 的双编码器结构与对称 InfoNCE 目标；
- 把数据替换为**可控的合成生成式基准**（`SyntheticWorld`），以便在无混杂因素干扰的条件下
  隔离测量「跨模态互补增益」这一机制；
- 围绕「标注稀缺」增设三臂对照与十一组实验（A–K），并以**标注替代率 Δ_ASR** 量化数据效率。

若需 CLEF 的原始实现、真实数据集与预训练权重，请移步上述官方仓库；
本仓库不复制、不分发其任何数据或权重。

### 1.1　框架要点

- **双编码器**：Encoder A（序列通路，2 层 Transformer）输出 128 维表征 `z`；
  Encoder B（模态通路，浅层 MLP）把拼接后的生物学特征映射到同一空间 `z′`。
- **对称 InfoNCE**（温度 `τ=0.5`）：$\mathcal{L}=\tfrac{1}{2}(\mathcal{L}_{A\to B}+\mathcal{L}_{B\to A})$，
  批内其余样本作为负样本。
- **两阶段训练**：阶段一在无标注池上做对比预训练；阶段二冻结骨干，
  只训练一个约 1.05 万参数的轻量分类头。
- **逐模态 L2 归一化**：五个模态维度悬殊（20 / 128 / 64 / 768 / 32），
  拼接前先各自归一化，避免 768 维的注释文本模态单靠维度优势压过低维干净模态。

### 1.2　标注替代率

$$\Delta_{\text{ASR}} = \frac{M_{\text{CLEF}} - M_{\text{Base}}}{M_{\text{Ceil}} - M_{\text{Base}}}$$

其中 $M_{\text{Base}}$ 为标注稀缺基线，$M_{\text{Ceil}}$ 为充足标注上界。
该量表示：CLEF 在给定标注预算下所达到的性能，相当于替代了多少比例的额外标注投入。
分母为 0 时以 `1e-8` 保护并返回 0.0。

---

## 二、数据来源声明

**本仓库不包含、不下载、也不分发任何真实生物学数据。**

- 全部实验结果基于 `data.py` 中的 `SyntheticWorld`，由固定随机种子程序化合成生成。
- 选用合成基准是为了实现**机制隔离**与**逐位可复现**：合成世界可人为设定真实效应方向
  与各模态观测到的坐标子集，从而排除同源泄漏、标注噪声、模态缺失等混杂因素。
- `data.py` 中的 `load_real_esm_embeddings` 是**接口桩函数**，
  未提供本地 `.npz` 时显式报错，不会静默伪造数据。其 docstring 给出迁移路径。
- CLEF 原文所用数据（如 CRISPRi 测序 NCBI `PRJNA1113837`、胞外蛋白组 PRIDE `PXD050280`）
  与官方仓库中的真实数据集，**均未被本仓库获取或再分发**。

---

## 三、快速开始

全程 **CPU** 可跑，无需 GPU。

```bash
git clone https://github.com/rschen923/clef_repro.git
cd clef_repro

python -m pip install -r requirements.txt
# 若 torch 未安装，装 CPU 版：
# python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

python run_experiment.py all        # 主实验 A / B / C / D
python run_supplementary.py all     # 补充实验 E – K
python plots.py                     # 主实验配图
python plots_supplementary.py       # 补充实验配图
python make_zh_figures.py           # 中文标注版配图
```

单组运行：

```bash
python run_experiment.py A      # A：三臂对照（8 臂，100-shot，50 epoch）
python run_experiment.py B      # B：标注预算扫描
python run_experiment.py C      # C：VIB β 扫描
python run_experiment.py D      # D：模态噪声鲁棒性

python run_supplementary.py E   # E：标注预算幂律（Δ_ASR 拟合）
python run_supplementary.py F   # F：无标注池规模缩放
python run_supplementary.py G   # G：InfoNCE 温度消融
python run_supplementary.py H   # H：表征维度消融
python run_supplementary.py I   # I：标签噪声鲁棒性
python run_supplementary.py J   # J：模态缺失矩阵
python run_supplementary.py K   # K：十种子配对显著性检验
```

全部随机性由种子统一控制（NumPy 与 PyTorch 均设种），结果可逐位复现。

---

## 四、目录结构

```
clef_repro/
├── data.py                 # 数据层：SyntheticWorld 合成基准 + 真实数据接口桩
├── model.py                # 模型层：Encoder A / B、对称 InfoNCE、下游分类头、VIB
├── train.py                # 训练层：对比预训练循环 + 下游 BCE（验证集 F1 早停）
├── evaluate.py             # 评估层：全部指标 + 标注替代率 Δ_ASR + 表征诊断
├── run_experiment.py       # 编排层：实验 A – D
├── run_supplementary.py    # 编排层：实验 E – K
├── plots.py                # 绘图层：主实验数据图
├── plots_supplementary.py  # 绘图层：补充实验数据图
├── schematic.py            # 示意图层：框架结构示意图
├── make_zh_figures.py      # 中文标注版配图（复用同一批结果文件）
├── style_journal.py        # 统一期刊绘图风格
├── export_data_package.py  # 数据与结果打包导出（含硬编码输出路径，跨平台需手动改）
├── requirements.txt
├── results/                # 运行产物：csv / json / RUN_LOG.md
├── figures/                # 运行产物：中文标注图
└── figures_en/             # 运行产物：英文标注图
```

---

## 五、合成基准设定

**共享潜变量世界**：效应蛋白方向 `e`、K=5 个家族方向 `u_k`（QR 正交化）、
序列投影 `A`、各模态投影 `P_m` 只构建一次；pool / train / val / test / ceiling
全部从同一世界采样，保证语义一致且互不重叠。

**两个视角**：序列视角 $x_{\text{seq}} = A z + \sigma_{\text{seq}}\varepsilon$，
reshape 为 `(seq_len=16, tok_dim=32)`；五个生物学模态视角如下（维度刻意异构）。

| 模态 | 维度 | 噪声 | 对应真实含义 |
|---|---|---|---|
| DP  | 20  | 1.0 | 进化谱 / PSSM 特征 |
| MSA | 128 | 1.2 | 多序列比对统计特征 |
| 3Di | 64  | 1.5 | Foldseek 3Di 结构字母表特征 |
| AT  | 768 | 8.0 | BioBERT 注释文本嵌入 |
| SE  | 32  | 1.3 | 他模型输出（如 DeepSecE） |

**模态互补性来源**：每个模态只随机观测潜变量 `z` 的一部分坐标（默认 `mod_mask_frac=0.3`），
单模态信息天然不完备，五模态拼接后才覆盖近乎全部坐标。

**划分规模**：无标注池 4000 / 标注训练 100（Ceiling 臂 1500）/ 验证 150 / 测试 800。

> **难度旋钮经过校准**。报告名义设定为 `σ_seq=6.0, e_shift=1.2, f_shift=0.9`，
> 该组合下信号被噪声完全淹没（ACC≈0.5，不可学）。为使下游难度落在合理区间，
> 本实现将世界尺度校准为 `signal_scale=8.0, d_z=64`，
> 难度旋钮设为 `σ_seq=13.5, e_shift=6.4, f_shift=3.5`。
> 所有旋钮均可通过命令行参数覆盖。

---

## 六、运行产物

- `results/exp_a.csv|json` —— 实验 A 八臂全指标 + Δ_ASR
- `results/exp_b*.csv|json`、`exp_c.*`、`exp_d.*` —— 标注预算 / VIB β / 模态噪声
- `results/exp_e.csv` … `exp_k.csv` —— 补充实验 E – K
- `results/supplementary.json` —— 补充实验汇总（含幂律拟合系数与显著性统计量）
- `results/curves.json` —— 对比预训练损失曲线
- `results/all_results.json` —— 主实验汇总
- `results/RUN_LOG.md` —— 每条命令、耗时与关键输出（复现时建议先看）
- `figures/`（中文标注，13 张）、`figures_en/`（英文标注，8 张）

注：`results/pca_data.npz`（约 829 KB）已由 `.gitignore` 排除，可由
`python run_experiment.py A` 重新生成。

---

## 七、致谢与贡献（Credits）

### 7.1　原始方法：CLEF

本项目的全部建模思路来源于以下已发表工作。若本仓库对您有帮助，请引用原始论文：

```bibtex
@article{peng2025clef,
  title   = {Contrastive-learning of language embedding and biological features
             for cross modality encoding and effector prediction},
  author  = {Peng, Yue and Wu, Junze and Sun, Yi and Zhang, Yuanxing
             and Wang, Qiyao and Shao, Shuai},
  journal = {Nature Communications},
  year    = {2025},
  doi     = {10.1038/s41467-025-56526-1}
}
```

| 项目 | 地址 / 标识 |
| :-- | :-- |
| 论文 | *Nature Communications* 2025，DOI `10.1038/s41467-025-56526-1`；PMID `39900608`；PMCID `PMC11791096` |
| 官方代码仓库 | <https://github.com/PengYueECUST1997/CLEF>（MIT 许可） |
| Zenodo 归档 | DOI `10.5281/zenodo.14587661`（MIT 许可） |
| 研究单位 | 华东理工大学生物反应器工程全国重点实验室（王启要教授课题组） |

特别致谢 CLEF 作者团队将代码、数据与预训练权重以 MIT 许可公开，
使本复现研究得以在其方法基础上开展标注稀缺条件下的扩展实验。

### 7.2　课程与作者

- **课程**：华东理工大学「人工智能技术与应用」
- **作者**：陈润声（Y17260029@mail.ecust.edu.cn）、吴思琪
- **性质**：课程作业。本仓库不含任何外部资助产出。

### 7.3　方法学参照文献

| 参照对象 | 作用 | 出处 |
| :-- | :-- | :-- |
| ESM-2 | 序列表征的预期来源 | Lin Z, 等. *Science*, 2023. DOI `10.1126/science.ade2574` |
| Foldseek 3Di | 3Di 模态设计依据 | van Kempen M, 等. *Nature Biotechnology*, 2023. DOI `10.1038/s41587-023-01773-0` |
| BioBERT | AT 模态设计依据 | Lee J, 等. *Bioinformatics*, 2019/2020. DOI `10.1093/bioinformatics/btz682` |
| DeepSecE | SE 模态设计依据 | Zhang H, 等. *Research*, 2023. DOI `10.34133/research.0258` |
| CD-HIT | 真实数据去冗余方案（本仓库未执行） | Fu L, 等. *Bioinformatics*, 2012. DOI `10.1093/bioinformatics/bts565` |
| UniProt / Swiss-Prot | 序列与注释的预期来源 | *Nucleic Acids Research*, 2023. DOI `10.1093/nar/gkac1052` |
| GO（基因本体） | AT 模态注释来源 | The Gene Ontology Consortium |
| PDB | 结构模态的预期来源 | RCSB Protein Data Bank |

对比学习目标本身的方法学来源：CLIP（Radford A, 等. ICML 2021，对称对比目标）、
SimCLR / InfoNCE（Chen T, 等. ICML 2020）、VIB（Alemi AA, 等. ICLR 2017，实验 C 的 KL 正则项）。

### 7.4　开源依赖

PyTorch、NumPy、Matplotlib、scikit-learn、pandas。
许可证分别为 BSD-3-Clause、BSD-3-Clause、PSF-based、BSD-3-Clause、BSD-3-Clause。

---

## 八、迁移到真实数据

1. 用 ESM-2 推理得到序列表征（官方权重见 <https://github.com/facebookresearch/esm>），存为 `.npz`；
2. 按第五节表格准备五个模态特征并与序列按样本 ID 对齐；
3. **先做同源去冗余**（CD-HIT，建议 30%–40% 一致性阈值）再划分 train / val / test，
   否则同源泄漏会使指标虚高；
4. 替换 `data.py` 中 `load_real_esm_embeddings` 的读取实现，
   其余训练与评测协议完全不变。

---

## 九、许可

MIT License。

**引用本项目**：

```
陈润声, 吴思琪. CLEF 复现与扩展研究：标注稀缺条件下的跨模态对比学习
[课程作业]. 华东理工大学「人工智能技术与应用」课程.
https://github.com/rschen923/clef_repro
```

---

## 十、已知限制

1. **合成基准的外推性有限**。SyntheticWorld 的潜变量结构是线性高斯的，
   真实蛋白序列的流形远比此复杂。本仓库结论应被理解为**机理受控验证**，
   而非真实数据上的性能承诺。
2. **`export_data_package.py` 内含硬编码的 Windows 输出路径**，
   在其他环境下运行前需手动修改。
