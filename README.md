# CLEF：跨模态对比学习在生物优质标注稀缺条件下的复现与扩展研究

**课程作业声明**：本项目为**华东理工大学「人工智能技术与应用」课程**的课程作业（Course Project），
由 `rschen923` 独立完成，用于课程考核与学术交流，非生产级软件，亦非已发表论文的官方代码库。

配套报告：《基于跨模态对比学习提升生物学优质标注稀缺条件下下游数据的效率与泛化》。

---

## 一、这是什么

**CLEF**（**C**ontrastive **L**earning of **L**anguage **E**mbedding and Biological **F**eatures）
是本课程作业中**自行设计并实现的**跨模态对比学习框架名（非第三方已发表方法，目前无对应公开论文）。
其思路直接继承自 CLIP 的对称对比目标，但把「图像—文本」两模态替换为
「蛋白序列表征—多源生物学特征」两模态。

**要解决的问题**：细菌效应蛋白（effector protein）的优质标注极度稀缺——
湿实验验证一条标注的成本远高于测序一条序列。因此现实中呈现
「海量无标注序列 + 少量昂贵标注」的分布。CLEF 的目标是：
**用大量无标注的多模态数据做对比预训练，把表征质量提前垫高，
使得下游只需极少量标注即可接近「充足标注」的上界。**

**核心设计**：

- **双编码器**：Encoder A（序列通路，Transformer）把序列映射到 128 维表征 `z`；
  Encoder B（模态通路，浅层 MLP）把拼接后的生物学特征映射到同一 128 维空间 `z′`。
- **对称 InfoNCE**（温度 `τ=0.5`）：$\mathcal{L}=\tfrac{1}{2}(\mathcal{L}_{A\to B}+\mathcal{L}_{B\to A})$，
  批内其余样本作为负样本。
- **两阶段优化**：阶段一用**无标注池**做对比预训练；阶段二**冻结骨干**，
  只训练一个约 1.05 万参数的轻量分类头。
- **关键工程细节**：五个模态维度悬殊（20 / 128 / 64 / 768 / 32），
  拼接前对每个模态**逐模态 L2 归一化**，避免 768 维的注释文本模态单靠维度优势压过低维干净模态。

---

## 二、重要声明：本仓库的数据来源

> **本仓库不包含、不分发、也不下载任何真实生物学数据。**

这一点需要明确说明，避免误读：

1. **全部实验结果所用的数据，均由本仓库 `data.py` 中的 `SyntheticWorld`
   以确定性随机种子程序化合成生成**。仓库中不存在任何从公开数据库下载的文件，
   代码中也不存在任何网络请求或下载逻辑。
2. 选用合成基准而非真实数据，是**有意为之的实验设计**：合成世界可以人为设定
   「真实效应方向」与「各模态观测到的坐标子集」，从而把「多模态互补增益」这一
   待验证机制**隔离**出来单独测量，不受真实数据中同源泄漏、标注噪声、
   模态缺失等混杂因素的干扰。
3. `data.py` 中的 `load_real_esm_embeddings` 是一个**接口桩函数（stub）**：
   它在未提供本地 `.npz` 文件时**显式报错并提示**，不会静默伪造数据。
   其函数 docstring 给出了迁移到真实数据的完整路径（见下文第五节）。
4. 报告中引用的公开数据库与预训练模型（UniProt、GO、PDB、ESM-2、Foldseek、
   BioBERT、DeepSecE、CD-HIT）在本仓库中**仅作为方法学设计参照与预期迁移来源**，
   其数据/权重**未被本仓库获取或再分发**。完整出处见第七节「致谢与贡献」。

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

分步运行：

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
python run_supplementary.py J   # J：模态缺失矩阵（单/双/全模态组合）
python run_supplementary.py K   # K：十种子配对显著性检验
```

所有随机性由种子统一控制（NumPy 与 PyTorch 均设种），结果可逐位复现。

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
├── style_journal.py        # 统一期刊绘图风格（Arial + 黑体）
├── export_data_package.py  # 把全部数据与结果打包导出（注意含硬编码输出路径）
├── requirements.txt
├── results/                # 运行产物：csv / json / RUN_LOG.md
├── figures/                # 运行产物：中文标注图
└── figures_en/             # 运行产物：英文标注图
```

---

## 五、合成基准（SyntheticWorld）设定

**共享潜变量世界**：效应蛋白方向 `e`、K=5 个家族方向 `u_k`（经 QR 正交化）、
序列投影 `A`、各模态投影 `P_m` **只构建一次**；pool / train / val / test / ceiling
全部从同一个世界采样，保证各 split 语义一致且互不重叠。

**两个视角**：

- 序列视角：$x_{\text{seq}} = A z + \sigma_{\text{seq}}\varepsilon$，reshape 为 `(seq_len=16, tok_dim=32)`。
- 五个生物学模态视角（维度刻意异构）：

| 模态 | 维度 | 噪声 | 对应真实含义 |
|---|---|---|---|
| DP  | 20  | 1.0 | 进化谱 / PSSM 特征 |
| MSA | 128 | 1.2 | 多序列比对统计特征 |
| 3Di | 64  | 1.5 | Foldseek 3Di 结构字母表特征 |
| AT  | 768 | 8.0 | BioBERT 注释文本嵌入（**噪声最大**） |
| SE  | 32  | 1.3 | DeepSecE 等他模型输出 |

**模态互补性来源**：每个模态只随机观测潜变量 `z` 的一部分坐标
（默认 `mod_mask_frac=0.3`），因此**单模态信息天然不完备**，
五个模态拼接后才覆盖近乎全部坐标——这正是多模态融合增益的机理来源。

**划分规模**：无标注池 4000 / 标注训练 100（Ceiling 臂 1500）/ 验证 150 / 测试 800。

> **关于难度旋钮的校准**：报告中的名义设定为 `σ_seq=6.0, e_shift=1.2, f_shift=0.9`。
> 在该名义组合下序列信号被噪声完全淹没（下游 ACC≈0.5，退化为不可学）。
> 为使合成基准的下游难度与报告经验区间对齐（Baseline ACC≈0.85、Ceiling ACC≈0.91），
> 本实现将世界尺度校准为 `signal_scale=8.0, d_z=64`，
> 并把难度旋钮设为 `σ_seq=13.5, e_shift=6.4, f_shift=3.5`。
> **所有旋钮均可通过命令行参数覆盖**，校准不改动机理，只改变信号—噪声比。

---

## 六、实验矩阵与核心结果

### 6.1 三臂对照协议

| 臂 | 输入表征 | 标注量 | 可训练参数 | 含义 |
|---|---|---|---|---|
| **Baseline** | 冻结随机序列编码器 | 100 | ≈1.06 万 | 标注稀缺下的现状 |
| **CLEF-\*** | 对比预训练后的序列表征 | 100 | ≈10–17 万 | 待验证方法 |
| **Ceiling** | 冻结随机序列编码器 | 1500 | ≈1.06 万 | 充足标注上界 |

**标注替代率**（本作业的核心评价指标）：

$$\Delta_{\text{ASR}} = \frac{M_{\text{CLEF}} - M_{\text{Base}}}{M_{\text{Ceil}} - M_{\text{Base}}}$$

含义为：CLEF 用 100 条标注所达到的性能，相当于在基线方法中「替代」了多少比例的
额外标注投入。分母为 0 时以 `1e-8` 保护并返回 0.0。

### 6.2 实验 A：三臂对照（100-shot，seed=7）

| 臂 | 模态 | ACC | F1 | MCC | AUPRC | ARI | NMI | ASW |
|---|---|---|---|---|---|---|---|---|
| Baseline | — | 0.8475 | 0.8416 | 0.6950 | 0.927 | 0.126 | 0.181 | 0.012 |
| CLEF-DP | DP | 0.8912 | 0.8854 | 0.7819 | 0.940 | 0.183 | 0.231 | 0.046 |
| CLEF-MSA | MSA | 0.8588 | 0.8579 | 0.7212 | 0.926 | 0.210 | 0.220 | 0.050 |
| CLEF-3Di | 3Di | 0.8888 | 0.8792 | 0.7774 | 0.943 | 0.177 | 0.209 | 0.033 |
| CLEF-AT | AT | 0.8550 | 0.8457 | 0.7090 | 0.918 | 0.130 | 0.182 | 0.030 |
| CLEF-SE | SE | 0.8862 | 0.8769 | 0.7722 | 0.942 | 0.229 | 0.224 | 0.056 |
| **CLEF-ALL** | 五模态 | **0.9075** | **0.9029** | **0.8146** | **0.947** | **0.242** | **0.283** | 0.044 |
| Ceiling | —（1500 条） | 0.9113 | 0.9060 | 0.8220 | 0.961 | 0.126 | 0.181 | 0.012 |

**Δ_ASR(ACC) = 0.941，Δ_ASR(MCC) = 0.942** —— 即 CLEF 用 100 条标注，
达到了「基线上界之间性能差距」的约 94%，等效于把标注需求压缩了一个数量级以上。

三个值得注意的现象：

1. **AT 模态噪声最大（σ=8.0），单模态增益最小**（MCC 0.709），符合预期；
   但并入 CLEF-ALL 后仍贡献正向增量，说明对比目标能自动降权噪声模态。
2. **表征质量指标（ARI / NMI）的提升幅度远大于分类指标**：
   ARI 从 0.126 → 0.242（约 +92%），而 ACC 仅 +6.0%。
   这说明对比预训练主要改善的是**表征空间的几何结构**，分类指标只是其下游投影。
3. **Ceiling 的表征质量指标与 Baseline 完全相同**（ARI 0.126 / NMI 0.181 / ASW 0.012）——
   这是正确行为：Ceiling 用的是同一套冻结序列编码器，只是标注更多，
   因此分类指标更高但表征结构不变。**这恰好证明性能差距来自标注量，而非表征质量。**

### 6.3 补充实验要点（E – K）

| 实验 | 内容 | 主要发现 |
|---|---|---|
| **E** | 标注预算幂律（n ∈ {25…300}） | Δ_ASR 随标注量 **上升**：$\Delta_{\text{ASR}}(n) = 0.0691\,n^{+0.627}$（$R^2=0.589$，对数空间拟合）<br>绝对增益则**衰减**：$\Delta_{\text{MCC}}(n)\approx 0.523\,n^{-0.430}$（$R^2=0.521$）<br>→ CLEF 的相对效率优势随标注增加而放大，但边际绝对收益递减 |
| **F** | 无标注池规模（500 → 8000） | 池规模过小（500）时 Δ_ASR 为负：对比预训练反而损害性能，存在**最小池规模阈值** |
| **G** | 温度 τ ∈ {0.1, 0.2, 0.5, 1.0} | 性能对 τ 不敏感，τ=0.2 附近略优；框架鲁棒 |
| **H** | 表征维度 d_z ∈ {32, 64, 128, 256} | d_z=32 时 ACC 最高（0.955），但此时 Baseline 也达 0.881 —— **世界变简单了，不是方法变强了**；须以 Δ_ASR 而非绝对指标判定 |
| **I** | 标签噪声率 ∈ {0, 5%, 10%, 20%} | CLEF 在 20% 噪声下仍保持对 Baseline 的稳定优势，退化斜率更缓 |
| **J** | 模态缺失矩阵（单/双/全组合） | 单模态均有增益；全模态最优；任删一个模态均可降级运行（框架不依赖模态完备） |
| **K** | 十种子配对 t 检验 | ΔMCC = **0.1005 ± 0.0293**，t = 10.86，p = 1.80×10⁻⁶，95% CI [0.0840, 0.1182] |

> **关于实验 H 的判读警告**：这是本作业中最容易误读的一组。
> 表征维度 `d_z` 同时决定了「世界的本征复杂度」与「模型的容量」，
> 降低 `d_z` 会让任务本身变简单，Baseline 与 CLEF 会**同步上升**。
> 因此必须用 Δ_ASR（相对增益）而非 ACC/MCC（绝对指标）来判断方法优劣。

### 6.4 运行产物

- `results/exp_a.csv|json` —— 实验 A 八臂全指标 + Δ_ASR
- `results/exp_b.csv`、`exp_b_per_cell.csv`、`exp_b.json` —— 标注预算扫描
- `results/exp_c.*`、`exp_d.*` —— VIB β 扫描、模态噪声鲁棒性
- `results/exp_e.csv` … `exp_k.csv` —— 补充实验 E – K
- `results/supplementary.json` —— 补充实验汇总（含幂律拟合系数与显著性统计量）
- `results/curves.json` —— 对比预训练损失曲线
- `results/all_results.json` —— 主实验汇总
- `results/RUN_LOG.md` —— 每条命令、耗时与关键输出（**复现时建议先看这个**）
- `figures/`（中文标注，13 张）、`figures_en/`（英文标注，8 张）

注：`results/pca_data.npz`（约 829 KB，PCA 可视化原始表征）已被 `.gitignore` 排除，
因为它可由 `python run_experiment.py A` 重新生成。

---

## 七、致谢与贡献（Credits）

### 7.1 课程与作者

- **课程**：华东理工大学「人工智能技术与应用」
- **作者**：`rschen923`（Y17260029@mail.ecust.edu.cn）
- **性质**：课程作业，独立完成。本报告与代码未接受任何外部资助。

### 7.2 方法学参照文献

下列工作为本作业的方法学来源。**本仓库未包含其任何代码或数据**，仅参照其公开发表的思想：

| 参照对象 | 在本项目中的作用 | 出处 |
|---|---|---|
| **CLIP** | 对称对比目标（$\mathcal{L}=\tfrac{1}{2}(\mathcal{L}_{A\to B}+\mathcal{L}_{B\to A})$）的直接来源 | Radford A, et al. *Learning Transferable Visual Models From Natural Language Supervision*. ICML 2021. |
| **SimCLR / InfoNCE** | 批内负样本对比损失的原始形式；$\tau$ 温度参数的作用机理 | Chen T, et al. *A Simple Framework for Contrastive Learning of Visual Representations*. ICML 2020. |
| **VIB**（变分信息瓶颈） | 实验 C 的 β·KL 正则项；重参数化技巧 | Alemi AA, et al. *Deep Variational Information Bottleneck*. ICLR 2017. |
| **ESM-2** | 真实数据迁移时序列表征的预期来源（逐残基 1280 维） | Lin Z, et al. *Evolutionary-scale prediction of atomic-level protein structure with a language model*. **Science**, 2023. DOI: `10.1126/science.ade2574` |
| **Foldseek 3Di** | 3Di 模态（结构字母表）的设计依据 | van Kempen M, et al. *Fast and accurate protein structure search with Foldseek*. **Nature Biotechnology**, 2023. DOI: `10.1038/s41587-023-01773-0` |
| **BioBERT** | AT 模态（768 维注释文本嵌入）的设计依据 | Lee J, et al. *BioBERT: a pre-trained biomedical language representation model for biomedical text mining*. **Bioinformatics**, 2019/2020. DOI: `10.1093/bioinformatics/btz682` |
| **DeepSecE** | SE 模态（他模型输出）的设计依据 | Zhang H, et al. *DeepSecE: A Deep-Learning-Based Framework for Multiclass Prediction of Secreted Proteins in Bacteria*. **Research** (AAAS), 2023. DOI: `10.34133/research.0258` |
| **CD-HIT** | 真实数据划分时防止同源泄漏的去冗余聚类方案（本仓库未执行，因使用合成数据） | Fu L, et al. *CD-HIT: accelerated for clustering the next-generation sequencing data*. **Bioinformatics**, 2012. DOI: `10.1093/bioinformatics/bts565` |
| **UniProt / Swiss-Prot** | 真实序列与功能注释的预期来源；报告引用的规模背景（约 2.5 亿条序列） | *UniProt: the Universal Protein Knowledgebase in 2023*. **Nucleic Acids Research**, 2023. DOI: `10.1093/nar/gkac1052` |
| **GO**（基因本体） | AT 模态注释文本的预期来源 | The Gene Ontology Consortium. |
| **PDB** | 结构模态的预期来源；报告引用的规模背景（约 20 万实验结构） | RCSB Protein Data Bank. |

### 7.3 开源依赖

- **PyTorch**（对比学习与训练循环）— BSD-3-Clause
- **NumPy**（合成世界与线性代数）— BSD-3-Clause
- **Matplotlib**（全部绘图）— PSF-based
- **scikit-learn**（ARI / NMI / ASW / 幂律拟合）— BSD-3-Clause
- **pandas**（结果表格读写）— BSD-3-Clause

本作业未使用任何大语言模型生成的第三方代码库作为核心实现；
全部建模、训练与评估逻辑为本作业独立编写。

---

## 八、迁移到真实数据

`data.py` 中的 `load_real_esm_embeddings` 是预留的接口桩。迁移路径：

1. 用 ESM-2 对全部序列推理，得到逐残基表征并池化为序列表征，存为 `.npz`；
2. 按第七节表格准备五个模态特征（DP / MSA / 3Di / AT / SE），
   与序列表征按样本 ID 对齐；
3. **先做同源去冗余**（CD-HIT，建议 30%–40% 序列一致性阈值）再划分
   train / val / test，否则同源泄漏会使指标虚高；
4. 替换 `load_real_esm_embeddings` 的读取实现，其余训练与评测协议**完全不变**。

若未提供本地 `.npz`，该桩函数会显式报错并提示，不会静默生成假数据。

---

## 九、许可

MIT License。作为课程作业，代码与报告可供学习、参考与引用。

**引用本项目**（如用于课程报告或后续研究）：

```
rschen923. CLEF: Cross-modal Contrastive Learning under Scarce High-Quality
Biological Annotation [Course Project]. East China University of Science and
Technology, "Artificial Intelligence Technology and Applications" course.
https://github.com/rschen923/clef_repro
```

---

## 十、已知限制

以下限制在本作业中被明确记录，未被掩饰：

1. **合成基准的外推性有限**。SyntheticWorld 的潜变量结构是线性高斯的，
   真实蛋白序列的流形远比此复杂。本仓库的结论应当被理解为
   **机理受控验证**，而非真实数据上的性能承诺。
2. **难度旋钮经过校准**。名义参数（6.0 / 1.2 / 0.9）会导致任务不可学，
   实际使用的是校准后的取值（13.5 / 6.4 / 3.5）。校准只改变信噪比，
   但意味着「绝对指标数值」不具跨实现可比性，**只有相对量 Δ_ASR 可比**。
3. **实验 F 显示存在最小池规模阈值**，小池下对比预训练可能有害。
   迁移到真实数据时必须先扫池规模，不能默认「预训练一定有帮助」。
4. **`export_data_package.py` 内含硬编码的 Windows 输出路径**，
   在其他环境下运行前需手动修改。
