# CLEF 复现与扩展研究：标注稀缺条件下的跨模态对比学习

**课程作业声明**：本项目为**华东理工大学「人工智能技术与应用」课程作业**，
由陈润声（`rschen923`，Y17260029@mail.ecust.edu.cn）与吴思琪完成。

配套报告：《基于跨模态对比学习提升生物学优质标注稀缺条件下下游数据的效率与泛化》。

---

## 一、项目定位与 CLEF 出处

**CLEF**（**C**ontrastive-learning of **L**anguage **E**mbedding and Biological **F**eatures）
是已发表的方法，非本项目提出。原始出处：

> Peng Y, Wu J, Sun Y, Zhang Y, Wang Q, Shao S.
> *Contrastive-learning of language embedding and biological features for cross modality
> encoding and effector prediction*. **Nature Communications**, 2025, 16:1299.
> DOI `10.1038/s41467-025-56526-1` · 官方仓库 <https://github.com/PengYueECUST1997/CLEF>

本仓库是对 CLEF 的**课程作业性质的复现与扩展研究**，非官方实现：
沿用其双编码器结构与对称对比目标，改用可控的合成基准，
围绕「标注稀缺」增设三臂对照与十一组实验（A–K），
以标注替代率 Δ_ASR（CLEF 相对于基线的增益占「基线—充足标注上界」差距的比例）衡量数据效率。

CLEF 的原始实现、真实数据集与预训练权重请见上述官方仓库，本仓库未包含其任何数据或权重。

---

## 二、数据来源

全部实验结果基于 `data.py` 中的合成生成式基准 `SyntheticWorld`，由固定随机种子程序化生成；
本仓库不包含、不下载、也不分发任何真实生物学数据。

`data.py` 的 `load_real_esm_embeddings` 为真实数据的接入接口（桩），
未提供本地 `.npz` 时会显式报错，不会静默伪造数据。

---

## 三、快速开始

全程 CPU 可跑，无需 GPU。

```bash
git clone https://github.com/rschen923/clef_repro.git
cd clef_repro

python -m pip install -r requirements.txt
# 若 torch 未安装，装 CPU 版：
# python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

python run_experiment.py all        # 主实验 A / B / C / D
python run_supplementary.py all     # 补充实验 E – K
```

结果写入 `results/`，配图写入 `figures/`。

单组运行：

```bash
python run_experiment.py A      # 三臂对照
python run_experiment.py B      # 标注预算扫描
python run_experiment.py C      # VIB β 扫描
python run_experiment.py D      # 模态噪声鲁棒性

python run_supplementary.py E   # 标注预算幂律
python run_supplementary.py F   # 无标注池规模缩放
python run_supplementary.py G   # 温度消融
python run_supplementary.py H   # 表征维度消融
python run_supplementary.py I   # 标签噪声鲁棒性
python run_supplementary.py J   # 模态缺失矩阵
python run_supplementary.py K   # 十种子配对显著性检验
```

绘图（可选）：

```bash
python plots.py                     # 主实验配图
python plots_supplementary.py       # 补充实验配图
python make_zh_figures.py           # 中文标注版配图
python schematic.py                 # 结构示意图
```

随机性由种子统一控制，结果可逐位复现。主实验全量约 6 分钟，补充实验约 20–30 分钟。

---

## 四、目录结构

```
clef_repro/
├── data.py                 # 数据层：合成基准 + 真实数据接入接口
├── model.py                # 模型层：双编码器、对比损失、下游分类头
├── train.py                # 训练层：对比预训练 + 下游分类
├── evaluate.py             # 评估层：指标计算
├── run_experiment.py       # 实验 A – D
├── run_supplementary.py    # 实验 E – K
├── plots.py                # 主实验配图
├── plots_supplementary.py  # 补充实验配图
├── schematic.py            # 结构示意图
├── make_zh_figures.py      # 中文标注版配图
├── style_journal.py        # 绘图风格
├── requirements.txt
└── results/                # 运行产物
```

---

## 五、运行产物

运行后写入 `results/`：

| 文件 | 内容 |
| :-- | :-- |
| `exp_a.csv` / `exp_a.json` | 实验 A 全指标 + Δ_ASR |
| `exp_b*.csv|json`、`exp_c.*`、`exp_d.*` | 实验 B / C / D |
| `exp_e.csv` … `exp_k.csv` | 补充实验 E – K |
| `supplementary.json` | 补充实验汇总 |
| `curves.json` | 预训练损失曲线 |
| `all_results.json` | 主实验汇总 |
| `RUN_LOG.md` | 命令、耗时与关键输出 |

---

## 六、致谢与引用

### 6.1　原始方法 CLEF

本项目的建模思路来源于以下工作，使用或引用本仓库时请一并引用原始论文：

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
| 论文 | DOI `10.1038/s41467-025-56526-1`；PMID `39900608`；PMCID `PMC11791096` |
| 官方代码仓库 | <https://github.com/PengYueECUST1997/CLEF> |
| Zenodo 归档 | DOI `10.5281/zenodo.14587661` |
| 研究单位 | 华东理工大学生物反应器工程全国重点实验室 |

### 6.2　课程与作者

华东理工大学「人工智能技术与应用」课程作业。
作者：陈润声（Y17260029@mail.ecust.edu.cn）、吴思琪。

### 6.3　方法学参照

ESM-2、Foldseek、BioBERT、DeepSecE、CD-HIT、UniProt 与 GO 为本合成基准各模态的设计参照；
对比学习目标参照 CLIP 与 SimCLR，实验 C 的信息瓶颈正则参照 VIB。

---

## 七、许可

MIT License。

引用本项目：

```
陈润声, 吴思琪. CLEF 复现与扩展研究：标注稀缺条件下的跨模态对比学习
[课程作业]. 华东理工大学「人工智能技术与应用」课程.
https://github.com/rschen923/clef_repro
```

---

## 八、已知限制

1. 合成基准的外推性有限，结论应理解为机理受控验证，而非真实数据上的性能承诺。
2. 实验 B 的 Δ_ASR 可能大于 1，成因见 `results/RUN_LOG.md` 第 6 节。
