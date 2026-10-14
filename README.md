# CLEF 跨模态对比学习复现 · 标注稀缺扩展研究

本项目是报告《基于跨模态对比学习提升生物学优质标注稀缺条件下下游数据的效率与泛化》
的**可运行实验代码**：在完全可控的合成生成式基准（SyntheticWorld）上复现 CLEF 双编码器
跨模态对比学习框架，并围绕“标注稀缺”做三臂对照、标注预算扫描、VIB 扫描与模态鲁棒性四组实验。

全程 **CPU** 运行，全部实验合计约 8-12 分钟。

---

## 1. 目录结构

```
clef_repro/
├── data.py            # 数据层：SyntheticWorld 生成式基准 + 真实数据接口桩
├── model.py           # 编码器层：Encoder A / Encoder B / 对称 InfoNCE / 下游分类头 / VIB
├── train.py           # 训练层：对比预训练循环 + 下游 BCE 训练（验证集 F1 早停）
├── evaluate.py        # 评估层：全部指标 + 标注替代率 + 表征诊断
├── plots.py           # 绘图层：报告里的全部数据图
├── schematic.py       # 示意图层：报告里的 2 张结构示意图
├── run_experiment.py  # 编排层：三臂对照 / 标注预算扫描 / VIB 扫描 / 模态鲁棒性
├── requirements.txt
├── README.md
├── results/           # 运行产物：csv / json / 日志
└── figures/           # 运行产物：png
```

## 2. 环境

```bash
python -m pip install -r requirements.txt
# 若 torch 未安装（CPU 版）：
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 3. 一键复现

```bash
python run_experiment.py all     # 依次跑实验 A / B / C / D（约 8-12 分钟）
python plots.py                  # 生成报告全部数据图
python schematic.py              # 生成报告 2 张结构示意图
```

分步运行：

```bash
python run_experiment.py A       # 三臂对照（8 臂，100-shot，50 epoch）
python run_experiment.py B       # 标注预算扫描（n_train 50/100/200 × seed 42/7）
python run_experiment.py C       # VIB β 扫描（0.0/0.01/0.05）
python run_experiment.py D       # 模态噪声鲁棒性（scale 1/2/4）
```

## 4. 实验设定

### 4.1 SyntheticWorld（可控生成式基准）

- **共享潜变量世界**：效应蛋白方向 `e`、K=5 个家族方向 `u_k`（QR 正交化）、
  序列投影 `A`、各模态投影 `P_m` **只构建一次**；pool / train / val / test 全部从同一世界采样。
- **序列视角**：`x_seq = A z + σ_seq · ε`，reshape 为 `(seq_len=16, tok_dim=32)`。
- **5 个生物模态视角**（维度刻意异构）：

  | 模态 | 维度 | 噪声 | 含义 |
  |---|---|---|---|
  | DP  | 20  | 1.0 | 进化 DP/PSSM |
  | MSA | 128 | 1.2 | MSA 统计 |
  | 3Di | 64  | 1.5 | 结构 Foldseek 3Di |
  | AT  | 768 | 8.0 | 注释文本 BioBERT（**噪声最大**） |
  | SE  | 32  | 1.3 | 他模型输出 DeepSecE |

- **模态互补性**：每个模态只随机观测潜变量 `z` 的一部分坐标（`mod_mask_frac=0.5`），
  因此单模态信息不完备、五模态拼接才覆盖近乎全部坐标 —— 这是多模态融合增益的来源。
- **逐模态 L2 归一化**后再拼接，避免 768 维的 AT 以维度优势压过低维干净模态。
- **划分规模**：无标注池 4000 / 标注训练 100（Ceiling 1500）/ 验证 150 / 测试 800。

> **关于难度旋钮的校准说明**：报告名义设定为 `σ_seq=6.0, eff_shift=1.2, fam_shift=0.9`。
> 该名义组合下序列信号被噪声完全淹没（下游 ACC≈0.5，不可学）。为使合成基准的下游难度
> 与报告经验值对齐（Baseline ACC≈0.84、Ceiling ACC≈0.92），本实现将“世界尺度”校准为
> `signal_scale=8.0, d_z=64`，并把难度旋钮设为 `σ_seq=13.5, eff_shift=6.4, fam_shift=3.5`。
> 所有旋钮均可通过命令行参数覆盖。

### 4.2 模型

- **Encoder A**（序列通路）：线性输入投影 → 位置编码 → 2 层 TransformerEncoder（4 头）
  → 平均池化（压掉序列维）→ 投影 MLP → **128 维**表征 `z`。
- **Encoder B**（模态通路）：浅层 MLP，把拼接后的模态特征映射到同一 128 维空间 `z'`。
- **对称 InfoNCE**：`τ=0.5`，批内其余样本为负样本。
- **DownstreamClassifier**：2 层 MLP + LayerNorm + sigmoid（输入 128 维，约 1.05 万参数）。
- **VIBEncoder**：变分信息瓶颈，损失 = InfoNCE + β·KL(q(z|x)‖N(0,I))，暴露终态 KL 项。
- **参数量**：Baseline / 下游分类头 ≈ **1.06 万**；CLEF-ALL 全模型 ≈ **15.5 万**（冻结骨干 0 参数）。

### 4.3 四组实验

| 实验 | 内容 | 关键设定 |
|---|---|---|
| A | 三臂对照（8 臂） | 100-shot，pre_epochs=50 |
| B | 标注预算扫描 | n_train ∈ {50,100,200}，seed ∈ {42,7}，pre_epochs=30 |
| C | VIB β 扫描 | β ∈ {0.0,0.01,0.05}，pre_epochs=30 |
| D | 模态噪声鲁棒性 | noise scale ∈ {1,2,4}，pre_epochs=30 |

## 5. 结果

运行产物：

- `results/exp_a.csv|json` —— 报告表 4-1（8 臂全指标 + Δ_ASR）
- `results/exp_b.csv` / `exp_b_per_cell.csv` / `exp_b.json` —— 标注预算扫描
- `results/exp_c.csv|json` —— VIB β 扫描
- `results/exp_d.csv|json` —— 模态鲁棒性
- `results/curves.json` —— 对比预训练曲线
- `results/pca_data.npz` —— PCA 可视化原始表征
- `results/all_results.json` —— 汇总
- `results/RUN_LOG.md` —— 每条命令、耗时、关键输出
- `figures/fig4_2/3/4/6/8/10_*.png` + `figures/fig2_2_*.png` + `figures/fig3_5_*.png`

## 6. 复现性与迁移

- 所有随机性由 `seed` 统一控制（numpy / torch 均设种子）。
- 迁移到真实数据：把 `data.py` 的 `load_real_esm_embeddings` 换成读取真实
  ESM-2 表征 + 真实模态特征的实现（接口签名与迁移路径见函数 docstring），
  训练 / 评测协议完全不变。
