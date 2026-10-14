#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""make_zh_figures.py — 生成中文标注版的论文正文配图。

原 ``plots.py`` / ``schematic.py`` 产出的是英文标注图（面向英文交付），
中文论文正文需要中文标注。本脚本复用同一批结果文件，输出中文版：

    fig2_2_architecture.png     CLEF 双编码器架构
    fig3_5_three_arm.png        三臂对照协议
    fig4_2_three_arm_bars.png   100-shot 三臂分组柱状图
    fig4_3_pretrain_curve.png   对比预训练收敛曲线
    fig4_4_pca.png              128 维表征 PCA
    fig4_6_budget_scan.png      标注预算扫描
    fig4_8_vib_beta.png         VIB β 扫描
    fig4_10_noise_robustness.png 模态噪声鲁棒性

同时去除图内自带标题（“Fig x-y ...”），标题职责交由 Word 图注承担。
用法：``python make_zh_figures.py``
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from evaluate import fit_pca, representation_diagnostics

# 输入 / 输出目录：相对当前运行目录（请在仓库根目录下运行）
RESULTS_DIR = "results"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "Microsoft YaHei",
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9.5,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# 与实验图统一的配色
BLUE, BLUE2, RED, GREY, DARK = "#1E4FA8", "#3D7BD9", "#D9534F", "#8B97A8", "#4A5568"
PALETTE = [BLUE, "#DD8452", "#55A868", RED]


def _load(name: str) -> Dict:
    with open(os.path.join(RESULTS_DIR, name), "r", encoding="utf-8") as f:
        return json.load(f)


def _save(fig, name: str) -> None:
    fig.savefig(os.path.join(FIGURES_DIR, name))
    plt.close(fig)
    print(f"  已保存 {name}")


# ------------------------------------------------------------------ 架构图
C_SEQ, C_MOD, C_ALIGN, C_HEAD, C_EDGE = "#DCE6F5", "#E2F0D9", "#FDE9D9", "#F2DCE6", "#4A4A4A"


def _box(ax, x, y, w, h, text, color, fontsize=9.5, bold=False, radius=0.02):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        linewidth=1.2, edgecolor=C_EDGE, facecolor=color, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold" if bold else "normal", zorder=3)


def _arrow(ax, p0, p1, color=C_EDGE, style="-|>", lw=1.4, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=13,
                                 linewidth=lw, color=color, linestyle=ls, zorder=1))


def draw_architecture() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.14, 0.955, "序列通路", ha="center", fontsize=12, fontweight="bold", color="#2F5496")
    _box(ax, 0.14, 0.86, 0.26, 0.085, "蛋白质氨基酸序列", C_SEQ)
    _box(ax, 0.14, 0.71, 0.26, 0.085, "冻结的 ESM-2\n（不引入可训练参数）", C_SEQ)
    _box(ax, 0.14, 0.55, 0.26, 0.095, "编码器 A\n2 层 Transformer（4 头）\n+ 平均池化 + 投影 MLP", C_SEQ, fontsize=9)
    _box(ax, 0.14, 0.40, 0.26, 0.075, "序列表征 $z$（128 维）", C_SEQ, bold=True)
    for y0, y1 in [(0.817, 0.755), (0.667, 0.60), (0.502, 0.44)]:
        _arrow(ax, (0.14, y0), (0.14, y1))

    ax.text(0.86, 0.955, "生物模态通路", ha="center", fontsize=12, fontweight="bold", color="#548235")
    _box(ax, 0.86, 0.86, 0.32, 0.085, "DP / MSA / 3Di / AT / SE\n+ 湿实验组学", C_MOD, fontsize=9)
    _box(ax, 0.86, 0.71, 0.32, 0.085, "各模态特征\n（20 / 128 / 64 / 768 / 32 维）", C_MOD, fontsize=8.5)
    _box(ax, 0.86, 0.55, 0.32, 0.095, "编码器 B\n浅层 MLP（融合 → 128 维）", C_MOD, fontsize=9)
    _box(ax, 0.86, 0.40, 0.32, 0.075, "模态表征 $z'$（128 维）", C_MOD, bold=True)
    for y0, y1 in [(0.817, 0.755), (0.667, 0.60), (0.502, 0.44)]:
        _arrow(ax, (0.86, y0), (0.86, y1))

    _box(ax, 0.50, 0.475, 0.30, 0.20,
         "对称 InfoNCE\n$\\mathcal{L}=\\frac{1}{2}(\\mathcal{L}_{A\\to B}+\\mathcal{L}_{B\\to A})$\n"
         "正样本：同一蛋白\n负样本：批内其余",
         C_ALIGN, fontsize=9.5)
    _arrow(ax, (0.27, 0.42), (0.35, 0.42))
    _arrow(ax, (0.73, 0.42), (0.65, 0.42))
    _arrow(ax, (0.35, 0.52), (0.27, 0.52), color="#C55A11")
    _arrow(ax, (0.65, 0.52), (0.73, 0.52), color="#C55A11")

    _arrow(ax, (0.14, 0.362), (0.14, 0.30))
    _box(ax, 0.32, 0.22, 0.36, 0.11, "下游分类器\n2 层 MLP + LayerNorm + sigmoid", C_HEAD, fontsize=9.5)
    _arrow(ax, (0.14, 0.30), (0.14, 0.22))
    ax.text(0.52, 0.075, "预训练结束后仅保留序列侧编码器：\n推理阶段只需一条蛋白序列，不再需要任何模态特征。",
            ha="center", va="center", fontsize=10, style="italic",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF7E6", edgecolor="#D6B656"))
    _save(fig, "fig2_2_architecture.png")


# ------------------------------------------------------------------ 三臂协议图
def draw_three_arm() -> None:
    fig, ax = plt.subplots(figsize=(12, 7.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    _box(ax, 0.5, 0.90, 0.88, 0.13,
         "SyntheticWorld　共享潜变量世界（$e$、$u_k$、$A$、$P_m$ 只构建一次）\n"
         "$z\\sim N(0,I)$　|　$x_{seq}=Az+\\sigma_{seq}\\epsilon$　|　$x_m=P_mz+\\eta_m\\epsilon$",
         "#EAF1FB", fontsize=10)
    ax.text(0.5, 0.80, "全部数据划分自同一世界采样（互不重叠、同分布）",
            ha="center", fontsize=9.5, style="italic")

    for x, t in [(0.16, "无标注池\n4000"), (0.42, "标注训练集\n100（上界臂 1500）"),
                 (0.66, "验证集\n150"), (0.86, "测试集\n800")]:
        _box(ax, x, 0.70, 0.24 if x < 0.5 else 0.20, 0.075, t, "#F2F2F2")
    for x in [0.16, 0.42, 0.66, 0.86]:
        _arrow(ax, (x, 0.662), (x, 0.60), color="#8C8C8C", lw=1.0, ls=":")

    y = 0.44
    _box(ax, 0.17, y, 0.28, 0.24,
         "Baseline（下界）\n冻结序列表征（平均池化）\n标注 100 条\n不使用无标注数据", "#FBE5E5", fontsize=9)
    _box(ax, 0.50, y, 0.28, 0.24,
         "CLEF（处理组）\n跨模态对比预训练\n（编码器 A ← 编码器 B 对齐）\n标注 100 条", "#E2F0D9", fontsize=9)
    _box(ax, 0.83, y, 0.28, 0.24,
         "Ceiling（上界）\n冻结序列表征（平均池化）\n标注 1500 条\n标注不再是瓶颈", "#EAF1FB", fontsize=9)
    for x in [0.17, 0.50, 0.83]:
        _arrow(ax, (x, 0.562), (x, 0.56), color="#8C8C8C", lw=1.0, ls=":")

    _box(ax, 0.5, 0.18, 0.74, 0.10,
         "在同一测试集上统一评测（分类器结构与训练协议完全一致）\n"
         "ACC · REC · PR · F1 · MCC · AUPRC　|　ARI · NMI · ASW",
         "#F2DCE6", fontsize=9.5)
    for x in [0.17, 0.50, 0.83]:
        _arrow(ax, (x, 0.32), (x, 0.24), color="#8C8C8C", lw=1.0, ls=":")

    _box(ax, 0.5, 0.055, 0.62, 0.075,
         "标注替代率　$\\Delta_{ASR}=\\dfrac{M_{CLEF}-M_{Base}}{M_{Ceil}-M_{Base}}$",
         "#FFF7E6", fontsize=11)
    _arrow(ax, (0.5, 0.128), (0.5, 0.095), color="#C55A11")
    _save(fig, "fig3_5_three_arm.png")


# ------------------------------------------------------------------ 柱状图
def plot_bars(exp_a: Dict) -> None:
    rows = exp_a["rows"]
    arms = [r["方法"] for r in rows]
    metrics = ["ACC", "F1", "MCC", "AUPRC"]
    x = np.arange(len(arms)); width = 0.2
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, [r[m] for r in rows], width, label=m,
               color=PALETTE[i], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(arms, rotation=18, ha="right")
    ax.set_ylabel("得分"); ax.set_ylim(0, 1.05)
    ax.legend(ncol=4, loc="upper left", framealpha=0.9)
    ax.axhline(rows[0]["ACC"], color="gray", linestyle=":", linewidth=1.0)
    _save(fig, "fig4_2_three_arm_bars.png")


# ------------------------------------------------------------------ 预训练曲线
def plot_curve(curves: Dict) -> None:
    curve = curves.get("CLEF-ALL", [])
    if not curve:
        print("  跳过 fig4_3：无曲线"); return
    epochs = np.arange(1, len(curve) + 1)
    rb = math.log(255)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(epochs, curve, color=BLUE, linewidth=2.0, marker="o", markersize=3, label="CLEF-ALL 的 InfoNCE")
    ax.axhline(rb, color=RED, linestyle="--", linewidth=1.4, label=f"随机基线 ln(B−1) = {rb:.2f}")
    ax.set_xlabel("预训练轮次"); ax.set_ylabel("InfoNCE 损失")
    ax.legend()
    _save(fig, "fig4_3_pretrain_curve.png")


# ------------------------------------------------------------------ PCA
def plot_pca(exp_a: Dict) -> None:
    path = os.path.join(RESULTS_DIR, "pca_data.npz")
    if not os.path.exists(path):
        print("  跳过 fig4_4：无 pca_data.npz"); return
    d = np.load(path); y_class = d["y_class"]
    rows = {r["方法"]: r for r in exp_a["rows"]}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for ax, key, title in zip(axes, ["baseline_rep", "clef_rep"],
                              ["Baseline（冻结序列表征）", "CLEF（跨模态对比表征）"]):
        rep = d[key]; xy = fit_pca(rep, 2)
        diag = representation_diagnostics(rep, y_class)
        acc = rows["Baseline" if key == "baseline_rep" else "CLEF-ALL"]["ACC"]
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=y_class, cmap="tab10", s=12, alpha=0.8, edgecolors="none")
        ax.set_title(f"{title}\nARI={diag['ARI']:.3f}　NMI={diag['NMI']:.3f}　ACC={acc:.4f}")
        ax.set_xlabel("主成分 1"); ax.set_ylabel("主成分 2")
    cbar = fig.colorbar(sc, ax=axes, ticks=range(len(np.unique(y_class))))
    cbar.set_label("真实组别（0 = 非效应蛋白，1…K = 家族）")
    _save(fig, "fig4_4_pca.png")


# ------------------------------------------------------------------ 预算扫描
def plot_budget(exp_b: Dict) -> None:
    agg = exp_b["agg"]; ns = [a["n_train"] for a in agg]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    for key, label, color in [("baseline", "Baseline", "#DD8452"), ("clef", "CLEF-ALL", BLUE),
                              ("ceiling", "Ceiling", "#55A868")]:
        ax.errorbar(ns, [a[f"{key}_ACC_mean"] for a in agg], yerr=[a[f"{key}_ACC_std"] for a in agg],
                    marker="o", capsize=4, linewidth=1.8, label=label, color=color)
    ax.set_xlabel("标注量 $n_{train}$"); ax.set_ylabel("ACC（跨种子均值 ± 标准差）")
    ax.set_title("（a）ACC 随标注预算的变化"); ax.legend()
    ax = axes[1]
    for key, label, color in [("asr_ACC", "ACC", BLUE), ("asr_MCC", "MCC", RED)]:
        ax.errorbar(ns, [a[f"{key}_mean"] for a in agg], yerr=[a[f"{key}_std"] for a in agg],
                    marker="s", capsize=4, linewidth=1.8, label=f"$\\Delta_{{ASR}}$（{label}）", color=color)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.0)
    ax.set_xlabel("标注量 $n_{train}$"); ax.set_ylabel("$\\Delta_{ASR}$")
    ax.set_title("（b）标注替代率随标注预算的变化"); ax.legend()
    _save(fig, "fig4_6_budget_scan.png")


# ------------------------------------------------------------------ VIB
def plot_vib(exp_c: Dict) -> None:
    rows = exp_c["rows"]; xs = np.arange(len(rows))
    labels = [f"$\\beta$={r['beta']}\nKL={r['final_kl']:.2f}" for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax, key, title, color in [(axes[0], "ACC", "（a）ACC 随 $\\beta$ 的变化", BLUE),
                                  (axes[1], "MCC", "（b）MCC 随 $\\beta$ 的变化", RED)]:
        vals = [r[key] for r in rows]
        bars = ax.bar(xs, vals, width=0.55, color=color, edgecolor="black", linewidth=0.5)
        ax.set_xticks(xs); ax.set_xticklabels(labels); ax.set_ylabel(key); ax.set_title(title)
        lo, hi = min(vals), max(vals); pad = max(0.01, (hi - lo) * 0.6)
        ax.set_ylim(max(0.0, lo - pad), min(1.0, hi + pad))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8.5)
    _save(fig, "fig4_8_vib_beta.png")


# ------------------------------------------------------------------ 鲁棒性
def plot_robust(exp_d: Dict) -> None:
    rows = exp_d["rows"]; scales = [r["scale"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax, key, title in [(axes[0], "ACC", "（a）模态噪声放大下的 ACC"),
                           (axes[1], "MCC", "（b）模态噪声放大下的 MCC")]:
        ax.plot(scales, [r[f"baseline_{key}"] for r in rows], marker="o", linestyle="--",
                color="#DD8452", linewidth=1.8, label="Baseline")
        ax.plot(scales, [r[f"clef_{key}"] for r in rows], marker="s", linestyle="-",
                color=BLUE, linewidth=1.8, label="CLEF-ALL")
        ax.set_xlabel("模态噪声放大倍数"); ax.set_ylabel(key); ax.set_title(title)
        ax.set_xticks(scales); ax.legend()
    _save(fig, "fig4_10_noise_robustness.png")


def main() -> None:
    need = ["exp_a.json", "exp_b.json", "exp_c.json", "exp_d.json", "curves.json"]
    missing = [n for n in need if not os.path.exists(os.path.join(RESULTS_DIR, n))]
    if missing:
        raise SystemExit(f"缺少结果文件 {missing}")
    draw_architecture()
    draw_three_arm()
    plot_bars(_load("exp_a.json"))
    plot_curve(_load("curves.json"))
    plot_pca(_load("exp_a.json"))
    plot_budget(_load("exp_b.json"))
    plot_vib(_load("exp_c.json"))
    plot_robust(_load("exp_d.json"))
    print("中文版配图全部生成完毕")


if __name__ == "__main__":
    main()
