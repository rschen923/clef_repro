"""本文件职责（绘图层）：
把 ``run_experiment.py`` 写入 ``results/`` 的真实结果渲染成报告正文中的全部数据图：

    fig4_2   100-shot 三臂对比分组柱状图（ACC/F1/MCC/AUPRC × 8 个实验臂）
    fig4_3   CLEF-ALL 对比预训练曲线（InfoNCE vs epoch，含 ln(B-1) 随机基线虚线）
    fig4_4   128 维表征 PCA 双子图（Baseline / CLEF，按真实组别着色，标 ARI/NMI/ACC）
    fig4_6   标注预算扫描双子图（ACC vs n_train；Δ_ASR vs n_train）
    fig4_8   VIB β 扫描双子图（ACC / MCC，标注每点 KL 值）
    fig4_10  模态噪声鲁棒性双子图（ACC / MCC，Baseline 虚线 vs CLEF-ALL 实线）

图形风格由 ``style_journal`` 统一给定（Nature/Cell 投稿规范）：Arial 5–7 pt、
Okabe–Ito 色盲友好配色、细线宽、无图内标题、粗体小写 panel 标签、无背景网格。
图内**不写图号**——图号与题注由报告正文给出，避免图文编号不一致。
用法：``python plots.py``（需先运行 ``python run_experiment.py all``）。
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

from evaluate import fit_pca, representation_diagnostics
from style_journal import C, MM, apply_style, panel_label

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
FIGURES_DIR = os.path.join(HERE, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

apply_style()

# 语义色：同一含义全篇同色
C_SEQ = C["blue"]        # CLEF / 序列侧
C_BASE = C["grey"]       # Baseline（冻结序列）
C_CEIL = C["orange"]     # Ceiling（充足标注上界）
C_ALT = C["vermillion"]  # 对照指标

W1 = 150 * MM            # 单幅宽 150 mm（版心 166 mm）
W2 = 150 * MM            # 并列双栏总宽


def _load(name: str) -> Dict:
    with open(os.path.join(RESULTS_DIR, name), "r", encoding="utf-8") as f:
        return json.load(f)


def _save(fig, name: str) -> None:
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  [plots] 已保存 {path}")


# ======================================================================================
# fig4_2 · 100-shot 三臂对比分组柱状图
# ======================================================================================
def plot_fig4_2(exp_a: Dict) -> None:
    rows: List[Dict] = exp_a["rows"]
    arms = [r["方法"] for r in rows]
    metrics = ["ACC", "F1", "MCC", "AUPRC"]
    x = np.arange(len(arms))
    width = 0.20
    colors = [C_SEQ, C_CEIL, C["green"], C_ALT]

    fig, ax = plt.subplots(figsize=(W1, 62 * MM))
    for i, m in enumerate(metrics):
        vals = [r[m] for r in rows]
        ax.bar(x + (i - 1.5) * width, vals, width, label=m, color=colors[i],
               linewidth=0)
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace("CLEF-", "") for a in arms], rotation=32, ha="right")
    ax.set_ylabel("得分")
    ax.set_ylim(0, 1.02)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    _save(fig, "fig4_2_three_arm_bars.png")


# ======================================================================================
# fig4_3 · 对比预训练曲线
# ======================================================================================
def plot_fig4_3(curves: Dict) -> None:
    curve = curves.get("CLEF-ALL", [])
    if not curve:
        print("  [plots] fig4_3 跳过：无 CLEF-ALL 曲线")
        return
    epochs = np.arange(1, len(curve) + 1)
    random_base = math.log(256 - 1)  # ln(B-1)，B=batch_size=256

    fig, ax = plt.subplots(figsize=(110 * MM, 62 * MM))
    ax.plot(epochs, curve, color=C_SEQ, linewidth=0.9, marker="o", markersize=2.2,
            label="CLEF-ALL 对比损失")
    ax.axhline(random_base, color=C_BASE, linestyle="--", linewidth=0.7,
               label=f"随机基线 ln(B−1) = {random_base:.2f}")
    ax.set_xlabel("预训练轮次")
    ax.set_ylabel("InfoNCE 损失")
    ax.set_xlim(1, len(curve))
    ax.set_ylim(0, max(max(curve), random_base) * 1.15)
    ax.legend(loc="upper right")
    _save(fig, "fig4_3_pretrain_curve.png")


# ======================================================================================
# fig4_4 · 表征 PCA 双子图
# ======================================================================================
def plot_fig4_4(exp_a: Dict) -> None:
    path = os.path.join(RESULTS_DIR, "pca_data.npz")
    if not os.path.exists(path):
        print("  [plots] fig4_4 跳过：无 pca_data.npz")
        return
    d = np.load(path)
    y_class = d["y_class"]
    rows = {r["方法"]: r for r in exp_a["rows"]}
    groups = np.unique(y_class)
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(1, 2, figsize=(W2, 56 * MM))
    for k, (ax, key, letter) in enumerate(zip(
            axes, ["baseline_rep", "clef_rep"], ["a", "b"])):
        rep = d[key]
        xy = fit_pca(rep, 2)
        diag = representation_diagnostics(rep, y_class)
        acc = rows["Baseline" if key == "baseline_rep" else "CLEF-ALL"]["ACC"]
        for gi, g in enumerate(groups):
            m = y_class == g
            ax.scatter(xy[m, 0], xy[m, 1], s=3.2, alpha=0.85, linewidths=0,
                       color=cmap(gi % 10), rasterized=True)
        ax.set_xlabel("PC 1")
        if k == 0:
            ax.set_ylabel("PC 2")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.02, 0.98, f"ARI = {diag['ARI']:.3f}\nNMI = {diag['NMI']:.3f}\n"
                            f"ACC = {acc:.4f}",
                transform=ax.transAxes, fontsize=6.2, va="top", ha="left",
                linespacing=1.35)
        panel_label(ax, letter, dx=-0.06, dy=1.05)
    fig.subplots_adjust(wspace=0.08)
    _save(fig, "fig4_4_pca.png")


# ======================================================================================
# fig4_6 · 标注预算扫描
# ======================================================================================
def plot_fig4_6(exp_b: Dict) -> None:
    agg = exp_b["agg"]
    ns = [a["n_train"] for a in agg]

    fig, axes = plt.subplots(1, 2, figsize=(W2, 58 * MM))
    ax = axes[0]
    for key, label, color, ls, mk in [("baseline", "Baseline（冻结序列）", C_BASE, "--", "o"),
                                      ("clef", "CLEF-ALL", C_SEQ, "-", "s"),
                                      ("ceiling", "Ceiling（充足标注）", C_CEIL, ":", "^")]:
        mean = [a[f"{key}_ACC_mean"] for a in agg]
        std = [a[f"{key}_ACC_std"] for a in agg]
        ax.errorbar(ns, mean, yerr=std, marker=mk, markersize=3.0, capsize=1.8,
                    elinewidth=0.6, linewidth=0.9, linestyle=ls, label=label, color=color)
    ax.set_xlabel("Labeled samples  n_tr")
    ax.set_ylabel("下游 ACC")
    ax.set_xticks(ns)
    ax.set_xlim(min(ns) - 12, max(ns) + 12)
    ax.legend(loc="lower right")
    panel_label(ax, "a")

    ax = axes[1]
    for key, label, color, mk in [("asr_ACC", "ACC", C_SEQ, "s"),
                                  ("asr_MCC", "MCC", C_ALT, "o")]:
        mean = [a[f"{key}_mean"] for a in agg]
        std = [a[f"{key}_std"] for a in agg]
        ax.errorbar(ns, mean, yerr=std, marker=mk, markersize=3.0, capsize=1.8,
                    elinewidth=0.6, linewidth=0.9, label=f"Δ_ASR（{label}）",
                    color=color)
    ax.axhline(1.0, color=C_BASE, linestyle=":", linewidth=0.7)
    ax.text(max(ns), 1.0, " ceiling = 1.0", fontsize=6.0, color=C["grey"],
            va="bottom", ha="right")
    ax.set_xlabel("Labeled samples  n_tr")
    ax.set_ylabel("标注替代率 Δ_ASR")
    ax.set_xticks(ns)
    ax.set_xlim(min(ns) - 12, max(ns) + 12)
    ax.legend(loc="upper left")
    panel_label(ax, "b")
    fig.subplots_adjust(wspace=0.30)
    _save(fig, "fig4_6_budget_scan.png")


# ======================================================================================
# fig4_8 · VIB β 扫描
# ======================================================================================
def plot_fig4_8(exp_c: Dict) -> None:
    rows = exp_c["rows"]
    betas = [r["beta"] for r in rows]
    xs = np.arange(len(betas))

    fig, axes = plt.subplots(1, 2, figsize=(W2, 56 * MM))
    for ax, key, letter, color in [(axes[0], "ACC", "a", C_SEQ),
                                   (axes[1], "MCC", "b", C["purple"])]:
        vals = [r[key] for r in rows]
        ax.plot(xs, vals, marker="o", markersize=3.2, linewidth=0.9, color=color)
        for i, (b, v) in enumerate(zip(betas, vals)):
            off, ha = ((6, 5), "left") if i == 0 else ((0, 6), "center")
            ax.annotate(f"KL = {rows[i]['final_kl']:.2f}", xy=(i, v), xytext=off,
                        textcoords="offset points", ha=ha, fontsize=5.8,
                        color=C["grey"])
        lo, hi = min(vals), max(vals)
        pad = max(0.004, (hi - lo) * 0.55)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("VIB 强度 β")
        ax.set_ylabel(f"下游 {key}")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{b:g}" for b in betas])
        panel_label(ax, letter)
    fig.subplots_adjust(wspace=0.34)
    _save(fig, "fig4_8_vib_beta.png")


# ======================================================================================
# fig4_10 · 模态噪声鲁棒性
# ======================================================================================
def plot_fig4_10(exp_d: Dict) -> None:
    rows = exp_d["rows"]
    scales = [r["scale"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(W2, 56 * MM))
    for ax, key, letter in [(axes[0], "ACC", "a"), (axes[1], "MCC", "b")]:
        ax.plot(scales, [r[f"baseline_{key}"] for r in rows], marker="o",
                markersize=3.0, linestyle="--", linewidth=0.9, color=C_BASE,
                label="Baseline（冻结序列）")
        ax.plot(scales, [r[f"clef_{key}"] for r in rows], marker="s",
                markersize=3.0, linestyle="-", linewidth=0.9, color=C_SEQ,
                label="CLEF-ALL")
        ax.set_xlabel("模态噪声放大倍数 ×")
        ax.set_ylabel(f"下游 {key}")
        ax.set_xticks(scales)
        ax.set_xlim(min(scales) - 0.3, max(scales) + 0.3)
        ax.legend(loc="lower right")
        panel_label(ax, letter)
    fig.subplots_adjust(wspace=0.30)
    _save(fig, "fig4_10_noise_robustness.png")


# ======================================================================================
# 主入口
# ======================================================================================
def main() -> None:
    need = ["exp_a.json", "exp_b.json", "exp_c.json", "exp_d.json", "curves.json"]
    missing = [n for n in need if not os.path.exists(os.path.join(RESULTS_DIR, n))]
    if missing:
        raise SystemExit(f"缺少结果文件 {missing}，请先运行 python run_experiment.py all")

    exp_a = _load("exp_a.json")
    plot_fig4_2(exp_a)
    plot_fig4_3(_load("curves.json"))
    plot_fig4_4(exp_a)
    plot_fig4_6(_load("exp_b.json"))
    plot_fig4_8(_load("exp_c.json"))
    plot_fig4_10(_load("exp_d.json"))
    print("[plots] 全部数据图已生成")


if __name__ == "__main__":
    main()
