#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""plots_supplementary.py — 把实验 E-K 的真实结果渲染成 5 张报告用图。

产物（写入 clef_repro/figures；如需同时输出到另一目录，
设置环境变量 CLEF_EXTRA_FIG_DIR 指向该目录即可）：
    fig_powerlaw.png    双栏   Δ_ASR 幂律拟合 + 绝对增益曲线
    fig_scale.png       双栏   无标注池规模缩放（ACC 三条线 + Δ_ASR）
    fig_ablation.png    三栏   温度 / 表征维度 / 标签噪声 三联消融
    fig_signif.png      双栏   模态缺失矩阵 + 配对 bootstrap 分布
    fig_overview.png    2×3    A-K 总览

图形风格由 ``style_journal`` 统一给定（Nature/Cell 投稿规范）：Arial + 黑体、
Okabe–Ito 色盲友好配色、细线宽、无图内大标题、粗体小写 panel 标签、无背景网格。
图内**不写图号**——图号与题注由报告正文给出。
用法：``python plots_supplementary.py``（需先跑 ``python run_supplementary.py all``）。
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from style_journal import C, MM, apply_style, panel_label

# 输入 / 输出目录：相对当前运行目录（请在仓库根目录下运行）
RESULTS = "results"
FIG1 = "figures"
# 可选的第二输出目录：由环境变量 CLEF_EXTRA_FIG_DIR 指定，默认不启用。
# （原实现硬编码写入仓库上级目录 ppt_clef_deck/assets，在其他机器上不可用，已改为可选。）
FIG2 = os.environ.get("CLEF_EXTRA_FIG_DIR", "").strip()
os.makedirs(FIG1, exist_ok=True)
OUT_DIRS = [FIG1] + ([FIG2] if FIG2 else [])
for _d in OUT_DIRS:
    os.makedirs(_d, exist_ok=True)

apply_style()

C_SEQ = C["blue"]        # CLEF
C_BASE = C["grey"]       # Baseline
C_CEIL = C["orange"]     # Ceiling
C_ALT = C["vermillion"]
C_ALT2 = C["purple"]

W2 = 150 * MM            # 双栏总宽 150 mm
W3 = 150 * MM            # 三栏总宽 150 mm
H_ROW = 52 * MM


def _load() -> dict:
    with open(os.path.join(RESULTS, "supplementary.json"), encoding="utf-8") as f:
        return json.load(f)


def _save(fig, name: str) -> None:
    for p in OUT_DIRS:
        fig.savefig(os.path.join(p, name))
    plt.close(fig)
    print("saved", name)


def _log_axis(ax, ticks, labels=None) -> None:
    """对数横轴 + 显式刻度标签（关闭次刻度标签，避免重叠）。"""
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks(list(ticks))
    ax.set_xticklabels(labels or [str(int(v)) for v in ticks])


# ======================================================================================
def plot_powerlaw(supp: dict) -> None:
    agg = supp["E"]["agg"]
    n = np.array([r["n_train"] for r in agg], float)
    asr = np.array([r["asr_MCC_mean"] for r in agg], float)
    asr_sd = np.array([r["asr_MCC_std"] for r in agg], float)
    gap = np.array([r["gap_MCC_mean"] for r in agg], float)
    gap_sd = np.array([r["gap_MCC_std"] for r in agg], float)

    fig, axes = plt.subplots(1, 2, figsize=(W2, H_ROW))
    ax = axes[0]
    ax.errorbar(n, asr, yerr=asr_sd, marker="o", markersize=3.0, linewidth=0.9,
                color=C_SEQ, capsize=1.8, elinewidth=0.6, label="Δ_ASR (MCC)")
    fit = supp["E"].get("powerlaw_fit", {})
    if fit.get("a") and fit.get("b") is not None:
        xx = np.linspace(n.min(), n.max(), 200)
        yy = fit["a"] * xx ** (-fit["b"])
        ax.plot(xx, yy, "--", linewidth=0.7, color=C_ALT,
                label=f"幂律拟合  R² = {fit['r2']:.3f}")
    ax.axhline(1.0, color=C_BASE, linestyle=":", linewidth=0.7)
    ax.text(n.max(), 1.0, " Δ_ASR = 1 ", fontsize=6.0, color=C["grey"],
            va="bottom", ha="right")
    _log_axis(ax, n)
    ax.set_xlabel("标注量  n")
    ax.set_ylabel("标注替代率 Δ_ASR (MCC)")
    ax.legend(loc="upper left")
    panel_label(ax, "a")

    ax = axes[1]
    ax.errorbar(n, gap, yerr=gap_sd, marker="s", markersize=3.0, linewidth=0.9,
                color=C_ALT, capsize=1.8, elinewidth=0.6, label="绝对增益 Δ (MCC)")
    ax.axhline(0, color=C_BASE, linestyle=":", linewidth=0.7)
    _log_axis(ax, n)
    ax.set_xlabel("标注量  n")
    ax.set_ylabel("绝对增益  MCC(CLEF) − MCC(Base)")
    ax.legend(loc="upper right")
    panel_label(ax, "b")
    fig.subplots_adjust(wspace=0.32)
    _save(fig, "fig_powerlaw.png")


def plot_scale(supp: dict) -> None:
    agg = supp["F"]["agg"]
    npool = np.array([r["n_pool"] for r in agg], float)
    clef = np.array([r["clef_ACC_mean"] for r in agg], float)
    base = np.array([r["baseline_ACC_mean"] for r in agg], float)
    ceil = np.array([r["ceiling_ACC_mean"] for r in agg], float)
    csd = np.array([r["clef_ACC_std"] for r in agg], float)
    bsd = np.array([r["baseline_ACC_std"] for r in agg], float)
    asr = np.array([r["asr_MCC_mean"] for r in agg], float)
    asd = np.array([r["asr_MCC_std"] for r in agg], float)

    fig, axes = plt.subplots(1, 2, figsize=(W2, H_ROW))
    ax = axes[0]
    ax.errorbar(npool, clef, yerr=csd, marker="o", markersize=3.0, linewidth=0.9,
                color=C_SEQ, capsize=1.8, elinewidth=0.6, label="CLEF-ALL")
    ax.errorbar(npool, base, yerr=bsd, marker="^", markersize=3.0, linewidth=0.9,
                color=C_BASE, capsize=1.8, elinewidth=0.6, label="Baseline（100 条标注）")
    ax.errorbar(npool, ceil, marker="s", markersize=3.0, linewidth=0.9,
                color=C_CEIL, label="Ceiling（1 500 条标注）")
    _log_axis(ax, npool)
    ax.set_xlabel("无标注池规模  n_pool")
    ax.set_ylabel("下游 ACC")
    ax.legend(loc="lower right")
    panel_label(ax, "a")

    ax = axes[1]
    ax.errorbar(npool, asr, yerr=asd, marker="D", markersize=3.0, linewidth=0.9,
                color=C_ALT, capsize=1.8, elinewidth=0.6, label="Δ_ASR (MCC)")
    ax.axhline(1.0, color=C_BASE, linestyle=":", linewidth=0.7)
    _log_axis(ax, npool)
    ax.set_xlabel("无标注池规模  n_pool")
    ax.set_ylabel("标注替代率 Δ_ASR (MCC)")
    ax.legend(loc="lower right")
    panel_label(ax, "b")
    fig.subplots_adjust(wspace=0.32)
    _save(fig, "fig_scale.png")


def plot_ablation(supp: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(W3, H_ROW))

    if "G" in supp:
        agg = supp["G"]["agg"]
        tau = [f"{r['tau']:g}" for r in agg]
        mcc = [r["MCC_mean"] for r in agg]
        sd = [r["MCC_std"] for r in agg]
        ax = axes[0]
        xs = np.arange(len(tau))
        ax.bar(xs, mcc, color=C_SEQ, width=0.58, linewidth=0)
        ax.errorbar(xs, mcc, yerr=sd, fmt="none", ecolor="black", capsize=1.8,
                    elinewidth=0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels(tau)
        ax.set_xlabel("InfoNCE 温度 τ")
        ax.set_ylabel("下游 MCC")
        ax.set_ylim(min(mcc) - 0.06, max(mcc) + 0.05)
        panel_label(ax, "a")

    if "H" in supp:
        agg = supp["H"]["agg"]
        dz = [str(int(r["d_z"])) for r in agg]
        mcc = [r["MCC_mean"] for r in agg]
        sd = [r["MCC_std"] for r in agg]
        ax = axes[1]
        xs = np.arange(len(dz))
        ax.bar(xs, mcc, color=C_ALT2, width=0.58, linewidth=0)
        ax.errorbar(xs, mcc, yerr=sd, fmt="none", ecolor="black", capsize=1.8,
                    elinewidth=0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels(dz)
        ax.set_xlabel("跨模态表征维度  d_z")
        ax.set_ylabel("下游 MCC")
        ax.set_ylim(min(mcc) - 0.06, max(mcc) + 0.05)
        panel_label(ax, "b")

    if "I" in supp:
        agg = supp["I"]["agg"]
        x = np.arange(len(agg))
        lab = [f"{float(r['rate'])*100:.0f}" for r in agg]
        b = [r["baseline_MCC_mean"] for r in agg]
        c = [r["clef_MCC_mean"] for r in agg]
        ax = axes[2]
        ax.plot(x, b, marker="^", markersize=3.0, linewidth=0.9, color=C_BASE,
                label="Baseline")
        ax.plot(x, c, marker="o", markersize=3.0, linewidth=0.9, color=C_SEQ,
                label="CLEF-ALL")
        ax.set_xticks(x)
        ax.set_xticklabels(lab)
        ax.set_xlabel("训练标签翻转率 (%)")
        ax.set_ylabel("下游 MCC")
        ax.legend(loc="lower left")
        panel_label(ax, "c")

    fig.subplots_adjust(wspace=0.46)
    _save(fig, "fig_ablation.png")


def plot_signif(supp: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(W2, H_ROW))

    if "J" in supp:
        agg = supp["J"]["agg"]
        names = [r["combo"] for r in agg]
        acc = [r["ACC_mean"] for r in agg]
        sd = [r["ACC_std"] for r in agg]
        order = np.argsort(acc)
        names = [names[i] for i in order]
        acc = [acc[i] for i in order]
        sd = [sd[i] for i in order]
        colors = [C_ALT if n == "ALL" else (C_SEQ if n.startswith("LOO") else C["sky"])
                  for n in names]
        ax = axes[0]
        y = np.arange(len(names))
        ax.barh(y, acc, color=colors, height=0.60, linewidth=0)
        ax.errorbar(acc, y, xerr=sd, fmt="none", ecolor="black", capsize=1.6,
                    elinewidth=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=6.0)
        ax.set_xlabel("下游 ACC")
        ax.set_xlim(min(acc) - 0.05, max(acc) + 0.035)
        panel_label(ax, "a", dx=-0.62)

    if "K" in supp:
        pairs = supp["K"]["pairs"]
        d = np.array([p["diff_MCC"] for p in pairs], float)
        rng = np.random.default_rng(20261013)
        boot = d[rng.integers(0, d.size, size=(20000, d.size))].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        ax = axes[1]
        ax.hist(boot, bins=42, color=C_SEQ, alpha=0.85, linewidth=0)
        ax.axvline(lo, color=C_ALT, linewidth=0.9, linestyle="--")
        ax.axvline(hi, color=C_ALT, linewidth=0.9, linestyle="--")
        ax.axvline(0.0, color=C_BASE, linewidth=0.9, linestyle=":")
        p = supp["K"]["stat_MCC"]["p_value"]
        ax.set_xlabel("配对差值 ΔMCC 的自助法均值")
        ax.set_ylabel("频次")
        ax.text(0.03, 0.95, f"95% CI = [{lo:.4f}, {hi:.4f}]\np = {p:.2e}（配对 t 检验）",
                transform=ax.transAxes, fontsize=6.2, va="top", ha="left",
                linespacing=1.4)
        panel_label(ax, "b")
    fig.subplots_adjust(wspace=0.34)
    _save(fig, "fig_signif.png")


def plot_overview(supp: dict) -> None:
    """A–K 总览：6 个子图统一风格，panel 标签为 a–f。"""
    fig, axes = plt.subplots(2, 3, figsize=(150 * MM, 84 * MM))
    fig.subplots_adjust(hspace=0.72, wspace=0.42)
    letters = [["a", "b", "c"], ["d", "e", "f"]]

    # (0,0) A 组三臂
    try:
        a = json.load(open(os.path.join(RESULTS, "exp_a.json"), encoding="utf-8"))
        rows = a["rows"]
        keep = ["Baseline", "CLEF-DP", "CLEF-3Di", "CLEF-AT", "CLEF-ALL", "Ceiling"]
        vals = [next(r["ACC"] for r in rows if r["方法"] == k) for k in keep]
        cols = [C_BASE] + [C["sky"]] * 3 + [C_SEQ, C_CEIL]
        ax = axes[0][0]
        ax.bar(range(len(keep)), vals, color=cols, width=0.64, linewidth=0)
        ax.set_xticks(range(len(keep)))
        ax.set_xticklabels(["Base", "DP", "3Di", "AT", "ALL", "Ceil"], fontsize=6.0)
        ax.set_ylim(min(vals) - 0.06, max(vals) + 0.03)
        ax.set_ylabel("ACC", fontsize=6.5)
    except Exception:
        pass

    # (0,1) E 绝对增益
    if "E" in supp:
        agg = supp["E"]["agg"]
        n = [r["n_train"] for r in agg]
        gap = [r["gap_MCC_mean"] for r in agg]
        ax = axes[0][1]
        ax.plot(n, gap, marker="s", markersize=2.6, linewidth=0.9, color=C_SEQ)
        _log_axis(ax, n, [str(int(v)) for v in n])
        ax.set_ylabel("ΔMCC", fontsize=6.5)

    # (0,2) F 池规模
    if "F" in supp:
        agg = supp["F"]["agg"]
        npool = [r["n_pool"] for r in agg]
        ax = axes[0][2]
        ax.plot(npool, [r["clef_ACC_mean"] for r in agg], marker="o", markersize=2.6,
                linewidth=0.9, color=C_SEQ, label="CLEF")
        ax.plot(npool, [r["baseline_ACC_mean"] for r in agg], marker="^", markersize=2.6,
                linewidth=0.9, color=C_BASE, label="Base")
        ax.plot(npool, [r["ceiling_ACC_mean"] for r in agg], marker="s", markersize=2.6,
                linewidth=0.9, color=C_CEIL, label="Ceil")
        _log_axis(ax, npool, [str(int(v)) for v in npool])
        ax.set_ylabel("ACC", fontsize=6.5)
        ax.legend(loc="lower right", fontsize=5.8)

    # (1,0) G 温度
    if "G" in supp:
        agg = supp["G"]["agg"]
        ax = axes[1][0]
        xs = np.arange(len(agg))
        vals = [r["MCC_mean"] for r in agg]
        ax.bar(xs, vals, color=C_SEQ, width=0.58, linewidth=0)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{r['tau']:g}" for r in agg], fontsize=6.0)
        ax.set_ylabel("MCC", fontsize=6.5)
        ax.set_ylim(min(vals) - 0.06, max(vals) + 0.04)

    # (1,1) H 维度
    if "H" in supp:
        agg = supp["H"]["agg"]
        ax = axes[1][1]
        xs = np.arange(len(agg))
        vals = [r["MCC_mean"] for r in agg]
        ax.bar(xs, vals, color=C_ALT2, width=0.58, linewidth=0)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(int(r["d_z"])) for r in agg], fontsize=6.0)
        ax.set_ylabel("MCC", fontsize=6.5)
        ax.set_ylim(min(vals) - 0.06, max(vals) + 0.04)

    # (1,2) I 标签噪声
    if "I" in supp:
        agg = supp["I"]["agg"]
        x = np.arange(len(agg))
        ax = axes[1][2]
        ax.plot(x, [r["baseline_MCC_mean"] for r in agg], marker="^", markersize=2.6,
                linewidth=0.9, color=C_BASE, label="Base")
        ax.plot(x, [r["clef_MCC_mean"] for r in agg], marker="o", markersize=2.6,
                linewidth=0.9, color=C_SEQ, label="CLEF")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{float(r['rate'])*100:.0f}" for r in agg], fontsize=6.0)
        ax.set_ylabel("MCC", fontsize=6.5)
        ax.legend(loc="lower left", fontsize=5.8)

    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            ax.tick_params(labelsize=6.0)
            panel_label(ax, letters[i][j], dx=-0.30 if j else -0.24, dy=1.14,
                        fontsize=7)
    _save(fig, "fig_overview.png")


def main() -> None:
    supp = _load()
    print("available groups:", list(supp.keys()))
    if "E" in supp:
        plot_powerlaw(supp)
    if "F" in supp:
        plot_scale(supp)
    if any(k in supp for k in ("G", "H", "I")):
        plot_ablation(supp)
    if any(k in supp for k in ("J", "K")):
        plot_signif(supp)
    plot_overview(supp)
    print("done")


if __name__ == "__main__":
    main()
