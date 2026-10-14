"""本文件职责（示意图层）：
用 matplotlib 绘制报告正文中的两张结构示意图：

    fig2_2_architecture.png   CLEF 双编码器架构图
        - A：训练框架总览（序列通路 / 生物模态通路 / 对称 InfoNCE 对比训练）
        - B：两个编码器的网络架构（编码器 A：Transformer + 池化 + 投影 MLP；编码器 B：浅层 MLP）
        - C：训练流程（无监督对比预训练 + 有监督分类训练）

    fig3_5_three_arm.png      三臂对照协议图

架构图采用 A/B/C 三联面板、圆角容器、直角与梯形模块、张量堆叠与热图矩阵、
旋转轴标签与底部图例；配色以浅蓝紫、青、粉、黄、绿为主色，
辅以靛蓝标题、红描边、灰描边与青色反向虚线。

用法：``python schematic.py``。
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

from style_journal import C, MM, apply_style

# 输出目录：相对当前运行目录（请在仓库根目录下运行）
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

apply_style()

# ------------------------------------------------------------------ 架构图配色
PAL = {
    "peri": "#DEE7F6",      # Transformer 模块 / PLM
    "peri_e": "#8C93AE",
    "cyan": "#A7D8E2",      # MLP
    "cyan_e": "#6FA8B8",
    "pink": "#E0C8E0",      # 层归一化
    "pink_e": "#B08CB0",
    "yellow": "#F8F5BE",    # 可训练 MLP
    "green": "#D2E5C9",     # 编码器 B
    "green_e": "#8FA882",
    "lav": "#D3D8EE",       # 编码器 A
    "lav_e": "#8C93AE",
    "purple": "#B680B2",    # 语言模型表征堆叠
    "ystack": "#F3F3AB",    # 生物模态特征堆叠
    "ostack": "#E0A860",    # 编码器 B 输出堆叠
    "indigo": "#3B3B98",    # 小标题文字
    "red": "#D9534F",       # 可训练模块描边
    "gray": "#B0B0B0",      # 冻结模块描边
    "teal": "#9FD4D4",      # 反向虚线
    "heat": "#DCEFF5",      # 热图底色
    "heat_d": "#4A90C0",    # 热图对角
    "edge": "#3A3A3A",
    "box_e": "#B9C4DC",
}

# 氨基酸字母配色
AA = {"M": "#C05AC0", "A": "#E08A3C", "T": "#E0A030", "P": "#4A6AC0",
      "G": "#3FA03F", "W": "#C05AC0", "I": "#3F8FC0", "R": "#3FA0A0"}

# 顶刊示意图配色（三臂协议图沿用）
FILL = "#FFFFFF"
FILL_ALT = "#F2F2F2"
FILL_KEY = "#E8F1F8"
EDGE = "#000000"
EDGE_SOFT = "#7F7F7F"
ACCENT = C["blue"]


# ======================================================================================
# 绘图图元
# ======================================================================================
def _rbox(ax, x, y, w, h, fc=FILL, ec=None, lw=0.7, r=0.012, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                mutation_scale=1, linewidth=lw,
                                edgecolor=ec or PAL["box_e"], facecolor=fc, zorder=z))


def _rtext(ax, x, y, w, h, s, fc=FILL, ec=None, lw=0.7, r=0.012, fs=7,
           color="black", bold=False, rot=0, z=3, ls=1.3):
    _rbox(ax, x, y, w, h, fc, ec, lw, r, z)
    ax.text(x + w / 2, y + h / 2, s, ha="center", va="center", fontsize=fs,
            color=color, fontweight="bold" if bold else "normal",
            rotation=rot, zorder=z + 1, linespacing=ls)


def _rect(ax, x, y, w, h, fc, ec, lw=0.7, z=2):
    ax.add_patch(Rectangle((x, y), w, h, linewidth=lw, edgecolor=ec,
                           facecolor=fc, zorder=z))


def _trap_v(ax, cx, cy, w_top, w_bot, h, s, fc, ec, fs=7, lw=0.8):
    """上宽下窄的梯形（纵向编码器）。"""
    ax.add_patch(Polygon([(cx - w_top / 2, cy + h / 2), (cx + w_top / 2, cy + h / 2),
                          (cx + w_bot / 2, cy - h / 2), (cx - w_bot / 2, cy - h / 2)],
                         closed=True, linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(cx, cy, s, ha="center", va="center", fontsize=fs, zorder=3)


def _trap_h(ax, x0, x1, cy, h_left, h_right, s, fc, ec, fs=7, lw=0.8):
    """左宽右窄的梯形（横向编码器）。"""
    ax.add_patch(Polygon([(x0, cy + h_left / 2), (x1, cy + h_right / 2),
                          (x1, cy - h_right / 2), (x0, cy - h_left / 2)],
                         closed=True, linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text((x0 + x1) / 2, cy, s, ha="center", va="center", fontsize=fs, zorder=3)


def _stack(ax, x, y, w, h, n, dx, dy, fc, ec, r=0.010, lw=0.6):
    """张量堆叠：n 层带偏移的圆角矩形（由后向前）。"""
    for i in range(n - 1, -1, -1):
        _rbox(ax, x + i * dx, y + i * dy, w, h, fc=fc, ec=ec, lw=lw, r=r, z=2 + (n - i))


def _heatmap(ax, x, y, w, h, n=8, off=0):
    """对比训练热图矩阵：浅蓝底 + 蓝色对角线。"""
    cw, ch = w / n, h / n
    for i in range(n):
        for j in range(n):
            k = (j - i) % n
            fc = PAL["heat_d"] if k == off else PAL["heat"]
            _rect(ax, x + j * cw, y + (n - 1 - i) * ch, cw, ch, fc, "#FFFFFF", 0.5, 1)


def _arrow(ax, p0, p1, color=None, lw=0.8, ls="-", style="-|>", ms=6):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms,
                                 linewidth=lw, color=color or PAL["edge"],
                                 linestyle=ls, shrinkA=0, shrinkB=0, zorder=4))


def _path(ax, pts, color, lw=0.8, ls=(0, (3, 2))):
    """折线路径，末段带箭头。"""
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=lw,
                linestyle=ls, zorder=1, solid_capstyle="butt")
    _arrow(ax, pts[-2], pts[-1], color=color, lw=lw, ls=ls, ms=5)


def _aa_row(ax, x, y, seq, fs=7.4, step=0.0190):
    for i, ch in enumerate(seq):
        ax.text(x + i * step, y, ch, ha="center", va="center", fontsize=fs,
                color=AA.get(ch, "black"), fontweight="bold", zorder=3)
    ax.text(x + len(seq) * step + 0.010, y, "· · · ·", ha="left", va="center",
            fontsize=fs - 1.0, color="#4FC8C8", zorder=3)


# ======================================================================================
# 图 1-1　CLEF 双编码器架构
# ======================================================================================
def draw_architecture() -> None:
    with plt.rc_context({
        "font.family": ["Arial", "Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "savefig.pad_inches": 0.015,
    }):
        fig, ax = plt.subplots(figsize=(15.0 * 10 * MM, 15.6 * 10 * MM))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # ============================================================ A　训练框架总览
        ax.text(0.008, 0.997, "A", fontsize=9.5, va="top", ha="left", zorder=5)

        # 序列通路输入
        ax.text(0.112, 0.972, "蛋白质序列", ha="center", va="center", fontsize=8.2,
                fontweight="bold", color=PAL["indigo"])
        ax.text(0.016, 0.925, "Protein 1 ：", ha="left", va="center", fontsize=6.4)
        _aa_row(ax, 0.118, 0.925, "MATPG", fs=7.0, step=0.0152)
        ax.text(0.016, 0.888, "Protein 2 ：", ha="left", va="center", fontsize=6.4)
        _aa_row(ax, 0.118, 0.888, "MWIRP", fs=7.0, step=0.0152)
        _arrow(ax, (0.214, 0.906), (0.258, 0.906))

        # ESM-2
        _rbox(ax, 0.262, 0.848, 0.150, 0.140, fc="#EEF2FA", ec="#DCE4F2", lw=1.0, r=0.022)
        _rtext(ax, 0.269, 0.856, 0.136, 0.124,
               "Evolutionary Scale\nModeling - 2\n(ESM-2)", fc=PAL["peri"],
               ec=PAL["peri_e"], lw=0.9, r=0.018, fs=6.6, ls=1.25)
        ax.text(0.337, 0.838, "预训练蛋白质语言模型", ha="center", va="center", fontsize=6.2)

        # 语言模型表征堆叠
        ax.text(0.518, 0.960, "语言模型表征", ha="center", va="center", fontsize=6.8)
        _stack(ax, 0.470, 0.868, 0.100, 0.052, 4, 0.0065, 0.0065,
               PAL["purple"], "#8F6389")
        _arrow(ax, (0.414, 0.900), (0.468, 0.900))

        # 编码器 A
        _arrow(ax, (0.518, 0.866), (0.518, 0.835))
        _trap_v(ax, 0.518, 0.798, 0.112, 0.064, 0.070, "编码器 A",
                PAL["lav"], PAL["lav_e"], fs=6.8)

        # 跨模态表征堆叠
        _arrow(ax, (0.518, 0.761), (0.518, 0.735))
        _stack(ax, 0.470, 0.690, 0.100, 0.038, 3, 0.0065, 0.0065,
               "#C9CFEA", "#8C93AE")
        ax.text(0.596, 0.712, "跨模态表征 z（128 维）", ha="left", va="center", fontsize=6.4)

        # 对比训练热图
        _arrow(ax, (0.518, 0.688), (0.518, 0.674))
        _heatmap(ax, 0.476, 0.582, 0.086, 0.086, 8)
        ax.text(0.580, 0.628, "对称 InfoNCE\n对比训练", ha="left", va="center",
                fontsize=6.4, linespacing=1.35)

        # 生物模态通路
        ax.text(0.112, 0.760, "生物模态特征", ha="center", va="center", fontsize=8.2,
                fontweight="bold", color=PAL["indigo"])
        ax.text(0.112, 0.700, "功能注释、蛋白质结构、\n实验数据 …", ha="center", va="center",
                fontsize=6.4, linespacing=1.35)
        _stack(ax, 0.188, 0.582, 0.050, 0.100, 3, 0.0065, 0.0065,
               PAL["ystack"], "#BDB36A")
        ax.text(0.214, 0.558, "DP / MSA / 3Di / AT / SE", ha="center", va="center",
                fontsize=6.0)
        _arrow(ax, (0.256, 0.632), (0.300, 0.632))
        _trap_h(ax, 0.300, 0.382, 0.632, 0.100, 0.050, "编码器 B",
                PAL["green"], PAL["green_e"], fs=6.4)
        _arrow(ax, (0.382, 0.632), (0.394, 0.632))
        _stack(ax, 0.394, 0.582, 0.050, 0.100, 3, 0.0065, 0.0065,
               PAL["ostack"], "#B07A34")

        # ============================================================ B　编码器网络架构
        ax.text(0.008, 0.542, "B", fontsize=9.5, va="top", ha="left", zorder=5)
        _rbox(ax, 0.048, 0.330, 0.472, 0.176, fc="#FDFDFD", ec="#D2D2D2", lw=0.9, r=0.020)
        _rbox(ax, 0.552, 0.330, 0.328, 0.176, fc="#FDFDFD", ec="#D2D2D2", lw=0.9, r=0.020)

        # 编码器 A 内部
        ax.text(0.080, 0.418, "语言模型\n表征", ha="center", va="center", fontsize=6.4,
                rotation=90, linespacing=1.25)
        _arrow(ax, (0.100, 0.418), (0.122, 0.418))
        _rtext(ax, 0.122, 0.360, 0.166, 0.116, "Transformer 模块",
               fc=PAL["peri"], ec=PAL["peri_e"], lw=0.8, r=0.008, fs=7.0)
        ax.text(0.205, 0.346, "n = 2,  d = 1 280", ha="center", va="center", fontsize=6.1)
        _arrow(ax, (0.288, 0.418), (0.310, 0.418))
        ax.text(0.326, 0.418, "跨模态\n表征", ha="center", va="center", fontsize=6.1,
                rotation=90, color="#C05A5A", linespacing=1.25)
        _arrow(ax, (0.344, 0.418), (0.366, 0.418))
        _rtext(ax, 0.366, 0.360, 0.042, 0.116, "MLP",
               fc=PAL["cyan"], ec=PAL["cyan_e"], lw=0.8, r=0.006, fs=7.0, rot=90)
        _arrow(ax, (0.408, 0.418), (0.430, 0.418))
        ax.text(0.446, 0.418, "嵌入", ha="center", va="center", fontsize=6.1, rotation=90)
        ax.text(0.387, 0.346, "d = 128", ha="center", va="center", fontsize=6.1)
        ax.text(0.284, 0.316, "编码器 A", ha="center", va="center", fontsize=7.6,
                fontweight="bold")

        # 编码器 B 内部
        ax.text(0.584, 0.418, "生物模态\n特征", ha="center", va="center", fontsize=6.4,
                rotation=90, linespacing=1.25)
        _arrow(ax, (0.604, 0.418), (0.622, 0.418))
        _rtext(ax, 0.622, 0.360, 0.042, 0.116, "MLP",
               fc=PAL["cyan"], ec=PAL["cyan_e"], lw=0.8, r=0.006, fs=7.0, rot=90)
        _arrow(ax, (0.664, 0.418), (0.680, 0.418))
        _rtext(ax, 0.680, 0.360, 0.046, 0.116, "层归一化",
               fc=PAL["pink"], ec=PAL["pink_e"], lw=0.8, r=0.006, fs=6.2, rot=90)
        _arrow(ax, (0.726, 0.418), (0.742, 0.418))
        _rtext(ax, 0.742, 0.360, 0.042, 0.116, "MLP",
               fc=PAL["cyan"], ec=PAL["cyan_e"], lw=0.8, r=0.006, fs=7.0, rot=90)
        _arrow(ax, (0.784, 0.418), (0.806, 0.418))
        ax.text(0.822, 0.418, "嵌入", ha="center", va="center", fontsize=6.1, rotation=90)
        ax.text(0.704, 0.346, "d = 128", ha="center", va="center", fontsize=6.1)
        ax.text(0.716, 0.316, "编码器 B", ha="center", va="center", fontsize=7.6,
                fontweight="bold")

        ax.text(0.892, 0.418, "n：编码器层数\nd：隐藏层维数", ha="left", va="center",
                fontsize=6.2, linespacing=1.45)

        # ============================================================ C　训练流程
        ax.text(0.008, 0.288, "C", fontsize=9.5, va="top", ha="left", zorder=5)
        ax.text(0.268, 0.262, "无监督对比预训练", ha="center", va="center", fontsize=8.0,
                fontweight="bold", color=PAL["indigo"])
        ax.text(0.748, 0.262, "有监督分类训练", ha="center", va="center", fontsize=8.0,
                fontweight="bold", color=PAL["indigo"])

        # 左：对比预训练
        ax.text(0.030, 0.158, "蛋白质序列", ha="center", va="center", fontsize=6.4,
                rotation=90)
        _arrow(ax, (0.048, 0.158), (0.070, 0.158))
        _rtext(ax, 0.070, 0.083, 0.106, 0.150, "PLM",
               fc=PAL["peri"], ec=PAL["gray"], lw=0.9, r=0.006, fs=7.2)
        _arrow(ax, (0.176, 0.158), (0.196, 0.158))
        _rtext(ax, 0.196, 0.083, 0.044, 0.150, "编码器 A",
               fc=PAL["lav"], ec=PAL["red"], lw=1.0, r=0.006, fs=6.4, rot=90)
        _arrow(ax, (0.240, 0.158), (0.268, 0.158))
        ax.text(0.322, 0.158, "InfoNCE 损失", ha="center", va="center", fontsize=7.0)
        _arrow(ax, (0.424, 0.158), (0.376, 0.158))
        _rtext(ax, 0.424, 0.083, 0.044, 0.150, "编码器 B",
               fc=PAL["green"], ec=PAL["red"], lw=1.0, r=0.006, fs=6.4, rot=90)
        ax.text(0.488, 0.158, "生物模态特征", ha="center", va="center", fontsize=6.4,
                rotation=90)
        _path(ax, [(0.218, 0.233), (0.218, 0.250), (0.446, 0.250), (0.446, 0.233)],
              PAL["teal"])
        _path(ax, [(0.446, 0.083), (0.446, 0.056), (0.218, 0.056), (0.218, 0.083)],
              PAL["teal"])

        # 右：有监督分类
        ax.text(0.548, 0.158, "蛋白质序列", ha="center", va="center", fontsize=6.4,
                rotation=90)
        _arrow(ax, (0.566, 0.158), (0.588, 0.158))
        _rtext(ax, 0.588, 0.083, 0.106, 0.150, "PLM",
               fc=PAL["peri"], ec=PAL["gray"], lw=0.9, r=0.006, fs=7.2)
        _arrow(ax, (0.694, 0.158), (0.712, 0.158))
        _rtext(ax, 0.712, 0.083, 0.044, 0.150, "编码器 A",
               fc=PAL["lav"], ec=PAL["gray"], lw=0.9, r=0.006, fs=6.4, rot=90)
        _arrow(ax, (0.756, 0.158), (0.774, 0.158))
        _rtext(ax, 0.774, 0.083, 0.044, 0.150, "MLP",
               fc=PAL["yellow"], ec=PAL["red"], lw=1.0, r=0.006, fs=6.6, rot=90)
        _arrow(ax, (0.818, 0.158), (0.842, 0.158))
        ax.text(0.884, 0.158, "BCE 损失", ha="center", va="center", fontsize=7.0)
        _path(ax, [(0.796, 0.233), (0.796, 0.250), (0.734, 0.250), (0.734, 0.233)],
              PAL["teal"])

        # 图例
        _rect(ax, 0.252, 0.010, 0.028, 0.017, FILL, PAL["red"], 1.0, 3)
        ax.text(0.286, 0.0185, "可训练模块", ha="left", va="center", fontsize=6.2)
        _rect(ax, 0.408, 0.010, 0.028, 0.017, FILL, PAL["gray"], 0.9, 3)
        ax.text(0.442, 0.0185, "冻结模块", ha="left", va="center", fontsize=6.2)
        ax.plot([0.570, 0.612], [0.0185, 0.0185], color=PAL["edge"], lw=0.8, zorder=3)
        ax.text(0.620, 0.0185, "前向", ha="left", va="center", fontsize=6.2)
        ax.plot([0.700, 0.742], [0.0185, 0.0185], color=PAL["teal"], lw=0.9,
                linestyle=(0, (3, 2)), zorder=3)
        ax.text(0.750, 0.0185, "反向", ha="left", va="center", fontsize=6.2)

        _save(fig, "fig2_2_architecture.png")


# ======================================================================================
# 三臂对照协议图（沿用顶刊风格）
# ======================================================================================
def _box(ax, x, y, w, h, text, face=FILL, edge=EDGE, fontsize=6.5,
         bold=False, lw=0.6, color="black"):
    """直角矩形 + 居中文字（坐标为 0~1 归一化，x,y 为左下角）。"""
    ax.add_patch(Rectangle((x, y), w, h, linewidth=lw, edgecolor=edge,
                           facecolor=face, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold" if bold else "normal",
            color=color, zorder=3, linespacing=1.35)


def _arrow2(ax, p0, p1, color=EDGE, lw=0.6, ls="-", style="-|>", ms=5):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms,
                                 linewidth=lw, color=color, linestyle=ls,
                                 shrinkA=0, shrinkB=0, zorder=1))


def _dash_arrow(ax, p0, p1):
    _arrow2(ax, p0, p1, color=EDGE_SOFT, lw=0.5, ls=(0, (2, 2)))


def draw_three_arm() -> None:
    fig, ax = plt.subplots(figsize=(150 * MM, 88 * MM))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---- 顶部：共享世界 ----
    _box(ax, 0.030, 0.800, 0.940, 0.170,
         "SyntheticWorld：共享潜变量世界（e, u_k, A, P_m 只构造一次）\n"
         "z ~ N(0, I)      x_seq = A z + σ_seq ε      x_m = P_m z + η_m ε",
         face=FILL_ALT, edge=EDGE, fontsize=6.2)
    ax.text(0.5, 0.775, "各划分自同一世界独立采样（互斥且同分布）",
            ha="center", va="center", fontsize=6.0, style="italic")

    for x, w, t in [(0.030, 0.215, "无标注池\n4 000"),
                    (0.285, 0.215, "标注训练集\n100（上界臂 1 500）"),
                    (0.540, 0.190, "验证集\n150"),
                    (0.780, 0.190, "测试集\n800")]:
        _box(ax, x, 0.640, w, 0.090, t, face=FILL)
        _dash_arrow(ax, (x + w / 2, 0.640), (x + w / 2, 0.575))

    # ---- 中部：三臂 ----
    arms = [
        (0.030, 0.290, "Baseline（下界）\n冻结序列表征（平均池化）\n+ 100 条标注", FILL),
        (0.355, 0.290, "CLEF（处理组）\n跨模态对比预训练\n（编码器 A ← 编码器 B 对齐）\n+ 100 条标注",
         FILL_KEY),
        (0.680, 0.290, "Ceiling（上界）\n冻结序列表征（平均池化）\n+ 1 500 条标注", FILL),
    ]
    for x, w, t, fc in arms:
        _box(ax, x, 0.370, w, 0.190, t, face=fc,
             edge=ACCENT if fc == FILL_KEY else EDGE,
             lw=0.9 if fc == FILL_KEY else 0.6, fontsize=6.2)
        _dash_arrow(ax, (x + w / 2, 0.370), (x + w / 2, 0.305))

    # ---- 底部：统一评测 ----
    _box(ax, 0.030, 0.185, 0.940, 0.100,
         "在同一测试集上统一评测（分类器与训练协议完全一致）\n"
         "ACC · REC · PR · F1 · MCC · AUPRC      ARI · NMI · ASW",
         face=FILL, edge=EDGE, fontsize=6.2)

    _box(ax, 0.130, 0.045, 0.740, 0.095,
         "标注替代率   Δ_ASR = (CLEF − Baseline) / (Ceiling − Baseline)",
         face=FILL_ALT, edge=EDGE, fontsize=6.6)
    _arrow2(ax, (0.500, 0.185), (0.500, 0.140), color=ACCENT, lw=0.8)
    _save(fig, "fig3_5_three_arm.png")


def _save(fig, name: str) -> None:
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  [schematic] 已保存 {path}")


def main() -> None:
    draw_architecture()
    draw_three_arm()
    print("[schematic] 全部示意图已生成")


if __name__ == "__main__":
    main()
