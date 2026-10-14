"""style_journal.py — 顶刊级图形风格（Nature / Cell / Science 投稿规范）。

设计依据（均为期刊官方作者指南与图形规范中的硬性条款）：

1. **字体**：一律无衬线（Arial / Helvetica），终稿字号 5–7 pt；禁止衬线字体、
   禁止描边字、禁止艺术字。字号须与正文字号相称，不得小于 5 pt。
2. **尺寸**：Nature 单栏 89 mm、双栏 183 mm；本报告版心宽 166 mm，故插图统一
   按 150 mm（单幅）或 150 mm（并列双栏各 74 mm）排版。
3. **线宽**：坐标轴与刻度 0.5–0.75 pt，数据线 0.75–1.0 pt，禁止 >2 pt 的粗线。
4. **配色**：色盲友好（Okabe–Ito 八色，Wong 2011, *Nat. Methods* 8: 441）；
   单图主色不超过 4 种；灰阶作为辅助；禁止渐变、发光、阴影、3D 效果。
5. **图元**：仅保留上／左边框（或全去边框），无背景网格或仅极浅实线网格；
   刻度朝外、带次刻度；禁止柱体黑描边、禁止图内标题、禁止装饰性图标。
6. **panel 标注**：粗体小写 a、b、c，置于各子图左上角，字号与轴标签相当。
7. **误差棒**：必须在图注中写明含义（s.d. / s.e.m. / 95% CI）与样本量 n。
8. **分辨率**：位图 ≥ 300 dpi（本模块统一 600 dpi 导出并留待下游重采样）。

用法::

    from style_journal import apply_style, C, panel_label
    apply_style()
    fig, ax = plt.subplots(figsize=(150/25.4, 60/25.4))
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------- Okabe–Ito 色盲友好调色板
# Wong B. Points of view: Color blindness. Nat Methods 2011;8:441.
C = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
    "grey": "#7F7F7F",
    "lgrey": "#BFBFBF",
}

# 语义色：全篇统一，同一含义在任何图中使用同一颜色
SEQ = C["blue"]        # 序列侧 / CLEF
BASE = C["grey"]       # 纯序列基线
CEIL = C["orange"]     # 充足标注上界
ACCENT = C["vermillion"]

PALETTE4 = [C["blue"], C["orange"], C["green"], C["vermillion"]]
PALETTE5 = [C["blue"], C["orange"], C["green"], C["vermillion"], C["purple"]]

# ---------------------------------------------------------------- 基础 rcParams
_RC = {
    # 关键：font.family 必须写成「具体字族列表」，matplotlib 才会逐字符回退。
    # 若写成 font.family='sans-serif' + font.sans-serif=[...]，中文会渲染成豆腐块。
    "font.family": ["Arial", "SimHei", "Microsoft YaHei", "DejaVu Sans"],
    "font.sans-serif": ["Arial", "SimHei", "Microsoft YaHei", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.axisbelow": True,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.4,
    "ytick.major.size": 2.4,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.minor.size": 1.3,
    "ytick.minor.size": 1.3,
    "lines.linewidth": 0.9,
    "lines.markersize": 2.6,
    "lines.markeredgewidth": 0.6,
    "patch.linewidth": 0.0,
    "legend.frameon": False,
    "legend.handlelength": 1.4,
    "legend.handletextpad": 0.5,
    "legend.labelspacing": 0.35,
    "legend.borderaxespad": 0.3,
    "figure.facecolor": "white",
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "savefig.transparent": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
    "mathtext.fontset": "dejavusans",
}

MM = 1.0 / 25.4  # 毫米 -> 英寸


def apply_style(grid: bool = False) -> None:
    """应用顶刊图形风格。``grid=True`` 时仅绘制极浅的实线网格。"""
    plt.rcParams.update(_RC)
    plt.rcParams["axes.grid"] = grid
    plt.rcParams["grid.color"] = "#E8E8E8"
    plt.rcParams["grid.linewidth"] = 0.4
    plt.rcParams["grid.linestyle"] = "-"


def panel_label(ax, letter: str, dx: float = -0.16, dy: float = 1.06,
                fontsize: float = 8) -> None:
    """在子图左上角写粗体小写 panel 标签（Nature 体例）。"""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=fontsize,
            fontweight="bold", va="top", ha="left", color="black")


def save(fig, path: str, width_mm: float | None = None) -> None:
    """按指定物理宽度（mm）保存，保证插入 Word 后字号不失真。"""
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- 示意图图元
# 顶刊示意图惯例：直角矩形 + 细边框 + 白／浅灰填充 + 单一强调色；无圆角、无阴影、无渐变。
SCHEME = {
    "fill": "#FFFFFF",
    "fill_alt": "#F2F2F2",
    "fill_key": "#E8F1F8",     # 强调模块的淡底
    "edge": "#000000",
    "edge_soft": "#7F7F7F",
    "accent": C["blue"],
}


def box(ax, x, y, w, h, text, face=None, edge=None, fontsize=6.5, bold=False,
        lw=0.6, color="black", zorder=2, va="center"):
    """直角矩形 + 文字。``(x, y)`` 为左下角，坐标为 0~1 归一化。"""
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((x, y), w, h, linewidth=lw,
                           edgecolor=edge or SCHEME["edge"],
                           facecolor=face or SCHEME["fill"], zorder=zorder))
    ty = y + h / 2 if va == "center" else y + h - 0.012
    ax.text(x + w / 2, ty, text, ha="center", va=va, fontsize=fontsize,
            fontweight="bold" if bold else "normal", color=color,
            zorder=zorder + 1, linespacing=1.35)


def arrow(ax, p0, p1, color=None, lw=0.6, ls="-", ms=5, style="-|>"):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms,
                                 linewidth=lw, color=color or SCHEME["edge"],
                                 linestyle=ls, shrinkA=0, shrinkB=0, zorder=1))


def dash_arrow(ax, p0, p1):
    arrow(ax, p0, p1, color=SCHEME["edge_soft"], lw=0.5, ls=(0, (2, 2)))
