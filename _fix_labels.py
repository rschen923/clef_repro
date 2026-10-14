# -*- coding: utf-8 -*-
"""一次性脚本：把绘图脚本中的标签统一为「中文说明 + Unicode 数学符号」。

原因：matplotlib 在「中文 + $...$ 数学模式」混排时，非数学片段会走
mathtext 的 default 字体（DejaVu Sans），中文渲染成豆腐块（□），
且该缺陷依赖调用顺序、不可靠。统一改用 Unicode 希腊字母与下标写法可彻底规避。
"""
import io

# 通用替换（对两个文件都适用）
COMMON = [
    ("$\\Delta_{ASR}$", "Δ_ASR"),
    ("$\\Delta$", "Δ"),
    ("$\\Delta$MCC", "ΔMCC"),
    ("$\\tau$", "τ"),
    ("$\\beta$", "β"),
    ("$\\times$", "×"),
    ("$n_{tr}$", "n_tr"),
    ("$n_{pool}$", "n_pool"),
    ("$d_z$", "d_z"),
    ("$n$", "n"),
    ("$R^2$", "R²"),
]

PLOTS = [
    ('ax.set_ylabel("Score")', 'ax.set_ylabel("得分")'),
    ('ax.set_xlabel("Epoch")', 'ax.set_xlabel("预训练轮次")'),
    ('ax.set_ylabel("InfoNCE loss")', 'ax.set_ylabel("InfoNCE 损失")'),
    ('label="CLEF-ALL InfoNCE"', 'label="CLEF-ALL 对比损失"'),
    ('label=f"Random baseline  ln(B$-$1) = {random_base:.2f}"',
     'label=f"随机基线 ln(B−1) = {random_base:.2f}"'),
    ('ax.set_xlabel("Labeled samples  $n_{tr}$")', 'ax.set_xlabel("标注量 n_tr")'),
    ('ax.set_ylabel("Downstream ACC")', 'ax.set_ylabel("下游 ACC")'),
    ('ax.set_ylabel("$\\Delta_{ASR}$")', 'ax.set_ylabel("标注替代率 Δ_ASR")'),
    ('label=f"$\\Delta_{{ASR}}$ ({label})"', 'label=f"Δ_ASR（{label}）"'),
    ('"Baseline (frozen-seq)"', '"Baseline（冻结序列）"'),
    ('"Ceiling (abundant)"', '"Ceiling（充足标注）"'),
    ('"ceiling = 1.0"', '"上界 = 1.0"'),
    ('ax.set_xlabel("VIB strength  $\\beta$")', 'ax.set_xlabel("VIB 强度 β")'),
    ('ax.set_ylabel(f"Downstream {key}")', 'ax.set_ylabel(f"下游 {key}")'),
    ('ax.set_xlabel("Modality noise scale  $\\times$")',
     'ax.set_xlabel("模态噪声放大倍数 ×")'),
    ('ax.set_xticklabels([f"{b:g}" for b in betas])',
     'ax.set_xticklabels([f"{b:g}" for b in betas])'),
]

SUPP = [
    ('label="$\\Delta_{ASR}$ (MCC)"', 'label="Δ_ASR (MCC)"'),
    ('label="绝对增益  $\\Delta$ (MCC)"', 'label="绝对增益 Δ (MCC)"'),
    ('label=f"幂律拟合  $R^2$ = {fit[\'r2\']:.3f}"',
     'label=f"幂律拟合  R² = {fit[\'r2\']:.3f}"'),
    ('ax.text(n.max(), 1.0, " $\\\\Delta_{ASR}$ = 1 "',
     'ax.text(n.max(), 1.0, " Δ_ASR = 1 "'),
    ('ax.set_xlabel("标注量  $n$")', 'ax.set_xlabel("标注量 n")'),
    ('ax.set_ylabel("标注替代率  $\\\\Delta_{ASR}$ (MCC)")',
     'ax.set_ylabel("标注替代率 Δ_ASR (MCC)")'),
    ('ax.set_ylabel("绝对增益  MCC$_{CLEF}$ − MCC$_{Base}$")',
     'ax.set_ylabel("绝对增益  MCC(CLEF) − MCC(Base)")'),
    ('ax.set_xlabel("无标注池规模  $n_{pool}$")', 'ax.set_xlabel("无标注池规模 n_pool")'),
    ('ax.set_ylabel("标注替代率  $\\\\Delta_{ASR}$ (MCC)")',
     'ax.set_ylabel("标注替代率 Δ_ASR (MCC)")'),
    ('label="$\\\\Delta_{ASR}$ (MCC)"', 'label="Δ_ASR (MCC)"'),
    ('ax.set_xlabel("InfoNCE 温度  $\\\\tau$")', 'ax.set_xlabel("InfoNCE 温度 τ")'),
    ('ax.set_xlabel("跨模态表征维度  $d_z$")', 'ax.set_xlabel("跨模态表征维度 d_z")'),
    ('ax.set_xlabel("训练标签翻转率  (%)")', 'ax.set_xlabel("训练标签翻转率 (%)")'),
    ('ax.set_xlabel("配对差值 $\\\\Delta$MCC 的 bootstrap 均值")',
     'ax.set_xlabel("配对差值 ΔMCC 的自助法均值")'),
    ('f"95% CI = [{lo:.4f}, {hi:.4f}]\\n$p$ = {p:.2e}（配对 $t$ 检验）"',
     'f"95% CI = [{lo:.4f}, {hi:.4f}]\\np = {p:.2e}（配对 t 检验）"'),
    ('ax.set_ylabel("$\\\\Delta$MCC", fontsize=6.5)', 'ax.set_ylabel("ΔMCC", fontsize=6.5)'),
]

SCHEM = [
    ('"$z \\\\sim \\\\mathcal{N}(0, I)$      $x_{seq} = A z + \\\\sigma_{seq}\\\\epsilon$"\n'
     '         "      $x_m = P_m z + \\\\eta_m \\\\epsilon$"',
     '"z ~ N(0, I)      x_seq = A z + σ_seq ε      x_m = P_m z + η_m ε"'),
    ('"序列表征 $z$ (128 维)"', '"序列表征 z（128 维）"'),
    ('"模态表征 $z\'$ (128 维)"', '"模态表征 z′（128 维）"'),
    ('"对称 InfoNCE\\n$\\\\mathcal{L}=\\\\frac{1}{2}(\\\\mathcal{L}_{A\\\\to B}+\\\\mathcal{L}_{B\\\\to A})$\\n"',
     '"对称 InfoNCE\\nL = (L_A→B + L_B→A) / 2\\n"'),
    ('"标注替代率   $\\\\Delta_{ASR} = \\\\dfrac{\\\\mathrm{CLEF} - \\\\mathrm{Baseline}}"\n'
     '         "{\\\\mathrm{Ceiling} - \\\\mathrm{Baseline}}$"',
     '"标注替代率   Δ_ASR = (CLEF − Baseline) / (Ceiling − Baseline)"'),
    ('"SyntheticWorld：共享潜变量世界（$e,\\\\ u_k,\\\\ A,\\\\ P_m$ 只构造一次）\\n"',
     '"SyntheticWorld：共享潜变量世界（e, u_k, A, P_m 只构造一次）\\n"'),
]


def apply(path, pairs):
    s = io.open(path, encoding="utf-8").read()
    n = 0
    for a, b in pairs:
        if a in s:
            s = s.replace(a, b)
            n += 1
        else:
            print(f"  [未命中] {path}: {a[:60]!r}")
    io.open(path, "w", encoding="utf-8").write(s)
    print(f"{path}: 替换 {n}/{len(pairs)}")


apply("plots.py", COMMON + PLOTS)
apply("plots_supplementary.py", COMMON + SUPP)
apply("schematic.py", COMMON + SCHEM)
