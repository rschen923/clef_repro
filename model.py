"""本文件职责（编码器层）：
定义 CLEF 跨模态对比学习框架的全部网络模块：
    - ``EncoderA``            序列通路编码器（TransformerEncoder + 均值池化 + 投影 MLP -> 128 维）
    - ``EncoderB``            模态通路编码器（浅层 MLP -> 同一 128 维潜空间）
    - ``symmetric_infonce``   对称 InfoNCE 损失
    - ``DownstreamClassifier``下游二分类头（2 层 MLP + LayerNorm + sigmoid）
    - ``VIBEncoder``          变分信息瓶颈正则（可选），暴露终态 KL 项
并提供一个 ``count_trainable_params`` 工具用于核对参数量量级。

设计动机（对应报告第 2 章“CLEF 双编码器架构”）：
- 序列侧（EncoderA）与模态侧（EncoderB）在预训练阶段被对称 InfoNCE 拉到同一 128 维空间；
  预训练结束后**只保留序列侧编码器 + 下游分类头**，推理仅需一条蛋白质序列。
- 因此“冻结骨干”（真实场景中的 ESM-2）不引入任何可训练参数；
  可训练参数只有 EncoderA / EncoderB / 下游分类头。
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================================
# 冻结序列基线（Baseline / Ceiling 臂使用）
# ======================================================================================
class FrozenSeqBaseline(nn.Module):
    """冻结的序列基线：均值池化 + 固定随机线性投影到 128 维。

    模拟“直接拿冻结骨干（ESM-2）的均值池化序列表征做下游”，
    投影矩阵随机初始化后**冻结**，因此不引入任何可训练参数，
    与报告“Baseline 只训练下游分类头”的设定一致。
    """

    def __init__(self, tok_dim: int, out_dim: int = 128, seed: int = 0) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        w = torch.randn(tok_dim, out_dim, generator=g) / math.sqrt(tok_dim)
        self.register_buffer("W", w)  # 冻结，不参与优化

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, L, tok_dim)
        return x.mean(dim=1) @ self.W


# ======================================================================================
# 序列通路：Encoder A
# ======================================================================================
class MeanPool(nn.Module):
    """沿序列维（dim=1）做平均池化，压掉序列维，得到固定长度向量。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, L, D)
        return x.mean(dim=1)


class EncoderA(nn.Module):
    """序列通路编码器：x_seq -> 128 维跨模态表征 z。

    结构：线性输入投影 -> 位置编码 -> 2 层 TransformerEncoder（4 头）-> MeanPool -> 投影 MLP。

    参数
    ----
    tok_dim   : 每个序列位置的 token 维度
    seq_len   : 序列长度
    d_model   : Transformer 隐层维度
    n_heads   : 注意力头数（报告固定 4）
    n_layers  : TransformerEncoder 层数（报告固定 2）
    out_dim   : 输出表征维度（报告固定 128）
    """

    def __init__(
        self,
        tok_dim: int,
        seq_len: int,
        d_model: int = 48,
        n_heads: int = 4,
        n_layers: int = 2,
        out_dim: int = 128,
    ) -> None:
        super().__init__()
        self.inp = nn.Linear(tok_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.pool = MeanPool()
        self.proj = nn.Sequential(
            nn.Linear(d_model, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, L, tok_dim)
        h = self.inp(x) + self.pos
        h = self.encoder(h)
        h = self.pool(h)
        return self.proj(h)


# ======================================================================================
# 模态通路：Encoder B
# ======================================================================================
class EncoderB(nn.Module):
    """模态通路编码器：拼接后的模态特征 -> 128 维潜空间 z'。

    浅层 MLP（一层隐藏层），把 ``in_dim`` 维拼接模态特征映射到与 EncoderA 相同的 128 维空间。

    参数
    ----
    in_dim   : 输入模态拼接维度（单模态时为该模态维度；ALL 时为五模态维度之和）
    hidden   : 隐藏层维度
    out_dim  : 输出维度（与 EncoderA 对齐，128）
    """

    def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, in_dim)
        return self.net(x)


# ======================================================================================
# 对称 InfoNCE
# ======================================================================================
def symmetric_infonce(
    z: torch.Tensor, z_prime: torch.Tensor, tau: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """对称 InfoNCE 损失。

    同一蛋白的两个视角（序列视角 z 与模态视角 z'）互为正样本，
    批内其余样本为负样本。损失为 A->B 与 B->A 两个方向的交叉熵平均。

    参数
    ----
    z       : (B, D) 序列侧表征
    z_prime : (B, D) 模态侧表征
    tau     : 温度系数（报告默认 0.5）

    返回
    ----
    (loss, loss_ab, loss_ba) —— 对称损失及其两个方向分量。
    """
    z = F.normalize(z, dim=1)
    z_prime = F.normalize(z_prime, dim=1)
    logits = z @ z_prime.t() / tau  # (B, B)
    labels = torch.arange(z.size(0), device=z.device)
    loss_ab = F.cross_entropy(logits, labels)
    loss_ba = F.cross_entropy(logits.t(), labels)
    loss = 0.5 * (loss_ab + loss_ba)
    return loss, loss_ab, loss_ba


# ======================================================================================
# 下游分类头
# ======================================================================================
class DownstreamClassifier(nn.Module):
    """下游二分类头：2 层 MLP + LayerNorm + sigmoid。

    输入 128 维冻结表征，输出二分类概率（logit 形式，配 ``BCEWithLogitsLoss``）。
    全实验所有实验臂共用同一结构、同一参数量，保证“除表征来源/标注预算外其余完全一致”。

    参数
    ----
    in_dim : 输入表征维度（128）
    hidden : 隐藏层维度（默认 80，使参数量约 1.05 万，与报告一致）
    """

    def __init__(self, in_dim: int = 128, hidden: int = 80) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, in_dim)
        return self.net(x).squeeze(-1)  # (B,) logits


# ======================================================================================
# 变分信息瓶颈（可选正则）
# ======================================================================================
class VIBEncoder(nn.Module):
    """变分信息瓶颈：把 128 维表征参数化为均值 μ / 对数方差 logσ²，重参数化采样。

    总损失 = InfoNCE + β·KL(q(z|x) ‖ N(0,I))。
    β=0 时退化为普通对比预训练；β 越大对表征的信息压缩越强。
    暴露 ``last_kl`` 供报告记录“终态 KL 项”。

    参数
    ----
    in_dim     : 输入表征维度（128）
    latent_dim : 潜变量维度（默认 128，与表征同维）
    """

    def __init__(self, in_dim: int = 128, latent_dim: int = 128) -> None:
        super().__init__()
        self.mu_layer = nn.Linear(in_dim, latent_dim)
        self.logvar_layer = nn.Linear(in_dim, latent_dim)
        self.last_kl: Optional[torch.Tensor] = None

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 ``(z_sample, mu, logvar)``。"""
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h).clamp(-8.0, 8.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q(z|x) ‖ N(0,I))，按 batch 求平均（单位：nats）。"""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()


# ======================================================================================
# 参数量核对
# ======================================================================================
def count_trainable_params(module: nn.Module) -> int:
    """统计可训练参数量（冻结骨干不引入任何可训练参数）。"""
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def build_clef(
    tok_dim: int,
    seq_len: int,
    modality_in_dim: int,
    *,
    d_model: int = 48,
    n_heads: int = 4,
    n_layers: int = 2,
    out_dim: int = 128,
    encoder_b_hidden: int = 64,
    use_vib: bool = False,
) -> Dict[str, nn.Module]:
    """便捷工厂：构建一套 CLEF 编码器。

    返回 dict：``{"encoder_a":..., "encoder_b":..., "vib":... or None}``。
    """
    encoder_a = EncoderA(tok_dim, seq_len, d_model, n_heads, n_layers, out_dim)
    encoder_b = EncoderB(modality_in_dim, hidden=encoder_b_hidden, out_dim=out_dim)
    vib = VIBEncoder(out_dim, out_dim) if use_vib else None
    return {"encoder_a": encoder_a, "encoder_b": encoder_b, "vib": vib}
