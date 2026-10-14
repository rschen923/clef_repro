"""本文件职责（训练层）：
    - ``set_seed``            统一随机性（numpy / torch），保证可复现
    - ``pretrain_contrastive`` 对称 InfoNCE 对比预训练循环（Adam，默认 50 epoch，batch 256）
    - ``train_downstream``     冻结表征 + 下游 BCE 训练（含 z-score 标准化 + 验证集 F1 早停）

设计动机（对应报告 3.3 节“训练协议”）：
- 预训练与下游两阶段解耦：预训练只做跨模态对齐，下游只训练一个小分类头，
  从而把“标注稀缺下的效率提升”归因到“表征质量”而非“分类器容量”。
- 下游表征先做 z-score 标准化（用训练集统计量），保证不同表征来源的尺度一致；
  以验证集 F1 作为早停准则（F1 对类别不平衡更稳健）。
"""

from __future__ import annotations

import copy
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from model import (
    DownstreamClassifier,
    EncoderA,
    EncoderB,
    VIBEncoder,
    kl_divergence,
    symmetric_infonce,
)


# ======================================================================================
# 随机性控制
# ======================================================================================
def set_seed(seed: int) -> None:
    """统一设置 numpy / torch / random 的随机种子，保证跨运行可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ======================================================================================
# 对比预训练
# ======================================================================================
def pretrain_contrastive(
    x_seq: np.ndarray,
    x_mod: np.ndarray,
    encoder_a: EncoderA,
    encoder_b: EncoderB,
    *,
    vib: Optional[VIBEncoder] = None,
    beta: float = 0.0,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    tau: float = 0.5,
    seed: int = 42,
    device: str = "cpu",
    verbose: bool = False,
) -> Dict[str, object]:
    """对称 InfoNCE 对比预训练。

    参数
    ----
    x_seq   : (N, L, tok_dim) 序列视角（无标注池）
    x_mod   : (N, D) 模态视角（无标注池）
    encoder_a / encoder_b : 两个编码器
    vib     : 可选变分信息瓶颈，作用于序列侧表征
    beta    : VIB 的 KL 权重（β=0 即普通对比预训练）
    epochs  : 预训练轮数（实验 A 用 50，B/C/D 用 30）
    batch_size : 批大小（默认 256）
    tau     : InfoNCE 温度（默认 0.5）
    seed    : 随机种子

    返回
    ----
    dict：
        ``curve``      : List[float] 每个 epoch 的平均 InfoNCE
        ``final_loss`` : 最后一个 epoch 的 InfoNCE
        ``final_kl``   : 终态 KL 项（未启用 VIB 时为 0.0）
        ``encoder_a``  : 训练后的序列编码器（原地更新）
        ``encoder_b``  : 训练后的模态编码器
    """
    set_seed(seed)
    device_t = torch.device(device)
    encoder_a.to(device_t).train()
    encoder_b.to(device_t).train()

    seq_t = torch.as_tensor(x_seq, dtype=torch.float32, device=device_t)
    mod_t = torch.as_tensor(x_mod, dtype=torch.float32, device=device_t)
    n = seq_t.size(0)

    params = list(encoder_a.parameters()) + list(encoder_b.parameters())
    if vib is not None:
        vib.to(device_t).train()
        params += list(vib.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    curve: List[float] = []
    final_kl = 0.0
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device_t)
        epoch_losses: List[float] = []
        epoch_kls: List[float] = []
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            if idx.numel() < 2:
                continue
            sb = seq_t[idx]
            mb = mod_t[idx]

            h_a = encoder_a(sb)
            h_b = encoder_b(mb)

            if vib is not None:
                z_a, mu, logvar = vib(h_a)
                kl = kl_divergence(mu, logvar)
                loss, _, _ = symmetric_infonce(z_a, h_b, tau=tau)
                loss = loss + beta * kl
                epoch_kls.append(float(kl.detach()))
            else:
                loss, _, _ = symmetric_infonce(h_a, h_b, tau=tau)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        curve.append(mean_loss)
        if vib is not None and epoch_kls:
            final_kl = float(np.mean(epoch_kls))
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  [pretrain] epoch {epoch + 1:>3}/{epochs}  InfoNCE={mean_loss:.4f}")

    return {
        "curve": curve,
        "final_loss": curve[-1] if curve else float("nan"),
        "final_kl": final_kl,
        "encoder_a": encoder_a,
        "encoder_b": encoder_b,
    }


# ======================================================================================
# 下游训练（含验证集 F1 早停）
# ======================================================================================
def _f1_from_logits(logits: torch.Tensor, y: torch.Tensor, thr: float = 0.5) -> float:
    """在固定阈值 0.5 下计算 F1（仅用于早停监控）。"""
    pred = (torch.sigmoid(logits) >= thr).float()
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    return 2 * prec * rec / (prec + rec + 1e-12)


def train_downstream(
    rep_train: np.ndarray,
    y_train: np.ndarray,
    rep_val: np.ndarray,
    y_val: np.ndarray,
    *,
    hidden: int = 80,
    epochs: int = 300,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 30,
    seed: int = 42,
    device: str = "cpu",
) -> Dict[str, object]:
    """下游二分类训练：z-score 标准化 + 2 层 MLP + BCE，验证集 F1 早停。

    参数
    ----
    rep_train / y_train : 训练集表征 (n,128) 与二分类标签 (n,)
    rep_val / y_val     : 验证集表征与标签（用于 F1 早停）
    hidden              : 分类头隐藏层维度（默认 80，参数量约 1.05 万）
    epochs              : 最大轮数
    patience            : 验证 F1 连续多少轮不提升则早停
    seed                : 随机种子

    返回
    ----
    dict：
        ``classifier``  : 训练好的分类头（最优权重）
        ``best_val_f1`` : 最优验证 F1
        ``best_epoch``  : 最优轮次
        ``mu`` / ``sigma`` : 表征标准化统计量（供评测集复用）
    """
    set_seed(seed)
    device_t = torch.device(device)

    # z-score 标准化：统计量只用训练集，避免信息泄漏
    mu = rep_train.mean(axis=0, keepdims=True)
    sigma = rep_train.std(axis=0, keepdims=True) + 1e-6
    rep_tr = (rep_train - mu) / sigma
    rep_va = (rep_val - mu) / sigma

    xt = torch.as_tensor(rep_tr, dtype=torch.float32, device=device_t)
    yt = torch.as_tensor(y_train, dtype=torch.float32, device=device_t)
    xv = torch.as_tensor(rep_va, dtype=torch.float32, device=device_t)
    yv = torch.as_tensor(y_val, dtype=torch.float32, device=device_t)

    clf = DownstreamClassifier(in_dim=rep_train.shape[1], hidden=hidden).to(device_t)
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    n = xt.size(0)
    best_f1 = -1.0
    best_epoch = -1
    best_state = copy.deepcopy(clf.state_dict())
    bad = 0

    for epoch in range(epochs):
        clf.train()
        perm = torch.randperm(n, device=device_t)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            logits = clf(xt[idx])
            loss = criterion(logits, yt[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 验证
        clf.eval()
        with torch.no_grad():
            val_logits = clf(xv)
            val_f1 = _f1_from_logits(val_logits, yv)

        if val_f1 > best_f1 + 1e-6:
            best_f1 = val_f1
            best_epoch = epoch
            best_state = copy.deepcopy(clf.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    clf.load_state_dict(best_state)
    clf.eval()
    return {
        "classifier": clf,
        "best_val_f1": best_f1,
        "best_epoch": best_epoch,
        "mu": mu,
        "sigma": sigma,
    }


def predict(
    classifier: DownstreamClassifier,
    rep: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """用训练好的分类头对表征打分，返回正类概率 (n,)。"""
    device_t = torch.device(device)
    x = (rep - mu) / sigma
    xt = torch.as_tensor(x, dtype=torch.float32, device=device_t)
    classifier.eval()
    with torch.no_grad():
        logits = classifier(xt)
        prob = torch.sigmoid(logits)
    return prob.cpu().numpy()
