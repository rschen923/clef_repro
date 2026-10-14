"""本文件职责（评估层）：
    - ``classification_metrics``        下游二分类指标（ACC/REC/PR/F1/MCC/AUPRC）
    - ``representation_diagnostics``    表征诊断指标（ARI/NMI/ASW，无需标签即可算）
    - ``annotation_substitution_rate``  核心指标：标注替代率 Δ_ASR
    - ``fit_pca``                       PCA 可视化降维

设计动机（对应报告 4.2 节“评估协议”）：
- 下游指标直接刻画任务性能；表征诊断指标刻画“表征本身是否按真实组别聚起来”，
  属于过程性证据——即便标注稀缺，只要表征诊断指标高，就说明对比预训练确实学到了结构。
- 标注替代率 Δ_ASR 把“CLEF 相对 Baseline 的提升”归一到“Ceiling 相对 Baseline 的上限”，
  量化“多少比例的标注被省下来”。
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE  # noqa: F401  (保留以便扩展)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
    silhouette_score,
    adjusted_rand_score,
)


# ======================================================================================
# 下游分类指标
# ======================================================================================
def classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5
) -> Dict[str, float]:
    """下游二分类指标：ACC / REC / PR / F1 / MCC / AUPRC。

    参数
    ----
    y_true : (n,) 真实二分类标签 {0,1}
    y_prob : (n,) 正类预测概率
    thr    : 判定阈值（默认 0.5）

    返回
    ----
    dict，键：``ACC`` ``REC`` ``PR`` ``F1`` ``MCC`` ``AUPRC``
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "ACC": float(accuracy_score(y_true, y_pred)),
        "REC": float(recall_score(y_true, y_pred, zero_division=0)),
        "PR": float(precision_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "AUPRC": float(average_precision_score(y_true, y_prob)),
    }


# ======================================================================================
# 表征诊断指标（无需标签即可算，属过程性证据）
# ======================================================================================
def representation_diagnostics(
    rep: np.ndarray, y_class: np.ndarray, seed: int = 42, max_asw: int = 2000
) -> Dict[str, float]:
    """表征诊断：ARI / NMI（对真实组别做 KMeans 聚类后比对）与 ASW（轮廓系数）。

    参数
    ----
    rep     : (n, d) 表征
    y_class : (n,) 真实细粒度组别（0=非效应蛋白，1..K=家族）
    seed    : KMeans / 采样随机种子
    max_asw : ASW 计算的最大样本数（轮廓系数 O(n^2)，超过则下采样）

    返回
    ----
    dict，键：``ARI`` ``NMI`` ``ASW``
    """
    rep = np.asarray(rep, dtype=np.float64)
    y_class = np.asarray(y_class)
    n_clusters = int(len(np.unique(y_class)))

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    pred = km.fit_predict(rep)
    ari = float(adjusted_rand_score(y_class, pred))
    nmi = float(normalized_mutual_info_score(y_class, pred))

    if rep.shape[0] > max_asw:
        rng = np.random.default_rng(seed)
        idx = rng.choice(rep.shape[0], size=max_asw, replace=False)
        asw = float(silhouette_score(rep[idx], y_class[idx]))
    else:
        asw = float(silhouette_score(rep, y_class))
    return {"ARI": ari, "NMI": nmi, "ASW": asw}


# ======================================================================================
# 标注替代率
# ======================================================================================
def annotation_substitution_rate(
    clef: float, baseline: float, ceiling: float
) -> float:
    """标注替代率 Δ_ASR = (CLEF − Baseline) / (Ceiling − Baseline)。

    - Δ_ASR ≈ 1 表示 CLEF 在稀缺标注下已达到 Ceiling 水平（标注几乎全部被替代）；
    - Δ_ASR ≈ 0 表示 CLEF 与 Baseline 无异；
    - 分母接近 0 时返回 0.0，避免除零。
    """
    denom = ceiling - baseline
    if abs(denom) < 1e-8:
        return 0.0
    return float((clef - baseline) / denom)


# ======================================================================================
# PCA 可视化降维
# ======================================================================================
def fit_pca(X: np.ndarray, n_components: int = 2, seed: int = 42) -> np.ndarray:
    """对表征做 PCA，返回前 ``n_components`` 个主成分坐标 (n, n_components)。"""
    X = np.asarray(X, dtype=np.float64)
    pca = PCA(n_components=n_components, random_state=seed)
    return pca.fit_transform(X)


# ======================================================================================
# 便捷：一次性评估一个实验臂
# ======================================================================================
def evaluate_arm(
    rep_test: np.ndarray,
    y_test: np.ndarray,
    y_class_test: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """合并下游指标 + 表征诊断指标，返回单个实验臂的完整结果字典。"""
    out = classification_metrics(y_test, y_prob)
    out.update(representation_diagnostics(rep_test, y_class_test))
    return out
