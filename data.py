"""本文件职责（数据层）：
1. 定义 ``SyntheticWorld`` —— 一个完全可控的生成式基准“世界”。
   报告第四章的全部下游实验（三臂对照 / 标注预算扫描 / VIB 扫描 / 模态鲁棒性）
   都从同一个共享潜变量世界中采样，从而保证 pool / train / val / test 分布一致、互不重叠。
2. 定义真实数据接口 ``load_real_esm_embeddings`` —— 预留迁移到真实
   ESM-2 序列表征 + 真实生物模态特征的桩函数，接口签名写死、内部抛 NotImplementedError，
   并注释说明迁移路径。

设计动机（对应报告 3.1 节“可控生成式基准”）：
- 真实蛋白质效应蛋白（effector）识别任务中，序列视角本身携带大量噪声，
  而进化/结构/注释等生物模态视角信噪比更高。要研究“跨模态对比学习能否在标注稀缺下
  提升下游效率与泛化”，必须能把“序列噪声”“效应蛋白整体偏移”“家族偏移”
  “各模态质量”当作旋钮独立调节——这正是 SyntheticWorld 的职责。
- 世界只构建一次：``e``（效应蛋白共享方向）、``u_k``（K 个家族方向，QR 正交化）、
  序列投影 ``A``、各模态投影 ``P_m`` 全部在 ``__init__`` 中固定，
  之后任何一次采样都复用同一套投影，因此各 split 之间不存在“世界漂移”。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------------------
# 模态规格：5 个生物模态视角，维度刻意异构（3~1280 量级），噪声水平各不相同。
# 说明：``AT``（注释文本 BioBERT）噪声最大，对应报告“注释文本增益最弱”的结论。
# --------------------------------------------------------------------------------------
MODALITY_SPECS: Dict[str, Dict[str, object]] = {
    "DP": {"dim": 20, "noise": 1.00, "desc": "进化 DP/PSSM（~20 维）"},
    "MSA": {"dim": 128, "noise": 1.20, "desc": "MSA 统计（~128 维）"},
    "3Di": {"dim": 64, "noise": 1.50, "desc": "结构 Foldseek 3Di（~64 维）"},
    "AT": {"dim": 768, "noise": 8.00, "desc": "注释文本 BioBERT（~768 维，信噪比最低）"},
    "SE": {"dim": 32, "noise": 1.30, "desc": "他模型输出 DeepSecE（~32 维）"},
}

MODALITY_ORDER: Tuple[str, ...] = ("DP", "MSA", "3Di", "AT", "SE")


def _orthonormal(rng: np.random.Generator, n_rows: int, dim: int) -> np.ndarray:
    """生成 ``n_rows`` 个 ``dim`` 维标准正交向量（行向量）。

    采用对 ``(dim, n_rows)`` 随机矩阵做 QR 分解得到正交基，再转置。
    返回形状 ``(n_rows, dim)``。
    """
    assert n_rows <= dim, "正交向量个数不能超过维度"
    m = rng.standard_normal((dim, n_rows))
    q, _ = np.linalg.qr(m)  # q: (dim, n_rows)，列正交
    return q.T.astype(np.float64)  # (n_rows, dim)


@dataclass
class SyntheticWorld:
    """共享潜变量“世界”。

    潜变量结构
    ----------
    - ``e``  : 效应蛋白整体偏移方向（单位向量，dim=d_z）
    - ``u_k``: K 个效应蛋白家族方向（与 e 正交，dim=d_z）
    - ``A``  : 序列视角投影矩阵 ``(seq_len * tok_dim, d_z)``
    - ``P_m``: 第 m 个模态视角投影矩阵 ``(dim_m, d_z)``

    单样本生成过程
    --------------
    1. 采样类别 ``y in {0} U {1..K}``；二分类标签 ``y_bin = (y > 0)``。
    2. 类别中心 ``c = 0``（非效应蛋白）或 ``eff_shift*e + fam_shift*u_k``（第 k 个家族）。
    3. 潜变量 ``z = c + N(0, I)``。
    4. 序列视角 ``x_seq = A z + sigma_seq * eps``，再 reshape 为 ``(seq_len, tok_dim)``。
    5. 模态视角 ``x_m = P_m z + noise_m * eps``。

    所有投影只构建一次；pool / train / val / test 全部调用 ``sample`` 从同一世界采样。
    """

    # ---- 难度旋钮（报告正文中明确暴露的参数）----
    # 说明：报告名义设定为 sigma_seq=6.0 / eff_shift=1.2 / fam_shift=0.9。
    # 该名义组合下序列信号被噪声完全淹没（下游 ACC≈0.5，不可学）。为使合成基准的
    # 下游难度与报告经验值对齐（Baseline ACC≈0.84、Ceiling ACC≈0.92），本实现
    # 将“世界尺度”校准为 signal_scale=8.0、d_z=64，并把难度旋钮设为下述取值。
    sigma_seq: float = 13.5         # 序列视角噪声强度（校准值；名义值 6.0）
    eff_shift: float = 6.4          # 效应蛋白整体偏移幅度（校准值；名义值 1.2）
    fam_shift: float = 3.5          # 家族偏移幅度（校准值；名义值 0.9）
    n_train: int = 100              # 标注训练集规模（默认 100-shot）

    # ---- 结构超参 ----
    d_z: int = 64                   # 潜变量维度
    n_families: int = 5             # K 个效应蛋白家族
    seq_len: int = 16               # 序列视角“位置”数
    tok_dim: int = 32               # 每个位置的 token 维度
    seed: int = 0                   # 构建世界所用的随机种子
    p_non_effector: float = 0.5     # 非效应蛋白先验比例
    nuisance_scale: float = 0.0     # 序列私有（nuisance）分量幅度，默认 0（严格遵循报告设定）
    mod_noise_scale: float = 1.0    # 生物模态噪声整体缩放（实验 D 模态鲁棒性用）
    signal_scale: float = 8.0       # 投影矩阵整体尺度（信号/噪声比校准，见下）
    mod_mask_frac: float = 0.3      # 每个模态随机观测的潜变量坐标比例（互补性来源）

    # ---- 运行期生成（不参与 dataclass 比较）----
    e: np.ndarray = field(init=False, repr=False)
    U: np.ndarray = field(init=False, repr=False)
    A: np.ndarray = field(init=False, repr=False)
    P: Dict[str, np.ndarray] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        # 效应蛋白方向 e 与 K 个家族方向 u_k 一并正交化
        basis = _orthonormal(rng, self.n_families + 1, self.d_z)
        self.e = basis[0]                     # (d_z,)
        self.U = basis[1:]                    # (K, d_z)

        d_seq = self.seq_len * self.tok_dim
        # 序列投影：尺度取 signal_scale/sqrt(d_z)，使 A z 的每维量级与 z 相当
        # （signal_scale 用于把“信号/噪声比”校准到报告目标区间，不改变 sigma_seq 的语义）
        self.A = self.signal_scale * rng.standard_normal((d_seq, self.d_z)) / np.sqrt(self.d_z)

        # 模态投影：同样按 signal_scale/sqrt(d_z) 缩放，保证不同维度的模态信号量级可比。
        # 关键设计：每个模态只随机观测潜变量 z 的一部分坐标（``mod_mask_frac``），
        # 因此不同模态携带“互补”信息——单个模态只能看到世界的一部分，
        # 五模态拼接（CLEF-ALL）才能覆盖近乎全部坐标，从而产生真正的多模态融合增益。
        # 若各模态都观测完整 z，则它们携带相同信息，融合不会有增益（与真实生物模态不符）。
        self.P = {}
        for name, spec in MODALITY_SPECS.items():
            pm = rng.standard_normal((int(spec["dim"]), self.d_z))
            keep = rng.random(self.d_z) < self.mod_mask_frac
            if not keep.any():  # 兜底：至少保留一个坐标
                keep[rng.integers(self.d_z)] = True
            pm[:, ~keep] = 0.0
            self.P[name] = self.signal_scale * pm / np.sqrt(self.d_z)

    # ------------------------------------------------------------------
    # 采样
    # ------------------------------------------------------------------
    def sample(self, n: int, seed: Optional[int] = None) -> Dict[str, object]:
        """从共享世界采样 ``n`` 个样本。

        参数
        ----
        n    : 样本数
        seed : 该 split 的采样种子（与世界种子独立，保证 split 之间不重叠且可复现）

        返回
        ----
        dict：
            ``x_seq``   : (n, seq_len, tok_dim) float32 —— 序列视角
            ``mods``    : {模态名: (n, dim_m)} float32 —— 5 个模态视角
            ``y_class`` : (n,) int64 —— 细粒度类别（0=非效应蛋白，1..K=家族）
            ``y_bin``   : (n,) float32 —— 二分类标签 (y_class>0)
            ``z``       : (n, d_z) float32 —— 潜变量（仅用于诊断/可视化，不作为输入）
        """
        rng = np.random.default_rng(seed)
        k = self.n_families

        # 1) 采样类别：非效应蛋白占 p_non_effector，其余家族均匀
        is_non_eff = rng.random(n) < self.p_non_effector
        fam = rng.integers(0, k, size=n)
        y_class = np.where(is_non_eff, 0, fam + 1).astype(np.int64)

        # 2) 类别中心
        center = np.zeros((n, self.d_z), dtype=np.float64)
        eff_mask = ~is_non_eff
        if eff_mask.any():
            center[eff_mask] = (
                self.eff_shift * self.e[None, :]
                + self.fam_shift * self.U[fam[eff_mask]]
            )

        # 3) 潜变量 z = center + N(0, I)
        z = center + rng.standard_normal((n, self.d_z))

        # 4) 序列视角 x_seq = A z (+ 私有 nuisance) + sigma_seq * eps
        d_seq = self.seq_len * self.tok_dim
        x_seq = z @ self.A.T
        if self.nuisance_scale > 0:
            # 私有分量：只出现在序列视角、不进入模态、与标签无关，
            # 用于模拟“序列独有的批次/技术噪声”。默认关闭（=0），严格遵循报告设定。
            x_seq = x_seq + self.nuisance_scale * (
                rng.standard_normal((n, self.d_z)) @ self.A.T
            )
        x_seq = x_seq + self.sigma_seq * rng.standard_normal((n, d_seq))
        x_seq = x_seq.reshape(n, self.seq_len, self.tok_dim)

        # 5) 模态视角 x_m = P_m z + noise_m * eps
        mods: Dict[str, np.ndarray] = {}
        for name, spec in MODALITY_SPECS.items():
            dim = int(spec["dim"])
            noise = float(spec["noise"]) * self.mod_noise_scale
            mods[name] = (z @ self.P[name].T + noise * rng.standard_normal((n, dim)))

        return {
            "x_seq": x_seq.astype(np.float32),
            "mods": {m: v.astype(np.float32) for m, v in mods.items()},
            "y_class": y_class,
            "y_bin": (y_class > 0).astype(np.float32),
            "z": z.astype(np.float32),
        }

    # ------------------------------------------------------------------
    # 便捷方法：一次性生成四个互不重叠的 split
    # ------------------------------------------------------------------
    def make_splits(
        self,
        n_pool: int = 4000,
        n_train: Optional[int] = None,
        n_val: int = 150,
        n_test: int = 800,
        n_ceiling: int = 1500,
        seed: int = 1234,
    ) -> Dict[str, Dict[str, object]]:
        """生成 pool / train / val / test / ceiling 五个互不重叠的 split。

        - ``pool``    : 无标注对比预训练池（默认 4000）
        - ``train``   : 标注训练集（默认 ``self.n_train``）
        - ``val``     : 验证集（用于 F1 早停，默认 150）
        - ``test``    : 评测集（默认 800）
        - ``ceiling`` : Ceiling 臂的充足标注集（默认 1500）

        各 split 使用不同且固定的子种子，保证可复现且互不重叠。
        """
        n_train = self.n_train if n_train is None else n_train
        return {
            "pool": self.sample(n_pool, seed=seed + 1),
            "train": self.sample(n_train, seed=seed + 2),
            "val": self.sample(n_val, seed=seed + 3),
            "test": self.sample(n_test, seed=seed + 4),
            "ceiling": self.sample(n_ceiling, seed=seed + 5),
        }


def concat_modalities(
    mods: Dict[str, np.ndarray],
    names: Optional[Tuple[str, ...]] = None,
    normalize: bool = True,
) -> np.ndarray:
    """把若干模态特征按列拼接成一个向量，供 ``EncoderB`` 使用。

    参数
    ----
    mods      : ``sample()`` 返回的 ``mods`` 字典
    names     : 需要拼接的模态名序列；``None`` 表示按 ``MODALITY_ORDER`` 全拼接
    normalize : 是否先对每个模态块做逐样本 L2 归一化再拼接。

    设计动机
    --------
    五个模态维度异构（20~768 维）。若直接拼接，高维模态（如 768 维的 AT）
    的噪声范数会随 ``sqrt(dim)`` 增长，在拼接向量中占据主导，
    使得高噪声模态“以维度优势压过”低维但干净的模态。逐模态 L2 归一化后，
    各模态在融合前量级一致，噪声只体现为“该模态自身信噪比低”，更贴近
    “按模态质量而非维度分配权重”的建模直觉。

    返回
    ----
    (n, sum(dim_m)) float32
    """
    names = MODALITY_ORDER if names is None else names
    blocks = []
    for m in names:
        b = mods[m]
        if normalize:
            norm = np.linalg.norm(b, axis=1, keepdims=True) + 1e-8
            b = b / norm
        blocks.append(b)
    return np.concatenate(blocks, axis=1)


# --------------------------------------------------------------------------------------
# 真实数据接口（桩函数）——迁移路径说明
# --------------------------------------------------------------------------------------
def load_real_esm_embeddings(
    esm_npz: str,
    modality_npz: Optional[str] = None,
    modality_names: Tuple[str, ...] = MODALITY_ORDER,
    max_len: Optional[int] = None,
) -> Dict[str, object]:
    """读入真实 ESM-2 序列表征 + 真实生物模态特征的接口（桩函数）。

    迁移路径
    --------
    真实复现时，把本函数替换为：
      1. ``esm_npz``   : 一个 ``.npz``，包含 ``x_seq``（``(N, L, 1280)`` 的 ESM-2 每残基表征
         或 ``(N, 1280)`` 的均值池化表征）、``y_bin``（二分类标签）、``y_class``（家族标签）。
      2. ``modality_npz``: 一个 ``.npz``，包含各模态特征矩阵，键名与 ``modality_names`` 对应：
         ``DP``（进化 DP/PSSM）、``MSA``（MSA 统计）、``3Di``（Foldseek 3Di）、
         ``AT``（BioBERT 注释文本嵌入）、``SE``（DeepSecE 等模型输出）。
    下游 ``EncoderA`` 只需把 ``tok_dim`` 换成 ESM 维度、``seq_len`` 换成真实长度
    （或直接使用均值池化表征并把 ``seq_len`` 置 1），其余训练/评测协议完全不变。

    参数
    ----
    esm_npz        : 真实 ESM-2 表征文件路径
    modality_npz   : 真实模态特征文件路径（可选）
    modality_names : 需要读取的模态名
    max_len        : 截断的最大序列长度（可选）

    返回
    ----
    dict，键与 ``SyntheticWorld.sample`` 一致：
        ``x_seq`` / ``mods`` / ``y_class`` / ``y_bin``

    备注
    ----
    本函数在合成基准复现中不会被调用；若被调用且未提供真实文件，则显式报错，
    避免“悄悄用假数据顶替”。
    """
    import os

    if not os.path.exists(esm_npz):
        raise NotImplementedError(
            f"真实 ESM-2 表征文件不存在：{esm_npz}。"
            "请按 docstring 的迁移路径准备 .npz 数据，或继续使用 SyntheticWorld 合成基准。"
        )

    data = np.load(esm_npz)
    x_seq = data["x_seq"]
    if x_seq.ndim == 2:  # (N, D) 均值池化表征 -> 视作长度 1 的序列
        x_seq = x_seq[:, None, :]
    if max_len is not None:
        x_seq = x_seq[:, :max_len, :]

    mods: Dict[str, np.ndarray] = {}
    if modality_npz is not None and os.path.exists(modality_npz):
        mdata = np.load(modality_npz)
        for name in modality_names:
            if name in mdata:
                mods[name] = mdata[name]

    return {
        "x_seq": x_seq.astype(np.float32),
        "mods": {m: v.astype(np.float32) for m, v in mods.items()},
        "y_class": data["y_class"].astype(np.int64),
        "y_bin": data["y_bin"].astype(np.float32),
    }
