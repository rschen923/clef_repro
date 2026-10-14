"""本文件职责（编排层）：
把数据层 / 模型层 / 训练层 / 评估层串成四组可复现实验，并通过 argparse 暴露子命令：

    实验 A  三臂对照（固定 100-shot）  —— 8 个实验臂 + Δ_ASR
    实验 B  标注预算扫描              —— n_train ∈ {50,100,200} × seed ∈ {42,7}
    实验 C  VIB β 扫描                —— β ∈ {0.0, 0.01, 0.05}
    实验 D  模态质量退化鲁棒性         —— 模态噪声 scale ∈ {1,2,4}

用法：
    python run_experiment.py A          # 只跑实验 A
    python run_experiment.py B
    python run_experiment.py C
    python run_experiment.py D
    python run_experiment.py all        # 依次跑 A/B/C/D

所有结果写入 ``results/``（csv + json），供 plots.py 绘图、供报告正文引用。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from data import MODALITY_ORDER, MODALITY_SPECS, SyntheticWorld, concat_modalities
from evaluate import (
    annotation_substitution_rate,
    classification_metrics,
    evaluate_arm,
    fit_pca,
    representation_diagnostics,
)
from model import (
    DownstreamClassifier,
    EncoderA,
    EncoderB,
    FrozenSeqBaseline,
    VIBEncoder,
    count_trainable_params,
)
from train import predict, pretrain_contrastive, set_seed, train_downstream

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
FIGURES_DIR = os.path.join(HERE, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def rel(p: str) -> str:
    """把绝对路径转成相对仓库根目录的路径，仅用于日志与终端输出。

    读写文件仍用绝对路径，但打印给用户看的路径一律相对化，
    以免日志中泄漏本机的绝对目录结构。
    """
    try:
        return os.path.relpath(p, HERE).replace("\\", "/")
    except ValueError:
        return p


# ======================================================================================
# 通用工具
# ======================================================================================
def _encode(module: torch.nn.Module, x: np.ndarray, device: str = "cpu", batch: int = 512) -> np.ndarray:
    """用冻结模块对输入做批量前向，返回 numpy 表征。"""
    module.eval()
    dev = torch.device(device)
    outs: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch):
            xb = torch.as_tensor(x[i : i + batch], dtype=torch.float32, device=dev)
            outs.append(module(xb).cpu().numpy())
    return np.concatenate(outs, axis=0)


def _build_world(args: argparse.Namespace, *, mod_noise_scale: float = 1.0) -> SyntheticWorld:
    """按命令行参数构建共享世界。"""
    return SyntheticWorld(
        sigma_seq=args.sigma_seq,
        eff_shift=args.eff_shift,
        fam_shift=args.fam_shift,
        n_train=args.n_train,
        seed=args.world_seed,
        mod_noise_scale=mod_noise_scale,
        signal_scale=args.signal_scale,
        d_z=args.d_z,
        tok_dim=args.tok_dim,
        seq_len=args.seq_len,
        mod_mask_frac=args.mod_mask_frac,
    )


def _frozen_baseline_rep(
    world: SyntheticWorld, splits: Dict[str, Dict[str, object]], seed: int = 0
) -> Tuple[Dict[str, np.ndarray], int]:
    """计算 Baseline/Ceiling 臂的冻结序列表征（每个 split 一份）。"""
    model = FrozenSeqBaseline(world.tok_dim, out_dim=128, seed=seed)
    reps = {name: _encode(model, sp["x_seq"]) for name, sp in splits.items()}  # type: ignore[arg-type]
    return reps, count_trainable_params(model)  # 冻结投影无参数 -> 0


def _clef_rep(
    world: SyntheticWorld,
    splits: Dict[str, Dict[str, object]],
    mod_names: Sequence[str],
    *,
    pre_epochs: int,
    beta: float = 0.0,
    tau: float = 0.5,
    seed: int = 42,
    device: str = "cpu",
    verbose: bool = False,
) -> Dict[str, object]:
    """对比预训练 + 生成各 split 的 CLEF 序列表征。

    返回 dict：``reps`` / ``curve`` / ``final_kl`` / ``params`` / ``inference_params``。
    """
    in_dim = int(sum(MODALITY_SPECS[m]["dim"] for m in mod_names))
    # 先固定随机种子再创建编码器，保证权重初始化可复现（不依赖调用前的环境 RNG 状态）
    set_seed(seed)
    enc_a = EncoderA(world.tok_dim, world.seq_len)
    enc_b = EncoderB(in_dim)
    vib = VIBEncoder() if beta > 0 else None

    pool_seq = splits["pool"]["x_seq"]  # type: ignore[index]
    pool_mod = concat_modalities(splits["pool"]["mods"], tuple(mod_names))  # type: ignore[index]

    info = pretrain_contrastive(
        pool_seq, pool_mod, enc_a, enc_b,
        vib=vib, beta=beta, epochs=pre_epochs, tau=tau, seed=seed,
        device=device, verbose=verbose,
    )

    # 表征：启用 VIB 时取均值 μ（确定性），否则取 EncoderA 输出
    def _rep(x_seq: np.ndarray) -> np.ndarray:
        h = _encode(enc_a, x_seq, device=device)
        if vib is not None:
            with torch.no_grad():
                mu = vib.mu_layer(torch.as_tensor(h, dtype=torch.float32))
            return mu.numpy()
        return h

    reps = {name: _rep(sp["x_seq"]) for name, sp in splits.items()}  # type: ignore[arg-type]

    params = count_trainable_params(enc_a) + count_trainable_params(enc_b)
    if vib is not None:
        params += count_trainable_params(vib)
    inference_params = count_trainable_params(enc_a)
    return {
        "reps": reps,
        "curve": info["curve"],
        "final_kl": info["final_kl"],
        "params": params,
        "inference_params": inference_params,
    }


def _downstream_and_eval(
    rep_source: Dict[str, np.ndarray],
    splits: Dict[str, Dict[str, object]],
    *,
    n_train: int,
    train_key: str = "train",
    hidden: int = 80,
    seed: int = 42,
    device: str = "cpu",
) -> Dict[str, float]:
    """在给定表征来源上训练下游分类头并评测 test 集，返回完整指标字典。

    ``train_key`` 指定标注来源 split：普通实验臂用 ``"train"``，
    Ceiling 臂用 ``"ceiling"``（充足标注）。
    """
    train_rep = rep_source[train_key][:n_train]
    y_train = splits[train_key]["y_bin"][:n_train]  # type: ignore[index]
    fit = train_downstream(
        train_rep, y_train,
        rep_source["val"], splits["val"]["y_bin"],  # type: ignore[index]
        hidden=hidden, seed=seed, device=device,
    )
    prob = predict(fit["classifier"], rep_source["test"], fit["mu"], fit["sigma"], device=device)
    out = evaluate_arm(
        rep_source["test"],
        splits["test"]["y_bin"],  # type: ignore[index]
        splits["test"]["y_class"],  # type: ignore[index]
        prob,
    )
    out["best_val_f1"] = float(fit["best_val_f1"])
    out["best_epoch"] = int(fit["best_epoch"])
    out["head_params"] = count_trainable_params(fit["classifier"])
    return out


# ======================================================================================
# 实验 A · 三臂对照（固定 100-shot）
# ======================================================================================
ARM_SPECS: List[Tuple[str, str, Optional[Tuple[str, ...]]]] = [
    ("Baseline", "baseline", None),
    ("CLEF-DP", "clef", ("DP",)),
    ("CLEF-MSA", "clef", ("MSA",)),
    ("CLEF-3Di", "clef", ("3Di",)),
    ("CLEF-AT", "clef", ("AT",)),
    ("CLEF-SE", "clef", ("SE",)),
    ("CLEF-ALL", "clef", tuple(MODALITY_ORDER)),
    ("Ceiling", "baseline", None),
]


def run_experiment_a(args: argparse.Namespace) -> Dict[str, object]:
    """实验 A：8 个实验臂的 100-shot 三臂对照，输出报告表 4-1。"""
    t0 = time.time()
    world = _build_world(args)
    splits = world.make_splits(
        n_pool=args.n_pool, n_train=args.n_train, n_val=args.n_val,
        n_test=args.n_test, n_ceiling=args.n_ceiling, seed=args.data_seed,
    )
    print(f"[A] 世界构建完成：sigma_seq={args.sigma_seq} eff_shift={args.eff_shift} "
          f"fam_shift={args.fam_shift} n_pool={args.n_pool}")

    rows: List[Dict[str, object]] = []
    extra: Dict[str, object] = {}

    baseline_reps, baseline_frozen_params = _frozen_baseline_rep(world, splits)
    # 基线/天花板共用同一冻结表征，只是标注预算不同
    for arm, kind, mods in ARM_SPECS:
        n_train = args.n_ceiling if arm == "Ceiling" else args.n_train
        if kind == "baseline":
            rep_source = baseline_reps
            trainable = baseline_frozen_params  # 冻结骨干 -> 0
            curve, final_kl = [], 0.0
        else:
            clef = _clef_rep(
                world, splits, mods, pre_epochs=args.pre_epochs, beta=0.0,
                tau=args.tau, seed=args.seed, device=args.device, verbose=False,
            )
            rep_source = clef["reps"]
            trainable = clef["params"]
            curve, final_kl = clef["curve"], clef["final_kl"]
            extra[f"curve_{arm}"] = curve

        metrics = _downstream_and_eval(
            rep_source, splits, n_train=n_train,
            train_key=("ceiling" if arm == "Ceiling" else "train"),
            seed=args.seed, device=args.device,
        )
        row = {
            "方法": arm,
            "模态": "-" if mods is None else "+".join(mods),
            "标注数": n_train,
            "可训练参数": int(trainable) + int(metrics["head_params"]) if kind == "clef" else int(metrics["head_params"]),
            "ACC": metrics["ACC"], "REC": metrics["REC"], "PR": metrics["PR"],
            "F1": metrics["F1"], "MCC": metrics["MCC"], "AUPRC": metrics["AUPRC"],
            "ARI": metrics["ARI"], "NMI": metrics["NMI"], "ASW": metrics["ASW"],
        }
        rows.append(row)
        print(f"  [A] {arm:<10} ACC={row['ACC']:.4f} F1={row['F1']:.4f} "
              f"MCC={row['MCC']:.4f} AUPRC={row['AUPRC']:.4f} ARI={row['ARI']:.4f}")

    # Δ_ASR
    def _get(arm: str, key: str) -> float:
        return float(next(r[key] for r in rows if r["方法"] == arm))

    asr_acc = annotation_substitution_rate(_get("CLEF-ALL", "ACC"), _get("Baseline", "ACC"), _get("Ceiling", "ACC"))
    asr_mcc = annotation_substitution_rate(_get("CLEF-ALL", "MCC"), _get("Baseline", "MCC"), _get("Ceiling", "MCC"))

    # 保存 PCA 可视化数据（Baseline vs CLEF-ALL）
    pca_pack = {
        "baseline_rep": baseline_reps["test"],
        "clef_rep": extra.get("rep_CLEF-ALL"),
        "y_class": splits["test"]["y_class"],  # type: ignore[index]
        "y_bin": splits["test"]["y_bin"],  # type: ignore[index]
    }
    # 若上面未保存 CLEF-ALL 表征，重算一次以用于 PCA
    if pca_pack["clef_rep"] is None:
        clef_all = _clef_rep(world, splits, tuple(MODALITY_ORDER), pre_epochs=args.pre_epochs,
                             seed=args.seed, device=args.device)
        pca_pack["clef_rep"] = clef_all["reps"]["test"]

    result = {
        "config": vars(args),
        "rows": rows,
        "delta_asr_acc": asr_acc,
        "delta_asr_mcc": asr_mcc,
        "elapsed_sec": time.time() - t0,
        "pca": pca_pack,
    }
    _save_exp("exp_a", result)
    # 另存对比预训练曲线（供 fig4_3）
    with open(os.path.join(RESULTS_DIR, "curves.json"), "w", encoding="utf-8") as f:
        json.dump({"CLEF-ALL": extra.get("curve_CLEF-ALL", [])}, f, ensure_ascii=False, indent=2)
    np.savez(os.path.join(RESULTS_DIR, "pca_data.npz"),
             baseline_rep=pca_pack["baseline_rep"], clef_rep=pca_pack["clef_rep"],
             y_class=pca_pack["y_class"], y_bin=pca_pack["y_bin"])
    print(f"[A] 完成，用时 {result['elapsed_sec']:.1f}s，Δ_ASR(ACC)={asr_acc:.3f} Δ_ASR(MCC)={asr_mcc:.3f}")
    return result


# ======================================================================================
# 实验 B · 标注预算扫描
# ======================================================================================
def run_experiment_b(args: argparse.Namespace) -> Dict[str, object]:
    """实验 B：n_train ∈ {50,100,200} × seed ∈ {42,7}，报告跨种子均值±标准差与 Δ_ASR。"""
    t0 = time.time()
    n_train_list = [50, 100, 200]
    seeds = [42, 7]
    per_cell: List[Dict[str, object]] = []

    for seed in seeds:
        world = _build_world(args)
        splits = world.make_splits(
            n_pool=args.n_pool, n_train=200, n_val=args.n_val,
            n_test=args.n_test, n_ceiling=args.n_ceiling, seed=args.data_seed + seed,
        )
        # 冻结骨干是固定的（与随机种子无关），故投影种子恒为 0；只有数据划分与训练种子随 seed 变化
        baseline_reps, _ = _frozen_baseline_rep(world, splits, seed=0)
        clef = _clef_rep(world, splits, tuple(MODALITY_ORDER), pre_epochs=args.pre_epochs,
                         seed=seed, device=args.device)
        for n_train in n_train_list:
            base = _downstream_and_eval(baseline_reps, splits, n_train=n_train, seed=seed, device=args.device)
            cl = _downstream_and_eval(clef["reps"], splits, n_train=n_train, seed=seed, device=args.device)
            ceil = _downstream_and_eval(baseline_reps, splits, n_train=args.n_ceiling,
                                        train_key="ceiling", seed=seed, device=args.device)
            asr_acc = annotation_substitution_rate(cl["ACC"], base["ACC"], ceil["ACC"])
            asr_mcc = annotation_substitution_rate(cl["MCC"], base["MCC"], ceil["MCC"])
            per_cell.append({
                "seed": seed, "n_train": n_train,
                "baseline_ACC": base["ACC"], "baseline_MCC": base["MCC"],
                "clef_ACC": cl["ACC"], "clef_MCC": cl["MCC"],
                "ceiling_ACC": ceil["ACC"], "ceiling_MCC": ceil["MCC"],
                "asr_ACC": asr_acc, "asr_MCC": asr_mcc,
            })
            print(f"  [B] seed={seed} n={n_train:>3} | base ACC={base['ACC']:.3f} "
                  f"CLEF ACC={cl['ACC']:.3f} ceil ACC={ceil['ACC']:.3f} | Δ_ASR(ACC)={asr_acc:.3f}")

    # 聚合跨种子
    agg: List[Dict[str, object]] = []
    for n_train in n_train_list:
        cells = [c for c in per_cell if c["n_train"] == n_train]
        row = {"n_train": n_train}
        for key in ["baseline_ACC", "baseline_MCC", "clef_ACC", "clef_MCC",
                    "ceiling_ACC", "ceiling_MCC", "asr_ACC", "asr_MCC"]:
            vals = np.array([c[key] for c in cells], dtype=float)
            row[key + "_mean"] = float(vals.mean())
            row[key + "_std"] = float(vals.std(ddof=0))
        agg.append(row)

    result = {"config": vars(args), "per_cell": per_cell, "agg": agg,
              "elapsed_sec": time.time() - t0}
    _save_exp("exp_b", result)
    print(f"[B] 完成，用时 {result['elapsed_sec']:.1f}s")
    return result


# ======================================================================================
# 实验 C · VIB β 扫描
# ======================================================================================
def run_experiment_c(args: argparse.Namespace) -> Dict[str, object]:
    """实验 C：β ∈ {0.0, 0.01, 0.05}，n_train=100，报告 ACC/MCC/终态 KL。"""
    t0 = time.time()
    betas = [0.0, 0.01, 0.05]
    world = _build_world(args)
    splits = world.make_splits(
        n_pool=args.n_pool, n_train=args.n_train, n_val=args.n_val,
        n_test=args.n_test, n_ceiling=args.n_ceiling, seed=args.data_seed,
    )
    rows: List[Dict[str, object]] = []
    for beta in betas:
        clef = _clef_rep(world, splits, tuple(MODALITY_ORDER), pre_epochs=args.pre_epochs,
                         beta=beta, seed=args.seed, device=args.device)
        m = _downstream_and_eval(clef["reps"], splits, n_train=args.n_train,
                                 seed=args.seed, device=args.device)
        rows.append({"beta": beta, "ACC": m["ACC"], "MCC": m["MCC"],
                     "F1": m["F1"], "AUPRC": m["AUPRC"], "final_kl": clef["final_kl"]})
        print(f"  [C] β={beta:<5} ACC={m['ACC']:.4f} MCC={m['MCC']:.4f} KL={clef['final_kl']:.3f}")

    result = {"config": vars(args), "rows": rows, "elapsed_sec": time.time() - t0}
    _save_exp("exp_c", result)
    print(f"[C] 完成，用时 {result['elapsed_sec']:.1f}s")
    return result


# ======================================================================================
# 实验 D · 模态质量退化鲁棒性
# ======================================================================================
def run_experiment_d(args: argparse.Namespace) -> Dict[str, object]:
    """实验 D：模态噪声 scale ∈ {1,2,4}，对比 Baseline 与 CLEF-ALL 的 ACC/MCC 退化。"""
    t0 = time.time()
    scales = [1, 2, 4]
    rows: List[Dict[str, object]] = []
    for scale in scales:
        world = _build_world(args, mod_noise_scale=float(scale))
        splits = world.make_splits(
            n_pool=args.n_pool, n_train=args.n_train, n_val=args.n_val,
            n_test=args.n_test, n_ceiling=args.n_ceiling, seed=args.data_seed,
        )
        baseline_reps, _ = _frozen_baseline_rep(world, splits)
        base = _downstream_and_eval(baseline_reps, splits, n_train=args.n_train,
                                    seed=args.seed, device=args.device)
        clef = _clef_rep(world, splits, tuple(MODALITY_ORDER), pre_epochs=args.pre_epochs,
                         seed=args.seed, device=args.device)
        cl = _downstream_and_eval(clef["reps"], splits, n_train=args.n_train,
                                  seed=args.seed, device=args.device)
        rows.append({"scale": scale, "baseline_ACC": base["ACC"], "baseline_MCC": base["MCC"],
                     "clef_ACC": cl["ACC"], "clef_MCC": cl["MCC"],
                     "clef_F1": cl["F1"], "baseline_F1": base["F1"]})
        print(f"  [D] scale={scale} | base ACC={base['ACC']:.4f} MCC={base['MCC']:.4f} "
              f"| CLEF ACC={cl['ACC']:.4f} MCC={cl['MCC']:.4f}")

    result = {"config": vars(args), "rows": rows, "elapsed_sec": time.time() - t0}
    _save_exp("exp_d", result)
    print(f"[D] 完成，用时 {result['elapsed_sec']:.1f}s")
    return result


# ======================================================================================
# 结果落盘
# ======================================================================================
def _save_exp(name: str, result: Dict[str, object]) -> None:
    """把实验结果写成 json + csv（csv 只写主表，便于报告引用）。"""
    import csv

    # json（含 config 与全部明细；numpy 数组转 list）
    def _default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return str(o)

    # 大数组（如 PCA 原始表征）单独存 npz，json 只保留数值表，避免文件过大
    payload = {k: v for k, v in result.items() if k != "pca"}
    with open(os.path.join(RESULTS_DIR, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_default)

    # csv
    if name == "exp_a":
        _write_csv(os.path.join(RESULTS_DIR, "exp_a.csv"), result["rows"])  # type: ignore[arg-type]
    elif name == "exp_b":
        _write_csv(os.path.join(RESULTS_DIR, "exp_b_per_cell.csv"), result["per_cell"])  # type: ignore[arg-type]
        _write_csv(os.path.join(RESULTS_DIR, "exp_b.csv"), result["agg"])  # type: ignore[arg-type]
    elif name == "exp_c":
        _write_csv(os.path.join(RESULTS_DIR, "exp_c.csv"), result["rows"])  # type: ignore[arg-type]
    elif name == "exp_d":
        _write_csv(os.path.join(RESULTS_DIR, "exp_d.csv"), result["rows"])  # type: ignore[arg-type]


def _write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    import csv

    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


# ======================================================================================
# 命令行
# ======================================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CLEF 跨模态对比学习复现实验编排")
    p.add_argument("exp", choices=["A", "B", "C", "D", "all"], help="要运行的实验")

    # 世界难度旋钮（报告名义设定为 6.0/1.2/0.9；此处为校准后使 Baseline ACC≈0.84 的取值）
    p.add_argument("--sigma_seq", type=float, default=13.5, help="序列视角噪声强度")
    p.add_argument("--eff_shift", type=float, default=6.4, help="效应蛋白整体偏移幅度")
    p.add_argument("--fam_shift", type=float, default=3.5, help="家族偏移幅度")
    p.add_argument("--world_seed", type=int, default=0, help="构建世界的随机种子")
    p.add_argument("--signal_scale", type=float, default=8.0, help="投影矩阵整体尺度（信噪比校准）")
    p.add_argument("--d_z", type=int, default=64, help="潜变量维度")
    p.add_argument("--tok_dim", type=int, default=32, help="序列位置 token 维度")
    p.add_argument("--seq_len", type=int, default=16, help="序列长度")
    p.add_argument("--mod_mask_frac", type=float, default=0.3,
                   help="每个模态随机观测的潜变量坐标比例（越小则单模态越弱、融合增益越明显）")

    # 划分规模
    p.add_argument("--n_pool", type=int, default=4000, help="无标注对比预训练池规模")
    p.add_argument("--n_train", type=int, default=100, help="标注训练集规模（100-shot）")
    p.add_argument("--n_val", type=int, default=150, help="验证集规模")
    p.add_argument("--n_test", type=int, default=800, help="测试集规模")
    p.add_argument("--n_ceiling", type=int, default=1500, help="Ceiling 臂标注规模")
    p.add_argument("--data_seed", type=int, default=1234, help="数据划分随机种子")

    # 训练协议
    p.add_argument("--pre_epochs", type=int, default=50, help="对比预训练轮数（A 用 50，B/C/D 用 30）")
    p.add_argument("--tau", type=float, default=0.5, help="InfoNCE 温度")
    p.add_argument("--seed", type=int, default=7,
                   help="协议随机种子（实验 B 固定用 {42,7} 两组；报告未指定 A 的具体种子，取 7）")
    p.add_argument("--device", type=str, default="cpu", help="计算设备（全程 CPU）")
    return p


def main() -> None:
    args = build_parser().parse_args()
    t0 = time.time()
    summary: Dict[str, object] = {}

    if args.exp in ("A", "all"):
        summary["A"] = run_experiment_a(args)
    if args.exp in ("B", "all"):
        summary["B"] = run_experiment_b(args)
    if args.exp in ("C", "all"):
        summary["C"] = run_experiment_c(args)
    if args.exp in ("D", "all"):
        summary["D"] = run_experiment_d(args)

    # 汇总 json（去掉大数组，避免文件过大）
    def _strip(o):
        if isinstance(o, dict):
            return {k: _strip(v) for k, v in o.items() if k != "pca"}
        if isinstance(o, list):
            return [_strip(v) for v in o]
        return o

    with open(os.path.join(RESULTS_DIR, "all_results.json"), "w", encoding="utf-8") as f:
        json.dump(_strip(summary), f, ensure_ascii=False, indent=2,
                  default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
    print(f"\n[ALL] 全部实验完成，总用时 {time.time() - t0:.1f}s，结果已写入 {rel(RESULTS_DIR)}/")


if __name__ == "__main__":
    main()
