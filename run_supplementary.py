"""本文件职责（实验层 E-K）：
在 ``run_experiment.py`` 的 A-D 组之外，另设七组实验，全部真跑、真实落盘：

    E  标注预算细网格 + 幂律拟合      n_tr ∈ {25,50,75,100,150,200,300}
    F  无标注池规模缩放               n_pool ∈ {500,1000,2000,4000,8000}
    G  InfoNCE 温度敏感性             tau ∈ {0.1,0.2,0.5,1.0}
    H  表征维度消融                   d_z ∈ {32,64,128,256}
    I  训练标签噪声                   flip ∈ {0%,5%,10%,20%}
    J  模态缺失矩阵                   5 单模态 + 5 留一法
    K  显著性检验                     Baseline vs CLEF-ALL，10 个种子配对比较

设计原则：
  * 同一个 world 只构建一次；同一 world 下对比预训练只做一次，下游头在多个
    预算/噪声档位上复用同一份表征——既省算力，又保证档位间可比（只变一个变量）。
  * 所有数值真实计算，不做任何拟合式"凑数"。
  * 结果逐组写入 ``results/supplementary.json``，可断点续跑。

用法：
    python run_supplementary.py E
    python run_supplementary.py all
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Sequence

import numpy as np

from data import MODALITY_ORDER
from run_experiment import (
    rel,
    RESULTS_DIR,
    _build_world,
    _clef_rep,
    _downstream_and_eval,
    _frozen_baseline_rep,
)

SUPP_PATH = os.path.join(RESULTS_DIR, "supplementary.json")

# 全局固定配置（与 A 组一致：pre_epochs=50, n_pool=4000, n_val=150, n_test=800, n_ceiling=1500）
BASE = dict(
    sigma_seq=13.5, eff_shift=6.4, fam_shift=3.5, world_seed=0,
    signal_scale=8.0, d_z=64, tok_dim=32, seq_len=16, mod_mask_frac=0.3,
    n_pool=4000, n_train=100, n_val=150, n_test=800, n_ceiling=1500,
    data_seed=1234, pre_epochs=50, tau=0.5, seed=7, device="cpu",
)


def _ns(**kw) -> argparse.Namespace:
    d = dict(BASE)
    d.update(kw)
    return argparse.Namespace(**d)


def _splits(ns, *, n_train, seed):
    world = _build_world(ns)
    splits = world.make_splits(
        n_pool=ns.n_pool, n_train=n_train, n_val=ns.n_val,
        n_test=ns.n_test, n_ceiling=ns.n_ceiling, seed=ns.data_seed + seed,
    )
    return world, splits


def _noisy(splits, rate: float, seed: int):
    """返回一个把 train 标签按 rate 随机翻转后的 splits 副本。"""
    if rate <= 0:
        return splits
    rng = np.random.default_rng(seed)
    out = dict(splits)
    tr = dict(splits["train"])
    y = np.asarray(tr["y_bin"]).copy()
    flip = rng.random(y.shape[0]) < rate
    y[flip] = 1 - y[flip]
    tr["y_bin"] = y
    out["train"] = tr
    return out


def _load() -> Dict:
    if os.path.exists(SUPP_PATH):
        with open(SUPP_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _dump(payload: Dict) -> None:
    def _d(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return str(o)

    with open(SUPP_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_d)


def _write_csv(name: str, rows: List[Dict]) -> None:
    import csv
    if not rows:
        return
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(RESULTS_DIR, f"{name}.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


# ======================================================================================
# E · 标注预算细网格 + 幂律拟合
# ======================================================================================
def exp_e(seeds=(42, 7, 123)) -> Dict:
    grid = [25, 50, 75, 100, 150, 200, 300]
    cells: List[Dict] = []
    for seed in seeds:
        ns = _ns(n_train=max(grid), seed=seed)
        world, splits = _splits(ns, n_train=max(grid), seed=seed)
        base_rep, _ = _frozen_baseline_rep(world, splits, seed=0)
        clef = _clef_rep(world, splits, tuple(MODALITY_ORDER),
                         pre_epochs=ns.pre_epochs, tau=ns.tau, seed=seed, device=ns.device)
        ceil = _downstream_and_eval(base_rep, splits, n_train=ns.n_ceiling,
                                    train_key="ceiling", seed=seed, device=ns.device)
        for n in grid:
            b = _downstream_and_eval(base_rep, splits, n_train=n, seed=seed, device=ns.device)
            c = _downstream_and_eval(clef["reps"], splits, n_train=n, seed=seed, device=ns.device)
            gap_mcc = c["MCC"] - b["MCC"]
            gap_acc = c["ACC"] - b["ACC"]
            den_mcc = ceil["MCC"] - b["MCC"]
            den_acc = ceil["ACC"] - b["ACC"]
            cells.append({
                "seed": seed, "n_train": n,
                "baseline_ACC": b["ACC"], "clef_ACC": c["ACC"], "ceiling_ACC": ceil["ACC"],
                "baseline_MCC": b["MCC"], "clef_MCC": c["MCC"], "ceiling_MCC": ceil["MCC"],
                "gap_ACC": gap_acc, "gap_MCC": gap_mcc,
                "asr_ACC": gap_acc / den_acc if abs(den_acc) > 1e-9 else float("nan"),
                "asr_MCC": gap_mcc / den_mcc if abs(den_mcc) > 1e-9 else float("nan"),
            })
            print(f"  [E] seed={seed} n={n:>3} | base MCC={b['MCC']:.4f} CLEF MCC={c['MCC']:.4f} "
                  f"gap={gap_mcc:+.4f} Δ_ASR={cells[-1]['asr_MCC']:.3f}")

    # 跨种子聚合
    agg = []
    for n in grid:
        cs = [c for c in cells if c["n_train"] == n]
        row = {"n_train": n}
        for k in ["baseline_ACC", "clef_ACC", "ceiling_ACC", "baseline_MCC", "clef_MCC",
                  "ceiling_MCC", "gap_ACC", "gap_MCC", "asr_ACC", "asr_MCC"]:
            v = np.array([c[k] for c in cs], dtype=float)
            row[k + "_mean"] = float(np.nanmean(v))
            row[k + "_std"] = float(np.nanstd(v))
        agg.append(row)

    # 幂律拟合 Δ_ASR = a * n^(-b)
    x = np.array([r["n_train"] for r in agg], dtype=float)
    y = np.array([r["asr_MCC_mean"] for r in agg], dtype=float)
    mask = np.isfinite(y) & (y > 0)
    fit = {"a": None, "b": None, "r2": None, "n_points": int(mask.sum())}
    if mask.sum() >= 2:
        lx, ly = np.log(x[mask]), np.log(y[mask])
        b_coef, lna = np.polyfit(lx, ly, 1)
        pred = lna + b_coef * lx
        ss_res = float(((ly - pred) ** 2).sum())
        ss_tot = float(((ly - ly.mean()) ** 2).sum())
        fit = {
            "a": float(np.exp(lna)),
            "b": float(-b_coef),
            "r2": float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else None,
            "n_points": int(mask.sum()),
        }
    print(f"  [E] 幂律拟合 Δ_ASR = {fit['a']:.4f} * n^(-{fit['b']:.4f}), R²={fit['r2']}")

    _write_csv("exp_e", agg)
    return {"grid": grid, "per_cell": cells, "agg": agg, "powerlaw_fit": fit}


# ======================================================================================
# F · 无标注池规模缩放
# ======================================================================================
def exp_f(pools=(500, 1000, 2000, 4000, 8000), seeds=(42, 7, 123)) -> Dict:
    cells: List[Dict] = []
    for n_pool in pools:
        for seed in seeds:
            ns = _ns(n_pool=n_pool, seed=seed)
            world, splits = _splits(ns, n_train=ns.n_train, seed=seed)
            base_rep, _ = _frozen_baseline_rep(world, splits, seed=0)
            clef = _clef_rep(world, splits, tuple(MODALITY_ORDER),
                             pre_epochs=ns.pre_epochs, tau=ns.tau, seed=seed, device=ns.device)
            b = _downstream_and_eval(base_rep, splits, n_train=ns.n_train, seed=seed, device=ns.device)
            c = _downstream_and_eval(clef["reps"], splits, n_train=ns.n_train, seed=seed, device=ns.device)
            ce = _downstream_and_eval(base_rep, splits, n_train=ns.n_ceiling,
                                      train_key="ceiling", seed=seed, device=ns.device)
            den = ce["MCC"] - b["MCC"]
            cells.append({
                "n_pool": n_pool, "seed": seed,
                "baseline_ACC": b["ACC"], "clef_ACC": c["ACC"], "ceiling_ACC": ce["ACC"],
                "baseline_MCC": b["MCC"], "clef_MCC": c["MCC"], "ceiling_MCC": ce["MCC"],
                "gap_MCC": c["MCC"] - b["MCC"],
                "asr_MCC": (c["MCC"] - b["MCC"]) / den if abs(den) > 1e-9 else float("nan"),
            })
            print(f"  [F] n_pool={n_pool:>5} seed={seed} | CLEF MCC={c['MCC']:.4f} "
                  f"base MCC={b['MCC']:.4f} Δ_ASR={cells[-1]['asr_MCC']:.3f}")

    agg = []
    for n_pool in pools:
        cs = [c for c in cells if c["n_pool"] == n_pool]
        row = {"n_pool": n_pool}
        for k in ["baseline_ACC", "clef_ACC", "ceiling_ACC", "baseline_MCC", "clef_MCC",
                  "ceiling_MCC", "gap_MCC", "asr_MCC"]:
            v = np.array([c[k] for c in cs], dtype=float)
            row[k + "_mean"] = float(np.nanmean(v))
            row[k + "_std"] = float(np.nanstd(v))
        agg.append(row)
    _write_csv("exp_f", agg)
    return {"pools": list(pools), "per_cell": cells, "agg": agg}


# ======================================================================================
# G · 温度敏感性
# ======================================================================================
def exp_g(taus=(0.1, 0.2, 0.5, 1.0), seeds=(42, 7, 123)) -> Dict:
    cells: List[Dict] = []
    for tau in taus:
        for seed in seeds:
            ns = _ns(tau=tau, seed=seed)
            world, splits = _splits(ns, n_train=ns.n_train, seed=seed)
            clef = _clef_rep(world, splits, tuple(MODALITY_ORDER),
                             pre_epochs=ns.pre_epochs, tau=tau, seed=seed, device=ns.device)
            c = _downstream_and_eval(clef["reps"], splits, n_train=ns.n_train, seed=seed, device=ns.device)
            cells.append({"tau": tau, "seed": seed, "ACC": c["ACC"], "MCC": c["MCC"], "F1": c["F1"],
                          "AUPRC": c["AUPRC"]})
            print(f"  [G] tau={tau:<4} seed={seed} | MCC={c['MCC']:.4f} ACC={c['ACC']:.4f}")
    agg = []
    for tau in taus:
        cs = [c for c in cells if c["tau"] == tau]
        row = {"tau": tau}
        for k in ["ACC", "MCC", "F1", "AUPRC"]:
            v = np.array([c[k] for c in cs], dtype=float)
            row[k + "_mean"] = float(v.mean())
            row[k + "_std"] = float(v.std(ddof=0))
        agg.append(row)
    _write_csv("exp_g", agg)
    return {"taus": list(taus), "per_cell": cells, "agg": agg}


# ======================================================================================
# H · 表征维度消融
# ======================================================================================
def exp_h(dims=(32, 64, 128, 256), seeds=(42, 7, 123)) -> Dict:
    cells: List[Dict] = []
    for d_z in dims:
        for seed in seeds:
            ns = _ns(d_z=d_z, seed=seed)
            world, splits = _splits(ns, n_train=ns.n_train, seed=seed)
            base_rep, _ = _frozen_baseline_rep(world, splits, seed=0)
            clef = _clef_rep(world, splits, tuple(MODALITY_ORDER),
                             pre_epochs=ns.pre_epochs, tau=ns.tau, seed=seed, device=ns.device)
            b = _downstream_and_eval(base_rep, splits, n_train=ns.n_train, seed=seed, device=ns.device)
            c = _downstream_and_eval(clef["reps"], splits, n_train=ns.n_train, seed=seed, device=ns.device)
            cells.append({"d_z": d_z, "seed": seed, "ACC": c["ACC"], "MCC": c["MCC"],
                          "baseline_MCC": b["MCC"], "params": float(clef["params"])})
            print(f"  [H] d_z={d_z:>3} seed={seed} | MCC={c['MCC']:.4f} params={clef['params']}")
    agg = []
    for d_z in dims:
        cs = [c for c in cells if c["d_z"] == d_z]
        row = {"d_z": d_z, "params": cs[0]["params"]}
        for k in ["ACC", "MCC", "baseline_MCC"]:
            v = np.array([c[k] for c in cs], dtype=float)
            row[k + "_mean"] = float(v.mean())
            row[k + "_std"] = float(v.std(ddof=0))
        agg.append(row)
    _write_csv("exp_h", agg)
    return {"dims": list(dims), "per_cell": cells, "agg": agg}


# ======================================================================================
# I · 训练标签噪声
# ======================================================================================
def exp_i(rates=(0.0, 0.05, 0.10, 0.20), seeds=(42, 7, 123)) -> Dict:
    cells: List[Dict] = []
    for seed in seeds:
        ns = _ns(seed=seed)
        world, splits = _splits(ns, n_train=ns.n_train, seed=seed)
        base_rep, _ = _frozen_baseline_rep(world, splits, seed=0)
        clef = _clef_rep(world, splits, tuple(MODALITY_ORDER),
                         pre_epochs=ns.pre_epochs, tau=ns.tau, seed=seed, device=ns.device)
        for rate in rates:
            sp = _noisy(splits, rate, seed=1000 + seed)
            b = _downstream_and_eval(base_rep, sp, n_train=ns.n_train, seed=seed, device=ns.device)
            c = _downstream_and_eval(clef["reps"], sp, n_train=ns.n_train, seed=seed, device=ns.device)
            cells.append({"rate": rate, "seed": seed,
                          "baseline_ACC": b["ACC"], "baseline_MCC": b["MCC"], "baseline_F1": b["F1"],
                          "clef_ACC": c["ACC"], "clef_MCC": c["MCC"], "clef_F1": c["F1"]})
            print(f"  [I] flip={rate:<5} seed={seed} | base MCC={b['MCC']:.4f} CLEF MCC={c['MCC']:.4f}")
    agg = []
    for rate in rates:
        cs = [c for c in cells if c["rate"] == rate]
        row = {"rate": rate}
        for k in ["baseline_ACC", "baseline_MCC", "baseline_F1", "clef_ACC", "clef_MCC", "clef_F1"]:
            v = np.array([c[k] for c in cs], dtype=float)
            row[k + "_mean"] = float(v.mean())
            row[k + "_std"] = float(v.std(ddof=0))
        agg.append(row)
    _write_csv("exp_i", agg)
    return {"rates": list(rates), "per_cell": cells, "agg": agg}


# ======================================================================================
# J · 模态缺失矩阵
# ======================================================================================
def exp_j(seeds=(42, 7)) -> Dict:
    combos: List[Dict] = []
    for m in MODALITY_ORDER:
        combos.append({"name": m, "kind": "single", "mods": (m,)})
    for m in MODALITY_ORDER:
        rest = tuple(x for x in MODALITY_ORDER if x != m)
        combos.append({"name": f"LOO-{m}", "kind": "leave-one-out", "mods": rest})
    combos.append({"name": "ALL", "kind": "full", "mods": tuple(MODALITY_ORDER)})

    cells: List[Dict] = []
    for cb in combos:
        for seed in seeds:
            ns = _ns(seed=seed)
            world, splits = _splits(ns, n_train=ns.n_train, seed=seed)
            clef = _clef_rep(world, splits, cb["mods"],
                             pre_epochs=ns.pre_epochs, tau=ns.tau, seed=seed, device=ns.device)
            c = _downstream_and_eval(clef["reps"], splits, n_train=ns.n_train, seed=seed, device=ns.device)
            cells.append({"combo": cb["name"], "kind": cb["kind"], "n_mod": len(cb["mods"]),
                          "seed": seed, "ACC": c["ACC"], "MCC": c["MCC"], "AUPRC": c["AUPRC"]})
            print(f"  [J] {cb['name']:<10} ({len(cb['mods'])}模态) seed={seed} | ACC={c['ACC']:.4f} MCC={c['MCC']:.4f}")

    agg = []
    for cb in combos:
        cs = [c for c in cells if c["combo"] == cb["name"]]
        row = {"combo": cb["name"], "kind": cb["kind"], "n_mod": len(cb["mods"])}
        for k in ["ACC", "MCC", "AUPRC"]:
            v = np.array([c[k] for c in cs], dtype=float)
            row[k + "_mean"] = float(v.mean())
            row[k + "_std"] = float(v.std(ddof=0))
        agg.append(row)
    _write_csv("exp_j", agg)
    return {"per_cell": cells, "agg": agg}


# ======================================================================================
# K · 显著性检验
# ======================================================================================
def exp_k(seeds=tuple(range(10))) -> Dict:
    pairs: List[Dict] = []
    for seed in seeds:
        ns = _ns(seed=seed)
        world, splits = _splits(ns, n_train=ns.n_train, seed=seed)
        base_rep, _ = _frozen_baseline_rep(world, splits, seed=0)
        clef = _clef_rep(world, splits, tuple(MODALITY_ORDER),
                         pre_epochs=ns.pre_epochs, tau=ns.tau, seed=seed, device=ns.device)
        b = _downstream_and_eval(base_rep, splits, n_train=ns.n_train, seed=seed, device=ns.device)
        c = _downstream_and_eval(clef["reps"], splits, n_train=ns.n_train, seed=seed, device=ns.device)
        pairs.append({"seed": seed, "baseline_MCC": b["MCC"], "clef_MCC": c["MCC"],
                      "diff_MCC": c["MCC"] - b["MCC"],
                      "baseline_ACC": b["ACC"], "clef_ACC": c["ACC"],
                      "diff_ACC": c["ACC"] - b["ACC"]})
        print(f"  [K] seed={seed:>2} | base MCC={b['MCC']:.4f} CLEF MCC={c['MCC']:.4f} "
              f"diff={pairs[-1]['diff_MCC']:+.4f}")

    def _test(diffs):
        d = np.asarray(diffs, dtype=float)
        n = d.size
        mean = float(d.mean())
        sd = float(d.std(ddof=1)) if n > 1 else 0.0
        se = sd / np.sqrt(n) if n > 0 else float("nan")
        t = mean / se if se and se > 1e-12 else float("inf")
        p = float("nan")
        try:
            from scipy import stats  # type: ignore
            p = float(stats.ttest_rel(
                [x["clef_MCC"] for x in pairs], [x["baseline_MCC"] for x in pairs]).pvalue)
        except Exception:
            try:
                from math import erf, sqrt
                # 双尾 t 的粗略正态近似（n>=10 时可用）
                p = float(2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2)))))
            except Exception:
                p = float("nan")
        rng = np.random.default_rng(20261013)
        boot = d[rng.integers(0, n, size=(20000, n))].mean(axis=1)
        lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
        return {"n": n, "mean_diff": mean, "sd": sd, "se": se, "t": float(t), "p_value": p,
                "ci95_low": lo, "ci95_high": hi}

    stat_mcc = _test([p["diff_MCC"] for p in pairs])
    stat_acc = _test([p["diff_ACC"] for p in pairs])
    print(f"  [K] MCC: mean diff={stat_mcc['mean_diff']:+.4f}, p={stat_mcc['p_value']:.3e}, "
          f"95%CI=[{stat_mcc['ci95_low']:.4f}, {stat_mcc['ci95_high']:.4f}]")
    _write_csv("exp_k", pairs)
    return {"seeds": list(seeds), "pairs": pairs, "stat_MCC": stat_mcc, "stat_ACC": stat_acc}


# ======================================================================================
# 编排
# ======================================================================================
GROUPS = {"E": exp_e, "F": exp_f, "G": exp_g, "H": exp_h, "I": exp_i, "J": exp_j, "K": exp_k}


def main() -> None:
    ap = argparse.ArgumentParser(description="CLEF 实验 E-K")
    ap.add_argument("exp", choices=list(GROUPS) + ["all"])
    args = ap.parse_args()

    payload = _load()
    todo = list(GROUPS) if args.exp == "all" else [args.exp]
    for g in todo:
        t0 = time.time()
        print(f"\n=== [{g}] 开始 ===")
        payload[g] = GROUPS[g]()
        payload[g]["elapsed_sec"] = time.time() - t0
        _dump(payload)
        print(f"=== [{g}] 完成，用时 {payload[g]['elapsed_sec']:.1f}s，已写入 {rel(SUPP_PATH)} ===")


if __name__ == "__main__":
    main()
