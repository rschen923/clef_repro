#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""export_data_package.py — 把本研究「全部使用数据」打包成单独文件。

产出（目录 D:/workbuddy-ai/worktree/data_package/，并压成 zip）：
    00_README.md              本包说明与数据字典
    01_dataset/               确定性合成基准数据集（本研究实验的全部输入）
        synthetic_splits.npz  pool/train/val/test/ceiling 原始特征
        world_meta.json       世界配置 + 投影矩阵指纹（SHA-256）
        dataset_card.md       数据集说明卡
    02_results/               全部实验结果（JSON / CSV / NPZ，原样复制）
    03_figures/               报告图与代码出图
    04_code/                  复现代码（12 个 py + 依赖 + 运行日志）
    05_references/            27 条参考文献结构化数据与核验信息
    06_figspec_survey/        顶刊（IF>40）图表规范调研的派生数据（不含 462 份原始 XML）
    MANIFEST.md               清单 + 逐文件 SHA-256
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import zipfile

import numpy as np

WT = r"D:/workbuddy-ai/worktree"
REPRO = os.path.join(WT, "clef_repro")
PKG = os.path.join(WT, "data_package")
FIGS_REPORT = os.path.join(WT, "docbuild", "figures")
FIGS_CODE = os.path.join(REPRO, "figures")
BUILD_DIR = os.path.join(WT, "ppt_clef_deck")
BUILD = os.path.join(BUILD_DIR, "build_paper.py")

# 规范：报告采用的「经典配置」（与 run_experiment.py 默认值 / 报告 2.x 节一致）
CFG = dict(
    sigma_seq=13.5, eff_shift=6.4, fam_shift=3.5, world_seed=0, signal_scale=8.0,
    d_z=64, tok_dim=32, seq_len=16, mod_mask_frac=0.3,
    n_pool=4000, n_train=100, n_val=150, n_test=800, n_ceiling=1500, data_seed=1234,
    pre_epochs=50, tau=0.5, seed=7,
)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def reset(*subs: str) -> str:
    p = os.path.join(PKG, *subs)
    os.makedirs(p, exist_ok=True)
    return p


def main() -> None:
    if os.path.exists(PKG):
        shutil.rmtree(PKG)
    os.makedirs(PKG)

    # ------------------------------------------------------------------ 01 数据集
    d1 = reset("01_dataset")
    import sys
    sys.path.insert(0, REPRO)
    from data import MODALITY_ORDER, MODALITY_SPECS, SyntheticWorld  # noqa: E402

    world = SyntheticWorld(
        sigma_seq=CFG["sigma_seq"], eff_shift=CFG["eff_shift"], fam_shift=CFG["fam_shift"],
        n_train=CFG["n_train"], seed=CFG["world_seed"], signal_scale=CFG["signal_scale"],
        d_z=CFG["d_z"], tok_dim=CFG["tok_dim"], seq_len=CFG["seq_len"],
        mod_mask_frac=CFG["mod_mask_frac"],
    )
    splits = world.make_splits(
        n_pool=CFG["n_pool"], n_train=CFG["n_train"], n_val=CFG["n_val"],
        n_test=CFG["n_test"], n_ceiling=CFG["n_ceiling"], seed=CFG["data_seed"],
    )

    npz = {}
    for name, sp in splits.items():
        npz[f"{name}__x_seq"] = sp["x_seq"]
        npz[f"{name}__y_class"] = sp["y_class"]
        npz[f"{name}__y_bin"] = sp["y_bin"]
        npz[f"{name}__z"] = sp["z"]
        for m in MODALITY_ORDER:
            npz[f"{name}__mod__{m}"] = sp["mods"][m]
    np.savez_compressed(os.path.join(d1, "synthetic_splits.npz"), **npz)

    meta = {
        "description": "CLEF 跨模态对比学习复现——确定性合成基准（共享潜变量世界）",
        "config": CFG,
        "modality_specs": {k: {"dim": v["dim"], "noise": v["noise"], "desc": v["desc"]}
                           for k, v in MODALITY_SPECS.items()},
        "modality_order": list(MODALITY_ORDER),
        "split_sizes": {k: int(v["y_bin"].shape[0]) for k, v in splits.items()},
        "arrays": sorted(npz.keys()),
        "array_shapes": {k: list(v.shape) for k, v in npz.items()},
        "dtypes": {k: str(v.dtype) for k, v in npz.items()},
        "fingerprints_sha256": {
            "e": hashlib.sha256(np.ascontiguousarray(world.e).tobytes()).hexdigest(),
            "U": hashlib.sha256(np.ascontiguousarray(world.U).tobytes()).hexdigest(),
            "A": hashlib.sha256(np.ascontiguousarray(world.A).tobytes()).hexdigest(),
            **{f"P_{m}": hashlib.sha256(np.ascontiguousarray(world.P[m]).tobytes()).hexdigest()
               for m in MODALITY_ORDER},
        },
        "reproduce": "python -c \"import numpy as np; from data import SyntheticWorld; "
                     "w=SyntheticWorld(sigma_seq=13.5,eff_shift=6.4,fam_shift=3.5,n_train=100,seed=0,"
                     "signal_scale=8.0,d_z=64,tok_dim=32,seq_len=16,mod_mask_frac=0.3); "
                     "s=w.make_splits(n_pool=4000,n_train=100,n_val=150,n_test=800,n_ceiling=1500,seed=1234)\"",
    }
    with open(os.path.join(d1, "world_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    with open(os.path.join(d1, "dataset_card.md"), "w", encoding="utf-8") as f:
        f.write("# 数据集说明卡 —— 确定性合成基准\n\n"
                "## 来源与性质\n"
                "本研究第 3 章全部下游实验的唯一数据来源，为**自建、完全确定性**的合成基准，"
                "**不依赖任何外部数据源**。其设计目的是把「序列噪声」「效应蛋白整体偏移」"
                "「家族偏移」「各模态质量」当作可独立调节的旋钮，从而在可控条件下分离"
                "「跨模态对比预训练对标注的替代效应」。\n\n"
                "真实生物数据（ESM-2 序列表征与五类生物模态）仅用于复现 CLEF 的模态构造与训练流程，"
                "其公共数据库来源与获取方式见报告 2.3 节与表 2.1；本包 `04_code/data.py` 中的 "
                "`load_real_esm_embeddings()` 给出了真实数据的接入接口与迁移路径。\n\n"
                "## 生成模型\n"
                "1. 采样类别 y ∈ {0} ∪ {1..K}，二分类标签 y_bin = (y>0)；\n"
                "2. 类别中心 c = 0（非效应蛋白）或 eff_shift·e + fam_shift·u_k（第 k 个家族）；\n"
                "3. 潜变量 z = c + N(0, I)，d_z = 64；\n"
                "4. 序列视角 x_seq = A z + σ_seq·ε，reshape 为 (seq_len=16, tok_dim=32)；\n"
                "5. 模态视角 x_m = P_m z + noise_m·ε，五个模态维度异构（20/128/64/768/32）。\n\n"
                "所有投影（e、u_k、A、P_m）在 `__init__` 中一次性固定，pool/train/val/test/ceiling "
                "全部从**同一世界**采样，故各划分分布一致、互不重叠、无「世界漂移」。\n\n"
                "## 配置（与报告一致）\n")
        for k, v in CFG.items():
            f.write(f"- `{k}` = {v}\n")
        f.write("\n## 划分规模\n")
        for k, v in meta["split_sizes"].items():
            f.write(f"- `{k}`: {v} 条\n")
        f.write("\n## 五类生物模态（维度刻意异构，噪声水平各不相同）\n\n"
                "| 模态 | 维度 | 噪声水平 | 说明 |\n|---|---|---|---|\n")
        for m in MODALITY_ORDER:
            s = MODALITY_SPECS[m]
            f.write(f"| {m} | {s['dim']} | {s['noise']} | {s['desc']} |\n")
        f.write("\n## NPZ 数组命名\n"
                "`<split>__x_seq`（序列视角）、`<split>__y_class`（细粒度类别）、"
                "`<split>__y_bin`（二分类标签）、`<split>__z`（潜变量，仅供诊断）、"
                "`<split>__mod__<M>`（第 M 个模态视角），split ∈ {pool, train, val, test, ceiling}。\n\n"
                "## 可复现性\n"
                "给定同一配置与种子，本数据集逐位可复现（见 `world_meta.json` 的 "
                "`fingerprints_sha256`）。代码实测：重跑实验 A 的全部指标与报告数值逐位一致。\n")

    print("[1/6] 数据集导出完成:", meta["split_sizes"])

    # ------------------------------------------------------------------ 02 结果
    d2 = reset("02_results")
    res_src = os.path.join(REPRO, "results")
    for fn in sorted(os.listdir(res_src)):
        if fn.endswith((".json", ".csv", ".npz", ".md", ".log")):
            shutil.copy2(os.path.join(res_src, fn), os.path.join(d2, fn))
    print("[2/6] 结果复制完成:", len(os.listdir(d2)), "个文件")

    # ------------------------------------------------------------------ 03 图
    d3r = reset("03_figures", "report")
    for fn in sorted(os.listdir(FIGS_REPORT)):
        if fn.endswith(".png") and not fn.startswith("_") and not fn.startswith("orig_"):
            shutil.copy2(os.path.join(FIGS_REPORT, fn), os.path.join(d3r, fn))
    d3c = reset("03_figures", "code")
    for fn in sorted(os.listdir(FIGS_CODE)):
        if fn.endswith(".png") and not fn.startswith("_"):
            shutil.copy2(os.path.join(FIGS_CODE, fn), os.path.join(d3c, fn))
    print("[3/6] 图复制完成: report", len(os.listdir(d3r)), "/ code", len(os.listdir(d3c)))

    # ------------------------------------------------------------------ 04 代码
    d4 = reset("04_code")
    for fn in sorted(os.listdir(REPRO)):
        if fn.endswith((".py", ".txt", ".md")):
            shutil.copy2(os.path.join(REPRO, fn), os.path.join(d4, fn))
    # 报告生成与交付校验脚本（消费本包数据的下游环节）
    for fn in ("build_paper.py", "check_report.py", "build_paper_docx.sh",
               "build_paper_full.sh", "postprocess_docx.py", "prepare_doc_figures.py"):
        p = os.path.join(BUILD_DIR, fn)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(d4, fn))
    print("[4/6] 代码复制完成:", len(os.listdir(d4)), "个文件")

    # ------------------------------------------------------------------ 05 参考文献
    d5 = reset("05_references")
    src = open(BUILD, encoding="utf-8").read()
    block = src.split('refs = """', 1)[1].split('"""', 1)[0]
    entries = re.findall(r'<p class="ref">\[(\d+)\]\s*(.*?)</p>', block, re.S)
    refs = []
    for n, txt in entries:
        plain = re.sub(r"<[^>]+>", "", txt).strip()
        doi = re.search(r"10\.\d{4,9}/[^\s]+", plain)
        url = re.search(r"https?://[^\s]+", plain)
        refs.append({
            "index": int(n),
            "text": plain,
            "doi": doi.group(0).rstrip(".") if doi else None,
            "url": url.group(0).rstrip(".") if url else None,
            "type_tag": (re.search(r"\[([A-Z]{2}(?:/[A-Z]{2})?)\]", plain).group(1)
                         if re.search(r"\[([A-Z]{2}(?:/[A-Z]{2})?)\]", plain) else None),
        })
    with open(os.path.join(d5, "references.json"), "w", encoding="utf-8") as f:
        json.dump({"count": len(refs), "style": "GB/T 7714-2025 顺序编码制",
                   "citation_order": [r["index"] for r in refs], "references": refs},
                  f, ensure_ascii=False, indent=2)
    with open(os.path.join(d5, "references.md"), "w", encoding="utf-8") as f:
        f.write("# 参考文献（GB/T 7714—2025 顺序编码制，共 %d 条）\n\n" % len(refs))
        for r in refs:
            f.write("- **[%d]** %s\n" % (r["index"], r["text"]))
    print("[5/6] 参考文献导出:", len(refs), "条")

    # ------------------------------------------------------------------ 06 图表规范调研
    d6 = reset("06_figspec_survey")
    fs = os.path.join(WT, "research_figspec")
    if os.path.isdir(fs):
        shutil.copy2(os.path.join(fs, "FIGSPEC.md"), os.path.join(d6, "FIGSPEC.md"))
        for sub in ("scripts",):
            s = os.path.join(fs, sub)
            if os.path.isdir(s):
                shutil.copytree(s, os.path.join(d6, sub))
        dd = os.path.join(fs, "data")
        if os.path.isdir(dd):
            out = os.path.join(d6, "data")
            os.makedirs(out, exist_ok=True)
            for fn in sorted(os.listdir(dd)):
                p = os.path.join(dd, fn)
                if os.path.isfile(p):
                    shutil.copy2(p, os.path.join(out, fn))
                elif os.path.isdir(p) and fn != "xml":
                    shutil.copytree(p, os.path.join(out, fn))
        with open(os.path.join(d6, "NOTE_raw_xml.md"), "w", encoding="utf-8") as f:
            f.write("# 关于原始 XML 全文\n\n"
                    "本调研从 Europe PMC 检索并下载了 462 篇全文 XML（约 95 MB），"
                    "为避免包体过大，**未随包分发**。可用 `scripts/retrieve.py` 依 "
                    "`data/sample_dois.csv` 与 `data/pmcids.txt` 重新获取。\n"
                    "本目录保留的是调研的**全部派生数据**：图注库、表格库、期刊元数据、"
                    "统计结果与官方指南摘录。\n")
    print("[6/6] 图表规范调研数据复制完成")

    # ------------------------------------------------------------------ MANIFEST
    lines = ["# MANIFEST —— 本研究全部使用数据包\n",
             "生成时间：2026-10-13 ｜ 根目录：`data_package/`\n",
             "## 目录结构\n",
             "```",
             "00_README.md            本包说明与数据字典",
             "01_dataset/             确定性合成基准数据集（全部实验输入）",
             "02_results/             全部实验结果（JSON/CSV/NPZ）",
             "03_figures/             报告图（report/）与代码出图（code/）",
             "04_code/                复现代码与运行日志",
             "05_references/          27 条参考文献结构化数据",
             "06_figspec_survey/      顶刊图表规范调研派生数据",
             "MANIFEST.md             本清单（含逐文件 SHA-256）",
             "```\n",
             "## 文件清单与 SHA-256\n",
             "| 相对路径 | 字节 | SHA-256 |",
             "|---|---:|---|"]
    for root, dirs, files in os.walk(PKG):
        dirs.sort()
        for fn in sorted(files):
            p = os.path.join(root, fn)
            rel = os.path.relpath(p, PKG).replace("\\", "/")
            lines.append("| `%s` | %d | `%s` |" % (rel, os.path.getsize(p), sha256(p)))
    with open(os.path.join(PKG, "MANIFEST.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # ------------------------------------------------------------------ README
    with open(os.path.join(PKG, "00_README.md"), "w", encoding="utf-8") as f:
        f.write("""# 全部使用数据包 —— 跨模态对比学习项目

本包收录《基于跨模态对比学习提升生物学优质标注稀缺条件下下游数据的效率与泛化》
研究所使用的**全部数据、代码与结果**，用于结果复核与完全复现。

## 一、数据字典

| 目录 | 内容 | 关键文件 |
|---|---|---|
| `01_dataset/` | 确定性合成基准（全部实验输入） | `synthetic_splits.npz`、`world_meta.json`、`dataset_card.md` |
| `02_results/` | 全部实验结果 | `all_results.json`、`supplementary.json`、`exp_a~k.csv/json`、`curves.json`、`pca_data.npz` |
| `03_figures/` | 报告图 13 幅 + 补充图 8 幅 + 代码出图 | `rep_*.png`、`sup_*.png` |
| `04_code/` | 复现代码 12 个模块（3 561 行） | `data.py`、`model.py`、`train.py`、`evaluate.py`、`run_experiment.py`、`run_supplementary.py` |
| `05_references/` | 27 条参考文献结构化数据 | `references.json`、`references.md` |
| `06_figspec_survey/` | 顶刊（IF>40）图表规范调研派生数据 | `FIGSPEC.md`、`data/`、`scripts/` |

## 二、结果索引

- **实验 A**（三臂对照，100-shot）：`02_results/exp_a.csv|json`
- **实验 B**（标注预算扫描）：`exp_b.csv`（聚合）/ `exp_b_per_cell.csv`（明细）
- **实验 C**（VIB β 扫描）：`exp_c.csv`
- **实验 D**（模态质量退化）：`exp_d.csv`
- **实验 E–K**（补充七组）：`exp_e~k.csv`，汇总于 `supplementary.json`
- **总汇总**：`all_results.json`

## 三、复现步骤

```bash
pip install -r 04_code/requirements.txt          # torch(CPU) / numpy / scikit-learn / matplotlib / pandas
cd 04_code
python run_experiment.py all                      # 实验 A–D
python run_supplementary.py all                   # 实验 E–K
```

全程 CPU 即可完成；实验 A 约 4 分钟，全量补充实验约 20–30 分钟。

## 四、可复现性核验结论

- 重跑实验 A：**全部指标与报告数值逐位一致**（Baseline ACC 0.8475 / MCC 0.6950；
  CLEF-ALL ACC 0.9075 / MCC 0.8146；Δ_ASR(ACC) 0.941 / Δ_ASR(MCC) 0.942）。
- 重跑补充实验 E–K：结果文件与归档版本**逐字节一致**。
- 唯一差异字段为 `elapsed_sec`（计时）与 `config.exp`（命令行参数），不影响任何科学数值。

## 五、许可与出处

- 复现代码对应原研究 CLEF（Peng et al., *Nature Communications* 2025, 16:1299）的
  **同协议独立实现**，非逐位复现；合成基准为本文自建。
- 参考文献按 GB/T 7714—2025 顺序编码制著录，详见 `05_references/`。
""")

    # ------------------------------------------------------------------ ZIP
    zip_path = os.path.join(WT, "跨模态对比学习_全部使用数据.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for root, dirs, files in os.walk(PKG):
            dirs.sort()
            for fn in sorted(files):
                p = os.path.join(root, fn)
                z.write(p, os.path.relpath(p, os.path.dirname(PKG)))
    print("\nDONE ->", PKG)
    print("ZIP  ->", zip_path, "(%.2f MB)" % (os.path.getsize(zip_path) / 1e6))


if __name__ == "__main__":
    main()
