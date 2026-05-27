"""
run_experiments.py
==================
Master script that runs ALL 3 experiment groups from the project plan:

  Exp 1 – Single-source baseline (train on 1 zone, test on all 3)
  Exp 2 – Multi-source baseline  (train on 2 zones, test on 1)
  Exp 3 – DG-Aug proposed method (train on 2 zones + aug, test on 1)

Then generates:
  - results/tables/summary_table.csv   (paper-style result table)
  - results/plots/map_id_vs_ood.png
  - results/plots/performance_drop.png

Usage:
    python run_experiments.py \
        --data_dir data \
        --model yolov8s.pt \
        --epochs 80 \
        --batch 16 \
        --device 0

    # Quick smoke-test (small GPU)
    python run_experiments.py \
        --data_dir data --model yolov8n.pt \
        --epochs 5 --imgsz 512 --batch 8 --device cpu
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# ─── Experiment definitions ──────────────────────────────────────────────────

ZONES = ["CZ_A", "CZ_B", "CZ_C"]

EXP1_RUNS = [
    # (train_zones, test_zone, run_name)
    (["CZ_A"], "CZ_A", "exp1_baseline_CZA"),
    (["CZ_B"], "CZ_B", "exp1_baseline_CZB"),
    (["CZ_C"], "CZ_C", "exp1_baseline_CZC"),
]

EXP2_RUNS = [
    (["CZ_A", "CZ_B"], "CZ_C", "exp2_multisrc_test_CZC"),
    (["CZ_A", "CZ_C"], "CZ_B", "exp2_multisrc_test_CZB"),
    (["CZ_B", "CZ_C"], "CZ_A", "exp2_multisrc_test_CZA"),
]

EXP3_RUNS = [
    (["CZ_A", "CZ_B"], "CZ_C", "exp3_dgaug_test_CZC"),
    (["CZ_A", "CZ_C"], "CZ_B", "exp3_dgaug_test_CZB"),
    (["CZ_B", "CZ_C"], "CZ_A", "exp3_dgaug_test_CZA"),
]


def get_yaml_for(src_zones: list, test_zone: str, cfg_dir: Path) -> Path:
    """Locate the YAML written by split_domain.py."""
    if len(src_zones) == 1:
        return cfg_dir / f"single_source_{src_zones[0].lower()}.yaml"
    else:
        return cfg_dir / f"multi_source_test_{test_zone.lower()}.yaml"


def run_cmd(cmd: list, desc: str = ""):
    """Run a subprocess command, streaming output."""
    print(f"\n{'─'*60}")
    print(f"  ▶ {desc}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'─'*60}")
    result = subprocess.run(cmd, check=True)
    return result


def train_and_eval(src_zones: list,
                   test_zone: str,
                   run_name: str,
                   use_dg_aug: bool,
                   args,
                   cfg_dir: Path) -> dict | None:
    """Train one model, evaluate cross-domain, return summary dict."""
    yaml_path = get_yaml_for(src_zones, test_zone, cfg_dir)
    if not yaml_path.exists():
        print(f"[SKIP] YAML not found: {yaml_path}")
        return None

    # After training, results are reorganized to: {project}/{name}/weights/best.pt
    weights_path = (Path(args.run_dir) / run_name / "weights" / "best.pt")

    # ─── Train (skip if weights already exist) ────────────────────────────
    if weights_path.exists() and not args.force_retrain:
        print(f"[SKIP training] {run_name} – weights already exist")
    else:
        train_script = (
            "scripts/train_dg_aug.py" if use_dg_aug
            else "scripts/train_baseline.py"
        )
        run_cmd(
            [sys.executable, train_script,
             "--cfg",      str(yaml_path),
             "--run_name", run_name,
             "--model",    args.model,
             "--imgsz",    str(args.imgsz),
             "--epochs",   str(args.epochs),
             "--batch",    str(args.batch),
             "--device",   args.device,
             "--project",  args.run_dir],
            desc=f"Training {run_name}"
        )
        
        # Verify weights exist after training
        if not weights_path.exists():
            # Check if they might be in the nested YOLO directory
            nested_weights = None
            for candidate in [
                Path(args.run_dir) / "detect" / run_name / "weights" / "best.pt",
                Path(args.run_dir) / "detect" / "runs" / run_name / "weights" / "best.pt",
                Path(args.run_dir) / "detect" / "results" / run_name / "weights" / "best.pt",
            ]:
                if candidate.exists():
                    nested_weights = candidate
                    break
            if nested_weights is not None:
                print(f"  → Found weights in nested YOLO structure, reorganizing...")
                import shutil
                final_output_dir = Path(args.run_dir) / run_name
                if final_output_dir.exists():
                    shutil.rmtree(final_output_dir)
                shutil.move(str(nested_weights.parent.parent), str(final_output_dir))
                print(f"  → Reorganized successfully")
            else:
                print(f"[ERROR] Training failed – weights not found at {weights_path}")
                return None

    # ─── Evaluate ─────────────────────────────────────────────────────────
    ood_zones = [z for z in ZONES if z not in src_zones]
    id_zones  = src_zones

    run_cmd(
        [sys.executable, "scripts/evaluate_domains.py",
         "--weights",   str(weights_path),
         "--data_dir",  args.data_dir,
         "--run_name",  run_name,
         "--id_zones",  *id_zones,
         "--ood_zones", *ood_zones,
         "--imgsz",     str(args.imgsz),
         "--batch",     str(args.batch),
         "--device",    args.device,
         "--out_dir",   "results/tables"],
        desc=f"Evaluating {run_name}"
    )

    # Load result JSON
    json_path = Path("results/tables") / f"{run_name}_results.json"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    return None


# ─── Table generation ────────────────────────────────────────────────────────

def build_summary_table(all_results: list) -> pd.DataFrame:
    """Build paper-style summary table from list of result dicts."""
    rows = []
    for res in all_results:
        if res is None:
            continue
        agg = res.get("aggregate", {})
        row = {
            "Method"    : res["run_name"],
            "Train Domain" : "+".join(res.get("id_zones", [])),
            "Test (OOD)": "+".join(res.get("ood_zones", [])),
            "ID mAP50"  : round(agg.get("mAP50_ID",    0), 4),
            "OOD mAP50" : round(agg.get("mAP50_OOD",   0), 4),
            "ID mAP5095": round(agg.get("mAP5095_ID",  0), 4),
            "OOD mAP5095":round(agg.get("mAP5095_OOD", 0), 4),
            "PD (%) ↓"  : round(agg.get("PD_5095",     0), 2),
            "H-score ↑" : round(agg.get("H_5095",      0), 4),
        }
        # Per-zone columns
        for zone in ZONES:
            row[f"mAP50_{zone}"] = round(
                res.get("per_zone", {}).get(zone, {}).get("mAP50", 0), 4)
        rows.append(row)
    return pd.DataFrame(rows)


# ─── Plotting ────────────────────────────────────────────────────────────────

def plot_results(df: pd.DataFrame, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[SKIP] matplotlib not installed – skipping plots")
        return

    if df.empty or "Method" not in df.columns:
        print("[WARN] Skipping plots: Results DataFrame is empty or missing 'Method' column.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. ID vs OOD mAP50 scatter plot ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"exp1": "#e74c3c", "exp2": "#3498db", "exp3": "#2ecc71"}
    labels = {"exp1": "Single-source Baseline",
              "exp2": "Multi-source Baseline",
              "exp3": "DG-Aug (Proposed)"}
    for prefix, color in colors.items():
        sub = df[df["Method"].str.startswith(prefix)]
        if sub.empty:
            continue
        ax.scatter(sub["ID mAP50"], sub["OOD mAP50"],
                   color=color, label=labels[prefix],
                   s=100, zorder=3, edgecolors="white", linewidths=0.8)
        for _, row in sub.iterrows():
            ax.annotate(row["Test (OOD)"],
                        (row["ID mAP50"], row["OOD mAP50"]),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=7)

    # Diagonal reference (ID = OOD)
    lim = max(df["ID mAP50"].max(), df["OOD mAP50"].max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, linewidth=1,
            label="ID = OOD (ideal)")
    ax.set_xlabel("ID mAP50", fontsize=12)
    ax.set_ylabel("OOD mAP50", fontsize=12)
    ax.set_title("ID vs OOD mAP50 across Experiments", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path1 = out_dir / "map_id_vs_ood.png"
    fig.savefig(path1, dpi=150)
    plt.close()
    print(f"  Saved → {path1}")

    # ── 2. Performance Drop bar chart ─────────────────────────────────────
    methods_map = {
        "exp1": "Single-src\nBaseline",
        "exp2": "Multi-src\nBaseline",
        "exp3": "DG-Aug\n(Proposed)",
    }
    pd_avgs = {}
    for prefix, label in methods_map.items():
        sub = df[df["Method"].str.startswith(prefix)]
        pd_avgs[label] = sub["PD (%) ↓"].mean() if not sub.empty else 0.0

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(pd_avgs.keys(), pd_avgs.values(),
                  color=["#e74c3c", "#3498db", "#2ecc71"],
                  edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars, pd_avgs.values()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Avg. Performance Drop (%)", fontsize=12)
    ax.set_title("Performance Drop: ID → OOD", fontsize=13)
    ax.set_ylim(0, max(pd_avgs.values()) * 1.25 + 5)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path2 = out_dir / "performance_drop.png"
    fig.savefig(path2, dpi=150)
    plt.close()
    print(f"  Saved → {path2}")

    # ── 3. Per-zone mAP50 grouped bar ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(ZONES))
    width = 0.25
    for i, (prefix, label) in enumerate(methods_map.items()):
        sub = df[df["Method"].str.startswith(prefix)]
        vals = []
        for zone in ZONES:
            col = f"mAP50_{zone}"
            vals.append(sub[col].mean() if col in sub and not sub.empty else 0)
        offset = (i - 1) * width
        rects = ax.bar(x + offset, vals, width,
                       label=label,
                       color=list(colors.values())[i],
                       edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(ZONES, fontsize=12)
    ax.set_ylabel("mAP50", fontsize=12)
    ax.set_title("Per-zone mAP50 by Method", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path3 = out_dir / "perzone_map50.png"
    fig.savefig(path3, dpi=150)
    plt.close()
    print(f"  Saved → {path3}")


# ─────────────────────────────── main ────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run all RWDS-CZ experiments end-to-end")
    parser.add_argument("--data_dir",     default="data")
    parser.add_argument("--cfg_dir",      default="configs")
    parser.add_argument("--run_dir",      default="runs",
                        help="Base directory for training runs (YOLO creates: run_dir/detect/results/)")
    parser.add_argument("--model",        default="yolov8s.pt",
                        help="YOLO model file or name (e.g., yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt, yolo11n.pt, etc.)")
    parser.add_argument("--imgsz",        type=int, default=640)
    parser.add_argument("--epochs",       type=int, default=80)
    parser.add_argument("--batch",        type=int, default=16)
    parser.add_argument("--device",       default="")
    parser.add_argument("--exp",          nargs="+",
                        choices=["1", "2", "3"],
                        default=["1", "2", "3"],
                        help="Which experiments to run (default: all)")
    parser.add_argument("--force_retrain", action="store_true",
                        help="Re-train even if weights exist")
    args = parser.parse_args()

    cfg_dir = Path(args.cfg_dir)
    all_results = []

    print("\n" + "═" * 60)
    print("  RWDS-CZ: Full Experiment Suite")
    print("═" * 60)

    # ── Experiment 1: Single-source baseline ──────────────────────────────
    if "1" in args.exp:
        print("\n▶ EXPERIMENT 1: Single-source Baseline")
        for src_zones, test_zone, run_name in EXP1_RUNS:
            res = train_and_eval(
                src_zones, test_zone, run_name,
                use_dg_aug=False, args=args, cfg_dir=cfg_dir)
            all_results.append(res)

    # ── Experiment 2: Multi-source baseline ───────────────────────────────
    if "2" in args.exp:
        print("\n▶ EXPERIMENT 2: Multi-source Baseline")
        for src_zones, test_zone, run_name in EXP2_RUNS:
            res = train_and_eval(
                src_zones, test_zone, run_name,
                use_dg_aug=False, args=args, cfg_dir=cfg_dir)
            all_results.append(res)

    # ── Experiment 3: Proposed DG-Aug method ─────────────────────────────
    if "3" in args.exp:
        print("\n▶ EXPERIMENT 3: DG-Aug (Proposed Method)")
        for src_zones, test_zone, run_name in EXP3_RUNS:
            res = train_and_eval(
                src_zones, test_zone, run_name,
                use_dg_aug=True, args=args, cfg_dir=cfg_dir)
            all_results.append(res)

    # ── Build summary table ───────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  Building Summary Table")
    print("═" * 60)
    df = build_summary_table([r for r in all_results if r])
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    csv_path = Path("results/tables/summary_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Summary table → {csv_path}")
    print(df.to_string(index=False))

    # ── Generate plots ────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  Generating Plots")
    print("═" * 60)
    if not df.empty and "Method" in df.columns:
        plot_results(df, Path("results/plots"))
    else:
        print("[WARN] No results gathered or invalid data. Skipping plots.")

    print("\n✓ All experiments complete!")
    print(f"  Tables  : results/tables/")
    print(f"  Plots   : results/plots/")
    print(f"  Weights : {args.run_dir}/")


if __name__ == "__main__":
    main()
