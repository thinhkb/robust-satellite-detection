#!/usr/bin/env python3
"""
Training Results Summary
==============================
Prints a comprehensive, well-formatted summary of all training experiment results
for inclusion in reports. Reads from the official results JSON files and CSV summaries.
"""

import json
import os
import csv
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "official"
RUNS_DIR = PROJECT_ROOT / "runs"

BOLD = "\033[1m"
DIM = "\033[2m"
UNDERLINE = "\033[4m"
RESET = "\033[0m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
WHITE = "\033[97m"
BLUE = "\033[94m"

LINE_W = 100


def hline(char="-"):
    print(char * LINE_W)


def section(title):
    print()
    hline("=")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    hline("=")


def subsection(title):
    print()
    print(f"  {BOLD}{YELLOW}> {title}{RESET}")
    hline("-")


def pct(v):
    return f"{v * 100:.2f}%"


def signed_pct(v):
    if v > 0:
        return f"{RED}{v:+.2f}%{RESET}"
    elif v < 0:
        return f"{GREEN}{v:+.2f}%{RESET}"
    else:
        return f"{v:+.2f}%"


def load_json_results():
    results = {}
    for f in sorted(RESULTS_DIR.glob("*_results.json")):
        with open(f) as fh:
            data = json.load(fh)
            results[data["run_name"]] = data
    return results


def get_training_info(run_name):
    args_path = RUNS_DIR / run_name / "args.yaml"
    if not args_path.exists():
        base_name = run_name.split("_seed")[0]
        args_path = RUNS_DIR / base_name / "args.yaml"
        if not args_path.exists():
            return {}
    info = {}
    with open(args_path) as fh:
        for line in fh:
            line = line.strip()
            if ":" in line:
                key, _, val = line.partition(":")
                info[key.strip()] = val.strip()
    return info


def get_epoch_count(run_name):
    csv_path = RUNS_DIR / run_name / "results.csv"
    if not csv_path.exists():
        return None
    count = 0
    with open(csv_path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            count += 1
    return count


METHOD_ORDER = ["Single-source", "Multi-source", "DG-Aug", "ACS-YOLO"]
METHOD_DESCRIPTIONS = {
    "Single-source": "Baseline - Train on one domain, test on others",
    "Multi-source": "Train on two domains, test on the held-out domain",
    "DG-Aug":       "Multi-source + Domain-Generalization Augmentation",
    "ACS-YOLO":     "Adaptive Channel Selection YOLO (proposed method)",
}
ZONE_NAMES = ["CZ_A", "CZ_B", "CZ_C"]
CLASS_NAMES = [
    "Building", "Small Car", "Truck", "Bus",
    "Cargo Truck", "Shipping Container", "Vehicle Lot", "Shed"
]


def main():
    results = load_json_results()
    if not results:
        print(f"{RED}No result files found in {RESULTS_DIR}{RESET}")
        return

    # Header
    print()
    hline("=")
    print(f"{BOLD}{CYAN}")
    print("   +==============================================================+")
    print("   |         ROBUST SATELLITE OBJECT DETECTION - RESULTS           |")
    print("   |       Domain Generalization Benchmark Training Summary        |")
    print("   +===============================================================+")
    print(f"{RESET}")
    print(f"  {DIM}Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}")
    print(f"  {DIM}Results dir: {RESULTS_DIR}{RESET}")
    hline("=")

    # 1. Environment
    section("1. TRAINING ENVIRONMENT & CONFIGURATION")

    sample = list(results.values())[0]
    env = sample.get("environment", {})

    print(f"  {'Python':<25} {env.get('python', 'N/A')}")
    print(f"  {'Platform':<25} {env.get('platform', 'N/A')}")
    print(f"  {'PyTorch':<25} {env.get('torch', 'N/A')}")
    print(f"  {'Ultralytics':<25} {env.get('ultralytics', 'N/A')}")

    args = get_training_info("exp1_baseline_CZA_seed42")
    if args:
        subsection("Hyperparameters")
        hp_keys = [
            ("model", "Base Model"),
            ("imgsz", "Image Size"),
            ("batch", "Batch Size"),
            ("epochs", "Max Epochs"),
            ("patience", "Early Stopping Patience"),
            ("optimizer", "Optimizer"),
            ("lr0", "Initial LR"),
            ("lrf", "Final LR Factor"),
            ("momentum", "Momentum"),
            ("weight_decay", "Weight Decay"),
            ("warmup_epochs", "Warmup Epochs"),
            ("seed", "Random Seed"),
            ("amp", "Mixed Precision (AMP)"),
            ("device", "GPU Device"),
        ]
        for key, label in hp_keys:
            val = args.get(key, "N/A")
            print(f"  {label:<25} {val}")

    subsection("Detection Classes (8 classes)")
    for i, cls in enumerate(CLASS_NAMES):
        print(f"  {i}: {cls}")

    # 2. Experiment Overview
    section("2. EXPERIMENT OVERVIEW")
    print(f"  {'#':<4} {'Experiment':<35} {'Method':<15} {'Train':<12} {'Test (OOD)':<12} {'Epochs'}")
    hline("-")

    grouped = {}
    for i, (name, data) in enumerate(sorted(results.items()), 1):
        method = data["method"]
        id_z = "+".join(data["id_zones"])
        ood_z = "+".join(data["ood_zones"])
        epoch_count = get_epoch_count(name)
        epoch_str = str(epoch_count) if epoch_count else "N/A"
        print(f"  {i:<4} {name:<35} {method:<15} {id_z:<12} {ood_z:<12} {epoch_str}")
        if method not in grouped:
            grouped[method] = []
        grouped[method].append(data)

    # 3. Main Results Table (mAP@[50])
    section("3. MAIN RESULTS - mAP@50")
    header = f"  {'Method':<15} {'OOD Target':<10} {'ID mAP50':>10} {'OOD mAP50':>10} {'PD(down)':>10} {'H-mean':>10} | {'CZ_A':>8} {'CZ_B':>8} {'CZ_C':>8}"
    print(header)
    hline("-")

    for method in METHOD_ORDER:
        if method not in grouped:
            continue
        for data in grouped[method]:
            agg = data["aggregate"]
            ood_target = "+".join(data["ood_zones"])
            pz = data["per_zone"]

            pd_val = agg["PD_50"]
            pd_str = signed_pct(pd_val)

            cza = pct(pz["CZ_A"]["mAP50"])
            czb = pct(pz["CZ_B"]["mAP50"])
            czc = pct(pz["CZ_C"]["mAP50"])

            print(f"  {method:<15} {ood_target:<10} {pct(agg['mAP50_ID']):>10} {pct(agg['mAP50_OOD']):>10} {pd_str:>20} {pct(agg['H_50']):>10} | {cza:>8} {czb:>8} {czc:>8}")
        print(f"  {'.'*96}")

    # 4. Main Results Table (mAP@[50-95])
    section("4. MAIN RESULTS - mAP@50-95")
    header = f"  {'Method':<15} {'OOD Target':<10} {'ID':>10} {'OOD':>10} {'PD(down)':>10} {'H-mean':>10} | {'CZ_A':>8} {'CZ_B':>8} {'CZ_C':>8}"
    print(header)
    hline("-")

    for method in METHOD_ORDER:
        if method not in grouped:
            continue
        for data in grouped[method]:
            agg = data["aggregate"]
            ood_target = "+".join(data["ood_zones"])
            pz = data["per_zone"]

            pd_val = agg["PD_5095"]
            pd_str = signed_pct(pd_val)

            cza = pct(pz["CZ_A"]["mAP50_95"])
            czb = pct(pz["CZ_B"]["mAP50_95"])
            czc = pct(pz["CZ_C"]["mAP50_95"])

            print(f"  {method:<15} {ood_target:<10} {pct(agg['mAP5095_ID']):>10} {pct(agg['mAP5095_OOD']):>10} {pd_str:>20} {pct(agg['H_5095']):>10} | {cza:>8} {czb:>8} {czc:>8}")
        print(f"  {'.'*96}")

    # 5. Method-wise Averages
    section("5. METHOD-WISE AVERAGES (across OOD targets)")
    print(f"  {'':>15} {'-- mAP@50 --':>42} | {'-- mAP@[50-95] --':>42}")
    header = f"  {'Method':<15} {'Avg ID':>10} {'Avg OOD':>10} {'Avg PD':>10} {'Avg H':>10} | {'Avg ID':>10} {'Avg OOD':>10} {'Avg PD':>10} {'Avg H':>10}"
    print(header)
    hline("-")

    for method in METHOD_ORDER:
        if method not in grouped:
            continue
        runs = grouped[method]
        n = len(runs)

        avg_id_50 = sum(d["aggregate"]["mAP50_ID"] for d in runs) / n
        avg_ood_50 = sum(d["aggregate"]["mAP50_OOD"] for d in runs) / n
        avg_pd_50 = sum(d["aggregate"]["PD_50"] for d in runs) / n
        avg_h_50 = sum(d["aggregate"]["H_50"] for d in runs) / n

        avg_id_95 = sum(d["aggregate"]["mAP5095_ID"] for d in runs) / n
        avg_ood_95 = sum(d["aggregate"]["mAP5095_OOD"] for d in runs) / n
        avg_pd_95 = sum(d["aggregate"]["PD_5095"] for d in runs) / n
        avg_h_95 = sum(d["aggregate"]["H_5095"] for d in runs) / n

        pd50_str = signed_pct(avg_pd_50)
        pd95_str = signed_pct(avg_pd_95)

        print(f"  {method:<15} {pct(avg_id_50):>10} {pct(avg_ood_50):>10} {pd50_str:>20} {pct(avg_h_50):>10} | {pct(avg_id_95):>10} {pct(avg_ood_95):>10} {pd95_str:>20} {pct(avg_h_95):>10}")

    # 6. Per-class breakdown
    section("6. PER-CLASS BREAKDOWN (mAP@50-95)")

    for method in METHOD_ORDER:
        if method not in grouped:
            continue
        subsection(f"{method}")
        cls_header = f"  {'Zone':<8}"
        for cls in CLASS_NAMES:
            cls_header += f" {cls[:10]:>12}"
        print(cls_header)
        hline("-")

        for data in grouped[method]:
            for zone in ZONE_NAMES:
                pz = data["per_zone"][zone]
                per_cls = pz.get("per_class_mAP50_95", {})
                is_ood = zone in data["ood_zones"]
                zone_label = f"{zone}{'*' if is_ood else ' '}"
                row = f"  {zone_label:<8}"
                for cls in CLASS_NAMES:
                    val = per_cls.get(cls, 0.0)
                    row += f" {pct(val):>12}"
                print(row)
            print(f"  {DIM}(* = OOD zone){RESET}")
            print()

    # 7. Improvement Analysis
    section("7. IMPROVEMENT ANALYSIS vs BASELINES")

    def delta_str(v):
        color = GREEN if v > 0 else RED if v < 0 else ""
        return f"{color}{v * 100:+.2f}%{RESET}"

    for method in ["Multi-source", "DG-Aug", "ACS-YOLO"]:
        if method not in grouped:
            continue
        subsection(f"{method} vs Single-source Baseline")
        print(f"  {'OOD Target':<12} {'d OOD mAP50':>14} {'d OOD mAP50-95':>16} {'d H-mean@50':>14} {'d H-mean@50-95':>16}")
        hline("-")

        for data in grouped[method]:
            ood_target = "+".join(data["ood_zones"])

            ood_zone = data["ood_zones"][0] if len(data["ood_zones"]) == 1 else None
            if ood_zone:
                best_baseline_ood = None
                for bdata in grouped.get("Single-source", []):
                    if ood_zone in bdata["ood_zones"]:
                        b_ood_map50 = bdata["per_zone"][ood_zone]["mAP50"]
                        b_ood_map95 = bdata["per_zone"][ood_zone]["mAP50_95"]
                        if best_baseline_ood is None or b_ood_map50 > best_baseline_ood[0]:
                            best_baseline_ood = (b_ood_map50, b_ood_map95,
                                                  bdata["aggregate"]["H_50"],
                                                  bdata["aggregate"]["H_5095"])

                if best_baseline_ood:
                    d_ood_50 = data["per_zone"][ood_zone]["mAP50"] - best_baseline_ood[0]
                    d_ood_95 = data["per_zone"][ood_zone]["mAP50_95"] - best_baseline_ood[1]
                    d_h_50 = data["aggregate"]["H_50"] - best_baseline_ood[2]
                    d_h_95 = data["aggregate"]["H_5095"] - best_baseline_ood[3]

                    print(f"  {ood_target:<12} {delta_str(d_ood_50):>24} {delta_str(d_ood_95):>26} {delta_str(d_h_50):>24} {delta_str(d_h_95):>26}")

    # 8. Cross-domain Performance Matrix
    section("8. CROSS-DOMAIR mAP@50 MATRIX")
    print(f"  {DIM}Rows = Method + Train config | Columns = Test zone{RESET}")
    print()
    print(f"  {'Method':<15} {'Train':<12} {'-> CZ_A':>10} {'-> CZ_B':>10} {'-> CZ_C':>10} {'Avg':>10}")
    hline("-")

    for method in METHOD_ORDER:
        if method not in grouped:
            continue
        for data in grouped[method]:
            id_z = "+".join(data["id_zones"])
            pz = data["per_zone"]
            vals = [pz[z]["mAP50"] for z in ZONE_NAMES]
            avg_val = sum(vals) / len(vals)

            row = f"  {method:<15} {id_z:<12}"
            for z in ZONE_NAMES:
                v = pz[z]["mAP50"]
                is_id = z in data["id_zones"]
                marker = f"{BOLD}" if is_id else ""
                end_marker = f"{RESET}" if is_id else ""
                row += f" {marker}{pct(v):>10}{end_marker}"
            row += f" {DIM}{pct(avg_val):>10}{RESET}"
            print(row)
        print(f"  {'.'*96}")

    # 9. Summary statistics
    section("9. KEY FINDINGS SUMMARY")

    for method in METHOD_ORDER:
        if method not in grouped:
            continue
        runs = grouped[method]
        best_ood = max(runs, key=lambda d: d["aggregate"]["mAP50_OOD"])
        worst_ood = min(runs, key=lambda d: d["aggregate"]["mAP50_OOD"])
        avg_ood = sum(d["aggregate"]["mAP50_OOD"] for d in runs) / len(runs)
        avg_pd = sum(d["aggregate"]["PD_50"] for d in runs) / len(runs)

        print(f"  {BOLD}{method}{RESET}: {METHOD_DESCRIPTIONS.get(method, '')}")
        print(f"    Avg OOD mAP@50: {pct(avg_ood)}  |  Avg PD: {signed_pct(avg_pd)}")
        print(f"    Best:  {'+'.join(best_ood['ood_zones']):<8} OOD mAP@50 = {pct(best_ood['aggregate']['mAP50_OOD'])}")
        print(f"    Worst: {'+'.join(worst_ood['ood_zones']):<8} OOD mAP@[50] = {pct(worst_ood['aggregate']['mAP50_OOD'])}")
        print()

    all_runs = list(results.values())
    best_overall = max(all_runs, key=lambda d: d["aggregate"]["H_50"])
    print(f"  {GREEN}{BOLD}Best Harmonic Mean (mAP@50):{RESET}")
    print(f"    {best_overall['run_name']} -- H = {pct(best_overall['aggregate']['H_50'])}")

    best_ood_overall = max(all_runs, key=lambda d: d["aggregate"]["mAP50_OOD"])
    print(f"  {GREEN}{BOLD}Best OOD mAP@50:{RESET}")
    print(f"    {best_ood_overall['run_name']} -- OOD mAP@[50] = {pct(best_ood_overall['aggregate']['mAP50_OOD'])}")

    lowest_pd = min(all_runs, key=lambda d: abs(d["aggregate"]["PD_50"]))
    print(f"  {GREEN}{BOLD}Lowest |Performance Degradation| @50:{RESET}")
    print(f"    {lowest_pd['run_name']} -- PD = {signed_pct(lowest_pd['aggregate']['PD_50'])}")

    print()
    hline("=")
    print(f"{BOLD}{CYAN}  Total experiments: {len(results)} | Methods: {len(METHOD_ORDER)} | Zones: {len(ZONE_NAMES)} | Classes: {len(CLASS_NAMES)}{RESET}")
    hline("=")
    print()


if __name__ == "__main__":
    main()
