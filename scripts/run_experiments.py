"""
Run the complete RWDS-CZ experiment suite with reproducible multi-seed reports.

Experiments:
  1. Single-source baseline
  2. Multi-source baseline
  3. Multi-source DG-Aug
  4. Multi-source ACS-YOLO

Official report example:
    python scripts/run_experiments.py --exp 1 2 3 4 --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ZONES = ["CZ_A", "CZ_B", "CZ_C"]


@dataclass(frozen=True)
class Experiment:
    group: str
    method: str
    train_zones: tuple[str, ...]
    test_zone: str
    base_run: str
    train_script: str


EXPERIMENTS = [
    *[
        Experiment(
            "1",
            "Single-source",
            (zone,),
            zone,
            f"exp1_baseline_{zone.replace('_', '')}",
            "scripts/train_baseline.py",
        )
        for zone in ZONES
    ],
    *[
        Experiment(
            "2",
            "Multi-source",
            tuple(zone for zone in ZONES if zone != test_zone),
            test_zone,
            f"exp2_multisrc_test_{test_zone.replace('_', '')}",
            "scripts/train_baseline.py",
        )
        for test_zone in ZONES
    ],
    *[
        Experiment(
            "3",
            "DG-Aug",
            tuple(zone for zone in ZONES if zone != test_zone),
            test_zone,
            f"exp3_dgaug_test_{test_zone.replace('_', '')}",
            "scripts/train_dg_aug.py",
        )
        for test_zone in ZONES
    ],
    *[
        Experiment(
            "4",
            "ACS-YOLO",
            tuple(zone for zone in ZONES if zone != test_zone),
            test_zone,
            f"exp4_acsyolo_test_{test_zone.replace('_', '')}",
            "scripts/train_acs_yolo.py",
        )
        for test_zone in ZONES
    ],
]


def yaml_for(experiment: Experiment, cfg_dir: Path) -> Path:
    if len(experiment.train_zones) == 1:
        return cfg_dir / f"single_source_{experiment.train_zones[0].lower()}.yaml"
    return cfg_dir / f"multi_source_test_{experiment.test_zone.lower()}.yaml"


def run_command(command: list[str], description: str) -> None:
    print(f"\n{'=' * 72}\n{description}\n{' '.join(command)}\n{'=' * 72}")
    subprocess.run(command, check=True)


def find_weights(run_dir: Path, run_name: str) -> Path | None:
    candidates = [
        run_dir / run_name / "weights" / "best.pt",
        run_dir / "detect" / run_name / "weights" / "best.pt",
        run_dir / "detect" / "runs" / run_name / "weights" / "best.pt",
        run_dir / "detect" / "results" / run_name / "weights" / "best.pt",
    ]
    return next((path for path in candidates if path.exists()), None)


def train_command(
    experiment: Experiment,
    run_name: str,
    yaml_path: Path,
    seed: int,
    args,
) -> list[str]:
    command = [
        sys.executable,
        experiment.train_script,
        "--cfg",
        str(yaml_path),
        "--run_name",
        run_name,
        "--model",
        args.model,
        "--imgsz",
        str(args.imgsz),
        "--epochs",
        str(args.epochs),
        "--batch",
        str(args.batch),
        "--workers",
        str(args.workers),
        "--device",
        args.device,
        "--project",
        str(args.run_dir),
        "--seed",
        str(seed),
        "--patience",
        str(args.patience),
    ]
    if experiment.group == "4":
        command.extend(["--warmup_epochs", str(args.acs_warmup_epochs)])
        if args.force_retrain:
            command.append("--force_retrain")
    return command


def evaluate_command(
    experiment: Experiment,
    run_name: str,
    weights: Path,
    seed: int,
    args,
) -> list[str]:
    ood_zones = [zone for zone in ZONES if zone not in experiment.train_zones]
    return [
        sys.executable,
        "scripts/evaluate_domains.py",
        "--weights",
        str(weights),
        "--data_dir",
        str(args.data_dir),
        "--run_name",
        run_name,
        "--base_run",
        experiment.base_run,
        "--method",
        experiment.method,
        "--seed",
        str(seed),
        "--id_zones",
        *experiment.train_zones,
        "--ood_zones",
        *ood_zones,
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--device",
        args.device,
        "--out_dir",
        str(args.results_dir),
    ]


def result_row(result: dict) -> dict:
    aggregate = result["aggregate"]
    row = {
        "run_name": result["run_name"],
        "base_run": result["base_run"],
        "method": result["method"],
        "seed": result["seed"],
        "train_domains": "+".join(result["id_zones"]),
        "ood_domains": "+".join(result["ood_zones"]),
        "ID_mAP50": aggregate["mAP50_ID"],
        "OOD_mAP50": aggregate["mAP50_OOD"],
        "ID_mAP50_95": aggregate["mAP5095_ID"],
        "OOD_mAP50_95": aggregate["mAP5095_OOD"],
        "PD_50": aggregate["PD_50"],
        "PD_50_95": aggregate["PD_5095"],
        "H_50": aggregate["H_50"],
        "H_50_95": aggregate["H_5095"],
    }
    for zone in ZONES:
        metrics = result.get("per_zone", {}).get(zone, {})
        row[f"{zone}_mAP50"] = metrics.get("mAP50")
        row[f"{zone}_mAP50_95"] = metrics.get("mAP50_95")
    return row


def write_reports(results: list[dict], results_dir: Path) -> None:
    if not results:
        raise RuntimeError("No experiment results were collected.")

    results_dir.mkdir(parents=True, exist_ok=True)
    individual = pd.DataFrame(result_row(result) for result in results)
    individual = individual.sort_values(["base_run", "seed"])
    individual.to_csv(results_dir / "individual_results.csv", index=False)

    identity = ["base_run", "method", "train_domains", "ood_domains"]
    metrics = [
        column
        for column in individual.columns
        if column not in {*identity, "run_name", "seed"}
    ]
    grouped = individual.groupby(identity, dropna=False)[metrics]
    mean = grouped.mean().add_suffix("_mean")
    std = grouped.std(ddof=1).fillna(0.0).add_suffix("_std")
    count = grouped.size().rename("n_seeds")
    aggregate = pd.concat([count, mean, std], axis=1).reset_index()
    aggregate.to_csv(results_dir / "summary_mean_std.csv", index=False)

    paper = aggregate[
        [
            "base_run",
            "method",
            "train_domains",
            "ood_domains",
            "n_seeds",
            "ID_mAP50_95_mean",
            "ID_mAP50_95_std",
            "OOD_mAP50_95_mean",
            "OOD_mAP50_95_std",
            "PD_50_95_mean",
            "PD_50_95_std",
            "H_50_95_mean",
            "H_50_95_std",
        ]
    ]
    paper.to_csv(results_dir / "paper_table.csv", index=False)
    print(f"\nReports written to {results_dir}")
    print(paper.to_string(index=False))


def require_leakage_safe_manifests(data_dir: Path) -> None:
    missing = [
        str(data_dir / zone / "split_manifest.csv")
        for zone in ZONES
        if not (data_dir / zone / "split_manifest.csv").exists()
    ]
    if missing:
        raise RuntimeError(
            "Leakage-safe split manifests are required. Run "
            "'python scripts/split_domain.py --force' first. Missing: "
            + ", ".join(missing)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--cfg_dir", type=Path, default=Path("configs"))
    parser.add_argument("--run_dir", type=Path, default=Path("runs"))
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("results/official"),
    )
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--acs_warmup_epochs", type=int, default=15)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument(
        "--exp",
        nargs="+",
        choices=["1", "2", "3", "4"],
        default=["1", "2", "3", "4"],
    )
    parser.add_argument("--force_retrain", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()

    args.cfg_dir = args.cfg_dir.resolve()
    args.data_dir = args.data_dir.resolve()
    args.run_dir = args.run_dir.resolve()
    args.results_dir = args.results_dir.resolve()
    require_leakage_safe_manifests(args.data_dir)
    selected = [experiment for experiment in EXPERIMENTS if experiment.group in args.exp]
    results = []

    for experiment in selected:
        yaml_path = yaml_for(experiment, args.cfg_dir)
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"Missing {yaml_path}. Rebuild leakage-safe splits first."
            )

        for seed in args.seeds:
            run_name = f"{experiment.base_run}_seed{seed}"
            weights = find_weights(args.run_dir, run_name)
            if not args.eval_only and (weights is None or args.force_retrain):
                run_command(
                    train_command(experiment, run_name, yaml_path, seed, args),
                    f"Training {run_name}",
                )
                weights = find_weights(args.run_dir, run_name)
            if weights is None:
                raise FileNotFoundError(f"Weights not found for {run_name}")

            run_command(
                evaluate_command(experiment, run_name, weights, seed, args),
                f"Cross-domain evaluation {run_name}",
            )
            json_path = args.results_dir / f"{run_name}_results.json"
            with open(json_path, encoding="utf-8") as handle:
                results.append(json.load(handle))

    write_reports(results, args.results_dir)
    run_command(
        [
            sys.executable,
            "scripts/visualize_official_results.py",
            "--input-dir",
            str(args.results_dir),
            "--output-dir",
            str(args.results_dir / "plots"),
        ],
        "Generating official report plots",
    )


if __name__ == "__main__":
    main()
