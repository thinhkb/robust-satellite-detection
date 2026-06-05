"""Create report-ready charts from multi-seed cross-domain results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_COLORS = {
    "Single-source": "#4C78A8",
    "Multi-source": "#F58518",
    "DG-Aug": "#54A24B",
    "ACS-YOLO": "#E45756",
}


def load_results(input_dir: Path) -> pd.DataFrame:
    path = input_dir / "individual_results.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/run_experiments.py first."
        )
    frame = pd.read_csv(path)
    required = {
        "method",
        "ood_domains",
        "seed",
        "ID_mAP50_95",
        "OOD_mAP50_95",
        "PD_50_95",
        "H_50_95",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing report columns: {sorted(missing)}")
    return frame


def save_ood_chart(frame: pd.DataFrame, output: Path) -> None:
    grouped = (
        frame.groupby(["method", "ood_domains"])["OOD_mAP50_95"]
        .agg(["mean", "std"])
        .fillna(0.0)
        .reset_index()
    )
    zones = sorted(grouped["ood_domains"].unique())
    methods = [method for method in METHOD_COLORS if method in set(grouped["method"])]
    x = np.arange(len(zones))
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(11, 6))
    for index, method in enumerate(methods):
        subset = grouped[grouped["method"] == method].set_index("ood_domains")
        means = np.array(
            [subset.loc[zone, "mean"] if zone in subset.index else np.nan for zone in zones]
        )
        stds = np.array(
            [subset.loc[zone, "std"] if zone in subset.index else 0.0 for zone in zones]
        )
        positions = x + (index - (len(methods) - 1) / 2) * width
        bars = ax.bar(
            positions,
            np.nan_to_num(means),
            width,
            yerr=stds,
            capsize=3,
            label=method,
            color=METHOD_COLORS[method],
        )
        for bar, value in zip(bars, means):
            if np.isnan(value):
                bar.set_alpha(0)
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.003,
                    f"{value:.3f}",
                    ha="center",
                    fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(zones)
    ax.set_ylabel("OOD test mAP50-95")
    ax.set_title("Cross-Domain Generalization (mean +/- std across seeds)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_tradeoff_chart(frame: pd.DataFrame, output: Path) -> None:
    grouped = (
        frame.groupby("method")[["ID_mAP50_95", "OOD_mAP50_95"]]
        .agg(["mean", "std"])
        .fillna(0.0)
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    for method, row in grouped.iterrows():
        ax.errorbar(
            row[("ID_mAP50_95", "mean")],
            row[("OOD_mAP50_95", "mean")],
            xerr=row[("ID_mAP50_95", "std")],
            yerr=row[("OOD_mAP50_95", "std")],
            fmt="o",
            markersize=10,
            capsize=4,
            color=METHOD_COLORS.get(method, "#777777"),
            label=method,
        )
    ax.set_xlabel("ID test mAP50-95")
    ax.set_ylabel("OOD test mAP50-95")
    ax.set_title("ID/OOD Accuracy Trade-off")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_table(frame: pd.DataFrame, output_csv: Path, output_png: Path) -> None:
    grouped = frame.groupby(
        ["method", "train_domains", "ood_domains"],
        as_index=False,
    ).agg(
        n_seeds=("seed", "nunique"),
        ID_mean=("ID_mAP50_95", "mean"),
        ID_std=("ID_mAP50_95", "std"),
        OOD_mean=("OOD_mAP50_95", "mean"),
        OOD_std=("OOD_mAP50_95", "std"),
        PD_mean=("PD_50_95", "mean"),
        PD_std=("PD_50_95", "std"),
        H_mean=("H_50_95", "mean"),
        H_std=("H_50_95", "std"),
    ).fillna(0.0)
    grouped.to_csv(output_csv, index=False, float_format="%.6f")

    display = grouped.copy()
    for metric in ["ID", "OOD", "PD", "H"]:
        display[metric] = display.apply(
            lambda row: f"{row[f'{metric}_mean']:.4f} +/- {row[f'{metric}_std']:.4f}",
            axis=1,
        )
    display = display[
        ["method", "train_domains", "ood_domains", "n_seeds", "ID", "OOD", "PD", "H"]
    ]

    fig, ax = plt.subplots(figsize=(15, max(4, len(display) * 0.55 + 1.5)))
    ax.axis("off")
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    for (row, _column), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#263238")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F3F6F8")
    ax.set_title("Official Cross-Domain Results", fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("results/official"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/official/plots"),
    )
    args = parser.parse_args()
    frame = load_results(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_ood_chart(frame, args.output_dir / "ood_map_by_zone.png")
    save_tradeoff_chart(frame, args.output_dir / "id_ood_tradeoff.png")
    save_table(
        frame,
        args.output_dir / "report_table.csv",
        args.output_dir / "report_table.png",
    )
    print(f"Official plots written to {args.output_dir}")


if __name__ == "__main__":
    main()
