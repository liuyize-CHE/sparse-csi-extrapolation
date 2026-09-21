"""Complex linear-interpolation baseline for the initial CSI dataset.

One spatial measurement reveals the complex CSI of all subcarriers at that
position. Official 5%, 10%, and 20% experiments run only when
``--run-experiment`` is supplied, so preregistered predictions can be saved in
Overleaf before producing results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sparse_csi.data import (
    build_coordinates,
    draw_random_spatial_mask,
    load_dataset,
)
from sparse_csi.interpolation import linear_reconstruction, nearest_reconstruction
from sparse_csi.metrics import calculate_metrics, summarize_records, write_csv
from sparse_csi.paths import DEFAULT_DATASET, PROJECT_ROOT, SAMPLING_RATES


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "interpolation_baselines"
METHOD_NAME = "complex_linear"


def save_metric_curves(
    summary: list[dict[str, float | int | str]],
    output_dir: Path,
) -> Path:
    metric_specs = (
        ("nmse_db", "Complex NMSE (dB)"),
        ("amplitude_nmae", "Amplitude NMAE"),
        ("phase_mae_rad", "Phase MAE (rad)"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)

    rows = sorted(summary, key=lambda row: float(row["sampling_rate"]))
    rates = np.array([float(row["sampling_rate"]) for row in rows])
    for axis, (metric_name, label) in zip(axes, metric_specs):
        means = np.array([float(row[f"{metric_name}_mean"]) for row in rows])
        standard_deviations = np.array(
            [float(row[f"{metric_name}_std"]) for row in rows]
        )
        axis.errorbar(
            rates * 100.0,
            means,
            yerr=standard_deviations,
            marker="o",
            capsize=4,
            label="Complex linear interpolation",
        )
        axis.set_xlabel("Observed spatial positions (%)")
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)
        axis.legend()

    output_path = output_dir / "baseline_metric_curves.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def save_reconstruction_figure(
    h_true: np.ndarray,
    sample_mask: np.ndarray,
    reconstruction: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    subcarrier_index: int,
    scene_index: int,
    output_dir: Path,
) -> Path:
    extent = [float(x_m.min()), float(x_m.max()), float(y_m.min()), float(y_m.max())]
    reference = float(np.max(np.abs(h_true[subcarrier_index])))
    display_order = ("truth", "linear")
    display_names = {"truth": "Ground truth", "linear": "Complex linear"}

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)

    xx, yy = np.meshgrid(x_m, y_m)
    axes[0, 0].scatter(xx[sample_mask], yy[sample_mask], s=3, c="black")
    axes[0, 0].set_title("Observed positions (10%)")
    axes[0, 0].set_xlim(extent[0], extent[1])
    axes[0, 0].set_ylim(extent[2], extent[3])
    axes[1, 0].axis("off")
    axes[1, 0].text(
        0.5,
        0.5,
        f"scene={scene_index}\nsubcarrier={subcarrier_index}",
        ha="center",
        va="center",
        fontsize=14,
    )

    for column, method in enumerate(display_order, start=1):
        field = (
            h_true[subcarrier_index]
            if method == "truth"
            else reconstruction[subcarrier_index]
        )
        amplitude_db = 20.0 * np.log10(
            np.abs(field) / max(reference, 1.0e-12) + 1.0e-12
        )
        amplitude_image = axes[0, column].imshow(
            amplitude_db,
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=-70,
            vmax=0,
        )
        axes[0, column].set_title(f"{display_names[method]} amplitude")
        fig.colorbar(amplitude_image, ax=axes[0, column])

        phase_image = axes[1, column].imshow(
            np.angle(field),
            origin="lower",
            extent=extent,
            cmap="twilight",
            vmin=-np.pi,
            vmax=np.pi,
        )
        axes[1, column].set_title(f"{display_names[method]} phase")
        fig.colorbar(phase_image, ax=axes[1, column])

    for axis in axes.flat:
        if axis.axison:
            axis.set_xlabel("x (m)")
            axis.set_ylabel("y (m)")

    output_path = output_dir / "baseline_10pct_reconstruction.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def run_experiment(
    dataset_path: Path,
    output_dir: Path,
    split: str,
    repeats: int,
    seed: int,
) -> None:
    dataset = load_dataset(dataset_path)
    csi = dataset["csi"]
    x_m = dataset["x_m"]
    y_m = dataset["y_m"]
    coordinates = build_coordinates(x_m, y_m)
    scene_indices = dataset[f"{split}_scene_indices"]
    records: list[dict[str, float | int | str]] = []
    representative = None

    for scene_index in scene_indices:
        h_true = csi[int(scene_index)]
        valid_mask = dataset["valid_masks"][int(scene_index)]

        for rate_index, sampling_rate in enumerate(SAMPLING_RATES):
            for repeat in range(repeats):
                run_seed = seed + int(scene_index) * 10_000 + rate_index * 100 + repeat
                rng = np.random.default_rng(run_seed)
                sample_mask = draw_random_spatial_mask(valid_mask, sampling_rate, rng)
                evaluation_mask = valid_mask & ~sample_mask

                nearest_result = nearest_reconstruction(
                    h_true,
                    coordinates,
                    sample_mask,
                )
                linear_result = linear_reconstruction(
                    h_true,
                    coordinates,
                    sample_mask,
                    nearest_result=nearest_result,
                )
                overall_metrics = calculate_metrics(
                    h_true,
                    linear_result,
                    evaluation_mask,
                )
                records.append(
                    {
                        "split": split,
                        "scene_index": int(scene_index),
                        "sampling_rate": sampling_rate,
                        "repeat": repeat,
                        "seed": run_seed,
                        "method": METHOD_NAME,
                        "sample_count": int(sample_mask.sum()),
                        "test_position_count": int(evaluation_mask.sum()),
                        **overall_metrics,
                    }
                )

                if (
                    representative is None
                    and sampling_rate == 0.10
                    and repeat == 0
                ):
                    representative = (
                        int(scene_index),
                        h_true,
                        sample_mask.copy(),
                        linear_result,
                    )

    if representative is None:
        raise RuntimeError("No representative 10% reconstruction was generated.")

    output_dir.mkdir(parents=True, exist_ok=True)
    detailed_path = output_dir / f"baseline_{split}_detailed_metrics.csv"
    summary_path = output_dir / f"baseline_{split}_summary.csv"
    write_csv(detailed_path, records)
    summary = summarize_records(records, methods=(METHOD_NAME,))
    write_csv(summary_path, summary)

    scene_index, h_true, sample_mask, reconstruction = representative
    metric_figure_path = save_metric_curves(summary, output_dir)
    reconstruction_path = save_reconstruction_figure(
        h_true,
        sample_mask,
        reconstruction,
        x_m,
        y_m,
        subcarrier_index=len(dataset["frequencies_hz"]) // 2,
        scene_index=scene_index,
        output_dir=output_dir,
    )
    run_metadata = {
        "dataset": str(dataset_path),
        "split": split,
        "scene_indices": scene_indices.tolist(),
        "sampling_rates": list(SAMPLING_RATES),
        "repeats": repeats,
        "seed": seed,
        "method": METHOD_NAME,
        "boundary_fill": "nearest measured position",
        "reported_metrics": ["nmse_db", "amplitude_nmae", "phase_mae_rad"],
        "spatial_measurement_reveals_all_subcarriers": True,
    }
    metadata_path = output_dir / f"baseline_{split}_run_metadata.json"
    metadata_path.write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Official complex linear-interpolation baseline completed.")
    print(f"Split: {split}, scenes: {scene_indices.tolist()}")
    print(f"Detailed metrics: {detailed_path}")
    print(f"Summary: {summary_path}")
    print(f"Metric curves: {metric_figure_path}")
    print(f"Reconstruction: {reconstruction_path}")


def run_self_test() -> None:
    """Validate complex linear interpolation on an affine complex field."""
    axis = np.linspace(0.0, 1.0, 9)
    xx, yy = np.meshgrid(axis, axis)
    coordinates = build_coordinates(axis, axis)

    h0 = (1.0 + 2.0 * xx - 0.5 * yy) + 1j * (-0.3 + 0.2 * xx + yy)
    h1 = (-0.4 + 0.7 * xx + 1.2 * yy) + 1j * (0.8 - xx + 0.1 * yy)
    h_true = np.stack((h0, h1), axis=0).astype(np.complex64)

    sample_mask = np.zeros_like(xx, dtype=bool)
    sample_mask[0, 0] = True
    sample_mask[0, -1] = True
    sample_mask[-1, 0] = True
    sample_mask[-1, -1] = True

    nearest_result = nearest_reconstruction(h_true, coordinates, sample_mask)
    linear_result = linear_reconstruction(
        h_true,
        coordinates,
        sample_mask,
        nearest_result=nearest_result,
    )

    maximum_linear_error = float(np.max(np.abs(linear_result - h_true)))
    if maximum_linear_error > 5.0e-6:
        raise AssertionError(
            f"Linear interpolation self-test failed: {maximum_linear_error}"
        )
    if nearest_result.shape != h_true.shape or not np.isfinite(nearest_result).all():
        raise AssertionError("Nearest interpolation produced an invalid result.")
    if not np.array_equal(linear_result[:, sample_mask], h_true[:, sample_mask]):
        raise AssertionError("Measured values were not preserved exactly.")

    print("Interpolation baseline self-test passed.")
    print(f"Shape: {linear_result.shape}")
    print(f"Maximum affine-field linear error: {maximum_linear_error:.3e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--self-test", action="store_true")
    action_group.add_argument("--run-experiment", action="store_true")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="test",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    run_experiment(
        dataset_path=args.dataset.resolve(),
        output_dir=args.output_dir.resolve(),
        split=args.split,
        repeats=args.repeats,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
