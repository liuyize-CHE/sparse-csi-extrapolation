"""Tx-aware phase-compensated Fourier MLP for sparse CSI extrapolation.

The known direct-path carrier phase is removed before fitting the same Fourier
MLP used in ``run_fourier_mlp.py``. Evaluation is always performed
after restoring the original complex CSI.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from sparse_csi.data import (
    build_coordinates,
    draw_random_spatial_mask,
    load_dataset,
)
from sparse_csi.interpolation import linear_reconstruction, nearest_reconstruction
from sparse_csi.metrics import calculate_metrics, summarize_records, write_csv
from sparse_csi.models import (
    ModelConfig,
    fit_and_reconstruct,
    run_seed_for,
    select_device,
)
from sparse_csi.paths import (
    DEFAULT_DATASET,
    PROJECT_ROOT,
    SAMPLING_RATES,
    SPEED_OF_LIGHT_MPS,
)
from sparse_csi.physics import (
    compensate_direct_phase,
    direct_phase_factor,
    restore_direct_phase,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "tx_aware_fourier_mlp"
DEFAULT_CONFIG_PATH = DEFAULT_OUTPUT_DIR / "selected_config.json"
DEFAULT_PLAIN_SUMMARY = (
    PROJECT_ROOT / "results" / "fourier_mlp" / "fourier_mlp_test_summary.csv"
)


def load_plain_summary(path: Path) -> list[dict[str, float | int | str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        rows = list(csv.DictReader(csv_file))
    return [row for row in rows if row["method"] == "fourier_mlp"]


def save_comparison_curves(
    summary: list[dict[str, float | int | str]],
    plain_summary: list[dict[str, float | int | str]],
    output_dir: Path,
) -> Path:
    combined = [*summary, *plain_summary]
    specifications = (
        ("nmse_db", "Complex NMSE (dB)"),
        ("amplitude_nmae", "Amplitude NMAE"),
        ("phase_mae_rad", "Phase MAE (rad)"),
    )
    method_order = (
        "complex_linear",
        "fourier_mlp",
        "tx_aware_fourier_mlp",
    )
    display_names = {
        "complex_linear": "Complex linear",
        "fourier_mlp": "Fourier MLP",
        "tx_aware_fourier_mlp": "Tx-aware Fourier MLP",
    }
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
    for axis, (metric, label) in zip(axes, specifications):
        for method in method_order:
            rows = sorted(
                [row for row in combined if row["method"] == method],
                key=lambda row: float(row["sampling_rate"]),
            )
            if not rows:
                continue
            rates = np.asarray([float(row["sampling_rate"]) for row in rows])
            means = np.asarray([float(row[f"{metric}_mean"]) for row in rows])
            deviations = np.asarray([float(row[f"{metric}_std"]) for row in rows])
            axis.errorbar(
                rates * 100.0,
                means,
                yerr=deviations,
                marker="o",
                capsize=4,
                label=display_names[method],
            )
        if metric == "nmse_db":
            axis.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
        axis.set_xlabel("Observed spatial positions (%)")
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    path = output_dir / "tx_aware_comparison_curves.png"
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def save_reconstruction_figure(
    h_true: np.ndarray,
    sample_mask: np.ndarray,
    linear_result: np.ndarray,
    tx_aware_result: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    scene_index: int,
    subcarrier_index: int,
    output_dir: Path,
) -> Path:
    fields = (
        ("Ground truth", h_true[subcarrier_index]),
        ("Complex linear", linear_result[subcarrier_index]),
        ("Tx-aware Fourier MLP", tx_aware_result[subcarrier_index]),
    )
    extent = [float(x_m.min()), float(x_m.max()), float(y_m.min()), float(y_m.max())]
    reference = max(float(np.max(np.abs(h_true[subcarrier_index]))), 1.0e-12)
    figure, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
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
    for column, (name, field) in enumerate(fields, start=1):
        amplitude_db = 20.0 * np.log10(np.abs(field) / reference + 1.0e-12)
        amplitude_image = axes[0, column].imshow(
            amplitude_db,
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=-70,
            vmax=0,
        )
        axes[0, column].set_title(f"{name} amplitude")
        figure.colorbar(amplitude_image, ax=axes[0, column])
        phase_image = axes[1, column].imshow(
            np.angle(field),
            origin="lower",
            extent=extent,
            cmap="twilight",
            vmin=-np.pi,
            vmax=np.pi,
        )
        axes[1, column].set_title(f"{name} phase")
        figure.colorbar(phase_image, ax=axes[1, column])
    for axis in axes.flat:
        if axis.axison:
            axis.set_xlabel("x (m)")
            axis.set_ylabel("y (m)")
    path = output_dir / "tx_aware_10pct_reconstruction.png"
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def tune_validation(
    dataset_path: Path,
    output_dir: Path,
    config_path: Path,
    repeats: int,
    seed: int,
    device: torch.device,
) -> None:
    dataset = load_dataset(dataset_path)
    coordinates = build_coordinates(dataset["x_m"], dataset["y_m"])
    scene_indices = dataset["validation_scene_indices"]
    candidates = (1.0, 2.0, 4.0, 8.0, 12.0)
    candidate_records: list[dict[str, float | int | str]] = []
    rate_index = SAMPLING_RATES.index(0.10)

    for maximum_frequency in candidates:
        config = ModelConfig(max_frequency_cpm=maximum_frequency)
        for scene_index_value in scene_indices:
            scene_index = int(scene_index_value)
            h_true = dataset["csi"][scene_index]
            valid_mask = dataset["valid_masks"][scene_index]
            phase_factor = direct_phase_factor(
                coordinates,
                dataset["tx_positions_m"][scene_index],
                dataset["frequencies_hz"],
                h_true.shape,
            )
            compensated_true = compensate_direct_phase(h_true, phase_factor)
            for repeat in range(repeats):
                run_seed = run_seed_for(scene_index, rate_index, repeat, seed)
                sample_mask = draw_random_spatial_mask(
                    valid_mask, 0.10, np.random.default_rng(run_seed)
                )
                evaluation_mask = valid_mask & ~sample_mask
                compensated_prediction, diagnostics = fit_and_reconstruct(
                    compensated_true,
                    coordinates,
                    sample_mask,
                    config,
                    run_seed + 70_000,
                    device,
                )
                prediction = restore_direct_phase(
                    compensated_prediction, phase_factor
                )
                prediction[:, sample_mask] = h_true[:, sample_mask]
                metrics = calculate_metrics(h_true, prediction, evaluation_mask)
                candidate_records.append(
                    {
                        "max_frequency_cpm": maximum_frequency,
                        "scene_index": scene_index,
                        "repeat": repeat,
                        "seed": run_seed,
                        **diagnostics,
                        "nmse_db": metrics["nmse_db"],
                        "amplitude_nmae": metrics["amplitude_nmae"],
                        "phase_mae_rad": metrics["phase_mae_rad"],
                    }
                )

    candidate_means = {}
    for maximum_frequency in candidates:
        selected = [
            record
            for record in candidate_records
            if float(record["max_frequency_cpm"]) == maximum_frequency
        ]
        candidate_means[maximum_frequency] = {
            metric: float(np.mean([float(record[metric]) for record in selected]))
            for metric in ("nmse_db", "amplitude_nmae", "phase_mae_rad")
        }

    eligible = [
        maximum_frequency
        for maximum_frequency, means in candidate_means.items()
        if means["amplitude_nmae"] < 0.8
        and means["phase_mae_rad"] < math.pi / 2.0
    ]
    selection_pool = eligible if eligible else list(candidates)
    best_frequency = min(
        selection_pool,
        key=lambda maximum_frequency: candidate_means[maximum_frequency]["nmse_db"],
    )
    selected_config = ModelConfig(max_frequency_cpm=best_frequency)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "validation_tuning_details.csv", candidate_records)
    payload = {
        **asdict(selected_config),
        "selection_split": "validation",
        "validation_scene_indices": scene_indices.tolist(),
        "selection_sampling_rate": 0.10,
        "selection_repeats": repeats,
        "candidate_means": candidate_means,
        "eligibility_rule": "amplitude_nmae < 0.8 and phase_mae_rad < pi/2",
        "eligible_frequencies_cpm": eligible,
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Tx-aware validation tuning completed.")
    print(f"Device: {device}")
    print(json.dumps(candidate_means, ensure_ascii=False, indent=2))
    print(f"Eligible frequencies: {eligible}")
    print(f"Selected max frequency: {best_frequency} cycles/m")
    print(f"Saved configuration: {config_path}")


def load_selected_config(config_path: Path) -> ModelConfig:
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration not found: {config_path}. Run --tune-validation first."
        )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    keys = set(ModelConfig.__dataclass_fields__)
    return ModelConfig(**{key: payload[key] for key in keys})


def run_test(
    dataset_path: Path,
    output_dir: Path,
    config_path: Path,
    plain_summary_path: Path,
    repeats: int,
    seed: int,
    device: torch.device,
) -> None:
    config = load_selected_config(config_path)
    dataset = load_dataset(dataset_path)
    coordinates = build_coordinates(dataset["x_m"], dataset["y_m"])
    scene_indices = dataset["test_scene_indices"]
    records: list[dict[str, float | int | str]] = []
    representative = None

    for scene_index_value in scene_indices:
        scene_index = int(scene_index_value)
        h_true = dataset["csi"][scene_index]
        valid_mask = dataset["valid_masks"][scene_index]
        phase_factor = direct_phase_factor(
            coordinates,
            dataset["tx_positions_m"][scene_index],
            dataset["frequencies_hz"],
            h_true.shape,
        )
        compensated_true = compensate_direct_phase(h_true, phase_factor)
        for rate_index, sampling_rate in enumerate(SAMPLING_RATES):
            for repeat in range(repeats):
                run_seed = run_seed_for(scene_index, rate_index, repeat, seed)
                sample_mask = draw_random_spatial_mask(
                    valid_mask, sampling_rate, np.random.default_rng(run_seed)
                )
                evaluation_mask = valid_mask & ~sample_mask
                nearest_result = nearest_reconstruction(
                    h_true, coordinates, sample_mask
                )
                linear_result = linear_reconstruction(
                    h_true,
                    coordinates,
                    sample_mask,
                    nearest_result=nearest_result,
                )
                compensated_prediction, diagnostics = fit_and_reconstruct(
                    compensated_true,
                    coordinates,
                    sample_mask,
                    config,
                    run_seed + 70_000,
                    device,
                )
                tx_aware_result = restore_direct_phase(
                    compensated_prediction, phase_factor
                )
                tx_aware_result[:, sample_mask] = h_true[:, sample_mask]

                common = {
                    "split": "test",
                    "scene_index": scene_index,
                    "sampling_rate": sampling_rate,
                    "repeat": repeat,
                    "seed": run_seed,
                    "sample_count": int(sample_mask.sum()),
                    "test_position_count": int(evaluation_mask.sum()),
                }
                for method, prediction in (
                    ("complex_linear", linear_result),
                    ("tx_aware_fourier_mlp", tx_aware_result),
                ):
                    metrics = calculate_metrics(h_true, prediction, evaluation_mask)
                    record: dict[str, float | int | str] = {
                        **common,
                        "method": method,
                        "nmse": metrics["nmse"],
                        "nmse_db": metrics["nmse_db"],
                        "amplitude_nmae": metrics["amplitude_nmae"],
                        "phase_mae_rad": metrics["phase_mae_rad"],
                    }
                    if method == "tx_aware_fourier_mlp":
                        record.update(diagnostics)
                    else:
                        record.update(
                            {
                                "selected_epoch": -1,
                                "internal_train_loss": "",
                                "internal_validation_loss": "",
                                "full_training_loss": "",
                            }
                        )
                    records.append(record)

                if representative is None and sampling_rate == 0.10 and repeat == 0:
                    representative = (
                        scene_index,
                        h_true,
                        sample_mask.copy(),
                        linear_result,
                        tx_aware_result,
                    )

    if representative is None:
        raise RuntimeError("No representative reconstruction was generated.")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "tx_aware_test_detailed_metrics.csv", records)
    summary = summarize_records(
        records,
        methods=("complex_linear", "tx_aware_fourier_mlp"),
    )
    write_csv(output_dir / "tx_aware_test_summary.csv", summary)
    curve_path = save_comparison_curves(
        summary, load_plain_summary(plain_summary_path), output_dir
    )
    scene_index, h_true, sample_mask, linear_result, tx_aware_result = representative
    reconstruction_path = save_reconstruction_figure(
        h_true,
        sample_mask,
        linear_result,
        tx_aware_result,
        dataset["x_m"],
        dataset["y_m"],
        scene_index,
        len(dataset["frequencies_hz"]) // 2,
        output_dir,
    )
    metadata = {
        "dataset": str(dataset_path),
        "split": "test",
        "scene_indices": scene_indices.tolist(),
        "sampling_rates": list(SAMPLING_RATES),
        "repeats": repeats,
        "seed": seed,
        "device": str(device),
        "config": asdict(config),
        "phase_compensation_uses": ["tx_position", "receiver_position", "frequency"],
        "phase_compensation_does_not_use": [
            "scatterer_position",
            "reflection_coefficient",
            "unobserved_csi",
        ],
        "reported_metrics": ["nmse_db", "amplitude_nmae", "phase_mae_rad"],
    }
    (output_dir / "tx_aware_test_run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Official Tx-aware Fourier MLP test completed.")
    print(f"Device: {device}")
    print(f"Summary: {output_dir / 'tx_aware_test_summary.csv'}")
    print(f"Curves: {curve_path}")
    print(f"Reconstruction: {reconstruction_path}")


def run_self_test(device: torch.device) -> None:
    axis = np.linspace(0.0, 1.0, 25, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)
    coordinates = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float64)
    tx_position = np.asarray([0.35, 0.55], dtype=np.float32)
    frequencies = np.asarray([2.99e9, 3.01e9], dtype=np.float64)
    distances = np.sqrt((xx - tx_position[0]) ** 2 + (yy - tx_position[1]) ** 2)
    amplitude = 1.0 / (distances + 0.2)
    h_true = np.stack(
        [
            amplitude
            * np.exp(-1j * 2.0 * np.pi * frequency * distances / SPEED_OF_LIGHT_MPS)
            for frequency in frequencies
        ],
        axis=0,
    ).astype(np.complex64)
    factor = direct_phase_factor(
        coordinates, tx_position, frequencies, h_true.shape
    )
    compensated = compensate_direct_phase(h_true, factor)
    round_trip = restore_direct_phase(compensated, factor)
    if float(np.max(np.abs(round_trip - h_true))) > 2.0e-6:
        raise AssertionError("Phase compensation round-trip failed.")

    sample_mask = np.zeros_like(xx, dtype=bool)
    rng = np.random.default_rng(24)
    sample_mask.flat[rng.choice(sample_mask.size, size=180, replace=False)] = True
    config = ModelConfig(
        max_frequency_cpm=2.0,
        hidden_width=64,
        hidden_depth=2,
        max_epochs=1000,
        patience=120,
    )
    compensated_prediction, diagnostics = fit_and_reconstruct(
        compensated,
        coordinates,
        sample_mask,
        config,
        seed=321,
        device=device,
    )
    prediction = restore_direct_phase(compensated_prediction, factor)
    prediction[:, sample_mask] = h_true[:, sample_mask]
    metrics = calculate_metrics(h_true, prediction, ~sample_mask)
    if not np.isfinite(prediction).all():
        raise AssertionError("Tx-aware self-test produced non-finite values.")
    if metrics["amplitude_nmae"] >= 0.25:
        raise AssertionError(
            f"Tx-aware synthetic amplitude NMAE is too high: {metrics['amplitude_nmae']}"
        )
    print("Tx-aware Fourier MLP synthetic self-test passed.")
    print(f"Device: {device}")
    print(f"Selected epoch: {diagnostics['selected_epoch']}")
    print(f"Complex NMSE: {metrics['nmse_db']:.3f} dB")
    print(f"Amplitude NMAE: {metrics['amplitude_nmae']:.4f}")
    print(f"Phase MAE: {metrics['phase_mae_rad']:.4f} rad")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--self-test", action="store_true")
    actions.add_argument("--tune-validation", action="store_true")
    actions.add_argument("--run-test", action="store_true")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--plain-summary", type=Path, default=DEFAULT_PLAIN_SUMMARY)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--tuning-repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats < 1 or args.tuning_repeats < 1:
        raise ValueError("Repeat counts must be positive.")
    device = select_device(args.device)
    if args.self_test:
        run_self_test(device)
    elif args.tune_validation:
        tune_validation(
            args.dataset.resolve(),
            args.output_dir.resolve(),
            args.config.resolve(),
            args.tuning_repeats,
            args.seed,
            device,
        )
    else:
        run_test(
            args.dataset.resolve(),
            args.output_dir.resolve(),
            args.config.resolve(),
            args.plain_summary.resolve(),
            args.repeats,
            args.seed,
            device,
        )


if __name__ == "__main__":
    main()
