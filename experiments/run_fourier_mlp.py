"""Train and evaluate a coordinate-based Fourier-feature MLP.

The model maps a 2-D location to the real and imaginary CSI values of every
subcarrier. Use validation scenes for bandwidth selection, then run the test
split once with the saved configuration.
"""

from __future__ import annotations

import argparse
import json
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
from sparse_csi.paths import DEFAULT_DATASET, PROJECT_ROOT, SAMPLING_RATES


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "fourier_mlp"
DEFAULT_CONFIG_PATH = DEFAULT_OUTPUT_DIR / "selected_config.json"
METHODS = ("complex_linear", "fourier_mlp")


def save_comparison_curves(
    summary: list[dict[str, float | int | str]], output_dir: Path
) -> Path:
    specifications = (
        ("nmse_db", "Complex NMSE (dB)"),
        ("amplitude_nmae", "Amplitude NMAE"),
        ("phase_mae_rad", "Phase MAE (rad)"),
    )
    display_names = {
        "complex_linear": "Complex linear",
        "fourier_mlp": "Fourier MLP",
    }
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for axis, (metric, label) in zip(axes, specifications):
        for method in METHODS:
            rows = sorted(
                [row for row in summary if row["method"] == method],
                key=lambda row: float(row["sampling_rate"]),
            )
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
        axis.set_xlabel("Observed spatial positions (%)")
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)
        axis.legend()
    path = output_dir / "fourier_mlp_comparison_curves.png"
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def save_reconstruction_comparison(
    h_true: np.ndarray,
    sample_mask: np.ndarray,
    linear_result: np.ndarray,
    mlp_result: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    scene_index: int,
    subcarrier_index: int,
    output_dir: Path,
) -> Path:
    fields = (
        ("Ground truth", h_true[subcarrier_index]),
        ("Complex linear", linear_result[subcarrier_index]),
        ("Fourier MLP", mlp_result[subcarrier_index]),
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
    path = output_dir / "fourier_mlp_10pct_reconstruction.png"
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
    candidates = (4.0, 8.0, 12.0)
    candidate_records: list[dict[str, float | int | str]] = []
    rate_index = SAMPLING_RATES.index(0.10)

    for maximum_frequency in candidates:
        config = ModelConfig(max_frequency_cpm=maximum_frequency)
        for scene_index_value in scene_indices:
            scene_index = int(scene_index_value)
            h_true = dataset["csi"][scene_index]
            valid_mask = dataset["valid_masks"][scene_index]
            for repeat in range(repeats):
                run_seed = run_seed_for(scene_index, rate_index, repeat, seed)
                sample_mask = draw_random_spatial_mask(
                    valid_mask,
                    0.10,
                    np.random.default_rng(run_seed),
                )
                evaluation_mask = valid_mask & ~sample_mask
                reconstruction, diagnostics = fit_and_reconstruct(
                    h_true,
                    coordinates,
                    sample_mask,
                    config,
                    run_seed + 50_000,
                    device,
                )
                metrics = calculate_metrics(h_true, reconstruction, evaluation_mask)
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

    means = {
        maximum_frequency: float(
            np.mean(
                [
                    float(record["nmse_db"])
                    for record in candidate_records
                    if float(record["max_frequency_cpm"]) == maximum_frequency
                ]
            )
        )
        for maximum_frequency in candidates
    }
    best_frequency = min(means, key=means.get)
    selected_config = ModelConfig(max_frequency_cpm=best_frequency)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "validation_tuning_details.csv", candidate_records)
    config_payload = {
        **asdict(selected_config),
        "selection_split": "validation",
        "validation_scene_indices": scene_indices.tolist(),
        "selection_sampling_rate": 0.10,
        "selection_repeats": repeats,
        "mean_nmse_db_by_max_frequency": means,
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Validation tuning completed.")
    print(f"Device: {device}")
    print(f"Mean NMSE (dB): {means}")
    print(f"Selected max frequency: {best_frequency} cycles/m")
    print(f"Saved configuration: {config_path}")


def load_selected_config(config_path: Path) -> ModelConfig:
    if not config_path.exists():
        raise FileNotFoundError(
            f"Selected configuration not found: {config_path}. "
            "Run --tune-validation first."
        )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    allowed = set(ModelConfig.__dataclass_fields__)
    return ModelConfig(**{key: payload[key] for key in allowed})


def run_test(
    dataset_path: Path,
    output_dir: Path,
    config_path: Path,
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
        for rate_index, sampling_rate in enumerate(SAMPLING_RATES):
            for repeat in range(repeats):
                run_seed = run_seed_for(scene_index, rate_index, repeat, seed)
                sample_mask = draw_random_spatial_mask(
                    valid_mask,
                    sampling_rate,
                    np.random.default_rng(run_seed),
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
                mlp_result, diagnostics = fit_and_reconstruct(
                    h_true,
                    coordinates,
                    sample_mask,
                    config,
                    run_seed + 50_000,
                    device,
                )

                common = {
                    "split": "test",
                    "scene_index": scene_index,
                    "sampling_rate": sampling_rate,
                    "repeat": repeat,
                    "seed": run_seed,
                    "sample_count": int(sample_mask.sum()),
                    "test_position_count": int(evaluation_mask.sum()),
                }
                for method, reconstruction in (
                    ("complex_linear", linear_result),
                    ("fourier_mlp", mlp_result),
                ):
                    metrics = calculate_metrics(
                        h_true, reconstruction, evaluation_mask
                    )
                    record: dict[str, float | int | str] = {
                        **common,
                        "method": method,
                        "nmse": metrics["nmse"],
                        "nmse_db": metrics["nmse_db"],
                        "amplitude_nmae": metrics["amplitude_nmae"],
                        "phase_mae_rad": metrics["phase_mae_rad"],
                    }
                    if method == "fourier_mlp":
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
                        mlp_result,
                    )

    if representative is None:
        raise RuntimeError("No representative 10% reconstruction was produced.")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "fourier_mlp_test_detailed_metrics.csv", records)
    summary = summarize_records(records, methods=METHODS)
    write_csv(output_dir / "fourier_mlp_test_summary.csv", summary)
    curve_path = save_comparison_curves(summary, output_dir)
    scene_index, h_true, sample_mask, linear_result, mlp_result = representative
    reconstruction_path = save_reconstruction_comparison(
        h_true,
        sample_mask,
        linear_result,
        mlp_result,
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
        "config_path": str(config_path),
        "config": asdict(config),
        "reported_metrics": ["nmse_db", "amplitude_nmae", "phase_mae_rad"],
        "spatial_measurement_reveals_all_subcarriers": True,
    }
    (output_dir / "fourier_mlp_test_run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Official Fourier MLP test completed.")
    print(f"Device: {device}")
    print(f"Scenes: {scene_indices.tolist()}")
    print(f"Summary: {output_dir / 'fourier_mlp_test_summary.csv'}")
    print(f"Metric curves: {curve_path}")
    print(f"Reconstruction comparison: {reconstruction_path}")


def run_self_test(device: torch.device) -> None:
    axis = np.linspace(0.0, 1.0, 25, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)
    coordinates = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float64)
    h0 = np.exp(1j * 2.0 * np.pi * 4.0 * xx)
    h1 = 0.7 * np.exp(1j * 2.0 * np.pi * 5.0 * yy)
    h_true = np.stack((h0, h1), axis=0).astype(np.complex64)
    sample_mask = np.zeros_like(xx, dtype=bool)
    rng = np.random.default_rng(91)
    sample_mask.flat[rng.choice(sample_mask.size, size=250, replace=False)] = True
    config = ModelConfig(
        max_frequency_cpm=6.0,
        frequency_band_count=6,
        direction_count=8,
        hidden_width=64,
        hidden_depth=2,
        max_epochs=800,
        patience=100,
    )
    reconstruction, diagnostics = fit_and_reconstruct(
        h_true,
        coordinates,
        sample_mask,
        config,
        seed=123,
        device=device,
    )
    evaluation_mask = ~sample_mask
    metrics = calculate_metrics(h_true, reconstruction, evaluation_mask)
    if not np.isfinite(reconstruction).all():
        raise AssertionError("Fourier MLP self-test produced non-finite values.")
    if metrics["nmse"] >= 0.25:
        raise AssertionError(f"Synthetic self-test NMSE is too high: {metrics['nmse']}")
    if not np.array_equal(reconstruction[:, sample_mask], h_true[:, sample_mask]):
        raise AssertionError("Measured CSI values were not preserved exactly.")
    print("Fourier MLP synthetic self-test passed.")
    print(f"Device: {device}")
    print(f"Selected epoch: {diagnostics['selected_epoch']}")
    print(f"Unobserved complex NMSE: {metrics['nmse']:.6f}")
    print(f"Unobserved phase MAE: {metrics['phase_mae_rad']:.6f} rad")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--self-test", action="store_true")
    actions.add_argument("--tune-validation", action="store_true")
    actions.add_argument("--run-test", action="store_true")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
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
            args.repeats,
            args.seed,
            device,
        )


if __name__ == "__main__":
    main()
