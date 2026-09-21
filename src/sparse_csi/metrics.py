"""Evaluation and result-table helpers shared by all experiments."""

import csv
from pathlib import Path

import numpy as np

from sparse_csi.paths import SAMPLING_RATES


Record = dict[str, float | int | str]
METRIC_NAMES = ("nmse_db", "amplitude_nmae", "phase_mae_rad")


def calculate_metrics(
    h_true: np.ndarray,
    h_predicted: np.ndarray,
    evaluation_mask: np.ndarray,
) -> dict[str, float]:
    """Evaluate all subcarriers at unobserved valid positions."""
    true_values = h_true[:, evaluation_mask]
    predicted_values = h_predicted[:, evaluation_mask]

    error_energy = np.sum(np.abs(predicted_values - true_values) ** 2)
    signal_energy = np.sum(np.abs(true_values) ** 2)
    nmse = float(error_energy / max(float(signal_energy), 1.0e-12))

    true_amplitude = np.abs(true_values)
    predicted_amplitude = np.abs(predicted_values)
    amplitude_nmae = float(
        np.mean(np.abs(predicted_amplitude - true_amplitude))
        / max(float(np.mean(true_amplitude)), 1.0e-12)
    )

    phase_error = np.abs(np.angle(predicted_values * np.conj(true_values)))
    return {
        "nmse": nmse,
        "nmse_db": 10.0 * np.log10(max(nmse, 1.0e-12)),
        "amplitude_nmae": amplitude_nmae,
        "phase_mae_rad": float(np.mean(phase_error)),
    }


def summarize_records(
    records: list[Record],
    methods: tuple[str, ...],
    sampling_rates: tuple[float, ...] = SAMPLING_RATES,
) -> list[Record]:
    """Aggregate repeated runs by sampling rate and method."""
    summary: list[Record] = []
    for sampling_rate in sampling_rates:
        for method in methods:
            selected = [
                record
                for record in records
                if float(record["sampling_rate"]) == sampling_rate
                and record["method"] == method
            ]
            if not selected:
                continue

            row: Record = {
                "sampling_rate": sampling_rate,
                "method": method,
                "run_count": len(selected),
            }
            for metric_name in METRIC_NAMES:
                values = np.asarray(
                    [float(record[metric_name]) for record in selected],
                    dtype=np.float64,
                )
                row[f"{metric_name}_mean"] = float(values.mean())
                row[f"{metric_name}_std"] = float(
                    values.std(ddof=1) if values.size > 1 else 0.0
                )
            summary.append(row)
    return summary


def write_csv(path: Path, records: list[Record]) -> None:
    """Write dictionaries as a UTF-8 CSV with a header."""
    if not records:
        raise ValueError(f"Cannot write empty records to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
