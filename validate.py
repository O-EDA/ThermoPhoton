"""Validate the released ThermoPhoton checkpoint against COMSOL references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import deepheat as dde
from thermophoton.network import (
    CASE_REGIONS,
    case_query_points,
    create_case_pattern,
    get_branch_model,
    get_track_model,
)


CASES = ("3x3_mrr", "3x3_mzi", "4x4_mzi", "random_blocks")
GRID_SIZE = 120
CELSIUS_TO_KELVIN = 273.15


def build_network() -> torch.nn.Module:
    """Build the architecture used by the released checkpoint."""
    return dde.nn.DeepONetCartesianProd(
        (GRID_SIZE**2, get_branch_model(256, GRID_SIZE)),
        (3, get_track_model("ResFC", 256, 3)),
        {"branch": "tanh", "trunk": "gelu"},
        kernel_initializer="Glorot normal",
    )


def load_checkpoint(path: Path, device: torch.device) -> torch.nn.Module:
    """Load a plain PyTorch state dictionary with strict key checking."""
    network = build_network()
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch versions predating the weights_only argument
        state = torch.load(path, map_location="cpu")
    network.load_state_dict(state, strict=True)
    return network.to(device).eval()


def predict_heater_temperatures(
    network: torch.nn.Module,
    case: str,
    reference: dict,
    device: torch.device,
) -> tuple[np.ndarray, list[int]]:
    """Predict the mean temperature in each paper-defined heater region."""
    points, heater_ids = case_query_points(
        case,
        grid_size=GRID_SIZE,
        z_center=reference["z_center_mm"],
        z_half_thickness=reference["z_half_thickness_mm"],
    )
    pattern = create_case_pattern(case, GRID_SIZE).reshape(1, -1)
    with torch.inference_mode():
        values = network(
            (
                torch.as_tensor(pattern, dtype=torch.float32, device=device),
                torch.as_tensor(points, dtype=torch.float32, device=device),
            )
        )
    values_kelvin = values.squeeze(0).detach().cpu().numpy() + CELSIUS_TO_KELVIN
    means = np.asarray(
        [values_kelvin[heater_ids == index].mean() for index in range(len(CASE_REGIONS[case]))]
    )
    counts = [int(np.count_nonzero(heater_ids == index)) for index in range(len(means))]
    return means, counts


def validate_case(
    network: torch.nn.Module,
    case: str,
    data_dir: Path,
    device: torch.device,
) -> dict:
    """Recompute the paper metric and checkpoint-reproduction delta."""
    reference = json.loads((data_dir / f"{case}.json").read_text(encoding="utf-8"))
    predicted, counts = predict_heater_temperatures(network, case, reference, device)
    if counts != reference["prediction_points_per_heater"]:
        raise RuntimeError(
            f"{case}: heater-region point counts differ: "
            f"{counts} != {reference['prediction_points_per_heater']}"
        )
    comsol = np.asarray(reference["comsol_heater_temperature_K"], dtype=np.float64)
    archived_prediction = np.asarray(
        reference["prediction_heater_temperature_K"], dtype=np.float64
    )
    absolute_error = np.abs(predicted - comsol)
    return {
        "case": case,
        "heaters": len(comsol),
        "points_per_heater": counts,
        "average_absolute_error_K": float(absolute_error.mean()),
        "mape_percent": float(np.mean(absolute_error / comsol) * 100),
        "maximum_archived_prediction_delta_K": float(
            np.max(np.abs(predicted - archived_prediction))
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare the released checkpoint with compact COMSOL references"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "thermophoton.pt",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT / "data" / "validation_cases"
    )
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="auto uses CUDA when available",
    )
    parser.add_argument(
        "--prediction-tolerance-k",
        type=float,
        default=5e-3,
        help="maximum allowed delta from the archived checkpoint predictions",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    network = load_checkpoint(args.checkpoint, device)

    print(f"device: {device}")
    print("case           heaters   avg |error| (K)   MAPE (%)   archived delta (K)")
    print("-------------  -------   ---------------   --------   ------------------")
    results = []
    for case in args.cases:
        result = validate_case(network, case, args.data_dir, device)
        results.append(result)
        print(
            f"{case:13}  {result['heaters']:7d}   "
            f"{result['average_absolute_error_K']:15.6f}   "
            f"{result['mape_percent']:8.6f}   "
            f"{result['maximum_archived_prediction_delta_K']:18.6g}"
        )

    maximum_delta = max(result["maximum_archived_prediction_delta_K"] for result in results)
    if maximum_delta > args.prediction_tolerance_k:
        print(
            f"FAIL: checkpoint predictions differ from the archived values by "
            f"{maximum_delta:.6g} K (tolerance {args.prediction_tolerance_k:g} K)."
        )
        return 1
    print("PASS: checkpoint inference reproduces the archived validation predictions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
