from __future__ import annotations

import ast
import subprocess
import sys
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class ReleaseTests(unittest.TestCase):
    def test_python_sources_parse(self) -> None:
        for path in ROOT.rglob("*.py"):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_repository_is_minimal(self) -> None:
        markdown = sorted(path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*.md"))
        self.assertEqual(markdown, ["README.md"])
        for removed in ("cluster", "docs", "experiments", "reports", "tools", "variants"):
            self.assertFalse(any(path.is_file() for path in (ROOT / removed).rglob("*")))

    def test_released_validation_artifacts(self) -> None:
        import json

        cases = {"3x3_mrr", "3x3_mzi", "4x4_mzi", "random_blocks"}
        data_dir = ROOT / "data" / "validation_cases"
        data_files = {path.stem for path in data_dir.glob("*.json")}
        self.assertEqual(data_files, cases)
        for case in cases:
            record = json.loads(
                (data_dir / f"{case}.json").read_text(encoding="utf-8")
            )
            self.assertEqual(record["pattern"], case)
            self.assertEqual(
                len(record["comsol_heater_temperature_K"]),
                len(record["prediction_heater_temperature_K"]),
            )
        self.assertGreater((ROOT / "checkpoints" / "thermophoton.pt").stat().st_size, 40_000_000)

    def test_pytorch_runtime_and_network(self) -> None:
        import deepheat
        import torch
        from thermophoton.network import FourierTrunk, GRF2D, TransformerBranch

        self.assertEqual(deepheat.backend.backend_name, "pytorch")
        sampler = GRF2D(N=4)
        self.assertEqual(sampler.sample_ratio, 0.75)
        with mock.patch(
            "thermophoton.network.np.random.randint", return_value=0
        ) as randint:
            self.assertEqual(sampler._rectangles(), [])
        randint.assert_called_once_with(5, 20)
        branch = TransformerBranch(output_dimension=16, grid_size=32).eval()
        trunk = FourierTrunk(input_dimension=3, output_dimension=16).eval()
        with torch.no_grad():
            self.assertEqual(branch(torch.zeros(1, 32 * 32)).shape, (1, 16))
            self.assertEqual(trunk(torch.zeros(4, 3)).shape, (4, 16))

    def test_top_level_help(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "main.py"), "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("--train-new-model", completed.stdout)

        completed = subprocess.run(
            [sys.executable, str(ROOT / "validate.py"), "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("--prediction-tolerance-k", completed.stdout)


if __name__ == "__main__":
    unittest.main()
