from __future__ import annotations

import ast
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class ReleaseTests(unittest.TestCase):
    def test_python_sources_parse(self) -> None:
        for path in ROOT.rglob("*.py"):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_repository_is_minimal(self) -> None:
        markdown = sorted(
            path.relative_to(ROOT).as_posix()
            for path in ROOT.rglob("*.md")
            if not any(part.startswith(".") for part in path.relative_to(ROOT).parts)
        )
        self.assertEqual(markdown, ["README.md"])
        for removed in ("cluster", "docs", "experiments", "reports", "tools", "variants"):
            self.assertFalse(any(path.is_file() for path in (ROOT / removed).rglob("*")))

    def test_released_artifacts(self) -> None:
        self.assertGreater((ROOT / "checkpoints" / "thermophoton.pt").stat().st_size, 40_000_000)
        self.assertTrue((ROOT / "data" / "comsol_ground_truth.7z").is_file())

    def test_pytorch_runtime_and_network(self) -> None:
        import torch
        from thermophoton.network import (
            FourierTrunk,
            ThermoPhotonNet,
            TransformerBranch,
            create_heat_source_from_config,
        )

        branch = TransformerBranch(output_dimension=16, grid_size=32).eval()
        trunk = FourierTrunk(input_dimension=3, output_dimension=16).eval()
        model = ThermoPhotonNet(grid_size=32, latent_dimension=16).eval()
        configs = sorted((ROOT / "data").glob("*.json"))
        self.assertEqual(
            [path.stem for path in configs],
            ["3x3_mrr", "3x3_mzi", "4x4_mzi", "random_blocks"],
        )
        for config in configs:
            heat_source = create_heat_source_from_config(config, 120)
            self.assertEqual(heat_source.shape, (120 * 120,))
            self.assertEqual(set(heat_source), {0.0, 1.0})
        with torch.no_grad():
            self.assertEqual(branch(torch.zeros(1, 32 * 32)).shape, (1, 16))
            self.assertEqual(trunk(torch.zeros(4, 3)).shape, (4, 16))
            self.assertEqual(
                model((torch.zeros(1, 32 * 32), torch.zeros(4, 3))).shape,
                (1, 4),
            )

    def test_top_level_help(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "main.py"), "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertNotIn("--train", completed.stdout)
        self.assertNotIn("--epochs", completed.stdout)
        self.assertIn("--config", completed.stdout)
        self.assertNotIn("--case", completed.stdout)

    def test_missing_checkpoint_is_reported(self) -> None:
        missing = ROOT / "checkpoint-that-does-not-exist.pt"
        completed = subprocess.run(
            [sys.executable, str(ROOT / "main.py"), "--checkpoint", str(missing)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn(f"checkpoint not found: {missing}", completed.stderr)

if __name__ == "__main__":
    unittest.main()
