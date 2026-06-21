"""
Tests for FlyLLM profiler.
"""

import os
import sys
import json
import torch
import tempfile
import unittest
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flyllm.profiler import (
    _kurtosis, _entropy, _max_abs, _score_layer, profile_model
)


def _make_fake_layer(path: str, kurtosis_level: str = "normal"):
    """Create a fake layer safetensors file for testing."""
    if kurtosis_level == "high":
        # High kurtosis — many zeros, few large values (outliers)
        t = torch.zeros(4096, 4096)
        t[0, 0] = 5.0
        t[1, 1] = -4.0
    elif kurtosis_level == "normal":
        # Normal distribution — low kurtosis
        t = torch.randn(4096, 4096) * 0.03
    else:
        # Very uniform — minimal kurtosis
        t = torch.ones(4096, 4096) * 0.01

    tensors = {
        "model.layers.0.self_attn.q_proj.weight": t,
        "model.layers.0.self_attn.k_proj.weight": torch.randn(1024, 4096) * 0.03,
        "model.layers.0.mlp.gate_proj.weight":    torch.randn(14336, 4096) * 0.03,
    }
    save_file(tensors, path)


class TestMetrics(unittest.TestCase):

    def test_kurtosis_normal_distribution(self):
        """Normal distribution should have kurtosis close to 3."""
        t    = torch.randn(10000)
        kurt = _kurtosis(t)
        self.assertGreater(kurt, 2.0)
        self.assertLess(kurt, 6.0)

    def test_kurtosis_high_outliers(self):
        """Distribution with outliers should have high kurtosis."""
        t = torch.zeros(10000)
        t[0] = 100.0   # extreme outlier
        kurt = _kurtosis(t)
        self.assertGreater(kurt, 50.0)

    def test_entropy_uniform(self):
        """Uniform distribution should have high entropy."""
        t    = torch.linspace(-1, 1, 10000)
        entr = _entropy(t)
        self.assertGreater(entr, 0.8)

    def test_entropy_concentrated(self):
        """Concentrated distribution should have low entropy."""
        t    = torch.zeros(10000)
        entr = _entropy(t)
        self.assertLess(entr, 0.1)

    def test_max_abs(self):
        """Max abs should return the largest absolute value."""
        t = torch.tensor([-5.0, 3.0, 2.0, -1.0])
        self.assertAlmostEqual(_max_abs(t), 5.0)

    def test_max_abs_positive(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        self.assertAlmostEqual(_max_abs(t), 3.0)


class TestScoreLayer(unittest.TestCase):

    def test_high_kurtosis_gets_float16(self):
        """Layer with outliers should get float16."""
        tensors = {}
        t = torch.zeros(4096, 4096)
        t[0, 0] = 10.0  # big outlier
        tensors["model.layers.0.q_proj.weight"] = t
        result = _score_layer(tensors)
        self.assertEqual(result["precision"], "float16")

    def test_normal_layer_gets_int4(self):
        """Normal layer should get int4."""
        tensors = {}
        t = torch.randn(4096, 4096) * 0.003  # small normal values
        tensors["model.layers.5.q_proj.weight"] = t
        result = _score_layer(tensors)
        self.assertIn(result["precision"], ["int4", "int8"])

    def test_empty_layer(self):
        """Empty tensors dict should return int4 with default values."""
        result = _score_layer({})
        self.assertEqual(result["precision"], "int4")
        self.assertEqual(result["kurtosis"], 3.0)

    def test_1d_tensors_ignored(self):
        """1D tensors (biases) should be ignored in scoring."""
        tensors = {
            "model.layers.0.bias": torch.randn(4096),  # 1D — ignored
            "model.layers.0.weight": torch.randn(4096, 4096) * 0.003,
        }
        result = _score_layer(tensors)
        self.assertIn("precision", result)
        self.assertIn("score", result)

    def test_score_between_0_and_1(self):
        """Score should always be in [0, 1]."""
        tensors = {"w": torch.randn(4096, 4096) * 0.03}
        result  = _score_layer(tensors)
        self.assertGreaterEqual(result["score"], 0.0)
        self.assertLessEqual(result["score"], 1.0)

    def test_result_has_all_keys(self):
        """Result should always have all expected keys."""
        tensors = {"w": torch.randn(512, 512)}
        result  = _score_layer(tensors)
        for key in ["kurtosis", "entropy", "max_abs", "score", "precision"]:
            self.assertIn(key, result)


class TestProfileModel(unittest.TestCase):

    def setUp(self):
        """Create a fake model directory."""
        self.tmpdir  = tempfile.mkdtemp()
        self.outdir  = tempfile.mkdtemp()

        # Create fake layer files
        for i in range(4):
            path = os.path.join(self.tmpdir, f"model.layers.{i}.safetensors")
            if i == 0 or i == 3:
                _make_fake_layer(path, "high")
            else:
                _make_fake_layer(path, "normal")

    def test_profile_returns_correct_structure(self):
        """Profile should return dict with layers and summary."""
        from unittest.mock import patch

        # Mock load_config to return 4 layers
        with patch("flyllm.profiler.load_config") as mock_cfg:
            mock_cfg.return_value = type("cfg", (), {
                "model_type": "mistral",
                "num_hidden_layers": 4,
            })()
            result = profile_model(
                "test/model",
                hf_dir=self.tmpdir,
                verbose=False,
            )

        self.assertIn("layers", result)
        self.assertIn("summary", result)
        self.assertEqual(len(result["layers"]), 4)

    def test_profile_saves_json(self):
        """Profile should save JSON when output_path given."""
        from unittest.mock import patch
        out_path = os.path.join(self.outdir, "profile.json")

        with patch("flyllm.profiler.load_config") as mock_cfg:
            mock_cfg.return_value = type("cfg", (), {
                "model_type": "mistral",
                "num_hidden_layers": 4,
            })()
            profile_model(
                "test/model",
                hf_dir=self.tmpdir,
                output_path=out_path,
                verbose=False,
            )

        self.assertTrue(os.path.exists(out_path))
        with open(out_path) as f:
            data = json.load(f)
        self.assertIn("layers", data)
        self.assertIn("summary", data)

    def test_summary_has_correct_keys(self):
        """Summary should have all expected keys."""
        from unittest.mock import patch

        with patch("flyllm.profiler.load_config") as mock_cfg:
            mock_cfg.return_value = type("cfg", (), {
                "model_type": "mistral",
                "num_hidden_layers": 4,
            })()
            result = profile_model(
                "test/model",
                hf_dir=self.tmpdir,
                verbose=False,
            )

        summary = result["summary"]
        for key in ["avg_bits", "size_reduction", "float16_layers",
                    "int8_layers", "int4_layers"]:
            self.assertIn(key, summary)

    def test_avg_bits_in_range(self):
        """Avg bits should be between 4 and 16."""
        from unittest.mock import patch

        with patch("flyllm.profiler.load_config") as mock_cfg:
            mock_cfg.return_value = type("cfg", (), {
                "model_type": "mistral",
                "num_hidden_layers": 4,
            })()
            result = profile_model(
                "test/model",
                hf_dir=self.tmpdir,
                verbose=False,
            )

        self.assertGreaterEqual(result["summary"]["avg_bits"], 4.0)
        self.assertLessEqual(result["summary"]["avg_bits"], 16.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
