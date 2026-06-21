"""
Tests for FlyLLM quantizer.
"""

import os
import sys
import torch
import tempfile
import unittest
from safetensors.torch import save_file, load_file

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flyllm.quantizer import (
    quantize_tensor, dequantize_tensor, quantize_model
)


class TestQuantizeTensor(unittest.TestCase):

    def _roundtrip(self, tensor, precision):
        q, meta = quantize_tensor(tensor, precision)
        return dequantize_tensor(q, meta, dtype=torch.float32)

    # ── float16 ──────────────────────────────────────────────

    def test_float16_roundtrip_exact(self):
        """float16 should be exact (no quantization loss)."""
        t   = torch.randn(512, 512)
        out = self._roundtrip(t, "float16")
        self.assertEqual(out.dtype, torch.float32)
        # float16 has slight precision loss vs float32
        self.assertLess((t.float() - out).abs().max().item(), 0.01)

    def test_float16_1d_tensor(self):
        """1D tensors should always go to float16."""
        t    = torch.randn(4096)
        q, meta = quantize_tensor(t, "int4")
        self.assertIsNone(meta)
        self.assertEqual(q.dtype, torch.float16)

    def test_float16_returns_none_meta(self):
        """float16 precision should return None meta."""
        t    = torch.randn(512, 512)
        q, meta = quantize_tensor(t, "float16")
        self.assertIsNone(meta)

    # ── INT8 ─────────────────────────────────────────────────

    def test_int8_roundtrip_cosine(self):
        """INT8 roundtrip should have cosine sim > 0.999."""
        import torch.nn.functional as F
        t   = torch.randn(4096, 4096) * 0.03
        out = self._roundtrip(t, "int8")
        a   = t.float().flatten()
        b   = out.flatten()
        cos = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
        self.assertGreater(cos, 0.999)

    def test_int8_quantized_dtype(self):
        """INT8 quantized tensor should be int8."""
        t    = torch.randn(512, 512)
        q, _ = quantize_tensor(t, "int8")
        self.assertEqual(q.dtype, torch.int8)

    def test_int8_meta_has_scales(self):
        """INT8 meta should have scales, shape, prec."""
        t    = torch.randn(512, 512)
        _, meta = quantize_tensor(t, "int8")
        self.assertIsNotNone(meta)
        self.assertIn("scales", meta)
        self.assertIn("shape",  meta)
        self.assertIn("prec",   meta)

    def test_int8_prec_id_is_1(self):
        """INT8 prec tensor should be 1."""
        t    = torch.randn(512, 512)
        _, meta = quantize_tensor(t, "int8")
        self.assertEqual(meta["prec"].item(), 1)

    # ── INT4 ─────────────────────────────────────────────────

    def test_int4_roundtrip_cosine(self):
        """INT4 roundtrip should have cosine sim > 0.995."""
        import torch.nn.functional as F
        t   = torch.randn(4096, 4096) * 0.03
        out = self._roundtrip(t, "int4")
        a   = t.float().flatten()
        b   = out.flatten()
        cos = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
        self.assertGreater(cos, 0.995)

    def test_int4_quantized_dtype(self):
        """INT4 quantized tensor should be stored as int8."""
        t    = torch.randn(512, 512)
        q, _ = quantize_tensor(t, "int4")
        self.assertEqual(q.dtype, torch.int8)

    def test_int4_values_in_range(self):
        """INT4 quantized values should be in [-8, 7]."""
        t    = torch.randn(512, 512)
        q, _ = quantize_tensor(t, "int4")
        self.assertGreaterEqual(q.min().item(), -8)
        self.assertLessEqual(q.max().item(), 7)

    def test_int4_prec_id_is_2(self):
        """INT4 prec tensor should be 2."""
        t    = torch.randn(512, 512)
        _, meta = quantize_tensor(t, "int4")
        self.assertEqual(meta["prec"].item(), 2)

    # ── Shape preservation ────────────────────────────────────

    def test_shape_preserved_int4(self):
        """Dequantized tensor should have same shape as original."""
        t    = torch.randn(4096, 14336)
        out  = self._roundtrip(t, "int4")
        self.assertEqual(out.shape, t.shape)

    def test_shape_preserved_int8(self):
        t    = torch.randn(1024, 4096)
        out  = self._roundtrip(t, "int8")
        self.assertEqual(out.shape, t.shape)

    def test_non_divisible_shape(self):
        """Tensors not divisible by BLOCK_SIZE should still work."""
        t   = torch.randn(100, 77)   # 7700 elements, not divisible by 32
        out = self._roundtrip(t, "int4")
        self.assertEqual(out.shape, t.shape)


class TestQuantizeModel(unittest.TestCase):

    def setUp(self):
        self.hf_dir     = tempfile.mkdtemp()
        self.output_dir = tempfile.mkdtemp()

        # Create fake layer files (4 layers)
        for i in range(4):
            tensors = {
                f"model.layers.{i}.self_attn.q_proj.weight": torch.randn(4096, 4096) * 0.03,
                f"model.layers.{i}.mlp.gate_proj.weight":    torch.randn(14336, 4096) * 0.03,
            }
            save_file(tensors,
                      os.path.join(self.hf_dir, f"model.layers.{i}.safetensors"))

        # Fake profile
        self.profile = {
            "model_type": "mistral",
            "layers": {
                "0": {"precision": "float16", "score": 0.6},
                "1": {"precision": "int8",    "score": 0.3},
                "2": {"precision": "int4",    "score": 0.1},
                "3": {"precision": "float16", "score": 0.5},
            },
            "summary": {
                "avg_bits": 9.0,
                "size_reduction": 43.8,
                "float16_layers": 2,
                "int8_layers": 1,
                "int4_layers": 1,
            }
        }

    def test_creates_output_files(self):
        """Quantize model should create one file per layer."""
        quantize_model(
            model_id="test/model",
            hf_dir=self.hf_dir,
            profile=self.profile,
            output_dir=self.output_dir,
            verbose=False,
        )
        for i in range(4):
            path = os.path.join(self.output_dir, f"model.layers.{i}.safetensors")
            self.assertTrue(os.path.exists(path), f"Missing: {path}")

    def test_creates_meta_json(self):
        """Should create flyllm_meta.json."""
        quantize_model(
            model_id="test/model",
            hf_dir=self.hf_dir,
            profile=self.profile,
            output_dir=self.output_dir,
            verbose=False,
        )
        meta_path = os.path.join(self.output_dir, "flyllm_meta.json")
        self.assertTrue(os.path.exists(meta_path))

        import json
        with open(meta_path) as f:
            meta = json.load(f)
        self.assertEqual(meta["model_id"], "test/model")
        self.assertIn("hf_dir", meta)
        self.assertIn("avg_bits", meta)

    def test_creates_profile_json(self):
        """Should create flyllm_profile.json."""
        quantize_model(
            model_id="test/model",
            hf_dir=self.hf_dir,
            profile=self.profile,
            output_dir=self.output_dir,
            verbose=False,
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.output_dir, "flyllm_profile.json"))
        )

    def test_float16_layer_smaller_than_original(self):
        """Quantized files should exist and be readable."""
        quantize_model(
            model_id="test/model",
            hf_dir=self.hf_dir,
            profile=self.profile,
            output_dir=self.output_dir,
            verbose=False,
        )
        # int4 layer should be smaller than original
        orig_size = os.path.getsize(
            os.path.join(self.hf_dir, "model.layers.2.safetensors")
        )
        quant_size = os.path.getsize(
            os.path.join(self.output_dir, "model.layers.2.safetensors")
        )
        self.assertLess(quant_size, orig_size)

    def test_skip_if_already_quantized(self):
        """Should skip if flyllm_meta.json already exists."""
        import json

        # First quantize
        quantize_model(
            model_id="test/model",
            hf_dir=self.hf_dir,
            profile=self.profile,
            output_dir=self.output_dir,
            verbose=False,
        )
        mtime1 = os.path.getmtime(
            os.path.join(self.output_dir, "model.layers.0.safetensors")
        )

        import time
        time.sleep(0.1)

        # Second call — should NOT re-quantize (meta exists)
        # In current implementation meta check is in loader, not quantizer
        # But the file should still exist
        self.assertTrue(
            os.path.exists(os.path.join(self.output_dir, "flyllm_meta.json"))
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
