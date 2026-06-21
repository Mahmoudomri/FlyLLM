"""
Tests for FlyLLM Mistral engine.
Tests dequantize, layer forward, and generation without needing real model weights.
"""

import os
import sys
import math
import torch
import tempfile
import unittest
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flyllm.quantizer import quantize_tensor, dequantize_tensor
from flyllm.engines.mistral import (
    _dequantize_layer, _build_rope, _apply_rope, _rms_norm, _layer_fwd
)
from flyllm.config import ModelConfig

# Tiny config for fast tests
TINY_CFG = ModelConfig(
    model_type              = "mistral",
    hidden_size             = 64,
    intermediate_size       = 128,
    num_hidden_layers       = 2,
    num_attention_heads     = 4,
    num_key_value_heads     = 2,
    max_position_embeddings = 512,
    rms_norm_eps            = 1e-5,
    rope_theta              = 10000.0,
    vocab_size              = 1000,
    chat_template           = "mistral",
    bos_token_id            = 1,
    eos_token_id            = 2,
)


def _make_tiny_layer_file(path: str, layer_idx: int,
                           cfg: ModelConfig, precision: str = "int4"):
    """Create a fake quantized layer safetensors file."""
    NH  = cfg.num_attention_heads
    NKV = cfg.num_key_value_heads
    H   = cfg.hidden_size
    IS  = cfg.intermediate_size
    HD  = H // NH

    raw = {
        f"model.layers.{layer_idx}.input_layernorm.weight":          torch.ones(H),
        f"model.layers.{layer_idx}.post_attention_layernorm.weight":  torch.ones(H),
        f"model.layers.{layer_idx}.self_attn.q_proj.weight":          torch.randn(NH*HD, H) * 0.02,
        f"model.layers.{layer_idx}.self_attn.k_proj.weight":          torch.randn(NKV*HD, H) * 0.02,
        f"model.layers.{layer_idx}.self_attn.v_proj.weight":          torch.randn(NKV*HD, H) * 0.02,
        f"model.layers.{layer_idx}.self_attn.o_proj.weight":          torch.randn(H, NH*HD) * 0.02,
        f"model.layers.{layer_idx}.mlp.gate_proj.weight":             torch.randn(IS, H) * 0.02,
        f"model.layers.{layer_idx}.mlp.up_proj.weight":               torch.randn(IS, H) * 0.02,
        f"model.layers.{layer_idx}.mlp.down_proj.weight":             torch.randn(H, IS) * 0.02,
    }

    # Quantize if needed
    out = {}
    for key, tensor in raw.items():
        q, meta = quantize_tensor(tensor, precision)
        out[key] = q.contiguous()
        if meta:
            out[f"{key}.__scales"] = meta["scales"].contiguous()
            out[f"{key}.__shape"]  = meta["shape"]
            out[f"{key}.__prec"]   = meta["prec"]

    save_file(out, path)
    return raw  # return originals for comparison


class TestDequantizeLayer(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_dequantize_int4_layer(self):
        """Dequantized layer should have same keys as original."""
        path    = os.path.join(self.tmpdir, "model.layers.0.safetensors")
        originals = _make_tiny_layer_file(path, 0, TINY_CFG, "int4")

        result = _dequantize_layer(path)

        for key in originals:
            self.assertIn(key, result)
            self.assertEqual(result[key].dtype, torch.float16)

    def test_dequantize_float16_layer(self):
        """float16 layer should be loaded exactly."""
        path = os.path.join(self.tmpdir, "model.layers.0.safetensors")
        _make_tiny_layer_file(path, 0, TINY_CFG, "float16")
        result = _dequantize_layer(path)
        # All values should be float16
        for v in result.values():
            self.assertEqual(v.dtype, torch.float16)

    def test_dequantize_shape_preserved(self):
        """Dequantized shapes should match original shapes."""
        path      = os.path.join(self.tmpdir, "model.layers.0.safetensors")
        originals = _make_tiny_layer_file(path, 0, TINY_CFG, "int4")
        result    = _dequantize_layer(path)

        for key in originals:
            self.assertEqual(
                result[key].shape, originals[key].shape,
                f"Shape mismatch for {key}"
            )

    def test_dequantize_cosine_quality(self):
        """Dequantized values should be close to originals (cosine > 0.99)."""
        import torch.nn.functional as F
        path      = os.path.join(self.tmpdir, "model.layers.0.safetensors")
        originals = _make_tiny_layer_file(path, 0, TINY_CFG, "int4")
        result    = _dequantize_layer(path)

        # Check q_proj weight
        key = "model.layers.0.self_attn.q_proj.weight"
        a   = originals[key].float().flatten()
        b   = result[key].float().flatten()
        cos = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
        self.assertGreater(cos, 0.99)


class TestRoPE(unittest.TestCase):

    def test_rope_output_shape(self):
        """RoPE cache should have shape [seq_len, head_dim/2]."""
        cos, sin = _build_rope(512, 16, 10000.0)
        self.assertEqual(cos.shape, (512, 8))
        self.assertEqual(sin.shape, (512, 8))

    def test_rope_values_unit_circle(self):
        """cos² + sin² should equal 1."""
        cos, sin = _build_rope(32, 8, 10000.0)
        norm = cos**2 + sin**2
        self.assertTrue(torch.allclose(norm, torch.ones_like(norm), atol=1e-4))

    def test_apply_rope_shape_preserved(self):
        """RoPE application should not change tensor shape."""
        B, NH, S, HD = 1, 4, 10, 16
        x            = torch.randn(B, NH, S, HD).half()
        cos, sin     = _build_rope(S, HD, 10000.0)
        out          = _apply_rope(x, cos, sin)
        self.assertEqual(out.shape, x.shape)

    def test_apply_rope_different_positions(self):
        """Different positions should produce different rotations."""
        x        = torch.ones(1, 1, 5, 8).half()
        cos, sin = _build_rope(5, 8, 10000.0)
        out      = _apply_rope(x, cos, sin)
        # Each position should be different
        self.assertFalse(torch.allclose(out[0, 0, 0], out[0, 0, 1]))


class TestRMSNorm(unittest.TestCase):

    def test_rms_norm_shape(self):
        """RMS norm should preserve shape."""
        x      = torch.randn(1, 10, 64).half()
        weight = torch.ones(64).half()
        out    = _rms_norm(x, weight, 1e-5)
        self.assertEqual(out.shape, x.shape)

    def test_rms_norm_normalizes(self):
        """RMS norm should normalize the values."""
        x      = torch.randn(1, 5, 64).half() * 100  # large values
        weight = torch.ones(64).half()
        out    = _rms_norm(x, weight, 1e-5)
        # Output should be roughly unit scale
        rms    = out.float().pow(2).mean(-1).sqrt()
        self.assertTrue((rms < 10).all())

    def test_rms_norm_weight_scaling(self):
        """Weight should scale the output."""
        x       = torch.randn(1, 5, 64).half()
        w1      = torch.ones(64).half()
        w2      = torch.ones(64).half() * 2.0
        out1    = _rms_norm(x, w1, 1e-5)
        out2    = _rms_norm(x, w2, 1e-5)
        self.assertTrue(torch.allclose(out2.float(), out1.float() * 2, atol=1e-2))


class TestLayerForward(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cfg    = TINY_CFG

        # Create fake weight dict (dequantized float16)
        NH  = self.cfg.num_attention_heads
        NKV = self.cfg.num_key_value_heads
        H   = self.cfg.hidden_size
        IS  = self.cfg.intermediate_size
        HD  = H // NH

        self.W = {
            "model.layers.0.input_layernorm.weight":         torch.ones(H).half(),
            "model.layers.0.post_attention_layernorm.weight":torch.ones(H).half(),
            "model.layers.0.self_attn.q_proj.weight":        torch.randn(NH*HD, H).half() * 0.02,
            "model.layers.0.self_attn.k_proj.weight":        torch.randn(NKV*HD, H).half() * 0.02,
            "model.layers.0.self_attn.v_proj.weight":        torch.randn(NKV*HD, H).half() * 0.02,
            "model.layers.0.self_attn.o_proj.weight":        torch.randn(H, NH*HD).half() * 0.02,
            "model.layers.0.mlp.gate_proj.weight":           torch.randn(IS, H).half() * 0.02,
            "model.layers.0.mlp.up_proj.weight":             torch.randn(IS, H).half() * 0.02,
            "model.layers.0.mlp.down_proj.weight":           torch.randn(H, IS).half() * 0.02,
        }

        HD_val = H // NH
        self.cos, self.sin = _build_rope(512, HD_val, self.cfg.rope_theta)

    def test_layer_fwd_output_shape(self):
        """Layer forward should preserve [B, S, H] shape."""
        h   = torch.randn(1, 5, self.cfg.hidden_size).half()
        out = _layer_fwd(h, self.W, self.cos, self.sin, self.cfg, "model.layers.0")
        self.assertEqual(out.shape, h.shape)

    def test_layer_fwd_output_dtype(self):
        """Layer forward should return float16."""
        h   = torch.randn(1, 5, self.cfg.hidden_size).half()
        out = _layer_fwd(h, self.W, self.cos, self.sin, self.cfg, "model.layers.0")
        self.assertEqual(out.dtype, torch.float16)

    def test_layer_fwd_batch_size_1(self):
        """Layer forward should work with batch size 1."""
        h   = torch.randn(1, 1, self.cfg.hidden_size).half()
        out = _layer_fwd(h, self.W, self.cos, self.sin, self.cfg, "model.layers.0")
        self.assertEqual(out.shape[0], 1)

    def test_layer_fwd_different_seq_lengths(self):
        """Layer forward should work with different sequence lengths."""
        for seq_len in [1, 5, 10, 20]:
            h   = torch.randn(1, seq_len, self.cfg.hidden_size).half()
            out = _layer_fwd(h, self.W, self.cos, self.sin, self.cfg, "model.layers.0")
            self.assertEqual(out.shape, (1, seq_len, self.cfg.hidden_size))

    def test_layer_fwd_residual_connection(self):
        """Output should be different from input (residual + transformation)."""
        h   = torch.randn(1, 5, self.cfg.hidden_size).half()
        out = _layer_fwd(h, self.W, self.cos, self.sin, self.cfg, "model.layers.0")
        # Output should not be identical to input
        self.assertFalse(torch.allclose(h, out))

    def test_layer_fwd_no_nan(self):
        """Layer forward should not produce NaN values."""
        h   = torch.randn(1, 5, self.cfg.hidden_size).half()
        out = _layer_fwd(h, self.W, self.cos, self.sin, self.cfg, "model.layers.0")
        self.assertFalse(torch.isnan(out).any())

    def test_layer_fwd_no_inf(self):
        """Layer forward should not produce Inf values."""
        h   = torch.randn(1, 5, self.cfg.hidden_size).half()
        out = _layer_fwd(h, self.W, self.cos, self.sin, self.cfg, "model.layers.0")
        self.assertFalse(torch.isinf(out).any())


class TestBaseEngineGeneration(unittest.TestCase):
    """Test the token generation loop in base engine."""

    def test_generate_stops_at_eos(self):
        """Generation should stop when EOS token is produced."""
        from flyllm.engines.base import BaseEngine
        import torch.nn.functional as F

        class FakeEngine(BaseEngine):
            def __init__(self):
                self.cfg = TINY_CFG
                self.call_count = 0

            def load(self): pass

            def forward(self, input_ids):
                self.call_count += 1
                # Always produce EOS token (id=2)
                logits = torch.zeros(1, input_ids.shape[1], TINY_CFG.vocab_size)
                logits[0, -1, 2] = 100.0  # EOS
                return logits

        eng    = FakeEngine()
        ids    = torch.tensor([[1, 10, 20]])  # start tokens
        tokens = list(eng.generate_tokens(ids, max_new_tokens=50, temperature=0.0))

        self.assertEqual(tokens[0], 2)   # first token is EOS
        self.assertEqual(len(tokens), 1)  # stopped immediately

    def test_generate_respects_max_tokens(self):
        """Generation should stop at max_new_tokens."""
        from flyllm.engines.base import BaseEngine

        class FakeEngine(BaseEngine):
            def __init__(self):
                self.cfg = TINY_CFG

            def load(self): pass

            def forward(self, input_ids):
                # Always produce token 42 (not EOS)
                logits = torch.zeros(1, input_ids.shape[1], TINY_CFG.vocab_size)
                logits[0, -1, 42] = 100.0
                return logits

        eng    = FakeEngine()
        ids    = torch.tensor([[1]])
        tokens = list(eng.generate_tokens(ids, max_new_tokens=10, temperature=0.0))
        self.assertEqual(len(tokens), 10)

    def test_generate_greedy_picks_argmax(self):
        """Greedy (temperature=0) should pick highest logit."""
        from flyllm.engines.base import BaseEngine

        class FakeEngine(BaseEngine):
            def __init__(self):
                self.cfg = TINY_CFG

            def load(self): pass

            def forward(self, input_ids):
                logits = torch.zeros(1, input_ids.shape[1], TINY_CFG.vocab_size)
                logits[0, -1, 99] = 10.0   # highest
                logits[0, -1, 2]  = -10.0  # EOS suppressed
                return logits

        eng    = FakeEngine()
        ids    = torch.tensor([[1]])
        tokens = list(eng.generate_tokens(ids, max_new_tokens=3, temperature=0.0))
        self.assertEqual(tokens[0], 99)


if __name__ == "__main__":
    unittest.main(verbosity=2)
