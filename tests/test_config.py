"""
Tests for FlyLLM config — auto-detection and prompt formatting.
"""

import os
import sys
import json
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flyllm.config import (
    load_config, format_prompt, get_model_name,
    get_flyllm_dir, FLYLLM_HOME, ModelConfig
)


def _make_fake_config(tmpdir: str, model_type: str = "mistral",
                      num_layers: int = 32, **kwargs):
    """Write a fake config.json to tmpdir."""
    cfg = {
        "model_type": model_type,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_hidden_layers": num_layers,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "max_position_embeddings": 32768,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "vocab_size": 32000,
        "bos_token_id": 1,
        "eos_token_id": 2,
        **kwargs,
    }
    with open(os.path.join(tmpdir, "config.json"), "w") as f:
        json.dump(cfg, f)
    return cfg


class TestLoadConfig(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_load_mistral_config(self):
        _make_fake_config(self.tmpdir, "mistral")
        cfg = load_config(self.tmpdir)
        self.assertEqual(cfg.model_type, "mistral")
        self.assertEqual(cfg.hidden_size, 4096)
        self.assertEqual(cfg.num_hidden_layers, 32)
        self.assertEqual(cfg.num_key_value_heads, 8)
        self.assertEqual(cfg.chat_template, "mistral")

    def test_load_llama_config(self):
        _make_fake_config(self.tmpdir, "llama", rope_theta=500000.0)
        cfg = load_config(self.tmpdir)
        self.assertEqual(cfg.model_type, "llama")
        self.assertEqual(cfg.chat_template, "llama3")
        self.assertEqual(cfg.rope_theta, 500000.0)

    def test_load_phi_config(self):
        _make_fake_config(self.tmpdir, "phi")
        cfg = load_config(self.tmpdir)
        self.assertEqual(cfg.model_type, "phi")
        self.assertEqual(cfg.chat_template, "phi")

    def test_load_qwen2_config(self):
        _make_fake_config(self.tmpdir, "qwen2")
        cfg = load_config(self.tmpdir)
        self.assertEqual(cfg.model_type, "qwen2")
        self.assertEqual(cfg.chat_template, "qwen2")

    def test_unknown_model_type_gets_default_template(self):
        _make_fake_config(self.tmpdir, "unknown_model_xyz")
        cfg = load_config(self.tmpdir)
        self.assertEqual(cfg.chat_template, "default")

    def test_gqa_fallback(self):
        """If num_key_value_heads missing, should use num_attention_heads."""
        cfg_data = {
            "model_type": "mistral",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            # no num_key_value_heads
            "vocab_size": 32000,
        }
        with open(os.path.join(self.tmpdir, "config.json"), "w") as f:
            json.dump(cfg_data, f)
        cfg = load_config(self.tmpdir)
        self.assertEqual(cfg.num_key_value_heads, 32)  # fallback to NH

    def test_returns_model_config_dataclass(self):
        _make_fake_config(self.tmpdir, "mistral")
        cfg = load_config(self.tmpdir)
        self.assertIsInstance(cfg, ModelConfig)

    def test_eos_bos_token_ids(self):
        _make_fake_config(self.tmpdir, "mistral",
                          bos_token_id=1, eos_token_id=2)
        cfg = load_config(self.tmpdir)
        self.assertEqual(cfg.bos_token_id, 1)
        self.assertEqual(cfg.eos_token_id, 2)

    def test_missing_config_raises(self):
        empty_dir = tempfile.mkdtemp()
        with self.assertRaises((ValueError, Exception)):
            load_config(empty_dir)


class TestFormatPrompt(unittest.TestCase):

    def _make_cfg(self, template: str) -> ModelConfig:
        return ModelConfig(
            model_type="mistral", hidden_size=4096, intermediate_size=14336,
            num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=8,
            max_position_embeddings=32768, rms_norm_eps=1e-5, rope_theta=10000.0,
            vocab_size=32000, chat_template=template,
        )

    def test_mistral_single_turn(self):
        cfg  = self._make_cfg("mistral")
        msgs = [{"role": "user", "content": "Hello"}]
        out  = format_prompt(msgs, cfg)
        self.assertIn("Hello", out)
        self.assertIn("[INST]", out)
        self.assertIn("[/INST]", out)

    def test_mistral_multi_turn(self):
        cfg  = self._make_cfg("mistral")
        msgs = [
            {"role": "user",      "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user",      "content": "How are you?"},
        ]
        out = format_prompt(msgs, cfg)
        self.assertIn("Hi", out)
        self.assertIn("Hello!", out)
        self.assertIn("How are you?", out)

    def test_system_prompt_included(self):
        cfg  = self._make_cfg("mistral")
        msgs = [{"role": "user", "content": "Hello"}]
        out  = format_prompt(msgs, cfg, system="You are a helpful assistant.")
        self.assertIn("You are a helpful assistant.", out)

    def test_no_system_prompt(self):
        cfg  = self._make_cfg("mistral")
        msgs = [{"role": "user", "content": "Hello"}]
        out  = format_prompt(msgs, cfg, system=None)
        self.assertNotIn("You are", out)

    def test_llama3_format(self):
        cfg  = self._make_cfg("llama3")
        msgs = [{"role": "user", "content": "What is AI?"}]
        out  = format_prompt(msgs, cfg)
        self.assertIn("What is AI?", out)
        self.assertIn("user", out.lower())

    def test_default_format(self):
        cfg  = self._make_cfg("default")
        msgs = [{"role": "user", "content": "Hello"}]
        out  = format_prompt(msgs, cfg)
        self.assertIn("Hello", out)
        self.assertIn("User:", out)

    def test_empty_messages(self):
        cfg = self._make_cfg("mistral")
        out = format_prompt([], cfg)
        self.assertEqual(out, "")

    def test_last_message_no_assistant_response(self):
        """Last user message should use last_turn template (no assistant response)."""
        cfg  = self._make_cfg("mistral")
        msgs = [{"role": "user", "content": "Final question"}]
        out  = format_prompt(msgs, cfg)
        # Should end with [/INST] waiting for response
        self.assertIn("[/INST]", out)
        # Should NOT contain assistant response after [/INST]
        after_inst = out.split("[/INST]")[-1]
        self.assertEqual(after_inst.strip(), "")


class TestHelperFunctions(unittest.TestCase):

    def test_get_model_name_slash(self):
        self.assertEqual(get_model_name("mistralai/Mistral-7B-v0.1"), "Mistral-7B-v0.1")

    def test_get_model_name_no_slash(self):
        self.assertEqual(get_model_name("Mistral-7B-v0.1"), "Mistral-7B-v0.1")

    def test_get_model_name_deep_path(self):
        self.assertEqual(get_model_name("org/sub/model-name"), "model-name")

    def test_get_flyllm_dir(self):
        path = get_flyllm_dir("mistralai/Mistral-7B-v0.1")
        self.assertTrue(path.endswith("Mistral-7B-v0.1"))
        self.assertIn("flyllmmodel", path)

    def test_get_flyllm_dir_is_absolute(self):
        path = get_flyllm_dir("mistralai/Mistral-7B-v0.1")
        self.assertTrue(os.path.isabs(path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
