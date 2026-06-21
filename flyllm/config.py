"""
FlyLLM - Auto Model Configuration
Detects any HuggingFace model architecture automatically.
"""

import os
import json
from dataclasses import dataclass
from typing import Optional


FLYLLM_HOME = os.path.expanduser("~/flyllmmodel")

CHAT_TEMPLATES = {
    "mistral": {
        "system":    "[INST] {system}\n\n{user} [/INST]",
        "turn":      "<s>[INST] {user} [/INST] {assistant}</s>",
        "last_turn": "<s>[INST] {user} [/INST]",
    },
    "llama3": {
        "system":    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>",
        "turn":      "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{assistant}<|eot_id|>",
        "last_turn": "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    },
    "phi": {
        "system":    "<|system|>\n{system}<|end|>\n",
        "turn":      "<|user|>\n{user}<|end|>\n<|assistant|>\n{assistant}<|end|>\n",
        "last_turn": "<|user|>\n{user}<|end|>\n<|assistant|>\n",
    },
    "qwen2": {
        "system":    "<|im_start|>system\n{system}<|im_end|>\n",
        "turn":      "<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant}<|im_end|>\n",
        "last_turn": "<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n",
    },
    "default": {
        "system":    "System: {system}\n\n",
        "turn":      "User: {user}\nAssistant: {assistant}\n",
        "last_turn": "User: {user}\nAssistant: ",
    },
}

MODEL_TYPE_TO_TEMPLATE = {
    "mistral": "mistral",
    "llama":   "llama3",
    "phi":     "phi",
    "phi3":    "phi",
    "qwen2":   "qwen2",
    "falcon":  "default",
}

CUSTOM_ENGINE_MODELS = {"mistral", "llama"}


@dataclass
class ModelConfig:
    model_type:              str
    hidden_size:             int
    intermediate_size:       int
    num_hidden_layers:       int
    num_attention_heads:     int
    num_key_value_heads:     int
    max_position_embeddings: int
    rms_norm_eps:            float
    rope_theta:              float
    vocab_size:              int
    chat_template:           str
    bos_token_id:            int = 1
    eos_token_id:            int = 2


def load_config(model_id_or_path: str) -> ModelConfig:
    """Load and auto-detect model config from HF hub or local path."""
    local = os.path.join(model_id_or_path, "config.json")
    if os.path.exists(local):
        with open(local) as f:
            hf_cfg = json.load(f)
    else:
        from huggingface_hub import hf_hub_download
        try:
            path = hf_hub_download(model_id_or_path, "config.json")
            with open(path) as f:
                hf_cfg = json.load(f)
        except Exception as e:
            raise ValueError(f"Cannot load config for '{model_id_or_path}': {e}")

    model_type = hf_cfg.get("model_type", "unknown").lower()
    template   = MODEL_TYPE_TO_TEMPLATE.get(model_type, "default")
    num_kv     = hf_cfg.get("num_key_value_heads",
                             hf_cfg.get("num_attention_heads", 32))

    return ModelConfig(
        model_type              = model_type,
        hidden_size             = hf_cfg.get("hidden_size", 4096),
        intermediate_size       = hf_cfg.get("intermediate_size", 11008),
        num_hidden_layers       = hf_cfg.get("num_hidden_layers", 32),
        num_attention_heads     = hf_cfg.get("num_attention_heads", 32),
        num_key_value_heads     = num_kv,
        max_position_embeddings = hf_cfg.get("max_position_embeddings", 4096),
        rms_norm_eps            = hf_cfg.get("rms_norm_eps", 1e-5),
        rope_theta              = hf_cfg.get("rope_theta", 10000.0),
        vocab_size              = hf_cfg.get("vocab_size", 32000),
        chat_template           = template,
        bos_token_id            = hf_cfg.get("bos_token_id", 1),
        eos_token_id            = hf_cfg.get("eos_token_id", 2),
    )


def format_prompt(messages: list, config: ModelConfig,
                  system: Optional[str] = None) -> str:
    """Format conversation into model's expected prompt format."""
    tmpl   = CHAT_TEMPLATES.get(config.chat_template, CHAT_TEMPLATES["default"])
    result = ""

    if system:
        result += tmpl["system"].format(system=system)

    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg["role"] == "user":
            if i + 1 < len(messages) and messages[i+1]["role"] == "assistant":
                result += tmpl["turn"].format(
                    user=msg["content"],
                    assistant=messages[i+1]["content"],
                    system=system or "",
                )
                i += 2
            else:
                result += tmpl["last_turn"].format(
                    user=msg["content"],
                    system=system or "",
                )
                i += 1
        else:
            i += 1

    return result


def get_model_name(model_id: str) -> str:
    """Get clean folder name from model ID. 'mistralai/Mistral-7B-v0.1' → 'Mistral-7B-v0.1'"""
    return model_id.split("/")[-1]


def get_flyllm_dir(model_id: str) -> str:
    """Get path where compressed layers are stored."""
    return os.path.join(FLYLLM_HOME, get_model_name(model_id))


def get_hf_cache_dir(model_id: str) -> Optional[str]:
    """
    Find the HuggingFace cache directory for a model.
    Returns the splitted_model path if it exists, else None.
    """
    hf_home    = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hub_dir    = os.path.join(hf_home, "hub")
    model_name = model_id.replace("/", "--")
    model_dir  = os.path.join(hub_dir, f"models--{model_name}")

    if not os.path.isdir(model_dir):
        return None

    # Find the snapshot with splitted_model
    snapshots_dir = os.path.join(model_dir, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return None

    for snapshot in os.listdir(snapshots_dir):
        split_path = os.path.join(snapshots_dir, snapshot, "splitted_model")
        if os.path.isdir(split_path):
            layer0 = os.path.join(split_path, "model.layers.0.safetensors")
            if os.path.exists(layer0):
                return split_path

    return None
