"""
FlyLLM - Engine Router
"""
from ..config import ModelConfig, CUSTOM_ENGINE_MODELS


def get_engine(flyllm_dir: str, hf_dir: str, cfg: ModelConfig, verbose: bool = True):
    """Pick the right engine and return it loaded."""
    if cfg.model_type == "mistral":
        from .mistral import MistralEngine
        engine = MistralEngine(flyllm_dir, hf_dir, cfg, verbose)
    elif cfg.model_type == "llama":
        from .llama import LlamaEngine
        engine = LlamaEngine(flyllm_dir, hf_dir, cfg, verbose)
    else:
        from .hf_engine import HFEngine
        engine = HFEngine(flyllm_dir, hf_dir, cfg, verbose)

    engine.load()
    return engine
