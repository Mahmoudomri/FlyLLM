"""
FlyLLM - Engine Router
"""
from ..config import ModelConfig


def get_engine(flyllm_dir: str, hf_dir: str, cfg: ModelConfig, verbose: bool = True):
    """One engine, works for every model architecture."""
    from .hf_engine import HFEngine
    engine = HFEngine(flyllm_dir, hf_dir, cfg, verbose)
    engine.load()
    return engine
