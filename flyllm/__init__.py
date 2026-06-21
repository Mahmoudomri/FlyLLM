"""
FlyLLM — Adaptive Quantization for Local LLMs

Usage:
    from flyllm import FlyLLM
    model = FlyLLM.from_pretrained("mistralai/Mistral-7B-v0.1")
    print(model.generate("What is AI?"))
"""

from .loader import FlyLLM
from .profiler import profile_model
from .quantizer import quantize_model
from .config import load_config

__version__ = "1.0.0"
__all__     = ["FlyLLM", "profile_model", "quantize_model", "load_config"]
