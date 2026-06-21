from setuptools import setup, find_packages

setup(
    name="flyllm",
    version="1.0.0",
    author="Mahmoud",
    description="Adaptive quantization for local LLMs — Kurtosis+Entropy+MaxAbs per-layer analysis",
    
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "safetensors>=0.4.0",
        "huggingface_hub>=0.19.0",
        "numpy>=1.24.0",
        "airllm>=0.1.0",
    ],
    entry_points={"console_scripts": ["flyllm=flyllm.cli:main"]},
    python_requires=">=3.9",
)
