from setuptools import find_packages, setup

# Core dependencies (excluding PyTorch which should be installed separately)
core_requires = [
    "datasets>=2.0.0",
    "numpy>=1.21.0",
    "openai>=1.0.0",
    "packaging>=21.0",
    "pyyaml>=6.0.1",
    "tqdm>=4.65.0",
    "transformers>=4.30.0",
    "wandb>=0.15.0",
]

# Full dependencies including PyTorch
# Note: Install PyTorch separately for your device (CPU/CUDA/ROCm/MPS)
# See: https://pytorch.org/get-started/locally/
full_requires = core_requires + [
    "torch>=2.0.0",
]

# Development dependencies
dev_requires = [
    "black>=24.10.0",
    "pre-commit>=3.8.0",
]

# Optional dependencies for specific features
extras_require = {
    "full": full_requires,
    "dev": dev_requires,
    "all": full_requires + dev_requires,
}

setup(
    name="comlrl",
    use_scm_version=True,
    setup_requires=["setuptools-scm"],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=core_requires,
    extras_require=extras_require,
)
