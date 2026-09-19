"""Monkeyinference: ternary Hadamard inference for Bonsai 2 27B on Apple silicon."""

__version__ = "0.1.0"

from monkeyinference.generate import GenerateResult, generate
from monkeyinference.load import DEFAULT_PACK, load_text_model

__all__ = ["DEFAULT_PACK", "GenerateResult", "generate", "load_text_model", "__version__"]
