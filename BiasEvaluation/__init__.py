"""
BiasEvaluation: bias analysis tooling for financial language models.
"""

from .core.models import (
    BERTModel,
    LlamaModelWrapper,
    load_bert_model,
    load_llama_model,
)
from .analysis.bias_analyzer import BiasAnalyzer
from .analysis.tokenShap import TokenSHAP
from .utils.data_loader import process_mutant_directory_multi_gpu
from .utils.helpers import build_full_prompt, jensen_shannon_distance

__all__ = [
    "BERTModel",
    "LlamaModelWrapper",
    "load_bert_model",
    "load_llama_model",
    "BiasAnalyzer",
    "TokenSHAP",
    "process_mutant_directory_multi_gpu",
    "build_full_prompt",
    "jensen_shannon_distance",
]
