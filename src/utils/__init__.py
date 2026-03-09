"""Utilities sub-package."""

from .evaluation import evaluate
from .outage_validator import OutageValidator, make_synthetic_dataset
from .logger import StructuredLogger, get_logger
from .metrics import MetricsRegistry, GLOBAL_REGISTRY
from .model_registry import ModelRegistry

# torch-dependent modules: imported conditionally so that the rest of the
# package remains usable in environments without PyTorch.
try:
    from .explainability import (
        vanilla_saliency,
        integrated_gradients,
        smooth_grad,
        feature_importance_summary,
    )
    _EXPLAINABILITY_AVAILABLE = True
except ImportError:
    _EXPLAINABILITY_AVAILABLE = False

__all__ = [
    "evaluate",
    "OutageValidator",
    "make_synthetic_dataset",
    "StructuredLogger",
    "get_logger",
    "MetricsRegistry",
    "GLOBAL_REGISTRY",
    "ModelRegistry",
    # explainability symbols available only when torch is installed
    "vanilla_saliency",
    "integrated_gradients",
    "smooth_grad",
    "feature_importance_summary",
]
