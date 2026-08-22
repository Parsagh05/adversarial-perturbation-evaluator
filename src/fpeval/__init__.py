"""Fixed-perturbation black-box anomaly-detection evaluation."""

from .config import EvaluationConfig
from .engine import evaluate

__all__ = ["EvaluationConfig", "evaluate"]
__version__ = "0.1.0"

