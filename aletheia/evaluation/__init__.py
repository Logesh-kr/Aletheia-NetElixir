"""
aletheia.evaluation
===================
Evaluation utilities for the Aletheia ML pipeline.

Public API
----------
    from aletheia.evaluation.metrics import RegressionEvaluator, RegressionMetrics
    from aletheia.evaluation.plots import RegressionPlotter
    from aletheia.evaluation.reporter import EvaluationReporter, EvaluationReport
"""

from .metrics import RegressionEvaluator, RegressionMetrics
from .plots import RegressionPlotter
from .reporter import EvaluationReporter, EvaluationReport

__all__ = [
    "RegressionEvaluator",
    "RegressionMetrics",
    "RegressionPlotter",
    "EvaluationReporter",
    "EvaluationReport",
]
