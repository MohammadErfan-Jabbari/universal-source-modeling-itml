"""Competition-local predictor interfaces and starter baselines."""

from competition.predictors.base import Predictor
from competition.predictors.ngram import NGramPredictor
from competition.predictors.uniform import UniformPredictor

__all__ = ["Predictor", "UniformPredictor", "NGramPredictor"]

