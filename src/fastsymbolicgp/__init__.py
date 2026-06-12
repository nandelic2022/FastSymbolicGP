from .estimators.classifier import FastSymbolicClassifier
from .estimators.regressor import FastSymbolicRegressor
from .estimators.multiclass import FastSymbolicMultiClassifier
from .ensemble.classifier import FastSymbolicEnsembleClassifier
from .ensemble.regressor import FastSymbolicEnsembleRegressor

__version__ = "0.2.0"

__all__ = [
    "FastSymbolicClassifier",
    "FastSymbolicRegressor",
    "FastSymbolicMultiClassifier",
    "FastSymbolicEnsembleClassifier",
    "FastSymbolicEnsembleRegressor",
]
