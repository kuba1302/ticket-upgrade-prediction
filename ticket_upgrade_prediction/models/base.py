from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor

# !TODO add lightgbm
SklearnClassifier = Union[
    LogisticRegression,
    KNeighborsClassifier,
    RandomForestClassifier,
    XGBClassifier,
]
SklearnRegressor = Union[
    LinearRegression,
    KNeighborsRegressor,
    RandomForestRegressor,
    XGBRegressor,
]

SklearnModel = Union[SklearnClassifier, SklearnRegressor]


class BaseModel(ABC):
    """
    All models should inherit from this interface
    In this way, we make sure, that every model is
    compatible with our custom Evaluator class
    that requires predict and predict_proba
    methods.
    """

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Sklearn-like predict function"""

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        """Sklearn-like predict probability function"""

    @abstractmethod
    def fit_model(self, **kwargs) -> None:
        """Fit model on new data"""

    @abstractmethod
    def get_fitted_model(self, **kwargs):
        """Retrieve fitted model from MLflow"""

    @abstractmethod
    def save_model_to_pickle(self, model_name) -> None:
        """Save model to .pkl"""

    @abstractmethod
    def save_model_to_mlflow(
        self, model_name, artifact_path, X_test, y_test
    ) -> None:
        """Save model to mlflow"""
