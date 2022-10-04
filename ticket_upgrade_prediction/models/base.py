from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    All models should inherit from this interface
    In this way, we make sure, that every model is
    compatible with our custom Evaluator class
    that requires predict and predict_proba
    methods.
    """

    @abstractmethod
    def predict(self, X):
        """Sklearn-like predict function"""

    @abstractmethod
    def predict_proba(self, X):
        """Sklearn-like predict probability function"""

    @abstractmethod
    def fit_model(self, **kwargs):
        """Fit model on new data"""

    @abstractmethod
    def get_fitted_model(self, **kwargs):
        """Retrieve fitted model from MLflow"""

    @abstractmethod
    def save_model_to_pickle(self, model_name):
        """Save model to .pkl"""

    @abstractmethod
    def save_model_to_mlflow(self, model_name, artifact_path, X_test, y_test):
        """Save model to mlflow"""
