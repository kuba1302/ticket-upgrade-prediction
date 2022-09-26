from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    All models shold inherit from this interface
    In this way, we make sure, that every model is
    compatible with our custem Evaluator class
    thath requires predict and predict_proba
    methods. 
    """
    @abstractmethod
    def predict(self, X):
        """Sklearn-like predict function"""

    def predict_proba(self, X):
        """Sklearn-like predict probability function"""