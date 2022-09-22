from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
import mlflow

@dataclass
class Metrics:
    accuracy: float = None
    roc_auc: float = None
    precision: float = None
    recall: float = None
    f1: float = None
    pr_auc: float = None

    def to_dict(self) -> dict:
        return self.__dict__


class Evaluator:
    def __init__(self, model: Any, X: pd.DataFrame, y: pd.Series) -> None:

        self.model = model
        self._assert_model_has_proper_methods()

        self.X = X
        self.y = y

        self.preds = self._get_preds()
        self.proba = self._get_proba()

    def _assert_model_has_proper_methods(self) -> None:
        # TODO! Add support for torchs
        required_methods = ["predict", "predict_proba"]
        methods = dir(self.model)

        if not all(
            required_method in methods for required_method in required_methods
        ):
            raise AttributeError(
                f"Wrong model class provided! Model needs to have {required_methods} methods"
            )

    def _get_preds(self) -> np.array:
        return self.model.predict(self.X)

    def _get_proba(self) -> np.array:
        return self.model.predict_proba(self.X)[:, 1]

    def get_accuracy(self) -> float:
        return accuracy_score(y_true=self.y, y_pred=self.preds)

    def get_roc_auc(self) -> float:
        return roc_auc_score(y_true=self.y, y_score=self.proba)

    def _get_pr_curve_properties(self) -> float:
        precision, recall, thresholds = precision_recall_curve(
            self.y, self.proba
        )
        # code is repeate to be clear what is returned
        return precision, recall, thresholds

    def get_precision(self) -> float:
        return precision_score(y_true=self.y, y_pred=self.preds)

    def get_recall(self) -> float:
        return recall_score(y_true=self.y, y_pred=self.preds)

    def get_f1_score(self) -> float:
        return f1_score(y_true=self.y, y_pred=self.preds)

    def get_pr_auc(self) -> float:
        precision, recall, _ = self._get_pr_curve_properties()
        return auc(recall, precision)

    def get_all_metrics(self, to_mlflow: bool = False) -> Metrics:
        metrics =  Metrics(
            accuracy=self.get_accuracy(),
            roc_auc=self.get_roc_auc(),
            precision=self.get_precision(),
            recall=self.get_recall(),
            f1=self.get_f1_score(),
            pr_auc=self.get_pr_auc(),
        )

        if to_mlflow: 
            mlflow.log_metrics(metrics)

        return metrics 


if __name__ == "__main__":
    # Just for testing purposes, to be removed later
    X, y = make_classification(n_samples=10000, weights=[0.5])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    model.predict_proba
    evaluator = Evaluator(model=model, X=X_test, y=y_test)
    print(evaluator.get_all_metrics())
