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
    roc_curve,
)
from pathlib import Path
from sklearn.model_selection import train_test_split
import mlflow
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.inspection import PartialDependenceDisplay


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
        metrics = Metrics(
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

    def plot_roc_curve(self, save_path: Path = None) -> Figure:
        fpr, tpr, thresholds = roc_curve(y_true=self.y, y_score=self.proba)
        auc_score = auc(fpr, tpr)

        gmeans = np.sqrt(tpr * (1 - fpr))
        ix = np.argmax(gmeans)

        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(15, 7)

        ax.plot([0, 1], [0, 1], linestyle="--", label="Random model")
        ax.plot(fpr, tpr, marker=".", label=f"Model - AUC: {auc_score:.3f}")
        ax.scatter(
            fpr[ix],
            tpr[ix],
            marker="o",
            color="black",
            label=f"Best Threshold={thresholds[ix]:.2f}, G-Mean={gmeans[ix]:.2f}",
        )

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="upper left")

        if save_path:
            fig.savefig(save_path / "roc_curve.png")

        return fig

    def plot_precision_recall_curve(self, save_path: Path = None) -> Figure:
        precision, recall, thresholds = precision_recall_curve(
            self.y, self.proba
        )
        auc_score = auc(recall, precision)

        fscore = (2 * precision * recall) / (precision + recall)
        ix = np.argmax(fscore)

        no_skill = len(self.y[self.y == 1]) / len(self.y)

        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(15, 7)

        ax.plot([0, 1], [no_skill, no_skill], linestyle="--", label="No Skill")
        ax.plot(
            recall,
            precision,
            marker=".",
            label=f"Model - AUC: {auc_score:.3f}",
        )
        ax.scatter(
            recall[ix],
            precision[ix],
            marker="o",
            color="black",
            label=f"Best Threshold={thresholds[ix]:.2f}, F-Score={fscore[ix]:.2f}",
        )

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(loc="upper left")

        if save_path:
            fig.savefig(save_path / "precision_recall_curve.png")

        return fig

    def plot_partial_dependency_plot(
        self, save_path: Path = None, kind: str = "average"
    ) -> Figure:

        cols = [col for col in self.X.columns]
        pdp = PartialDependenceDisplay.from_estimator(
            self.model, self.X, cols, kind=kind
        )

        number_of_cols = len(cols)
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(1 * number_of_cols, 1.5 * number_of_cols)
        fig.suptitle("Partial dependency plot", size=15)
        pdp.plot(ax=ax)

        if save_path:
            fig.savefig(save_path / "partial_dependence_plot.png")

        return fig


if __name__ == "__main__":
    # Just for testing purposes, to be removed later
    save_path = Path(__file__).parents[0] / "plots"
    X, y = make_classification(n_samples=10000, weights=[0.5])
    X = pd.DataFrame(data=X, columns=[f"col_{x}" for x in range(X.shape[1])])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    model.predict_proba
    evaluator = Evaluator(model=model, X=X_test, y=y_test)
    evaluator.plot_precision_recall_curve(save_path=save_path)
    evaluator.plot_roc_curve(save_path=save_path)
    evaluator.plot_partial_dependency_plot(save_path=save_path)

    print(evaluator.get_all_metrics())
