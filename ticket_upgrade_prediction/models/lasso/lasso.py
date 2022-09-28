import gc

import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from ticket_upgrade_prediction.models.base import BaseModel
from ticket_upgrade_prediction.pipeline import Pipeline


class LassoModel(BaseModel):
    def __init__(self, from_mlflow: bool = False, **kwargs):
        self.model = None
        self.get_fitted_model(
            model_name=kwargs.pop("model_name", "LASSO"),
            version=kwargs.pop("version", 1),
        ) if from_mlflow else self.fit_model(
            dataset=kwargs.pop("dataset", Pipeline().df),
            class_weight_balance=kwargs.pop(
                "class_weight_balance", "balanced"
            ),
            verbose=kwargs.pop("verbose", 2),
            max_iter=kwargs.pop("max_iter", 333),
            target=kwargs.pop("target", "UPGRADED_FLAG"),
        )

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)

    def fit_model(
        self,
        dataset: pd.DataFrame,
        class_weight_balance: str,
        verbose: int,
        max_iter: int,
        target: str,
    ) -> LogisticRegression:
        model = LogisticRegression(
            penalty="l1",
            solver="saga",
            class_weight=class_weight_balance,
            verbose=verbose,
            max_iter=max_iter,
        )
        dataset.dropna(inplace=True)
        X_train, _, y_train, _ = train_test_split(
            dataset.drop([target], axis=1),
            dataset[[target]],
            test_size=0.2,
            random_state=42,
        )
        del dataset
        gc.collect()
        model.fit(X_train, y_train)
        self.model = model

    def get_fitted_model(
        self, model_name: str, version: int
    ) -> LogisticRegression:
        model_info = mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}/{version}"
        )
        self.model = model_info._model_impl


if __name__ == "__main__":
    lasso_model = LassoModel(from_mlflow=True, version=1)
