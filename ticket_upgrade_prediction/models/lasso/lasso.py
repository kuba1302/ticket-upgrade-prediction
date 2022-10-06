import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import mlflow
from ticket_upgrade_prediction.evaluator import Evaluator
from ticket_upgrade_prediction.models.base import BaseModel
from ticket_upgrade_prediction.pipeline import Pipeline


class LassoModel(BaseModel):
    def __init__(self):
        self.model = None

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)

    def fit_model(
        self,
        dataset: dict = Pipeline().df,
        class_weight_balance: str = "balanced",
        verbose: int = 2,
        max_iter: int = 333,
        target: str = "UPGRADED_FLAG",
    ) -> LogisticRegression:
        model = LogisticRegression(
            penalty="l1",
            solver="saga",
            class_weight=class_weight_balance,
            verbose=verbose,
            max_iter=max_iter,
        )
        model.fit(dataset["X_train"], dataset["y_train"])
        self.model = model

    def get_fitted_model(
        self, model_name: str = "LASSO", version: int = 1
    ) -> LogisticRegression:
        model_info = mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}/{version}"
        )
        self.model = model_info._model_impl

    def save_model_to_pickle(self, model_name):
        with open(f"{model_name}.pkl", "wb") as file:
            pickle.dump(self.model, file)

    def save_model_to_mlflow(self, model_name, artifact_path, X_test, y_test):
        with mlflow.start_run():
            evaluator = Evaluator(model=self.model, X=X_test, y=y_test)
            evaluator.get_all_metrics(to_mlflow=True)
            mlflow.sklearn.log_model(
                sk_model=self.model,
                artifact_path=artifact_path,
                registered_model_name=model_name,
            )


if __name__ == "__main__":
    lasso_model = LassoModel()
    lasso_model.get_fitted_model(version=1)
