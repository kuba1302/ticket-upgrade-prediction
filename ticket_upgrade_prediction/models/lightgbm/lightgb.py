import pickle

import numpy as np
from lightgb import LGBMClassifier

import mlflow
from ticket_upgrade_prediction.evaluator import Evaluator
from ticket_upgrade_prediction.models.base import BaseModel
from ticket_upgrade_prediction.pipeline import Pipeline


class LightGBMModel(BaseModel):
    def __init__(self):
        self.model = None

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)

    def fit_model(
        self,
        dataset: dict = Pipeline().df,
        target: str = "UPGRADED_FLAG",
        **kwargs,
    ) -> LGBMClassifier:
        model = LGBMClassifier(kwargs)
        model.fit(dataset.X_train, dataset.y_train)
        self.model = model

    def get_fitted_model(
        self, model_name: str = "LGBM", version: int = 1
    ) -> LGBMClassifier:
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
    lasso_model = LightGBMModel()
    lasso_model.get_fitted_model(version=1)
