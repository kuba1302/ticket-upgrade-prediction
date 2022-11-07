import random
import warnings

import catboost as ctb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger
from pandas.core.common import SettingWithCopyWarning
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, ParameterGrid, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

from typing import Tuple


class HyperparamPipeline:
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model: str,
        param_space: dict,
        stratify: bool,
        cols_to_scale: list,
        classification: bool,
        metric: str,
    ):
        self.X, self.y = X, y
        self.model = model
        self.classification = classification
        self.cols_to_scale = cols_to_scale
        self.metric = metric
        self.scores = []
        self.params = []
        self.param_space = param_space
        self.stratify = stratify

    def get_best_params(self) -> Tuple[dict, float]:
        return self.params[np.argmax(self.scores)], np.max(self.scores)

    def search_for_params(
        self, searching_algo: str = "random", n_splits: int = 5, **kwargs
    ) -> None:
        logger.info(
            f"starting to serach for params using {searching_algo} search"
        )
        if searching_algo == "random":
            self.optimize_hypers_using_random_search(
                n_splits, kwargs["n_iters"]
            )
        elif searching_algo == "grid":
            self.optimize_hypers_using_grid_search(n_splits)
        else:
            raise ValueError(
                "Unrecognizable searching_algo param. Currently available are: random, grid."
            )

    @staticmethod
    def get_random_params(param_space: dict) -> dict:
        return {
            k: (random.choice(v) if type(v) == list else v)
            for k, v in param_space.items()
        }

    def optimize_hypers_using_grid_search(self, n_splits: int) -> None:
        for iteration_params in ParameterGrid(self.param_space):
            self.append_params_and_calculate_scores(iteration_params, n_splits)

    def append_params_and_calculate_scores(
        self, iteration_params: dict, n_splits: int
    ) -> None:
        self.params.append(iteration_params)
        score = self.create_splits_and_calc_scores(n_splits, iteration_params)
        logger.info(f"score for params {iteration_params} -> {score}")
        self.scores.append(score)

    def optimize_hypers_using_random_search(
        self, n_splits: int, n_iters: int
    ) -> None:
        for i in range(n_iters):
            iteration_params = self.get_random_params(self.param_space)
            self.append_params_and_calculate_scores(iteration_params, n_splits)

    def create_splits_and_calc_scores(
        self, n_splits: int, iteration_params: dict
    ) -> np.ndarray:
        kf = StratifiedKFold(n_splits) if self.stratify else KFold(n_splits)
        return np.mean(
            [
                self.create_preds_for_hypers(
                    train_index, test_index, iteration_params
                )
                for train_index, test_index in kf.split(self.X, self.y)
            ]
        )

    def map_cols_to_scale_to_boolean(self):
        return [col in self.cols_to_scale for col in list(X.columns)]

    def get_scaled_train_and_test_sets(
        self, train_index: np.ndarray, test_index: np.ndarray
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        scaler = StandardScaler()
        X_train, X_test, y_train, y_test = (
            self.X.iloc[train_index],
            self.X.iloc[test_index],
            self.y.iloc[train_index],
            self.y.iloc[test_index],
        )
        X_train.iloc[:, self.map_cols_to_scale_to_boolean()] = scaler.fit_transform(
            X_train.iloc[:, self.map_cols_to_scale_to_boolean()]
        )
        X_test.iloc[:, self.map_cols_to_scale_to_boolean()] = scaler.transform(
            X_test.iloc[:, self.map_cols_to_scale_to_boolean()]
        )
        return X_train, X_test, y_train, y_test

    def create_preds_for_hypers(
        self,
        train_index: np.ndarray,
        test_index: np.ndarray,
        iteration_params: dict,
    ) -> float:
        model = self.determine_model(iteration_params)
        X_train, X_test, y_train, y_test = self.get_scaled_train_and_test_sets(
            train_index, test_index
        )
        model.fit(X_train, y_train.values.ravel())
        # predictions = model.predict_proba(X_test)[:, 1] if self.classification else model.predict(X_test)
        predictions = model.predict(X_test)
        return self.calculate_metric(predictions, y_test)

    def calculate_metric(self, y_pred: pd.Series, y_true: pd.Series) -> float:
        # add support for more metrics
        return 1

    def determine_model(self, params: dict):
        if self.model == "xgb":
            if self.classification:
                return xgb.XGBClassifier(
                    objective="binary:logistic", verbosity=0, **params
                )
            return xgb.XGBRegressor(
                objective="reg:squarederror", verbosity=0, **params
            )
        elif self.model == "lr":
            return LogisticRegression(**params)
        elif self.model == "knn":
            if self.classification:
                return KNeighborsClassifier(**params)
            return KNeighborsRegressor(**params)
        elif self.model == "cat":
            if self.classification:
                return ctb.CatBoostClassifier(**params)
            return ctb.CatBoostRegressor(**params)
        elif self.model == "lgb":
            if self.classification:
                return lgb.LGBMClassifier(**params)
            return lgb.LGBMRegressor(**params)
        else:
            raise ValueError(
                "this model is currently not supported. try any of: xgb, lr, knn, cat, lgb"
            )


if __name__ == "__main__":
    X, y = make_classification(n_samples=10000, weights=[0.5])
    X = pd.DataFrame(data=X, columns=[f"col_{x}" for x in range(X.shape[1])])
    y = pd.DataFrame(data=y, columns=["y"])
    space = {
        "n_estimators": [5, 10],
        "eta": [0.025, 0.5, 0.025],
        "max_depth": [4, 8, 12],
    }
    hp = HyperparamPipeline(
        X,
        y,
        model="xgb",
        param_space=space,
        stratify=True,
        cols_to_scale=list(X.columns),
        classification=True,
        metric="accuracy_score",
    )
    hp.search_for_params(searching_algo="grid", n_splits=5)
    print(hp.get_best_params())
