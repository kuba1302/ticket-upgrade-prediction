from sklearn.pipeline import Pipeline as sk_pipeline
from ticket_upgrade_prediction.pipeline import Pipeline as df_pipeline
import pandas as pd
from sklearn.preprocessing import StandardScaler
from loguru import logger
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import random
from sklearn.metrics import accuracy_score
import itertools
import numpy as np
from sklearn.datasets import make_classification
import warnings
from sklearn.model_selection import ParameterGrid
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)


class HyperparamPipeline:
    def __init__(self, X: pd.DataFrame, y: pd.Series, model: str, param_space: dict, stratify: bool, cols_to_scale: list, classification: bool, metric: str):
        self.X, self.y = X, y
        self.model = model
        self.classification = classification
        self.cols_to_scale = cols_to_scale
        self.metric = metric
        self.scores = []
        self.params = []
        self.param_space = param_space
        self.stratify = stratify

    def get_best_params(self):
        return self.params[np.argmax(self.scores)], np.max(self.scores)

    def search_for_params(self, searching_algo: str = 'random', n_splits: int = 5, **kwargs):
        logger.info(f"starting to serach for params using {searching_algo} search")
        if searching_algo == 'random':
            self.optimize_hypers_using_random_search(n_splits, kwargs['n_iters'])
        elif searching_algo == 'grid':
            self.optimize_hypers_using_grid_search(n_splits)
        else:
            raise ValueError('Unrecognizable searching_algo param. Currently available are: random, grid, bayes.')

    @staticmethod
    def get_random_params(param_space):
        return {k: (random.choice(v) if type(v) == list else v) for k, v in param_space.items()}

    def optimize_hypers_using_grid_search(self, n_splits):
        for iteration_params in ParameterGrid(self.param_space):
            self.params.append(iteration_params)
            score = self.create_splits_and_calc_scores(n_splits, iteration_params)
            logger.info(f'score for params {iteration_params} -> {score}')
            self.scores.append(score)

    def optimize_hypers_using_random_search(self, n_splits, n_iters):
        for i in range(n_iters):
            iteration_params = self.get_random_params(self.param_space)
            self.params.append(iteration_params)
            score = self.create_splits_and_calc_scores(n_splits, iteration_params)
            logger.info(f'score for params {iteration_params} -> {score}')
            self.scores.append(score)

    def create_splits_and_calc_scores(self, n_splits, iteration_params):
        kf = StratifiedKFold(n_splits) if self.stratify else KFold(n_splits)
        return np.mean(
            [self.create_preds_for_hypers(train_index, test_index, iteration_params) for train_index, test_index in
             kf.split(self.X, self.y)])

    def get_scaled_train_and_test_sets(self, train_index, test_index):
        scaler = StandardScaler()
        X_train, X_test, y_train, y_test = self.X.iloc[train_index], self.X.iloc[test_index], self.y.iloc[train_index], self.y.iloc[
            test_index]
        X_train[self.cols_to_scale] = scaler.fit_transform(
            X_train[self.cols_to_scale]
        )
        X_test[self.cols_to_scale] = scaler.transform(X_test[self.cols_to_scale])
        return X_train, X_test, y_train, y_test

    def create_preds_for_hypers(self, train_index, test_index, iteration_params):
        model = self.determine_model(iteration_params)
        X_train, X_test, y_train, y_test = self.get_scaled_train_and_test_sets(train_index, test_index)
        model.fit(X_train, y_train)
        #predictions = model.predict_proba(X_test)[:, 1] if self.classification else model.predict(X_test)
        predictions = model.predict(X_test)
        return self.calculate_metric(predictions, y_test)

    def calculate_metric(self, y_pred, y_true):
        #add support for more metrics
        if self.metric == 'accuracy_score':
            return accuracy_score(y_pred, y_true)
        else:
            raise ValueError('this model is currently not supported. try any of: xgb')

    def determine_model(self, params):
        #add support for more models
        if self.model == 'xgb' and self.classification:
            return xgb.XGBClassifier(objective='binary:logistic', verbosity=0, **params)
        else:
            raise ValueError('this model is currently not supported. try any of: xgb')


if __name__ == '__main__':
    X, y = make_classification(n_samples=10000, weights=[0.5])
    X = pd.DataFrame(data=X, columns=[f"col_{x}" for x in range(X.shape[1])])
    y = pd.DataFrame(data=y, columns=["y"])
    space = {
        "n_estimators": [5, 10],
        "eta": [0.025, 0.5, 0.025],
        "max_depth": [4, 8, 12]
    }
    hp = HyperparamPipeline(X, y, model='xgb', param_space=space, stratify=True, cols_to_scale=X.columns, classification=True, metric='accuracy_score')
    hp.search_for_params(searching_algo='random', n_splits=5, n_iters=3)
    print(hp.get_best_params())
