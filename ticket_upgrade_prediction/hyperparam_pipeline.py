from sklearn.pipeline import Pipeline as sk_pipeline
from ticket_upgrade_prediction.pipeline import Pipeline as df_pipeline
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from loguru import logger
import xgb
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import random
from sklearn.metrics import accuracy_score
import itertools
import numpy as np


class HyperparamPipeline:
    def __init__(self, X: pd.DataFrame, y: pd.Series, model: str, param_space: dict, taget_col: str, stratify: bool, cols_to_scale: list, classification: bool, metric: str):
        self.X, self.y = X, y
        self.model = model
        self.classification = classification
        self.cols_to_scale = cols_to_scale
        self.target_col = taget_col
        self.pipe = None
        self.scores = []
        self.params = []
        self.param_space = param_space
        self.stratify = stratify
        self.best_params = None

    def get_best_params(self):
        return self.params[np.argmax(self.scores)], np.max(self.scores)

    def search_for_params(self, searching_algo: str = 'random', n_splits: int = 5, **kwargs):
        #dodatkowa metoda do wyciagniecia najlepszego wyniku + hiperkow
        if searching_algo == 'random':
            self.optimize_hypers_using_random_search(n_splits, kwargs)
        elif searching_algo == 'grid':
            self.optimize_hypers_using_grid_search(n_splits)
        else:
            raise ValueError('Unrecognizable searching_algo param. Currently available are: random, grid, bayes.')

    @staticmethod
    def get_random_params(param_space):
        return {k: random.choice(v) for k, v in param_space.items()}

    def optimize_hypers_using_grid_search(self, n_splits):
        perm_dicts = self.get_permutation_dicts(self.param_space)
        for iteration_params in perm_dicts:
            self.params.append(iteration_params)
            self.create_splits_and_calc_scores(n_splits, iteration_params)

    @staticmethod
    def get_permutation_dicts(param_space):
        keys, values = zip(*param_space.values())
        return [dict(zip(keys, v)) for v in itertools.product(*values)]

    def optimize_hypers_using_random_search(self, n_splits, n_iters):
        for i in range(n_iters):
            iteration_params = self.get_random_params(self.param_space)
            self.params.append(iteration_params)
            self.create_splits_and_calc_scores(n_splits, iteration_params)

    def create_splits_and_calc_scores(self, n_splits, iteration_params):
        kf = StratifiedKFold(n_splits) if self.stratify else kf = KFold(n_splits)
        self.scores.append(np.mean(
            [self.create_preds_for_hypers(train_index, test_index, iteration_params) for train_index, test_index in
             kf.split(self.X, self.y)]))

    def get_scaled_train_and_test_sets(self, train_index, test_index):
        scaler = StandardScaler()
        X_train, X_test, y_train, y_test = self.X[train_index], self.X[test_index], self.y[train_index], self.y[
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
        predictions = model.predict_proba(X_test) if self.classification else model.predict(X_test)
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
            return xgb.XGBClassifier(params)
        else:
            raise ValueError('this model is currently not supported. try any of: xgb')


if __name__ == '__main__':
    d = df_pipeline(model_type="predict_when_upgrade")
    d.get_oh_encoding()
    df = d.df