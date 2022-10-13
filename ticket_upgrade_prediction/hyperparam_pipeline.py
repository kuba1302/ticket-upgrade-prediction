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

    def search_for_params(self, searching_algo: str = 'random'):
        #wez goly dataset
        #wylosuj parametry (albo wez kolejne z grida)
        #wylosuj probke pod kfold x-validation
        #zrob skalowanie na kazdej probce
        #zapisz gdzies sredni wynik z walidacji (oraz paramsy)
        #dodatkowa metoda do wyciagniecia najlepszego wyniku + hiperkow
        if searching_algo == 'random':
            pass
        elif searching_algo == 'grid':
            pass
        elif searching_algo == 'bayes':
            self.best_params = self.optimize_bayes()
        else:
            raise ValueError('Unrecognizable searching_algo param. Currently available are: random, grid, bayes.')

    @staticmethod
    def get_random_params(param_space):
        return {k: random.choice(v) for k, v in param_space.items()}

    def optimize_hypers_using_random_search(self, n_iters, n_splits):
        for i in range(n_iters):
            iteration_params = self.get_random_params(self.param_space)
            kf = StratifiedKFold(n_splits) if self.stratify else kf = KFold(n_splits)
            for train_index, test_index in kf.split(self.X, self.y):
                X_train, X_test, y_train, y_test = self.X[train_index], self.X[test_index], self.y[train_index], self.y[test_index]
                scaler = StandardScaler()
                X_train[self.cols_to_scale] = scaler.fit_transform(
                    X_train[self.cols_to_scale]
                )
                X_test[self.cols_to_scale] = scaler.transform(X_test[self.cols_to_scale])
                model = self.determine_model(iteration_params)
                model.fit(X_train, y_train)
                predictions = model.predict_proba(X_test) if self.classification else model.predict(X_test)
                self.scores.append(predictions)
                self.params.append(iteration_params)

    def determine_metric(self, y_pred, y_true):
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

    def optimize_bayes(self):
        return fmin(self.score,
             self.param_space,
             algo=tpe.suggest,
             trials=self.trials,
             max_evals=50)

    #def prepare_dmatrices_for_xgb(self):
    #    return xgb.DMatrix(self.dataset[col for col  in self.dataset.columns if col != self.train_cols],
    #    label=self.dataset[self.test_cols]), xgb.DMatrix(X_test, label=y_test)

    #def score(self, params):
    #    print("Training with params: ")
    #    print(params)
    #    if 'n_estimators' in params:
    #        num_round = int(params['n_estimators'])
    #        del params['n_estimators']
    #
    #    #watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    #    model_for_training = self.model.train(self.param_space)
    #    gbm_model = xgb.train(params, dtrain, num_round,
    #                          verbose_eval=True, evals=watchlist,
    #                          early_stopping_rounds=20)
    #    predictions = gbm_model.predict(dvalid,
    #                                    ntree_limit=gbm_model.best_iteration + 1)
    #    score = roc_auc_score(y_test, predictions)
    #    print("\tScore {0}\n\n".format(score))
    #    loss = 1 - score
    #    return {'loss': loss, 'status': STATUS_OK}



if __name__ == '__main__':
    d = df_pipeline(model_type="predict_when_upgrade")
    d.get_oh_encoding()
    df = d.df