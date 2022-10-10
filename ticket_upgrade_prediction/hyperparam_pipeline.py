from sklearn.pipeline import Pipeline as sk_pipeline
from ticket_upgrade_prediction.pipeline import Pipeline as df_pipeline
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


class HyperparamPipeline:
    def __init__(self, dataset: pd.DataFrame, model, param_dict: dict):
        self.dataset = dataset
        self.model = model
        self.pipe = None
        self.param_dict = None

    def make_pipeline(self):
        self.pipe = sk_pipeline(StandardScaler(), self.model)

    def search_for_params(self, searching_algo: str = 'random'):
        if searching_algo == 'random':
            pass
        elif searching_algo == 'grid':
            pass
        elif searching_algo == 'bayes':
            pass
        else:
            raise ValueError('Unrecognizable searching_algo param. Currently available are: random, grid, bayes.')


if __name__ == '__main__':
    d = df_pipeline(model_type="predict_when_upgrade")
    d.get_oh_encoding()
    df = d.df