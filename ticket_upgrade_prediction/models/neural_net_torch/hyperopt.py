import itertools
from typing import Any, Optional

import mlflow
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from ticket_upgrade_prediction.evaluator import Metrics
from ticket_upgrade_prediction.models.neural_net_torch.trainer import (
    HyperParams,
    NetworkTrainer,
    mlflow_run_start_handle,
)
from ticket_upgrade_prediction.pipeline import Dataset


class NeuralNetHyperopt:
    def __init__(
        self,
        data: pd.DataFrame,
        hyper_params: dict,
        n_splits: int = 5,
        scaler: Any = StandardScaler,
        y_col: str = "UPGRADED_FLAG",
        train_size: float = 0.75,
        per_fold_epoch: int = 10,
        batch_size: int = 16,
    ) -> None:
        self.n_splits = n_splits
        self.data = data
        self.scaler = scaler
        self.per_fold_epoch = per_fold_epoch
        self.metrics = []
        self.y_col = y_col
        self.batch_size = batch_size
        self.train_size = train_size
        self.hyper_params_combinations = self._get_params_combinations(
            hyper_params=hyper_params
        )

    def _get_params_combinations(self, hyper_params) -> list[dict]:
        keys, values = zip(*hyper_params.items())
        return [dict(zip(keys, v)) for v in itertools.product(*values)]

    def _one_hparam_combination(self, hparams: dict, to_mlflow: bool):
        s_kfold = StratifiedKFold(n_splits=self.n_splits)
        metrics_list = []

        for train_index, test_index in s_kfold.split(
            X=self.data.drop(columns=self.y_col), y=self.data[self.y_col]
        ):
            dataset = Dataset(
                X_train=self.data.loc[train_index, :].drop(columns=self.y_col),
                X_test=self.data.loc[test_index, :].drop(columns=self.y_col),
                y_train=self.data.loc[train_index, self.y_col],
                y_test=self.data.loc[test_index, self.y_col],
            )
            hyper_params = HyperParams(
                layers=hparams["layers"],
                optimizer_name=hparams["optimizer_name"],
                learning_rate=hparams["learning_rate"],
            )
            trainer = NetworkTrainer(
                dataset=dataset,
                hparams=hyper_params,
                epochs=self.per_fold_epoch,
                batch_size=self.batch_size,
            )
            # Don't save every fold to mlflow, do it manually later!
            trainer.fit(mlflow_run_name=None)

            # In future - change last merics to best metrics?
            last_metrics = trainer.get_results()[-1]
            metrics_list.append(last_metrics)

        metrics = Metrics.from_multiple_metrics(*metrics_list)

        if to_mlflow:
            mlflow.log_params(hyper_params.to_dict())
            mlflow.log_metrics(metrics)

        return metrics

    @mlflow_run_start_handle
    def hyperopt(
        self,
        target_metric: str,
        number_of_hparams_combinations: int,
        mlflow_run_name: Optional[str] = None,
    ):
        to_mlflow = bool(mlflow_run_name)

        for param_combination in self.hyper_params_combinations[
            :number_of_hparams_combinations
        ]:
            result = self._one_hparam_combination(
                hparams=param_combination, to_mlflow=to_mlflow
            ).get_metric_from_string(metric_name=target_metric)

            self.metrics.append(result)

    def get_metrics(self):
        return self.metrics
