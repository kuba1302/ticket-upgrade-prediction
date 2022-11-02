from copyreg import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import click
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
from data_loader import UpgradeDataset
from loguru import logger
from neural_net_model import Network
from sklearn.model_selection import train_test_split
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from ticket_upgrade_prediction.pipeline import Dataset
from ticket_upgrade_prediction import Evaluator, Metrics
from sklearn.model_selection import train_test_split
import itertools
import mlflow
from ticket_upgrade_prediction.pipeline import Pipeline
from ticket_upgrade_prediction.config.env_config import EXPERIMENT_NAME


def mlflow_run_start_handle(method):
    def wrapper(*args, **kwargs):
        mlflow_run_name = kwargs.get("mlflow_run_name", None)
        to_mlflow = True if mlflow_run_name else False

        if to_mlflow:
            client = mlflow.MlflowClient()
            experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

            with mlflow.start_run(
                run_name=mlflow_run_name,
                experiment_id=experiment.experiment_id,
            ):
                method(*args, **kwargs)

        else:
            method(*args, **kwargs)

    return wrapper


@dataclass
class HyperParams:
    layers: list[int]
    optimizer_name: str
    learning_rate: float

    def to_dict(self) -> dict:
        return self.__dict__


class NetworkTrainer:
    def __init__(
        self,
        dataset: Dataset,
        hparams: HyperParams = HyperParams(
            layers=[5], optimizer_name="Adam", learning_rate=0.0001
        ),
        epochs: int = 2,
        batch_size: int = 64,
        criterion: Any = BCEWithLogitsLoss(),
        plots_save_path: Optional[Path] = None,
    ) -> None:
        self.dataset = dataset
        self.hparams = hparams
        self.plot_save_path = plots_save_path
        self.model = Network(
            input_size=dataset.X_train.shape[1],
            hidden_layers_sizes=self.hparams.layers,
        )
        self.optimizer = self._get_optimizer(
            optimizer_name=self.hparams.optimizer_name,
            learning_rate=self.hparams.learning_rate,
        )
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.epochs = epochs
        self.criterion = criterion
        self.batch_size = batch_size

        self.training_results = []

    def _get_optimizer(self, optimizer_name: str, learning_rate: float) -> Any:
        if optimizer_name.lower() == "adam":
            return Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_name.lower() == "sgd":
            return SGD(self.model.parameters(), lr=learning_rate)
        else:
            raise ValueError("Wrong type of optimizer provided!")

    def _one_batch(self, X: torch.Tensor, y: torch.Tensor) -> None:
        X.to(self.device)
        y.to(self.device)

        preds_logit = self.model(X.float())
        loss = self.criterion(preds_logit.float(), y.float())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    @mlflow_run_start_handle
    def fit(self, mlflow_run_name: Optional[int] = None) -> None:
        to_mlflow = True if mlflow_run_name else False

        train_dataset = UpgradeDataset(
            X=self.dataset.X_train, y=self.dataset.y_train.values
        )
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=self.batch_size, shuffle=True
        )

        if to_mlflow:
            mlflow.log_params(self.hparams.to_dict())

        for epoch in tqdm(range(self.epochs), desc="Epoch"):
            for X, y in tqdm(iter(train_loader), desc="Batch"):
                self._one_batch(X=X, y=y)

            with torch.no_grad():
                self._evaluate(to_mlflow=to_mlflow, epoch=epoch)

    def _evaluate(self, to_mlflow: bool, epoch: int) -> None:
        evaluator = Evaluator(
            model=self.model, X=self.dataset.X_test, y=self.dataset.y_test
        )
        metrics = evaluator.get_all_metrics(epoch=epoch, to_mlflow=to_mlflow)
        # _ = evaluator.plot_all_plots(
        #     save_path=self.plot_save_path, to_mlflow=to_mlflow
        # )
        logger.info(metrics)
        self.training_results.append(metrics)

    def get_results(self) -> list[Metrics]:
        return self.training_results


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
        mlflow_run_name: bool = False,
    ):
        to_mlflow = True if mlflow_run_name else False

        for param_combination in self.hyper_params_combinations[
            :number_of_hparams_combinations
        ]:
            result = self._one_hparam_combination(
                hparams=param_combination, to_mlflow=to_mlflow
            ).get_metric_from_string(metric_name=target_metric)

            self.metrics.append(result)

    def get_metrics(self):
        return self.metrics

    def get_highest_metric(self):
        pass


"""
1. Checkpointy - zapis wszystkich kombinacji i tych jakie juz poszly
"""


def train_model(
    layers: list = [5],
    optimizer: str = "Adam",
    epochs: int = 2,
    learning_rate: int = 0.0001,
    batch_size: int = 64,
    train_size: float = 0.75,
):
    layers = [int(layer) for layer in layers]
    data_path = str(
        Path(__file__).parents[3] / "data" / "preprocessed_upgrade.csv",
    )
    data = pd.read_csv(data_path).dropna().head(1000)

    y_col = "UPGRADED_FLAG"
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(columns=y_col),
        data[y_col],
        train_size=train_size,
        random_state=1,
    )

    train_dataset = UpgradeDataset(X=X_train, y=y_train.values)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = X_train.shape[1]
    model = Network(input_size=input_size, hidden_layers_sizes=layers)

    criterion = BCEWithLogitsLoss()
    optimizer = (
        Adam(model.parameters(), lr=learning_rate)
        if optimizer.lower() == "adam"
        else SGD(model.parameters(), lr=learning_rate)
    )

    training_results = []

    for epoch in tqdm(range(epochs), desc="Epoch"):
        for X, y in tqdm(iter(train_loader), desc="Batch"):
            X.to(device)
            y.to(device)

            preds_logit = model(X.float())
            loss = criterion(preds_logit.float(), y.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                evaluator = Evaluator(model=model, X=X_test, y=y_test)
                metrics = evaluator.get_all_metrics(epoch=epoch)
                logger.info(metrics)

                training_results.append(metrics)


# @click.command()
# @click.option(
#     "--layers",
#     "-l",
#     multiple=True,
#     required=True,
#     type=int,
# )
# @click.option(
#     "--optimizer",
#     "-o",
#     type=click.Choice(
#         ["Adam", "SGD"],
#         case_sensitive=False,
#     ),
#     default="Adam",
# )
# @click.option("--epochs", "-e", default=10, type=int)
# @click.option("--learning-rate", "-lr", default=0.001, type=int)
# @click.option("--batch-size", "-b", default=64, type=int)
# @click.option("--train-size", "-tr", default=0.75, type=float)
# def main(
#     layers: list,
#     optimizer: str,
#     epochs: int,
#     learning_rate: int,
#     batch_size: int,
#     train_size: float,
# ):
#     train_model(
#         layers=layers,
#         optimizer=optimizer,
#         epochs=epochs,
#         learning_rate=learning_rate,
#         batch_size=batch_size,
#         train_size=train_size,
#     )


if __name__ == "__main__":
    import pickle

    data_path = Path(__file__).parents[3] / "data" / "dataset.pickle"
    with open(data_path, "rb") as file:
        dataset = pickle.load(file)

    # trainer = NetworkTrainer(dataset=dataset, epochs=20)
    # trainer.fit(mlflow_run_name="Neural-net-test")

    data_path_csv = Path(__file__).parents[3] / "data" / "dataset.csv"
    df = (
        pd.read_csv(data_path_csv, index_col=False)
        .sample(10000)
        .reset_index(drop=True)
    )
    params = {
        "layers": [[5], [5, 10], [5, 10, 15]],
        "optimizer_name": ["adam"],
        "learning_rate": [0.001, 0.01, 0.0001],
    }

    hyperopt = NeuralNetHyperopt(data=df, hyper_params=params)
    hyperopt.hyperopt(
        target_metric="roc_auc", number_of_hparams_combinations=3
    )

    print(hyperopt.get_metrics())
