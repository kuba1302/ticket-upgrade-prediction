from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

@dataclass
class HyperParams:
    layers: list[int]
    optimizer_name: str
    learning_rate: float


class ModelTrainer:
    def __init__(
        self,
        dataset: Dataset,
        layers: list = [5],
        optimizer_name: str = "Adam",
        epochs: int = 2,
        learning_rate: float = 0.0001,
        batch_size: int = 64,
        criterion: Any = BCEWithLogitsLoss,
    ) -> None:
        self.layers = layers
        self.model = Network(
            input_size=dataset.X_train.shape[1], hidden_layers_sizes=layers
        )
        self.optimizer = self._get_optimizer(
            optimizer_name=optimizer_name, learning_rate=learning_rate
        )
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.epochs = epochs
        self.criterion = criterion
        self.batch_size = batch_size

        self.training_results = []

    def _get_optimizer(
        self, optimizer_name: str, learning_rate: float
    ) -> None:
        if optimizer_name.lower() == "adam":
            self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_name.lower() == "sgd":
            self.optimizer = SGD(self.model.parameters(), lr=learning_rate)
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

    def fit(self, to_mlflow: bool = False):
        train_dataset = UpgradeDataset(
            X=self.dataset.X_train, y=self.dataset.y_train.values
        )
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=self.batch_size, shuffle=True
        )
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
        batch_size: int = 16
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

    def _one_hparam_combination(self, hparams):
        s_kfold = StratifiedKFold(n_splits=self.n_splits)
        metrics_list = []

        for train_index, test_index in s_kfold.split(
            self.data.drop(columns=self.y_col), self.data[:, self.y_col]
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
            trainer = ModelTrainer(
                dataset=dataset,
                layers=hyper_params.layers,
                optimizer_name=hyper_params.optimizer_name,
                epochs=self.per_fold_epoch, 
                learning_rate=hyper_params.learning_rate, 
                batch_size=self.batch_size, 
            )
            
            trainer.fit()
            # In future - change last merics to best metrics? 
            last_metrics = trainer.get_results()[-1]
            metrics_list.append(last_metrics)
        
        return Metrics.from_multiple_metrics(*metrics_list)

    def _one_fold_train(self):
        pass

    def hyperopt(
        self,
        target_metric: str,
        params: dict,
        number_of_hparams_combinations: int,
    ):
        param_combinations: list[dict] = self._get_params_combinations()

        for param_combination in param_combinations[
            :number_of_hparams_combinations
        ]:
            result = self._one_hparam_combination()
            self.metrics.append(result)

    def get_metrics(self):
        return self.metrics

    def get_highest_metric(self):
        pass


"""
1. Podajemy df
2. Wybieramy losowe kombinacje parametrow
3. Dla kazdej kombinacji: 
    - Dzielimy na foldy.
    - Dla kazdego folda: 
        - trening modelu
        - ewaluacja
        - zapisanie metryk
    - srednia z metryk 
    - zapis w mlflow 
    - trzymamy w pamieci jakie parametry juz poszy (na wypadek jakby sie wypierdolilo)



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
    data = pd.read_csv(data_path).dropna()

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


@click.command()
@click.option(
    "--layers",
    "-l",
    multiple=True,
    required=True,
    type=int,
)
@click.option(
    "--optimizer",
    "-o",
    type=click.Choice(
        ["Adam", "SGD"],
        case_sensitive=False,
    ),
    default="Adam",
)
@click.option("--epochs", "-e", default=10, type=int)
@click.option("--learning-rate", "-lr", default=0.001, type=int)
@click.option("--batch-size", "-b", default=64, type=int)
@click.option("--train-size", "-tr", default=0.75, type=float)
def main(
    layers: list,
    optimizer: str,
    epochs: int,
    learning_rate: int,
    batch_size: int,
    train_size: float,
):
    print(layers)
    train_model(
        layers=layers,
        optimizer=optimizer,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        train_size=train_size,
    )


if __name__ == "__main__":
    main()
