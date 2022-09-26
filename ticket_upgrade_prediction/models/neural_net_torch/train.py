from dataclasses import dataclass
import click
import torch
from neural_net_model import Network
from data_loader import UpgradeDataset
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss
from ticket_upgrade_prediction import Evaluator, Metrics
from loguru import logger


@dataclass
class Results:
    epoch: int
    metrics: Metrics


# @click.command()
# @click.option(
#     "--layers",
#     "-l",S
#     multiple=True,
#     required=True,
#     type=list[int],
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
def main(
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
    data = pd.read_csv(data_path)
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
        for X, y in iter(train_loader):
            X.to(device)
            y.to(device)

            preds_logit = model(X.float())
            loss = criterion(preds_logit.float(), y.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                evaluator = Evaluator(model=model, X=X_test, y=y_test)
                metrics = evaluator.get_all_metrics()
                logger.info(metrics)

                result = Results(epoch=epoch, metrics=metrics)
                training_results.append(result)


if __name__ == "__main__":
    main()
