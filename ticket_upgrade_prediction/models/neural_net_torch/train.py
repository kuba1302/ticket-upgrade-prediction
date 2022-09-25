import click
import torch
from neural_net_model import Network
from data_loader import UpgradeDataset
from torch.optim import Adam, SGD
from torch.nn import BCELoss
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm


@click.command()
@click.option(
    "--layer",
    "-l",
    multiple=True,
    required=True,
    type=list,
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
    layer: list,
    optimizer: str,
    epochs: int,
    learning_rate: int,
    batch_size: int,
    train_size: float,
):
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
    model = Network(layers_sizes=layer)

    criterion = BCELoss()
    optimizer = (
        Adam(model.parameters(), lr=learning_rate)
        if optimizer.lower() == "adam"
        else SGD(model.parameters(), lr=learning_rate)
    )

    for epoch in tqdm(epochs, desc="Epoch"):
        for X, y in iter(train_loader):
            X.to(device)
            y.to(device)

            preds = model(X)
            loss = criterion(preds, y)

            optimizer.zego_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    main()
