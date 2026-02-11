from typing import Generator, Tuple
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from const import DATA_PATH, DEVICE,  BatchSize, isMain, seed


print(DEVICE)

seed()


class DatasetManager:
    numberOfChannels = 1
    shape = (32, 32)
    trainPriorSize = 400

    def __init__(self) -> None:
        transform = transforms.Compose(
            [
                # We add 2 of padding to make them 32x32
                transforms.Pad(2),
                transforms.ToTensor()
            ]
        )

        self.trainDataset = datasets.MNIST(
            root=DATA_PATH,
            train=True,
            download=True,
            transform=transform
        )

        self.testDataset = datasets.MNIST(
            root=DATA_PATH,
            train=False,
            download=True,
            transform=transform
        )

        self.trainDataLoaded: DataLoader[Tuple[torch.Tensor, torch.Tensor]] = DataLoader(
            dataset=self.trainDataset, batch_size=BatchSize.train, shuffle=True,)

        self.testDataLoaded: DataLoader[Tuple[torch.Tensor, torch.Tensor]] = DataLoader(
            dataset=self.testDataset, batch_size=BatchSize.test, shuffle=False)

    def testEntries(self) -> Generator[Tuple[int, torch.Tensor]]:
        for index, (X, y) in enumerate(self.testDataLoaded):
            X.to(DEVICE)
            y.to(DEVICE)

            yield index, X

    def trainEntries(self) -> Generator[Tuple[int, torch.Tensor]]:
        for index, (X, y) in enumerate(self.trainDataLoaded):
            X.to(DEVICE)
            y.to(DEVICE)

            yield index, X

    def priorEntriesTrain(self) -> Generator[Tuple[int, torch.Tensor]]:
        for index in range(self.trainPriorSize):
            yield index, torch.randn((BatchSize.train, self.numberOfChannels, self.shape[0], self.shape[1])).to(DEVICE)

    def priorEntriesTest(self) -> Generator[Tuple[int, torch.Tensor]]:
        for index in range(self.trainPriorSize):
            yield index, torch.randn((BatchSize.test, self.numberOfChannels, self.shape[0], self.shape[1])).to(DEVICE)


datasetManager = DatasetManager()

if isMain(__name__):

    for index, X in datasetManager.trainEntries():
        print(X.shape)
        break

    for index, X in datasetManager.testEntries():
        print(X.shape)
        break
