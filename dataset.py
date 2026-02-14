import random
from typing import Generator, Tuple
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from const import DATA_PATH, DEVICE,  BatchSize, isMain, seed


seed()


class DatasetManager:
    numberOfChannels = 1
    shape = (32, 32)

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

        self.trainSize = len(self.trainDataset)

        self.testDataset = datasets.MNIST(
            root=DATA_PATH,
            train=False,
            download=True,
            transform=transform
        )

        self.testSize = len(self.testDataset)

        self.trainDataLoaded: DataLoader[Tuple[torch.Tensor, torch.Tensor]] = DataLoader(
            dataset=self.trainDataset, batch_size=BatchSize.train, shuffle=True,)

        self.testDataLoaded: DataLoader[Tuple[torch.Tensor, torch.Tensor]] = DataLoader(
            dataset=self.testDataset, batch_size=BatchSize.test, shuffle=False)

    def testEntries(self) -> Generator[Tuple[int, torch.Tensor]]:
        for index, (X, y) in enumerate(self.testDataLoaded):
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            yield index, X

    def trainEntries(self, proportion: float = 1.0) -> Generator[Tuple[int, torch.Tensor]]:
        for index, (X, y) in enumerate(self.trainDataLoaded):
            if random.random() > proportion:
                continue
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            yield index, X

    def priorEntriesTrain(self, proportion: float = 1.0) -> Generator[Tuple[int, torch.Tensor]]:
        for index in range(self.trainSize//BatchSize.train):
            if random.random() > proportion:
                continue

            yield index, torch.randn((BatchSize.train, self.numberOfChannels, self.shape[0], self.shape[1])).to(DEVICE)

    def priorEntriesTest(self) -> Generator[Tuple[int, torch.Tensor]]:
        for index in range(self.testSize//BatchSize.test):
            yield index, torch.randn((BatchSize.test, self.numberOfChannels, self.shape[0], self.shape[1])).to(DEVICE)

    def getRandomPriors(self) -> torch.Tensor:
        return torch.randn((self.numberOfChannels, self.shape[0], self.shape[1])).to(DEVICE)

    def getRandomTest(self) -> torch.Tensor:
        idx = random.randint(0, self.testSize-1)
        return self.testDataset[idx][0].to(DEVICE)

    def getRandomTrain(self) -> torch.Tensor:
        idx = random.randint(0, self.trainSize-1)
        return self.trainDataset[idx][0].to(DEVICE)


datasetManager = DatasetManager()

if isMain(__name__):

    for index, X in datasetManager.trainEntries():
        print(X.shape, X.device)
        break

    for index, X in datasetManager.testEntries():
        print(X.shape, X.device)
        break
