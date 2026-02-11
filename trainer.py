from math import sqrt
from torch import nn, optim
import torch
from tqdm import tqdm

from const import DEVICE, Optimizer, isMain, lossFn
from dataset import DatasetManager
from meanPredictor import PredManager


class MeanPredictorTrainer():
    def __init__(self, epochs, lr, T, nChannels, numberOfTimesSteps) -> None:
        self.epochs = epochs
        self.lr = lr
        self.T = T
        self.nChannels = nChannels
        self.numberOfTimesSteps = numberOfTimesSteps

    def trainForward(self, backward: nn.Module, priorIterator):
        nEpochs = self.epochs
        T = self.T

        lr = self.lr

        forward = PredManager.getForwardPredUntrained(
            numberOfChannels=self.nChannels)

        numberOfTimesSteps = self.numberOfTimesSteps

        optimizer = Optimizer.getOptimizer(params=forward.parameters(), lr=lr)

        for _ in tqdm(range(nEpochs)):
            for _, X in priorIterator:

                print("prior", X.shape)
                forward.train()
                optimizer.zero_grad()

                curX: torch.Tensor = X
                listX: list[torch.Tensor] = [X]
                for i in range(numberOfTimesSteps):
                    dt = T/numberOfTimesSteps

                    timeStep = numberOfTimesSteps-i-1
                    backward.eval()
                    with torch.inference_mode():
                        curX = backward(curX, timeStep+1)+sqrt(2)*sqrt(dt) * \
                            torch.randn(curX.shape).to(DEVICE)

                    # If we don't clone we get an error
                    # Cause curX is a inference tensor
                    listX.append(curX.clone())

                loss = torch.zeros(1).to(DEVICE)

                oldX = listX.pop()
                for i in range(numberOfTimesSteps):
                    curX = listX.pop()

                    yPred = forward(oldX, i)

                    backward.eval()
                    with torch.inference_mode():
                        yTrue = oldX+backward(curX, i+1) - \
                            backward(oldX, i+1)

                    loss += lossFn(yPred, yTrue.clone())

                    oldX = curX

                loss /= numberOfTimesSteps

                loss.backward()

                optimizer.step()

        return forward

    def trainBackward(self, forward: nn.Module, XtrainIterator):
        nEpochs = self.epochs
        T = self.T
        lr = self.lr

        backward = PredManager.getBackwardPredUntrained(
            numberOfChannels=self.nChannels)

        numberOfTimesSteps = self.numberOfTimesSteps

        optimizer = Optimizer.getOptimizer(params=backward.parameters(), lr=lr)

        for _ in tqdm(range(nEpochs)):
            for _, X in XtrainIterator:

                print("img", X.shape)
                backward.train()
                optimizer.zero_grad()

                curX: torch.Tensor = X
                listX: list[torch.Tensor] = [X]
                for i in range(numberOfTimesSteps):
                    dt = T/numberOfTimesSteps

                    forward.eval()
                    with torch.inference_mode():
                        curX = forward(curX, i)+sqrt(2)*sqrt(dt) * \
                            torch.randn(curX.shape).to(DEVICE)

                    listX.append(curX.clone())

                loss = torch.zeros(1).to(DEVICE)

                oldX = listX.pop()
                for i in range(numberOfTimesSteps):
                    curX = listX.pop()

                    timeStep = numberOfTimesSteps-i-1

                    yPred = backward(oldX, timeStep+1)

                    forward.eval()
                    with torch.inference_mode():
                        yTrue = oldX+forward(curX, timeStep) - \
                            forward(oldX, timeStep)

                    loss += lossFn(yPred, yTrue.clone())

                    oldX = curX

                loss /= numberOfTimesSteps

                loss.backward()

                optimizer.step()
        return backward

    def train(self, nSteps):

        for i in range(nSteps):
            pass


if isMain(__name__):
    epochs = 10
    lr = 0.01
    T = 10

    meanPredictorTrainer = MeanPredictorTrainer(
        epochs=epochs, lr=lr, T=T, nChannels=1, numberOfTimesSteps=10)

    datasetManager = DatasetManager()

    print("Backward")
    meanPredictorTrainer.trainBackward(
        forward=PredManager.getForwardPredUntrained(numberOfChannels=1, ), XtrainIterator=datasetManager.trainEntries())

    print("Forward")

    meanPredictorTrainer.trainForward(backward=PredManager.getBackwardPredUntrained(
        numberOfChannels=1), priorIterator=datasetManager.priorEntriesTrain())
