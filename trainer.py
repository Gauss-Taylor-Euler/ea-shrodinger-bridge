from math import sqrt
from typing import Any
from torch import nn
import torch
from tqdm import tqdm

from const import DEVICE, Optimizer, isMain, lossFn
from dataset import DatasetManager
from meanPredictor import PredManager


class MeanPredictorTrainer():
    def __init__(self, epochs: int, lr: float, T: float, nChannels: int, numberOfTimesSteps: int, dsbIterationNumber: int) -> None:
        self.epochs = epochs
        self.lr = lr
        self.T = T
        self.nChannels = nChannels
        self.numberOfTimesSteps = numberOfTimesSteps
        self.dsbIterationNumber = dsbIterationNumber

    def trainForward(self, backward: nn.Module, forward: nn.Module, nEpochs: int, T: float, lr: float, numberOfTimesSteps: int, priorIterator):
        optimizer = Optimizer.getOptimizer(params=forward.parameters(), lr=lr)

        for _ in tqdm(range(nEpochs)):
            for _, X in priorIterator:
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

    def trainBackward(self, backward: nn.Module, forward: nn.Module, nEpochs: int, T: float, lr: float, numberOfTimesSteps: int, XtrainIterator):

        optimizer = Optimizer.getOptimizer(params=backward.parameters(), lr=lr)

        for _ in tqdm(range(nEpochs)):
            for _, X in XtrainIterator:

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

    def train(self, backward: nn.Module | None = None, forward: nn.Module | None = None,  nEpochs: int | None = None, lr: float | None = None, T: float | None = None, nChannels: int | None = None, numberOfTimesSteps: int | None = None, dsbIterationNumber: int | None = None):

        # sadly no ?? in python :( lol
        def getOtherIfNone(x, defaultX) -> Any:
            if x == None:
                return defaultX
            return x

        nChannelsUsed: int = getOtherIfNone(nChannels, self.nChannels)

        backwardUsed: nn.Module = getOtherIfNone(
            backward, PredManager.getBackwardPredUntrained(numberOfChannels=nChannelsUsed))
        forwardUsed: nn.Module = getOtherIfNone(
            forward, PredManager.getForwardPredUntrained(numberOfChannels=nChannelsUsed))

        dsbIterationNumberUsed = getOtherIfNone(
            dsbIterationNumber, self.dsbIterationNumber)

        nEpochsUsed = getOtherIfNone(nEpochs, self.epochs)

        lrUsed = getOtherIfNone(lr, self.lr)

        TUsed = getOtherIfNone(T, self.T)

        numberOfTimesStepsUsed: int = getOtherIfNone(
            numberOfTimesSteps, self.numberOfTimesSteps)

        for _ in range(dsbIterationNumberUsed):
            self.trainBackward(backward=backwardUsed, forward=forwardUsed, nEpochs=nEpochsUsed,
                               T=TUsed, lr=lrUsed, numberOfTimesSteps=numberOfTimesStepsUsed, XtrainIterator=None)

            self.trainForward(backward=backwardUsed, forward=forwardUsed, nEpochs=nEpochsUsed,
                              T=TUsed, lr=lrUsed, numberOfTimesSteps=numberOfTimesStepsUsed, priorIterator=None)


if isMain(__name__):
    epochs = 10
    lr = 0.01
    T = 10
    dsbIterationNumber = 10
    nChannels = 1
    numberOfTimesSteps = 10

    meanPredictorTrainer = MeanPredictorTrainer(
        epochs=epochs, lr=lr, T=T, nChannels=nChannels, numberOfTimesSteps=numberOfTimesSteps, dsbIterationNumber=dsbIterationNumber)

    datasetManager = DatasetManager()
