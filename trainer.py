from math import sqrt
from dataset import datasetManager
from typing import Any
from torch import nn
import torch
from tqdm import tqdm

from const import DEVICE, Optimizer, isMain, lossFn
from meanPredictor import OUMeanPredictor, PredManager


class MeanPredictorTrainer():
    def __init__(self, epochs: int, lr: float, T: float, nChannels: int, numberOfTimesSteps: int, dsbIterationNumber: int, alphaOu: float, XtrainIteratorFun, priorIteratorFun, proportion: float) -> None:
        self.epochs = epochs
        self.lr = lr
        self.T = T
        self.nChannels = nChannels
        self.numberOfTimesSteps = numberOfTimesSteps
        self.dsbIterationNumber = dsbIterationNumber
        self.alphaOu = alphaOu
        self.XtrainIteratorFun = XtrainIteratorFun
        self.priorIteratorFun = priorIteratorFun
        self.proportion = proportion

    def trainForward(self, backward: nn.Module, forward: nn.Module, nEpochs: int, T: float, lr: float, numberOfTimesSteps: int, priorIteratorFun, proportion):
        optimizer = Optimizer.getOptimizer(params=forward.parameters(), lr=lr)

        print(f"Starting Forward Training with {nEpochs} epochs, lr={lr}")

        meanEpochLoss = 0
        for epoch in tqdm(range(nEpochs), desc="Forward Epochs"):
            epochLoss = 0.0
            batchCount = 0

            for batchIndex, X in priorIteratorFun(proportion):
                forward.train()
                optimizer.zero_grad()

                curX: torch.Tensor = X
                listX: list[torch.Tensor] = [X]

                # Forward pass through time steps
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
                step_losses = []
                # Compute loss across time steps
                for i in range(numberOfTimesSteps):
                    curX = listX.pop()

                    yPred = forward(oldX, i)

                    backward.eval()
                    with torch.inference_mode():
                        yTrue = oldX+backward(curX, i+1) - \
                            backward(oldX, i+1)

                    step_loss = lossFn(yPred, yTrue.clone())
                    loss += step_loss
                    step_losses.append(step_loss.item())

                    oldX = curX

                loss /= numberOfTimesSteps
                epochLoss += loss.item()
                batchCount += 1

                print(
                    f"  Batch {batchIndex} - Average loss: {loss.item():.6f}")

                loss.backward()

                optimizer.step()

            avgEpochLoss = epochLoss / batchCount if batchCount > 0 else 0
            print(
                f"\nForward - Epoch {epoch+1}/{nEpochs}, Average Loss: {avgEpochLoss:.6f}\n\n")

            meanEpochLoss += avgEpochLoss

        if nEpochs > 0:
            meanEpochLoss = meanEpochLoss/nEpochs

        print("Forward Training Completed")

        return forward, meanEpochLoss

    def trainBackward(self, backward: nn.Module, forward: nn.Module, nEpochs: int, T: float, lr: float, numberOfTimesSteps: int, XtrainIteratorFun, proportion):

        optimizer = Optimizer.getOptimizer(params=backward.parameters(), lr=lr)

        print(f"Starting Backward Training with {nEpochs} epochs, lr={lr}")

        meanEpochLoss = 0
        for epoch in tqdm(range(nEpochs), desc="Backward Epochs"):
            epochLoss = 0.0
            batchCount = 0

            for batchIndex, X in XtrainIteratorFun(proportion):
                backward.train()
                optimizer.zero_grad()

                curX: torch.Tensor = X
                listX: list[torch.Tensor] = [X]

                # Backward pass through time steps
                for i in range(numberOfTimesSteps):
                    dt = T/numberOfTimesSteps

                    forward.eval()
                    with torch.inference_mode():
                        curX = forward(curX, i)+sqrt(2)*sqrt(dt) * \
                            torch.randn(curX.shape).to(DEVICE)

                    listX.append(curX.clone())

                loss = torch.zeros(1).to(DEVICE)

                oldX = listX.pop()
                step_losses = []
                # Compute loss across time steps
                for i in range(numberOfTimesSteps):
                    curX = listX.pop()

                    timeStep = numberOfTimesSteps-i-1

                    yPred = backward(oldX, timeStep+1)

                    forward.eval()
                    with torch.inference_mode():
                        yTrue = oldX+forward(curX, timeStep) - \
                            forward(oldX, timeStep)

                    step_loss = lossFn(yPred, yTrue.clone())
                    loss += step_loss
                    step_losses.append(step_loss.item())

                    oldX = curX

                loss /= numberOfTimesSteps
                epochLoss += loss.item()
                batchCount += 1

                print(
                    f"  Batch {batchIndex} - Average loss: {loss.item():.6f}")

                loss.backward()

                optimizer.step()

            avgEpochLoss = epochLoss / batchCount if batchCount > 0 else 0
            print(
                f"\nBackward - Epoch {epoch+1}/{nEpochs}, Average Loss: {avgEpochLoss:.6f}\n\n")
            meanEpochLoss += avgEpochLoss

        if nEpochs > 0:
            meanEpochLoss = meanEpochLoss/nEpochs
        print("Backward Training Completed")
        return backward, meanEpochLoss

    def train(self, priorIterator: Any | None = None, XtrainIteraton: Any | None = None, backward: nn.Module | None = None, forward: nn.Module | None = None,  nEpochs: int | None = None, lr: float | None = None, T: float | None = None, nChannels: int | None = None, numberOfTimesSteps: int | None = None, dsbIterationNumber: int | None = None, alphaOu=None, proportion: float | None = None):

        # sadly no ?? in python :( lol
        def getOtherIfNone(x, defaultX) -> Any:
            if x == None:
                return defaultX
            return x

        XtrainIteratonUsedFun: Any = getOtherIfNone(
            XtrainIteraton, self.XtrainIteratorFun)

        priorIteratorUsedFun: Any = getOtherIfNone(
            priorIterator, self.priorIteratorFun)

        nChannelsUsed: int = getOtherIfNone(nChannels, self.nChannels)

        backwardUsed: nn.Module = getOtherIfNone(
            backward, PredManager.getBackwardPredUntrained(numberOfChannels=nChannelsUsed))

        wasForwadGiven = forward != None

        forwardUsed: nn.Module = getOtherIfNone(
            forward, PredManager.getForwardPredUntrained(numberOfChannels=nChannelsUsed))

        dsbIterationNumberUsed = getOtherIfNone(
            dsbIterationNumber, self.dsbIterationNumber)

        nEpochsUsed = getOtherIfNone(nEpochs, self.epochs)

        lrUsed = getOtherIfNone(lr, self.lr)

        TUsed = getOtherIfNone(T, self.T)

        numberOfTimesStepsUsed: int = getOtherIfNone(
            numberOfTimesSteps, self.numberOfTimesSteps)

        alphaOuUsed = getOtherIfNone(alphaOu, self.alphaOu)

        proportionUsed = getOtherIfNone(proportion, self.proportion)

        ouProcess = OUMeanPredictor(
            alphaOu=alphaOuUsed, T=TUsed, numberOfTimesSteps=numberOfTimesStepsUsed)

        print(f"Training on {proportionUsed*100}% of the data")

        lastLoss = 0

        for i in range(dsbIterationNumberUsed):

            print("="*60)
            print(f"DSB ITERATION NUMBER: {i+1}/{dsbIterationNumberUsed}")
            print("="*60)

            # For the first backward training
            # We used OU process as a baseline instead of the neural forward neural networks
            # If someone give use the forward we use it as a baseline
            curForward = ouProcess if i == 0 and not wasForwadGiven else forwardUsed

            print("@"*60)
            print("Starting Backward Training Phase")
            backwardUsed, backMeanLoss = self.trainBackward(backward=backwardUsed, forward=curForward, nEpochs=nEpochsUsed,
                                                            T=TUsed, lr=lrUsed, numberOfTimesSteps=numberOfTimesStepsUsed, XtrainIteratorFun=XtrainIteratonUsedFun, proportion=proportionUsed)

            print("#"*60)
            print("\n\nStarting Forward Training Phase")
            forwardUsed, forwardMeanLoss = self.trainForward(backward=backwardUsed, forward=forwardUsed, nEpochs=nEpochsUsed,
                                                             T=TUsed, lr=lrUsed, numberOfTimesSteps=numberOfTimesStepsUsed, priorIteratorFun=priorIteratorUsedFun, proportion=proportionUsed)

            lastLoss = (backMeanLoss+forwardMeanLoss)/2

        return (backwardUsed, forwardUsed, lastLoss)


if isMain(__name__):
    epochs = 10
    lr = 1e-4
    T = 1.0
    dsbIterationNumber = 5
    nChannels = 1
    numberOfTimesSteps = 12
    alphaOu = 1
    XtrainIteratorFun = datasetManager.trainEntries
    priorIteratorFun = datasetManager.priorEntriesTrain
    proportion = 0.005

    meanPredictorTrainer = MeanPredictorTrainer(
        epochs=epochs, lr=lr, T=T, nChannels=nChannels, numberOfTimesSteps=numberOfTimesSteps, dsbIterationNumber=dsbIterationNumber, alphaOu=alphaOu, XtrainIteratorFun=XtrainIteratorFun, priorIteratorFun=priorIteratorFun, proportion=proportion)

    meanPredictorTrainer.train()
