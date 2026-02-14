from math import sqrt
import matplotlib.pyplot as plt
from torch import nn
import torch
from dataset import datasetManager

from const import DEVICE, isMain
from meanPredictor import PredManager


class Predictor:
    def __init__(self, backward: nn.Module, forward: nn.Module, numberOfTimesSteps: int, T: float, calcDevice: str) -> None:
        self.backward = backward.to(device=calcDevice)
        self.forward = forward.to(device=calcDevice)
        self.numberOfTimesSteps = numberOfTimesSteps
        self.T = T
        self.device = calcDevice

    def getBackwardTrajectory(self, X: torch.Tensor) -> list[torch.Tensor]:
        out = [X]

        orginalDevice = X.device.type

        X = X.to(self.device)
        numberOfTimesSteps = self.numberOfTimesSteps
        T = self.T

        curX: torch.Tensor = X
        for i in range(numberOfTimesSteps):
            dt = T/numberOfTimesSteps

            timeStep = numberOfTimesSteps-i-1
            self.backward.eval()
            with torch.inference_mode():
                curX = self.backward(curX, timeStep+1)+sqrt(2)*sqrt(dt) * \
                    torch.randn(curX.shape).to(DEVICE)

            out.append(curX.to(orginalDevice))

        return out

    def getForwardTrajectory(self, X: torch.Tensor) -> list[torch.Tensor]:
        out = [X]

        orginalDevice = X.device.type

        out: list[torch.Tensor] = [X]

        X = X.to(self.device)

        numberOfTimesSteps = self.numberOfTimesSteps
        T = self.T

        forward = self.forward

        curX: torch.Tensor = X

        for i in range(numberOfTimesSteps):
            dt = T/numberOfTimesSteps

            forward.eval()
            with torch.inference_mode():
                curX = forward(curX, i)+sqrt(2)*sqrt(dt) * \
                    torch.randn(curX.shape).to(DEVICE)

            out.append(curX.to(orginalDevice))

        return out

    def generateBackward(self, X: torch.Tensor):
        trajectories = self.getBackwardTrajectory(X)
        return trajectories[-1]

    def generateFoward(self, X: torch.Tensor):
        trajectories = self.getForwardTrajectory(X)
        return trajectories[-1]


if isMain(__name__):
    forward = PredManager.getForwardPredUntrained(numberOfChannels=1)
    backward = PredManager.getBackwardPredUntrained(numberOfChannels=1)

    # forward.load_state_dict(torch.load(
    #    "models/forward_id_195913_0_loss_0p1241.pth"))

    # backward.load_state_dict(torch.load(
    #    "models/back_id_195913_0_loss_0p1241.pth"))

    # Seem like the best so far
    # Correspond to 5 epochs
    # lr ~ 1e-4 T= 1, dsb=4,numberOfTimesSteps =  1, alpha = 1,proportions=3%
    forward.load_state_dict(torch.load(
        "models/id_000254_1e5_lr0p0001_T1p0_dsb4_nts5_ao1p0_prop0p03_0p2037/forward.pth"))

    backward.load_state_dict(torch.load(
        "models/id_000254_1e5_lr0p0001_T1p0_dsb4_nts5_ao1p0_prop0p03_0p2037/backward.pth"))

    calcDevice = DEVICE

    predictor = Predictor(backward=backward, forward=forward,
                          calcDevice=calcDevice, T=22, numberOfTimesSteps=100)

    """
    for i in range(2):
        X = datasetManager.getRandomTrain().unsqueeze(0)
        forwardX = predictor.generateFoward(X)

        X = X.cpu()
        forwardX = forwardX.cpu()

        plt.imshow(X.squeeze())
        plt.show()

        plt.imshow(forwardX.squeeze())
        plt.show()
        """

    for i in range(10):
        X = datasetManager.getRandomPriors().unsqueeze(0)

        print(X.shape)

        forwardX = predictor.generateBackward(X)

        X = X.cpu()
        forwardX = forwardX.cpu()

        plt.imshow(X.squeeze(), )
        plt.show()

        plt.imshow(forwardX.squeeze(), )
        plt.show()
