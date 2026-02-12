from math import sqrt
from torch import nn
import torch

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
    forward = PredManager.getForwardPredUntrained()
    backward = PredManager.getBackwardPredUntrained()
    calcDevice = DEVICE
