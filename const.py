import torch
import numpy as np
import random

DATA_PATH = "./data"

MAIN = "__main__"


class Devices:
    cuda = "cuda"
    cpu = "cpu"


def isMain(name):
    return name == MAIN


SEED = None


DEVICE = Devices.cuda if torch.cuda.is_available() else Devices.cpu


lossFn = torch.nn.MSELoss()


class ParamsMeanPredictor:
    numEncoders = 3
    numDecoders = 3
    conv2DKernelSize = (3, 3)
    conv2DPadding = 1
    maxPool2DKernelSize = (2, 2)
    maxPoolingStride = 2
    numGroups = 32
    firstExpansion = 64
    dimPosEncoding = 64


class Optimizer:
    @staticmethod
    def getOptimizer(params, lr):
        return torch.optim.AdamW(params=params, lr=lr)


class BatchSize:
    test = 1000
    train = 32


def seed():
    if SEED == None:
        return
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
