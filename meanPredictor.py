import matplotlib.pyplot as plt
from torch import nn
import torch

from const import DEVICE, Devices, ParamsMeanPredictor, isMain
from dataset import DatasetManager
from filmLayer import FilmLayer
from positionalEncoder import PositionalEncoder


class MeanPredictor(nn.Module):
    def __init__(self,  nChannels: int) -> None:
        super().__init__()

        self.positionalEncoder = PositionalEncoder(
            posisionalEncodingSize=ParamsMeanPredictor.dimPosEncoding,)

        prevChannels = nChannels

        allPartsExceptLastLayers = []

        def wrap(e):
            allPartsExceptLastLayers.append(e)
            return e

        def getFilm(numChannels):
            return FilmLayer(positionalEncodingSize=ParamsMeanPredictor.dimPosEncoding, nbFeatures=numChannels)

        def ConvAndNorm(inChannels, outChannels):
            return nn.Sequential(
                nn.Conv2d(in_channels=inChannels,
                          out_channels=outChannels, kernel_size=ParamsMeanPredictor.conv2DKernelSize, padding=ParamsMeanPredictor.conv2DPadding),
                nn.GroupNorm(num_channels=outChannels,
                             num_groups=ParamsMeanPredictor.numGroups),
            )

        # Encoder

        def getEncoderFrom(prevChannels, nextChannels):

            firstPart = wrap(ConvAndNorm(
                prevChannels,
                nextChannels
            ))

            filmFirstPart = wrap(getFilm(nextChannels))

            firstActivation = nn.ReLU()

            secondPart = wrap(ConvAndNorm(
                nextChannels,
                nextChannels
            ))

            filmSecondPart = wrap(getFilm(nextChannels))

            secondActivation = nn.ReLU()

            afterEncoder = nn.MaxPool2d(
                kernel_size=ParamsMeanPredictor.maxPool2DKernelSize,
                stride=ParamsMeanPredictor.maxPoolingStride
            )

            return (firstPart, filmFirstPart, firstActivation, secondPart, filmSecondPart, secondActivation, afterEncoder)

        encoders = []
        for _ in range(ParamsMeanPredictor.numEncoders):
            nextChannels = max(
                2*prevChannels, ParamsMeanPredictor.firstExpansion)

            encoder = getEncoderFrom(prevChannels, nextChannels)

            prevChannels = nextChannels

            encoders.append(encoder)

        self.encoders = encoders

        # Now the Bottleneck
        def getBottleneck(prevChannels, nextChannels):

            firstPartBottleneck = wrap(ConvAndNorm(inChannels=prevChannels,
                                                   outChannels=nextChannels, ))

            filmFirstBotttleneck = wrap(getFilm(nextChannels))

            firstActivation = nn.ReLU()

            secondPartBottleneck = wrap(ConvAndNorm(inChannels=nextChannels,
                                                    outChannels=nextChannels))

            filmSecondBotttleneck = wrap(getFilm(nextChannels))

            secondActivation = nn.ReLU()

            return (firstPartBottleneck, filmFirstBotttleneck, firstActivation, secondPartBottleneck, filmSecondBotttleneck, secondActivation)

        nextChannels = 2*prevChannels

        bottleneck = getBottleneck(prevChannels, nextChannels)

        prevChannels = nextChannels

        self.bottleneck = bottleneck

        # Decoder pipeline
        def getDecoderFrom(prevChannels, nextChannels):
            beforeDecoder = wrap(nn.ConvTranspose2d(in_channels=prevChannels,
                                                    out_channels=nextChannels, kernel_size=ParamsMeanPredictor.maxPool2DKernelSize, stride=ParamsMeanPredictor.maxPoolingStride))

            firstPartDecoder = wrap(ConvAndNorm(inChannels=2*nextChannels,
                                                outChannels=nextChannels))

            filmFirstPart = wrap(getFilm(nextChannels))

            firstActivation = nn.ReLU()

            secondPartDecoder = wrap(ConvAndNorm(inChannels=nextChannels,
                                                 outChannels=nextChannels))

            filmSecondPart = wrap(getFilm(nextChannels))

            secondActivation = nn.ReLU()

            return (beforeDecoder, firstPartDecoder, filmFirstPart, firstActivation, secondPartDecoder, filmSecondPart, secondActivation)

        decoders = []
        for _ in range(ParamsMeanPredictor.numDecoders):
            nextChannels = prevChannels//2

            decoders.append(getDecoderFrom(prevChannels, nextChannels))

            prevChannels = nextChannels

        self.decoders = decoders

        # Last layer to get back our image dimension
        self.lastLayer = nn.Conv2d(in_channels=prevChannels,
                                   out_channels=nChannels, kernel_size=1, padding=0)

        self.allPartsExceptLastLayers = nn.ModuleList(allPartsExceptLastLayers)

    def forward(self, X: torch.Tensor, pos: int):

        posVector = self.positionalEncoder.encode(k=pos)

        def getValueFromFilmAndPos(film, tens):
            filmOutput: torch.Tensor = film(posVector).unsqueeze(0)

            n = filmOutput.shape[1]

            alpha = filmOutput[:, :n//2].unsqueeze(-1).unsqueeze(-1)
            beta = filmOutput[:, n//2:].unsqueeze(-1).unsqueeze(-1)

            return alpha * tens + beta

        nb = len(self.encoders)

        out = X

        skipConnectionValues = []

        for i in range(nb):
            (firstPartEncoder, filmFirstPartEncoder, firstActivationEncoder, secondPartEncoder,
             filmSecondPartEncoder, secondActivationEncoder, afterEncoder) = self.encoders[i]

            out = firstPartEncoder(out)

            out = getValueFromFilmAndPos(filmFirstPartEncoder, out)

            out = firstActivationEncoder(out)

            out = secondPartEncoder(out)

            out = getValueFromFilmAndPos(filmSecondPartEncoder, out)

            out = secondActivationEncoder(out)

            skipConnectionValues.append(out)

            out = afterEncoder(out)

        skipConnectionValues.reverse()

        (firstPartBottleneck, filmFirstBotttleneck, firstActivationBottleneck,
         secondPartBottleneck, filmSecondBotttleneck, secondActivationBottleneck) = self.bottleneck

        out = firstPartBottleneck(out)

        out = getValueFromFilmAndPos(filmFirstBotttleneck, out)

        out = firstActivationBottleneck(out)

        out = secondPartBottleneck(out)

        out = getValueFromFilmAndPos(filmSecondBotttleneck, out)

        out = secondActivationBottleneck(out)

        for i in range(nb):
            (beforeDecoder, firstPartDecoder, filmFirstPartDecoder, firstActivationDecoder,
             secondPartDecoder, filmSecondPartDecoder, secondActivationDecoder) = self.decoders[i]

            out = beforeDecoder(out)

            skipValue = skipConnectionValues[i]

            out = torch.concat([out, skipValue], dim=1)

            out = firstPartDecoder(out)

            out = getValueFromFilmAndPos(filmFirstPartDecoder, out)

            out = firstActivationDecoder(out)

            out = secondPartDecoder(out)

            out = getValueFromFilmAndPos(filmSecondPartDecoder, out)

            out = secondActivationDecoder(out)

        out = self.lastLayer(out)

        return out


class PredManager:
    @staticmethod
    def getForwardPredUntrained(numberOfChannels):
        return MeanPredictor(nChannels=numberOfChannels).to(DEVICE)

    @staticmethod
    def getBackwardPredUntrained(numberOfChannels):
        return MeanPredictor(nChannels=numberOfChannels).to(DEVICE)


if isMain(__name__):
    mPred = PredManager.getForwardPredUntrained(
        DatasetManager.numberOfChannels)
    datasetManager = DatasetManager()

    for index in range(3):

        X = datasetManager.getRandomTrain().unsqueeze(0)

        with torch.inference_mode():
            out = mPred(X, 1)

        imgIn = X.squeeze().to(Devices.cpu)

        plt.imshow(imgIn)

        plt.show()

        imgOut = out.squeeze().to(Devices.cpu)

        plt.imshow(imgOut)

        plt.show()
