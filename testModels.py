import torch
import time
import os
import matplotlib.pyplot as plt
import sys
import json

from const import DEVICE
from meanPredictor import PredManager
from predictor import Predictor
from dataset import datasetManager


def testModelAndGenerateImages(modelFolderPath: str, numImages: int, displayTime: float, numberOfTimesSteps: int | None = None, T: float | None = None):
    paramsFilePath = os.path.join(modelFolderPath, "paramsUsed.json")
    try:
        with open(paramsFilePath, 'r') as f:
            params = json.load(f)
    except FileNotFoundError:
        print(f"Error: paramsUsed.json not found in {modelFolderPath}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {paramsFilePath}")
        return

    nChannels = params.get('nChannels', 1)

    if T == None:
        T = params.get('T', 1.0)

    if numberOfTimesSteps == None:
        numberOfTimesSteps = params.get('numberOfTimesSteps', 22)

    forwardModel = PredManager.getForwardPredUntrained(
        numberOfChannels=nChannels)
    backwardModel = PredManager.getBackwardPredUntrained(
        numberOfChannels=nChannels)

    backwardPath = os.path.join(modelFolderPath, "backward.pth")
    forwardPath = os.path.join(modelFolderPath, "forward.pth")

    try:
        backwardModel.load_state_dict(
            torch.load(backwardPath, map_location=DEVICE))
        forwardModel.load_state_dict(
            torch.load(forwardPath, map_location=DEVICE))
    except FileNotFoundError:
        print(f"Error: Model files not found in {modelFolderPath}")
        return
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        return

    backwardModel.eval()
    forwardModel.eval()

    predictor = Predictor(
        backward=backwardModel,
        forward=forwardModel,
        numberOfTimesSteps=numberOfTimesSteps,
        T=T,
        calcDevice=DEVICE
    )

    imageShape = (nChannels, datasetManager.shape[0], datasetManager.shape[1])

    print(
        f"Generating {numImages} images from model: {os.path.basename(modelFolderPath)}...")
    print(
        f"Displaying each image for {displayTime} seconds. Close the window to exit early.")

    for i in range(numImages):
        randomPrior = torch.randn(1, *imageShape, device=DEVICE)
        generatedImage = predictor.generateBackward(randomPrior)

        randomPriorCpu = randomPrior.squeeze().cpu()
        generatedImageCpu = generatedImage.squeeze().cpu()

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(randomPriorCpu, cmap="gray")
        plt.title(f"Random Prior (Noise) - Image {i+1}")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(generatedImageCpu, cmap="gray")
        plt.title(f"Generated Image (Backward Process) - Image {i+1}")
        plt.axis('off')

        plt.suptitle(f"Model: {os.path.basename(modelFolderPath)}")
        plt.show(block=False)
        plt.pause(displayTime)
        plt.close()

    print("Finished displaying images.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python testModels.py path_to_model_folder [optional_number_of_images] [optional_display_time_seconds] [optional_number_of_times_steps] [optional_T] ")
        print("Example: python testModels.py models/id_000254_1e5_lr0p0001_T1p0_dsb4_nts5_ao1p0_prop0p03_0p2037")
        sys.exit(1)

    modelFolderPath = sys.argv[1]
    numberOfTimesSteps = None
    T = None
    numberOfImages = 10
    displayTime = 0.6

    if len(sys.argv) >= 3:
        numberOfImages = int(sys.argv[2])

    if len(sys.argv) >= 4:
        displayTime = float(sys.argv[3])

    if len(sys.argv) >= 5:
        numberOfTimesSteps = int(sys.argv[4])

    if len(sys.argv) >= 6:
        T = float(sys.argv[5])

    testModelAndGenerateImages(
        modelFolderPath, numImages=numberOfImages, numberOfTimesSteps=numberOfTimesSteps, T=T, displayTime=displayTime)
