import torch
import sys
import random
import os
import json
from datetime import datetime
from math import inf

from const import isMain
from dataset import datasetManager
from trainer import MeanPredictorTrainer

import json


def saveModelsSimple(backwardModel, forwardModel, iterationLoss, paramsId, numericParams, baseDir="models"):
    lossStr = f"{iterationLoss:.4f}".replace('.', 'p').replace('-', 'neg')

    numericParamStr = "_".join([
        f"e{numericParams['epochs']}",
        f"lr{str(numericParams['lr']).replace('.', 'p')}",
        f"T{str(numericParams['T']).replace('.', 'p')}",
        f"dsb{numericParams['dsbIterationNumber']}",
        f"nts{numericParams['numberOfTimesSteps']}",
        f"ao{str(numericParams['alphaOu']).replace('.', 'p')}",
        f"prop{str(numericParams['proportion']).replace('.', 'p')}"
    ])

    runFolder = os.path.join(baseDir, "id_"+paramsId+"_" +
                             numericParamStr+"_"+lossStr)

    with open(os.path.join(runFolder, "paramsUsed.json"), "w") as f:
        f.write(json.dumps(numericParams))

    os.makedirs(runFolder, exist_ok=True)

    backFilename = os.path.join(
        runFolder, f"backward.pth")
    forwardFilename = os.path.join(
        runFolder, f"forward.pth")

    torch.save(backwardModel.state_dict(), backFilename)
    torch.save(forwardModel.state_dict(), forwardFilename)

    print(f"Models saved: {backFilename}, {forwardFilename}")
    return backFilename, forwardFilename


def trainModelWithParamAndSave(
    epochs, lr, T, dsbIterationNumber, nChannels, numberOfTimesSteps,
    alphaOu, xTrainIteratorFun, priorIteratorFun, proportion, paramsId, numericParams
):
    print(f"--- Starting Training Run (ID: {paramsId}) ---")
    print(f"Parameters: epochs={epochs}, lr={lr}, T={T}, dsbIterations={dsbIterationNumber}, "
          f"nChannels={nChannels}, numTimeSteps={numberOfTimesSteps}, alphaOu={alphaOu}, "
          f"proportion={proportion}")

    trainer = MeanPredictorTrainer(
        epochs=epochs, lr=lr, T=T, nChannels=nChannels,
        numberOfTimesSteps=numberOfTimesSteps, dsbIterationNumber=dsbIterationNumber,
        alphaOu=alphaOu, XtrainIteratorFun=xTrainIteratorFun,
        priorIteratorFun=priorIteratorFun, proportion=proportion
    )

    try:
        (backwardModel, forwardModel, currentLoss) = trainer.train()
        print(
            f"Training Run (ID: {paramsId}) Completed. Final DSB Iteration Loss: {currentLoss:.6f}")

        saveModelsSimple(backwardModel, forwardModel,
                         currentLoss, paramsId, numericParams)
        print(f"Models saved successfully for run {paramsId}.")
        return backwardModel, forwardModel, currentLoss

    except Exception as e:
        print(f"Training Run (ID: {paramsId}) FAILED with error: {e}")
        return None, None, inf


if isMain(__name__):
    paramsJsonPath = sys.argv[1]
    try:
        with open(paramsJsonPath, 'r') as f:
            config = json.load(f)
        paramSpace = config['paramSpace']
        numRandomTrials = config['numRandomTrials']
        print(f"Parameters loaded from {paramsJsonPath}")
    except FileNotFoundError:
        print(f"Error: {paramsJsonPath} not found. Please create it.")
        exit()
    except json.JSONDecodeError:
        print(
            f"Error: Could not decode JSON from {paramsJsonPath}. Check file format.")
        exit()
    except KeyError as e:
        print(
            f"Error: Missing key {e} in {paramsJsonPath}. Check file content.")
        exit()

    fixedParams = {
        'xTrainIteratorFun': datasetManager.trainEntries,
        'priorIteratorFun': datasetManager.priorEntriesTrain,
    }

    bestLoss = inf
    bestParams = None
    bestParamsId = None

    print(
        f"Starting Hyperparameter Search with {numRandomTrials} random trials.")

    for trialIdx in range(numRandomTrials):
        print(f"\n--- Trial {trialIdx + 1}/{numRandomTrials} ---")

        currentParams = {k: random.choice(v) for k, v in paramSpace.items()}
        currentParams.update(fixedParams)

        paramsId = f"{datetime.now().strftime('%H%M%S')}_{trialIdx}"

        print(f"Trial {trialIdx + 1} Parameters: {currentParams}")

        numericParamsForFilename = {
            'epochs': currentParams['epochs'],
            'lr': currentParams['lr'],
            'T': currentParams['T'],
            'dsbIterationNumber': currentParams['dsbIterationNumber'],
            'numberOfTimesSteps': currentParams['numberOfTimesSteps'],
            'alphaOu': currentParams['alphaOu'],
            'proportion': currentParams['proportion']
        }

        backwardModel, forwardModel, currentLoss = trainModelWithParamAndSave(
            epochs=currentParams['epochs'],
            lr=currentParams['lr'],
            T=currentParams['T'],
            dsbIterationNumber=currentParams['dsbIterationNumber'],
            nChannels=currentParams['nChannels'],
            numberOfTimesSteps=currentParams['numberOfTimesSteps'],
            alphaOu=currentParams['alphaOu'],
            xTrainIteratorFun=currentParams['xTrainIteratorFun'],
            priorIteratorFun=currentParams['priorIteratorFun'],
            proportion=currentParams['proportion'],
            paramsId=paramsId,
            numericParams=numericParamsForFilename
        )

        if currentLoss < bestLoss:
            bestLoss = currentLoss
            bestParams = currentParams
            bestParamsId = paramsId
            print(
                f"NEW BEST LOSS: {bestLoss:.6f} with parameters: {bestParams}")

    print("\n--- Hyperparameter Search Finished ---")
    print(f"Best Loss Found: {bestLoss:.6f}")
    print(f"Best Parameters: {bestParams}")
    print(f"Best Run ID: {bestParamsId}")
