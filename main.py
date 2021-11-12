#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from imageio import imread
from difflib import SequenceMatcher
import pandas as pd

images_dir = './assets/images'
images_ext = ['jpg', 'jpeg', 'png']

images_max_learn_nb = 1000
images_max_test_nb = 1000

# Cf printSimilarPokemonNames()
pokemons_manual_associations = {
    "Farfetch'd":  "Farfetchd",
    "Mr. Mime":  "MrMime"
}

""" TRAITEMENT PRINCIPAL """

def getLearningDataset():
    pokemonDataset = {}

    datasets = ['CompletePokemonImageDataset', 'PokemonGenerationOne']
    i = 0
    for dataset in datasets:
        for ext in images_ext:
            paths = [images_dir + '/' + dataset + '/data/shapes/*/*.' + ext]
            for path in paths:
                for gl in glob.glob(os.path.join(path)):

                    if i == images_max_learn_nb:
                        return pokemonDataset
                    i += 1

                    pokemonName = gl.split('\\')[1]
                    if (pokemonName in pokemons_manual_associations):
                        pokemonName = pokemons_manual_associations[pokemonName]

                    pokemonImg = imread(gl)

                    if pokemonName in pokemonDataset:
                        pokemonDataset[pokemonName].append(pokemonImg)
                    else:
                        pokemonDataset[pokemonName] = [pokemonImg]

    return pokemonDataset

def learn(learningDataset):
    pass

def predictPokemon(image):
    return "Pikachu"

def getTestingDataset():
    csv = pd.read_csv(
        './assets/csv/pokedex.csv',
        names=['pokemon_name', 'pokedex_number']
    )
    print(csv)
    return {}

    pokemonDataset = {}

    i = 0
    for ext in images_ext:
        paths = [images_dir + '/OneShotPokemon/data/shapes/*.' + ext]
        for path in paths:
            for gl in glob.glob(os.path.join(path)):

                if i == images_max_test_nb:
                    return pokemonDataset
                i += 1

                pokemonName = gl.split('\\')[1]
                if (pokemonName in pokemons_manual_associations):
                    pokemonName = pokemons_manual_associations[pokemonName]

                pokemonImg = imread(gl)

                if pokemonName in pokemonDataset:
                    pokemonDataset[pokemonName].append(pokemonImg)
                else:
                    pokemonDataset[pokemonName] = [pokemonImg]

    return pokemonDataset

def makeTests(testDataset):

    (success, error)  = (0, 0)
    for testResult, testEntries in testDataset:
        predictedResult = predict(testEntries)
        if (testResult == predictedResult):
            sucess += 1
        else:
            error += 1

    if (success + error != 0):
        print("Success rate : ", success / (success + error))
    else:
        print("No test launched...")

    print("Success : ", success)
    print("Error   : ", error)

""" DEBUG """

"""
    Cette fonction permet de lister les pokémons ayant des noms très similaires
    au sein de datasets CompletePokemonImageDataset et OneShotPokemon. En effet,
    ceux-ci sont répertoriés par leur nom dans le dataset et par l'id du pokédex,
    d'où de potentielles erreurs d'identification.

    En particulier, nous avons contaté des incohérences entre les datasets concernant
    les pokémons suivants :
    - Farfetch'd  =>  Farfetchd ;
    - Mr. Mime    =>  MrMime    ;
"""

def printSimilarPokemonNames():
    datasets = ['CompletePokemonImageDataset', 'PokemonGenerationOne']

    d = dict()
    for dataset in datasets:
        s = set()
        for ext in images_ext:
            paths = [images_dir + '/' + dataset + '/data/shapes/*/*.' + ext]
            for path in paths:
                for gl in glob.glob(os.path.join(path)):
                    s.add(gl.split('\\')[1])
        d[dataset] = s

    similarElements = []
    for p0 in d[datasets[0]]:
        for p1 in d[datasets[1]]:
            ratio = SequenceMatcher(None, p0, p1).ratio()
            if (0.8 < ratio and ratio < 1):
                if not((p1, p0) in similarElements):
                    similarElements.append((p0, p1))
                    print(p0, p1)

if __name__ == '__main__':

    # learningDataset = getLearningDataset()

    # learn(learningDataset)

    testDataset = getTestingDataset()
    
    # makeTests(testDataset)

    # printSimilarPokemonNames()