#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from imageio import imread
from difflib import SequenceMatcher

images_max_nb = 1000
images_dir = './images'
images_ext = ['jpg', 'jpeg', 'png']

# Cf printSimilarPokemonNames()
pokemons_manual_associations = {
    "Farfetch'd":  "Farfetchd",
    "Mr. Mime":  "MrMime"
}

""" TRAITEMENT PRINCIPAL """

def getPokemonDataset():
    pokemonDataset = {}

    datasets = ['CompletePokemonImageDataset', 'PokemonGenerationOne']
    i = 0
    for dataset in datasets:
        for ext in images_ext:
            paths = [images_dir + '/' + dataset + '/data/shapes/*/*.' + ext]
            for path in paths:
                for gl in glob.glob(os.path.join(path)):

                    if i == images_max_nb:
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

    pokemonDataset = getPokemonDataset()

    # printSimilarPokemonNames()