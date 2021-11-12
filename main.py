#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from imageio import imread
from difflib import SequenceMatcher
import pandas as pd
import re
import random
import tqdm

images_dir = './assets/images'
images_ext = ['jpg', 'jpeg', 'png']

images_max_learn_nb = 10
images_max_test_nb  = 10000000

pokemons_manual_associations = {
    # "MrMime"     :  "Mr. Mime"
}

display_names_associations     = True
display_max_number_reached     = True
display_unknowned_pokemon_name = True
display_unreachable_file_name  = True
display_detailed_results       = False

global pokedex

""" TRAITEMENT PRINCIPAL """

def getPokedex():
    
    print()
    print("<========== LOADING POKEDEX ==========>")
    print()

    file = pd.read_csv(
        './assets/csv/pokedex.csv',
        usecols=[ 'pokedex_number', 'name' ]
    )
    
    N = len(file)
    pokedex = dict()
    for i in range(N):
        pokedex[file['name'][i]] = file['pokedex_number'][i]

    print(str(N) + " pokemons loaded!")
    
    print()
    print("<=====================================>")
    print()
    
    return pokedex

def getMaxReliableStringRatio(a, b):
    lenA = len(a)
    lenB = len(b)

    if lenA > lenB:
        (a, b) = (b, a)
        (lenA, lenB) = (lenB, lenA)

    if (a in b or b in a):
        return 1.0

    return max([ 
        SequenceMatcher(None, a, b[i:i+lenA]).ratio() 
        for i in range(lenB - lenA + 1) 
    ])

def getLearningDataset():
    
    print("<====== LOADING LEARNING DATASET =====>")
    print()

    # Get all images
    globs = []
    datasets = ['CompletePokemonImageDataset', 'PokemonGenerationOne']
    for dataset in datasets:
        for ext in images_ext:
            paths = [images_dir + '/' + dataset + '/data/shapes/*/*.' + ext]
            for path in paths:
                globs += glob.glob(os.path.join(path))
    
    # Associates images to pokemons
    pokedexKeys = pokedex.keys()
    pokemonDataset = {}
    replaced = {}
    unknowned = {}

    for i in tqdm.tqdm(range(min(len(globs), images_max_learn_nb))):
        gl = globs[i]

        # Get the pokemon id
        pokemonName = gl.split('\\')[1]
        if (pokemonName in pokemons_manual_associations):
            pokemonName = pokemons_manual_associations[pokemonName]

        if not(pokemonName in pokedex):
            if not(pokemonName in unknowned.keys()):
                similarPokemonName = max(pokedexKeys, key = lambda x : getMaxReliableStringRatio(pokemonName, x))
                ratio = getMaxReliableStringRatio(pokemonName, similarPokemonName)

                if (0.7 <= ratio):

                    if not(pokemonName in replaced.keys()):
                        replaced[pokemonName] = "INFOS => The pokemon \"" + pokemonName + "\" has been classified as \"" + similarPokemonName + "\" in the pokedex"
                    
                    pokemonName = similarPokemonName

                else:
                    unknowned[pokemonName] = "WARNING => Impossible to find the pokemon id of the name \"" + pokemonName + "\" in the pokedex"
                    continue
            else:
                continue
        
        pokemonId = pokedex[pokemonName]

        # Get the pokemon image
        pokemonImg = None # imread(gl)
        if pokemonId in pokemonDataset:
            pokemonDataset[pokemonId].append(pokemonImg)
        else:
            pokemonDataset[pokemonId] = [pokemonImg]

    # Check if max number of images reached
    if display_max_number_reached and i == images_max_learn_nb - 1:
        print()
        print("WARNING => Max images number reached ({}).".format(images_max_learn_nb))

    if display_names_associations and len(replaced) != 0:
        print()
        for msg in replaced.values():
            print(msg)

    if display_unknowned_pokemon_name and len(unknowned) != 0:
        print()
        for msg in unknowned.values():
            print(msg)

    print()
    print(str(i) + " images loaded!")

    print()
    print("<=====================================>")
    print()

    return pokemonDataset

def learn(learningDataset):

    print()
    print("<============== LEARNING =============>")
    print()

    print("AI is ready!")
    
    print()
    print("<=====================================>")
    print()

def makePrediction(image):
    return random.randint(0, len(pokedex))

def getTestingDataset():

    print()
    print("<====== LOADING TESTING DATASET ======>")
    print()
    
    # Get all images
    globs = []
    for ext in images_ext:
        paths = [images_dir + '/OneShotPokemon/data/shapes/*.' + ext]
        for path in paths:
            globs += glob.glob(os.path.join(path))
                
    # Associates images to pokemons
    pokemonDataset = {}
    pokemonIds = pokedex.values()
    unreachabled = []
    unknowned = {}
    for i in tqdm.tqdm(range(min(len(globs), images_max_test_nb))):
        gl = globs[i]

        # Get the pokemon id
        fileName = gl.split('\\')[1]
        intFounds = re.findall('\d+', fileName)

        if (len(intFounds) == 0):
            unreachabled.append("WARNING => No id found in the file name \"" + fileName + "\" (no integer found)")
            continue
                            
        pokemonId = int(intFounds[0])
        if not(pokemonId in pokemonIds):
            if not(pokemonId in unknowned.keys()):
                unknowned[pokemonId] = "WARNING => The pokemon id \"" + str(pokemonId) + "\" is not in the pokedex"
            continue

        # Get the pokemon image
        pokemonImg = None # imread(gl)
        if pokemonId in pokemonDataset:
            pokemonDataset[pokemonId].append(pokemonImg)
        else:
            pokemonDataset[pokemonId] = [pokemonImg]
        
    # Check if max number of images reached
    if display_max_number_reached and i == images_max_test_nb - 1:
        print()
        print("WARNING => Max images number reached ({}).".format(images_max_test_nb))

    if display_unreachable_file_name and len(unreachabled) != 0:
        print()
        for msg in unreachabled:
            print(msg)

    if display_unknowned_pokemon_name and len(unknowned) != 0:
        print()
        for msg in unknowned.values():
            print(msg)

    print()
    print(str(i) + " images loaded!")

    print()
    print("<=====================================>")
    print()

    return pokemonDataset

def makeTests(testDataset):

    print()
    print("<======== MAKE AUTOMATIC TESTS =======>")
    print()

    results = {}
    (success, error)  = (0, 0)
    for testResult, testEntries in tqdm.tqdm(testDataset.items()):
        predictedResult = makePrediction(testEntries)

        if not(testResult in results):
            results[testResult] = {
                'success' : 0,
                'error' : 0
            }

        if (testResult == predictedResult):
            success += 1
            results[testResult]['success'] += 1
        else:
            error += 1
            results[testResult]['error'] += 1
        
        results[testResult]['avg'] = results[testResult]['success'] / (results[testResult]['success'] +  results[testResult]['error'])

    total = success + error
    if (total != 0):
        print()
        print(str(total) + " tests made!")
        print()

        rate = success / total
        print("Total success rate : {0:.2f}%".format(rate)) 

        ratesByPokemon = map(lambda x : x['avg'], results.values())
        rate = sum(ratesByPokemon) / len(results)
        print("Success rate by pokemon : {0:.2f}%".format(rate))
    else:
        print("No test launched...")

    print("Success : ", success)
    print("Error   : ", error)

    if display_detailed_results:
        print()
        for pokemonName in sorted(results.keys()):
            print("INFO => \"{:03d}\" prediction rate = {:.2f}%".format(pokemonName, results[pokemonName]['avg']))       

    print()
    print("<=====================================>")
    print()

if __name__ == '__main__':

    pokedex = getPokedex()

    learningDataset = getLearningDataset()
    learn(learningDataset)

    testDataset = getTestingDataset()
    makeTests(testDataset)
