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

pokedex_path = './assets/csv/pokedex.csv'
images_dir = './assets/images'
images_ext = ['jpg', 'jpeg', 'png']

images_max_learn_nb = 10000000
images_max_test_nb = 10000000

pokemons_manual_associations = {
    "MrMime":  "Mr. Mime"
}

display_names_associations = True
display_max_number_reached = True
display_unknowned_pokemon_name = True
display_unreachable_file_name = True
display_detailed_results = True

global pokedex

""" TRAITEMENT PRINCIPAL """

""" 
    Sortie  : 
        - dict pokedex ;

    Description :
        Création d'un dictionnaire qui prends en entrée le nom d'un 
        pokemon et retourne son id 
"""


def getPokedex():

    print()
    print("<========== LOADING POKEDEX ==========>")
    print()

    file = pd.read_csv(
        pokedex_path,
        usecols=['pokedex_number', 'name']
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


"""
    Entrées : 
    - string a (de taille n) ; 
    - string b (de taille m) ;
    (on peut avoir n != m)

    Sortie  : 
    - float ratio ;

    Description :
        Retourne le ratio de similarité maximum (entre 0 et 1) 
        entre les deux string de taille min(n, m).
"""


def getMaxReliableStringRatio(a, b):
    lenA = len(a)
    lenB = len(b)

    # On fait en sorte que len(a) <= len(b)
    if lenA > lenB:
        (a, b) = (b, a)
        (lenA, lenB) = (lenB, lenA)

    # On considère le ratio comme 1 si il y a une inclusion de a dans b
    if (a in b):
        return 1.0

    # On compare chaque sous chaine de b de taille a, avec a et on
    # retourne le ratio le plus élevé
    return max([
        SequenceMatcher(None, a, b[i:i+lenA]).ratio()
        for i in range(lenB - lenA + 1)
    ])


""" 
    Sortie  : 
    - dict learningDataset ;

    Description :
        Retourne un dictionnaire associant pour chaque id de pokémon,
        un liste d'images issues des datasets "CompletePokemonImageDataset" 
        et "PokemonGenerationOne". Une image est en fait représentée par
        une matrice de pixels.
"""


def getLearningDataset():

    print("<====== LOADING LEARNING DATASET =====>")
    print()

    # On récupère l'ensemble des images de pokemons contenues dans
    # les datasets "CompletePokemonImageDataset" et "PokemonGenerationOne"
    globs = []
    datasets = ['CompletePokemonImageDataset', 'PokemonGenerationOne']
    for dataset in datasets:
        for ext in images_ext:
            paths = [images_dir + '/' + dataset + '/data/shapes/*/*.' + ext]
            for path in paths:
                globs += glob.glob(os.path.join(path))

    # Initialisation des variables
    pokedexKeys = pokedex.keys()
    pokemonDataset = {}
    replaced = {}
    unknowned = {}

    # Pour chaque image des datasets (max : images_max_learn_nb)
    for i in tqdm.tqdm(range(min(len(globs), images_max_learn_nb))):
        gl = globs[i]

        # On récupère le nom du pokemon (nom du sous dossier)
        pokemonName = gl.split('\\')[1]

        # On récupère l'id du pokemon à partir de son nom
        if (pokemonName in pokemons_manual_associations):
            pokemonName = pokemons_manual_associations[pokemonName]

        # Si le nom du pokemon ne correspond pas au nom de pokémon inscrit
        # dans le pokedex, on essaye de l'associer au mieux
        if not(pokemonName in pokedex):
            if not(pokemonName in unknowned.keys()):

                # On récupère le nom du pokemon du pokedex ressemblant le plus
                # à celui acuel, ainsi que son ratio de ressemblance
                similarPokemonName = max(
                    pokedexKeys, key=lambda x: getMaxReliableStringRatio(pokemonName, x))
                ratio = getMaxReliableStringRatio(
                    pokemonName, similarPokemonName)

                # Si le nom est ressemblant à plus de 70%, on considère que
                # l'association est faite
                if (0.7 <= ratio):

                    if not(pokemonName in replaced.keys()):
                        replaced[pokemonName] = "INFOS => The pokemon \"" + pokemonName + \
                            "\" has been classified as \"" + similarPokemonName + "\" in the pokedex"

                    pokemonName = similarPokemonName

                else:
                    unknowned[pokemonName] = "WARNING => Impossible to find the pokemon id of the name \"" + \
                        pokemonName + "\" in the pokedex"
                    continue
            else:
                continue

        pokemonId = pokedex[pokemonName]

        # On charge la matrice de pixels de l'image
        pokemonImg = None  # imread(gl)

        # On ajoute les données dans le dictionnaire
        if pokemonId in pokemonDataset:
            pokemonDataset[pokemonId].append(pokemonImg)
        else:
            pokemonDataset[pokemonId] = [pokemonImg]

    # Gestion de l'affichage sur la console
    if display_max_number_reached and i == images_max_learn_nb - 1:
        print()
        print("WARNING => Max images number reached ({}).".format(
            images_max_learn_nb))

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

    # On renvoie les données
    return pokemonDataset


""" 
    Entrée  : 
    - dict learningDataset ;

    Description :
        Phase d'apprentissage de l'IA.
"""


def learn(learningDataset):

    print()
    print("<============== LEARNING =============>")
    print()

    print("AI is ready!")

    print()
    print("<=====================================>")
    print()


""" 
    Sortie  : 
    - mat image ;

    Sortie  : 
    - int pokedexId ;

    Description :
        Analyse la matrice de pixels de l'image envoyée en paramètres
        et renvoie l'id du pokemon identifié dans l'image.
"""


def makePrediction(image):
    return random.randint(0, len(pokedex))


""" 
    Sortie  : 
    - dict testingDataset ;

    Description :
        Retourne un dictionnaire associant pour chaque id de pokémon,
        un liste d'images issues du dataset "OneShotPokemon". Une 
        image est en fait représentée par une matrice de pixels.
"""


def getTestingDataset():

    print()
    print("<====== LOADING TESTING DATASET ======>")
    print()

    # On récupère l'ensemble des images de pokemons contenues dans
    # le dataset "OneShotPokemon"
    globs = []
    for ext in images_ext:
        paths = [images_dir + '/OneShotPokemon/data/shapes/*.' + ext]
        for path in paths:
            globs += glob.glob(os.path.join(path))

    # Initialisation des variables
    pokemonDataset = {}
    pokemonIds = pokedex.values()
    unreachabled = []
    unknowned = {}

    # Pour chaque image du dataset (max : images_max_test_nb)
    for i in tqdm.tqdm(range(min(len(globs), images_max_test_nb))):
        gl = globs[i]

        # On récupère l'id du pokemon (dans le nom de fichier)
        fileName = gl.split('\\')[1]
        intFounds = re.findall('\d+', fileName)
        if (len(intFounds) == 0):
            unreachabled.append(
                "WARNING => No id found in the file name \"" + fileName + "\" (no integer found)")
            continue
        pokemonId = int(intFounds[0])

        # On traite uniquement les pokemons ayant un id contenu dans le pokedex
        if not(pokemonId in pokemonIds):
            if not(pokemonId in unknowned.keys()):
                unknowned[pokemonId] = "WARNING => The pokemon id \"" + \
                    str(pokemonId) + "\" is not in the pokedex"
            continue

        # On charge la matrice de pixels de l'image
        pokemonImg = None  # imread(gl)

        # On ajoute les données dans le dictionnaire
        if pokemonId in pokemonDataset:
            pokemonDataset[pokemonId].append(pokemonImg)
        else:
            pokemonDataset[pokemonId] = [pokemonImg]

    # Gestion de l'affichage sur la console
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

    # On renvoie les données
    return pokemonDataset


""" 
    Sortie  : 
    - dict testingDataset ;

    Description :
        Pour chaque image de test, on compare le résultat attendu avec la
        prédiction effectue par notre IA, et on affiche les résultat associés.
"""


def makeTests(testDataset):

    print()
    print("<======== MAKE AUTOMATIC TESTS =======>")
    print()

    # Pour chaque image de test
    results = {}
    (success, error) = (0, 0)
    for testResult, testEntries in tqdm.tqdm(testDataset.items()):
        for testEntry in testEntries:

            # On prédit l'id du pokemon par rapport à l'image
            predictedResult = makePrediction(testEntry)

            if not(testResult in results):
                results[testResult] = {
                    'success': 0,
                    'error': 0
                }

            # On compare l'id prédit avec l'id attendu
            if (testResult == predictedResult):
                success += 1
                results[testResult]['success'] += 1
            else:
                error += 1
                results[testResult]['error'] += 1

            results[testResult]['avg'] = results[testResult]['success'] / \
                (results[testResult]['success'] + results[testResult]['error'])

    # On affiche les résultats de traitement
    total = success + error
    if (total != 0):
        print()
        print(str(total) + " tests made!")
        print()

        rate = success / total
        print("Total success rate : {0:.2f}%".format(rate))

        ratesByPokemon = map(lambda x: x['avg'], results.values())
        rate = sum(ratesByPokemon) / len(results)
        print("Success rate by pokemon : {0:.2f}%".format(rate))
    else:
        print("No test launched...")

    print("Success : ", success)
    print("Error   : ", error)

    # Gestion de l'affichage sur la console
    if display_detailed_results:
        print()
        for pokemonName in sorted(results.keys()):
            print("INFO => \"{:03d}\" prediction rate = {:.2f}%".format(
                pokemonName, results[pokemonName]['avg']))

    print()
    print("<=====================================>")
    print()


if __name__ == '__main__':

    # On charge le pokédex
    pokedex = getPokedex()

    # On entraîne notre IA à partir des données d'apprentissage
    learningDataset = getLearningDataset()
    learn(learningDataset)

    # On teste notre IA à partir des données de test
    testDataset = getTestingDataset()
    makeTests(testDataset)
