import glob
import os
from imageio import imread
from difflib import SequenceMatcher
import pandas as pd
import re
import tqdm


class DataManager(object):

    """
        Cette classe permet de charger en mémoire les datasets 
        d'images d'apprentissage et de test.
    """

    def __init__(self, consts):
        self._consts = consts
        self._pokedex = self._loadPokedex()
        self._reversedPokedex = self._loadReversedPokedex()

    """ 
        Sortie  : 
            - dict pokedex ;

        Description :
            Création d'un dictionnaire qui prends en entrée le nom d'un 
            pokemon et retourne son id 
    """

    def _loadPokedex(self):

        print()
        print("<========== LOADING POKEDEX ==========>")
        print()

        file = pd.read_csv(
            self._consts.getPokedexPath(),
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
        Sortie  : 
            - dict pokedex ;

        Description :
            Retourne le pokédex précedamment chargé en mémoire
    """

    def getPokedex(self):
        return self._pokedex

    """ 
        Sortie  : 
            - dict reversed pokedex ;

        Description :
            Création d'un dictionnaire qui prends en entrée l'id d'un 
            pokemon et retourne son nom
    """

    def _loadReversedPokedex(self):
        revPokedex = dict()

        for pokemonName, pokemonId in self._pokedex.items():
            revPokedex[pokemonId] = pokemonName

        return revPokedex

    """ 
        Sortie  : 
            - dict reversedPokedex ;

        Description :
            Retourne le pokédex inversé précedamment chargé en mémoire
    """

    def getReversedPokedex(self):
        return self._reversedPokedex

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

    def _getMaxReliableStringRatio(self, a, b):
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

    def getLearningDataset(self):

        print("<====== LOADING LEARNING DATASET =====>")
        print()

        # On récupère l'ensemble des images de pokemons contenues dans
        # les datasets "CompletePokemonImageDataset" et "PokemonGenerationOne"
        globs = []
        datasets = ['CompletePokemonImageDataset', 'PokemonGenerationOne']
        for dataset in datasets:
            for ext in self._consts.getImageExtensions():
                paths = [self._consts.getImagesDirectory()
                         + '/' + dataset +
                         '/data/shapes/*/*.' + ext]
                for path in paths:
                    globs += glob.glob(os.path.join(path))

        # Initialisation des variables
        pokedexKeys = self._pokedex.keys()
        pokemonDataset = {}
        replaced = {}
        unknowned = {}
        invalids = []
        manualAssociations = self._consts.getPokemonManualAssociations()
        regexpPokemonName = self._consts.getPokemonRegexpName()
        maxLearnNb = self._consts.getImagesMaxLearnNb()
        n = 0

        # Pour chaque image des datasets
        for i in tqdm.tqdm(range(len(globs))):
            gl = globs[i]

            # On récupère le nom du pokemon (nom du sous dossier)
            pokemonName = gl.split('\\')[1]

            # On récupère l'id du pokemon à partir de son nom
            if (pokemonName in manualAssociations):
                pokemonName = manualAssociations[pokemonName]

            # On ignore les pokemons ne correspondant pas à la regexp
            if (regexpPokemonName and not re.match(regexpPokemonName, pokemonName)):
                continue

            # Si le nom du pokemon ne correspond pas au nom de pokémon inscrit
            # dans le pokedex, on essaye de l'associer au mieux
            if not(pokemonName in self._pokedex):
                if not(pokemonName in unknowned.keys()):

                    # On récupère le nom du pokemon du pokedex ressemblant le plus
                    # à celui acuel, ainsi que son ratio de ressemblance
                    similarPokemonName = max(
                        pokedexKeys, key=lambda x: self._getMaxReliableStringRatio(pokemonName, x))
                    ratio = self._getMaxReliableStringRatio(
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

            pokemonId = self._pokedex[pokemonName]

            # On charge la matrice de pixels de l'image
            pokemonImg = imread(gl)

            # On garde le images 3D avec un nombre de pixels multiple 3
            if not(len(pokemonImg.shape) == 3 and (pokemonImg.shape[0] * pokemonImg.shape[1] * pokemonImg.shape[2]) % 3 == 0):
                if self._consts.getDisplayInvalidImage():
                    invalids.append("WARNING => The image \"" + gl +
                                    "\" has been ignored because it has a wrong shape!")
                    continue

            # On ajoute les données dans le dictionnaire
            if pokemonId in pokemonDataset:
                pokemonDataset[pokemonId].append(pokemonImg)
            else:
                pokemonDataset[pokemonId] = [pokemonImg]

            n += 1
            if (n == maxLearnNb):
                print()
                print("WARNING => Max images number reached ({}).".format(n))
                break

        # Gestion de l'affichage sur la console
        if self._consts.getDisplayNamesAssociations() and len(replaced) != 0:
            print()
            for msg in replaced.values():
                print(msg)

        if self._consts.getDisplayUnknownedPokemonName() and len(unknowned) != 0:
            print()
            for msg in unknowned.values():
                print(msg)

        if self._consts.getDisplayInvalidImage():
            print()
            for msg in invalids:
                print(msg)

        self._recognizablePokemonIds = pokemonDataset.keys()

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

        Sortie  : 
        -  ImageAnalysor pokemonAi ;

        Description :
            Phase d'apprentissage de l'IA.
    """

    def getTestingDataset(self):

        print()
        print("<====== LOADING TESTING DATASET ======>")
        print()

        # On récupère l'ensemble des images de pokemons contenues dans
        # le dataset "OneShotPokemon"
        globs = []
        for ext in self._consts.getImageExtensions():
            paths = [self._consts.getImagesDirectory(
            ) + '/OneShotPokemon/data/shapes/*.' + ext]
            for path in paths:
                globs += glob.glob(os.path.join(path))

        # Initialisation des variables
        pokemonDataset = {}
        pokemonIds = self._pokedex.values()
        unreachabled = []
        unknowned = {}
        unRecognizable = {}
        invalids = {}
        regexpPokemonName = self._consts.getPokemonRegexpName()
        maxTestNb = self._consts.getImagesMaxTestNb()
        n = 0

        # Pour chaque image du dataset (max : images_max_test_nb)
        for i in tqdm.tqdm(range(len(globs))):
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

            # On ignore les pokemons ne correspondant pas à la regexp
            if (regexpPokemonName and not re.match(regexpPokemonName, self._reversedPokedex[pokemonId])):
                continue

            # On traite uniquement les pokemons qui peuvent être reconnus par l'IA
            if not(pokemonId in self._recognizablePokemonIds):
                if not(pokemonId in unRecognizable.keys()):
                    unRecognizable[pokemonId] = "WARNING => The pokemon id \"" + \
                        str(pokemonId) + \
                        "\" was skipped because it is not recognizable by the AI"
                continue

            # On charge la matrice de pixels de l'image
            pokemonImg = imread(gl)

            # On garde le images 3D avec un nombre de pixels multiple 3
            if not(len(pokemonImg.shape) == 3 and (pokemonImg.shape[0] * pokemonImg.shape[1] * pokemonImg.shape[2]) % 3 == 0):
                if self._consts.getDisplayInvalidImage():
                    invalids.append("WARNING => The image \"" + gl +
                                    "\" has been ignored because it has a wrong shape!")
                    continue

            # On ajoute les données dans le dictionnaire
            if pokemonId in pokemonDataset:
                pokemonDataset[pokemonId].append(pokemonImg)
            else:
                pokemonDataset[pokemonId] = [pokemonImg]

            n += 1

            if (n == maxTestNb):
                print()
                print("WARNING => Max images number reached ({}).".format(maxTestNb))
                break

        # Gestion de l'affichage sur la console
        if self._consts.getDisplayUnreachableFileName() and len(unreachabled) != 0:
            print()
            for msg in unreachabled:
                print(msg)

        if self._consts.getDisplayUnknownedPokemonName() and len(unknowned) != 0:
            print()
            for msg in unknowned.values():
                print(msg)

        if self._consts.getDisplayUnrecognizablePokemon():
            print()
            for msg in unRecognizable.values():
                print(msg)

        if self._consts.getDisplayInvalidImage():
            print()
            for msg in invalids:
                print(msg)

        print()
        print(str(i) + " images loaded!")

        print()
        print("<=====================================>")
        print()

        # On renvoie les données
        return pokemonDataset
