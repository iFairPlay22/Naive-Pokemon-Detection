import tqdm
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
import colorsys
from sklearn.metrics.pairwise import euclidean_distances
from statistics import mode as most_common


class AiManager(object):

    """
        Cette classe permet, suite à un apprentissage, de reconnaitre quel pokemon
        est dans une image, à partir d'une analyse de son histogramme de couleurs.
    """

    def __init__(self, consts):
        self._consts = consts

    """ 
        Entrée  : 
        - dict learningDataset ;

        Description :
            Phase d'apprentissage de l'IA. On calcule l'ensemble des diagrammes de couleurs
            des images de chaque pokemon du dataset d'apprentissage, et on les stocke dans 
            un tableau.
    """

    def learn(self, learningDataset):

        print()
        print("<========== LOADING POKEDEX ==========>")
        print()

        self._ids = []
        self._imgs = []
        for pokemonId, pokemonImages in learningDataset.items():
            for pokemonImage in pokemonImages:
                self._ids.append(pokemonId)
                self._imgs.append(pokemonImage)

        self._kms = self._getKMeans(self._getRandomRgbPixels())

        self._histograms = np.zeros(
            (len(self._imgs), self._kms.n_clusters), dtype=np.float32)
        for i in range(len(self._imgs)):
            self._histograms[i, :] = self._getImgHistogram(
                self._kms, self._imgs[i])

        print()
        print("<=====================================>")
        print()

    """
        Entrée :
            - numpy array img ;

        Description:
            Affiche une image à l'écran
    """

    def _displayImg(self, img):
        px.imshow(img).show()

    """
        Entrée :
            - numpy array images ;
            - int pixelNumber ;

        Sortie :
            - numpy array randomRbgPixels ;

        Description:
            Retourne ${pixelNumber} pixels RGB de manière aléatoire depuis
            le dataset fourni en paramètre
    """

    def _getRandomRgbPixels(self, pixelNumber=100000):
        allRgbPixels = [img.reshape((-1, 3)) for img in self._imgs]
        allRgbPixels = np.vstack(allRgbPixels)
        randomPixelsIndexes = np.random.choice(
            allRgbPixels.shape[0], pixelNumber, replace=False)
        randomRbgPixels = allRgbPixels[randomPixelsIndexes, :]
        return randomRbgPixels

    """
        Entrée :
            - numpy array randomRbgPixels ;

        Sortie :
            - KMeans kms ;

        Description:
            Applique l'algorithme de KMeans sur les pixels. On regarde les occurences
            des pixels dans l'image, afin de pouvoir batir un modèle prédictif.
    """

    def _getKMeans(self, pixels):
        kms = KMeans(128, n_init=1, verbose=0)
        kms.fit(pixels)

        h = np.zeros((kms.n_clusters, ))
        for i in range(kms.n_clusters):
            r, g, b = kms.cluster_centers_[i, :]/255.0
            h[i] = colorsys.rgb_to_hsv(r, g, b)[0]
        idx = np.argsort(h)
        kms.cluster_centers_ = kms.cluster_centers_[idx, :]

        if (self._consts.getDisplayImagePalette()):
            palette = kms.cluster_centers_.astype(np.uint8)
            self._displayImg(palette.reshape((8, -1, 3)))

        return kms

    """
        Entrée :
            - KMeans kms ;
            - numpy array img ;

        Sortie :
            - numpy array histogram ;

        Description:
            On revoie l'histogramme des couleurs de l'image.
    """

    def _getImgHistogram(self, kms, img):
        x = np.zeros((kms.n_clusters, ))
        idx = kms.predict(img.astype(np.float64).reshape((-1, 3)))
        idv, v = np.unique(idx, return_counts=True)
        x[idv] = v
        return x/np.sum(x).astype(np.float32)

    """ 
        Entrée  : 
        - numpy array img ;
        - int imgNumber ;

        Sortie  : 
        -  int pokemonId ;

        Description :
            On compare les diagrammes de couleurs de l'image et de celles enregistrées 
            dans le dataset. On retourne le pokemonId le plus proche.
    """

    def predict(self, img, imgNumber=5):
        imgHistogram = self._getImgHistogram(self._kms, img)
        dist = euclidean_distances([imgHistogram], self._histograms)
        sortDist = np.argsort(dist[0, :])

        foundPokemonIds = [self._ids[sortDist[i]] for i in range(imgNumber)]

        return most_common(foundPokemonIds)

    """
        Sortie  :
        - dict testingDataset ;

        Description :
            Pour chaque image de test, on compare le résultat attendu avec la
            prédiction effectue par notre IA, et on affiche les résultat associés.
    """

    def makeTests(self, dataManager, testDataset):

        print()
        print("<======== MAKE AUTOMATIC TESTS =======>")
        print()

        # Pour chaque image de test
        results = {}
        (success, error) = (0, 0)
        for testResult, testEntries in tqdm.tqdm(testDataset.items()):
            for testEntry in testEntries:

                # On prédit l'id du pokemon par rapport à l'image
                predictedResult = self.predict(testEntry)
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
                    (results[testResult]['success'] +
                     results[testResult]['error'])

        # On affiche les résultats de traitement
        total = success + error
        if (total != 0):
            print()
            print(str(total) + " tests made!")
            print()

            rate = success / total * 100
            print("Total success rate : {0:.2f}%".format(rate))

            ratesByPokemon = map(lambda x: x['avg'], results.values())
            rate = sum(ratesByPokemon) / len(results) * 100
            print("Success rate by pokemon : {0:.2f}%".format(rate))
        else:
            print("No test launched...")

        print("Success : ", success)
        print("Error   : ", error)

        # Gestion de l'affichage sur la console
        if self._consts.getDisplayDetailedResults():
            print()
            reversedPokedex = dataManager.getReversedPokedex()
            for pokemonId in sorted(results.keys()):
                print("INFO => {} (\"{:03d}\"), prediction rate over {} test(s) = {:.2f}%".format(
                    reversedPokedex[pokemonId], pokemonId, results[pokemonId]['success'] + results[pokemonId]['error'], results[pokemonId]['avg'] * 100))

        print()
        print("<=====================================>")
        print()
