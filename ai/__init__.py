import numpy as np
import plotly.express as px
import glob
import os
from imageio import imread
from sklearn.cluster import KMeans
import colorsys
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import statistics
from statistics import mode as most_common

display_image_palette = False


class ImageAnalysor(object):
    def __init__(self, learningDataset):
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
        allRgbPixels = np.vstack([img.reshape((-1, 3)) for img in self._imgs])
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

        if (display_image_palette):
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

    def getSimilarImageId(self, img, imgNumber=5):
        imgHistogram = self._getImgHistogram(self._kms, img)
        dist = euclidean_distances([imgHistogram], self._histograms)
        sortDist = np.argsort(dist[0, :])

        foundPokemonIds = [self._ids[sortDist[i]] for i in range(imgNumber)]
        print(foundPokemonIds)
        print(most_common(foundPokemonIds))
        return most_common(foundPokemonIds)
