#!/usr/bin/python
# -*- coding: utf-8 -*-

from consts import *
from ai import *
from load import *

if __name__ == '__main__':

    # Inititalisation
    constantManager = ConstantsManager()
    dataManager = DataManager(constantManager)
    aiManager = AiManager(constantManager, dataManager)

    # On lance l'apprentissage
    aiManager.learn(dataManager.getLearningDataset())

    # On lance les tests
    aiManager.makeTests(dataManager, dataManager.getTestingDataset())
