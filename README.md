# WhosThatPokemon 

## Réalisateurs

**E4FI** - **Groupe 1** - **Machine learning**

| Membre                | Github                                             |
|-----------------------|----------------------------------------------------|
| Ewen BOUQUET          | [@iFairPlay22](https://github.com/iFairPlay22)     |
| Matthias JOUEN        | [@MatthiasJouen](https://github.com/MatthiasJouen)           |
| Alexandre JOUDIOUX    | [@Vitrox77](https://github.com/Vitrox77)           |
| Guillaume DAVY        | [@GuillaumeDavy](https://github.com/GuillaumeDavy) |

## Contexte

### Notre projet

Le but de notre projet est de simuler le comportement d'un **pokédex**, mais à partir d'images. Ce dernier contient un algorithme de **machine learning**, fait en python, permettant de **reconnaitre un pokémon** à partir **d'une image**. 

L'algorithme contient 2 parties principales :
- **L'apprentissage** : 

	> On calcule les **histogrammes de couleur** de chaque pokémon contenu dans des datasets d'apprentissage (`CompletePokemonImageDataset` et `PokemonGenerationOne`) ;
- **La prédiction** :

	> On lance une série de tests par **comparaison vectorielle** de diagrammes de couleurs entre les données d'apprentissage et les images du dataset de test (`OneShotPokemon`) ;

### Datasets utilisés

- Informations sur les pokémons (ids, noms, descriptions, caractéristiques) :

	> [Complete Pokemon Dataset](https://www.kaggle.com/mariotormo/complete-pokemon-dataset-updated-090420), Information about 1045 Pokemon (including varieties) until 8th Generation ;
 
- Images de pokémons :

	> [Complete Pokemon Image Dataset](https://www.kaggle.com/hlrhegemony/pokemon-image-dataset), 2,500+ clean labeled images, all official art, for Generations 1 through 8 ;
    
	> [One-Shot-Pokemon Images](https://www.kaggle.com/aaronyin/oneshotpokemon), Colorful and fun dataset for one shot learning problem, gotta recognize them all ;
    
	> [Pokemon Generation One](https://www.kaggle.com/thedagger/pokemon-generation-one), Gotta train them all! ;

## Organisation du projet 

### Code du projet (`/code`)

Vous trouverez dans ce programme les différents codes python utilisés pour lancer l'algorithme de machine learning.

- `main.py` :

	> Le programme principal à exécuter pour obtenir les résultats du traitement. Ce dernier utilise les classes définies dans les répertoires suivants.

- `consts/__init__.py` :

	> Définis une classe `ConstantsManager` permettant de gérer les paramètres du traitement comme le chemins d'accès aux fichiers, de gérer ce qui est affiché dans la console, etc.

- `load/__init__.py` :

	> Définis une classe `DataManager` permettant de charger en mémoire les données du csv (pokédex) et des différents datasets (images de pokemons) de manière uniforme.

- `ai/__init__.py` :

	> Définis une classe `AiManager` permettant de calculer les histogrammes de couleurs de chaque image de pokemon (apprentissage), et de trouver le nom du pokémon associé à une image de test (prédiction).

### Ressources du projet (`/assets`)

Afin de simplifier le traitement, nous avons organisé les ressources (csv et images) dans le répertoire de manière bien précise :

- `/assets/csv/pokedex.csv` 

	> Contient le CSV le plus récent fourni via Kaggle : [Complete Pokemon Dataset](https://www.kaggle.com/mariotormo/complete-pokemon-dataset-updated-090420/download) ;

- `/assets/images/CompletePokemonImageDatatset/data/shapes/` 

	> Contient toutes les images fournies via Kaggle : [Complete Pokemon Image Datatset](https://www.kaggle.com/hlrhegemony/pokemon-image-dataset/download) ;

- `/assets/images/OneShotPokemon/data/shapes/` 

	> Contient toutes les images (de pokemon standard) fournies via Kaggle : [One Shot Pokemon](https://www.kaggle.com/aaronyin/oneshotpokemon/download) ;

- `/assets/images/PokemonGenerationOne/data/shapes/` 

	> Contient toutes les images fournies via Kaggle : [Pokemon Generation One](https://www.kaggle.com/thedagger/pokemon-generation-one/download) ;

- `/assets/images/OneShotPokemon/data/shapes/` 


### Description du projet (`/notebook`)

Vous trouverez dans ce répertoire le notebook (`.ipynb`) traitant des sujets suivants :
- Etat de l'art ;

	> Convolutional Neural Networks ;

	> Regression Logistique ;

	> K plus proches voisins ;

- Description détaillée de la méthode utilisée ;

## Lancer le projet

### Inclure les images

Téléchargez les images des datasets et collez les dans les 3 répertoires suivants :
- `./assets/images/CompletePokemonImageDataset/data/shapes` ;
- `./assets/images/OneShotPokemon/data/shapes` ;
- `./assets/images/PokemonGenerationOne/data/shapes` ;

### Dépendances

Installer les librairies : 
- tqdm ;
- numpy ;
- pandas ;
- plotly ;
- imageio ;
- sklearn ; 
- difflib ;

### Execution

Ouvrir un terminal et éxecuter les commandes suivantes : 

```bash
cd ./WhosThatPokemon
python3 ./code/main.py
```

Le 02/01/2022.
