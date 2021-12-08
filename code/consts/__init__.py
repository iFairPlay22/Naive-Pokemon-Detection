class ConstantsManager(object):

    """
        Le chemin du csv du pokedex
    """
    def getPokedexPath(self):
        return './assets/csv/pokedex.csv'

    """
        Le répertoire dans lequel trouver les datasets
    """

    def getImagesDirectory(self):
        return './assets/images/'

    """
        Le type des différentes images qu'on veut charger
    """
    def getImageExtensions(self):
        return ['jpg', 'jpeg', 'png']

    """
        Le nombre d'images max à charger en mémoire pour l'apprentissage
    """
    def getImagesMaxLearnNb(self):
        return 1000

    """
        Le nombre d'images max à charger en mémoire pour les tests
    """
    def getImagesMaxTestNb(self):
        return 1000

    """
        Le nom des pokemons qu'on veut charger (sous forme de regexp)
        NB : Ici, tous les pokémons dont le nom commence par un a ou A
    """
    def getPokemonRegexpName(self):
        return r"[aA].*"

    """
        Les association manuelles de noms de pokemon faites dans le traitement
    """
    def getPokemonManualAssociations(self):
        return {
            "MrMime":  "Mr. Mime"
        }

    """
        Afficher les noms des associations de pokemons (noms similaires entre 
        les datasets et le pokedex).
    """
    def getDisplayNamesAssociations(self):
        return True

    """
        Afficher si on a restreint le jeu de données d'apprentissage / de test
        avec les limites de nombre d'images définies plus haut.
    """
    def getDisplayMaxNumberReached(self):
        return True

    """
        Afficher les noms de pokemons qui n'ont pas pu être identifiés lors de la
        lecture des fichiers.
    """
    def getDisplayUnknownedPokemonName(self):
        return True

    """
        Afficher les noms de fichiers inexploitables
    """
    def getDisplayUnreachableFileName(self):
        return True

    """
        Afficher les noms des pokémons ignorés dans la phase de test car on n'a pas 
        appris à les reconnaître dans la phase d'apprentissage.
    """
    def getDisplayUnrecognizablePokemon(self):
        return True

    """
        Afficher les noms des images qui ont un format de données invalides.
    """
    def getDisplayInvalidImage(self):
        return True

    """
        Afficher les résultats du traitement en détail (statistiques par pokémon).
    """
    def getDisplayDetailedResults(self):
        return True

    """
        Afficher la palette d'images utilisée pour la reconnaissance d'images.
    """
    def getDisplayImagePalette(self):
        return False
