Énoncé
===

Vous devez rendre une archive au format zip, nommée avec vos noms et prénoms (pour les binômes, les deux noms et deux prénoms) contenant:
- les sources pour les fonctions d'apprentissage et de test de la méthode Bagging avec des arbres de décisions
- les sources pour les fonctions d'apprentissage et de test de la méthode Adaboost.M1 avec des arbres de décisions
- les sources pour les fonctions d'apprentissage et de test de la méthode Forest-RI
- une ou plusieurs fonctions permettant de lancer les tests de toutes ces fonctions (à la manière des fichiers test_tree.m et test_forest.m qui étaient fournis pour le TP3)
Commentez abondamment votre code pour expliquer ce qu'il réalise étape après étape.

## Rendu

Chaque méthode se situe dans un dossier. Les fichiers d'apprentissage, de test et de prédiction sont disponibles ainsi qu'un fichier main afin de lancer le programme.

Pour chaque méthode il est possible de modifier dans le ou les fichiers principaux le jeu de données utilisé (changer la variable `fileName`).

### Bagging (1.ensembleClassifier(bagging))

Lancer `main.m` pour lancer Bagging.

## Boosting (1.ensembleClassifier(boosting))

Lancer `adaboostMain.m` pour AdaBoost et `adaboostM1Main.m` pour AdaBoost.M1.

Il est possible de modifier le classifieur faible (changer la variable `learningType` en `tree` ou `stump`).

## Random Forest (3.ensembleClassifier(random_forest))

Lancer `test_rf.m` pour les Random Forest.