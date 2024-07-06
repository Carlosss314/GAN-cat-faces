# GAN-cat-faces

réseau de neurone adversaire génératif (GAN) qui génère des images de chat

Je n'ai mis ici qu'un petit échantillon du jeu de données dans le dossier "cats", le jeu de donné entier contient un peu plus de 15000 images.

Le dossier "resultats" contient des exemples d'images générées (à droite) avec à gauche une vraie image du jeu de données pour la comparaison.



"load_images.py" convertit le dossier d'images "cats" en un jeu de données "dataset.pt"

"model.py" contient le réseau de neurones

"train.py" permet d'entraîner le model

"test.py" permet de tester le model sur des exemmples du jeu de données
