# Deep Learning Project for Geometric and Spectral Image Classification

This project explores the use of deep learning for automated analysis of brain tumor
MRI images by combining geometric and frequency domains. Addressing the challenges
of computer-aided medical diagnosis, we propose an innovative approach that simulta-
neously exploits traditional spatial features and spectral information obtained through
Fourier transformation.
Our developed system is based on a hybrid convolutional neural network architecture
composed of two parallel branches. The first branch, dedicated to geometric analysis,
processes MRI images in their native spatial representation to extract morphological
features such as shapes, contours, and textures. The second branch, focused on frequency
analysis, applies Fast Fourier Transform (FFT) to capture periodic patterns and textural
signatures not apparent in the spatial domain.
The dataset used, consisting of 7,022 MRI images, comprises four distinct clinical
classes : gliomas, meningiomas, pituitary tumors, and no-tumor cases. Each image under-
goes specific preprocessing according to the concerned branch, including normalization,
resizing, and augmentation for the geometric branch, as well as FFT transformation,
spectral centering, and magnitude extraction for the frequency branch.
This study demonstrates the potential of the multi-domain approach for improving
automated brain tumor diagnosis, paving the way for promising clinical applications in
the field of AI-assisted radiology.


## Structure du dépôt

- `geometric_model.py` : modèle CNN pour la branche géométrique.
- `Spectral_model.py` : modèle CNN pour la branche spectrale.
- `fusion.py` : fusion des deux branches dans un modèle unique.
- `SpectralDataGenerator.py` : générateur de données pour charger et prétraiter les images spectrales.
- `Partitionnement.py` : utilitaires pour le partitionnement, la normalisation et la préparation des données.
- `README.md` : documentation du projet.

## Réseaux CNN pour la détection de tumeurs

Plusieurs architectures CNN ont été adoptées avec succès pour les tâches de classifi-
cation et segmentation des tumeurs :
• U-Net : Architecture conçue spécifiquement pour la segmentation biomédicale. Elle
repose sur une structure en “U” avec des chemins d’encodage (compression) et de
décodage (décompression), facilitant la localisation précise des régions tumorales.
• ResNet : Introduit des connexions résiduelles qui permettent de construire des
réseaux très profonds sans souffrir du problème de dégradation des performances.
ResNet est souvent utilisé pour la classification de tumeurs.
• VGG : Réseau profond basé sur des couches convolutives empilées avec des filtres
de petite taille (3x3). Bien que plus coûteux en calculs, il fournit de bonnes perfor-
mances de base pour la reconnaissance d’images médicales.

###Analyse fréquentielle appliquée aux images médicales

Outre les approches basées sur les caractéristiques spatiales, l’analyse fréquentielle
représente une autre stratégie pertinente. Elle consiste à transformer l’image dans le
domaine des fréquences, souvent via la Fast Fourier Transform (FFT), afin de capturer
des motifs globaux, des textures ou des informations structurelles qui peuvent être moins
visibles dans le domaine spatial.

###Technologies utilisées

• Python 3.10 : langage principal de développement.
• TensorFlow & Keras : pour la conception, l’entraînement et l’évaluation des
modèles de deep learning.
• NumPy & OpenCV : pour la manipulation des images et la transformation de
Fourier.
• Matplotlib & Seaborn : pour la visualisation des résultats.


### Installation

Sur Windows, PowerShell ou CMD :

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install tensorflow opencv-python numpy scikit-learn
```

Sur Linux/macOS :

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install tensorflow opencv-python numpy scikit-learn
```

## Lancer le projet

### 1) Cloner le dépôt

```bash
git clone <url-du-repo>
cd <nom-du-repo>
```

### 2) Créer l'environnement virtuel

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

### 3) Installer les dépendances

```bash
pip install tensorflow opencv-python numpy scikit-learn
```

### 4) Vérifier que le code compile

```bash
python -m py_compile geometric_model.py Spectral_model.py fusion.py SpectralDataGenerator.py Partitionnement.py
```

### 5) Tester les modèles

```bash
python geometric_model.py
python Spectral_model.py
python fusion.py
```

Ces commandes doivent afficher les summaries des modèles sans erreur si tout est correctement configuré.

## Exemple d'utilisation en Python

```python
from geometric_model import build_geometric_model
from Spectral_model import build_spectral_model
from fusion import build_fusion_model

geometric_model = build_geometric_model(input_shape=(128, 128, 1), num_classes=10)
spectral_model = build_spectral_model(input_shape=(128, 128, 1), num_classes=10)
fusion_model = build_fusion_model(input_shape=(128, 128, 1), spectral_shape=(128, 128, 1), num_classes=10)

fusion_model.summary()
```

## Tester la préparation des données

```bash
python Partitionnement.py
```

Ce script vérifie la préparation des données en les normalisant et en appliquant les transformations attendues.

## Remarques importantes

- Le projet a été reconstruit et nettoyé pour corriger des erreurs de syntaxe et d'import.
- Les chemins vers les datasets peuvent varier selon votre environnement local ou votre dépôt GitHub.
- Si vous publiez sur GitHub, il est conseillé de déposer le dataset dans un autre dépôt ou de fournir un script de téléchargement et une description claire des données.
- Avant un entraînement complet, il faut préparer les images dans le bon format : tableau NumPy, taille uniforme, labels encodés.

## Prochaine étape recommandée

1. Ajouter le dataset Kaggle ou un sous-ensemble public.
2. Préparer un script d'entraînement complet.
3. Ajouter les fichiers de configuration pour les dépendances.
4. Documenter les hyperparamètres, métriques et résultats.
5. Publier le repo avec un vrai notebook ou un script d'entraînement.

## Licence

Ce projet est à adapter selon votre usage personnel ou académique.
