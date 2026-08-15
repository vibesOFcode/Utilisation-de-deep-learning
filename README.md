# Deep Learning Project for Geometric and Spectral Image Classification

Ce dépôt contient une implémentation de modèles de deep learning pour la classification d'images en combinant des informations géométriques et spectrales.

## Objectif du projet

Le projet vise à explorer une approche de fusion de deux branches de réseaux de neurones :

- une branche géométrique basée sur l'image originale,
- une branche spectrale basée sur la transformée de Fourier de l'image.

L'idée est d'utiliser les deux types de représentations pour améliorer la capacité du modèle à distinguer les classes.

## Structure du dépôt

- `geometric_model.py` : modèle CNN pour la branche géométrique.
- `Spectral_model.py` : modèle CNN pour la branche spectrale.
- `fusion.py` : fusion des deux branches dans un modèle unique.
- `SpectralDataGenerator.py` : générateur de données pour charger et prétraiter les images spectrales.
- `Partitionnement.py` : utilitaires pour le partitionnement, la normalisation et la préparation des données.
- `README.md` : documentation du projet.

## Dépendances

Ce projet utilise généralement :

- Python 3.9+
- TensorFlow / Keras
- NumPy
- OpenCV
- scikit-learn (si utilisé pour le train/test split)

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
