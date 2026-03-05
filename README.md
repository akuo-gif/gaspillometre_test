# Gaspillomètre

Détection du gaspillage alimentaire sur photos de plateaux repas avec YOLOv8.

## Classes détectées

food, meat, pasta, apple, banana, bread

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Entraîner

```bash
python src/train.py
```

## Tester sur une image

```bash
python src/inference.py --image imageplateau/P1400073.JPG
```

## Tester sur un dossier

```bash
python src/inference.py --dir imageplateau/
```

## Structure

```
config/          # Configuration (classes, paramètres)
data/            # Dataset YOLO (images + labels train/val)
imageplateau/    # Photos originales des plateaux
models/best.pt   # Modèle entraîné
src/train.py     # Entraînement
src/inference.py # Détection
src/prepare_data.py # Préparation du dataset
```
