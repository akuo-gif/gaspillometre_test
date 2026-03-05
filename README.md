# 🍽️ Gaspillomètre — Prototype IA

**Système intelligent de mesure du gaspillage alimentaire** par détection d'objets sur photos de plateaux repas.

## 🎯 Ce que fait ce prototype

| Fonctionnalité | Méthode | Statut |
|---|---|---|
| Identifier les aliments jetés | YOLOv8 (détection d'objets) | ✅ Prêt |
| Estimer le poids des restes | Surface détectée × densité alimentaire | ✅ Prêt |
| Calculer des statistiques | Agrégation par aliment/jour/saison/menu | ✅ Prêt |
| Affichage dynamique | Dashboard Streamlit | ✅ Prêt |
| Comparer gaspillage / production | Ratio déchets / production | ✅ Prêt |

## 📁 Structure du projet

```
gaspillometre/
├── config/
│   ├── classes.yaml          # Classes d'aliments (YOLO format)
│   └── config.yaml           # Configuration modèle, poids, stats
├── src/
│   ├── prepare_data.py       # Préparer le dataset train/val
│   ├── launch_annotation.py  # Lancer Label Studio
│   ├── train.py              # Entraîner YOLOv8
│   ├── inference.py          # Détecter + estimer le poids
│   ├── stats.py              # Calculer les statistiques
│   └── dashboard.py          # Dashboard Streamlit
├── imagesplateau/            # Vos 78 images brutes
├── data/
│   ├── images/train/         # Images d'entraînement
│   ├── images/val/           # Images de validation
│   ├── labels/train/         # Annotations train
│   ├── labels/val/           # Annotations val
│   ├── menus.csv             # (optionnel) Menus du jour
│   └── production.csv        # (optionnel) Quantités produites
├── models/                   # Modèles entraînés
├── results/                  # Résultats d'entraînement
├── logs/                     # Logs de détection
└── requirements.txt
```

## 🚀 Démarrage rapide

### 1. Installation

```bash
cd gaspillometre
pip install -r requirements.txt
```

### 2. Annoter les images (ÉTAPE CRITIQUE ⚡)

C'est **l'étape la plus importante**. La qualité du modèle dépend directement de la qualité des annotations.

#### Option A : Label Studio (recommandé)
```bash
python src/launch_annotation.py
```
- S'ouvre dans le navigateur sur `http://localhost:8080`
- Importez les images de `imagesplateau/`
- Dessinez des rectangles autour de chaque aliment
- Exportez en **format YOLO**

#### Option B : Roboflow (gratuit, semi-automatique)
1. Allez sur [roboflow.com](https://roboflow.com)
2. Créez un projet "Object Detection"
3. Uploadez vos images
4. Annotez (Roboflow propose de l'auto-annotation)
5. Exportez en **YOLOv8 format**
6. Placez les fichiers dans `data/`

#### Option C : CVAT (open source, en ligne)
1. Allez sur [app.cvat.ai](https://app.cvat.ai)
2. Même processus, exportez en YOLO

#### Format d'annotation YOLO
Un fichier `.txt` par image, même nom que l'image :
```
# <class_id> <x_center> <y_center> <width> <height>
# Coordonnées normalisées entre 0 et 1
1 0.45 0.32 0.15 0.12
4 0.72 0.61 0.20 0.18
7 0.30 0.80 0.10 0.08
```

### 3. Préparer le dataset

```bash
python src/prepare_data.py
# Organise les images annotées en train/val (80%/20%)
```

### 4. Entraîner le modèle

```bash
python src/train.py
# Options :
python src/train.py --epochs 200 --batch 16 --model yolov8s
python src/train.py --resume  # reprendre un entraînement
```

### 5. Tester le modèle

```bash
# Sur une image
python src/inference.py --image imagesplateau/P1400033.JPG

# Sur tout le dossier
python src/inference.py --dir imagesplateau/

# En temps réel (webcam)
python src/inference.py --camera 0
```

### 6. Lancer le dashboard

```bash
streamlit run src/dashboard.py
```

## 📋 Guide d'annotation — Bonnes pratiques

### Classes d'aliments (15 classes)

| ID | Classe | Exemples |
|----|--------|----------|
| 0 | pain | Bout de pain, croûton, tartine |
| 1 | viande | Poulet, boeuf, porc, agneau |
| 2 | poisson | Filet, pané, crevettes |
| 3 | legumes | Haricots, carottes, courgettes |
| 4 | feculents | Pommes de terre, purée |
| 5 | fromage | Portion, râpé |
| 6 | dessert | Gâteau, crème, mousse |
| 7 | fruits | Pomme, banane, orange |
| 8 | salade | Verte, composée |
| 9 | soupe | Bol, assiette |
| 10 | riz | Blanc, pilaf |
| 11 | pates | Spaghetti, penne, coquillettes |
| 12 | sauce | Tomate, béchamel, jus |
| 13 | yaourt | Pot, fromage blanc |
| 14 | compote | Pot, coupelle |

### Règles d'annotation

1. **Dessinez des rectangles serrés** autour de chaque aliment
2. **Annotez TOUT** ce qui est visible, même les petits restes
3. **Un rectangle par aliment**, pas par groupe
4. **En cas de doute** sur la classe → choisir la plus proche
5. **Plateau vide** → pas de fichier d'annotation (ou fichier vide)
6. **Aliments mélangés** → annoter la classe dominante

### Combien d'annotations faut-il ?

| Nb images annotées | Qualité attendue |
|---|---|
| 80 (actuel) | 🟡 Prototype fonctionnel |
| 200-500 | 🟢 Bon pour démo |
| 500-1000 | 🟢🟢 Fiable |
| 1000+ | 🟢🟢🟢 Production |

> **Avec 80 images**, le modèle sera capable de détecter les grandes catégories. 
> Pour améliorer la précision, **ajoutez simplement plus d'images annotées** et relancez l'entraînement.

## 🔄 Comment améliorer le modèle

Le prototype est conçu pour s'améliorer **uniquement en ajoutant plus de données** :

```
1. Prenez de nouvelles photos de plateaux
2. Annotez-les (Label Studio / Roboflow)
3. Relancez :
   python src/prepare_data.py
   python src/train.py
4. Le modèle sera automatiquement meilleur !
```

### Améliorations progressives

- **Court terme** : Plus d'annotations → meilleure détection
- **Moyen terme** : Passer de YOLOv8n → YOLOv8s ou YOLOv8m
- **Long terme** : Segmentation (YOLOv8-seg) pour une estimation de poids plus précise
- **Optionnel** : Caméra de profondeur pour estimation volumétrique

## 🛠️ Architecture technique

```
Images plateau → YOLOv8 (détection) → Bounding boxes + classes
                                          ↓
                              Surface relative calculée
                                          ↓
                              × densité alimentaire (g/cm²)
                                          ↓
                              = Poids estimé par aliment
                                          ↓
                              Agrégation → Statistiques → Dashboard
```

## 📊 Données optionnelles

Pour enrichir les analyses, vous pouvez fournir :

- **`data/menus.csv`** : menus du jour → analyse du gaspillage par menu
- **`data/production.csv`** : quantités produites → calcul du taux de gaspillage

Templates téléchargeables depuis le dashboard Streamlit.
