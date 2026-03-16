"""
GASPILLOMÈTRE - Préparation des données
========================================
Ce script :
1. Organise les images en dossiers train/val
2. Vérifie la cohérence images/annotations
3. Génère des statistiques sur le dataset

Usage:
    python src/prepare_data.py
    python src/prepare_data.py --split 0.8  # 80% train, 20% val
"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path
from collections import Counter

import yaml
from tqdm import tqdm
from PIL import Image


# ── Chemins ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RAW_IMAGES_DIR = PROJECT_ROOT / "imageplateau"


def load_config():
    """Charge la configuration des classes."""
    with open(CONFIG_DIR / "classes.yaml", "r") as f:
        return yaml.safe_load(f)


def setup_directories():
    """Crée l'arborescence YOLO attendue."""
    dirs = [
        DATA_DIR / "images" / "train",
        DATA_DIR / "images" / "val",
        DATA_DIR / "labels" / "train",
        DATA_DIR / "labels" / "val",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  📁 {d.relative_to(PROJECT_ROOT)}")
    return dirs


def find_images(source_dir: Path) -> list:
    """Trouve toutes les images dans un dossier."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    images = []
    for f in sorted(source_dir.iterdir()):
        if f.suffix.lower() in extensions:
            images.append(f)
    return images


def find_annotation(image_path: Path) -> Path | None:
    """
    Cherche le fichier d'annotation YOLO correspondant à une image.
    Recherche dans plusieurs emplacements possibles.
    """
    stem = image_path.stem
    search_dirs = [
        image_path.parent,                          # même dossier
        image_path.parent / "labels",               # sous-dossier labels
        PROJECT_ROOT / "annotations",               # dossier annotations
        PROJECT_ROOT / "labels",                     # dossier labels
    ]
    for d in search_dirs:
        label_file = d / f"{stem}.txt"
        if label_file.exists():
            return label_file
    return None


def validate_annotation(label_path: Path, num_classes: int) -> tuple:
    """
    Valide un fichier d'annotation YOLO.
    Retourne (is_valid, num_objects, class_counts, errors).
    """
    errors = []
    class_counts = Counter()
    num_objects = 0

    with open(label_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"Ligne {line_num}: attendu 5 valeurs, trouvé {len(parts)}")
                continue

            try:
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
            except ValueError:
                errors.append(f"Ligne {line_num}: valeurs non numériques")
                continue

            if class_id < 0 or class_id >= num_classes:
                errors.append(f"Ligne {line_num}: classe {class_id} hors limites [0, {num_classes-1}]")

            for val, name in zip([x_center, y_center, width, height],
                                  ["x_center", "y_center", "width", "height"]):
                if val < 0 or val > 1:
                    errors.append(f"Ligne {line_num}: {name}={val} hors [0, 1]")

            class_counts[class_id] += 1
            num_objects += 1

    is_valid = len(errors) == 0
    return is_valid, num_objects, class_counts, errors


def split_dataset(images: list, labels: dict, train_ratio: float = 0.8, seed: int = 42):
    """
    Sépare les images annotées en ensembles train/val.
    Les images sans annotation sont listées séparément.
    """
    random.seed(seed)

    annotated = [(img, labels[img]) for img in images if img in labels]
    unannotated = [img for img in images if img not in labels]

    random.shuffle(annotated)
    split_idx = int(len(annotated) * train_ratio)

    train_set = annotated[:split_idx]
    val_set = annotated[split_idx:]

    return train_set, val_set, unannotated


def copy_files(file_pairs: list, img_dest: Path, lbl_dest: Path):
    """Copie les paires image/annotation vers les dossiers de destination."""
    for img_path, lbl_path in tqdm(file_pairs, desc=f"  → {img_dest.parent.name}/{img_dest.name}"):
        # Copier l'image (convertir en .jpg si nécessaire)
        dest_img = img_dest / img_path.name
        shutil.copy2(img_path, dest_img)

        # Copier le label
        dest_lbl = lbl_dest / lbl_path.name
        shutil.copy2(lbl_path, dest_lbl)


def generate_stats(train_set, val_set, unannotated, class_names, num_classes):
    """Affiche les statistiques du dataset."""
    print("\n" + "=" * 60)
    print("📊 STATISTIQUES DU DATASET")
    print("=" * 60)

    total = len(train_set) + len(val_set) + len(unannotated)
    annotated = len(train_set) + len(val_set)

    print(f"\n  Images totales     : {total}")
    print(f"  Images annotées    : {annotated} ({100*annotated/total:.0f}%)")
    print(f"  Images NON annotées: {len(unannotated)} ({100*len(unannotated)/total:.0f}%)")
    print(f"  ├── Train          : {len(train_set)}")
    print(f"  └── Val            : {len(val_set)}")

    # Comptage par classe
    all_counts = Counter()
    for _, lbl_path in train_set + val_set:
        _, _, counts, _ = validate_annotation(lbl_path, num_classes)
        all_counts.update(counts)

    if all_counts:
        print(f"\n  Objets annotés par classe :")
        for class_id in sorted(all_counts.keys()):
            name = class_names.get(class_id, f"classe_{class_id}")
            count = all_counts[class_id]
            bar = "█" * min(count, 40)
            print(f"    {name:12s} : {count:4d} {bar}")

    if unannotated:
        print(f"\n  ⚠️  Images à annoter :")
        for img in unannotated[:10]:
            print(f"    - {img.name}")
        if len(unannotated) > 10:
            print(f"    ... et {len(unannotated) - 10} autres")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Préparation des données GASPILLOMÈTRE")
    parser.add_argument("--split", type=float, default=0.8, help="Ratio train/val (défaut: 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="Seed aléatoire")
    parser.add_argument("--source", type=str, default=None, help="Dossier source des images")
    args = parser.parse_args()

    print("\n🍽️  GASPILLOMÈTRE - Préparation des données")
    print("=" * 50)

    # Charger config
    config = load_config()
    class_names = config["names"]
    num_classes = config["nc"]
    print(f"\n  Classes configurées : {num_classes}")
    for k, v in class_names.items():
        print(f"    {k}: {v}")

    # Créer les dossiers
    print("\n📁 Création de l'arborescence YOLO...")
    setup_directories()

    # Trouver les images
    source = Path(args.source) if args.source else RAW_IMAGES_DIR
    print(f"\n🔍 Recherche d'images dans : {source}")
    images = find_images(source)
    print(f"  {len(images)} images trouvées")

    if not images:
        print("  ❌ Aucune image trouvée ! Vérifiez le dossier source.")
        sys.exit(1)

    # Chercher les annotations
    print("\n🏷️  Recherche des annotations YOLO...")
    labels = {}
    errors_found = []
    for img in images:
        lbl = find_annotation(img)
        if lbl:
            is_valid, n_obj, counts, errs = validate_annotation(lbl, num_classes)
            if is_valid:
                labels[img] = lbl
            else:
                errors_found.append((img.name, errs))

    print(f"  {len(labels)} annotations valides trouvées")

    if errors_found:
        print(f"  ⚠️  {len(errors_found)} annotations avec erreurs :")
        for name, errs in errors_found:
            print(f"    {name}:")
            for e in errs:
                print(f"      - {e}")

    # Séparer train/val
    print(f"\n✂️  Séparation train/val (ratio={args.split})...")
    train_set, val_set, unannotated = split_dataset(images, labels, args.split, args.seed)

    # Copier les fichiers
    if train_set or val_set:
        print("\n📋 Copie des fichiers...")
        copy_files(train_set, DATA_DIR / "images" / "train", DATA_DIR / "labels" / "train")
        copy_files(val_set, DATA_DIR / "images" / "val", DATA_DIR / "labels" / "val")

    # Statistiques
    generate_stats(train_set, val_set, unannotated, class_names, num_classes)

    if not labels:
        print("\n" + "=" * 60)
        print("🚀 PROCHAINE ÉTAPE : ANNOTER VOS IMAGES !")
        print("=" * 60)
        print("""
  Vos 78 images n'ont pas encore d'annotations YOLO.
  
  Option 1 - Label Studio (recommandé) :
    python src/launch_annotation.py
    
  Option 2 - CVAT (en ligne) :
    https://app.cvat.ai
    
  Option 3 - Roboflow (semi-auto) :
    https://roboflow.com
    
  Format attendu : fichier .txt par image avec :
    <class_id> <x_center> <y_center> <width> <height>
    (coordonnées normalisées entre 0 et 1)
    
  Placez les annotations dans :
    imagesplateau/<nom_image>.txt
    ou
    annotations/<nom_image>.txt
""")

    print("✅ Préparation terminée !\n")


if __name__ == "__main__":
    main()
