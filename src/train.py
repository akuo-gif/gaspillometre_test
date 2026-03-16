"""
GASPILLOMÈTRE - Entraînement du modèle YOLOv8
================================================
Entraîne un modèle YOLOv8 pour la détection d'aliments
sur les plateaux de cantine.

Utilise le transfer learning depuis les poids pré-entraînés
sur COCO pour être efficace même avec peu d'images (~80).

Usage:
    python src/train.py
    python src/train.py --epochs 200 --batch 16 --model yolov8s
    python src/train.py --resume  # reprendre un entraînement
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_config():
    """Charge la configuration d'entraînement."""
    with open(CONFIG_DIR / "config.yaml", "r") as f:
        return yaml.safe_load(f)


def check_dataset():
    """Vérifie que le dataset est prêt pour l'entraînement."""
    train_imgs = list((DATA_DIR / "images" / "train").glob("*"))
    val_imgs = list((DATA_DIR / "images" / "val").glob("*"))
    train_lbls = list((DATA_DIR / "labels" / "train").glob("*.txt"))
    val_lbls = list((DATA_DIR / "labels" / "val").glob("*.txt"))

    print("\n📊 État du dataset :")
    print(f"  Train : {len(train_imgs)} images, {len(train_lbls)} labels")
    print(f"  Val   : {len(val_imgs)} images, {len(val_lbls)} labels")

    if len(train_imgs) == 0:
        print("\n  ❌ Aucune image d'entraînement trouvée !")
        print("  Lancez d'abord :")
        print("    1. python src/launch_annotation.py  (annoter les images)")
        print("    2. python src/prepare_data.py        (préparer le dataset)")
        return False

    if len(train_lbls) == 0:
        print("\n  ❌ Aucune annotation trouvée !")
        print("  Annotez d'abord vos images avec Label Studio.")
        return False

    # Vérifier la cohérence
    train_img_stems = {Path(f).stem for f in train_imgs}
    train_lbl_stems = {Path(f).stem for f in train_lbls}
    missing = train_img_stems - train_lbl_stems
    if missing:
        print(f"\n  ⚠️  {len(missing)} images sans annotation dans train/")

    return True


def train(args):
    """Lance l'entraînement YOLOv8."""
    config = load_config()
    model_cfg = config["model"]
    train_cfg = config["training"]

    # Overrides depuis les arguments CLI
    model_name = args.model or model_cfg["name"]
    epochs = args.epochs or train_cfg["epochs"]
    batch_size = args.batch or train_cfg["batch_size"]
    imgsz = args.imgsz or model_cfg["imgsz"]

    print("\n🍽️  GASPILLOMÈTRE - Entraînement du modèle")
    print("=" * 50)

    # Vérifier le dataset
    if not check_dataset():
        sys.exit(1)

    # Charger le modèle
    data_yaml = str(CONFIG_DIR / "classes.yaml")

    if args.resume:
        # Reprendre un entraînement
        last_model = MODELS_DIR / "last.pt"
        if not last_model.exists():
            # Chercher dans les résultats YOLO
            last_candidates = list(RESULTS_DIR.rglob("last.pt"))
            if last_candidates:
                last_model = last_candidates[-1]
            else:
                print("  ❌ Aucun modèle à reprendre trouvé.")
                sys.exit(1)
        print(f"\n  🔄 Reprise depuis : {last_model}")
        model = YOLO(str(last_model))
    else:
        # Nouveau modèle avec transfer learning
        model_file = f"{model_name}.pt"
        print(f"\n  🧠 Modèle     : {model_name}")
        print(f"  📐 Image size : {imgsz}")
        print(f"  📦 Batch size : {batch_size}")
        print(f"  🔄 Epochs     : {epochs}")
        print(f"  ⏱️  Patience   : {train_cfg['patience']}")
        print(f"  📊 Dataset    : {data_yaml}")
        model = YOLO(model_file)

    # Créer le dossier de résultats
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    run_name = f"gaspillo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\n  🚀 Lancement de l'entraînement...")
    print(f"  Run: {run_name}")
    print("  " + "─" * 45)

    # Entraînement
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        patience=train_cfg["patience"],
        lr0=train_cfg["lr0"],
        lrf=train_cfg["lrf"],
        # Augmentations pour compenser le petit dataset
        augment=train_cfg["augment"],  # Active/désactive toutes les augmentations (True/False)
        hsv_h=0.015,        # Variation de teinte (couleur) : ±1.5% sur le cercle chromatique (léger, pour ne pas dénaturer les aliments)
        hsv_s=0.7,          # Variation de saturation : ±70% (simule des éclairages/caméras différents)
        hsv_v=0.4,          # Variation de luminosité : ±40% (simule des conditions de lumière variées, cantine plus ou moins éclairée)
        degrees=10.0,       # Rotation aléatoire de ±10° (les plateaux ne sont pas toujours droits sur la photo)
        translate=0.1,      # Translation : décale l'image de ±10% (le plateau n'est pas toujours centré)
        scale=0.5,          # Zoom aléatoire de ±50% (simule des distances de prise de vue différentes)
        shear=2.0,          # Cisaillement de ±2° (légère déformation en parallélogramme, simule l'angle de la caméra)
        flipud=0.5,         # 50% de chance de retourner l'image verticalement (haut ↔ bas)
        fliplr=0.5,         # 50% de chance de retourner l'image horizontalement (gauche ↔ droite)
        mosaic=1.0,         # Mosaïque à 100% : combine 4 images en une seule, force le modèle à détecter des objets plus petits et dans des contextes variés
        mixup=0.1,          # 10% de chance de superposer 2 images semi-transparentes, aide à la généralisation
        # Sortie
        project=str(RESULTS_DIR),
        name=run_name,
        save=True,
        save_period=10,      # Sauvegarder toutes les 10 epochs
        plots=True,
        verbose=True,
    )

    # Sauvegarder le meilleur modèle
    best_model_src = RESULTS_DIR / run_name / "weights" / "best.pt"
    if best_model_src.exists():
        best_model_dst = MODELS_DIR / "best.pt"
        import shutil
        shutil.copy2(best_model_src, best_model_dst)
        print(f"\n  ✅ Meilleur modèle sauvegardé : {best_model_dst.relative_to(PROJECT_ROOT)}")

        last_model_src = RESULTS_DIR / run_name / "weights" / "last.pt"
        if last_model_src.exists():
            shutil.copy2(last_model_src, MODELS_DIR / "last.pt")

    # Afficher les résultats
    print("\n" + "=" * 50)
    print("📈 RÉSULTATS DE L'ENTRAÎNEMENT")
    print("=" * 50)
    print(f"\n  Résultats complets : {RESULTS_DIR / run_name}")
    print(f"  Meilleur modèle   : models/best.pt")
    print(f"\n  Pour tester le modèle :")
    print(f"    python src/inference.py --image <chemin_image>")
    print(f"    python src/inference.py --dir imageplateau/")
    print(f"\n  Pour lancer le dashboard :")
    print(f"    streamlit run src/dashboard.py")

    return results


def validate(args):
    """Valide le modèle sur le jeu de validation."""
    model_path = MODELS_DIR / "best.pt"
    if not model_path.exists():
        print("  ❌ Aucun modèle entraîné trouvé (models/best.pt)")
        sys.exit(1)

    model = YOLO(str(model_path))
    data_yaml = str(CONFIG_DIR / "classes.yaml")

    print("\n📊 Validation du modèle...")
    results = model.val(data=data_yaml, verbose=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="Entraînement GASPILLOMÈTRE")
    parser.add_argument("--model", type=str, default=None,
                        help="Modèle YOLOv8 (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)")
    parser.add_argument("--epochs", type=int, default=None, help="Nombre d'epochs")
    parser.add_argument("--batch", type=int, default=None, help="Taille du batch")
    parser.add_argument("--imgsz", type=int, default=None, help="Taille des images")
    parser.add_argument("--resume", action="store_true", help="Reprendre le dernier entraînement")
    parser.add_argument("--validate", action="store_true", help="Mode validation uniquement")
    args = parser.parse_args()

    if args.validate:
        validate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
