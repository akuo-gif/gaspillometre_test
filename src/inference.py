"""
GASPILLOMÈTRE - Inférence et estimation de gaspillage
=======================================================
Détecte les aliments sur un plateau, estime le poids
des restes et calcule le gaspillage.

Usage:
    python src/inference.py --image chemin/image.jpg
    python src/inference.py --dir imagesplateau/
    python src/inference.py --camera 0  # webcam temps réel
"""

import argparse
import json
import csv
import sys
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"


def load_configs():
    """Charge toutes les configurations."""
    with open(CONFIG_DIR / "config.yaml", "r") as f:
        config = yaml.safe_load(f)
    with open(CONFIG_DIR / "classes.yaml", "r") as f:
        classes = yaml.safe_load(f)
    return config, classes


def load_model(model_path: str = None) -> YOLO:
    """Charge le modèle YOLOv8 entraîné."""
    if model_path:
        path = Path(model_path)
    else:
        path = MODELS_DIR / "best.pt"

    if not path.exists():
        print(f"  ❌ Modèle non trouvé : {path}")
        print(f"  Entraînez d'abord avec : python src/train.py")
        sys.exit(1)

    print(f"  🧠 Modèle chargé : {path}")
    return YOLO(str(path))


class WeightEstimator:
    """
    Estimateur de poids basé sur la surface détectée.
    
    Principe :
    1. On détecte la boîte englobante de chaque aliment
    2. On calcule la surface relative par rapport au plateau
    3. On multiplie par la densité surfacique de l'aliment
    
    Note : C'est une estimation grossière. Pour plus de précision,
    il faudrait une caméra de profondeur ou une balance.
    """

    def __init__(self, config: dict, class_names: dict):
        self.ref_area = config["weight_estimation"]["reference_tray_area_cm2"]
        self.densities = config["weight_estimation"]["density_g_per_cm2"]
        self.class_names = class_names

    def estimate_weight(self, class_name: str, bbox_area_ratio: float) -> float:
        """
        Estime le poids d'un aliment détecté.
        
        Args:
            class_name: Nom de la classe d'aliment
            bbox_area_ratio: Ratio surface bbox / surface image
        
        Returns:
            Poids estimé en grammes
        """
        # Surface estimée en cm²
        area_cm2 = bbox_area_ratio * self.ref_area

        # Densité de l'aliment (g/cm²)
        density = self.densities.get(class_name, 0.5)  # 0.5 par défaut

        # Poids estimé
        # Facteur 0.7 : la bbox contient ~70% d'aliment en moyenne
        weight_g = area_cm2 * density * 0.7

        return round(weight_g, 1)


class WasteDetector:
    """
    Détecteur de gaspillage alimentaire.
    Combine détection YOLO + estimation de poids.
    """

    def __init__(self, model: YOLO, config: dict, class_names: dict):
        self.model = model
        self.config = config
        self.class_names = class_names
        self.weight_estimator = WeightEstimator(config, class_names)
        self.conf_threshold = config["model"]["confidence_threshold"]
        self.iou_threshold = config["model"]["iou_threshold"]

    def detect(self, image_path: str | np.ndarray) -> dict:
        """
        Analyse une image de plateau.
        
        Returns:
            dict avec détections, poids estimés, et image annotée
        """
        # Inférence
        results = self.model(
            image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        result = results[0]
        img = result.orig_img.copy()
        img_h, img_w = img.shape[:2]
        img_area = img_h * img_w

        detections = []
        total_weight = 0

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                # Extraire les infos
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.class_names.get(class_id, f"classe_{class_id}")

                # Surface relative
                bbox_w = x2 - x1
                bbox_h = y2 - y1
                bbox_area_ratio = (bbox_w * bbox_h) / img_area

                # Estimation du poids
                weight = self.weight_estimator.estimate_weight(class_name, bbox_area_ratio)
                total_weight += weight

                detection = {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(conf, 3),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "area_ratio": round(bbox_area_ratio, 4),
                    "weight_g": weight,
                }
                detections.append(detection)

                # Dessiner sur l'image
                color = self._get_color(class_id)
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                label = f"{class_name} {conf:.0%} ~{weight}g"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(img,
                              (int(x1), int(y1) - label_size[1] - 10),
                              (int(x1) + label_size[0], int(y1)),
                              color, -1)
                cv2.putText(img, label,
                            (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Résumé en bas de l'image
        summary = f"Aliments: {len(detections)} | Poids total: ~{total_weight:.0f}g"
        cv2.rectangle(img, (0, img_h - 40), (img_w, img_h), (0, 0, 0), -1)
        cv2.putText(img, summary, (10, img_h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return {
            "detections": detections,
            "total_weight_g": round(total_weight, 1),
            "num_items": len(detections),
            "annotated_image": img,
            "image_size": (img_w, img_h),
            "timestamp": datetime.now().isoformat(),
        }

    def _get_color(self, class_id: int) -> tuple:
        """Couleur unique par classe."""
        colors = [
            (107, 107, 255), (76, 205, 196), (69, 183, 209),
            (150, 206, 180), (255, 234, 167), (221, 160, 221),
            (152, 216, 200), (247, 220, 111), (187, 143, 206),
            (133, 193, 233), (248, 196, 113), (130, 224, 170),
            (241, 148, 138), (174, 214, 241), (215, 189, 226),
        ]
        return colors[class_id % len(colors)]


def log_detection(result: dict, image_name: str, log_file: Path):
    """Enregistre les résultats dans un fichier CSV."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_file.exists()

    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "image", "num_items", "total_weight_g",
                "aliments_detectes", "details_json"
            ])

        aliments = ", ".join([d["class_name"] for d in result["detections"]])
        # Convertir les float32 numpy en float Python pour la sérialisation JSON
        detections_serializable = [
            {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in d.items()}
            for d in result["detections"]
        ]
        details = json.dumps(detections_serializable, ensure_ascii=False)
        writer.writerow([
            result["timestamp"],
            image_name,
            result["num_items"],
            result["total_weight_g"],
            aliments,
            details,
        ])


def process_image(detector: WasteDetector, image_path: Path, output_dir: Path, log_file: Path):
    """Traite une seule image."""
    print(f"\n  📸 {image_path.name}")

    result = detector.detect(str(image_path))

    # Afficher les résultats
    if result["detections"]:
        for det in result["detections"]:
            print(f"    🍽️  {det['class_name']:12s} | confiance: {det['confidence']:.0%} | ~{det['weight_g']}g")
        print(f"    {'─' * 45}")
        print(f"    📊 Total: {result['num_items']} aliments, ~{result['total_weight_g']}g")
    else:
        print(f"    ⚪ Aucun aliment détecté (plateau vide ?)")

    # Sauvegarder l'image annotée
    output_path = output_dir / f"detected_{image_path.stem}.jpg"
    cv2.imwrite(str(output_path), result["annotated_image"])
    print(f"    💾 Sauvegardé : {output_path.relative_to(PROJECT_ROOT)}")

    # Logger
    log_detection(result, image_path.name, log_file)

    return result


def run_camera(detector: WasteDetector, camera_id: int = 0):
    """Mode caméra temps réel."""
    print(f"\n  📹 Mode caméra (ID: {camera_id})")
    print(f"  Appuyez sur 'q' pour quitter, 's' pour sauvegarder")

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("  ❌ Impossible d'ouvrir la caméra")
        return

    log_file = LOGS_DIR / "camera_log.csv"
    output_dir = RESULTS_DIR / "camera"
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Détection toutes les 30 frames pour la fluidité
        if frame_count % 30 == 0:
            result = detector.detect(frame)
            display = result["annotated_image"]
        else:
            display = frame

        cv2.imshow("Gaspillometre - Temps Reel", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = output_dir / f"capture_{ts}.jpg"
            cv2.imwrite(str(save_path), result["annotated_image"])
            log_detection(result, f"capture_{ts}.jpg", log_file)
            print(f"  💾 Capture sauvegardée : {save_path.name}")

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Inférence GASPILLOMÈTRE")
    parser.add_argument("--image", type=str, help="Chemin vers une image")
    parser.add_argument("--dir", type=str, help="Dossier d'images à traiter")
    parser.add_argument("--camera", type=int, default=None, help="ID de la caméra (0 pour webcam)")
    parser.add_argument("--model", type=str, default=None, help="Chemin vers le modèle .pt")
    parser.add_argument("--conf", type=float, default=None, help="Seuil de confiance")
    parser.add_argument("--output", type=str, default=None, help="Dossier de sortie")
    args = parser.parse_args()

    print("\n🍽️  GASPILLOMÈTRE - Détection et estimation")
    print("=" * 50)

    # Charger configs et modèle
    config, classes_cfg = load_configs()
    class_names = classes_cfg["names"]

    if args.conf:
        config["model"]["confidence_threshold"] = args.conf

    model = load_model(args.model)
    detector = WasteDetector(model, config, class_names)

    # Dossier de sortie
    output_dir = Path(args.output) if args.output else RESULTS_DIR / "detections"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "detections.csv"

    if args.camera is not None:
        run_camera(detector, args.camera)

    elif args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"  ❌ Image non trouvée : {image_path}")
            sys.exit(1)
        process_image(detector, image_path, output_dir, log_file)

    elif args.dir:
        dir_path = Path(args.dir)
        if not dir_path.exists():
            print(f"  ❌ Dossier non trouvé : {dir_path}")
            sys.exit(1)

        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images = sorted([f for f in dir_path.iterdir() if f.suffix.lower() in extensions])
        print(f"\n  📂 {len(images)} images trouvées dans {dir_path}")

        all_results = []
        for img_path in images:
            result = process_image(detector, img_path, output_dir, log_file)
            all_results.append(result)

        # Résumé global
        total_items = sum(r["num_items"] for r in all_results)
        total_weight = sum(r["total_weight_g"] for r in all_results)
        print(f"\n{'=' * 50}")
        print(f"📊 RÉSUMÉ GLOBAL")
        print(f"  Plateaux analysés : {len(all_results)}")
        print(f"  Aliments détectés : {total_items}")
        print(f"  Poids total estimé: ~{total_weight:.0f}g ({total_weight/1000:.1f}kg)")
        print(f"  Moyenne/plateau   : ~{total_weight/len(all_results):.0f}g")
        print(f"\n  📋 Log : {log_file.relative_to(PROJECT_ROOT)}")
        print(f"  🖼️  Images : {output_dir.relative_to(PROJECT_ROOT)}/")

    else:
        parser.print_help()

    print("\n✅ Terminé !\n")


if __name__ == "__main__":
    main()
