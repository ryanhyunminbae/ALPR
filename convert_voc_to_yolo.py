from pathlib import Path
import xml.etree.ElementTree as ET
from tqdm import tqdm
import shutil
import argparse

# Map dataset-specific class names to YOLO class IDs.
CLASS_MAP = {
    "license-plate": 0,
    "licence": 0,  # spelling used in the Kaggle dataset
    "plate": 0,
}


def voc_to_yolo_box(size, box):
    w, h = size
    xmin, ymin, xmax, ymax = box
    x_center = (xmin + xmax) / 2.0 / w
    y_center = (ymin + ymax) / 2.0 / h
    width = (xmax - xmin) / w
    height = (ymax - ymin) / h
    return x_center, y_center, width, height


def _resolve_subdir(root: Path, candidates) -> Path:
    for name in candidates:
        candidate_path = root / name
        if candidate_path.exists():
            return candidate_path
    raise FileNotFoundError(f"None of {candidates} exist under {root}")


def convert_dataset(voc_root, yolo_root, overwrite: bool = True):
    voc_root = Path(voc_root)
    yolo_root = Path(yolo_root)

    annotations_dir = _resolve_subdir(voc_root, ["Annotations", "annotations", "labels"])
    images_dir = _resolve_subdir(voc_root, ["Images", "images"])

    images_out = yolo_root / "images"
    labels_out = yolo_root / "labels"

    if overwrite and images_out.exists():
        shutil.rmtree(images_out)
    if overwrite and labels_out.exists():
        shutil.rmtree(labels_out)

    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(annotations_dir.glob("*.xml"))
    for xml_file in tqdm(xml_files):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename = root.find("filename").text
        img_path = images_dir / filename
        if not img_path.exists():
            print(f"[skip] image missing: {img_path}")
            continue

        size = root.find("size")
        width = float(size.find("width").text)
        height = float(size.find("height").text)

        yolo_lines = []
        for obj in root.findall("object"):
            name = obj.find("name").text.strip()
            if name not in CLASS_MAP:
                continue
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
            x_c, y_c, w, h = voc_to_yolo_box((width, height), (xmin, ymin, xmax, ymax))
            class_id = CLASS_MAP[name]
            yolo_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

        if not yolo_lines:
            continue

        dest_img = images_out / filename
        shutil.copy2(img_path, dest_img)

        dest_label = labels_out / (Path(filename).stem + ".txt")
        dest_label.write_text("\n".join(yolo_lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--voc-root", required=True)
    parser.add_argument("--yolo-root", required=True)
    parser.add_argument("--no-overwrite", action="store_true", help="Append instead of replacing output directories.")
    args = parser.parse_args()
    convert_dataset(args.voc_root, args.yolo_root, overwrite=not args.no_overwrite)
