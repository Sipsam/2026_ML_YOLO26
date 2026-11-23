
from pathlib import Path
from shutil import copy2
from collections import Counter
from ultralytics import YOLO
import csv
import zipfile

ROOT = Path.home() / "lol-yolo"

MODEL_PATH = (
    ROOT / "runs/detect/runs/seed_model/weights/best.pt"
)
SOURCE_DIR = ROOT / "unlabeled"
OUTPUT_DIR = ROOT / "auto_labeled"

IMAGE_DIR = OUTPUT_DIR / "images"
LABEL_DIR = OUTPUT_DIR / "labels"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO(str(MODEL_PATH))

image_paths = sorted(
    p for p in SOURCE_DIR.rglob("*")
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

print(f"Images: {len(image_paths)}")

if not image_paths:
    raise RuntimeError("No images found")

counts = Counter()
records = []

for i, image_path in enumerate(image_paths, 1):

    # 원본 파일명 유지 (같은 이름이 있으면 오류 방지)
    filename = image_path.name
    dest_image = IMAGE_DIR / filename
    dest_label = LABEL_DIR / f"{image_path.stem}.txt"

    if dest_image.exists():
        raise FileExistsError(f"Duplicate: {filename}")

    results = model.predict(
        source=str(image_path),
        imgsz=960,
        conf=0.15,
        device=0,
        verbose=False,
    )

    result = results[0]
    lines = []

    if result.boxes is not None:
        for box in result.boxes:
            cls = int(box.cls.item())
            x, y, w, h = box.xywhn[0].tolist()

            lines.append(
                f"{cls} {x:.6f} {y:.6f} "
                f"{w:.6f} {h:.6f}"
            )
            counts[cls] += 1

    copy2(image_path, dest_image)

    dest_label.write_text(
        "\n".join(lines) + ("\n" if lines else ""),
        encoding="utf-8",
    )

    records.append((filename, len(lines)))

    if i % 50 == 0 or i == len(image_paths):
        print(f"[{i}/{len(image_paths)}] processed")

# 이미지별 탐지 수 저장
with (OUTPUT_DIR / "manifest.csv").open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "detections"])
    writer.writerows(records)

# 이미지 + 라벨 압축
zip_path = ROOT / "auto_labeled.zip"

with zipfile.ZipFile(
    zip_path, "w", zipfile.ZIP_DEFLATED
) as z:
    for folder in [IMAGE_DIR, LABEL_DIR]:
        for path in sorted(folder.iterdir()):
            z.write(
                path,
                arcname=str(path.relative_to(OUTPUT_DIR))
            )

print("\nAuto labeling completed")
print("Class detections:", dict(counts))
print("Images with zero detections:",
      sum(n == 0 for _, n in records))
print("ZIP:", zip_path)
