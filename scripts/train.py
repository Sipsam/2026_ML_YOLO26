
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

def main():
    model = YOLO("yolo26s.pt")

    model.train(
        data=str(ROOT / "datasets/final/data.yaml"),
        epochs=100,
        imgsz=960,
        batch=16,
        device=0,
        patience=20,
        workers=8,
        project=str(ROOT / "runs"),
        name="lol_final_yolo26s",
    )

if __name__ == "__main__":
    main()
