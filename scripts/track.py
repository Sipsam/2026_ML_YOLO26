
from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict, deque
import cv2
import numpy as np

ROOT = Path("/home/sipsam/lol-yolo")

MODEL = ROOT / "runs/lol_final_yolo26s/weights/best.pt"
VIDEO = ROOT / "gameplay.mp4"
OUTPUT = ROOT / "results/tracking.mp4"

# 원하는 설정
START_SECONDS = 0       # 영상 시작 지점
DURATION_SECONDS = 25    # 제출용 영상 길이
CONFIDENCE = 0.25
BOX_WIDTH = 1
FONT_SIZE = 9
TRAIL_LENGTH = 30       # 이동 경로 최대 30프레임
TRAIL_WIDTH = 1

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

model = YOLO(str(MODEL))
cap = cv2.VideoCapture(str(VIDEO))

if not cap.isOpened():
    raise RuntimeError("영상 파일을 열 수 없습니다.")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cap.set(cv2.CAP_PROP_POS_MSEC, START_SECONDS * 1000)

writer = cv2.VideoWriter(
    str(OUTPUT),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

if not writer.isOpened():
    raise RuntimeError("출력 영상 파일을 만들 수 없습니다.")

history = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))
max_frames = round(fps * DURATION_SECONDS)

for frame_idx in range(max_frames):
    success, frame = cap.read()
    if not success:
        break

    result = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        imgsz=960,
        conf=CONFIDENCE,
        device=0,
        verbose=False
    )[0]

    # Bounding Box + 클래스 + 추적 ID
    annotated = result.plot(
        line_width=BOX_WIDTH,
        font_size=FONT_SIZE,
        conf=True,
        labels=True
    )

    # 객체별 이동 궤적
    if result.boxes is not None and result.boxes.id is not None:
        boxes = result.boxes.xywh.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            history[track_id].append((int(x), int(y)))

            points = np.array(
                history[track_id],
                dtype=np.int32
            ).reshape(-1, 1, 2)

            if len(points) > 1:
                cv2.polylines(
                    annotated,
                    [points],
                    False,
                    (0, 255, 255),
                    TRAIL_WIDTH
                )

    writer.write(annotated)

    if (frame_idx + 1) % 30 == 0:
        print(f"Processed {frame_idx + 1}/{max_frames}")

cap.release()
writer.release()

print("Saved:", OUTPUT)
