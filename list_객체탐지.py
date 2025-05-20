import cv2
import os
import numpy as np
from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image

def put_text_on_image(img, text, position, font_size=30, font_color=(0, 255, 0)):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("malgun.ttf", font_size)
    except:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=font_color[::-1])
    return np.array(img_pil)

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 < x1 or y2 < y1:
        return 0.0
    inter_area = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter_area
    return inter_area / union

# === 설정 ===
video_path = r"C:\workspace\list\0_8_IPC1_20230108162038.mp4"
model_path = r"C:\workspace\list\best.pt"
grid_size = 3
selected_cell = (2, 0)  # (행, 열)
scale_factor = 1.0
conf_threshold = 0.5

# === 초기화 ===
if not os.path.exists(video_path):
    print(f"❌ 파일이 존재하지 않습니다: {video_path}")
    exit()

model = YOLO(model_path)
cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)

# 이전 프레임에서 탐지한 객체들의 정보(바운딩 박스 위치 + ID)(x1, y1, x2, y2), obj_id)를 저장 / 다음 프레임에서 ID가 없어진 객체와 비교할 때 사용 (IOU 기반 ID 복원)
last_detections = []

# === 메인 루프 ===
while True:
    # 프레임 읽기 (ret - True, False / frame - 이미지 데이터)
    ret, frame = cap.read()
    if not ret:
        break
    
    # 아래 (2,0) 셀 선택
    row, col = selected_cell
    # 해당 셀만 잘라서 cell_frame 사용
    cell_h = height // grid_size
    cell_w = width // grid_size
    x, y = col * cell_w, row * cell_h
    cell_frame = frame[y:y+cell_h, x:x+cell_w]

    # 객체 탐지 + 추적
    results = model.track(cell_frame, conf=conf_threshold, persist=True, verbose=False)[0]
    #  YOLO 모델이 반환한 탐지 결과를 리스트 형태로 받음
    # 탐지된 객체들의 좌표 / 모든 객체의 바운딩 박스 좌표
    boxes = results.boxes.xyxy.cpu().tolist()
    # 각 객체의 고유 ID / 없으면 -1로 채움 
    ids = results.boxes.id.int().cpu().tolist() if results.boxes.id is not None else [-1] * len(boxes)

    #  현재 프레임에서 새로 탐지된 객체들의 정보를 저장 / 현재 프레임 시각화 및 다음 프레임 비교를 위해 사용
    new_detections = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        obj_id = ids[i]

        # ID가 없을 경우 IOU로 ID 복원
        # 이전 프레임에 있던 박스들과 IOU(겹치는 정도) 비교
        # IOU (두 개의 바운딩 박스(직사각형)가 얼마나 겹치는지를 나타내는 지표) 가 0.5 이상이면 같은 객체로 간주하고 이전 ID를 붙임
        if obj_id == -1:  # ID가 없는 경우(잠깐 사라진 객체), 이전 박스와 비교
            for last_box, last_id in last_detections:
                iou = calculate_iou(box, last_box)
                if iou > 0.5:
                    obj_id = last_id
                    break
        
        # 추적 결과 저장 / 객체 하나의 위치와 ID를 새 리스트에 저장
        new_detections.append(((x1, y1, x2, y2), obj_id))

    # === 시각화 ===
    plotted = cell_frame.copy()
    for (x1, y1, x2, y2), obj_id in new_detections:
        cv2.rectangle(plotted, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(plotted, f"ID {obj_id}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    if scale_factor != 1.0:
        h, w = plotted.shape[:2]
        new_size = (int(w * scale_factor), int(h * scale_factor))
        plotted = cv2.resize(plotted, new_size)

    chicken_count = len(new_detections)
    plotted = put_text_on_image(plotted, f"(2,0) 셀 - 닭 {chicken_count}마리", (10, 30), 25)

    cv2.imshow("Selected Cell Detection with ID Recovery", plotted)

    # 현재 프레임 정보 저장, 다음 프레임에서 IOU 기반 비교할 수 있도록 함
    last_detections = [((x1, y1, x2, y2), obj_id) for (x1, y1, x2, y2), obj_id in new_detections]

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

