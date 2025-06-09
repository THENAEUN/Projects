import cv2                             # OpenCV 라이브러리 불러오기: 이미지∙비디오 처리
import os                              # 운영체제(OS) 경로 확인 및 파일 존재 여부 확인
import numpy as np                     # 수치 계산 및 배열 처리용 NumPy
from ultralytics import YOLO           # Ultralytics YOLO 모델 불러오기
from PIL import ImageFont, ImageDraw, Image  # 한글 텍스트 출력을 위한 PIL 모듈
import time                            # 실행 시간 측정용

def put_text_on_image(img, text, position, font_size=30, font_color=(0,255,0)):
    # OpenCV(BGR) 이미지에 PIL을 이용해 한글 텍스트를 그린 뒤 다시 NumPy 배열로 반환
    img_pil = Image.fromarray(img)     # NumPy 배열 → PIL 이미지
    draw = ImageDraw.Draw(img_pil)     # 텍스트 드로잉 객체 생성
    try:
        font = ImageFont.truetype("malgun.ttf", font_size)  # 맑은고딕 폰트 로드
    except:
        font = ImageFont.load_default()  # 폰트 로드 실패 시 기본 폰트 사용
    # fill에 BGR→RGB 변환을 위해 font_color[::-1] 사용
    draw.text(position, text, font=font, fill=font_color[::-1])
    return np.array(img_pil)           # PIL 이미지 → NumPy 배열

def calculate_iou(box1, box2):
    # 두 바운딩 박스(box1, box2)의 교집합 비율(IOU) 계산
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    if x2 < x1 or y2 < y1:
        return 0.0                      # 교집합이 없으면 IOU=0
    inter = (x2 - x1) * (y2 - y1)       # 교집합 넓이
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])  # 박스1 넓이
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])  # 박스2 넓이
    return inter / (area1 + area2 - inter)       # IOU 계산식

def center_distance(box1, box2):
    # 두 박스의 중심점 간 유클리드 거리 계산
    cx1, cy1 = (box1[0]+box1[2])/2, (box1[1]+box1[3])/2
    cx2, cy2 = (box2[0]+box2[2])/2, (box2[1]+box2[3])/2
    return np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)

# ==== 설정 파라미터 ====
video_path     = r"C:\workspace\list\0_8_IPC1_20230108162038.mp4"  # 입력 비디오 파일 경로
model_path     = r"C:\workspace\list\best.pt"                    # YOLO 모델 파일 경로
grid_size      = 3                                              # 프레임을 N×N 그리드로 분할할 때 N
selected_cell  = (2, 0)                                         # 처리할 셀 위치 (row, col)
conf_threshold = 0.25                                           # 검출 신뢰도 임계값
DIST_THRESHOLD = 50                                             # 복원용 최대 중심 거리(픽셀)
scale_factor   = 1.0                                            # 출력 영상 스케일 비율

# 비디오/모델 파일 존재 여부 확인
if not os.path.exists(video_path): exit(f"파일이 없습니다: {video_path}")

model = YOLO(model_path)             # YOLO 모델 로드
cap   = cv2.VideoCapture(video_path) # 비디오 캡처 객체 생성

# 비디오 정보 추출
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # 프레임 폭
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 프레임 높이
fps    = cap.get(cv2.CAP_PROP_FPS)                # 초당 프레임 수
delay  = int(1000 / fps)                          # 화면 표시 딜레이(ms)

# 추적∙메모리 관리용 변수 초기화
last_detections       = []  # 직전 프레임 검출 기록
frame_memory          = []  # 과거 N프레임 검출 기록(복원용)
tracked_ids_per_frame = []  # 프레임별 ID 집합 기록
id_counter            = 0   # 새 객체에 부여할 순차 ID
MAX_MEMORY            = 5   # 기억할 과거 프레임 개수

start_time = time.time()      # 처리 시작 시간 기록

while True:
    ret, frame = cap.read()   # 프레임 읽기
    if not ret: break         # 더 이상 읽을 프레임 없으면 종료

    # 전체 프레임을 grid_size×grid_size로 분할 → 선택한 셀만 잘라냄
    cell_w, cell_h = width//grid_size, height//grid_size
    r, c = selected_cell
    y1, y2 = r*cell_h, (r+1)*cell_h
    x1, x2 = c*cell_w, (c+1)*cell_w
    cell_frame = frame[y1:y2, x1:x2]

    # YOLO 예측 수행 (셀 단위)
    results = model.predict(source=cell_frame, conf=conf_threshold, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # 검출된 박스 좌표 리스트

    new_detections = []        # 이번 프레임 검출+ID 저장용
    ids = [-1]*len(boxes)      # 박스별 ID 초기값

    # --- ID 매칭 로직 ---
    for i, box in enumerate(boxes):
        # cell-relative 좌표 → 전체 프레임 절대 좌표로 변환
        x1_abs, y1_abs, x2_abs, y2_abs = box + np.array([c*cell_w, r*cell_h, c*cell_w, r*cell_h])
        # 직전 프레임과 IOU 비교하여 best match 찾기
        best_iou, best_id = 0, -1
        for prev_box, prev_id in last_detections:
            iou = calculate_iou((x1_abs,y1_abs,x2_abs,y2_abs), prev_box)
            if iou > best_iou:
                best_iou, best_id = iou, prev_id
        if best_iou > 0.5:
            ids[i] = best_id          # IOU 기준 매칭 성공 → 같은 ID
        else:
            ids[i] = id_counter       # 매칭 실패 → 새 ID 부여
            id_counter += 1

        # --- 메모리 복원 로직 (ID == -1일 경우) ---
        if ids[i] == -1:
            for memory in reversed(frame_memory):
                for mem_box, mem_id in memory:
                    if (center_distance((x1_abs,y1_abs,x2_abs,y2_abs), mem_box) < DIST_THRESHOLD
                        and calculate_iou((x1_abs,y1_abs,x2_abs,y2_abs), mem_box) > 0.25):
                        ids[i] = mem_id  # 과거 메모리와 기준 충족 시 복원
                        break
                if ids[i] != -1:
                    break

        new_detections.append(((x1_abs,y1_abs,x2_abs,y2_abs), ids[i])) 
    # 메모리 및 직전 검출값 업데이트
    last_detections = new_detections.copy()
    frame_memory.append(new_detections)
    if len(frame_memory) > MAX_MEMORY:
        frame_memory.pop(0)

    # 프레임별 ID 집합 기록 (ID 스위칭 수 계산용)
    tracked_ids_per_frame.append(set(ids))

    # === 시각화 ===
    plotted = cell_frame.copy()
    for (bx1,by1,bx2,by2), obj_id in new_detections:
        # 박스 그리기 (셀 내 좌표로 변환 모듈러 연산)
        cv2.rectangle(plotted,
                      (int(bx1)%cell_w, int(by1)%cell_h),
                      (int(bx2)%cell_w, int(by2)%cell_h),
                      (0,255,0), 2)
        # ID 텍스트 출력
        cv2.putText(plotted, f"ID {obj_id}",
                    (int(bx1)%cell_w, int(by1)%cell_h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255,0,255), 2)
    # 닭 마리수 텍스트 출력 (한글 지원)
    chicken_count = len(new_detections)
    plotted = put_text_on_image(plotted,
                                f"(2,0) 셀 - 닭 {chicken_count}마리",
                                (10,30), 25)

    cv2.imshow("Selected Cell", plotted)  # 결과 화면 표시
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break  # 'q' 키 입력 시 루프 종료

# === 성능 측정 결과 출력 ===
end_time = time.time()
total_time = end_time - start_time

id_switch_count = 0
for i in range(1, len(tracked_ids_per_frame)):
    new_ids = tracked_ids_per_frame[i] - tracked_ids_per_frame[i-1]
    id_switch_count += len(new_ids)

print("\n=== 성능 측정 결과 ===")
print(f"총 프레임 수: {len(tracked_ids_per_frame)}")
print(f"ID Switching 횟수: {id_switch_count}")
print(f"총 실행 시간: {total_time:.2f}초")
