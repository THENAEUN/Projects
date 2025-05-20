# OpenCV 라이브러리 임포트 - 이미지 및 비디오 처리 기능 제공
import cv2
# 파일 및 디렉토리 경로 관리를 위한 os 모듈 임포트
import os
# 수치 계산을 위한 넘파이 라이브러리 임포트
import numpy as np
# Ultralytics의 YOLO 객체 탐지 모델 임포트
from ultralytics import YOLO
# 한글 텍스트 처리를 위한 PIL 라이브러리 임포트
from PIL import ImageFont, ImageDraw, Image

# 이미지에 한글 텍스트를 표시하는 함수 정의
def put_text_on_image(img, text, position, font_size=30, font_color=(0, 255, 0)):
    # OpenCV 이미지를 PIL 이미지로 변환
    img_pil = Image.fromarray(img)
    # 그리기 객체 생성
    draw = ImageDraw.Draw(img_pil)
    try:
        # 맑은 고딕 폰트 로드 시도 (한글 지원)
        font = ImageFont.truetype("malgun.ttf", font_size)
    except:
        # 폰트 로드 실패 시 기본 폰트 사용
        font = ImageFont.load_default()
    # PIL에서는 RGB 순서가 아닌 BGR 순서로 색상 지정 필요 (OpenCV와 반대)
    draw.text(position, text, font=font, fill=font_color[::-1])
    # PIL 이미지를 다시 NumPy 배열로 변환하여 반환
    return np.array(img_pil)

# 두 바운딩 박스 간의 IoU(Intersection over Union) 계산 함수
def calculate_iou(box1, box2):
    # 두 박스의 겹치는 영역의 좌상단 좌표 계산
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    # 두 박스의 겹치는 영역의 우하단 좌표 계산
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    # 겹치는 영역이 없는 경우 IoU는 0
    if x2 < x1 or y2 < y1:
        return 0.0
    # 겹치는 영역의 면적 계산
    inter_area = (x2 - x1) * (y2 - y1)
    # 첫 번째 박스의 면적 계산
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    # 두 번째 박스의 면적 계산
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    # 합집합 면적 계산 (두 면적의 합 - 겹치는 부분)
    union = area1 + area2 - inter_area
    # IoU 계산 및 반환 (교집합 / 합집합)
    return inter_area / union

# === 설정 부분 시작 ===
# 분석할 비디오 파일 경로 지정
video_path = r"C:\workspace\list\0_8_IPC1_20230108162038.mp4"
# YOLO 모델 파일 경로 지정
model_path = r"C:\workspace\list\best.pt"
# 영상을 몇 개의 격자로 나눌지 설정 (3x3 그리드)
grid_size = 3
# 처리할 특정 셀 위치 지정 (2행, 0열) - 좌상단이 (0,0)
selected_cell = (2, 0)  # (행, 열)
# 출력 화면 크기 조정 비율
scale_factor = 1.0
# 객체 탐지 신뢰도 임계값 (0.5 이상인 객체만 탐지)
conf_threshold = 0.5

# === 초기화 부분 시작 ===
# 비디오 파일 존재 여부 확인
if not os.path.exists(video_path):
    # 파일이 없으면 에러 메시지 출력 후 종료
    print(f"❌ 파일이 존재하지 않습니다: {video_path}")
    exit()

# YOLO 모델 로드
model = YOLO(model_path)
# 비디오 파일 열기
cap = cv2.VideoCapture(video_path)

# 비디오 프레임 너비 가져오기
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# 비디오 프레임 높이 가져오기
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# 비디오 초당 프레임 수(FPS) 가져오기
fps = cap.get(cv2.CAP_PROP_FPS)
# 각 프레임 간 지연 시간 계산 (밀리초 단위)
delay = int(1000 / fps)

# 이전 프레임에서 탐지한 객체들의 정보를 저장할 리스트 초기화
# 형식: [((x1, y1, x2, y2), obj_id), ...] - 바운딩 박스 좌표와 객체 ID
# 다음 프레임에서 ID가 없어진 객체와 비교할 때 사용 (IOU 기반 ID 복원)
last_detections = []

# === 메인 처리 루프 시작 ===
while True:
    # 비디오에서 다음 프레임 읽기
    # ret: 프레임 읽기 성공 여부 (True/False)
    # frame: 읽은 프레임 이미지 데이터
    ret, frame = cap.read()
    # 더 이상 읽을 프레임이 없으면 루프 종료
    if not ret:
        break
    
    # 선택된 셀의 행과 열 값 가져오기
    row, col = selected_cell
    # 그리드 하나의 높이 계산 (전체 높이 / 그리드 크기)
    cell_h = height // grid_size
    # 그리드 하나의 너비 계산 (전체 너비 / 그리드 크기)
    cell_w = width // grid_size
    # 선택된 셀의 좌상단 좌표 계산
    x, y = col * cell_w, row * cell_h
    # 전체 프레임에서 선택된 셀 부분만 추출
    cell_frame = frame[y:y+cell_h, x:x+cell_w]

    # YOLO 모델을 사용하여 객체 탐지 및 추적 수행
    # conf: 신뢰도 임계값 - 이 값 이상의 신뢰도를 가진 객체만 탐지
    # persist: 객체가 일시적으로 가려져도 ID 유지
    # verbose: 상세 출력 여부
    # [0]: 결과의 첫 번째 항목 가져오기 (배치 처리 시 여러 결과가 반환될 수 있음)
    results = model.track(cell_frame, conf=conf_threshold, persist=True, verbose=False)[0]
    
    # 탐지된 객체들의 바운딩 박스 좌표 추출 (x1, y1, x2, y2 형식)
    # cpu(): GPU 텐서를 CPU로 이동
    # tolist(): 텐서를 파이썬 리스트로 변환
    boxes = results.boxes.xyxy.cpu().tolist()
    
    # 각 객체의 고유 ID 추출
    # ID가 있으면 해당 ID를 가져오고, 없으면 -1로 채움
    ids = results.boxes.id.int().cpu().tolist() if results.boxes.id is not None else [-1] * len(boxes)

    # 현재 프레임에서 탐지된 객체들의 정보를 저장할 리스트 초기화
    # 현재 프레임 시각화 및 다음 프레임에서의 비교를 위해 사용
    new_detections = []
    
    # 탐지된 각 객체에 대해 처리
    for i, box in enumerate(boxes):
        # 바운딩 박스 좌표 추출
        x1, y1, x2, y2 = box
        # 객체 ID 가져오기
        obj_id = ids[i]

        # ID가 없는 경우(-1), 이전 프레임의 객체들과 비교하여 ID 복원 시도
        if obj_id == -1:  # ID가 없는 경우(객체가 잠시 사라졌다가 다시 나타난 경우)
            # 이전 프레임의 모든 객체와 IOU 계산
            for last_box, last_id in last_detections:
                # 현재 박스와 이전 박스의 IOU 계산
                iou = calculate_iou(box, last_box)
                # IOU가 0.5 이상이면 같은 객체로 간주하고 이전 ID 사용
                if iou > 0.5:
                    obj_id = last_id
                    break
        
        # 현재 객체의 정보(박스 좌표와 ID)를 새 리스트에 추가
        new_detections.append(((x1, y1, x2, y2), obj_id))

    # === 결과 시각화 부분 시작 ===
    # 원본 프레임 복사하여 시각화용 이미지 생성
    plotted = cell_frame.copy()
    
    # 탐지된 각 객체에 대해 시각화 작업 수행
    for (x1, y1, x2, y2), obj_id in new_detections:
        # 바운딩 박스 그리기 (녹색, 두께 2)
        cv2.rectangle(plotted, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # 객체 ID 텍스트 표시 (바운딩 박스 위에 분홍색으로 표시)
        cv2.putText(plotted, f"ID {obj_id}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    # 필요시 출력 이미지 크기 조정
    if scale_factor != 1.0:
        # 현재 이미지 크기 가져오기
        h, w = plotted.shape[:2]
        # 새 크기 계산
        new_size = (int(w * scale_factor), int(h * scale_factor))
        # 이미지 크기 조정
        plotted = cv2.resize(plotted, new_size)

    # 탐지된 닭의 수 계산
    chicken_count = len(new_detections)
    # 한글 텍스트로 셀 위치와 닭 수 표시
    plotted = put_text_on_image(plotted, f"(2,0) 셀 - 닭 {chicken_count}마리", (10, 30), 25)

    # 결과 화면 표시
    cv2.imshow("Selected Cell Detection with ID Recovery", plotted)

    # 현재 프레임의 탐지 정보를 이전 탐지 정보로 저장
    # 다음 프레임에서 IOU 기반 비교를 위해 사용
    last_detections = [((x1, y1, x2, y2), obj_id) for (x1, y1, x2, y2), obj_id in new_detections]

    # 키 입력 확인 (q 키를 누르면 프로그램 종료)
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# 비디오 캡처 객체 해제
cap.release()
# 모든 OpenCV 창 닫기
cv2.destroyAllWindows()
