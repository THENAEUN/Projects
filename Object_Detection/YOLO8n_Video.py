import cv2
from ultralytics import YOLO  # YOLO 모델을 불러옵니다.

# YOLO 모델 로드 (yolov8n.pt사용)
model = YOLO("yolov8n.pt")

# 웹캠 신호 가져오기 (0번 디바이스 열기기)
VideoSignal = cv2.VideoCapture(0)

# 웹캠이 정상적으로 열렸는지 확인
if not VideoSignal.isOpened():
    print("웹캠을 열 수 없습니다.")
else:
    print("웹캠이 정상적으로 열렸습니다.")

# 웹캠 영상 출력
while True:
    # 프레임 읽기 (ret : 프레임을 읽었는지 여부부)
    ret, frame = VideoSignal.read()  
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # YOLO 모델을 사용하여 객체 인식 (프레임을 모델에 전달)
    results = model(frame)

    # 인식된 객체를 바운딩 박스로 표시 - YOLO 모델에서 바운딩 박스를 그려주는 함수
    frame = results[0].plot() 

    # 화면에 출력
    cv2.imshow("Webcam - YOLO", frame)

    # 'q' 키를 누르면 종료 (waitKey : 1ms동안 키 입력 대기 (프레임 처리 속도가 느리면 대기 시간을 늘려서 완화화))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 해제 및 창 닫기
VideoSignal.release()
cv2.destroyAllWindows()





