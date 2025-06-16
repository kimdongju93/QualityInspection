import cv2
import numpy as np
from ultralytics import YOLO
from playsound import playsound
import time
import os
from dotenv import load_dotenv
import torch
import json
from datetime import datetime
import tkinter as tk
from tkinter import filedialog

# torch.load monkey patch (safe)
import builtins
_builtin_torch_load = torch.load

def patched_torch_load(f, *args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _builtin_torch_load(f, *args, **kwargs)

torch.load = patched_torch_load

def select_video_file():
    """MP4 파일 선택 대화상자"""
    root = tk.Tk()
    root.withdraw()  # 기본 창 숨기기
    file_path = filedialog.askopenfilename(
        title="분석할 MP4 파일을 선택하세요",
        filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
    )
    return file_path if file_path else None

class WorkerDetectionSystem:
    def __init__(self, camera_url=0):
        # YOLO 모델 로드 (small 모델 사용)
        self.model = YOLO('yolov8s.pt')
        
        # 카메라 설정
        self.cap = cv2.VideoCapture(camera_url)
        if not self.cap.isOpened():
            raise ValueError("카메라를 열 수 없습니다.")
        
        # ROI (관심 영역) 설정
        self.roi_points = []
        self.roi_set = False
        self.roi_file = "roi_config.json"
        
        # 저장된 ROI가 있다면 불러오기
        self.load_roi()
        
        # 알람 설정
        self.alarm_sound = "alarm.wav"
        self.last_alarm_time = 0
        self.alarm_cooldown = 5  # 알람 간격 (초)
        
        # 프레임 크기 설정
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 처리할 프레임 크기 설정 (더 큰 해상도로 처리)
        self.process_width = 1280
        self.process_height = 720

        # 스냅샷 관련 설정
        self.snapshots_dir = "snapshots"
        self.snapshots = []  # (timestamp, image_path) 튜플 리스트
        self.max_snapshots = 10  # 표시할 최대 스냅샷 수
        self.snapshot_size = (160, 120)  # 스냅샷 표시 크기
        self.last_snapshot_time = 0  # 마지막 스냅샷 촬영 시간
        self.snapshot_interval = 3  # 스냅샷 촬영 간격 (초)
        
        # 스냅샷 디렉토리 생성
        if not os.path.exists(self.snapshots_dir):
            os.makedirs(self.snapshots_dir)

    def save_roi(self):
        """ROI 좌표를 JSON 파일로 저장"""
        if self.roi_set:
            roi_data = {
                "points": self.roi_points.tolist(),
                "frame_width": self.frame_width,
                "frame_height": self.frame_height
            }
            with open(self.roi_file, 'w') as f:
                json.dump(roi_data, f)
            print(f"ROI 설정이 {self.roi_file}에 저장되었습니다.")

    def load_roi(self):
        """저장된 ROI 좌표 불러오기"""
        try:
            if os.path.exists(self.roi_file):
                with open(self.roi_file, 'r') as f:
                    roi_data = json.load(f)
                
                # 저장된 프레임 크기와 현재 프레임 크기가 같은지 확인
                if (roi_data["frame_width"] == self.frame_width and 
                    roi_data["frame_height"] == self.frame_height):
                    self.roi_points = np.array(roi_data["points"], np.int32)
                    self.roi_set = True
                    print(f"저장된 ROI 설정을 불러왔습니다.")
                else:
                    print("저장된 ROI 설정의 프레임 크기가 현재 영상과 다릅니다.")
        except Exception as e:
            print(f"ROI 설정을 불러오는 중 오류 발생: {e}")

    def set_roi(self, points):
        """관심 영역(ROI) 설정"""
        self.roi_points = np.array(points, np.int32)
        self.roi_set = True
        # ROI 설정 후 자동 저장
        self.save_roi()

    def is_point_in_roi(self, point):
        """점이 ROI 내부에 있는지 확인"""
        if not self.roi_set:
            return False
        return cv2.pointPolygonTest(self.roi_points, point, False) >= 0

    def play_alarm(self):
        """알람 소리 재생"""
        current_time = time.time()
        if current_time - self.last_alarm_time >= self.alarm_cooldown:
            try:
                playsound(self.alarm_sound)
                self.last_alarm_time = current_time
            except Exception as e:
                print(f"알람 재생 중 오류 발생: {e}")

    def save_snapshot(self, frame):
        """위험구역 진입 시 스냅샷 저장"""
        current_time = time.time()
        
        # 마지막 스냅샷으로부터 3초가 지났는지 확인
        if current_time - self.last_snapshot_time >= self.snapshot_interval:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snapshot_{timestamp}.jpg"
            filepath = os.path.join(self.snapshots_dir, filename)
            
            # 스냅샷 저장
            cv2.imwrite(filepath, frame)
            
            # 스냅샷 리스트에 추가
            self.snapshots.append((timestamp, filepath))
            
            # 최대 개수 유지
            while len(self.snapshots) > self.max_snapshots:
                # 가장 오래된 스냅샷 파일 삭제
                old_timestamp, old_filepath = self.snapshots.pop(0)
                if os.path.exists(old_filepath):
                    os.remove(old_filepath)
            
            # 마지막 스냅샷 시간 업데이트
            self.last_snapshot_time = current_time

    def create_snapshot_display(self, frame):
        """스냅샷 리스트를 포함한 디스플레이 생성"""
        # 원본 프레임 복사
        display = frame.copy()
        
        # 왼쪽에 스냅샷 리스트를 표시할 영역 생성
        list_width = 180
        list_height = self.frame_height
        
        # 스냅샷 리스트 배경
        cv2.rectangle(display, (0, 0), (list_width, list_height), (0, 0, 0), -1)
        
        # 스냅샷 표시
        for i, (timestamp, filepath) in enumerate(self.snapshots):
            if os.path.exists(filepath):
                # 스냅샷 이미지 로드 및 크기 조정
                snapshot = cv2.imread(filepath)
                snapshot = cv2.resize(snapshot, self.snapshot_size)
                
                # 스냅샷 위치 계산
                y_pos = 10 + i * (self.snapshot_size[1] + 10)
                
                # 스냅샷 표시
                display[y_pos:y_pos + self.snapshot_size[1], 
                       10:10 + self.snapshot_size[0]] = snapshot
                
                # 타임스탬프 표시
                cv2.putText(display, timestamp, 
                          (10, y_pos + self.snapshot_size[1] + 15),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return display

    def process_frame(self, frame):
        """프레임 처리 및 작업자 감지"""
        # 프레임 크기 조정 (더 큰 해상도로 처리)
        process_frame = cv2.resize(frame, (self.process_width, self.process_height))
        
        # YOLO로 객체 감지
        results = self.model(process_frame, conf=0.3)  # 신뢰도 임계값 낮춤
        
        # ROI 표시
        if self.roi_set:
            # ROI 좌표를 처리 크기에 맞게 조정
            scale_x = self.process_width / self.frame_width
            scale_y = self.process_height / self.frame_height
            scaled_roi = self.roi_points * np.array([scale_x, scale_y])
            cv2.polylines(process_frame, [scaled_roi.astype(np.int32)], True, (0, 255, 0), 2)
        
        # 감지된 객체 처리
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 사람 클래스만 감지 (COCO 데이터셋에서 사람 클래스는 0)
                if int(box.cls) == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    
                    # 신뢰도가 0.3 이상인 경우만 처리 (임계값 낮춤)
                    if confidence > 0.3:
                        # 바운딩 박스 그리기
                        cv2.rectangle(process_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # 사람의 중심점 계산
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        
                        # ROI 내부에 있는지 확인
                        if self.is_point_in_roi((center_x / scale_x, center_y / scale_y)):
                            cv2.putText(process_frame, f"WARNING: Worker in restricted area! ({confidence:.2f})", 
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            self.play_alarm()
                            # 스냅샷 저장
                            self.save_snapshot(frame)  # 원본 프레임으로 스냅샷 저장
        
        # 처리된 프레임을 원래 크기로 조정
        display_frame = cv2.resize(process_frame, (self.frame_width, self.frame_height))
        
        # 스냅샷 리스트가 포함된 디스플레이 생성
        display = self.create_snapshot_display(display_frame)
        
        return display

    def run(self):
        """메인 실행 루프"""
        print("시스템을 시작합니다.")
        print("'q'를 눌러 종료하세요.")
        print("'r'를 눌러 ROI를 새로 설정하세요.")
        print("'s'를 눌러 현재 ROI 설정을 저장하세요.")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 프레임 처리
            processed_frame = self.process_frame(frame)
            
            # 결과 표시
            cv2.imshow('Worker Detection System', processed_frame)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # ROI 설정 모드
                print("ROI를 설정하세요. 4개의 점을 클릭하세요.")
                points = []
                
                def mouse_callback(event, x, y, flags, param):
                    if event == cv2.EVENT_LBUTTONDOWN:
                        points.append((x, y))
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                        cv2.imshow('Worker Detection System', frame)
                
                cv2.setMouseCallback('Worker Detection System', mouse_callback)
                
                while len(points) < 4:
                    cv2.waitKey(1)
                
                self.set_roi(points)
                cv2.setMouseCallback('Worker Detection System', lambda *args: None)
            elif key == ord('s'):
                # ROI 설정 저장
                if self.roi_set:
                    self.save_roi()
                else:
                    print("저장할 ROI 설정이 없습니다.")
        
        # 정리
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # 환경 변수에서 카메라 URL 로드 (선택사항)
    load_dotenv()
    
    # MP4 파일 선택
    video_file = select_video_file()
    if video_file is None:
        print("파일이 선택되지 않았습니다. 프로그램을 종료합니다.")
        exit()
    
    try:
        system = WorkerDetectionSystem(video_file)
        system.run()
    except Exception as e:
        print(f"오류 발생: {e}") 