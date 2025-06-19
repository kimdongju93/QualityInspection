import cv2
import numpy as np
import os
import json
from datetime import datetime
import time
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

class VideoAnalysisSystem:
    def __init__(self, video_path):
        self.video_path = video_path
        
        # YOLOv8 모델 초기화 (더 큰 모델 사용)
        try:
            self.model = YOLO('yolov8m.pt')  # YOLOv8 medium 모델 사용
            print("YOLOv8m 모델 로드 성공")
        except Exception as e:
            print(f"YOLOv8m 모델 로드 실패: {e}")
            # fallback으로 YOLOv8n 사용
            try:
                self.model = YOLO('yolov8n.pt')
                print("YOLOv8n 모델로 대체")
            except:
                print("모델 로드 실패")
                return
        
        self.roi_points = []
        self.roi_set = False
        self.detection_mode = False  # ROI 설정 모드/감지 모드 구분
        
        # 분석 결과 저장
        self.detection_results = []
        self.frame_count = 0
        
        # 스냅샷 관련 설정
        self.snapshots_dir = 'snapshots'
        self.max_snapshots = 10
        self.snapshot_interval = 3.0  # 같은 객체에 대해 3초마다 한 번씩
        self.last_snapshot_times = {}  # 객체별 마지막 스냅샷 시간
        self.snapshot_files = []  # 스냅샷 파일 목록
        
        # 스냅샷 확대 관련 설정
        self.enlarged_snapshot = None  # 확대된 스냅샷
        self.enlarged_snapshot_index = -1  # 확대된 스냅샷 인덱스
        
        # 통합 창 레이아웃 설정
        self.snapshot_panel_width = 250  # 스냅샷 패널의 고정 너비
        self.combined_window_height = 720  # 통합 창의 높이 (영상이 이 높이로 리사이즈됨)
        
        # 탐지 설정 (속도 최적화)
        self.conf_threshold = 0.2  # 신뢰도 임계값
        self.iou_threshold = 0.3   # IoU 임계값 (NMS)
        self.input_size = 416      # 입력 이미지 크기 (640에서 416으로 줄임 - 속도 향상)
        
        # 한글 폰트 설정
        self.setup_korean_font()
        
        # 스냅샷 디렉토리 생성
        if not os.path.exists(self.snapshots_dir):
            os.makedirs(self.snapshots_dir)
        
        # 기존 스냅샷 파일들 로드
        self.load_existing_snapshots()
        
    def setup_korean_font(self):
        """한글 폰트 설정"""
        try:
            # Windows 시스템 폰트 경로들
            font_paths = [
                "C:/Windows/Fonts/malgun.ttf",  # 맑은 고딕
                "C:/Windows/Fonts/gulim.ttc",   # 굴림
                "C:/Windows/Fonts/batang.ttc",  # 바탕
                "C:/Windows/Fonts/dotum.ttc",   # 돋움
            ]
            
            self.font = None
            for font_path in font_paths:
                if os.path.exists(font_path):
                    self.font = ImageFont.truetype(font_path, 20)
                    print(f"한글 폰트 로드 성공: {font_path}")
                    break
            
            if self.font is None:
                print("한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
                self.font = ImageFont.load_default()
                
        except Exception as e:
            print(f"폰트 설정 실패: {e}")
            self.font = ImageFont.load_default()

    def put_korean_text(self, img, text, position, font_size=20, color=(255, 255, 255)):
        """한글 텍스트를 이미지에 추가"""
        try:
            # PIL 이미지로 변환
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            
            # 폰트 크기 조정
            if font_size != 20:
                try:
                    font_paths = [
                        "C:/Windows/Fonts/malgun.ttf",
                        "C:/Windows/Fonts/gulim.ttc",
                        "C:/Windows/Fonts/batang.ttc",
                        "C:/Windows/Fonts/dotum.ttc",
                    ]
                    
                    temp_font = None
                    for font_path in font_paths:
                        if os.path.exists(font_path):
                            temp_font = ImageFont.truetype(font_path, font_size)
                            break
                    
                    if temp_font:
                        draw.text(position, text, font=temp_font, fill=color)
                    else:
                        draw.text(position, text, font=self.font, fill=color)
                except:
                    draw.text(position, text, font=self.font, fill=color)
            else:
                draw.text(position, text, font=self.font, fill=color)
            
            # OpenCV 이미지로 변환
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            return img_cv
            
        except Exception as e:
            print(f"한글 텍스트 추가 실패: {e}")
            # 실패 시 기본 OpenCV 텍스트 사용
            cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            return img

    def load_roi_config(self):
        """저장된 ROI 설정을 로드"""
        try:
            if os.path.exists('roi_config.json'):
                with open('roi_config.json', 'r') as f:
                    config = json.load(f)
                    self.roi_points = config.get('roi_points', [])
                    self.roi_set = len(self.roi_points) == 4
                    print(f"ROI 설정을 로드했습니다: {self.roi_points}")
        except Exception as e:
            print(f"ROI 설정 로드 실패: {e}")
    
    def set_roi_from_video(self, frame):
        """비디오에서 ROI 설정"""
        print("ROI를 설정해주세요. 4개의 점을 클릭하세요.")
        print("완료되면 'Enter' 키를 누르세요.")
        
        roi_points = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(roi_points) < 4:
                    roi_points.append([x, y])
                    print(f"ROI 포인트 {len(roi_points)} 추가: ({x}, {y})")
        
        cv2.namedWindow('ROI Setup')
        cv2.setMouseCallback('ROI Setup', mouse_callback)
        
        while len(roi_points) < 4:
            display_frame = frame.copy()
            
            # 기존 포인트 표시
            for i, point in enumerate(roi_points):
                cv2.circle(display_frame, tuple(point), 5, (0, 255, 0), -1)
                cv2.putText(display_frame, str(i+1), (point[0]+10, point[1]-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imshow('ROI Setup', display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                cv2.destroyAllWindows()
                return False
        
        self.roi_points = roi_points
        self.roi_set = True
        
        # ROI 설정 저장
        config = {'roi_points': self.roi_points}
        with open('roi_config.json', 'w') as f:
            json.dump(config, f)
        
        cv2.destroyAllWindows()
        return True
    
    def point_in_roi(self, x, y):
        """점이 ROI 내부에 있는지 확인"""
        if len(self.roi_points) != 4:
            return False
        
        # Ray casting 알고리즘 사용
        n = len(self.roi_points)
        inside = False
        
        p1x, p1y = self.roi_points[0]
        for i in range(n + 1):
            p2x, p2y = self.roi_points[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def analyze_frame(self, frame, timestamp):
        """단일 프레임 분석 (YOLOv8 사용)"""
        try:
            # YOLOv8 모델로 객체 감지 (속도 최적화)
            results = self.model(frame, 
                               conf=self.conf_threshold,  # 신뢰도 임계값
                               iou=self.iou_threshold,    # IoU 임계값
                               imgsz=self.input_size,    # 입력 이미지 크기
                               verbose=False,             # 출력 줄이기
                               half=True)                 # FP16 추론으로 속도 향상
            
            detections = []
            workers_in_roi = 0
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # 사람 클래스만 감지 (class 0)
                        if int(box.cls) == 0:  # person class
                            confidence = float(box.conf)
                            if confidence > self.conf_threshold:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                
                                # 바운딩 박스 그리기
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                
                                # 신뢰도 표시
                                cv2.putText(frame, f'Person: {confidence:.2f}', 
                                          (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                
                                # ROI 내부인지 확인
                                in_roi = False
                                if self.roi_set and len(self.roi_points) == 4:
                                    # 객체의 중심점이 ROI 내부인지 확인
                                    center_x = (x1 + x2) // 2
                                    center_y = (y1 + y2) // 2
                                    in_roi = self.point_in_roi(center_x, center_y)
                                    
                                    if in_roi:
                                        workers_in_roi += 1
                                        # ROI 내부 객체는 빨간색으로 표시
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                        cv2.putText(frame, 'IN ROI', (x1, y1-30), 
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                        
                                        # 스냅샷 촬영 (같은 객체에 대해 3초마다)
                                        print(f"[DEBUG] ROI 내 작업자 탐지: ({center_x}, {center_y})")
                                        self.take_snapshot_if_needed(frame, center_x, center_y, timestamp)
                                else:
                                    # ROI가 설정되지 않은 경우에도 스냅샷 촬영 (디버깅용)
                                    center_x = (x1 + x2) // 2
                                    center_y = (y1 + y2) // 2
                                    print(f"[DEBUG] ROI 미설정 상태에서 작업자 탐지: ({center_x}, {center_y})")
                                    self.take_snapshot_if_needed(frame, center_x, center_y, timestamp)
                                
                                detections.append({
                                    'bbox': [x1, y1, x2, y2],
                                    'confidence': confidence,
                                    'in_roi': in_roi,
                                    'center': [center_x, center_y]
                                })
            
            # ROI 그리기
            if self.roi_set and len(self.roi_points) == 4:
                roi_points = np.array(self.roi_points, np.int32)
                cv2.polylines(frame, [roi_points], True, (255, 0, 0), 2)
                cv2.putText(frame, 'ROI', (self.roi_points[0][0], self.roi_points[0][1]-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            return frame, detections
            
        except Exception as e:
            print(f"YOLOv8 분석 중 오류: {e}")
            # fallback으로 빈 결과 반환
            return frame, []

    def take_snapshot_if_needed(self, frame, center_x, center_y, timestamp):
        """같은 객체에 대해 3초마다 스냅샷 촬영"""
        # 객체의 위치를 기반으로 고유 ID 생성 (간단한 그리드 기반)
        grid_size = 50  # 50x50 픽셀 그리드
        grid_x = center_x // grid_size
        grid_y = center_y // grid_size
        object_id = f"{grid_x}_{grid_y}"
        
        # 마지막 스냅샷 시간 확인
        if object_id in self.last_snapshot_times:
            time_diff = timestamp - self.last_snapshot_times[object_id]
            if time_diff < self.snapshot_interval:
                print(f"[DEBUG] 스냅샷 스킵: {object_id}, 시간차: {time_diff:.1f}초")
                return  # 3초가 지나지 않았으면 스냅샷 촬영하지 않음
        
        # 스냅샷 촬영
        timestamp_str = f"{timestamp:.1f}".replace('.', '_')
        filename = f"snapshot_{timestamp_str}_{object_id}.jpg"
        filepath = os.path.join(self.snapshots_dir, filename)

        # ROI(위험구역) 표시 후 저장
        snapshot_frame = frame.copy()
        if self.roi_set and len(self.roi_points) == 4:
            roi_points = np.array(self.roi_points, np.int32)
            cv2.polylines(snapshot_frame, [roi_points], True, (255, 0, 0), 3)
            cv2.putText(snapshot_frame, '위험구역', (self.roi_points[0][0], self.roi_points[0][1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 3)
        try:
            cv2.imwrite(filepath, snapshot_frame)
            self.last_snapshot_times[object_id] = timestamp
            
            # 스냅샷 파일 목록 업데이트 (최신 10개만 유지)
            self.update_snapshot_files()
            
            print(f"스냅샷 촬영: {filename}")
        except Exception as e:
            print(f"스냅샷 촬영 실패: {e}")

    def update_snapshot_files(self):
        """스냅샷 파일 목록 업데이트 (최신 10개만 표시)"""
        if os.path.exists(self.snapshots_dir):
            files = [f for f in os.listdir(self.snapshots_dir) if f.endswith('.jpg')]
            files.sort(key=lambda x: os.path.getctime(os.path.join(self.snapshots_dir, x)))
            # 파일은 삭제하지 않고, 표시할 파일만 최신 10개로 제한
            self.snapshot_files = files[-self.max_snapshots:] if len(files) > self.max_snapshots else files
            print(f"[DEBUG] 스냅샷 파일 목록 업데이트: {len(files)}개 중 {len(self.snapshot_files)}개 표시")

    def analyze_video(self):
        """비디오 전체 분석"""
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print(f"비디오 파일을 열 수 없습니다: {self.video_path}")
            return
        
        # 비디오 정보
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        
        print(f"비디오 정보:")
        print(f"- 총 프레임: {total_frames}")
        print(f"- FPS: {fps:.2f}")
        print(f"- 길이: {duration:.2f}초")
        
        # ROI 설정 로드
        self.load_roi_config()
        
        # 윈도우 설정
        cv2.namedWindow('Video Analysis with Snapshots')
        cv2.setMouseCallback('Video Analysis with Snapshots', self.combined_mouse_callback)
        
        print("=== 비디오 분석 시스템 ===")
        print("'r': ROI 설정 모드")
        print("'q': 프로그램 종료")
        print("스냅샷 패널 클릭: 확대/축소")
        
        if not self.roi_set:
            print("ROI를 설정해주세요. 4개의 점을 클릭하세요.")
        
        frame_count = 0
        workers_in_roi_count = 0
        frame_skip = 2  # 프레임 스킵 (2면 1프레임씩 건너뛰기 - 속도 향상)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break
            
            frame_count += 1
            
            # 프레임 스킵으로 속도 향상
            if frame_count % frame_skip != 0:
                continue
            
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"진행률: {progress:.1f}% ({frame_count}/{total_frames})")
            
            # 프레임 분석
            processed_frame, detections = self.analyze_frame(frame, frame_count / fps)
            
            # ROI 내 작업자 수 계산
            workers_in_roi = sum(1 for det in detections if det['in_roi'])
            if workers_in_roi > 0:
                workers_in_roi_count += 1
            
            # 결과 저장
            self.detection_results.append({
                'frame': frame_count,
                'timestamp': frame_count / fps,
                'detections': detections,
                'workers_in_roi': workers_in_roi
            })
            
            # 통합 프레임 생성 및 표시
            combined_frame = self.create_combined_frame(processed_frame, frame_count, total_frames, workers_in_roi)
            cv2.imshow('Video Analysis with Snapshots', combined_frame)
            
            # 실시간 분석을 위한 대기 시간 (1ms로 설정하여 최대한 빠르게)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                cv2.waitKey(0)
            elif key == ord('r'):
                self.roi_points = []
                self.roi_set = False
                self.detection_mode = False
                print("[DEBUG] detection_mode -> False (ROI 재설정)")
                print("ROI 설정 모드로 전환됩니다. 4개의 점을 클릭하세요.")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # 분석 결과 출력
        self.print_analysis_results()
        
        # 결과를 JSON 파일로 저장
        self.save_analysis_results()
    
    def print_analysis_results(self):
        """분석 결과 출력"""
        print("\n=== 비디오 분석 결과 ===")
        
        total_frames = len(self.detection_results)
        frames_with_workers = sum(1 for result in self.detection_results if result['workers_in_roi'] > 0)
        
        print(f"총 프레임 수: {total_frames}")
        print(f"작업자가 감지된 프레임: {frames_with_workers}")
        print(f"작업자 감지 비율: {(frames_with_workers/total_frames)*100:.2f}%")
        
        # 최대 동시 작업자 수
        max_workers = max(result['workers_in_roi'] for result in self.detection_results)
        print(f"최대 동시 작업자 수: {max_workers}")
        
        # 작업자가 감지된 구간들
        worker_segments = []
        start_frame = None
        
        for i, result in enumerate(self.detection_results):
            if result['workers_in_roi'] > 0 and start_frame is None:
                start_frame = i
            elif result['workers_in_roi'] == 0 and start_frame is not None:
                worker_segments.append((start_frame, i-1))
                start_frame = None
        
        if start_frame is not None:
            worker_segments.append((start_frame, len(self.detection_results)-1))
        
        print(f"작업자 감지 구간 수: {len(worker_segments)}")
        for i, (start, end) in enumerate(worker_segments):
            start_time = start / 30  # FPS가 30이라고 가정
            end_time = end / 30
            print(f"  구간 {i+1}: {start_time:.1f}초 ~ {end_time:.1f}초")
    
    def save_analysis_results(self):
        """분석 결과를 JSON 파일로 저장"""
        output_file = 'video_analysis_results.json'
        
        results = {
            'video_path': self.video_path,
            'roi_points': self.roi_points,
            'analysis_results': self.detection_results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n분석 결과가 {output_file}에 저장되었습니다.")

    def load_existing_snapshots(self):
        """기존 스냅샷 파일들을 로드"""
        print(f"[DEBUG] Loading existing snapshots from: {self.snapshots_dir}")
        if os.path.exists(self.snapshots_dir):
            files = [f for f in os.listdir(self.snapshots_dir) if f.endswith('.jpg')]
            files.sort(key=lambda x: os.path.getctime(os.path.join(self.snapshots_dir, x)))
            self.snapshot_files = files[-self.max_snapshots:]  # 최신 10개만 유지
            print(f"[DEBUG] Found {len(files)} snapshot files, loaded {len(self.snapshot_files)}")
            for i, filename in enumerate(self.snapshot_files):
                print(f"[DEBUG] Snapshot {i+1}: {filename}")
        else:
            print(f"[DEBUG] Snapshots directory does not exist: {self.snapshots_dir}")
            self.snapshot_files = []

    def mouse_callback(self, event, x, y, flags, param):
        """마우스 클릭 이벤트 처리 (더 이상 사용되지 않음 - combined_mouse_callback 사용)"""
        pass

    def draw_snapshots_panel_content(self, panel_height):
        """스냅샷 썸네일 패널의 내용만 그리는 메서드"""
        panel = np.zeros((panel_height, self.snapshot_panel_width, 3), dtype=np.uint8)
        panel[:] = (50, 50, 50)  # 배경색 조정 (파란색 대신)
        
        # 패널 테두리와 제목
        cv2.rectangle(panel, (0, 0), (self.snapshot_panel_width-1, panel_height-1), (255, 255, 255), 3)
        
        # 한글 텍스트 추가
        panel = self.put_korean_text(panel, "스냅샷", (int(self.snapshot_panel_width / 2) - 40, 10), 24, (255, 255, 255))
        panel = self.put_korean_text(panel, "(클릭하면 확대)", (10, 35), 16, (200, 200, 200))
        
        # 스냅샷 개수 표시
        panel = self.put_korean_text(panel, f"총: {len(self.snapshot_files)}개", (10, panel_height - 30), 18, (255, 255, 255))
        
        # 스냅샷 썸네일 표시
        y_offset = 80
        thumbnail_height = 80  # 썸네일 높이
        
        for i, filename in enumerate(self.snapshot_files[-10:]):  # 최신 10개만
            if y_offset + thumbnail_height > panel_height - 50:  # 하단 여백 확보
                break
                
            filepath = os.path.join(self.snapshots_dir, filename)
            if os.path.exists(filepath):
                snapshot = cv2.imread(filepath)
                if snapshot is not None:
                    h, w = snapshot.shape[:2]
                    aspect_ratio = w / h
                    thumb_w = self.snapshot_panel_width - 20
                    thumb_h = int(thumb_w / aspect_ratio)
                    
                    if thumb_h > thumbnail_height:
                        thumb_h = thumbnail_height
                        thumb_w = int(thumb_h * aspect_ratio)
                    
                    thumbnail = cv2.resize(snapshot, (thumb_w, thumb_h))
                    x_start = 10
                    y_start = y_offset
                    
                    border_color = (0, 255, 255) if i == self.enlarged_snapshot_index else (0, 255, 0)
                    
                    panel[y_start:y_start+thumb_h, x_start:x_start+thumb_w] = thumbnail
                    cv2.rectangle(panel, (x_start, y_start), (x_start+thumb_w, y_start+thumb_h), border_color, 3)
                    
                    # 파일명 표시 (한글 텍스트 사용)
                    panel = self.put_korean_text(panel, f"{i+1}. {filename[-15:]}", (x_start, y_start+thumb_h+5), 14, (255, 255, 255))
                    
                    y_offset += thumbnail_height + 25
            
        return panel

    def create_combined_frame(self, video_frame, frame_count, total_frames, workers_in_roi):
        """영상 분석 화면과 스냅샷 리스트를 하나의 프레임으로 합성"""
        
        # 영상 프레임 크기를 통합 창 높이에 맞춰 리사이즈
        video_h, video_w = video_frame.shape[:2]
        aspect_ratio = video_w / video_h
        
        new_video_h = self.combined_window_height
        new_video_w = int(new_video_h * aspect_ratio)
        resized_video = cv2.resize(video_frame, (new_video_w, new_video_h))
        
        # 스냅샷 패널 이미지 가져오기
        snapshot_panel_image = self.draw_snapshots_panel_content(self.combined_window_height)
        
        # 통합 프레임 크기
        combined_width = self.snapshot_panel_width + new_video_w + 3 # 구분선 너비 추가
        combined_height = self.combined_window_height
        
        # 첫 번째 프레임에서만 디버깅 정보 출력
        if frame_count == 1:
            print(f"[DEBUG] Original video size: {video_w}x{video_h}")
            print(f"[DEBUG] Resized video size: {new_video_w}x{new_video_h}")
            print(f"[DEBUG] Snapshot panel width: {self.snapshot_panel_width}, height: {self.combined_window_height}")
            print(f"[DEBUG] Combined size: {combined_width}x{combined_height}")
            print(f"[DEBUG] Snapshot files count (in create_combined_frame): {len(self.snapshot_files)}")
            if snapshot_panel_image is None or snapshot_panel_image.size == 0:
                print("[DEBUG] snapshot_panel_image is empty or None!")
            else:
                print(f"[DEBUG] snapshot_panel_image shape: {snapshot_panel_image.shape}")
            if resized_video is None or resized_video.size == 0:
                print("[DEBUG] resized_video is empty or None!")
            else:
                print(f"[DEBUG] resized_video shape: {resized_video.shape}")

        # 통합 프레임 생성
        combined_frame = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
        # 배경색을 어둡게 설정하여 명확하게 구분
        combined_frame[:] = (50, 50, 50)

        # 왼쪽: 스냅샷 패널
        combined_frame[:, :self.snapshot_panel_width] = snapshot_panel_image
        
        # 가운데 구분선
        divider_x = self.snapshot_panel_width
        cv2.line(combined_frame, (divider_x, 0), 
                 (divider_x, combined_height), (0, 255, 0), 3)
                 
        # 오른쪽: 영상 분석 화면
        target_x_start = self.snapshot_panel_width + 3 # 구분선 너비 고려
        target_x_end = target_x_start + new_video_w
        
        # 복사 전에 대상 영역과 원본 이미지 크기 확인
        if target_x_end <= combined_width and new_video_h <= combined_height:
            combined_frame[:, target_x_start:target_x_end] = resized_video
        else:
            print("[DEBUG] Resized video does not fit in combined frame slice!")
            print(f"[DEBUG] Combined frame shape: {combined_frame.shape}")
            print(f"[DEBUG] Resized video shape: {resized_video.shape}")
            print(f"[DEBUG] Target slice: [:, {target_x_start}:{target_x_end}]")

        # 영상 정보 텍스트 오버레이 (영상 분석 부분에만) - 한글 텍스트 사용
        combined_frame = self.put_korean_text(combined_frame, "영상분석", 
                                            (self.snapshot_panel_width + int(new_video_w / 2) - 40, 10), 
                                            24, (255, 255, 255))
        combined_frame = self.put_korean_text(combined_frame, f"Frame: {frame_count}/{total_frames}", 
                                            (self.snapshot_panel_width + 10, 40), 
                                            18, (255, 255, 255))
        combined_frame = self.put_korean_text(combined_frame, f"Workers in ROI: {workers_in_roi}", 
                                            (self.snapshot_panel_width + 10, 70), 
                                            18, (0, 255, 255))
        
        return combined_frame

    def combined_mouse_callback(self, event, x, y, flags, param):
        """통합 프레임의 마우스 콜백"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 마우스 클릭이 확대된 스냅샷 닫기 버튼인지 확인
            if self.enlarged_snapshot is not None:
                # 현재는 show_enlarged_snapshot에서 별도 창을 띄우므로, 해당 창에 대한 클릭 처리가 필요함.
                # 통합 프레임에서는 확대 스냅샷이 표시되지 않으므로, 이 로직은 불필요하거나 별도 창의 마우스 콜백에 속해야 함.
                pass
            
            if x < self.snapshot_panel_width:  # 왼쪽 스냅샷 패널 영역
                # 스냅샷 패널 내 좌표는 그대로 사용
                self.handle_snapshot_click_in_panel(x, y)
            else:  # 오른쪽 영상 분석 영역
                if not self.detection_mode:
                    # ROI 설정 모드 (영상 영역에서만)
                    # 영상 영역 내 좌표로 변환
                    roi_x = x - (self.snapshot_panel_width + 3) # 구분선 너비 고려
                    roi_y = y
                    
                    if len(self.roi_points) < 4:
                        self.roi_points.append([roi_x, roi_y])
                        print(f"ROI 포인트 {len(self.roi_points)} 추가: ({roi_x}, {roi_y})")
                        if len(self.roi_points) == 4:
                            self.roi_set = True
                            self.detection_mode = True
                            print("[DEBUG] detection_mode -> True (ROI 설정 완료)")
                            self.save_roi_config()
                            print("ROI 설정 완료! 감지 모드로 전환됩니다.")

    def handle_snapshot_click_in_panel(self, x, y):
        """스냅샷 패널 내 클릭 처리 (패널 내 좌표 기준)"""
        print(f"[DEBUG] Snapshot panel click: x={x}, y={y}")
        y_offset = 80  # 패널 제목 아래 여백
        thumbnail_height = 80
        
        for i, filename in enumerate(self.snapshot_files[-10:]):
            if y_offset + thumbnail_height > self.combined_window_height - 50:  # 하단 여백 확보
                break
                
            thumbnail_y1 = y_offset
            thumbnail_y2 = y_offset + thumbnail_height
            
            # 썸네일 클릭 영역을 더 정확하게 정의 (패널 내 X축 범위 고려)
            # 스냅샷 썸네일은 패널 좌측 10px 시작, 너비는 panel_width - 20px
            if thumbnail_y1 <= y <= thumbnail_y2 and 10 <= x <= (self.snapshot_panel_width - 10):
                print(f"[DEBUG] Thumbnail {i} clicked!")
                # 클릭하면 확대 이미지 윈도우를 띄움 (새 창)
                filepath = os.path.join(self.snapshots_dir, filename)
                if os.path.exists(filepath):
                    enlarged = cv2.imread(filepath)
                    if enlarged is not None:
                        self.show_enlarged_snapshot(enlarged, filename)
                break
                
            y_offset += thumbnail_height + 25

    def show_enlarged_snapshot(self, image, filename):
        """확대된 스냅샷을 별도 창에 표시"""
        h, w = image.shape[:2]
        # 화면에 맞게 스케일 조정 (최대 너비 1000px, 최대 높이 800px)
        scale = min(1.0, 1000 / w, 800 / h) 
        new_w = int(w * scale)
        new_h = int(h * scale)
        enlarged = cv2.resize(image, (new_w, new_h))
        
        win_name = 'Enlarged Snapshot'
        cv2.imshow(win_name, enlarged)
        cv2.setWindowTitle(win_name, f'Enlarged Snapshot - {filename}')
        cv2.waitKey(1) # 창이 바로 닫히는 것을 방지

        # 닫기 버튼 클릭 처리 (새로운 콜백 함수를 여기에 연결)
        def enlarged_mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # 이미지 크기를 기준으로 'X' 버튼 영역을 대략적으로 계산 (우상단)
                # 닫기 버튼이 실제 이미지 내부에 그려진다면 해당 좌표를 기준으로 판단
                # 현재는 이미지 자체에 X 버튼을 그리지 않으므로, 창 닫는 기능만 구현
                cv2.destroyWindow(win_name)
                print("확대 스냅샷 창 닫힘.")

        cv2.setMouseCallback(win_name, enlarged_mouse_callback)

    def save_roi_config(self):
        """ROI 설정을 저장"""
        try:
            import json
            config = {'roi_points': self.roi_points}
            with open('roi_config.json', 'w') as f:
                json.dump(config, f)
            print("ROI 설정을 저장했습니다.")
        except Exception as e:
            print(f"ROI 설정 저장 실패: {e}")

if __name__ == "__main__":
    video_path = "cctv_video2.mp4"
    
    if not os.path.exists(video_path):
        print(f"비디오 파일을 찾을 수 없습니다: {video_path}")
    else:
        analyzer = VideoAnalysisSystem(video_path)
        analyzer.analyze_video() 