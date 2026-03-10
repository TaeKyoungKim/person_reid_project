import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from ..config.settings import config
from ..utils.logger import get_logger

logger = get_logger("video_processor")

class VideoProcessor:
    def __init__(self, source: str, output_path: str = None):
        self.source = source
        self.output_path = output_path
        
        # 입력 소스가 숫자면 웹캠 인덱스로 간주
        if str(source).isdigit():
            self.source = int(source)
            
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            logger.error(f"Failed to open video source: {self.source}")
            raise ValueError(f"Cannot open video source: {self.source}")
            
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        if self.fps == 0:
            self.fps = config.video.fps
            if self.fps == 0:
                self.fps = 30
            
        logger.info(f"Video Source Initialized: {self.source} ({self.width}x{self.height} @ {self.fps}fps)")
        
        self.writer = None
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))
            logger.info(f"Video Writer Initialized: {self.output_path}")
            
    def read_frame(self) -> Tuple[bool, np.ndarray]:
        return self.cap.read()
        
    def write_frame(self, frame: np.ndarray):
        if self.writer:
            self.writer.write(frame)
            
    def draw_results(self, frame: np.ndarray, detections: List[Dict[str, Any]], summary: Dict[str, Any]) -> np.ndarray:
        """분석 결과 시각화 모듈"""
        vis_frame = frame.copy()
        
        # 탐지 결과 및 경고 표출
        for det in detections:
            bbox = det.get("bbox")
            track_id = det.get("id", -1)
            alert = det.get("alert")
            keypoints = det.get("keypoints")
            
            if not bbox: continue
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # 알람 여부에 따라 BBox 색상 지정 (빨강: 쓰러짐/배회, 초록: 정상)
            color = (0, 0, 255) if alert else (0, 255, 0)
            
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # ID 및 상태 표출 텍스트
            label = f"ID: {track_id}"
            if alert:
                label += f" | {alert}"
                
            # 텍스트 백그라운드 추가로 가독성 확보
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis_frame, (x1, y1 - text_height - baseline - 5), (x1 + text_width, y1), color, -1)
            cv2.putText(vis_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
            # 포즈 키포인트 렌더링
            if keypoints is not None:
                for pt in keypoints:
                    px, py, conf = pt
                    if conf > 0.5:
                        cv2.circle(vis_frame, (int(px), int(py)), 3, (255, 0, 0), -1)
                        
        # 전체 통계/알림 요약 표출
        total_count = summary.get("total_count", 0)
        cv2.putText(vis_frame, f"Total Count: {total_count}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2, cv2.LINE_AA)
                    
        return vis_frame
        
    def release(self):
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()
