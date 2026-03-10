import numpy as np
from typing import List, Dict, Any, Tuple
from ultralytics import YOLO

from ..config.settings import config
from ..utils.logger import get_logger

logger = get_logger("person_detector")

class PersonDetector:
    """
    YOLOv8 기반 사람 탐지 및 추적 클래스
    """
    
    def __init__(self):
        self.model_name = config.detection.model_name
        self.conf_threshold = config.detection.confidence_threshold
        
        # device "auto" 처리는 ultralytics에서 자연스럽게 처리되지 않을 수 있으므로 None으로 설정
        self.device = None if config.detection.device == "auto" else config.detection.device
        
        logger.info(f"Loading YOLO model: {self.model_name}")
        try:
            # YOLOv8 모델 로드
            self.model = YOLO(self.model_name)
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
            
        # 클래스 ID 중 사람(0)만 필터링하기 위한 설정
        self.person_class_id = 0

    def detect_and_track(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        프레임에서 사람을 탐지하고 추적합니다.
        
        Args:
            frame: 영상의 BGR 프레임 (numpy array)
            
        Returns:
            Tuple[np.ndarray, List[Dict]]: (시각화된 프레임, 탐지 결과 리스트)
            탐지 결과는 다음과 같은 형태를 가집니다:
            [
                {
                    "id": 추적 ID (int),
                    "bbox": [x1, y1, x2, y2] (list of int),
                    "confidence": 신뢰점수 (float)
                }, ...
            ]
        """
        # YOLOv8의 track 메서드 사용. persist=True로 프레임 간 추적 ID 유지.
        results = self.model.track(
            frame, 
            persist=True, 
            classes=[self.person_class_id],
            conf=self.conf_threshold,
            device=self.device,
            tracker="bytetrack.yaml",
            verbose=False
        )
        
        detections = []
        annotated_frame = frame.copy()
        
        if len(results) > 0 and results[0].boxes is not None:
            # 기본 시각화 결과 사용
            annotated_frame = results[0].plot()
            
            boxes = results[0].boxes
            for i in range(len(boxes)):
                box = boxes[i]
                
                # 아직 추적 ID가 할당되지 않은 경우 예외 처리
                track_id = -1
                if box.id is not None:
                    track_id = int(box.id.item())
                
                # Bounding box 좌표 추출 (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf.item())
                
                detections.append({
                    "id": track_id,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf
                })
                
        return annotated_frame, detections
