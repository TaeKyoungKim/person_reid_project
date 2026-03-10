import numpy as np
from typing import List, Dict, Any
from ultralytics import YOLO

from ..config.settings import config
from ..utils.logger import get_logger

logger = get_logger("pose_estimator")

def compute_iou(box1: List[float], box2: List[float]) -> float:
    """두 바운딩 박스 간의 IoU(Intersection over Union)를 계산합니다."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = intersection / float(area1 + area2 - intersection)
    return iou


class PoseEstimator:
    """
    YOLOv8-pose 기반 관절(Keypoints) 추출 클래스
    """
    def __init__(self):
        self.model_name = config.pose.model_name
        self.device = None if config.detection.device == "auto" else config.detection.device
        
        logger.info(f"Loading Pose model: {self.model_name}")
        try:
            self.model = YOLO(self.model_name)
        except Exception as e:
            logger.error(f"Failed to load pose model {self.model_name}: {e}")
            raise

    def estimate(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        프레임과 이전 탐지/추적 결과를 받아 포즈를 추정하고 결과를 업데이트합니다.
        
        Args:
            frame: 영상의 BGR 프레임 (numpy array)
            detections: 탐지된 사람 리스트 (id, bbox 정보 포함)
            
        Returns:
            List[Dict]: 포즈 정보(keypoints)가 추가된 탐지 결과
            keypoints shape: (17, 3) -> 각 키포인트는 [x, y, confidence] 형태
        """
        if not detections:
            return detections
            
        # 포즈 모델 추론 수행 (사람만 탐지)
        results = self.model(frame, classes=[0], verbose=False, device=self.device)
        
        pose_results = []
        if len(results) > 0 and results[0].keypoints is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            keypoints = results[0].keypoints.data.cpu().numpy()  # (N, 17, 3)
            
            for i in range(len(boxes)):
                pose_results.append({
                    "bbox": boxes[i].tolist(),
                    "keypoints": keypoints[i].tolist()
                })
        
        # 추적된 바운딩 박스와 포즈 모델의 바운딩 박스를 IoU로 매칭
        for det in detections:
            det["keypoints"] = None  # 기본적으로 None 할당
            
            best_iou = 0
            best_pose = None
            
            for pose in pose_results:
                iou = compute_iou(det["bbox"], pose["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_pose = pose
                    
            # IoU가 0.5 이상인 경우에만 포즈 정보 할당
            if best_iou > 0.5 and best_pose is not None:
                det["keypoints"] = best_pose["keypoints"]
                
        return detections
