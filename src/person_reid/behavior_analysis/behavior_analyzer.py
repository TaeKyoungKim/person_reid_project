import numpy as np
import time
from typing import List, Dict, Any

from ..config.settings import config
from ..utils.logger import get_logger

logger = get_logger("behavior_analyzer")


class BehaviorAnalyzer:
    """
    탐지된 객체의 추적 ID와 포즈를 기반으로 이상행동(쓰러짐, 배회 등)을 감지하는 클래스
    """
    def __init__(self):
        self.fall_enabled = config.behavior.fall_detection_enabled
        self.motion_enabled = config.behavior.motion_analysis_enabled
        self.inactivity_threshold = config.behavior.inactivity_threshold  # 단위: 초
        self.fall_thresh = config.behavior.fall_confidence_threshold
        
        # 사람별 상태 히스토리 저장 (id -> state dict)
        self.person_states = {}

    def analyze(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        탐지된 결과를 바탕으로 행동(쓰러짐, 배회 등)을 분석합니다.
        
        Args:
            detections: List of dict (id, bbox, confidence, keypoints 포함)
            
        Returns:
            Dict: 현재 프레임의 전체 상태 요약(total_count, alerts 등)
        """
        current_time = time.time()
        current_ids = set()
        alerts = []
        
        for det in detections:
            person_id = det["id"]
            if person_id == -1:
                # 추적 ID가 할당되지 않은 미식별 객체는 분석에서 제외 (또는 일시 보류)
                continue
                
            current_ids.add(person_id)
            
            # 박스 중심 좌표 계산
            x1, y1, x2, y2 = det["bbox"]
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            
            if person_id not in self.person_states:
                self.person_states[person_id] = {
                    "first_seen": current_time,
                    "last_seen": current_time,
                    "history": [(center_x, center_y, current_time)],
                    "is_falling": False,
                    "is_loitering": False
                }
            else:
                state = self.person_states[person_id]
                state["last_seen"] = current_time
                state["history"].append((center_x, center_y, current_time))
                
                # 히스토리는 inactivity threshold의 약 1.5배 기간까지만 보관하여 메모리 관리
                max_history_time = self.inactivity_threshold * 1.5
                state["history"] = [
                    h for h in state["history"] 
                    if current_time - h[2] <= max_history_time
                ]
            
            state = self.person_states[person_id]
            det["alert"] = None  # 기본 상태
            
            # 1. 쓰러짐 감지(Fall Detection)
            if self.fall_enabled:
                is_fall = self._detect_fall(det.get("keypoints"), det["bbox"])
                state["is_falling"] = is_fall
                if is_fall:
                    det["alert"] = "FALL DETECTED"
                    alerts.append(f"Person {person_id} fall detected!")
            
            # 2. 배회 및 정지 감지(Loitering/Inactivity Detection)
            # 쓰러짐이 감지된 상황에서는 배회가 아닌 응급상황이므로 예외 처리
            if self.motion_enabled and not state["is_falling"]:
                is_loitering = self._detect_loitering(state["history"], current_time)
                state["is_loitering"] = is_loitering
                if is_loitering:
                    # 기존에 다른 알람이 없다면 설정
                    if det["alert"] is None:
                        det["alert"] = "LOITERING DETECTED"
                    alerts.append(f"Person {person_id} is loitering!")
                    
        # 일정 시간(예: 5초) 동안 탐지되지 않은 ID의 히스토리는 정리
        ids_to_remove = [
            pid for pid, state in self.person_states.items() 
            if current_time - state["last_seen"] > 5.0
        ]
        
        for pid in ids_to_remove:
            del self.person_states[pid]
            
        summary = {
            "total_count": len(current_ids),
            "alerts": alerts
        }
        
        return summary
        
    def _detect_fall(self, keypoints: List[List[float]], bbox: List[int]) -> bool:
        """
        관절 위치(Keypoints)와 바운딩 박스를 이용하여 쓰러짐 여부를 판단합니다.
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        if height == 0:
            return False
            
        aspect_ratio = width / float(height)
        
        # 키포인트 존재 여부 확인
        if keypoints is not None and len(keypoints) >= 13: # 최소 골반 위치까지는 있어야 판단 가능
            try:
                # YOLOv8 Pose Keypoints 인덱스:
                # 5: 좌측 어깨, 6: 우측 어깨
                # 11: 좌측 골반, 12: 우측 골반
                ls = keypoints[5]
                rs = keypoints[6]
                lh = keypoints[11]
                rh = keypoints[12]
                
                # 각 파츠의 추론 신뢰도가 thresholds 이상일 때만 각도 계산
                if ls[2] > 0.5 and rs[2] > 0.5 and lh[2] > 0.5 and rh[2] > 0.5:
                    shoulder_cx = (ls[0] + rs[0]) / 2.0
                    shoulder_cy = (ls[1] + rs[1]) / 2.0
                    hip_cx = (lh[0] + rh[0]) / 2.0
                    hip_cy = (lh[1] + rh[1]) / 2.0
                    
                    dx = hip_cx - shoulder_cx
                    dy = hip_cy - shoulder_cy
                    
                    # 어깨-골반 벡터의 각도 계산 (x축 기준 각도, 누우면 0도 혹은 180도에 가까워짐)
                    angle = np.degrees(np.arctan2(abs(dy), abs(dx)))
                    
                    # 각도가 45도 미만이면 몸이 상당히 기울었음을 의미
                    if angle < 45.0:
                        return True
                    else:
                        return False
            except Exception as e:
                logger.debug(f"Keypoint parsing error in fall detection: {e}")
                
        # 키포인트를 사용할 수 없거나 신뢰도가 너무 낮을 경우: bbox의 종횡비로 Fallback 판단
        # 일반적으로 서 있는 사람은 H > W (비율 < 1.0), 쓰러진 사람은 W > H (비율 > 1.2 등)
        if aspect_ratio > 1.3:
            return True
            
        return False
        
    def _detect_loitering(self, history: List[tuple], current_time: float) -> bool:
        """
        히스토리 기반으로 설정된 시간(inactivity_threshold) 이상 움직임(변위)이 적은지 판단합니다.
        """
        if len(history) < 2:
            return False
            
        first_time = history[0][2]
        # 해당 객체가 추적된 전체 시간이 threshold보다 작으면 아직 판단 보류
        if current_time - first_time < self.inactivity_threshold:
            return False
            
        # threshold 시간 이전에 저장된 위치를 타겟 위치로 선정
        target_hist = None
        for h in reversed(history):
            if current_time - h[2] >= self.inactivity_threshold:
                target_hist = h
                break
                
        if target_hist is None:
            return False
            
        curr_x, curr_y = history[-1][0], history[-1][1]
        prev_x, prev_y = target_hist[0], target_hist[1]
        
        distance = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
        
        # 이동 거리가 threshold 픽셀 이하인 경우 배회/정지로 판단 (해상도에 따라 거리 임계치 조정 필요)
        # TODO: camera perspective calibration 없이 대략적인 픽셀 사용 (현재 50px 기반)
        if distance < 50.0:
            return True
            
        return False
