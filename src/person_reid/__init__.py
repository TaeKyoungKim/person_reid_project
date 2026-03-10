"""
AI 기반 사람 탐지 및 행동 분석 시스템

이 패키지는 실시간 이미지/영상에서 사람을 식별하고,
인원수 카운팅, 쓰러짐 감지, 움직임 분석을 통해 이상 상황을 감지하는 AI 시스템입니다.
"""

__version__ = "0.1.0"
__author__ = "Person ReID Team"
__description__ = "AI 기반 사람 탐지 및 행동 분석 시스템"

from .detection.person_detector import PersonDetector
from .pose_estimation.pose_estimator import PoseEstimator
from .behavior_analysis.behavior_analyzer import BehaviorAnalyzer
from .api.server import start_server

__all__ = [
    "PersonDetector",
    "PoseEstimator", 
    "BehaviorAnalyzer",
    "start_server",
] 