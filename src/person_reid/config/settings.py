"""
프로젝트 설정 파일

이 파일은 모든 모듈에서 사용하는 설정값들을 정의합니다.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


class AlertLevel(Enum):
    """알림 레벨 정의"""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class DetectionConfig:
    """사람 탐지 설정"""
    model_name: str = "yolov8n.pt"
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    input_size: tuple = (640, 640)
    device: str = "auto"  # "cpu", "cuda", "auto"


@dataclass
class PoseConfig:
    """포즈 추정 설정"""
    model_name: str = "yolov8n-pose.pt"
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    static_image_mode: bool = False


@dataclass
class BehaviorConfig:
    """행동 분석 설정"""
    fall_detection_enabled: bool = True
    motion_analysis_enabled: bool = True
    inactivity_threshold: int = 30  # 초
    fall_confidence_threshold: float = 0.7
    motion_sensitivity: float = 0.1


@dataclass
class VideoConfig:
    """비디오 처리 설정"""
    fps: int = 30
    frame_width: int = 1920
    frame_height: int = 1080
    buffer_size: int = 100
    night_mode: bool = False


@dataclass
class APIConfig:
    """API 설정"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    workers: int = 1
    timeout: int = 30


@dataclass
class AlertConfig:
    """알림 설정"""
    server_url: str = "http://localhost:8001"
    webhook_url: Optional[str] = None
    email_enabled: bool = False
    sms_enabled: bool = False
    alert_cooldown: int = 60  # 초


@dataclass
class LoggingConfig:
    """로깅 설정"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


class Config:
    """전체 설정 클래스"""
    
    def __init__(self):
        self.detection = DetectionConfig()
        self.pose = PoseConfig()
        self.behavior = BehaviorConfig()
        self.video = VideoConfig()
        self.api = APIConfig()
        self.alert = AlertConfig()
        self.logging = LoggingConfig()
        
        # 환경 변수에서 설정 로드
        self._load_from_env()
    
    def _load_from_env(self):
        """환경 변수에서 설정 로드"""
        # Detection 설정
        if os.getenv("DETECTION_CONFIDENCE"):
            self.detection.confidence_threshold = float(os.getenv("DETECTION_CONFIDENCE"))
        
        if os.getenv("DETECTION_DEVICE"):
            self.detection.device = os.getenv("DETECTION_DEVICE")
        
        # API 설정
        if os.getenv("API_HOST"):
            self.api.host = os.getenv("API_HOST")
        
        if os.getenv("API_PORT"):
            self.api.port = int(os.getenv("API_PORT"))
        
        # 알림 설정
        if os.getenv("ALERT_SERVER_URL"):
            self.alert.server_url = os.getenv("ALERT_SERVER_URL")
        
        if os.getenv("ALERT_WEBHOOK_URL"):
            self.alert.webhook_url = os.getenv("ALERT_WEBHOOK_URL")
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            "detection": self.detection.__dict__,
            "pose": self.pose.__dict__,
            "behavior": self.behavior.__dict__,
            "video": self.video.__dict__,
            "api": self.api.__dict__,
            "alert": self.alert.__dict__,
            "logging": self.logging.__dict__,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """딕셔너리에서 설정 생성"""
        config = cls()
        
        if "detection" in config_dict:
            for key, value in config_dict["detection"].items():
                setattr(config.detection, key, value)
        
        if "pose" in config_dict:
            for key, value in config_dict["pose"].items():
                setattr(config.pose, key, value)
        
        if "behavior" in config_dict:
            for key, value in config_dict["behavior"].items():
                setattr(config.behavior, key, value)
        
        if "video" in config_dict:
            for key, value in config_dict["video"].items():
                setattr(config.video, key, value)
        
        if "api" in config_dict:
            for key, value in config_dict["api"].items():
                setattr(config.api, key, value)
        
        if "alert" in config_dict:
            for key, value in config_dict["alert"].items():
                setattr(config.alert, key, value)
        
        if "logging" in config_dict:
            for key, value in config_dict["logging"].items():
                setattr(config.logging, key, value)
        
        return config


# 전역 설정 인스턴스
config = Config() 