"""
로깅 유틸리티

이 모듈은 프로젝트 전반에서 사용하는 로깅 설정을 제공합니다.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from loguru import logger
from ..config.settings import config


def setup_logger(
    name: str = "person_reid",
    level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
    format: str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
) -> logger:
    """
    로거 설정
    
    Args:
        name: 로거 이름
        level: 로그 레벨
        log_file: 로그 파일 경로
        rotation: 로그 파일 로테이션 설정
        retention: 로그 파일 보관 기간
        format: 로그 포맷
    
    Returns:
        설정된 로거 인스턴스
    """
    # 기존 핸들러 제거
    logger.remove()
    
    # 콘솔 출력 핸들러 추가
    logger.add(
        sys.stdout,
        format=format,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # 파일 출력 핸들러 추가 (선택사항)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            format=format,
            level=level,
            rotation=rotation,
            retention=retention,
            backtrace=True,
            diagnose=True,
            encoding="utf-8"
        )
    
    # 표준 라이브러리 로깅과 연동
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1
            
            logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )
    
    # 표준 라이브러리 로깅 핸들러 설정
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    return logger


def get_logger(name: str = "person_reid") -> logger:
    """
    로거 인스턴스 반환
    
    Args:
        name: 로거 이름
    
    Returns:
        로거 인스턴스
    """
    return logger.bind(name=name)


# 기본 로거 설정
default_logger = setup_logger(
    name="person_reid",
    level=config.logging.level,
    log_file=config.logging.file_path,
    rotation=f"{config.logging.max_size // (1024*1024)} MB",
    retention=f"{config.logging.backup_count} days"
) 