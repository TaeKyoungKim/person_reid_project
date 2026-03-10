import argparse
import cv2
import sys
from pathlib import Path

# 소스 모듈 import
from src.person_reid.detection.person_detector import PersonDetector
from src.person_reid.pose_estimation.pose_estimator import PoseEstimator
from src.person_reid.behavior_analysis.behavior_analyzer import BehaviorAnalyzer
from src.person_reid.utils.video_processor import VideoProcessor
from src.person_reid.api.server import start_server, app as fastapi_app
from src.person_reid.utils.logger import get_logger

logger = get_logger("main")

def parse_args():
    parser = argparse.ArgumentParser(description="AI 기반 사람 탐지 및 행동 분석 시스템")
    parser.add_argument("--source", type=str, default="0", help="비디오 소스 (0: 웹캠, 또는 비디오 파일 경로)")
    parser.add_argument("--output", type=str, default=None, help="결과 비디오 저장 경로 (예: output.mp4)")
    parser.add_argument("--no-api", action="store_true", help="API 서버를 실행하지 않음")
    parser.add_argument("--api-port", type=int, default=8000, help="API 서버 포트")
    parser.add_argument("--headless", action="store_true", help="OpenCV 화면 출력을 비활성화")
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info("시스템 초기화를 시작합니다...")
    
    # 코어 분석 모듈 클래스 초기화
    try:
        detector = PersonDetector()
        pose_estimator = PoseEstimator()
        behavior_analyzer = BehaviorAnalyzer()
        video_processor = VideoProcessor(source=args.source, output_path=args.output)
    except Exception as e:
        logger.error(f"모듈 초기화 중 오류가 발생했습니다: {e}")
        sys.exit(1)
    
    # API 서버 백그라운드 시작
    if not args.no_api:
        start_server(port=args.api_port)
        
    logger.info("비디오 분석 루프를 시작합니다. (중지하려면 터미널에서 Ctrl+C 누르기 또는 화면 클릭 후 'q' 누르기)")
    
    try:
        while True:
            ret, frame = video_processor.read_frame()
            if not ret:
                logger.info("비디오 스트림이 종료되었습니다.")
                break
                
            # 1. 탐지 및 추적 
            # (YOLOv8 ByteTrack을 이용하여 BBox 및 ID 발급)
            _, detections = detector.detect_and_track(frame)
            
            # 2. 포즈(관절) 추출
            # (YOLO-pose 모델로 전체 이미지 내 키포인트를 추론 후 IoU 매칭 통해 Box와 결합)
            detections = pose_estimator.estimate(frame, detections)
            
            # 3. 행동 분석 (쓰러짐, 배회 감지 및 시스템 카운트)
            summary = behavior_analyzer.analyze(detections)
            
            # 전역 변수에 현재 상태를 덮어써서 FastAPI 호출 시 반환되도록 설정
            fastapi_app.state.current_summary = summary
            
            # 4. 시각화 렌더링 요소 적용
            vis_frame = video_processor.draw_results(frame, detections, summary)
            
            # 콘솔에 심각한 알람 내용 출력
            for alert in summary.get("alerts", []):
                logger.warning(f"경보 통지: {alert}")
                
            # 결과 프레임 저장
            if args.output:
                video_processor.write_frame(vis_frame)
                
            # 결과 화면 출력
            if not args.headless:
                cv2.imshow("Person ReID & Behavior Analysis", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("종료 명령(q)이 입력되었습니다.")
                    break
                
    except KeyboardInterrupt:
        logger.info("사용자에 의해 강제 종료되었습니다. (Ctrl+C)")
    except Exception as e:
        logger.error(f"메인 루프 실행 중 오류 발생: {e}", exc_info=True)
    finally:
        # 종료 전 자원 수거
        video_processor.release()
        cv2.destroyAllWindows()
        logger.info("시스템이 안전하게 종료되었습니다.")

if __name__ == "__main__":
    main()
