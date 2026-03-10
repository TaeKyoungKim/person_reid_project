from fastapi import FastAPI
import threading
import uvicorn
from ..utils.logger import get_logger

logger = get_logger("api_server")

app = FastAPI(title="Person ReID API", description="현재 CCTV 시스템 내의 인원 카운팅 및 알람 정보 제공 API")

# 전역 상태 (메인 루프에서 주기적으로 업데이트)
app.state.current_summary = {"total_count": 0, "alerts": []}

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Person ReID Server is running."}

@app.get("/api/status")
def get_status():
    """현재 카메라의 인원 상태 및 발생한 알람 수를 반환합니다."""
    return app.state.current_summary

def start_server(host="0.0.0.0", port=8000):
    """API 서버를 백그라운드 스레드에서 비동기적(데몬 스레드)으로 실행합니다."""
    def run():
        logger.info(f"Starting API Server on {host}:{port}")
        uvicorn.run(app, host=host, port=port, log_level="warning")
        
    server_thread = threading.Thread(target=run, daemon=True)
    server_thread.start()
    return app
