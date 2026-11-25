"""Main FastAPI application for Metro Pulse."""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from metro_pulse.core.config import settings
from metro_pulse.api.v1.routes import router as v1_router
from metro_pulse.ml.loader import MetroDataLoader
from metro_pulse.ml.simulator import get_simulator


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(">> [Metro Pulse] 서버 초기화 중: 데이터 및 AI 모델 로드...")
    loader = MetroDataLoader.get_instance()
    loader.load_data()
    _ = get_simulator()
    print(">> [Metro Pulse] AI 시뮬레이터 준비 완료!")
    yield
    print(">> [Metro Pulse] 서버 종료.")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="서울교통공사 실데이터 기반 AI 관제·승무 통합 시뮬레이터",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(v1_router, prefix="/api/v1")


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
    }


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


# Docker 볼륨 마운트 경로(/app/frontend) 우선 탐색, 로컬 실행 시 상대 경로 폴백
FRONTEND_DIR = Path("/app/frontend")
if not FRONTEND_DIR.exists():
    FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def serve_index():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse(
        status_code=404,
        content={"detail": f"index.html not found at {index_file}"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("metro_pulse.main:app", host="127.0.0.1", port=8000, reload=True)