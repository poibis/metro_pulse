"""Main FastAPI application for Metro Pulse."""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from metro_pulse.core.config import settings
from metro_pulse.api.v1.routes import router as v1_router
from metro_pulse.ml.loader import MetroDataLoader
from metro_pulse.ml.simulator import get_simulator


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 스타트업 시 데이터 로딩 및 모델 예열
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

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 라우터 등록
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


# 프론트엔드 정적 서빙
frontend_dir = settings.FRONTEND_DIR if hasattr(settings, "FRONTEND_DIR") else Path(__file__).resolve().parent.parent.parent / "frontend"

if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        index_file = frontend_dir / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return {"message": "Metro Pulse API Server is running. Visit /docs for Swagger UI."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("metro_pulse.main:app", host="127.0.0.1", port=8000, reload=True)
