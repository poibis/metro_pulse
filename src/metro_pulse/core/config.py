from pathlib import Path
from pydantic_settings import BaseSettings

# Docker 볼륨 마운트 경로(/app/data)가 존재하면 /app을 루트로 사용, 아니면 로컬 프로젝트 루트 사용
_DOCKER_ROOT = Path("/app")
if (_DOCKER_ROOT / "data").exists():
    PROJECT_ROOT = _DOCKER_ROOT
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FRONTEND_DIR = PROJECT_ROOT / "frontend"


class Settings(BaseSettings):
    app_name: str = "Metro Pulse AI Simulator"
    app_version: str = "0.1.0"
    debug: bool = False

    # Paths
    hourly_csv_path: Path = DATA_DIR / "seoul_metro_hourly.csv"
    distance_csv_path: Path = DATA_DIR / "seoul_metro_distance.csv"
    model_artifact_path: Path = ARTIFACTS_DIR / "metro_pulse_lgbm.joblib"

    # Model defaults
    default_capacity: int = 1600  # 10량 1편성 기준 정원
    default_target_congestion: float = 130.0  # 목표 혼잡도 %

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()