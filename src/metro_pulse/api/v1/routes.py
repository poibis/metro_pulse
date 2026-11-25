"""FastAPI endpoints for Metro Pulse simulation."""

from typing import Dict, List, Any
from fastapi import APIRouter, HTTPException

from metro_pulse.schemas.simulation import SimulationRequest, SimulationResponse
from metro_pulse.ml.simulator import get_simulator
from metro_pulse.ml.loader import MetroDataLoader

router = APIRouter(tags=["Simulation"])


@router.get("/stations", response_model=Dict[str, Any])
async def get_stations():
    """사용 가능한 지하철 노선 및 역 목록 반환."""
    loader = MetroDataLoader.get_instance()
    loader.load_data()
    return {
        "lines": loader.lines,
        "stations_by_line": loader.stations_by_line,
    }


@router.post("/simulate", response_model=SimulationResponse)
async def simulate(req: SimulationRequest):
    """
    AI 수요 예측 및 관제·승무 통합 시뮬레이션 수행.
    - LightGBM 승차인원 추론
    - 목표 혼잡도 달성을 위한 배차간격 동적 최적화 (Self-Correction)
    - 승객 과밀 하중에 따른 기관사 주행 안전 가이던스 산출
    - XAI 변수 기여도 도출
    """
    try:
        sim = get_simulator()
        return sim.run_simulation(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"시뮬레이션 수행 오류: {str(e)}")
