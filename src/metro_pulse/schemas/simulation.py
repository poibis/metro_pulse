from typing import List, Dict, Optional
from pydantic import BaseModel, Field

class SimulationRequest(BaseModel):
    line_name: str = Field(default="2호선", description="지하철 호선 (예: 2호선)")
    departure_station: str = Field(default="잠실", description="출발역명")
    arrival_station: Optional[str] = Field(default="잠실새내", description="도착역명 (구간 분석용)")
    scenario: str = Field(default="event", description="시나리오 유형 (weekday, weekend, rain, event)")
    weather_coeff: float = Field(default=1.1, ge=0.5, le=2.0, description="기상 계수 (1.0 = 맑음, >1.0 = 우천/악천후)")
    event_coeff: float = Field(default=1.3, ge=0.5, le=2.5, description="이벤트 계수 (1.0 = 일반, >1.0 = 콘서트/행사)")
    target_congestion_pct: float = Field(default=130.0, ge=80.0, le=250.0, description="목표 혼잡도 상한선 (%)")
    train_capacity: int = Field(default=1600, ge=500, le=3000, description="열차 1편성 기준 정원")

class DriverGuidance(BaseModel):
    recommended_max_speed_kmh: float = Field(description="권장 역간 최고속도 (km/h)")
    coasting_start_ratio_pct: float = Field(description="타력주행 개시 비율 (%)")
    dwell_time_limit_sec: int = Field(description="승객 과밀 반영 역 정차 허용 시간 (초)")
    braking_alert: str = Field(description="제동 안전 및 운전 주의 안내 브리핑")

class HourlySimulationItem(BaseModel):
    time_label: str = Field(description="시간대 레이블 (예: 18-19)")
    baseline_pax: int = Field(description="기준 일평균 승차인원")
    predicted_pax: int = Field(description="AI 시나리오 예측 승차인원")
    current_interval_min: float = Field(description="AI 초기 수요 비례 배차간격 (분)")
    optimal_interval_min: float = Field(description="물리 제약 반영 최적 배차간격 (분)")
    min_physical_headway_min: float = Field(description="역간거리·주행시간·정차지연에 따른 물리적 최소 안전 시격 한계 (분)")
    dwell_time_sec: int = Field(description="시간대별 탑승 인원에 의한 실제 승강장 정차 시간 (초)")
    required_extra_trains: int = Field(description="필요 추가 증편 수 (차량기지 가용 한계 내)")
    current_congestion_pct: float = Field(description="초기 배차 시 예상 혼잡도 (%)")
    optimal_congestion_pct: float = Field(description="최적 배차 시 최종 혼잡도 (%)")
    target_achieved: bool = Field(description="목표 혼잡도 달성 여부 (False: 물리적 한계 도달)")
    bottleneck_cause: Optional[str] = Field(default=None, description="목표 미달 시 물리적 병목 원인 요약")
    extra_load_ton: float = Field(description="정원 초과로 인한 추가 승객 하중 (톤)")
    braking_distance_increase_pct: float = Field(description="제동거리 증가율 (%)")
    driver_guidance: DriverGuidance

class ProfilingDetail(BaseModel):
    phase1_data_lookup_ms: float = Field(description="1단계: 공공데이터 역사/구간 조회 소요시간 (ms)")
    phase2_feature_prep_ms: float = Field(description="2단계: ML 입력 피처 매트릭스 변환 소요시간 (ms)")
    phase3_lgbm_inference_ms: float = Field(description="3단계: LightGBM 모델 실시간 회귀 추론 소요시간 (ms)")
    phase4_self_correction_ms: float = Field(description="4단계: 물리 제약 기반 Self-Correction 탐색 루프 (ms)")
    phase5_guidance_calc_ms: float = Field(description="5단계: 물리 제동거리 및 승무 가이던스 연산 소요시간 (ms)")
    total_time_ms: float = Field(description="총 소요시간 (ms)")

class SimulationMeta(BaseModel):
    line_name: str
    departure_station: str
    arrival_station: Optional[str]
    distance_km: float
    standard_travel_sec: int
    inference_time_ms: float = 0.0
    model_name: str = "Pure LightGBM + Real-world Railway Physics Constraints"
    profiling: ProfilingDetail
    cache_hit: bool = False

class SimulationResponse(BaseModel):
    meta: SimulationMeta
    xai_feature_importance_pct: Dict[str, float]
    hourly_simulation: List[HourlySimulationItem]
