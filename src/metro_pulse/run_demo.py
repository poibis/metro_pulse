"""Standalone demo script for Metro Pulse AI Simulator."""

import sys
from metro_pulse.schemas.simulation import SimulationRequest
from metro_pulse.ml.simulator import get_simulator


def main():
    print("=" * 80)
    print("       Metro Pulse AI 관제·승무 통합 시뮬레이터 실행 데모 (순수 수식 & ML)")
    print("=" * 80)

    # 1. 시뮬레이터 초기화
    print("\n[1] AI 모델(LightGBM) 및 서울교통공사 실데이터 로딩...")
    sim = get_simulator()
    print("    -> 로딩 완료!")

    # 2. 시뮬레이션 파라미터 설정
    req = SimulationRequest(
        line_name="2호선",
        departure_station="잠실",
        arrival_station="잠실새내",
        scenario="event",
        weather_coeff=1.15,
        event_coeff=1.35,
        target_congestion_pct=130.0,
        train_capacity=1600,
    )

    print("\n[2] 시뮬레이션 입력 조건:")
    print(f"    - 대상 노선: {req.line_name} ({req.departure_station}역 -> {req.arrival_station}역)")
    print(f"    - 시나리오: {req.scenario} (기상 계수: {req.weather_coeff}, 행사 계수: {req.event_coeff})")
    print(f"    - 목표 혼잡도 상한선: {req.target_congestion_pct}% | 열차 기준 정원: {req.train_capacity}명")

    # 3. AI 시뮬레이션 실행
    print("\n[3] AI 추론 및 배차 Self-Correction 동적 최적화 실행 중...")
    res = sim.run_simulation(req)

    # 4. 연산 단계별 정밀 프로파일링 출력
    prof = res.meta.profiling
    print("\n" + "-" * 80)
    print(" [⏱️ AI 연산 단계별 정밀 프로파일링 (어디서 시간이 소요되었는가?)]")
    print(f"  • 1단계 (공공데이터베이스 조회)       : {prof.phase1_data_lookup_ms:>6.3f} ms")
    print(f"  • 2단계 (피처 매트릭스 변환)          : {prof.phase2_feature_prep_ms:>6.3f} ms")
    print(f"  • 3단계 (LightGBM 머신러닝 회귀 추론) : {prof.phase3_lgbm_inference_ms:>6.3f} ms  <-- 최다 소요 구간")
    print(f"  • 4단계 (Self-Correction 최적화 루프) : {prof.phase4_self_correction_ms:>6.3f} ms")
    print(f"  • 5단계 (열차 물리 제동 및 안전 연산) : {prof.phase5_guidance_calc_ms:>6.3f} ms")
    print(f"  -------------------------------------------------------------")
    print(f"  • 총 연산 소요 시간                   : {prof.total_time_ms:>6.2f} ms")

    # 5. XAI 변수 기여도 출력
    print("\n [XAI 변수 기여도 (Feature Importance %)]")
    for feat, imp in res.xai_feature_importance_pct.items():
        bar = "■" * int(imp // 2)
        print(f"  • {feat:<15}: {imp:>5.1f}% | {bar}")

    # 6. 주요 시간대별 시뮬레이션 결과 표 출력
    print("\n" + "=" * 80)
    print(" [시간대별 AI 관제 최적화 및 승무원 주행 가이던스 (순수 공식 산출)]")
    print("=" * 80)
    print(f"{'시간대':^7} | {'예측승객':^7} | {'초기시격':^6} -> {'최적시격':^6} | {'증편':^4} | {'초기혼잡도':^8} -> {'최적혼잡도':^8} | {'추가하중':^6} | {'최고속도':^6}")
    print("-" * 80)

    for item in res.hourly_simulation:
        print(
            f"{item.time_label:^7} | "
            f"{item.predicted_pax:>6}명 | "
            f"{item.current_interval_min:>4.1f}분 -> {item.optimal_interval_min:>4.1f}분 | "
            f"+{item.required_extra_trains:>2}편 | "
            f"{item.current_congestion_pct:>7.1f}% -> {item.optimal_congestion_pct:>7.1f}% | "
            f"+{item.extra_load_ton:>4.1f}t | "
            f"{item.driver_guidance.recommended_max_speed_kmh:>5.1f}km/h"
        )

    # 7. 퇴근 피크시간대 (18-19시) 승무 브리핑 및 감사 근거
    peak_18 = next(x for x in res.hourly_simulation if x.time_label == "18-19")
    print("-" * 80)
    print(f"\n[📐 의사결정 근거 및 감사 리포트 (Audit Report: 18-19시 피크)]")
    print(f"  • 기준 수요: {peak_18.baseline_pax:,}명 -> AI 시나리오 예측: {peak_18.predicted_pax:,}명")
    print(f"  • 관제 지침: 초기 시격 {peak_18.current_interval_min}분 -> 최적 {peak_18.optimal_interval_min}분 단축, 열차 +{peak_18.required_extra_trains}편 긴급 투입")
    print(f"  • 혼잡도 완화 효과: {peak_18.current_congestion_pct}% -> {peak_18.optimal_congestion_pct}% (목표 혼잡도 달성)")
    print(f"  • 기관사 브리핑: {peak_18.driver_guidance.braking_alert}")
    print(f"  • 정차 허용 시간(Dwell Time): {peak_18.driver_guidance.dwell_time_limit_sec}초 | 타력주행 개시율: {peak_18.driver_guidance.coasting_start_ratio_pct}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
