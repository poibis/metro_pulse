"""Data loading and preprocessing module for Metro Pulse."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import numpy as np
import pandas as pd

from metro_pulse.core.config import settings

STANDARD_TIME_SLOTS = [
    ("06시이전", "06시이전"),
    ("06-07시간대", "06-07"),
    ("07-08시간대", "07-08"),
    ("08-09시간대", "08-09"),
    ("09-10시간대", "09-10"),
    ("10-11시간대", "10-11"),
    ("11-12시간대", "11-12"),
    ("12-13시간대", "12-13"),
    ("13-14시간대", "13-14"),
    ("14-15시간대", "14-15"),
    ("15-16시간대", "15-16"),
    ("16-17시간대", "16-17"),
    ("17-18시간대", "17-18"),
    ("18-19시간대", "18-19"),
    ("19-20시간대", "19-20"),
    ("20-21시간대", "20-21"),
    ("21-22시간대", "21-22"),
    ("22-23시간대", "22-23"),
    ("23-24시간대", "23-24"),
    ("24시이후", "24시이후"),
]

MAJOR_STATIONS = [
    "강남", "잠실", "홍대입구", "신도림", "신림", "구로디지털단지",
    "역삼", "삼성", "을지로입구", "시청", "서울역", "종로3가", "고속터미널"
]


def clean_station_name(name: str) -> str:
    """역명의 괄호 부역명 제거 (예: '잠실(송파구청)' -> '잠실')"""
    return re.sub(r"\(.*?\)", "", str(name)).strip()


def parse_travel_sec(time_str: str) -> int:
    """MM:SS 형식을 초(sec)로 변환."""
    try:
        parts = str(time_str).strip().split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        return 90
    except Exception:
        return 90


class MetroDataLoader:
    _instance: Optional["MetroDataLoader"] = None

    def __init__(self):
        self.station_baselines: Dict[str, Dict[str, float]] = {}  # f"{line}_{station}": {time_label: avg_pax}
        self.distance_map: Dict[str, Dict[str, Tuple[float, int]]] = {}  # f"{line}_{st}": (dist_km, sec)
        self.lines: List[str] = []
        self.stations_by_line: Dict[str, List[str]] = {}
        self.loaded = False

    @classmethod
    def get_instance(cls) -> "MetroDataLoader":
        if cls._instance is None:
            cls._instance = MetroDataLoader()
        return cls._instance

    def load_data(self) -> None:
        """CSV 2종을 파싱하여 메모리에 캐싱."""
        if self.loaded:
            return

        self._load_distance_csv()
        self._load_hourly_csv()
        self._sort_stations()
        self.loaded = True

    def _sort_stations(self) -> None:
        """주요 환승역/과밀역이 목록 상단에 오도록 정렬."""
        for line, st_list in self.stations_by_line.items():
            major_in_line = [s for s in MAJOR_STATIONS if s in st_list]
            others = [s for s in st_list if s not in MAJOR_STATIONS]
            self.stations_by_line[line] = major_in_line + others

    def _load_distance_csv(self) -> None:
        path = settings.distance_csv_path
        if not path.exists():
            return

        try:
            df = pd.read_csv(path, encoding="cp949")
        except Exception:
            df = pd.read_csv(path, encoding="utf-8")

        for _, row in df.iterrows():
            line_raw = str(row["호선"]).strip()
            line = f"{line_raw}호선" if not line_raw.endswith("호선") else line_raw
            raw_station = str(row["역명"]).strip()
            station = clean_station_name(raw_station)
            dist_km = float(row.get("역간거리(km)", 1.2) or 1.2)
            travel_sec = parse_travel_sec(row.get("소요시간", "01:30"))
            if travel_sec <= 0:
                travel_sec = 120
            if dist_km <= 0:
                dist_km = 1.2

            if line not in self.stations_by_line:
                self.stations_by_line[line] = []
            if station not in self.stations_by_line[line]:
                self.stations_by_line[line].append(station)

            self.distance_map[f"{line}_{station}"] = (dist_km, travel_sec)
            self.distance_map[f"{line}_{raw_station}"] = (dist_km, travel_sec)

        self.lines = sorted(list(self.stations_by_line.keys()))

    def _load_hourly_csv(self) -> None:
        path = settings.hourly_csv_path
        if not path.exists():
            return

        try:
            df = pd.read_csv(path, encoding="cp949")
        except Exception:
            df = pd.read_csv(path, encoding="utf-8")

        df_board = df[df["승하차구분"] == "승차"]
        raw_cols = [src for src, _ in STANDARD_TIME_SLOTS if src in df_board.columns]
        target_labels = [dst for src, dst in STANDARD_TIME_SLOTS if src in df_board.columns]

        grouped = df_board.groupby(["호선", "역명"])[raw_cols].mean().round(1).reset_index()

        for _, row in grouped.iterrows():
            line = str(row["호선"]).strip()
            raw_station = str(row["역명"]).strip()
            station = clean_station_name(raw_station)

            hourly_dict = {}
            for raw_col, label in zip(raw_cols, target_labels):
                hourly_dict[label] = float(row[raw_col])

            # 원본 역명 및 정제된 역명 둘 다 저장
            self.station_baselines[f"{line}_{raw_station}"] = hourly_dict
            self.station_baselines[f"{line}_{station}"] = hourly_dict

            if line not in self.stations_by_line:
                self.stations_by_line[line] = []
            if station not in self.stations_by_line[line]:
                self.stations_by_line[line].append(station)

        self.lines = sorted(list(self.stations_by_line.keys()))

    def get_station_baseline(self, line: str, station: str) -> Dict[str, float]:
        """역사 기준 시간대별 일평균 승차 인원 반환 (정확 매칭 + 퍼지 매칭)."""
        clean_st = clean_station_name(station)
        
        # 1. 정확 매칭
        key = f"{line}_{clean_st}"
        if key in self.station_baselines:
            return self.station_baselines[key]
            
        key_raw = f"{line}_{station}"
        if key_raw in self.station_baselines:
            return self.station_baselines[key_raw]

        # 2. 호선 일치 부분 매칭
        for k, v in self.station_baselines.items():
            if k.startswith(f"{line}_") and (clean_st in k or k.endswith(clean_st)):
                return v

        # 3. 타 호선 동일 역명 매칭
        for k, v in self.station_baselines.items():
            if clean_st in k:
                return v

        # 기본 폴백: 출퇴근 피크 곡선
        return {
            label: 4500.0 if "08-09" in label or "18-19" in label else 1200.0
            for _, label in STANDARD_TIME_SLOTS
        }

    def get_section_info(self, line: str, dep: str, arr: Optional[str]) -> Tuple[float, int]:
        """두 역사 간의 거리(km)와 소요시간(초) 반환."""
        clean_dep = clean_station_name(dep)
        clean_arr = clean_station_name(arr) if arr else None

        key_arr = f"{line}_{clean_arr}" if clean_arr else None
        key_dep = f"{line}_{clean_dep}"

        if key_arr and key_arr in self.distance_map:
            return self.distance_map[key_arr]
        if key_dep in self.distance_map:
            return self.distance_map[key_dep]

        return (1.2, 120)

    def generate_training_dataset(self, num_samples_per_station: int = 12) -> pd.DataFrame:
        """LightGBM 학습을 위한 시나리오 증강 데이터셋 생성."""
        self.load_data()

        # CSV 컬럼 불일치 등으로 station_baselines가 비어 있을 경우 기본 샘플 데이터 생성
        if not self.station_baselines:
            print(">> [Warning] CSV에서 승하차 기준 데이터를 찾지 못해 기본 역 데이터셋을 생성합니다.")
            base_curve = {
                label: (6500.0 if "08-09" in label else 9500.0 if "18-19" in label else 2400.0)
                for _, label in STANDARD_TIME_SLOTS
            }
            for st in MAJOR_STATIONS:
                self.station_baselines[f"2호선_{st}"] = base_curve.copy()
                if "2호선" not in self.stations_by_line:
                    self.stations_by_line["2호선"] = []
                if st not in self.stations_by_line["2호선"]:
                    self.stations_by_line["2호선"].append(st)
            self.lines = sorted(list(self.stations_by_line.keys()))

        rows = []
        scenario_types = [
            (0, "weekday", 1.0),
            (1, "weekend", 0.85),
            (2, "rain", 1.15),
            (3, "event", 1.35),
        ]

        line_code_map = {f"{i}호선": i for i in range(1, 10)}
        np.random.seed(42)

        for key, hourly_dict in self.station_baselines.items():
            parts = key.split("_", 1)
            line = parts[0]
            station = parts[1]
            line_code = line_code_map.get(line, 2)
            station_hash = abs(hash(station)) % 1000

            for time_idx, (_, label) in enumerate(STANDARD_TIME_SLOTS):
                base_pax = hourly_dict.get(label, 500.0)

                for sc_code, _, sc_mult in scenario_types:
                    for _ in range(max(1, num_samples_per_station // len(scenario_types))):
                        weather_coeff = 1.0
                        event_coeff = 1.0

                        if sc_code == 2:  # rain
                            weather_coeff = round(np.random.uniform(1.08, 1.25), 2)
                        elif sc_code == 3:  # event
                            event_coeff = round(np.random.uniform(1.15, 1.50), 2)
                            weather_coeff = round(np.random.uniform(0.95, 1.10), 2)
                        elif sc_code == 1:  # weekend
                            weather_coeff = round(np.random.uniform(0.95, 1.05), 2)

                        noise = np.random.normal(1.0, 0.03)
                        predicted = base_pax * sc_mult * weather_coeff * event_coeff * noise
                        predicted = max(10.0, round(predicted, 1))

                        rows.append({
                            "station_id": station_hash,
                            "line_code": line_code,
                            "time_idx": time_idx,
                            "base_pax": round(base_pax, 1),
                            "scenario_code": sc_code,
                            "weather_coeff": weather_coeff,
                            "event_coeff": event_coeff,
                            "predicted_pax": predicted,
                        })

        return pd.DataFrame(rows)

