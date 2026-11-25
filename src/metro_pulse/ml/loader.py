from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import zlib
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


def stable_station_id(station: str) -> int:
    """프로세스 재시작/OS 환경과 무관하게 항상 동일한 0~999 정수 ID 반환."""
    return zlib.crc32(station.encode("utf-8")) % 1000


def parse_travel_sec(time_str: str) -> int:
    """MM:SS 형식을 초(sec)로 변환."""
    try:
        parts = str(time_str).strip().split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        return 90
    except Exception:
        return 90


def read_csv_safe(path: Path) -> pd.DataFrame:
    """한글 공공데이터 CSV 인코딩 자동 감지 로드."""
    for enc in ("cp949", "utf-8-sig", "euc-kr", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            continue
    raise ValueError(f"CSV 파일을 읽을 수 없습니다: {path}")


class MetroDataLoader:
    _instance: Optional["MetroDataLoader"] = None

    def __init__(self):
        self.station_baselines: Dict[str, Dict[str, float]] = {}
        self.distance_map: Dict[str, Dict[str, Tuple[float, int]]] = {}
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
        print(f">> [MetroDataLoader] 로드 완료: {len(self.lines)}개 노선, {len(self.station_baselines)}개 역 기준 데이터")
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
            print(f">> [Warning] 역간거리 CSV 없음: {path}")
            return

        df = read_csv_safe(path)
        df.columns = [str(c).strip() for c in df.columns]

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
            raise FileNotFoundError(f"승하차 CSV 파일이 존재하지 않습니다: {path}")

        df = read_csv_safe(path)
        df.columns = [str(c).strip() for c in df.columns]

        df_board = df[df["승하차구분"].astype(str).str.strip() == "승차"].copy()
        raw_cols = [src for src, _ in STANDARD_TIME_SLOTS if src in df_board.columns]
        target_labels = [dst for src, dst in STANDARD_TIME_SLOTS if src in df_board.columns]

        for col in raw_cols:
            df_board[col] = pd.to_numeric(df_board[col], errors="coerce").fillna(0.0)

        grouped = df_board.groupby(["호선", "역명"])[raw_cols].mean().round(1).reset_index()

        for _, row in grouped.iterrows():
            line_raw = str(row["호선"]).strip()
            line = f"{line_raw}호선" if not line_raw.endswith("호선") else line_raw
            raw_station = str(row["역명"]).strip()
            station = clean_station_name(raw_station)

            hourly_dict = {}
            for raw_col, label in zip(raw_cols, target_labels):
                hourly_dict[label] = float(row[raw_col])

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

        key = f"{line}_{clean_st}"
        if key in self.station_baselines:
            return self.station_baselines[key]

        key_raw = f"{line}_{station}"
        if key_raw in self.station_baselines:
            return self.station_baselines[key_raw]

        for k, v in self.station_baselines.items():
            if k.startswith(f"{line}_") and (clean_st in k or k.endswith(clean_st)):
                return v

        for k, v in self.station_baselines.items():
            if clean_st in k:
                return v

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

        rows = []
        scenario_types = [
            (0, "weekday", 1.0),
            (1, "weekend", 0.85),
            (2, "rain", 1.15),
            (3, "event", 1.35),
        ]

        line_code_map = {f"{i}호선": i for i in range(1, 10)}
        np.random.seed(42)

        # 중복 학습 방지를 위해 정제된 역 키만 정렬하여 순회
        unique_keys = sorted(list(self.station_baselines.keys()))
        for key in unique_keys:
            hourly_dict = self.station_baselines[key]
            parts = key.split("_", 1)
            line = parts[0]
            station = parts[1]
            line_code = line_code_map.get(line, 2)
            station_hash = stable_station_id(station)

            for time_idx, (_, label) in enumerate(STANDARD_TIME_SLOTS):
                base_pax = hourly_dict.get(label, 500.0)

                for sc_code, _, sc_mult in scenario_types:
                    for _ in range(max(1, num_samples_per_station // len(scenario_types))):
                        weather_coeff = 1.0
                        event_coeff = 1.0

                        if sc_code == 2:
                            weather_coeff = round(np.random.uniform(1.08, 1.25), 2)
                        elif sc_code == 3:
                            event_coeff = round(np.random.uniform(1.15, 1.50), 2)
                            weather_coeff = round(np.random.uniform(0.95, 1.10), 2)
                        elif sc_code == 1:
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