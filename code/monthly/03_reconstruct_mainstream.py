# -*- coding: utf-8 -*-
"""
月尺度：Z1-Z5 分区分月 QM 校正结果回算五个干流控制站，并与实测逐月径流深验证

核心逻辑：
1. 读取 Z1-Z5 逐月 Raw TC 与 QM 校正结果；
2. 按面积加权回算五个干流完整控制流域：
   屏山 = Z1
   寸滩 = Z1 + Z2
   宜昌 = Z1 + Z2 + Z3
   汉口 = Z1 + Z2 + Z3 + Z4
   大通 = Z1 + Z2 + Z3 + Z4 + Z5
3. 读取五个干流站逐月 OBS 径流深；
4. 在 station + year + month 上合并 OBS / Raw / QM；
5. 统一限定 2007-2024 年核心验证期，并输出全部月份总体指标和 1-12 月分月指标。

注意：
- 回算使用“分区径流深 × 分区面积”的水量守恒思想，再除以控制站面积。
- 不做汛期/枯水期分组统计。
- 不对 OBS 缺测月份插补。
- 指标计算限定为 2007-2024 年，与月尺度 QM 训练和验证期保持一致。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from areas import STATION_AREAS_KM2 as STATION_AREAS
from areas import ZONE_AREAS_KM2 as ZONE_AREAS
from areas import validate_areas

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# 1. 路径设置
# ============================================================

WORKFLOW_DIR = Path(__file__).resolve().parent
INPUT_DIR = WORKFLOW_DIR / "inputs"
OUTPUT_DIR = WORKFLOW_DIR / "outputs"

ZONE_QM_WIDE_PATH = (
    OUTPUT_DIR / "02_zone_month_qm" / "02_corrected_runoff"
    / "monthly_TC_QM_corrected_Z1_Z5_1958_2024_wide.csv"
)

OBS_DIR = INPUT_DIR / "observed_runoff" / "mainstream"
OBS_PATHS: Dict[str, Path] = {
    "pingshan": OBS_DIR / "obs_pingshan_monthly_ro_2007_2024.csv",
    "cuntan": OBS_DIR / "obs_cuntan_monthly_ro_2006-2024.csv",
    "yichang": OBS_DIR / "obs_yichang_monthly_ro_2006_2024.csv",
    "hankou": OBS_DIR / "obs_hankou_monthly_ro_2006_2024.csv",
    "datong": OBS_DIR / "obs_datong_monthly_2006_2024.csv",
}

# 输出目录
OUT_DIR = OUTPUT_DIR / "03_mainstream_reconstruction"

# ============================================================
# 2. 空间嵌套关系（在 areas.py 中输入实际面积）
# ============================================================

ZONES: List[str] = ["Z1", "Z2", "Z3", "Z4", "Z5"]
STATIONS: List[str] = ["pingshan", "cuntan", "yichang", "hankou", "datong"]

STATION_CN = {
    "pingshan": "屏山",
    "cuntan": "寸滩",
    "yichang": "宜昌",
    "hankou": "汉口",
    "datong": "大通",
}

# 控制站由哪些独立分区面积加权组成
STATION_ZONE_MAP = {
    "pingshan": ["Z1"],
    "cuntan": ["Z1", "Z2"],
    "yichang": ["Z1", "Z2", "Z3"],
    "hankou": ["Z1", "Z2", "Z3", "Z4"],
    "datong": ["Z1", "Z2", "Z3", "Z4", "Z5"],
}

# 核心验证期：与月尺度分区分月 QM 训练 / LOOCV 时段一致
VALID_START_YEAR = 2007
VALID_END_YEAR = 2024

# 是否额外输出所有中间 long 表
SAVE_LONG_TABLES = True

# ============================================================
# 3. 基础工具函数
# ============================================================

def read_csv_smart(path: str) -> pd.DataFrame:
    """自动尝试常见编码读取 CSV。"""
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"无法读取文件编码：{path}; last_error={last_err}")


def clean_numeric(series: pd.Series) -> pd.Series:
    """将可能带逗号、空格的文本数字转为 float。"""
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .replace({"nan": np.nan, "None": np.nan, "": np.nan}),
        errors="coerce",
    )


def parse_chinese_year_month(value) -> Tuple[Optional[int], Optional[int]]:
    """解析类似 '1959年1月' 的年月。"""
    if pd.isna(value):
        return None, None
    s = str(value).strip()
    m = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月", s)
    if m:
        return int(m.group(1)), int(m.group(2))

    dt = pd.to_datetime(s, errors="coerce")
    if pd.isna(dt):
        return None, None
    return int(dt.year), int(dt.month)


def parse_datong_month(value) -> Tuple[Optional[int], Optional[int]]:
    """
    解析大通站月份字段。

    兼容两类格式：
    1. 中文年月：2006年1月、2024年12月；
    2. 英文缩写：Jan-58、Feb-06、Mar-2024。

    之前大通站读取为 0 条的主要原因是：
    当前 CSV 的 data 字段为“2006年1月”这类中文年月，
    而旧函数只优先解析 Jan-58 这类英文月份格式。
    """
    if pd.isna(value):
        return None, None

    s = str(value).strip()

    # 先解析中文年月：2006年1月
    m_cn = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月", s)
    if m_cn:
        return int(m_cn.group(1)), int(m_cn.group(2))

    # 再解析英文月份：Jan-58 / Feb-06 / Mar-2024
    month_map = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4,
        "may": 5, "jun": 6, "jul": 7, "aug": 8,
        "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }

    m_en = re.search(r"([A-Za-z]{3})\s*[-_/ ]\s*(\d{2,4})", s)
    if m_en:
        mon = month_map[m_en.group(1).lower()]
        yy = int(m_en.group(2))
        if yy < 100:
            year = 2000 + yy if yy <= 24 else 1900 + yy
        else:
            year = yy
        return year, mon

    # 最后尝试 pandas 常规日期解析
    dt = pd.to_datetime(s, errors="coerce")
    if pd.isna(dt):
        return None, None

    return int(dt.year), int(dt.month)


def make_date_col(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(
        df["year"].astype("Int64").astype(str) + "-" +
        df["month"].astype("Int64").astype(str).str.zfill(2) + "-01",
        errors="coerce",
    )
    return df


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# ============================================================
# 4. 指标函数
# ============================================================

def calc_metrics(obs, sim) -> Dict[str, float]:
    """计算 R²、RMSE、MAE、Bias、PBIAS、NSE 等指标。"""
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[mask]
    sim = sim[mask]
    n = len(obs)

    out = {
        "N": int(n),
        "OBS_mean": np.nan,
        "SIM_mean": np.nan,
        "R2": np.nan,
        "RMSE": np.nan,
        "MAE": np.nan,
        "Bias": np.nan,
        "PBIAS": np.nan,
        "NSE": np.nan,
    }
    if n == 0:
        return out

    err = sim - obs
    out["OBS_mean"] = float(np.mean(obs))
    out["SIM_mean"] = float(np.mean(sim))
    out["RMSE"] = float(np.sqrt(np.mean(err ** 2)))
    out["MAE"] = float(np.mean(np.abs(err)))
    out["Bias"] = float(np.mean(err))

    obs_sum = float(np.sum(obs))
    if abs(obs_sum) > 1e-12:
        out["PBIAS"] = float(100.0 * np.sum(err) / obs_sum)

    if n >= 2 and np.std(obs) > 1e-12 and np.std(sim) > 1e-12:
        r = np.corrcoef(obs, sim)[0, 1]
        out["R2"] = float(r ** 2)

    denom = float(np.sum((obs - np.mean(obs)) ** 2))
    if denom > 1e-12:
        out["NSE"] = float(1.0 - np.sum((sim - obs) ** 2) / denom)

    return out


def metrics_for_group(df: pd.DataFrame, obs_col: str, sim_col: str) -> Dict[str, float]:
    return calc_metrics(df[obs_col].values, df[sim_col].values)


def filter_validation_period(df: pd.DataFrame) -> pd.DataFrame:
    """限定核心验证期，避免长序列 OBS 文件使指标期超出 2007-2024。"""
    out = df.copy()
    out["year"] = clean_numeric(out["year"]).astype("Int64")
    out = out[(out["year"] >= VALID_START_YEAR) & (out["year"] <= VALID_END_YEAR)].copy()
    return out.reset_index(drop=True)

# ============================================================
# 5. OBS 读取与标准化
# ============================================================

def read_obs_one_station(station: str, path: str) -> pd.DataFrame:
    """读取一个站点 OBS 月径流深，统一为 station/year/month/date/obs_ro_mm。"""
    df = read_csv_smart(path)
    df.columns = [str(c).strip() for c in df.columns]

    if station == "cuntan":
        # 字段：year/month/cuntan_ro_mm
        ro_col = "cuntan_ro_mm" if "cuntan_ro_mm" in df.columns else None
        if ro_col is None:
            candidates = [c for c in df.columns if "ro" in c.lower() and "mm" in c.lower()]
            if not candidates:
                raise ValueError(f"{station} 未找到径流深字段：{path}")
            ro_col = candidates[0]
        out = pd.DataFrame({
            "year": clean_numeric(df["year"]).astype("Int64"),
            "month": clean_numeric(df["month"]).astype("Int64"),
            "station": station,
            "obs_ro_mm": clean_numeric(df[ro_col]),
        })

    elif station == "datong":
        # 大通站字段：station / data / obs_ro_monthly / datong_volume_亿m3
        # data 可能为中文年月（如 2006年1月），也可能为英文缩写（如 Jan-58）
        date_col = "data"
        ro_col = "obs_ro_monthly"

        if date_col not in df.columns:
            raise ValueError(f"大通站未找到 data 字段：{path}")

        ym = df[date_col].apply(parse_datong_month)

        # 为保持面积口径一致，若存在月水量字段，则优先用水量按最终大通控制面积重算径流深。
        # 换算关系：OBS_ro_mm = volume_亿m3 × 100000 / area_km2
        volume_candidates = [
            "datong_volume_亿m3",
            "datong_volume",
            "volume_亿m3",
            "volume",
        ]
        volume_col = next((c for c in volume_candidates if c in df.columns), None)

        if volume_col is not None:
            obs_ro = clean_numeric(df[volume_col]) * 100000.0 / STATION_AREAS["datong"]
        else:
            if ro_col not in df.columns:
                candidates = [c for c in df.columns if "ro" in c.lower()]
                if not candidates:
                    raise ValueError(f"大通站未找到 obs_ro_monthly 或水量字段：{path}")
                ro_col = candidates[0]
            obs_ro = clean_numeric(df[ro_col])

        out = pd.DataFrame({
            "year": [v[0] for v in ym],
            "month": [v[1] for v in ym],
            "station": station,
            "obs_ro_mm": obs_ro,
        })
        out["year"] = out["year"].astype("Int64")
        out["month"] = out["month"].astype("Int64")

    else:
        # 汉口、屏山、宜昌：字段 date 类似 1959年1月，径流深 obs_ro_mm
        date_col = "date"
        ro_col = "obs_ro_mm"
        if date_col not in df.columns:
            raise ValueError(f"{station} 未找到 date 字段：{path}")
        if ro_col not in df.columns:
            candidates = [c for c in df.columns if "ro" in c.lower() and "mm" in c.lower()]
            if not candidates:
                raise ValueError(f"{station} 未找到 obs_ro_mm 字段：{path}")
            ro_col = candidates[0]
        ym = df[date_col].apply(parse_chinese_year_month)
        out = pd.DataFrame({
            "year": [v[0] for v in ym],
            "month": [v[1] for v in ym],
            "station": station,
            "obs_ro_mm": clean_numeric(df[ro_col]),
        })
        out["year"] = out["year"].astype("Int64")
        out["month"] = out["month"].astype("Int64")

    out = out.dropna(subset=["year", "month"])
    out = out[(out["month"] >= 1) & (out["month"] <= 12)]
    out = make_date_col(out)
    out = out[["station", "year", "month", "date", "obs_ro_mm"]]
    out = out.sort_values(["station", "year", "month"]).reset_index(drop=True)

    # 同一站点同一年月重复时取均值
    dup_n = out.duplicated(subset=["station", "year", "month"]).sum()
    if dup_n > 0:
        print(f"[提示] {station} 存在 {dup_n} 条重复年月记录，将按均值合并。")
        out = out.groupby(["station", "year", "month"], as_index=False)["obs_ro_mm"].mean()
        out = make_date_col(out)
        out = out[["station", "year", "month", "date", "obs_ro_mm"]]

    print(f"[OBS] {station}: {len(out)} 条记录，{out['year'].min()}-{out['year'].max()}")
    return out


def read_all_obs() -> pd.DataFrame:
    frames = []
    for station in STATIONS:
        frames.append(read_obs_one_station(station, OBS_PATHS[station]))
    return pd.concat(frames, ignore_index=True).sort_values(["station", "year", "month"]).reset_index(drop=True)

# ============================================================
# 6. 回算函数
# ============================================================

def build_area_closure_check() -> pd.DataFrame:
    """检查五个干流控制站面积是否等于对应 Z1-Z5 分区面积和。"""
    rows = []
    for station in STATIONS:
        zones = STATION_ZONE_MAP[station]
        zone_area_sum = float(sum(ZONE_AREAS[z] for z in zones))
        station_area = float(STATION_AREAS[station])
        rows.append({
            "station": station,
            "station_cn": STATION_CN.get(station, station),
            "zones": ";".join(zones),
            "station_area_km2": station_area,
            "sum_zone_area_km2": zone_area_sum,
            "area_diff_km2": zone_area_sum - station_area,
            "area_diff_pct": (zone_area_sum - station_area) / station_area * 100.0 if station_area > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def read_zone_qm_wide(path: str) -> pd.DataFrame:
    df = read_csv_smart(path)
    df.columns = [str(c).strip() for c in df.columns]

    if "year" not in df.columns or "month" not in df.columns:
        raise ValueError("Z1-Z5 校正结果表必须包含 year 和 month 字段。")

    df["year"] = clean_numeric(df["year"]).astype("Int64")
    df["month"] = clean_numeric(df["month"]).astype("Int64")
    df = df.dropna(subset=["year", "month"])
    df = df[(df["month"] >= 1) & (df["month"] <= 12)]

    required_cols = []
    for z in ZONES:
        required_cols += [f"{z}_tc_raw", f"{z}_tc_qm"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Z1-Z5 校正结果表缺少字段：{missing}")

    for c in required_cols:
        df[c] = clean_numeric(df[c])

    df = make_date_col(df)
    df = df.sort_values(["year", "month"]).reset_index(drop=True)
    print(f"[Zone QM] 读取 {len(df)} 条 year-month 记录。年份范围：{df['year'].min()}-{df['year'].max()}")
    return df


def backcalc_mainstream_from_zones(zone_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    从 Z1-Z5 raw/qm 区间径流深回算五个干流控制站。
    返回：wide 表、long 表。
    """
    out = zone_df[["year", "month", "date"]].copy()

    for station in STATIONS:
        zones = STATION_ZONE_MAP[station]
        station_area = STATION_AREAS[station]

        raw_sum = np.zeros(len(zone_df), dtype=float)
        qm_sum = np.zeros(len(zone_df), dtype=float)

        # 如果某个分区值是 NaN，则加权结果应该变为 NaN。
        raw_valid = np.ones(len(zone_df), dtype=bool)
        qm_valid = np.ones(len(zone_df), dtype=bool)

        for z in zones:
            area = ZONE_AREAS[z]
            raw_val = zone_df[f"{z}_tc_raw"].to_numpy(dtype=float)
            qm_val = zone_df[f"{z}_tc_qm"].to_numpy(dtype=float)

            raw_valid &= np.isfinite(raw_val)
            qm_valid &= np.isfinite(qm_val)
            raw_sum += np.where(np.isfinite(raw_val), raw_val * area, 0.0)
            qm_sum += np.where(np.isfinite(qm_val), qm_val * area, 0.0)

        raw_back = raw_sum / station_area
        qm_back = qm_sum / station_area
        raw_back[~raw_valid] = np.nan
        qm_back[~qm_valid] = np.nan

        out[f"{station}_tc_raw_backcalc"] = raw_back
        out[f"{station}_tc_qm_backcalc"] = qm_back

    # long 表
    long_rows = []
    for _, row in out.iterrows():
        for station in STATIONS:
            long_rows.append({
                "station": station,
                "station_cn": STATION_CN[station],
                "year": int(row["year"]),
                "month": int(row["month"]),
                "date": row["date"],
                "tc_raw_backcalc": row[f"{station}_tc_raw_backcalc"],
                "tc_qm_backcalc": row[f"{station}_tc_qm_backcalc"],
            })
    long_df = pd.DataFrame(long_rows)
    long_df = long_df.sort_values(["station", "year", "month"]).reset_index(drop=True)

    return out, long_df

# ============================================================
# 7. OBS / 回算结果合并与指标计算
# ============================================================

def merge_obs_backcalc(obs_long: pd.DataFrame, backcalc_long: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(
        obs_long[["station", "year", "month", "obs_ro_mm"]],
        backcalc_long[["station", "station_cn", "year", "month", "tc_raw_backcalc", "tc_qm_backcalc"]],
        on=["station", "year", "month"],
        how="inner",
    )
    merged = make_date_col(merged)
    merged = merged[[
        "station", "station_cn", "year", "month", "date",
        "obs_ro_mm", "tc_raw_backcalc", "tc_qm_backcalc",
    ]]
    merged = merged.sort_values(["station", "year", "month"]).reset_index(drop=True)
    return merged


def make_merged_wide(merged_long: pd.DataFrame) -> pd.DataFrame:
    base = merged_long[["year", "month", "date"]].drop_duplicates().sort_values(["year", "month"])
    wide = base.copy()

    for station in STATIONS:
        sub = merged_long[merged_long["station"] == station][[
            "year", "month", "obs_ro_mm", "tc_raw_backcalc", "tc_qm_backcalc"
        ]].copy()
        sub = sub.rename(columns={
            "obs_ro_mm": f"{station}_obs",
            "tc_raw_backcalc": f"{station}_tc_raw_backcalc",
            "tc_qm_backcalc": f"{station}_tc_qm_backcalc",
        })
        wide = pd.merge(wide, sub, on=["year", "month"], how="left")

    wide = wide.sort_values(["year", "month"]).reset_index(drop=True)
    return wide


def calc_all_months_metrics(merged_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for station, sub in merged_long.groupby("station"):
        station_cn = STATION_CN.get(station, station)

        raw_m = metrics_for_group(sub, "obs_ro_mm", "tc_raw_backcalc")
        qm_m = metrics_for_group(sub, "obs_ro_mm", "tc_qm_backcalc")

        row = {
            "station": station,
            "station_cn": station_cn,
            "N": raw_m["N"],
            "OBS_mean": raw_m["OBS_mean"],
            "Raw_mean": raw_m["SIM_mean"],
            "QM_mean": qm_m["SIM_mean"],
            "Raw_R2": raw_m["R2"],
            "QM_R2": qm_m["R2"],
            "Raw_RMSE": raw_m["RMSE"],
            "QM_RMSE": qm_m["RMSE"],
            "Delta_RMSE_QM_minus_Raw": qm_m["RMSE"] - raw_m["RMSE"] if np.isfinite(qm_m["RMSE"]) and np.isfinite(raw_m["RMSE"]) else np.nan,
            "RMSE_improve_percent": (raw_m["RMSE"] - qm_m["RMSE"]) / raw_m["RMSE"] * 100 if np.isfinite(raw_m["RMSE"]) and raw_m["RMSE"] > 0 and np.isfinite(qm_m["RMSE"]) else np.nan,
            "Raw_MAE": raw_m["MAE"],
            "QM_MAE": qm_m["MAE"],
            "Raw_Bias": raw_m["Bias"],
            "QM_Bias": qm_m["Bias"],
            "Raw_PBIAS": raw_m["PBIAS"],
            "QM_PBIAS": qm_m["PBIAS"],
            "Raw_NSE": raw_m["NSE"],
            "QM_NSE": qm_m["NSE"],
            "Delta_NSE_QM_minus_Raw": qm_m["NSE"] - raw_m["NSE"] if np.isfinite(qm_m["NSE"]) and np.isfinite(raw_m["NSE"]) else np.nan,
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    out["station_order"] = out["station"].map({s: i for i, s in enumerate(STATIONS)})
    out = out.sort_values("station_order").drop(columns="station_order").reset_index(drop=True)
    return out


def calc_by_month_metrics(merged_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (station, month), sub in merged_long.groupby(["station", "month"]):
        station_cn = STATION_CN.get(station, station)
        raw_m = metrics_for_group(sub, "obs_ro_mm", "tc_raw_backcalc")
        qm_m = metrics_for_group(sub, "obs_ro_mm", "tc_qm_backcalc")

        rows.append({
            "station": station,
            "station_cn": station_cn,
            "month": int(month),
            "N": raw_m["N"],
            "OBS_mean": raw_m["OBS_mean"],
            "Raw_mean": raw_m["SIM_mean"],
            "QM_mean": qm_m["SIM_mean"],
            "Raw_R2": raw_m["R2"],
            "QM_R2": qm_m["R2"],
            "Raw_RMSE": raw_m["RMSE"],
            "QM_RMSE": qm_m["RMSE"],
            "Delta_RMSE_QM_minus_Raw": qm_m["RMSE"] - raw_m["RMSE"] if np.isfinite(qm_m["RMSE"]) and np.isfinite(raw_m["RMSE"]) else np.nan,
            "RMSE_improve_percent": (raw_m["RMSE"] - qm_m["RMSE"]) / raw_m["RMSE"] * 100 if np.isfinite(raw_m["RMSE"]) and raw_m["RMSE"] > 0 and np.isfinite(qm_m["RMSE"]) else np.nan,
            "Raw_MAE": raw_m["MAE"],
            "QM_MAE": qm_m["MAE"],
            "Raw_Bias": raw_m["Bias"],
            "QM_Bias": qm_m["Bias"],
            "Raw_PBIAS": raw_m["PBIAS"],
            "QM_PBIAS": qm_m["PBIAS"],
            "Raw_NSE": raw_m["NSE"],
            "QM_NSE": qm_m["NSE"],
            "Delta_NSE_QM_minus_Raw": qm_m["NSE"] - raw_m["NSE"] if np.isfinite(qm_m["NSE"]) and np.isfinite(raw_m["NSE"]) else np.nan,
        })

    out = pd.DataFrame(rows)
    out["station_order"] = out["station"].map({s: i for i, s in enumerate(STATIONS)})
    out = out.sort_values(["station_order", "month"]).drop(columns="station_order").reset_index(drop=True)
    return out


def calc_sample_coverage(merged_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for station in STATIONS:
        sub = merged_long[merged_long["station"] == station]
        for month in range(1, 13):
            sm = sub[sub["month"] == month]
            years = sorted(sm["year"].dropna().astype(int).unique().tolist())
            rows.append({
                "station": station,
                "station_cn": STATION_CN[station],
                "month": month,
                "N": len(years),
                "year_min": min(years) if years else np.nan,
                "year_max": max(years) if years else np.nan,
                "years": ";".join(map(str, years)) if years else "",
            })
    return pd.DataFrame(rows)

# ============================================================
# 8. 主程序
# ============================================================

def main() -> None:
    validate_areas(STATION_AREAS)
    validate_areas(ZONE_AREAS)
    out_dir = OUT_DIR
    ensure_dir(out_dir)

    print("=" * 80)
    print("月尺度：Z1-Z5 校正结果回算五个干流控制站")
    print("输出目录：", out_dir)
    print("=" * 80)

    # 1) 读取分区校正结果
    zone_qm = read_zone_qm_wide(ZONE_QM_WIDE_PATH)

    # 2) 回算五个干流控制站
    mainstream_wide, mainstream_long = backcalc_mainstream_from_zones(zone_qm)

    # 3) 读取 OBS
    obs_long = read_all_obs()

    # 4) 合并 OBS / Raw / QM
    merged_long_all = merge_obs_backcalc(obs_long, mainstream_long)
    merged_long = filter_validation_period(merged_long_all)
    merged_wide = make_merged_wide(merged_long)

    # 5) 指标：仅使用 2007-2024 年核心验证期
    metrics_all = calc_all_months_metrics(merged_long)
    metrics_month = calc_by_month_metrics(merged_long)
    coverage = calc_sample_coverage(merged_long)
    area_check = build_area_closure_check()

    # 6) 输出
    mainstream_wide.to_csv(
        os.path.join(out_dir, "monthly_TC_raw_recalc_and_QM_corrected_mainstream_1958_2024_wide.csv"),
        index=False, encoding="utf-8-sig"
    )
    # 兼容旧文件名：该表同时包含 RawRecalc 与 QM 回算结果。
    mainstream_wide.to_csv(
        os.path.join(out_dir, "monthly_TC_QM_corrected_mainstream_1958_2024_wide.csv"),
        index=False, encoding="utf-8-sig"
    )
    if SAVE_LONG_TABLES:
        mainstream_long.to_csv(
            os.path.join(out_dir, "monthly_TC_QM_corrected_mainstream_1958_2024_long.csv"),
            index=False, encoding="utf-8-sig"
        )

    obs_long_core = filter_validation_period(obs_long)
    obs_wide = obs_long_core.pivot_table(
        index=["year", "month", "date"],
        columns="station",
        values="obs_ro_mm",
        aggfunc="mean",
    ).reset_index()
    obs_wide.columns.name = None
    obs_wide = obs_wide.rename(columns={st: f"{st}_obs" for st in STATIONS if st in obs_wide.columns})
    obs_wide.to_csv(
        os.path.join(out_dir, "OBS_monthly_ro_mainstream_wide.csv"),
        index=False, encoding="utf-8-sig"
    )
    if SAVE_LONG_TABLES:
        obs_long_core.to_csv(
            os.path.join(out_dir, "OBS_monthly_ro_mainstream_long.csv"),
            index=False, encoding="utf-8-sig"
        )
        obs_long.to_csv(
            os.path.join(out_dir, "OBS_monthly_ro_mainstream_all_available_long.csv"),
            index=False, encoding="utf-8-sig"
        )

    merged_wide.to_csv(
        os.path.join(out_dir, "monthly_TC_QM_corrected_mainstream_vs_OBS_merged_wide.csv"),
        index=False, encoding="utf-8-sig"
    )
    if SAVE_LONG_TABLES:
        merged_long.to_csv(
            os.path.join(out_dir, "monthly_TC_QM_corrected_mainstream_vs_OBS_merged_long.csv"),
            index=False, encoding="utf-8-sig"
        )
        merged_long_all.to_csv(
            os.path.join(out_dir, "monthly_TC_QM_corrected_mainstream_vs_OBS_merged_all_available_long.csv"),
            index=False, encoding="utf-8-sig"
        )

    area_check.to_csv(
        os.path.join(out_dir, "monthly_mainstream_backcalc_area_closure_check.csv"),
        index=False, encoding="utf-8-sig"
    )

    metrics_all.to_csv(
        os.path.join(out_dir, "monthly_TC_QM_corrected_mainstream_metrics_all_months_wide.csv"),
        index=False, encoding="utf-8-sig"
    )
    metrics_month.to_csv(
        os.path.join(out_dir, "monthly_TC_QM_corrected_mainstream_metrics_by_calendar_month_wide.csv"),
        index=False, encoding="utf-8-sig"
    )
    coverage.to_csv(
        os.path.join(out_dir, "monthly_TC_QM_corrected_mainstream_sample_coverage_by_calendar_month_wide.csv"),
        index=False, encoding="utf-8-sig"
    )

    # 7) 打印摘要
    print("\n[完成] 输出文件：")
    for f in sorted(os.listdir(out_dir)):
        print("  -", f)

    print("\n[总体指标摘要]")
    show_cols = [
        "station", "station_cn", "N",
        "Raw_NSE", "QM_NSE", "Delta_NSE_QM_minus_Raw",
        "Raw_PBIAS", "QM_PBIAS",
        "Raw_RMSE", "QM_RMSE", "RMSE_improve_percent",
    ]
    print(metrics_all[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
