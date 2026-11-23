# -*- coding: utf-8 -*-
"""
月尺度支流站点外部验证：分区-分月 QM 校正结果迁移到支流站点

功能：
1. 读取支流站点逐月 OBS 径流深；
2. 读取站点 TerraClimate 逐月 ro；
3. 读取前一步“Z1-Z5 独立分区分月 QM”最终模型参数与稠密映射曲线；
4. 按支流所属分区，将 station TC 逐月序列代入对应 zone-month QM；
5. 输出支流站点 Raw TC 与 QM TC 的逐月精度验证结果。

重要口径：
- 支流站点只用于外部验证，不参与 QM 建模；
- 支流属于哪个 Z 分区，就使用该 Z 分区对应月份的最终 QM；
- 校正是逐月进行，验证同时输出 all-months 总体指标和 1-12 月分月指标；
- 本脚本验证高场、城陵矶、湘潭、桃源、湖口和外洲六个支流站点。

依赖：pandas, numpy
"""

from __future__ import annotations

import os
import re
import math
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# 0. 用户配置区
# =============================================================================

# 若本地缺少某个 OBS 文件：
# False：直接报错，适合正式运行；
# True ：跳过缺失站点，适合只用部分文件测试格式。
ALLOW_MISSING_OBS_FILES = os.environ.get("ALLOW_MISSING_OBS_FILES", "0") == "1"

WORKFLOW_DIR = Path(__file__).resolve().parent
INPUT_DIR = WORKFLOW_DIR / "inputs"
OUTPUT_DIR = WORKFLOW_DIR / "outputs"

OUT_DIR = OUTPUT_DIR / "04_tributary_validation"
TC_STATION_MONTHLY_CSV = INPUT_DIR / "terraclimate" / "TC_monthly_ro_1958_2024.csv"

MODEL_DIR = OUTPUT_DIR / "02_zone_month_qm" / "01_final_models"
MODEL_SUMMARY_CSV = MODEL_DIR / "monthly_QM_final_model_summary_Z1_Z5_by_month_wide.csv"
DENSE_CURVE_CSV = MODEL_DIR / "monthly_QM_final_dense_mapping_curves_Z1_Z5_by_month_wide.csv"

# 支流 OBS 文件
OBS_DIR = INPUT_DIR / "observed_runoff" / "tributaries"
TRIBUTARY_OBS_FILES: Dict[str, Path] = {
    "chenglingji": OBS_DIR / "obs_chenglingji_monthly_1987_2012.csv",
    "gaochang": OBS_DIR / "obs_gaochang__monthly_1959_2018.csv",
    "hukou": OBS_DIR / "obs_hukou_monthly_2000_2010.csv",
    "taoyuan": OBS_DIR / "obs_taoyuan_monthly_1988_2012.csv",
    "waizhou": OBS_DIR / "obs_waizhou_monthly_2000_2010.csv",
    "xiangtan": OBS_DIR / "obs_xiangtan_monthly_ro_1988_2012.csv",
}

# 中文名
STATION_CN: Dict[str, str] = {
    "chenglingji": "城陵矶",
    "gaochang": "高场",
    "hukou": "湖口",
    "taoyuan": "桃源",
    "waizhou": "外洲",
    "xiangtan": "湘潭",
}

# 支流站点所属分区
STATION_ZONE: Dict[str, str] = {
    "gaochang": "Z2",
    "chenglingji": "Z4",
    "taoyuan": "Z4",
    "xiangtan": "Z4",
    "hukou": "Z5",
    "waizhou": "Z5",
}

# TC 站点列名。如果后续 TC 表字段不同，只改这里即可。
TC_STATION_COLUMNS: Dict[str, str] = {
    "chenglingji": "chenglingji",
    "gaochang": "gaochang",
    "hukou": "hukou",
    "taoyuan": "taoyuan",
    "waizhou": "waizhou",
    "xiangtan": "xiangtan",
}

# 物理下限约束
NON_NEGATIVE_QM = True


# =============================================================================
# 1. 通用工具函数
# =============================================================================

def read_csv_auto(path: Path) -> pd.DataFrame:
    """自动尝试常见编码读取 CSV。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到文件：{path}")

    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030"]
    last_error = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"读取 CSV 失败：{path}\n最后错误：{last_error}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_colname(col: str) -> str:
    """列名归一化，便于模糊匹配。"""
    return str(col).strip().replace(" ", "").replace("_", "").lower()


def find_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """从 df 中按归一化列名查找候选列。"""
    normalized_map = {normalize_colname(c): c for c in df.columns}
    for cand in candidates:
        key = normalize_colname(cand)
        if key in normalized_map:
            return normalized_map[key]
    return None


def parse_year_month_from_text(value) -> Tuple[Optional[int], Optional[int]]:
    """从 Jan-58、1959年1月、1959-01 等文本中解析 year/month。"""
    if pd.isna(value):
        return None, None
    s = str(value).strip()
    if not s:
        return None, None

    # 中文年月：1959年1月
    m = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月", s)
    if m:
        return int(m.group(1)), int(m.group(2))

    # 标准形态：1959-01, 1959/1, 1959.1
    m = re.search(r"(\d{4})[-/.年\s]+(\d{1,2})", s)
    if m:
        return int(m.group(1)), int(m.group(2))

    # Jan-58 / Jan-1958
    m = re.search(r"([A-Za-z]{3,9})[-_/\s]+(\d{2,4})", s)
    if m:
        month_name = m.group(1).lower()[:3]
        month_map = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        }
        if month_name in month_map:
            yy = int(m.group(2))
            if yy < 100:
                yy = 1900 + yy if yy >= 50 else 2000 + yy
            return yy, month_map[month_name]

    # 兜底：让 pandas 试一下
    try:
        dt = pd.to_datetime(s, errors="coerce")
        if pd.notna(dt):
            return int(dt.year), int(dt.month)
    except Exception:  # noqa: BLE001
        pass

    return None, None


def standardize_obs_monthly(df: pd.DataFrame, station: str) -> pd.DataFrame:
    """将各种格式的支流 OBS 表统一为 station/year/month/obs_ro_mm。"""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    year_col = find_column(df, ["year", "年份", "年"])
    month_col = find_column(df, ["month", "月份", "月"])

    if year_col is not None and month_col is not None:
        df["year"] = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")
        df["month"] = pd.to_numeric(df[month_col], errors="coerce").astype("Int64")
    else:
        date_col = find_column(df, ["date", "data", "time", "年月", "日期", "yearmonth"])
        if date_col is None:
            raise ValueError(
                f"{station} OBS 表无法识别 year/month，也找不到 date/data/年月列。列名：{list(df.columns)}"
            )
        parsed = df[date_col].apply(parse_year_month_from_text)
        df["year"] = [x[0] for x in parsed]
        df["month"] = [x[1] for x in parsed]
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")

    obs_col = find_column(
        df,
        [
            "obs_ro_mm", "runoff_mm", "ro_mm", "径流深", "月径流深", "runoff", "ro",
            "obs", "value", "径流深mm", "月径流深mm",
        ],
    )

    if obs_col is None:
        # 兜底：选择除 year/month/station/date 外的第一个数值列
        exclude_norm = {
            normalize_colname(x)
            for x in ["year", "month", "station", "station_cn", "站点", "date", "data", "time", "年月", "日期"]
        }
        numeric_candidates = []
        for c in df.columns:
            if normalize_colname(c) in exclude_norm:
                continue
            ser = pd.to_numeric(df[c], errors="coerce")
            if ser.notna().sum() > 0:
                numeric_candidates.append(c)
        if not numeric_candidates:
            raise ValueError(f"{station} OBS 表找不到径流深数值列。列名：{list(df.columns)}")
        obs_col = numeric_candidates[0]
        warnings.warn(f"{station}: 未识别标准 OBS 列名，自动使用数值列 {obs_col}")

    out = pd.DataFrame(
        {
            "station": station,
            "station_cn": STATION_CN.get(station, station),
            "zone": STATION_ZONE.get(station),
            "year": pd.to_numeric(df["year"], errors="coerce"),
            "month": pd.to_numeric(df["month"], errors="coerce"),
            "obs_ro_mm": pd.to_numeric(df[obs_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["year", "month"])
    out["year"] = out["year"].astype(int)
    out["month"] = out["month"].astype(int)
    out = out[(out["month"] >= 1) & (out["month"] <= 12)]
    out = out.drop_duplicates(subset=["station", "year", "month"], keep="first")
    out = out.sort_values(["station", "year", "month"]).reset_index(drop=True)
    return out


def metric_pair(obs: np.ndarray, sim: np.ndarray) -> Dict[str, float]:
    """计算单组 obs/sim 精度指标。"""
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[mask]
    sim = sim[mask]
    n = len(obs)

    result = {
        "N": n,
        "R2": np.nan,
        "RMSE": np.nan,
        "MAE": np.nan,
        "Bias": np.nan,
        "PBIAS": np.nan,
        "NSE": np.nan,
        "OBS_mean": np.nan,
        "SIM_mean": np.nan,
    }
    if n == 0:
        return result

    err = sim - obs
    result["RMSE"] = float(np.sqrt(np.mean(err ** 2)))
    result["MAE"] = float(np.mean(np.abs(err)))
    result["Bias"] = float(np.mean(err))
    result["OBS_mean"] = float(np.mean(obs))
    result["SIM_mean"] = float(np.mean(sim))

    obs_sum = float(np.sum(obs))
    if abs(obs_sum) > 1e-12:
        result["PBIAS"] = float(np.sum(err) / obs_sum * 100.0)

    if n >= 2 and np.nanstd(obs) > 0 and np.nanstd(sim) > 0:
        r = np.corrcoef(obs, sim)[0, 1]
        result["R2"] = float(r ** 2)

    denom = float(np.sum((obs - np.mean(obs)) ** 2))
    if denom > 1e-12:
        result["NSE"] = float(1.0 - np.sum(err ** 2) / denom)

    return result


def build_metrics_wide(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """针对 Raw 与 QM 同时计算指标，并输出宽表。"""
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}

        raw = metric_pair(g["obs_ro_mm"].values, g["tc_raw"].values)
        qm = metric_pair(g["obs_ro_mm"].values, g["tc_qm"].values)

        # N 使用两者共同有效样本数，一般 Raw/QM 相同
        row["N"] = int(min(raw["N"], qm["N"]))
        row["OBS_mean"] = raw["OBS_mean"]
        row["Raw_mean"] = raw["SIM_mean"]
        row["QM_mean"] = qm["SIM_mean"]

        for k in ["R2", "RMSE", "MAE", "Bias", "PBIAS", "NSE"]:
            row[f"Raw_{k}"] = raw[k]
            row[f"QM_{k}"] = qm[k]

        if np.isfinite(row.get("Raw_RMSE", np.nan)) and abs(row["Raw_RMSE"]) > 1e-12:
            row["RMSE_improve_pct"] = (row["Raw_RMSE"] - row["QM_RMSE"]) / row["Raw_RMSE"] * 100.0
        else:
            row["RMSE_improve_pct"] = np.nan

        if np.isfinite(row.get("Raw_NSE", np.nan)) and np.isfinite(row.get("QM_NSE", np.nan)):
            row["NSE_delta"] = row["QM_NSE"] - row["Raw_NSE"]
        else:
            row["NSE_delta"] = np.nan

        if np.isfinite(row.get("Raw_PBIAS", np.nan)) and np.isfinite(row.get("QM_PBIAS", np.nan)):
            row["abs_PBIAS_improve"] = abs(row["Raw_PBIAS"]) - abs(row["QM_PBIAS"])
        else:
            row["abs_PBIAS_improve"] = np.nan

        rows.append(row)

    out = pd.DataFrame(rows)
    return out.sort_values(group_cols).reset_index(drop=True)


def apply_dense_mapping_one(x: float, model: pd.Series, curve: pd.DataFrame) -> Tuple[float, str, bool]:
    """使用某个 zone-month 最终 QM 稠密曲线与尾部斜率校正单个 TC 值。"""
    if pd.isna(x) or not np.isfinite(x):
        return np.nan, "missing", False

    status = str(model.get("status", "normal"))
    if status != "normal":
        return np.nan, f"model_{status}", False

    train_min = float(model["train_min"])
    train_max = float(model["train_max"])
    y_min = float(model["y_min"])
    y_max = float(model["y_max"])
    low_slope = float(model["low_slope"])
    high_slope = float(model["high_slope"])

    if x < train_min:
        y = y_min + low_slope * (x - train_min)
        etype = "lower"
        slope_adjusted = bool(model.get("low_slope_adjusted", False))
    elif x > train_max:
        y = y_max + high_slope * (x - train_max)
        etype = "upper"
        slope_adjusted = bool(model.get("high_slope_adjusted", False))
    else:
        curve2 = curve.sort_values("x_tc")
        y = float(np.interp(x, curve2["x_tc"].values, curve2["y_obs_mapped"].values))
        etype = "none"
        slope_adjusted = False

    if NON_NEGATIVE_QM:
        y = max(y, 0.0)
    return y, etype, slope_adjusted


# =============================================================================
# 2. 数据读取与整理
# =============================================================================

def load_tributary_obs() -> pd.DataFrame:
    obs_list = []
    missing = []

    for station, path in TRIBUTARY_OBS_FILES.items():
        if not path.is_file():
            msg = f"缺少 {station} OBS 文件：{path}"
            if ALLOW_MISSING_OBS_FILES:
                warnings.warn(msg + "，已跳过。")
                missing.append(station)
                continue
            raise FileNotFoundError(msg)

        df = read_csv_auto(path)
        std = standardize_obs_monthly(df, station)
        obs_list.append(std)

    if not obs_list:
        raise RuntimeError("没有成功读取任何支流 OBS 文件。")

    obs = pd.concat(obs_list, ignore_index=True)
    obs = obs.sort_values(["station", "year", "month"]).reset_index(drop=True)
    return obs


def load_station_tc_long(stations: Iterable[str]) -> pd.DataFrame:
    df = read_csv_auto(TC_STATION_MONTHLY_CSV)
    df.columns = [str(c).strip() for c in df.columns]

    year_col = find_column(df, ["year", "年份", "年"])
    month_col = find_column(df, ["month", "月份", "月"])
    if year_col is None or month_col is None:
        raise ValueError(f"TC 站点表必须包含 year/month 列。当前列名：{list(df.columns)}")

    df["year"] = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")
    df["month"] = pd.to_numeric(df[month_col], errors="coerce").astype("Int64")

    records = []
    for station in stations:
        col = TC_STATION_COLUMNS.get(station, station)
        if col not in df.columns:
            raise ValueError(f"TC 表缺少站点列：{station} 对应列 {col}。当前列名：{list(df.columns)}")
        tmp = df[["year", "month", col]].copy()
        tmp = tmp.rename(columns={col: "tc_raw"})
        tmp["station"] = station
        tmp["station_cn"] = STATION_CN.get(station, station)
        tmp["zone"] = STATION_ZONE.get(station)
        tmp["tc_raw"] = pd.to_numeric(tmp["tc_raw"], errors="coerce")
        records.append(tmp[["station", "station_cn", "zone", "year", "month", "tc_raw"]])

    out = pd.concat(records, ignore_index=True)
    out = out.dropna(subset=["year", "month"])
    out["year"] = out["year"].astype(int)
    out["month"] = out["month"].astype(int)
    out = out.sort_values(["station", "year", "month"]).reset_index(drop=True)
    return out


def load_qm_models() -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary = read_csv_auto(MODEL_SUMMARY_CSV)
    curve = read_csv_auto(DENSE_CURVE_CSV)

    # 类型统一
    for df in [summary, curve]:
        df["month"] = pd.to_numeric(df["month"], errors="coerce").astype(int)
        df["zone"] = df["zone"].astype(str)
        if "method" in df.columns:
            df["method"] = df["method"].astype(str)
    summary["best_method"] = summary["best_method"].astype(str)

    needed_summary = {
        "zone", "month", "best_method", "N_train", "status", "sample_warning",
        "train_min", "train_max", "y_min", "y_max", "low_slope", "high_slope",
        "low_slope_adjusted", "high_slope_adjusted",
    }
    missing_summary = needed_summary - set(summary.columns)
    if missing_summary:
        raise ValueError(f"模型摘要表缺少字段：{missing_summary}")

    needed_curve = {"zone", "month", "method", "x_tc", "y_obs_mapped"}
    missing_curve = needed_curve - set(curve.columns)
    if missing_curve:
        raise ValueError(f"稠密曲线表缺少字段：{missing_curve}")

    return summary, curve


# =============================================================================
# 3. 支流 TC 应用分区分月 QM
# =============================================================================

def correct_tributary_tc(tc_long: pd.DataFrame, summary: pd.DataFrame, curve: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model_lookup = {
        (str(r["zone"]), int(r["month"])): r
        for _, r in summary.iterrows()
    }

    # 按 zone/month/method 建曲线索引
    curve_lookup: Dict[Tuple[str, int, str], pd.DataFrame] = {}
    for (z, m, method), g in curve.groupby(["zone", "month", "method"]):
        curve_lookup[(str(z), int(m), str(method))] = g.sort_values("x_tc").reset_index(drop=True)

    corrected_rows = []
    extrap_rows = []

    for _, row in tc_long.iterrows():
        station = row["station"]
        station_cn = row["station_cn"]
        zone = row["zone"]
        year = int(row["year"])
        month = int(row["month"])
        tc_raw = row["tc_raw"]

        model_key = (str(zone), month)
        if model_key not in model_lookup:
            tc_qm = np.nan
            etype = "missing_model"
            slope_adjusted = False
            best_method = np.nan
            sample_warning = np.nan
            model_status = "missing_model"
            n_train = np.nan
        else:
            model = model_lookup[model_key]
            best_method = str(model["best_method"])
            sample_warning = bool(model.get("sample_warning", False))
            model_status = str(model.get("status", "normal"))
            n_train = model.get("N_train", np.nan)

            curve_key = (str(zone), month, best_method)
            if curve_key not in curve_lookup:
                tc_qm = np.nan
                etype = "missing_curve"
                slope_adjusted = False
                model_status = "missing_curve"
            else:
                tc_qm, etype, slope_adjusted = apply_dense_mapping_one(
                    float(tc_raw) if pd.notna(tc_raw) else np.nan,
                    model,
                    curve_lookup[curve_key],
                )

        corrected_rows.append(
            {
                "station": station,
                "station_cn": station_cn,
                "zone": zone,
                "year": year,
                "month": month,
                "tc_raw": tc_raw,
                "tc_qm": tc_qm,
                "best_method": best_method,
                "N_train_zone_month": n_train,
                "extrapolation_type": etype,
                "slope_adjusted": slope_adjusted,
                "sample_warning": sample_warning,
                "model_status": model_status,
            }
        )

        if etype in ["lower", "upper", "missing", "missing_model", "missing_curve"] or slope_adjusted:
            extrap_rows.append(
                {
                    "station": station,
                    "station_cn": station_cn,
                    "zone": zone,
                    "year": year,
                    "month": month,
                    "tc_raw": tc_raw,
                    "tc_qm": tc_qm,
                    "best_method": best_method,
                    "N_train_zone_month": n_train,
                    "extrapolation_type": etype,
                    "slope_adjusted": slope_adjusted,
                    "sample_warning": sample_warning,
                    "model_status": model_status,
                }
            )

    corrected = pd.DataFrame(corrected_rows)
    extrap = pd.DataFrame(extrap_rows)
    return corrected, extrap


# =============================================================================
# 4. 输出宽表与汇总
# =============================================================================

def pivot_obs_wide(obs_long: pd.DataFrame) -> pd.DataFrame:
    wide = obs_long.pivot_table(index=["year", "month"], columns="station", values="obs_ro_mm", aggfunc="first")
    wide.columns = [f"{c}_obs" for c in wide.columns]
    wide = wide.reset_index().sort_values(["year", "month"])
    return wide


def pivot_corrected_wide(corr_long: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for value_col in ["tc_raw", "tc_qm"]:
        p = corr_long.pivot_table(index=["year", "month"], columns="station", values=value_col, aggfunc="first")
        p.columns = [f"{c}_{value_col}" for c in p.columns]
        parts.append(p)
    wide = pd.concat(parts, axis=1).reset_index().sort_values(["year", "month"])
    return wide


def pivot_merged_wide(merged_long: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for value_col in ["obs_ro_mm", "tc_raw", "tc_qm"]:
        p = merged_long.pivot_table(index=["year", "month"], columns="station", values=value_col, aggfunc="first")
        suffix = {"obs_ro_mm": "obs", "tc_raw": "tc_raw", "tc_qm": "tc_qm"}[value_col]
        p.columns = [f"{c}_{suffix}" for c in p.columns]
        parts.append(p)
    wide = pd.concat(parts, axis=1).reset_index().sort_values(["year", "month"])
    return wide


def build_extrapolation_summary(extrap: pd.DataFrame, corrected: pd.DataFrame) -> pd.DataFrame:
    total = corrected.groupby(["station", "station_cn", "zone"]).size().reset_index(name="total_months")
    if extrap.empty:
        total["extrapolation_months"] = 0
        total["lower_months"] = 0
        total["upper_months"] = 0
        total["missing_months"] = 0
        total["slope_adjusted_months"] = 0
        total["extrapolation_ratio_pct"] = 0.0
        return total

    e = extrap.copy()
    e["is_lower"] = e["extrapolation_type"].eq("lower")
    e["is_upper"] = e["extrapolation_type"].eq("upper")
    e["is_missing"] = e["extrapolation_type"].astype(str).str.contains("missing", na=False)
    summary = e.groupby(["station", "station_cn", "zone"]).agg(
        extrapolation_months=("extrapolation_type", lambda x: int(((x == "lower") | (x == "upper")).sum())),
        lower_months=("is_lower", "sum"),
        upper_months=("is_upper", "sum"),
        missing_months=("is_missing", "sum"),
        slope_adjusted_months=("slope_adjusted", "sum"),
    ).reset_index()

    out = total.merge(summary, on=["station", "station_cn", "zone"], how="left")
    for c in ["extrapolation_months", "lower_months", "upper_months", "missing_months", "slope_adjusted_months"]:
        out[c] = out[c].fillna(0).astype(int)
    out["extrapolation_ratio_pct"] = out["extrapolation_months"] / out["total_months"] * 100.0
    return out


def build_sample_coverage(obs_long: pd.DataFrame, merged_long: pd.DataFrame) -> pd.DataFrame:
    obs_cov = obs_long.groupby(["station", "station_cn", "zone", "month"]).agg(
        obs_month_count=("obs_ro_mm", lambda x: int(x.notna().sum())),
        obs_start_year=("year", "min"),
        obs_end_year=("year", "max"),
    ).reset_index()
    val_cov = merged_long.groupby(["station", "station_cn", "zone", "month"]).agg(
        validation_N=("obs_ro_mm", lambda x: int(x.notna().sum())),
    ).reset_index()
    return obs_cov.merge(val_cov, on=["station", "station_cn", "zone", "month"], how="left")


# =============================================================================
# 5. 主流程
# =============================================================================

def main() -> None:
    ensure_dir(OUT_DIR)

    print("[1/7] 读取支流 OBS...")
    obs_long = load_tributary_obs()
    stations = sorted(obs_long["station"].unique().tolist())
    print(f"  成功读取站点：{stations}")

    print("[2/7] 读取站点 TC 逐月数据...")
    tc_long = load_station_tc_long(stations)

    print("[3/7] 读取区间分月 QM 最终模型与稠密曲线...")
    model_summary, dense_curve = load_qm_models()

    print("[4/7] 将支流 TC 代入所属分区的分月 QM...")
    corrected_long, extrap_records = correct_tributary_tc(tc_long, model_summary, dense_curve)

    print("[5/7] 合并 OBS 与 Raw/QM TC...")
    merged_long = obs_long.merge(
        corrected_long,
        on=["station", "station_cn", "zone", "year", "month"],
        how="inner",
    )
    # 只保留 OBS、Raw、QM 均有效的验证样本
    merged_long_valid = merged_long.dropna(subset=["obs_ro_mm", "tc_raw", "tc_qm"]).copy()
    merged_long_valid = merged_long_valid.sort_values(["station", "year", "month"]).reset_index(drop=True)

    print("[6/7] 计算精度指标...")
    metrics_all = build_metrics_wide(merged_long_valid, ["station", "station_cn", "zone"])
    metrics_month = build_metrics_wide(merged_long_valid, ["station", "station_cn", "zone", "month"])
    coverage = build_sample_coverage(obs_long, merged_long_valid)
    extrap_summary = build_extrapolation_summary(extrap_records, corrected_long)

    print("[7/7] 输出结果...")
    # OBS
    obs_long.to_csv(OUT_DIR / "OBS_monthly_ro_tributary_stations_long.csv", index=False, encoding="utf-8-sig")
    pivot_obs_wide(obs_long).to_csv(OUT_DIR / "OBS_monthly_ro_tributary_stations_wide.csv", index=False, encoding="utf-8-sig")

    # 校正后的支流 TC
    corrected_long.to_csv(OUT_DIR / "monthly_TC_QM_corrected_tributary_1958_2024_long.csv", index=False, encoding="utf-8-sig")
    pivot_corrected_wide(corrected_long).to_csv(OUT_DIR / "monthly_TC_QM_corrected_tributary_1958_2024_wide.csv", index=False, encoding="utf-8-sig")

    # 外推记录
    extrap_records.to_csv(OUT_DIR / "monthly_QM_application_extrapolation_records_tributary_wide.csv", index=False, encoding="utf-8-sig")
    extrap_summary.to_csv(OUT_DIR / "monthly_QM_application_extrapolation_summary_tributary_wide.csv", index=False, encoding="utf-8-sig")

    # 合并验证表
    merged_long_valid.to_csv(OUT_DIR / "monthly_TC_QM_corrected_tributary_vs_OBS_merged_long.csv", index=False, encoding="utf-8-sig")
    pivot_merged_wide(merged_long_valid).to_csv(OUT_DIR / "monthly_TC_QM_corrected_tributary_vs_OBS_merged_wide.csv", index=False, encoding="utf-8-sig")

    # 指标与样本覆盖
    metrics_all.to_csv(OUT_DIR / "monthly_TC_QM_corrected_tributary_metrics_all_months_wide.csv", index=False, encoding="utf-8-sig")
    metrics_month.to_csv(OUT_DIR / "monthly_TC_QM_corrected_tributary_metrics_by_calendar_month_wide.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(OUT_DIR / "monthly_TC_QM_corrected_tributary_sample_coverage_by_calendar_month_wide.csv", index=False, encoding="utf-8-sig")

    print("\n完成。输出目录：")
    print(OUT_DIR)
    print("\n总体指标预览：")
    preview_cols = [
        "station", "station_cn", "zone", "N",
        "Raw_NSE", "QM_NSE", "Raw_PBIAS", "QM_PBIAS", "Raw_RMSE", "QM_RMSE", "RMSE_improve_pct",
    ]
    print(metrics_all[preview_cols].to_string(index=False))


if __name__ == "__main__":
    main()
