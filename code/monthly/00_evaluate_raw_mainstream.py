# -*- coding: utf-8 -*-
"""
月尺度：五个干流站完整控制流域 原始 TC vs OBS 精度验证

只做两层评估：
1) 所有逐月样本总体评估
2) 按自然月 1-12 月分别评估

不做：汛期/枯水期分组统计、Z1-Z5 独立分区、QM 校正。
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import math
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# 1. 路径设置
# ============================================================

WORKFLOW_DIR = Path(__file__).resolve().parent
INPUT_DIR = WORKFLOW_DIR / "inputs"
OUTPUT_DIR = WORKFLOW_DIR / "outputs"

OBS_DIR = INPUT_DIR / "observed_runoff" / "mainstream"
OBS_PATHS: Dict[str, Path] = {
    "pingshan": OBS_DIR / "obs_pingshan_monthly_ro_2007_2024.csv",
    "cuntan":   OBS_DIR / "obs_cuntan_monthly_ro_2006-2024.csv",
    "yichang":  OBS_DIR / "obs_yichang_monthly_ro_2006_2024.csv",
    "hankou":   OBS_DIR / "obs_hankou_monthly_ro_1959_1979_2006_2024.csv",
    "datong":   OBS_DIR / "obs_datong_monthly_1958_1985_2006_2011_2024.csv",
}

TC_PATH = INPUT_DIR / "terraclimate" / "TC_monthly_ro_1958_2024.csv"

OUT_DIR = OUTPUT_DIR / "00_raw_mainstream"

STATIONS = ["pingshan", "cuntan", "yichang", "hankou", "datong"]

STATION_CN = {
    "pingshan": "屏山",
    "cuntan": "寸滩",
    "yichang": "宜昌",
    "hankou": "汉口",
    "datong": "大通",
}

# ============================================================
# 2. 基础函数
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
    raise UnicodeDecodeError(f"无法读取文件编码：{path}", b"", 0, 1, str(last_err))


def parse_chinese_year_month(value) -> Tuple[Optional[int], Optional[int]]:
    """解析类似 '1959年1月' 的年月。"""
    if pd.isna(value):
        return None, None
    s = str(value).strip()
    m = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月", s)
    if not m:
        # 兜底：尝试常规日期解析
        dt = pd.to_datetime(s, errors="coerce")
        if pd.isna(dt):
            return None, None
        return int(dt.year), int(dt.month)
    return int(m.group(1)), int(m.group(2))


def parse_datong_month(value) -> Tuple[Optional[int], Optional[int]]:
    """
    解析大通数据中的 'Jan-58'、'Feb-06'、'Mar-2024' 等格式。
    规则：两位年份 yy <= 24 视为 2000+yy，否则视为 1900+yy。
    """
    if pd.isna(value):
        return None, None

    s = str(value).strip()
    month_map = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4,
        "may": 5, "jun": 6, "jul": 7, "aug": 8,
        "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }

    m = re.search(r"([A-Za-z]{3})\s*[-_/ ]\s*(\d{2,4})", s)
    if not m:
        dt = pd.to_datetime(s, errors="coerce")
        if pd.isna(dt):
            return None, None
        return int(dt.year), int(dt.month)

    mon = month_map[m.group(1).lower()]
    yy = int(m.group(2))
    if yy < 100:
        year = 2000 + yy if yy <= 24 else 1900 + yy
    else:
        year = yy
    return year, mon


def clean_numeric(series: pd.Series) -> pd.Series:
    """把可能带空格、逗号的数字列转成 float。"""
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .replace({"nan": np.nan, "None": np.nan, "": np.nan}),
        errors="coerce"
    )


def make_date_col(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(
        df["year"].astype("Int64").astype(str) + "-" +
        df["month"].astype("Int64").astype(str).str.zfill(2) + "-01",
        errors="coerce"
    )
    return df

# ============================================================
# 3. OBS 读取与标准化
# ============================================================

def read_obs_one_station(station: str, path: str) -> pd.DataFrame:
    """读取一个站点 OBS 月径流深，统一为 year/month/station/obs_ro_mm。"""
    df = read_csv_smart(path)
    df.columns = [str(c).strip() for c in df.columns]

    if station == "cuntan":
        # 已有 year/month/cuntan_ro_mm
        year_col = "year"
        month_col = "month"
        ro_col = "cuntan_ro_mm"

        out = pd.DataFrame({
            "year": clean_numeric(df[year_col]).astype("Int64"),
            "month": clean_numeric(df[month_col]).astype("Int64"),
            "station": station,
            "obs_ro_mm": clean_numeric(df[ro_col]),
        })

    elif station == "datong":
        # 字段：station, data, obs_ro_monthly；data 形如 Jan-58
        date_col = "data"
        ro_col = "obs_ro_monthly"

        ym = df[date_col].apply(parse_datong_month)
        out = pd.DataFrame({
            "year": [v[0] for v in ym],
            "month": [v[1] for v in ym],
            "station": station,
            "obs_ro_mm": clean_numeric(df[ro_col]),
        })
        out["year"] = out["year"].astype("Int64")
        out["month"] = out["month"].astype("Int64")

    else:
        # 汉口、屏山、宜昌：字段 date 形如 1959年1月，径流深 obs_ro_mm
        date_col = "date"
        ro_col = "obs_ro_mm"

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

    # 同一站点同一年月如有重复，取平均并提示。
    dup_n = out.duplicated(subset=["station", "year", "month"]).sum()
    if dup_n > 0:
        print(f"[提示] {station} 存在 {dup_n} 条重复年月记录，将按均值合并。")
        out = (
            out.groupby(["station", "year", "month"], as_index=False)["obs_ro_mm"]
            .mean()
        )
        out = make_date_col(out)
        out = out[["station", "year", "month", "date", "obs_ro_mm"]]

    return out


def read_all_obs() -> pd.DataFrame:
    frames = []
    for st in STATIONS:
        print(f"读取 OBS: {st} -> {OBS_PATHS[st]}")
        frames.append(read_obs_one_station(st, OBS_PATHS[st]))
    obs_long = pd.concat(frames, ignore_index=True)
    return obs_long.sort_values(["station", "year", "month"]).reset_index(drop=True)

# ============================================================
# 4. TC 读取与标准化
# ============================================================

def read_tc_monthly(path: str) -> pd.DataFrame:
    df = read_csv_smart(path)
    df.columns = [str(c).strip() for c in df.columns]

    need_cols = ["year", "month"] + STATIONS
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"TC 文件缺少必要字段：{missing}")

    df["year"] = clean_numeric(df["year"]).astype("Int64")
    df["month"] = clean_numeric(df["month"]).astype("Int64")

    for st in STATIONS:
        df[st] = clean_numeric(df[st])

    tc_long = df[["year", "month"] + STATIONS].melt(
        id_vars=["year", "month"],
        value_vars=STATIONS,
        var_name="station",
        value_name="tc_ro_mm"
    )
    tc_long = make_date_col(tc_long)
    tc_long = tc_long[["station", "year", "month", "date", "tc_ro_mm"]]
    return tc_long.sort_values(["station", "year", "month"]).reset_index(drop=True)

# ============================================================
# 5. 指标计算
# ============================================================

def calc_metrics(obs: pd.Series, sim: pd.Series) -> Dict[str, float]:
    """计算 R2/RMSE/MAE/Bias/PBIAS/NSE 等。"""
    data = pd.DataFrame({"obs": obs, "sim": sim}).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(data)

    if n == 0:
        return {
            "N": 0, "R2": np.nan, "RMSE": np.nan, "MAE": np.nan,
            "Bias": np.nan, "PBIAS": np.nan, "NSE": np.nan,
            "OBS_mean": np.nan, "TC_mean": np.nan,
        }

    o = data["obs"].to_numpy(dtype=float)
    s = data["sim"].to_numpy(dtype=float)
    err = s - o

    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))

    obs_sum = float(np.sum(o))
    pbias = float(100.0 * np.sum(err) / obs_sum) if abs(obs_sum) > 1e-12 else np.nan

    denom = float(np.sum((o - np.mean(o)) ** 2))
    nse = float(1.0 - np.sum(err ** 2) / denom) if denom > 1e-12 else np.nan

    if n >= 2 and np.std(o, ddof=1) > 1e-12 and np.std(s, ddof=1) > 1e-12:
        r = float(np.corrcoef(o, s)[0, 1])
        r2 = r ** 2
    else:
        r2 = np.nan

    return {
        "N": int(n),
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "Bias": bias,
        "PBIAS": pbias,
        "NSE": nse,
        "OBS_mean": float(np.mean(o)),
        "TC_mean": float(np.mean(s)),
    }


def build_all_months_metrics(merged_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for st in STATIONS:
        sub = merged_long[merged_long["station"] == st].dropna(subset=["obs_ro_mm", "tc_ro_mm"])
        m = calc_metrics(sub["obs_ro_mm"], sub["tc_ro_mm"])
        if len(sub) > 0:
            m["start_year"] = int(sub["year"].min())
            m["end_year"] = int(sub["year"].max())
            m["start_date"] = sub["date"].min().strftime("%Y-%m")
            m["end_date"] = sub["date"].max().strftime("%Y-%m")
        else:
            m["start_year"] = np.nan
            m["end_year"] = np.nan
            m["start_date"] = np.nan
            m["end_date"] = np.nan
        m["station"] = st
        m["station_cn"] = STATION_CN[st]
        rows.append(m)

    cols = [
        "station", "station_cn", "N", "start_year", "end_year", "start_date", "end_date",
        "R2", "RMSE", "MAE", "Bias", "PBIAS", "NSE", "OBS_mean", "TC_mean"
    ]
    out = pd.DataFrame(rows)[cols]
    return out


def build_by_calendar_month_metrics_wide(merged_long: pd.DataFrame) -> pd.DataFrame:
    """输出宽表：每个站点一行，每个月的每个指标展开为列。"""
    metric_names = ["N", "R2", "RMSE", "MAE", "Bias", "PBIAS", "NSE", "OBS_mean", "TC_mean"]
    rows = []

    for st in STATIONS:
        row = {"station": st, "station_cn": STATION_CN[st]}
        st_df = merged_long[merged_long["station"] == st]
        for mon in range(1, 13):
            sub = st_df[st_df["month"] == mon].dropna(subset=["obs_ro_mm", "tc_ro_mm"])
            m = calc_metrics(sub["obs_ro_mm"], sub["tc_ro_mm"])
            for metric in metric_names:
                row[f"M{mon:02d}_{metric}"] = m[metric]
        rows.append(row)

    return pd.DataFrame(rows)

# ============================================================
# 6. 宽表输出
# ============================================================

def obs_long_to_wide(obs_long: pd.DataFrame) -> pd.DataFrame:
    obs_wide = obs_long.pivot_table(
        index=["year", "month", "date"],
        columns="station",
        values="obs_ro_mm",
        aggfunc="mean"
    ).reset_index()
    obs_wide.columns.name = None
    rename = {st: f"{st}_obs" for st in STATIONS if st in obs_wide.columns}
    obs_wide = obs_wide.rename(columns=rename)
    order_cols = ["year", "month", "date"] + [f"{st}_obs" for st in STATIONS if f"{st}_obs" in obs_wide.columns]
    return obs_wide[order_cols].sort_values(["year", "month"]).reset_index(drop=True)


def merged_long_to_wide(merged_long: pd.DataFrame) -> pd.DataFrame:
    obs_wide = merged_long.pivot_table(
        index=["year", "month", "date"],
        columns="station",
        values="obs_ro_mm",
        aggfunc="mean"
    ).reset_index()
    obs_wide.columns.name = None
    obs_wide = obs_wide.rename(columns={st: f"{st}_obs" for st in STATIONS if st in obs_wide.columns})

    tc_wide = merged_long.pivot_table(
        index=["year", "month", "date"],
        columns="station",
        values="tc_ro_mm",
        aggfunc="mean"
    ).reset_index()
    tc_wide.columns.name = None
    tc_wide = tc_wide.rename(columns={st: f"{st}_tc" for st in STATIONS if st in tc_wide.columns})

    out = pd.merge(obs_wide, tc_wide, on=["year", "month", "date"], how="outer")

    order_cols = ["year", "month", "date"]
    for st in STATIONS:
        for suffix in ["obs", "tc"]:
            col = f"{st}_{suffix}"
            if col in out.columns:
                order_cols.append(col)

    return out[order_cols].sort_values(["year", "month"]).reset_index(drop=True)


def write_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.6f")
    print(f"已输出：{path}")

# ============================================================
# 7. 主程序
# ============================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) 读取并标准化 OBS/TC
    obs_long = read_all_obs()
    tc_long = read_tc_monthly(TC_PATH)

    # 2) 合并：保留 TC 全时期，OBS 缺测留空；指标计算时只使用 OBS/TC 同时存在的样本
    merged_long = pd.merge(
        tc_long,
        obs_long[["station", "year", "month", "obs_ro_mm"]],
        on=["station", "year", "month"],
        how="left"
    )
    merged_long = merged_long[["station", "year", "month", "date", "obs_ro_mm", "tc_ro_mm"]]
    merged_long = merged_long.sort_values(["station", "year", "month"]).reset_index(drop=True)

    # 3) 生成宽表
    obs_wide = obs_long_to_wide(obs_long)
    merged_wide = merged_long_to_wide(merged_long)

    # 4) 两层指标
    metrics_all = build_all_months_metrics(merged_long)
    metrics_by_month_wide = build_by_calendar_month_metrics_wide(merged_long)

    # 5) 样本覆盖检查表，可用于核对各站有效月份数量
    coverage_rows = []
    for st in STATIONS:
        obs_sub = obs_long[obs_long["station"] == st]
        valid_sub = merged_long[(merged_long["station"] == st) & merged_long["obs_ro_mm"].notna() & merged_long["tc_ro_mm"].notna()]
        coverage_rows.append({
            "station": st,
            "station_cn": STATION_CN[st],
            "obs_month_count": int(obs_sub["obs_ro_mm"].notna().sum()),
            "valid_eval_month_count": int(len(valid_sub)),
            "obs_start_date": obs_sub["date"].min().strftime("%Y-%m") if len(obs_sub) else np.nan,
            "obs_end_date": obs_sub["date"].max().strftime("%Y-%m") if len(obs_sub) else np.nan,
            "valid_start_date": valid_sub["date"].min().strftime("%Y-%m") if len(valid_sub) else np.nan,
            "valid_end_date": valid_sub["date"].max().strftime("%Y-%m") if len(valid_sub) else np.nan,
        })
    coverage = pd.DataFrame(coverage_rows)

    # 6) 输出
    write_csv(obs_wide, os.path.join(OUT_DIR, "OBS_monthly_ro_mainstream_wide.csv"))
    write_csv(merged_wide, os.path.join(OUT_DIR, "monthly_raw_TC_vs_OBS_merged_mainstream_wide.csv"))
    write_csv(metrics_all, os.path.join(OUT_DIR, "monthly_raw_TC_vs_OBS_metrics_mainstream_all_months_wide.csv"))
    write_csv(metrics_by_month_wide, os.path.join(OUT_DIR, "monthly_raw_TC_vs_OBS_metrics_mainstream_by_calendar_month_wide.csv"))
    write_csv(coverage, os.path.join(OUT_DIR, "monthly_raw_TC_vs_OBS_sample_coverage_mainstream_wide.csv"))

    # 7) 控制台摘要
    print("\n========== 月尺度原始验证完成 ==========")
    print("输出文件夹：", OUT_DIR)
    print("\n总体指标预览：")
    print(metrics_all.to_string(index=False))


if __name__ == "__main__":
    main()
