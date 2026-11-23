# -*- coding: utf-8 -*-
"""
月尺度 Z1-Z5 独立区间原始精度验证（剔除负值月份版）

功能：
1. 读取五个干流站逐月水量（亿 m3）CSV；
2. 按“相邻控制站水量相减”构建 Z1-Z5 逐月区间水量；
3. 将区间水量换算为区间径流深 mm；
4. 检查并输出相减为负的区间水量记录；
5. 将负区间水量对应的 zone-month OBS 从干净表、合并表、精度评估和后续 QM 输入中剔除；
6. 读取 Z1-Z5 区间 TC 月径流深；
7. 合并 OBS 与 TC，计算区间原始精度指标：
   - 所有月份整体评估；
   - 1-12 月分月评估。

重要口径：
- OBS 区间径流深不能直接用相邻站径流深相减，必须先用水量相减，再除以区间面积。
- 负区间水量不参与任何精度指标和后续 QM 建模。
- 负区间水量仍单独输出 raw 表、负值明细和负值汇总，用于论文/汇报中说明剔除依据。
- 宽表中，如果某个 zone 在某个 year-month 为负值，则该 zone 的 OBS 水量和径流深置为空值；同一 year-month 的其他 zone 若有效则保留。
"""

from __future__ import annotations

from pathlib import Path
from functools import reduce
from typing import Dict, Iterable, Optional
import numpy as np
import pandas as pd

from areas import ZONE_AREAS_KM2, validate_areas

# =============================================================================
# 1. 路径配置
# =============================================================================

WORKFLOW_DIR = Path(__file__).resolve().parent
INPUT_DIR = WORKFLOW_DIR / "inputs"
OUTPUT_DIR = WORKFLOW_DIR / "outputs"

OBS_DIR = INPUT_DIR / "observed_volume" / "mainstream"
OBS_FLOW_PATHS: Dict[str, Path] = {
    "pingshan": OBS_DIR / "obs_pingshan_monthly_flow_2007-2024.csv",
    "cuntan":   OBS_DIR / "obs_cuntan_monthly_flow_2006-2024.csv",
    "yichang":  OBS_DIR / "obs_yichang_monthly_flow_2006-2024.csv",
    "hankou":   OBS_DIR / "obs_hankou_monthly_flow_1959-1979-2006-2024.csv",
    "datong":   OBS_DIR / "obs_datong_monthly_flow_1956-1985-2006-2024.csv",
}

TC_ZONE_PATH = INPUT_DIR / "terraclimate" / "TC_monthly_ro_Z1_Z5_1958_2024.csv"

OUT_DIR = OUTPUT_DIR / "01_raw_zones"

# 本项目 TC 统一验证时期
START_YEAR = 1958
END_YEAR = 2024

# 固定口径：负区间水量样本不进入指标计算和后续 QM 建模输入表
# 注意：这里不是删除整行 year-month，而是只剔除发生负值的 zone-month 样本；
# 同一月份其他分区若为正值，仍正常参与对应分区评估。
REMOVE_NEGATIVE_INTERVALS = True

# =============================================================================
# 2. 区间定义（在 areas.py 中输入实际面积）
# =============================================================================

# 区间水量构建：downstream - upstream；Z1 只有 downstream
ZONE_DEFINITIONS = {
    "Z1": {"downstream": "pingshan", "upstream": None,       "formula": "pingshan"},
    "Z2": {"downstream": "cuntan",   "upstream": "pingshan", "formula": "cuntan - pingshan"},
    "Z3": {"downstream": "yichang",  "upstream": "cuntan",   "formula": "yichang - cuntan"},
    "Z4": {"downstream": "hankou",   "upstream": "yichang",  "formula": "hankou - yichang"},
    "Z5": {"downstream": "datong",   "upstream": "hankou",   "formula": "datong - hankou"},
}

ZONE_ORDER = ["Z1", "Z2", "Z3", "Z4", "Z5"]
STATION_ORDER = ["pingshan", "cuntan", "yichang", "hankou", "datong"]

# =============================================================================
# 3. 工具函数
# =============================================================================

def read_csv_auto(path: Path) -> pd.DataFrame:
    """自动尝试常见编码读取 CSV。"""
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")

    last_err: Optional[Exception] = None
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"无法读取 CSV：{path}\n最后一次错误：{last_err}")


def standardize_year_month(df: pd.DataFrame) -> pd.DataFrame:
    """确保 year、month 为整数，并生成 YYYY-MM 格式 date。"""
    if "year" not in df.columns or "month" not in df.columns:
        raise ValueError("输入表必须包含 year 和 month 字段。")

    out = df.copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["month"] = pd.to_numeric(out["month"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["year", "month"]).copy()
    out["year"] = out["year"].astype(int)
    out["month"] = out["month"].astype(int)

    bad_month = ~out["month"].between(1, 12)
    if bad_month.any():
        raise ValueError("存在 month 不在 1-12 范围内的记录，请检查原始数据。")

    out["date"] = out["year"].astype(str) + "-" + out["month"].astype(str).str.zfill(2)
    return out


def find_volume_col(df: pd.DataFrame, station: str) -> str:
    """识别某站水量字段，单位应为 亿m3。"""
    preferred = [
        f"{station}_volume_亿m3",
        f"{station}_volume_1e8m3",
        f"{station}_volume",
    ]
    for c in preferred:
        if c in df.columns:
            return c

    candidates = [c for c in df.columns if ("volume" in c.lower() and ("亿" in c or "1e8" in c.lower() or "m3" in c.lower()))]
    if len(candidates) == 1:
        return candidates[0]

    # 如果只有一个 volume 字段，也接受，但后面输出中明确重命名为 *_volume_1e8m3
    candidates = [c for c in df.columns if "volume" in c.lower()]
    if len(candidates) == 1:
        return candidates[0]

    raise ValueError(
        f"无法识别 {station} 的月水量字段。\n"
        f"请确保存在类似 {station}_volume_亿m3 的字段。\n"
        f"当前字段：{list(df.columns)}"
    )


def load_station_volume(station: str, path_str: Path) -> pd.DataFrame:
    """读取单个站点逐月水量，并标准化为 station_volume_1e8m3。"""
    df = read_csv_auto(path_str)
    df = standardize_year_month(df)
    vol_col = find_volume_col(df, station)

    out = df[["year", "month", "date", vol_col]].copy()
    out = out.rename(columns={vol_col: f"{station}_volume_1e8m3"})
    out[f"{station}_volume_1e8m3"] = pd.to_numeric(out[f"{station}_volume_1e8m3"], errors="coerce")

    out = out[(out["year"] >= START_YEAR) & (out["year"] <= END_YEAR)].copy()
    out = out.drop_duplicates(subset=["year", "month"], keep="last")
    return out


def volume_to_ro_mm(volume_1e8m3: pd.Series, area_km2: float) -> pd.Series:
    """亿 m3 → mm：RO = V * 100000 / Area。"""
    return volume_1e8m3 * 100000.0 / area_km2


def calc_metrics(obs: Iterable[float], sim: Iterable[float]) -> dict:
    """计算 R2/RMSE/MAE/Bias/PBIAS/NSE 等指标。"""
    obs_arr = np.asarray(obs, dtype=float)
    sim_arr = np.asarray(sim, dtype=float)
    mask = np.isfinite(obs_arr) & np.isfinite(sim_arr)
    obs_arr = obs_arr[mask]
    sim_arr = sim_arr[mask]

    n = int(len(obs_arr))
    if n == 0:
        return {
            "N": 0,
            "R2": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "Bias": np.nan,
            "PBIAS": np.nan,
            "NSE": np.nan,
            "OBS_mean": np.nan,
            "TC_mean": np.nan,
        }

    err = sim_arr - obs_arr
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    obs_mean = float(np.mean(obs_arr))
    sim_mean = float(np.mean(sim_arr))

    if np.isclose(np.sum(obs_arr), 0):
        pbias = np.nan
    else:
        pbias = float(100.0 * np.sum(err) / np.sum(obs_arr))

    if n >= 2 and np.std(obs_arr) > 0 and np.std(sim_arr) > 0:
        r = np.corrcoef(obs_arr, sim_arr)[0, 1]
        r2 = float(r ** 2)
    else:
        r2 = np.nan

    denom = np.sum((obs_arr - obs_mean) ** 2)
    if n >= 2 and denom > 0:
        nse = float(1.0 - np.sum(err ** 2) / denom)
    else:
        nse = np.nan

    return {
        "N": n,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "Bias": bias,
        "PBIAS": pbias,
        "NSE": nse,
        "OBS_mean": obs_mean,
        "TC_mean": sim_mean,
    }


def round_float_cols(df: pd.DataFrame, digits: int = 6) -> pd.DataFrame:
    """统一数值输出精度，避免 CSV 小数过长。"""
    out = df.copy()
    float_cols = out.select_dtypes(include=["float", "float64", "float32"]).columns
    out[float_cols] = out[float_cols].round(digits)
    return out

# =============================================================================
# 4. 主流程
# =============================================================================

def main() -> None:
    validate_areas(ZONE_AREAS_KM2)
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 4.1 读取五个干流站逐月水量
    # -------------------------------------------------------------------------
    station_tables = []
    for station in STATION_ORDER:
        station_tables.append(load_station_volume(station, OBS_FLOW_PATHS[station]))

    station_volume_wide = reduce(
        lambda left, right: pd.merge(left, right, on=["year", "month", "date"], how="outer"),
        station_tables,
    )
    station_volume_wide = station_volume_wide.sort_values(["year", "month"]).reset_index(drop=True)

    # 确保 date 由 year/month 生成，避免 outer merge 时受到源文件格式影响
    station_volume_wide["date"] = (
        station_volume_wide["year"].astype(str)
        + "-"
        + station_volume_wide["month"].astype(str).str.zfill(2)
    )

    station_volume_wide = station_volume_wide[
        ["year", "month", "date"] + [f"{s}_volume_1e8m3" for s in STATION_ORDER]
    ]
    station_volume_wide.to_csv(out_dir / "OBS_monthly_volume_mainstream_stations_wide.csv", index=False, encoding="utf-8-sig")

    # -------------------------------------------------------------------------
    # 4.2 构建 Z1-Z5 区间水量与区间径流深
    # -------------------------------------------------------------------------
    zone_raw = station_volume_wide.copy()

    negative_records = []
    coverage_records = []

    for zone in ZONE_ORDER:
        info = ZONE_DEFINITIONS[zone]
        downstream = info["downstream"]
        upstream = info["upstream"]
        formula = info["formula"]

        down_col = f"{downstream}_volume_1e8m3"
        up_col = f"{upstream}_volume_1e8m3" if upstream else None
        zone_vol_col = f"{zone}_obs_volume_1e8m3_raw"
        zone_ro_col = f"{zone}_obs_ro_mm_raw"

        if upstream is None:
            zone_raw[zone_vol_col] = zone_raw[down_col]
        else:
            zone_raw[zone_vol_col] = zone_raw[down_col] - zone_raw[up_col]

        zone_raw[zone_ro_col] = volume_to_ro_mm(zone_raw[zone_vol_col], ZONE_AREAS_KM2[zone])

        computed = zone_raw[zone_vol_col].notna()
        neg = computed & (zone_raw[zone_vol_col] < 0)

        coverage_records.append({
            "zone": zone,
            "formula": formula,
            "n_months_computed": int(computed.sum()),
            "n_negative": int(neg.sum()),
            "negative_ratio_percent": (100.0 * neg.sum() / computed.sum()) if computed.sum() > 0 else np.nan,
            "min_volume_1e8m3": zone_raw.loc[computed, zone_vol_col].min(),
            "max_volume_1e8m3": zone_raw.loc[computed, zone_vol_col].max(),
            "first_year": zone_raw.loc[computed, "year"].min() if computed.any() else np.nan,
            "last_year": zone_raw.loc[computed, "year"].max() if computed.any() else np.nan,
        })

        if neg.any():
            tmp_cols = ["year", "month", "date", zone_vol_col, zone_ro_col, down_col]
            if up_col is not None:
                tmp_cols.append(up_col)
            tmp = zone_raw.loc[neg, tmp_cols].copy()
            tmp.insert(0, "zone", zone)
            tmp.insert(1, "formula", formula)
            tmp = tmp.rename(columns={
                zone_vol_col: "zone_volume_1e8m3",
                zone_ro_col: "zone_ro_mm",
                down_col: "downstream_volume_1e8m3",
            })
            if up_col is not None:
                tmp = tmp.rename(columns={up_col: "upstream_volume_1e8m3"})
            else:
                tmp["upstream_volume_1e8m3"] = np.nan
            negative_records.append(tmp)

    # raw 表：负值保留，用于追溯
    raw_cols = ["year", "month", "date"]
    for z in ZONE_ORDER:
        raw_cols.extend([f"{z}_obs_volume_1e8m3_raw", f"{z}_obs_ro_mm_raw"])
    zone_raw_out = zone_raw[raw_cols].copy()
    round_float_cols(zone_raw_out).to_csv(
        out_dir / "OBS_monthly_volume_ro_Z1_Z5_raw_wide.csv", index=False, encoding="utf-8-sig"
    )

    # 负值明细与汇总
    if negative_records:
        negative_detail = pd.concat(negative_records, ignore_index=True)
        negative_detail = negative_detail[
            [
                "zone", "formula", "year", "month", "date",
                "downstream_volume_1e8m3", "upstream_volume_1e8m3",
                "zone_volume_1e8m3", "zone_ro_mm",
            ]
        ].sort_values(["zone", "year", "month"])
    else:
        negative_detail = pd.DataFrame(columns=[
            "zone", "formula", "year", "month", "date",
            "downstream_volume_1e8m3", "upstream_volume_1e8m3",
            "zone_volume_1e8m3", "zone_ro_mm",
        ])

    negative_summary = pd.DataFrame(coverage_records).sort_values("zone")
    round_float_cols(negative_detail).to_csv(
        out_dir / "OBS_monthly_negative_interval_records_Z1_Z5.csv", index=False, encoding="utf-8-sig"
    )
    round_float_cols(negative_summary).to_csv(
        out_dir / "OBS_monthly_negative_interval_summary_Z1_Z5.csv", index=False, encoding="utf-8-sig"
    )

    # 剔除说明表：明确哪些负值样本将从 clean OBS / merged / metrics / QM 输入中剔除。
    removed_summary = negative_summary[["zone", "formula", "n_months_computed", "n_negative", "negative_ratio_percent"]].copy()
    removed_summary = removed_summary.rename(columns={
        "n_negative": "n_removed_negative_months",
        "negative_ratio_percent": "removed_ratio_percent",
    })
    removed_summary["remove_rule"] = "zone-month OBS set to NaN when interval volume < 0"
    round_float_cols(removed_summary).to_csv(
        out_dir / "OBS_monthly_removed_negative_interval_summary_Z1_Z5.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # 4.3 生成用于验证/QM 的 OBS 区间表：负值 zone-month 已剔除
    # -------------------------------------------------------------------------
    zone_valid = zone_raw[["year", "month", "date"]].copy()
    flag_cols = []

    for z in ZONE_ORDER:
        raw_vol_col = f"{z}_obs_volume_1e8m3_raw"
        raw_ro_col = f"{z}_obs_ro_mm_raw"
        valid_vol_col = f"{z}_obs_volume_1e8m3"
        valid_ro_col = f"{z}_obs_ro_mm"
        neg_flag_col = f"{z}_negative_flag"

        zone_valid[valid_vol_col] = zone_raw[raw_vol_col]
        zone_valid[valid_ro_col] = zone_raw[raw_ro_col]
        zone_valid[neg_flag_col] = zone_raw[raw_vol_col] < 0
        zone_valid.loc[zone_raw[raw_vol_col].isna(), neg_flag_col] = False
        flag_cols.append(neg_flag_col)

        if REMOVE_NEGATIVE_INTERVALS:
            neg_mask = zone_raw[raw_vol_col] < 0
            # 剔除该 zone-month 的 OBS 样本：后续合并表和指标计算中不再参与。
            zone_valid.loc[neg_mask, [valid_vol_col, valid_ro_col]] = np.nan

    valid_cols = ["year", "month", "date"]
    for z in ZONE_ORDER:
        valid_cols.extend([f"{z}_obs_volume_1e8m3", f"{z}_obs_ro_mm", f"{z}_negative_flag"])

    clean_obs_wide = zone_valid[valid_cols].copy()
    round_float_cols(clean_obs_wide).to_csv(
        out_dir / "OBS_monthly_volume_ro_Z1_Z5_wide.csv", index=False, encoding="utf-8-sig"
    )

    # 同时输出一张长表，作为后续 zone-month 分月 QM 的直接输入检查表。
    clean_long_records = []
    for z in ZONE_ORDER:
        tmp = zone_valid[["year", "month", "date", f"{z}_obs_volume_1e8m3", f"{z}_obs_ro_mm", f"{z}_negative_flag"]].copy()
        tmp.insert(0, "zone", z)
        tmp = tmp.rename(columns={
            f"{z}_obs_volume_1e8m3": "obs_volume_1e8m3",
            f"{z}_obs_ro_mm": "obs_ro_mm",
            f"{z}_negative_flag": "negative_flag_raw",
        })
        tmp = tmp[tmp["obs_ro_mm"].notna()].copy()
        clean_long_records.append(tmp)

    obs_clean_long = pd.concat(clean_long_records, ignore_index=True) if clean_long_records else pd.DataFrame()
    round_float_cols(obs_clean_long).to_csv(
        out_dir / "OBS_monthly_volume_ro_Z1_Z5_clean_long_for_QM.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # 4.4 读取 TC 区间逐月径流深并合并
    # -------------------------------------------------------------------------
    tc = read_csv_auto(TC_ZONE_PATH)
    tc = standardize_year_month(tc)
    tc = tc[(tc["year"] >= START_YEAR) & (tc["year"] <= END_YEAR)].copy()

    missing_tc_cols = [z for z in ZONE_ORDER if z not in tc.columns]
    if missing_tc_cols:
        raise ValueError(f"TC 区间表缺少字段：{missing_tc_cols}")

    tc_out = tc[["year", "month", "date"] + ZONE_ORDER].copy()
    tc_out = tc_out.rename(columns={z: f"{z}_tc_ro_mm" for z in ZONE_ORDER})

    merged = pd.merge(
        zone_valid[["year", "month", "date"] + [f"{z}_obs_ro_mm" for z in ZONE_ORDER] + flag_cols],
        tc_out,
        on=["year", "month", "date"],
        how="outer",
    ).sort_values(["year", "month"]).reset_index(drop=True)

    # 调整列顺序：Z1_obs, Z1_tc, Z1_negative_flag, ...
    merged_cols = ["year", "month", "date"]
    for z in ZONE_ORDER:
        merged_cols.extend([f"{z}_obs_ro_mm", f"{z}_tc_ro_mm", f"{z}_negative_flag"])
    merged = merged[merged_cols]

    round_float_cols(merged).to_csv(
        out_dir / "monthly_raw_TC_vs_OBS_merged_wide_Z1_Z5.csv", index=False, encoding="utf-8-sig"
    )

    # -------------------------------------------------------------------------
    # 4.5 计算所有月份整体指标
    # -------------------------------------------------------------------------
    overall_rows = []
    for z in ZONE_ORDER:
        m = calc_metrics(merged[f"{z}_obs_ro_mm"], merged[f"{z}_tc_ro_mm"])
        m.update({
            "zone": z,
            "area_km2": ZONE_AREAS_KM2[z],
            "negative_removed_from_metrics": REMOVE_NEGATIVE_INTERVALS,
        })
        overall_rows.append(m)

    overall_metrics = pd.DataFrame(overall_rows)
    overall_metrics = overall_metrics[
        [
            "zone", "area_km2", "N", "R2", "RMSE", "MAE", "Bias", "PBIAS", "NSE",
            "OBS_mean", "TC_mean", "negative_removed_from_metrics",
        ]
    ]
    round_float_cols(overall_metrics).to_csv(
        out_dir / "monthly_raw_TC_vs_OBS_metrics_Z1_Z5_all_months_wide.csv", index=False, encoding="utf-8-sig"
    )

    # -------------------------------------------------------------------------
    # 4.6 计算 1-12 月分月指标
    # -------------------------------------------------------------------------
    by_month_rows = []
    for z in ZONE_ORDER:
        for month in range(1, 13):
            sub = merged[merged["month"] == month]
            m = calc_metrics(sub[f"{z}_obs_ro_mm"], sub[f"{z}_tc_ro_mm"])
            m.update({
                "zone": z,
                "month": month,
                "area_km2": ZONE_AREAS_KM2[z],
                "negative_removed_from_metrics": REMOVE_NEGATIVE_INTERVALS,
            })
            by_month_rows.append(m)

    by_month_metrics = pd.DataFrame(by_month_rows)
    by_month_metrics = by_month_metrics[
        [
            "zone", "month", "area_km2", "N", "R2", "RMSE", "MAE", "Bias", "PBIAS", "NSE",
            "OBS_mean", "TC_mean", "negative_removed_from_metrics",
        ]
    ]
    round_float_cols(by_month_metrics).to_csv(
        out_dir / "monthly_raw_TC_vs_OBS_metrics_Z1_Z5_by_calendar_month_wide.csv", index=False, encoding="utf-8-sig"
    )

    # -------------------------------------------------------------------------
    # 4.7 样本覆盖度表：每个区间每个月有效样本数、负值样本数
    # -------------------------------------------------------------------------
    coverage_rows = []
    for z in ZONE_ORDER:
        raw_vol_col = f"{z}_obs_volume_1e8m3_raw"
        valid_ro_col = f"{z}_obs_ro_mm"
        tc_col = f"{z}_tc_ro_mm"
        for month in range(1, 13):
            raw_sub = zone_raw[zone_raw["month"] == month]
            mer_sub = merged[merged["month"] == month]
            raw_n = int(raw_sub[raw_vol_col].notna().sum())
            neg_n = int((raw_sub[raw_vol_col] < 0).sum())
            obs_valid_n = int(mer_sub[valid_ro_col].notna().sum())
            tc_valid_n = int(mer_sub[tc_col].notna().sum())
            pair_valid_n = int((mer_sub[valid_ro_col].notna() & mer_sub[tc_col].notna()).sum())
            coverage_rows.append({
                "zone": z,
                "month": month,
                "raw_obs_months": raw_n,
                "negative_months": neg_n,
                "obs_valid_months_after_negative_rule": obs_valid_n,
                "tc_valid_months": tc_valid_n,
                "paired_valid_months_for_metrics": pair_valid_n,
            })

    coverage_by_month = pd.DataFrame(coverage_rows)
    coverage_by_month.to_csv(
        out_dir / "monthly_raw_TC_vs_OBS_sample_coverage_Z1_Z5_by_calendar_month_wide.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # 4.8 控制台摘要
    # -------------------------------------------------------------------------
    print("\n完成：月尺度 Z1-Z5 区间原始验证（负值月份已剔除）。")
    print(f"输出目录：{out_dir}")
    print("\n负区间水量汇总：")
    print(round_float_cols(negative_summary).to_string(index=False))
    print("\n总体指标：")
    print(round_float_cols(overall_metrics, 3).to_string(index=False))


if __name__ == "__main__":
    main()
