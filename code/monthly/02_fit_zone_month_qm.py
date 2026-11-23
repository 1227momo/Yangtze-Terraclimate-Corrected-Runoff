# -*- coding: utf-8 -*-
"""
月尺度 Z1-Z5 独立分区 + 分月 QM 校正完整代码

功能：
1. 读取剔除负值后的 Z1-Z5 逐月 OBS 区间径流深；
2. 读取 Z1-Z5 逐月 TerraClimate ro；
3. 对每个 zone × month 独立执行 QUANT / RQUANT / SSPLIN + LOOCV；
4. 保留距最高 NSE 不超过 0.01 的候选，再按 RMSE、|PBIAS|、|Bias| 排序；
5. 使用全部重叠样本建立最终分区分月 QM；
6. 应用到 1958-2024 全时期逐月 TC；
7. 输出校正结果、外推记录、比例系数表和校正后精度验证表。
8. 输出 LOOCV 失败原因汇总表，用于检查候选方法是否正常参与筛选。

本修正版重点：
- 修复 RQUANT 分支中平滑数组可能为 read-only 而导致端点赋值失败的问题；
- 对所有需要后续修改的 NumPy 数组显式使用 copy=True 或 .copy()；
- 增加 monthly_QM_LOOCV_failure_reason_summary_Z1_Z5_by_month_wide.csv 输出。

注意：
- 输入 OBS 应使用上一阶段已经剔除负值月份后的表；
- 本代码不再重新构建 OBS 区间，也不再处理负值相减；
- 小样本月份采用方案 A：严格分月，不扩窗，只限制候选方法；
- 6 <= N < 10 时不使用 SSPLIN；N < 6 时不建模；
- 所有输出均为 CSV 宽表/长表，编码 utf-8-sig。
"""

from __future__ import annotations

import os
from pathlib import Path
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

try:
    from scipy.interpolate import UnivariateSpline
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# =============================================================================
# 0. 路径设置：按需修改
# =============================================================================

WORKFLOW_DIR = Path(__file__).resolve().parent
INPUT_DIR = WORKFLOW_DIR / "inputs"
OUTPUT_DIR = WORKFLOW_DIR / "outputs"

OBS_ZONE_MONTHLY_CSV = OUTPUT_DIR / "01_raw_zones" / "OBS_monthly_volume_ro_Z1_Z5_wide.csv"
TC_ZONE_MONTHLY_CSV = INPUT_DIR / "terraclimate" / "TC_monthly_ro_Z1_Z5_1958_2024.csv"
OUTPUT_ROOT = OUTPUT_DIR / "02_zone_month_qm"


# =============================================================================
# 1. 全局参数
# =============================================================================

ZONES = ["Z1", "Z2", "Z3", "Z4", "Z5"]
MONTHS = list(range(1, 13))
CALIBRATION_START_YEAR = 2007
CALIBRATION_END_YEAR = 2024
BASE_METHODS = ["QUANT", "RQUANT", "SSPLIN"]

# LOOCV 筛选规则
NSE_TIE_TOL = 0.01

# 样本量保险机制：方案 A
N_NORMAL = 15      # N >= 15：正常
N_WARN = 10        # 10 <= N < 15：警告，但允许三种方法
N_MIN_MODEL = 6    # N < 6：不建模
N_MIN_SSPLIN = 10  # 6 <= N < 10：不允许 SSPLIN

# 稠密映射曲线与外推参数
N_DENSE_POINTS = 1001
TAIL_SLOPE_SPAN_RATIO_LIST = [0.05, 0.10, 0.15]
TAIL_SLOPE_MIN_POINTS = 20
SLOPE_EPS = 1e-6

# 比例系数分母保护
TC_FACTOR_EPS = 1e-6

# 是否对最终校正值施加非负约束
APPLY_NONNEGATIVE_CONSTRAINT = True

# RQUANT 平滑窗口比例：用于降低局部异常分位点影响
RQUANT_SMOOTH_WINDOW_RATIO = 0.03

# SSPLIN 平滑强度；越大越平滑。若效果不理想，可小幅调整。
SSPLIN_S_FACTOR_RATIO = 0.10


# =============================================================================
# 2. 通用工具函数
# =============================================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_csv_smart(path: str) -> pd.DataFrame:
    """自动尝试常见编码读取 CSV。"""
    last_err = None
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"无法读取 CSV：{path}\n最后一次错误：{last_err}")


def write_csv(df: pd.DataFrame, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False, encoding="utf-8-sig")


def safe_float_array(values: Any) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float, copy=True)
    return arr.copy()


def calc_metrics(obs: np.ndarray, sim: np.ndarray) -> Dict[str, float]:
    """计算 R2/RMSE/MAE/Bias/PBIAS/NSE。误差方向 sim - obs。"""
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[mask]
    sim = sim[mask]
    n = len(obs)

    out = {
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
        return out

    err = sim - obs
    out["RMSE"] = float(np.sqrt(np.mean(err ** 2)))
    out["MAE"] = float(np.mean(np.abs(err)))
    out["Bias"] = float(np.mean(err))
    out["OBS_mean"] = float(np.mean(obs))
    out["SIM_mean"] = float(np.mean(sim))

    obs_sum = np.sum(obs)
    if abs(obs_sum) > 1e-12:
        out["PBIAS"] = float(100.0 * np.sum(err) / obs_sum)

    if n >= 2 and np.nanstd(obs) > 0 and np.nanstd(sim) > 0:
        r = np.corrcoef(obs, sim)[0, 1]
        out["R2"] = float(r ** 2)

    denom = np.sum((obs - np.mean(obs)) ** 2)
    if denom > 1e-12:
        out["NSE"] = float(1.0 - np.sum(err ** 2) / denom)

    return out


def add_date_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")
    df["date"] = df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
    return df


# =============================================================================
# 3. 数据整理：OBS/TC 宽表 -> 长表
# =============================================================================

def obs_wide_to_long(obs_wide: pd.DataFrame) -> pd.DataFrame:
    """
    支持两种输入：
    1. 宽表：year, month, Z1_obs_ro_mm, ..., Z5_obs_ro_mm
    2. 长表：year, month, zone, obs_ro_mm
    """
    df = obs_wide.copy()
    df.columns = [str(c).strip() for c in df.columns]

    lower_cols = {c.lower(): c for c in df.columns}
    if {"year", "month", "zone"}.issubset(set(lower_cols.keys())):
        obs_col = None
        for cand in ["obs_ro_mm", "obs", "ro_mm", "obs_value"]:
            if cand in lower_cols:
                obs_col = lower_cols[cand]
                break
        if obs_col is None:
            raise ValueError("OBS 长表中未找到 obs_ro_mm / obs / ro_mm 字段。")

        out = df[[lower_cols["year"], lower_cols["month"], lower_cols["zone"], obs_col]].copy()
        out.columns = ["year", "month", "zone", "obs_ro_mm"]
    else:
        records = []
        for z in ZONES:
            # 优先匹配上一阶段输出字段
            possible = [
                # 常见写法 1：Z1 在前
                f"{z}_obs_ro_mm",
                f"{z}_obs_ro",
                f"{z}_ro_mm",
                f"{z}_ro",
                f"{z}_obs",
                # 常见写法 2：obs / ro 在前；你的 clean OBS 表属于 obs_ro_Z1 这种格式
                f"obs_ro_{z}",
                f"obs_ro_mm_{z}",
                f"obs_{z}_ro",
                f"obs_{z}_ro_mm",
                f"ro_{z}",
                f"ro_mm_{z}",
                # 兜底：字段名直接为 Z1/Z2/...
                z,
            ]
            value_col = None
            for c in possible:
                if c in df.columns:
                    value_col = c
                    break
            if value_col is None:
                raise ValueError(
                    f"OBS 宽表中未找到 {z} 的径流深字段，已尝试：{possible}\n"
                    f"当前 OBS 表字段为：{list(df.columns)}"
                )

            tmp = df[["year", "month", value_col]].copy()
            tmp["zone"] = z
            tmp.rename(columns={value_col: "obs_ro_mm"}, inplace=True)
            records.append(tmp[["year", "month", "zone", "obs_ro_mm"]])

        out = pd.concat(records, ignore_index=True)

    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["month"] = pd.to_numeric(out["month"], errors="coerce")
    out["obs_ro_mm"] = pd.to_numeric(out["obs_ro_mm"], errors="coerce")
    out = out.dropna(subset=["year", "month", "zone"])
    out["year"] = out["year"].astype(int)
    out["month"] = out["month"].astype(int)
    out["zone"] = out["zone"].astype(str).str.upper().str.strip()

    # 保险：负值和非有限值不进入 QM
    out.loc[out["obs_ro_mm"] < 0, "obs_ro_mm"] = np.nan
    out = out[out["zone"].isin(ZONES)].copy()
    return out.sort_values(["zone", "year", "month"]).reset_index(drop=True)


def tc_wide_to_long(tc_wide: pd.DataFrame) -> pd.DataFrame:
    """
    支持两种输入：
    1. 宽表：year, month, Z1, Z2, Z3, Z4, Z5
    2. 长表：year, month, zone, tc_ro_mm
    """
    df = tc_wide.copy()
    df.columns = [str(c).strip() for c in df.columns]

    lower_cols = {c.lower(): c for c in df.columns}
    if {"year", "month", "zone"}.issubset(set(lower_cols.keys())):
        tc_col = None
        for cand in ["tc_ro_mm", "tc", "ro_mm", "tc_value"]:
            if cand in lower_cols:
                tc_col = lower_cols[cand]
                break
        if tc_col is None:
            raise ValueError("TC 长表中未找到 tc_ro_mm / tc / ro_mm 字段。")

        out = df[[lower_cols["year"], lower_cols["month"], lower_cols["zone"], tc_col]].copy()
        out.columns = ["year", "month", "zone", "tc_ro_mm"]
    else:
        records = []
        for z in ZONES:
            possible = [
                # 常见写法 1：字段名直接为 Z1/Z2/... 或 Z1 在前
                z,
                f"{z}_tc",
                f"{z}_tc_ro_mm",
                f"{z}_tc_ro",
                f"{z}_ro_mm",
                f"{z}_ro",
                # 常见写法 2：tc / ro 在前
                f"tc_ro_{z}",
                f"tc_ro_mm_{z}",
                f"tc_{z}",
                f"ro_{z}",
                f"ro_mm_{z}",
            ]
            value_col = None
            for c in possible:
                if c in df.columns:
                    value_col = c
                    break
            if value_col is None:
                raise ValueError(
                    f"TC 宽表中未找到 {z} 字段，已尝试：{possible}\n"
                    f"当前 TC 表字段为：{list(df.columns)}"
                )

            tmp = df[["year", "month", value_col]].copy()
            tmp["zone"] = z
            tmp.rename(columns={value_col: "tc_ro_mm"}, inplace=True)
            records.append(tmp[["year", "month", "zone", "tc_ro_mm"]])

        out = pd.concat(records, ignore_index=True)

    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["month"] = pd.to_numeric(out["month"], errors="coerce")
    out["tc_ro_mm"] = pd.to_numeric(out["tc_ro_mm"], errors="coerce")
    out = out.dropna(subset=["year", "month", "zone"])
    out["year"] = out["year"].astype(int)
    out["month"] = out["month"].astype(int)
    out["zone"] = out["zone"].astype(str).str.upper().str.strip()
    out = out[out["zone"].isin(ZONES)].copy()
    return out.sort_values(["zone", "year", "month"]).reset_index(drop=True)


def make_wide_from_long_corrected(corrected_long: pd.DataFrame) -> pd.DataFrame:
    """校正结果长表转宽表。"""
    parts = []
    base = corrected_long[["year", "month"]].drop_duplicates().copy()
    base = base.sort_values(["year", "month"]).reset_index(drop=True)

    for z in ZONES:
        sub = corrected_long[corrected_long["zone"] == z].copy()
        keep_cols = ["year", "month", "tc_raw", "tc_qm", "best_method", "extrapolation_type", "sample_warning"]
        sub = sub[keep_cols].rename(columns={
            "tc_raw": f"{z}_tc_raw",
            "tc_qm": f"{z}_tc_qm",
            "best_method": f"{z}_best_method",
            "extrapolation_type": f"{z}_extrapolation_type",
            "sample_warning": f"{z}_sample_warning",
        })
        base = base.merge(sub, on=["year", "month"], how="left")

    base = add_date_column(base)
    first_cols = ["year", "month", "date"]
    other_cols = [c for c in base.columns if c not in first_cols]
    return base[first_cols + other_cols]


# =============================================================================
# 4. QM 映射方法
# =============================================================================

@dataclass
class MappingModel:
    zone: str
    month: int
    method: str
    n_train: int
    x_dense: np.ndarray
    y_dense: np.ndarray
    train_min: float
    train_max: float
    y_min: float
    y_max: float
    low_slope: float
    high_slope: float
    low_slope_adjusted: bool
    high_slope_adjusted: bool
    model_warning: str = ""


def _rolling_median_smooth(arr: np.ndarray, window: int) -> np.ndarray:
    """
    对一维数组进行滚动中位数平滑，窗口自动修正为奇数。

    说明：
    RQUANT 分支后续会对平滑后的数组首尾端点赋值。某些 pandas / NumPy
    版本中，rolling(...).median().to_numpy(...) 返回的数组可能继承只读状态。
    因此这里统一返回可写副本，避免 ValueError: assignment destination is read-only。
    """
    arr = np.array(arr, dtype=float, copy=True)
    if window <= 1 or len(arr) < 3:
        return arr.copy()
    if window % 2 == 0:
        window += 1
    window = min(window, len(arr) if len(arr) % 2 == 1 else len(arr) - 1)
    if window < 3:
        return arr.copy()

    s = pd.Series(arr.copy())
    sm = s.rolling(window=window, center=True, min_periods=1).median().to_numpy(dtype=float, copy=True)
    return sm.copy()


def _make_strict_mapping(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    清理映射曲线：排序、去除 NaN、合并重复 x，并保证 y 非递减。
    """
    x = np.array(x, dtype=float, copy=True)
    y = np.array(y, dtype=float, copy=True)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        raise ValueError("有效映射点不足。")

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # 合并重复 x，取 y 均值；重复值较多时可避免 np.interp / polyfit 不稳定
    tmp = pd.DataFrame({"x": x, "y": y}).groupby("x", as_index=False)["y"].mean()
    x2 = tmp["x"].to_numpy(dtype=float, copy=True)
    y2 = tmp["y"].to_numpy(dtype=float, copy=True)

    if len(x2) < 2:
        raise ValueError("TC 训练值几乎无变化，无法建立映射。")

    # 分位数映射应保持单调，这里防止样条震荡造成局部下降
    y2 = np.maximum.accumulate(y2).copy()

    return x2.copy(), y2.copy()


def _tail_slope(x: np.ndarray, y: np.ndarray, side: str) -> Tuple[float, bool, float, int]:
    """
    使用尾部普通线性回归估计外推斜率。
    若斜率 <= 0，则扩大窗口；若仍失败，回退 SLOPE_EPS。
    返回：slope, adjusted, used_ratio, used_n
    """
    assert side in ["low", "high"]
    x = np.array(x, dtype=float, copy=True)
    y = np.array(y, dtype=float, copy=True)
    n = len(x)

    if n < 2:
        return SLOPE_EPS, True, np.nan, 0

    for ratio in TAIL_SLOPE_SPAN_RATIO_LIST:
        k = max(TAIL_SLOPE_MIN_POINTS, int(math.ceil(n * ratio)))
        k = min(k, n)
        if k < 2:
            continue

        if side == "low":
            xx = x[:k]
            yy = y[:k]
        else:
            xx = x[-k:]
            yy = y[-k:]

        # 如果 x 变化太小，polyfit 不可靠
        if np.nanmax(xx) - np.nanmin(xx) <= 1e-12:
            continue

        try:
            slope = float(np.polyfit(xx, yy, 1)[0])
        except Exception:
            continue

        if np.isfinite(slope) and slope > 0:
            return slope, False, ratio, k

    return SLOPE_EPS, True, TAIL_SLOPE_SPAN_RATIO_LIST[-1], min(max(TAIL_SLOPE_MIN_POINTS, int(math.ceil(n * TAIL_SLOPE_SPAN_RATIO_LIST[-1]))), n)


def build_mapping_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    method: str,
    zone: str,
    month: int,
) -> MappingModel:
    """根据训练样本建立单个 QM 映射模型。"""
    x = np.array(x_train, dtype=float, copy=True)
    y = np.array(y_train, dtype=float, copy=True)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask].copy()
    y = y[mask].copy()

    if len(x) < 2:
        raise ValueError("训练样本少于 2，无法建立 QM。")
    if np.nanmax(x) - np.nanmin(x) <= 1e-12:
        raise ValueError("训练期 TC 取值无变化，无法建立 QM。")

    method = method.upper()
    p_grid = np.linspace(0.0, 1.0, N_DENSE_POINTS)
    warning_msg = ""

    if method == "QUANT":
        x_dense = np.quantile(x, p_grid).copy()
        y_dense = np.quantile(y, p_grid).copy()

    elif method == "RQUANT":
        # RQUANT 修复要点：
        # 1. quantile 输出显式 copy；
        # 2. rolling median 输出显式 copy；
        # 3. 端点赋值前再次确保数组可写。
        x_dense = np.quantile(x, p_grid).copy()
        y_dense = np.quantile(y, p_grid).copy()
        # 在概率空间对 OBS 分位数曲线做滚动中位数平滑，降低局部异常分位点影响
        win = max(3, int(round(N_DENSE_POINTS * RQUANT_SMOOTH_WINDOW_RATIO)))
        y_dense = _rolling_median_smooth(y_dense, win).copy()
        y_dense.setflags(write=True)
        # 保留端点，避免极端范围被平滑过度
        y_dense[0] = np.nanmin(y)
        y_dense[-1] = np.nanmax(y)

    elif method == "SSPLIN":
        if not SCIPY_AVAILABLE:
            raise RuntimeError("当前 Python 环境未安装 scipy，无法使用 SSPLIN。")

        # 先构造经验分位数节点，再拟合 y=f(x) 的平滑样条
        xq = np.quantile(x, p_grid).copy()
        yq = np.quantile(y, p_grid).copy()
        xq, yq = _make_strict_mapping(xq, yq)

        if len(xq) < 4:
            # 样本太少时退化为线性插值曲线；正常流程中小样本不会允许 SSPLIN
            x_dense = np.linspace(np.nanmin(x), np.nanmax(x), N_DENSE_POINTS).copy()
            y_dense = np.interp(x_dense, xq, yq).copy()
            warning_msg = "SSPLIN_unique_points_too_few_fallback_to_linear_interp"
        else:
            k = min(3, len(xq) - 1)
            y_var = float(np.nanvar(yq))
            s_val = SSPLIN_S_FACTOR_RATIO * len(xq) * y_var
            try:
                spline = UnivariateSpline(xq, yq, k=k, s=s_val)
                x_dense = np.linspace(np.nanmin(x), np.nanmax(x), N_DENSE_POINTS).copy()
                y_dense = np.asarray(spline(x_dense), dtype=float).copy()
                # 锚定端点，降低样条端点漂移
                y_dense[0] = yq[0]
                y_dense[-1] = yq[-1]
            except Exception as e:
                x_dense = np.linspace(np.nanmin(x), np.nanmax(x), N_DENSE_POINTS).copy()
                y_dense = np.interp(x_dense, xq, yq).copy()
                warning_msg = f"SSPLIN_failed_fallback_to_linear_interp: {e}"

    else:
        raise ValueError(f"未知 QM 方法：{method}")

    x_dense, y_dense = _make_strict_mapping(x_dense, y_dense)

    # 若样条或平滑产生负值，模型曲线可以暂留；最终应用时统一施加非负约束。
    train_min = float(np.nanmin(x))
    train_max = float(np.nanmax(x))
    y_min = float(np.interp(train_min, x_dense, y_dense))
    y_max = float(np.interp(train_max, x_dense, y_dense))

    low_slope, low_adj, _, _ = _tail_slope(x_dense, y_dense, side="low")
    high_slope, high_adj, _, _ = _tail_slope(x_dense, y_dense, side="high")

    return MappingModel(
        zone=zone,
        month=int(month),
        method=method,
        n_train=int(len(x)),
        x_dense=x_dense,
        y_dense=y_dense,
        train_min=train_min,
        train_max=train_max,
        y_min=y_min,
        y_max=y_max,
        low_slope=low_slope,
        high_slope=high_slope,
        low_slope_adjusted=low_adj,
        high_slope_adjusted=high_adj,
        model_warning=warning_msg,
    )


def apply_mapping_model(model: MappingModel, x_value: float) -> Dict[str, Any]:
    """应用单个 QM 模型，返回校正值和外推信息。"""
    x = float(x_value) if pd.notna(x_value) else np.nan
    if not np.isfinite(x):
        return {
            "tc_qm": np.nan,
            "extrapolation_type": "missing_tc",
            "slope_adjusted": False,
        }

    if x < model.train_min:
        y = model.y_min + model.low_slope * (x - model.train_min)
        extrap = "lower"
        slope_adj = model.low_slope_adjusted
    elif x > model.train_max:
        y = model.y_max + model.high_slope * (x - model.train_max)
        extrap = "upper"
        slope_adj = model.high_slope_adjusted
    else:
        y = float(np.interp(x, model.x_dense, model.y_dense))
        extrap = "none"
        slope_adj = False

    if APPLY_NONNEGATIVE_CONSTRAINT and np.isfinite(y):
        y = max(y, 0.0)

    return {
        "tc_qm": float(y) if np.isfinite(y) else np.nan,
        "extrapolation_type": extrap,
        "slope_adjusted": bool(slope_adj),
    }


# =============================================================================
# 5. 样本量规则与 LOOCV
# =============================================================================

def allowed_methods_by_sample_size(n: int) -> Tuple[List[str], bool, bool, str]:
    """
    根据样本量返回候选方法与警告。
    返回：allowed_methods, sample_warning, method_limited, status
    """
    if n < N_MIN_MODEL:
        return [], True, True, "insufficient_sample"
    if n < N_MIN_SSPLIN:
        return ["QUANT", "RQUANT"], True, True, "small_sample_method_limited"
    if n < N_NORMAL:
        return BASE_METHODS.copy(), True, False, "small_sample_warning"
    return BASE_METHODS.copy(), False, False, "normal"


def loocv_for_group(group: pd.DataFrame, method: str, zone: str, month: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    对单个 zone-month-method 执行 LOOCV。
    返回：predictions_df, extrap_records_df
    """
    rows_pred = []
    rows_extrap = []

    g = group.dropna(subset=["obs_ro_mm", "tc_ro_mm"]).copy()
    g = g.sort_values("year").reset_index(drop=True)
    n = len(g)

    for i in range(n):
        valid = g.drop(index=i).copy()
        test = g.iloc[i]

        try:
            model = build_mapping_model(
                x_train=valid["tc_ro_mm"].to_numpy(dtype=float),
                y_train=valid["obs_ro_mm"].to_numpy(dtype=float),
                method=method,
                zone=zone,
                month=month,
            )
            app = apply_mapping_model(model, float(test["tc_ro_mm"]))
            qm_pred = app["tc_qm"]
            extrap = app["extrapolation_type"]
            slope_adjusted = app["slope_adjusted"]
            err_msg = ""

            rows_extrap.append({
                "zone": zone,
                "month": month,
                "method": method,
                "year": int(test["year"]),
                "tc_raw": float(test["tc_ro_mm"]),
                "tc_qm": qm_pred,
                "train_min": model.train_min,
                "train_max": model.train_max,
                "extrapolation_type": extrap,
                "low_slope": model.low_slope,
                "high_slope": model.high_slope,
                "slope_adjusted": slope_adjusted,
                "low_slope_adjusted": model.low_slope_adjusted,
                "high_slope_adjusted": model.high_slope_adjusted,
                "model_warning": model.model_warning,
            })

        except Exception as e:
            qm_pred = np.nan
            extrap = "model_failed"
            slope_adjusted = False
            err_msg = str(e)

        rows_pred.append({
            "zone": zone,
            "month": month,
            "method": method,
            "year": int(test["year"]),
            "obs": float(test["obs_ro_mm"]),
            "tc_raw": float(test["tc_ro_mm"]),
            "qm_pred": qm_pred,
            "extrapolation_type": extrap,
            "slope_adjusted": slope_adjusted,
            "error_message": err_msg,
        })

    return pd.DataFrame(rows_pred), pd.DataFrame(rows_extrap)


def choose_best_method(metrics_df: pd.DataFrame) -> Optional[pd.Series]:
    """先按 NSE 容差筛选，再依次比较 RMSE、|PBIAS| 和 |Bias|。"""
    df = metrics_df.copy()
    df = df[np.isfinite(df["NSE"])]
    if df.empty:
        # 如果 NSE 全部为 NaN，则退而按 RMSE 选择
        df = metrics_df.copy()
        df = df[np.isfinite(df["RMSE"])]
        if df.empty:
            return None
        df = df.sort_values(["RMSE", "PBIAS_abs", "Bias_abs"], ascending=[True, True, True])
        return df.iloc[0]

    max_nse = df["NSE"].max()
    candidate = df[df["NSE"] >= max_nse - NSE_TIE_TOL].copy()
    candidate = candidate.sort_values(["RMSE", "PBIAS_abs", "Bias_abs"], ascending=[True, True, True])
    return candidate.iloc[0]


# =============================================================================
# 6. 主流程：LOOCV 筛选
# =============================================================================

def run_loocv_selection(train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_pred = []
    all_extrap = []
    all_metrics = []
    sample_checks = []
    best_rows = []

    for zone in ZONES:
        for month in MONTHS:
            g = train_df[(train_df["zone"] == zone) & (train_df["month"] == month)].copy()
            g = g.dropna(subset=["obs_ro_mm", "tc_ro_mm"])
            g = g.sort_values("year").reset_index(drop=True)
            n = len(g)
            allowed, sample_warning, method_limited, status = allowed_methods_by_sample_size(n)

            sample_checks.append({
                "zone": zone,
                "month": month,
                "N": n,
                "status": status,
                "allowed_methods": ";".join(allowed),
                "sample_warning": sample_warning,
                "method_limited_by_small_sample": method_limited,
                "min_year": int(g["year"].min()) if n > 0 else np.nan,
                "max_year": int(g["year"].max()) if n > 0 else np.nan,
            })

            if not allowed:
                best_rows.append({
                    "zone": zone,
                    "month": month,
                    "N": n,
                    "candidate_methods": "",
                    "best_method": "insufficient_sample",
                    "sample_warning": True,
                    "method_limited_by_small_sample": True,
                    "status": status,
                    "R2": np.nan,
                    "RMSE": np.nan,
                    "MAE": np.nan,
                    "Bias": np.nan,
                    "PBIAS": np.nan,
                    "NSE": np.nan,
                    "OBS_mean": np.nan,
                    "SIM_mean": np.nan,
                })
                continue

            method_metric_rows = []
            for method in allowed:
                pred_df, extrap_df = loocv_for_group(g, method, zone, month)
                if not pred_df.empty:
                    all_pred.append(pred_df)
                if not extrap_df.empty:
                    all_extrap.append(extrap_df)

                m = calc_metrics(pred_df["obs"].to_numpy(dtype=float), pred_df["qm_pred"].to_numpy(dtype=float))
                mrow = {
                    "zone": zone,
                    "month": month,
                    "method": method,
                    "candidate_methods": ";".join(allowed),
                    "sample_warning": sample_warning,
                    "method_limited_by_small_sample": method_limited,
                    "status": status,
                    **m,
                }
                mrow["PBIAS_abs"] = abs(mrow["PBIAS"]) if np.isfinite(mrow["PBIAS"]) else np.nan
                mrow["Bias_abs"] = abs(mrow["Bias"]) if np.isfinite(mrow["Bias"]) else np.nan
                method_metric_rows.append(mrow)
                all_metrics.append(mrow)

            method_metrics_df = pd.DataFrame(method_metric_rows)
            best = choose_best_method(method_metrics_df)
            if best is None:
                best_rows.append({
                    "zone": zone,
                    "month": month,
                    "N": n,
                    "candidate_methods": ";".join(allowed),
                    "best_method": "no_valid_method",
                    "sample_warning": sample_warning,
                    "method_limited_by_small_sample": method_limited,
                    "status": "no_valid_method",
                })
            else:
                best_rows.append({
                    "zone": zone,
                    "month": month,
                    "N": n,
                    "candidate_methods": ";".join(allowed),
                    "best_method": best["method"],
                    "sample_warning": sample_warning,
                    "method_limited_by_small_sample": method_limited,
                    "status": status,
                    "R2": best.get("R2", np.nan),
                    "RMSE": best.get("RMSE", np.nan),
                    "MAE": best.get("MAE", np.nan),
                    "Bias": best.get("Bias", np.nan),
                    "PBIAS": best.get("PBIAS", np.nan),
                    "NSE": best.get("NSE", np.nan),
                    "OBS_mean": best.get("OBS_mean", np.nan),
                    "SIM_mean": best.get("SIM_mean", np.nan),
                })

            print(f"[LOOCV] {zone} 月{month:02d}: N={n}, status={status}, methods={allowed}")

    pred_all = pd.concat(all_pred, ignore_index=True) if all_pred else pd.DataFrame()
    extrap_all = pd.concat(all_extrap, ignore_index=True) if all_extrap else pd.DataFrame()
    metrics_all = pd.DataFrame(all_metrics)
    sample_check_df = pd.DataFrame(sample_checks)
    best_df = pd.DataFrame(best_rows)

    # 排序
    for df in [pred_all, extrap_all, metrics_all, sample_check_df, best_df]:
        if not df.empty:
            sort_cols = [c for c in ["zone", "month", "method", "year"] if c in df.columns]
            df.sort_values(sort_cols, inplace=True)
            df.reset_index(drop=True, inplace=True)

    return pred_all, metrics_all, best_df, extrap_all, sample_check_df


def build_loocv_failure_summary(pred_all: pd.DataFrame) -> pd.DataFrame:
    """
    汇总 LOOCV 阶段各候选方法的失败原因。

    用途：
    1. 检查 RQUANT / QUANT / SSPLIN 是否因为代码或样本问题未能有效参与筛选；
    2. 若 failure_count 为 0，说明对应方法在 LOOCV 预测阶段没有运行错误；
    3. 若存在失败，可根据 fail_reason 定位具体原因。
    """
    if pred_all is None or pred_all.empty or "error_message" not in pred_all.columns:
        return pd.DataFrame(columns=[
            "zone", "month", "method", "fail_reason", "failure_count",
            "N_total", "failure_ratio_pct"
        ])

    df = pred_all.copy()
    df["error_message"] = df["error_message"].fillna("").astype(str)
    df_fail = df[df["error_message"].str.strip() != ""].copy()

    if df_fail.empty:
        # 仍然输出一个空表结构，便于后续自动检查。
        return pd.DataFrame(columns=[
            "zone", "month", "method", "fail_reason", "failure_count",
            "N_total", "failure_ratio_pct"
        ])

    total = (
        df.groupby(["zone", "month", "method"], as_index=False)
          .size()
          .rename(columns={"size": "N_total"})
    )
    fail = (
        df_fail.groupby(["zone", "month", "method", "error_message"], as_index=False)
               .size()
               .rename(columns={"error_message": "fail_reason", "size": "failure_count"})
    )
    out = fail.merge(total, on=["zone", "month", "method"], how="left")
    out["failure_ratio_pct"] = np.where(
        out["N_total"] > 0,
        100.0 * out["failure_count"] / out["N_total"],
        np.nan,
    )
    out = out.sort_values(["zone", "month", "method", "failure_count"], ascending=[True, True, True, False])
    return out.reset_index(drop=True)


# =============================================================================
# 7. 最终模型建立与应用
# =============================================================================

def build_final_models_and_apply(
    train_df: pd.DataFrame,
    tc_all_long: pd.DataFrame,
    best_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[Tuple[str, int], MappingModel]]:
    final_summary_rows = []
    dense_rows = []
    corrected_rows = []
    app_extrap_rows = []
    model_dict: Dict[Tuple[str, int], MappingModel] = {}

    for zone in ZONES:
        for month in MONTHS:
            best_sub = best_df[(best_df["zone"] == zone) & (best_df["month"] == month)]
            if best_sub.empty:
                best_method = "no_valid_method"
                sample_warning = True
                status = "no_best_record"
            else:
                best_row = best_sub.iloc[0]
                best_method = str(best_row.get("best_method", "no_valid_method"))
                sample_warning = bool(best_row.get("sample_warning", True))
                status = str(best_row.get("status", "unknown"))

            tc_sub = tc_all_long[(tc_all_long["zone"] == zone) & (tc_all_long["month"] == month)].copy()
            tc_sub = tc_sub.sort_values("year")

            if best_method in ["insufficient_sample", "no_valid_method", "nan", "None"]:
                final_summary_rows.append({
                    "zone": zone,
                    "month": month,
                    "best_method": best_method,
                    "N_train": 0,
                    "status": status,
                    "sample_warning": sample_warning,
                    "train_min": np.nan,
                    "train_max": np.nan,
                    "low_slope": np.nan,
                    "high_slope": np.nan,
                    "low_slope_adjusted": np.nan,
                    "high_slope_adjusted": np.nan,
                    "model_warning": "model_not_built",
                })
                for _, r in tc_sub.iterrows():
                    corrected_rows.append({
                        "year": int(r["year"]),
                        "month": int(month),
                        "zone": zone,
                        "tc_raw": float(r["tc_ro_mm"]) if pd.notna(r["tc_ro_mm"]) else np.nan,
                        "tc_qm": np.nan,
                        "best_method": best_method,
                        "extrapolation_type": "model_not_built",
                        "slope_adjusted": False,
                        "sample_warning": sample_warning,
                        "status": status,
                    })
                continue

            g = train_df[(train_df["zone"] == zone) & (train_df["month"] == month)].dropna(subset=["obs_ro_mm", "tc_ro_mm"]).copy()
            g = g.sort_values("year")

            try:
                model = build_mapping_model(
                    x_train=g["tc_ro_mm"].to_numpy(dtype=float),
                    y_train=g["obs_ro_mm"].to_numpy(dtype=float),
                    method=best_method,
                    zone=zone,
                    month=month,
                )
                model_dict[(zone, month)] = model
                model_warning = model.model_warning

                final_summary_rows.append({
                    "zone": zone,
                    "month": month,
                    "best_method": best_method,
                    "N_train": model.n_train,
                    "status": status,
                    "sample_warning": sample_warning,
                    "train_min": model.train_min,
                    "train_max": model.train_max,
                    "y_min": model.y_min,
                    "y_max": model.y_max,
                    "low_slope": model.low_slope,
                    "high_slope": model.high_slope,
                    "low_slope_adjusted": model.low_slope_adjusted,
                    "high_slope_adjusted": model.high_slope_adjusted,
                    "model_warning": model_warning,
                })

                for i, (xd, yd) in enumerate(zip(model.x_dense, model.y_dense)):
                    dense_rows.append({
                        "zone": zone,
                        "month": month,
                        "method": best_method,
                        "curve_index": i,
                        "x_tc": float(xd),
                        "y_obs_mapped": float(yd),
                    })

                for _, r in tc_sub.iterrows():
                    app = apply_mapping_model(model, r["tc_ro_mm"])
                    corrected_rows.append({
                        "year": int(r["year"]),
                        "month": int(month),
                        "zone": zone,
                        "tc_raw": float(r["tc_ro_mm"]) if pd.notna(r["tc_ro_mm"]) else np.nan,
                        "tc_qm": app["tc_qm"],
                        "best_method": best_method,
                        "extrapolation_type": app["extrapolation_type"],
                        "slope_adjusted": app["slope_adjusted"],
                        "sample_warning": sample_warning,
                        "status": status,
                    })
                    if app["extrapolation_type"] in ["lower", "upper"] or app["slope_adjusted"]:
                        app_extrap_rows.append({
                            "year": int(r["year"]),
                            "month": int(month),
                            "zone": zone,
                            "method": best_method,
                            "tc_raw": float(r["tc_ro_mm"]) if pd.notna(r["tc_ro_mm"]) else np.nan,
                            "tc_qm": app["tc_qm"],
                            "train_min": model.train_min,
                            "train_max": model.train_max,
                            "extrapolation_type": app["extrapolation_type"],
                            "low_slope": model.low_slope,
                            "high_slope": model.high_slope,
                            "slope_adjusted": app["slope_adjusted"],
                            "low_slope_adjusted": model.low_slope_adjusted,
                            "high_slope_adjusted": model.high_slope_adjusted,
                            "sample_warning": sample_warning,
                            "status": status,
                            "model_warning": model_warning,
                        })

            except Exception as e:
                final_summary_rows.append({
                    "zone": zone,
                    "month": month,
                    "best_method": best_method,
                    "N_train": len(g),
                    "status": "final_model_failed",
                    "sample_warning": sample_warning,
                    "train_min": np.nan,
                    "train_max": np.nan,
                    "low_slope": np.nan,
                    "high_slope": np.nan,
                    "low_slope_adjusted": np.nan,
                    "high_slope_adjusted": np.nan,
                    "model_warning": str(e),
                })
                for _, r in tc_sub.iterrows():
                    corrected_rows.append({
                        "year": int(r["year"]),
                        "month": int(month),
                        "zone": zone,
                        "tc_raw": float(r["tc_ro_mm"]) if pd.notna(r["tc_ro_mm"]) else np.nan,
                        "tc_qm": np.nan,
                        "best_method": best_method,
                        "extrapolation_type": "final_model_failed",
                        "slope_adjusted": False,
                        "sample_warning": sample_warning,
                        "status": "final_model_failed",
                    })

            print(f"[Final] {zone} 月{month:02d}: method={best_method}, status={status}")

    final_summary = pd.DataFrame(final_summary_rows).sort_values(["zone", "month"]).reset_index(drop=True)
    dense_df = pd.DataFrame(dense_rows).sort_values(["zone", "month", "curve_index"]).reset_index(drop=True) if dense_rows else pd.DataFrame()
    corrected_long = pd.DataFrame(corrected_rows).sort_values(["zone", "year", "month"]).reset_index(drop=True)
    app_extrap = pd.DataFrame(app_extrap_rows).sort_values(["zone", "month", "year"]).reset_index(drop=True) if app_extrap_rows else pd.DataFrame()

    return final_summary, dense_df, corrected_long, app_extrap, model_dict


# =============================================================================
# 8. 比例系数与校正后验证
# =============================================================================

def build_factor_tables(corrected_long: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = corrected_long.copy()
    df["factor"] = np.where(
        (df["tc_raw"].notna()) & (df["tc_raw"] > TC_FACTOR_EPS) & (df["tc_qm"].notna()),
        df["tc_qm"] / df["tc_raw"],
        np.nan,
    )
    df["factor_invalid_reason"] = ""
    df.loc[(df["tc_raw"].notna()) & (df["tc_raw"] <= TC_FACTOR_EPS), "factor_invalid_reason"] = "tc_raw_too_small"
    df.loc[df["tc_qm"].isna(), "factor_invalid_reason"] = np.where(
        df.loc[df["tc_qm"].isna(), "factor_invalid_reason"].eq(""),
        "tc_qm_missing",
        df.loc[df["tc_qm"].isna(), "factor_invalid_reason"] + ";tc_qm_missing",
    )

    factor_long = df[[
        "year", "month", "zone", "factor", "tc_raw", "tc_qm", "best_method",
        "extrapolation_type", "sample_warning", "factor_invalid_reason"
    ]].sort_values(["year", "month", "zone"]).reset_index(drop=True)

    check = factor_long.pivot_table(index=["year", "month"], columns="zone", values="factor", aggfunc="first").reset_index()
    check.columns = [str(c) if c in ["year", "month"] else f"{c}_factor" for c in check.columns]
    check = add_date_column(check)
    first_cols = ["year", "month", "date"]
    check = check[first_cols + [c for c in check.columns if c not in first_cols]]

    summary_rows = []
    for z in ZONES:
        sub = factor_long[factor_long["zone"] == z]
        vals = sub["factor"].dropna()
        summary_rows.append({
            "zone": z,
            "N_total": len(sub),
            "N_valid_factor": int(vals.count()),
            "N_invalid_factor": int(sub["factor"].isna().sum()),
            "factor_min": float(vals.min()) if len(vals) else np.nan,
            "factor_p05": float(vals.quantile(0.05)) if len(vals) else np.nan,
            "factor_median": float(vals.median()) if len(vals) else np.nan,
            "factor_mean": float(vals.mean()) if len(vals) else np.nan,
            "factor_p95": float(vals.quantile(0.95)) if len(vals) else np.nan,
            "factor_max": float(vals.max()) if len(vals) else np.nan,
        })
    summary = pd.DataFrame(summary_rows)
    return factor_long, check, summary


def build_application_extrap_summary(corrected_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for zone in ZONES:
        for month in MONTHS:
            sub = corrected_long[(corrected_long["zone"] == zone) & (corrected_long["month"] == month)]
            n = len(sub)
            rows.append({
                "zone": zone,
                "month": month,
                "N_total": n,
                "N_none": int((sub["extrapolation_type"] == "none").sum()),
                "N_lower": int((sub["extrapolation_type"] == "lower").sum()),
                "N_upper": int((sub["extrapolation_type"] == "upper").sum()),
                "N_missing_or_failed": int((~sub["extrapolation_type"].isin(["none", "lower", "upper"])).sum()),
                "N_slope_adjusted": int(sub["slope_adjusted"].fillna(False).astype(bool).sum()),
                "lower_ratio_pct": 100.0 * int((sub["extrapolation_type"] == "lower").sum()) / n if n else np.nan,
                "upper_ratio_pct": 100.0 * int((sub["extrapolation_type"] == "upper").sum()) / n if n else np.nan,
            })
    return pd.DataFrame(rows)


def build_corrected_vs_obs_metrics(
    corrected_long: pd.DataFrame,
    obs_long: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = obs_long.merge(
        corrected_long[["year", "month", "zone", "tc_raw", "tc_qm", "best_method", "extrapolation_type"]],
        on=["year", "month", "zone"],
        how="left",
    )
    merged = merged.sort_values(["zone", "year", "month"]).reset_index(drop=True)

    # 宽表合并结果
    wide = merged.pivot_table(index=["year", "month"], columns="zone", values=["obs_ro_mm", "tc_raw", "tc_qm"], aggfunc="first")
    wide.columns = [f"{z}_{var}" for var, z in wide.columns]
    wide = wide.reset_index()
    wide = add_date_column(wide)
    first_cols = ["year", "month", "date"]
    wide = wide[first_cols + [c for c in wide.columns if c not in first_cols]]

    # 所有月份总体指标：Raw/QM 两套
    all_rows = []
    for z in ZONES:
        sub = merged[merged["zone"] == z]
        raw_m = calc_metrics(sub["obs_ro_mm"].to_numpy(dtype=float), sub["tc_raw"].to_numpy(dtype=float))
        qm_m = calc_metrics(sub["obs_ro_mm"].to_numpy(dtype=float), sub["tc_qm"].to_numpy(dtype=float))
        all_rows.append({"zone": z, "series": "raw", **raw_m})
        all_rows.append({"zone": z, "series": "qm", **qm_m})
    all_metrics = pd.DataFrame(all_rows)

    # 分月指标：Raw/QM 两套
    month_rows = []
    for z in ZONES:
        for m in MONTHS:
            sub = merged[(merged["zone"] == z) & (merged["month"] == m)]
            raw_m = calc_metrics(sub["obs_ro_mm"].to_numpy(dtype=float), sub["tc_raw"].to_numpy(dtype=float))
            qm_m = calc_metrics(sub["obs_ro_mm"].to_numpy(dtype=float), sub["tc_qm"].to_numpy(dtype=float))
            month_rows.append({"zone": z, "month": m, "series": "raw", **raw_m})
            month_rows.append({"zone": z, "month": m, "series": "qm", **qm_m})
    month_metrics = pd.DataFrame(month_rows)

    return wide, all_metrics, month_metrics


# =============================================================================
# 9. 主函数
# =============================================================================

def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    obs_path = OBS_ZONE_MONTHLY_CSV
    tc_path = TC_ZONE_MONTHLY_CSV
    for input_path in (obs_path, tc_path):
        if not input_path.is_file():
            raise FileNotFoundError(input_path)

    out00 = os.path.join(OUTPUT_ROOT, "00_model_selection")
    out01 = os.path.join(OUTPUT_ROOT, "01_final_models")
    out02 = os.path.join(OUTPUT_ROOT, "02_corrected_runoff")
    out03 = os.path.join(OUTPUT_ROOT, "03_zone_validation")
    for d in [out00, out01, out02, out03]:
        ensure_dir(d)

    print("=" * 80)
    print("月尺度 Z1-Z5 独立分区 + 分月 QM 校正")
    print("=" * 80)
    print(f"OBS 输入：{obs_path}")
    print(f"TC  输入：{tc_path}")
    print(f"输出目录：{OUTPUT_ROOT}")
    print(f"scipy 可用：{SCIPY_AVAILABLE}")

    # 读取与整理
    obs_wide = read_csv_smart(obs_path)
    tc_wide = read_csv_smart(tc_path)
    obs_long = obs_wide_to_long(obs_wide)
    obs_long = obs_long[obs_long["year"].between(CALIBRATION_START_YEAR, CALIBRATION_END_YEAR)].copy()
    tc_long = tc_wide_to_long(tc_wide)

    # 训练样本：OBS 与 TC 重叠期
    train_df = obs_long.merge(tc_long, on=["year", "month", "zone"], how="inner")
    train_df = train_df.dropna(subset=["obs_ro_mm", "tc_ro_mm"]).copy()
    train_df = train_df.sort_values(["zone", "month", "year"]).reset_index(drop=True)

    print(f"OBS long 行数：{len(obs_long)}")
    print(f"TC  long 行数：{len(tc_long)}")
    print(f"训练重叠样本行数：{len(train_df)}")

    # 输出训练样本备查
    write_csv(train_df, os.path.join(out00, "monthly_QM_training_samples_Z1_Z5_by_month_long.csv"))

    # Step 1: LOOCV 筛选
    print("\n[Step 1] 开始 LOOCV 筛选...")
    pred_all, metrics_all, best_df, loocv_extrap, sample_check = run_loocv_selection(train_df)

    write_csv(pred_all, os.path.join(out00, "monthly_QM_LOOCV_predictions_Z1_Z5_by_month_wide.csv"))
    write_csv(metrics_all, os.path.join(out00, "monthly_QM_LOOCV_metrics_Z1_Z5_by_month_wide.csv"))
    write_csv(best_df, os.path.join(out00, "monthly_QM_best_method_Z1_Z5_by_month_wide.csv"))
    write_csv(loocv_extrap, os.path.join(out00, "monthly_QM_LOOCV_extrapolation_records_Z1_Z5_by_month_wide.csv"))
    write_csv(sample_check, os.path.join(out00, "monthly_QM_sample_size_check_Z1_Z5_by_month_wide.csv"))

    failure_summary = build_loocv_failure_summary(pred_all)
    write_csv(failure_summary, os.path.join(out00, "monthly_QM_LOOCV_failure_reason_summary_Z1_Z5_by_month_wide.csv"))

    # Step 2: 最终模型与 1958-2024 应用
    print("\n[Step 2] 建立最终分区分月 QM 并应用到 1958-2024...")
    final_summary, dense_df, corrected_long, app_extrap, model_dict = build_final_models_and_apply(train_df, tc_long, best_df)

    write_csv(final_summary, os.path.join(out01, "monthly_QM_final_model_summary_Z1_Z5_by_month_wide.csv"))
    write_csv(dense_df, os.path.join(out01, "monthly_QM_final_dense_mapping_curves_Z1_Z5_by_month_wide.csv"))

    corrected_wide = make_wide_from_long_corrected(corrected_long)
    write_csv(corrected_long, os.path.join(out02, "monthly_TC_QM_corrected_Z1_Z5_1958_2024_long.csv"))
    write_csv(corrected_wide, os.path.join(out02, "monthly_TC_QM_corrected_Z1_Z5_1958_2024_wide.csv"))
    write_csv(app_extrap, os.path.join(out02, "monthly_QM_application_extrapolation_records_Z1_Z5_by_month_wide.csv"))

    app_extrap_summary = build_application_extrap_summary(corrected_long)
    write_csv(app_extrap_summary, os.path.join(out02, "monthly_QM_application_extrapolation_summary_Z1_Z5_by_month_wide.csv"))

    # Step 3: 比例系数
    print("\n[Step 3] 生成 GEE 下推比例系数...")
    factor_long, factor_check_wide, factor_summary = build_factor_tables(corrected_long)
    write_csv(factor_long, os.path.join(out02, "monthly_zone_correction_factor_Z1_Z5_1958_2024_for_GEE.csv"))
    write_csv(factor_check_wide, os.path.join(out02, "monthly_zone_correction_factor_Z1_Z5_1958_2024_check_wide.csv"))
    write_csv(factor_summary, os.path.join(out02, "monthly_zone_correction_factor_Z1_Z5_1958_2024_summary_wide.csv"))

    # Step 4: 校正后区间验证
    print("\n[Step 4] 生成校正后区间验证指标...")
    merged_wide, all_metrics, month_metrics = build_corrected_vs_obs_metrics(corrected_long, obs_long)
    write_csv(merged_wide, os.path.join(out03, "monthly_TC_QM_corrected_Z1_Z5_vs_OBS_merged_wide.csv"))
    write_csv(all_metrics, os.path.join(out03, "monthly_TC_QM_corrected_Z1_Z5_metrics_all_months_wide.csv"))
    write_csv(month_metrics, os.path.join(out03, "monthly_TC_QM_corrected_Z1_Z5_metrics_by_calendar_month_wide.csv"))

    # 简要打印最优方法与样本警告
    print("\n" + "=" * 80)
    print("运行完成。关键结果预览：")
    print("=" * 80)
    print("\n[最优方法表前 20 行]")
    print(best_df.head(20).to_string(index=False))

    print("\n[样本量警告]")
    warn = sample_check[sample_check["sample_warning"] == True]
    if warn.empty:
        print("无样本量警告。")
    else:
        print(warn.to_string(index=False))

    print("\n[LOOCV 失败原因汇总]")
    if failure_summary.empty:
        print("无 LOOCV 运行错误；RQUANT/QUANT/SSPLIN 均已正常参与其允许样本单元的筛选。")
    else:
        print(failure_summary.head(20).to_string(index=False))

    print("\n[输出目录]")
    print(out00)
    print(out01)
    print(out02)
    print(out03)


if __name__ == "__main__":
    main()
