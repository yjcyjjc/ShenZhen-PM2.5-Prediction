"""用于 PM2.5 预测的端到端数据准备流水线。

本模块将探索性 Notebook 中的处理逻辑整合为可复用函数。
它读取已经预处理完成的 CSV 文件（空气质量、PM 详细数据、
地面气象观测、ERA5 类再分析栅格数据），
并生成一个按小时粒度合并的数据集，
同时与日尺度空气质量观测结果进行时间对齐。

最低字段要求（至少应包含以下列）：
    air_quality: ['O3', 'PM25', 'district', 'CO', 'NO2', 'datetime', 'SO2', 'PM10']
    pm_detail:   ['datetime', 'site_id', 'average_pm25_hour', 'district']
    weather:     ['pressure_hPa', 'datetime', 'precipitation_mm', 'wind_direction_deg',
                  'temperature_C', 'relative_humidity_pct', 'visibility_m',
                  'wind_speed_ms']
    reanalysis:  ['datetime', 'latitude', 'longitude', 'number', 'expver', 'u10', 'v10',
                  't2m_c', 'd2m_c', 'sp_hpa', 'blh_m', 'wind_speed', 'wind_dir', 'rh',
                  'source_zip']
"""
from __future__ import annotations

import argparse
import pathlib
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Basic datetime utilities
# ---------------------------------------------------------------------------

def _ensure_datetime_col(df: pd.DataFrame, col: str = "datetime") -> pd.DataFrame:
    """返回一个副本，其中 ``col`` 存在并被解析为 datetime 类型。
        如果 ``col`` 仅作为索引存在，则先将其移到列中。
    """

    out = df.copy()
    if col not in out.columns and out.index.name == col:
        out = out.reset_index()
    out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def _floor_to_hour(df: pd.DataFrame, col: str = "datetime") -> pd.DataFrame:
    """返回一个副本，其中 ``col`` 被向下取整到小时。"""

    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce").dt.floor("H")
    return out


# ---------------------------------------------------------------------------
# Weather aggregation
# ---------------------------------------------------------------------------

def hourly_inhour_rolling(
    weather: pd.DataFrame,
    dt_col: str = "datetime",
    wind_dir_col: str = "wind_direction_deg",
    min_obs: int = 1,
    strict_wind_dir: bool = True,
) -> pd.DataFrame:
    """将不规则频率的气象观测数据聚合为小时尺度数据。

- 均值聚合：temperature_C、pressure_hPa、relative_humidity_pct、
  visibility_m、wind_speed_ms
- 累计求和：precipitation_mm（小时累计降水量）
- 风向处理：当 ``strict_wind_dir`` 为 True 时，采用向量平均法，
  以避免在 0/360 度附近产生角度回绕偏差。

输出结果的索引为向下取整后的整点小时时间戳，并按时间升序排列。
"""

    df = weather.copy()
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.dropna(subset=[dt_col]).sort_values(dt_col).set_index(dt_col)

    mean_cols = [
        "temperature_C",
        "pressure_hPa",
        "relative_humidity_pct",
        "visibility_m",
        "wind_speed_ms",
    ]
    sum_cols = ["precipitation_mm"]

    mean_cols = [c for c in mean_cols if c in df.columns]
    sum_cols = [c for c in sum_cols if c in df.columns]

    rolling_obj = df.rolling("60min", closed="right")
    out_parts: list[pd.DataFrame] = []

    if mean_cols:
        out_parts.append(rolling_obj[mean_cols].mean())
    if sum_cols:
        out_parts.append(rolling_obj[sum_cols].sum())

    if wind_dir_col in df.columns:
        if strict_wind_dir:
            rad = np.deg2rad(df[wind_dir_col].astype(float))
            sin_mean = np.sin(rad).rolling("60min", closed="right").mean()
            cos_mean = np.cos(rad).rolling("60min", closed="right").mean()
            ang = np.arctan2(sin_mean, cos_mean)
            deg = (np.rad2deg(ang) + 360.0) % 360.0
            out_parts.append(deg.to_frame(wind_dir_col))
        else:
            out_parts.append(rolling_obj[[wind_dir_col]].mean())

    if not out_parts:
        raise ValueError("No aggregatable columns found in weather DataFrame.")

    rolled = pd.concat(out_parts, axis=1).reset_index()
    rolled["hour"] = rolled[dt_col].dt.floor("H")
    rolled = (
        rolled.sort_values(dt_col)
        .drop_duplicates(subset=["hour"], keep="last")
        .drop(columns=[dt_col])
        .rename(columns={"hour": dt_col})
        .set_index(dt_col)
    )

    counts = (
        df.iloc[:, [0]]
        .rolling("60min", closed="right")
        .count()
        .reset_index()
    )
    counts["hour"] = counts[dt_col].dt.floor("H")
    counts = (
        counts.sort_values(dt_col)
        .drop_duplicates(subset=["hour"], keep="last")
        .drop(columns=[dt_col])
        .set_index("hour")
        .iloc[:, 0]
    )

    rolled.loc[counts < min_obs, :] = np.nan
    return rolled.sort_index()


# ---------------------------------------------------------------------------
# Reanalysis processing
# ---------------------------------------------------------------------------
"""将栅格化再分析数据压缩为行政区级时间序列。

检查栅格的最南一行（最小纬度）：其中最西侧点定义为西南角（SW），
最东侧点定义为东南角（SE）。将东南角（SE）网格代表“⼤鹏区”，
其余网格点（剔除 SW 与 SE 后）求平均作为“其它”。对每个时间戳，
对数值型列（不含经纬度）进行均值聚合，得到区级序列。
"""
def build_reanalysis_by_district(reanalysis: pd.DataFrame) -> pd.DataFrame:
    df = reanalysis.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    grid = (
        df[["latitude", "longitude"]]
        .drop_duplicates()
        .sort_values(["latitude", "longitude"], ascending=[False, True])
        .reset_index(drop=True)
    )

    lat_min = grid["latitude"].min()
    south_row = grid[np.isclose(grid["latitude"].values, lat_min)]
    sw = south_row.sort_values("longitude").iloc[0]
    se = south_row.sort_values("longitude").iloc[-1]

    sw_lat, sw_lon = float(sw["latitude"]), float(sw["longitude"])
    se_lat, se_lon = float(se["latitude"]), float(se["longitude"])

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ["latitude", "longitude"]]

    mask_dp = np.isclose(df["latitude"].values, se_lat) & np.isclose(
        df["longitude"].values, se_lon
    )
    df_dp = (
        df.loc[mask_dp]
        .groupby("datetime", as_index=False)[numeric_cols]
        .mean()
    )
    df_dp["target_district"] = "大鹏区"

    mask_sw = np.isclose(df["latitude"].values, sw_lat) & np.isclose(
        df["longitude"].values, sw_lon
    )
    mask_other = ~(mask_sw | mask_dp)

    df_other = (
        df.loc[mask_other]
        .groupby("datetime", as_index=False)[numeric_cols]
        .mean()
    )
    df_other["target_district"] = "其它"

    out = (
        pd.concat([df_other, df_dp], ignore_index=True)
        .sort_values(["datetime", "target_district"])
        .reset_index(drop=True)
    )

    return out


# ---------------------------------------------------------------------------
# End-to-end merging
# ---------------------------------------------------------------------------

def align_time_range(reference: pd.DataFrame, *others: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp, list[pd.DataFrame]]:
    """将其他 DataFrame 的时间范围修剪到 ``reference`` 的时间跨度。
    return (start, end, trimmed_others).
    """

    start = reference["datetime"].min()
    end = reference["datetime"].max()
    trimmed = []
    for df in others:
        tmp = df.copy()
        tmp = tmp[(tmp["datetime"] >= start) & (tmp["datetime"] <= end)]
        trimmed.append(tmp)
    return start, end, trimmed


def merge_hourly_sources(
    reanalysis_by_district: pd.DataFrame,
    weather_hourly: pd.DataFrame,
    pm_detail: pd.DataFrame,
) -> pd.DataFrame:
    """合并再分析（目标区）与小时气象和 PM 详细数据。"""

    weather_h = _floor_to_hour(_ensure_datetime_col(weather_hourly, "datetime"), "datetime")
    weather_h = weather_h.sort_values("datetime").drop_duplicates("datetime", keep="last")

    pm_d = _floor_to_hour(_ensure_datetime_col(pm_detail, "datetime"), "datetime")

    rea = _floor_to_hour(_ensure_datetime_col(reanalysis_by_district, "datetime"), "datetime")

    main = rea.merge(weather_h, on="datetime", how="left")
    main = main.merge(pm_d, on="datetime", how="left")
    return main.dropna()


def merge_daily_air_quality(main: pd.DataFrame, air_quality: pd.DataFrame) -> pd.DataFrame:
    """、将日尺度空气质量通过向下取整的日期和区映射附加到小时主表。"""

    main = main.copy()
    main["datetime"] = pd.to_datetime(main["datetime"], errors="coerce")
    main["date00"] = main["datetime"].dt.floor("D")

    air = air_quality.copy()
    air["datetime"] = pd.to_datetime(air["datetime"], errors="coerce")
    air["date00"] = air["datetime"].dt.floor("D")

    merged = main.merge(air, on=["date00", "district"], how="left", suffixes=("", "_air"))

    # 筛除大鹏区与其它区之间的错误匹配（因为它们共享同一再分析网格点）：
    mask = (
        (merged["target_district"] == "大鹏区") & (merged["district"] == "大鹏新区")
    ) | (
        (merged["target_district"] != "大鹏区") & (merged["district"] != "大鹏新区")
    )
    return merged.loc[mask].reset_index(drop=True)


def run_pipeline(
    air_quality_path: pathlib.Path,
    pm_detail_path: pathlib.Path,
    weather_path: pathlib.Path,
    reanalysis_path: pathlib.Path,
    output_csv: Optional[pathlib.Path] = None,
    *,
    min_obs_per_hour: int = 1,
    strict_wind_dir: bool = True,
) -> pd.DataFrame:
    """最终整合函数：从输入 CSV 文件读取数据，执行所有处理步骤，并返回合并后的 DataFrame。"""

    air_quality = pd.read_csv(air_quality_path)
    pm_detail = pd.read_csv(pm_detail_path)
    weather = pd.read_csv(weather_path)

    reanalysis = pd.read_csv(reanalysis_path)
    reanalysis = reanalysis.rename(columns={"time": "datetime"})
    reanalysis["datetime"] = pd.to_datetime(reanalysis["datetime"], errors="coerce")
    reanalysis = reanalysis.sort_values("datetime")

    reanalysis_by_district = build_reanalysis_by_district(reanalysis)
    reanalysis_by_district = reanalysis_by_district.drop(columns=["number", "expver"], errors="ignore")

    weather_hourly = hourly_inhour_rolling(
        weather,
        dt_col="datetime",
        wind_dir_col="wind_direction_deg",
        min_obs=min_obs_per_hour,
        strict_wind_dir=strict_wind_dir,
    )

    # Align time ranges with reanalysis window
    _, _, trimmed = align_time_range(reanalysis_by_district, air_quality, weather_hourly.reset_index(), pm_detail)
    air_quality_trimmed, weather_hourly_trimmed, pm_detail_trimmed = trimmed

    main = merge_hourly_sources(reanalysis_by_district, weather_hourly_trimmed, pm_detail_trimmed)
    merged = merge_daily_air_quality(main, air_quality_trimmed)

    if output_csv is not None:
        output_csv = pathlib.Path(output_csv)
        merged.to_csv(output_csv, index=False)

    return merged


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def _parse_args(args: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build merged PM2.5 dataset")
    parser.add_argument("--air-quality", required=True, type=pathlib.Path, dest="air_quality")
    parser.add_argument("--pm-detail", required=True, type=pathlib.Path, dest="pm_detail")
    parser.add_argument("--weather", required=True, type=pathlib.Path)
    parser.add_argument("--reanalysis", required=True, type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, help="Path to save merged CSV")
    parser.add_argument("--min-obs", type=int, default=1, help="Minimum observations per hour window")
    parser.add_argument(
        "--strict-wind-dir",
        action="store_true",
        help="Use vector mean for wind direction (recommended)",
    )
    parser.add_argument(
        "--no-strict-wind-dir",
        dest="strict_wind_dir",
        action="store_false",
        help="Use arithmetic mean for wind direction",
    )
    parser.set_defaults(strict_wind_dir=True)
    return parser.parse_args(args=args)


def main(argv: Optional[Iterable[str]] = None) -> None:
    ns = _parse_args(argv)
    merged = run_pipeline(
        air_quality_path=ns.air_quality,
        pm_detail_path=ns.pm_detail,
        weather_path=ns.weather,
        reanalysis_path=ns.reanalysis,
        output_csv=ns.output,
        min_obs_per_hour=ns.min_obs,
        strict_wind_dir=ns.strict_wind_dir,
    )
    if ns.output:
        print(f"Merged dataset saved to {ns.output}")
    else:
        print(merged.head())


if __name__ == "__main__":
    main()
