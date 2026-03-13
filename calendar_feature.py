"""节假日日历与日级时间特征工程（CSV版本）。

本模块仅基于结构化 `holiday.csv` 构建节假日日历与日级特征：
1) holiday.csv 解析
2) holiday calendar 构建
3) merged_dataset.date00 日级特征生成
4) 输出一致性校验

说明：
- 不再包含半结构化节假日文本解析逻辑。
- 周/月周期特征与高阶 Fourier 已移除。
- 仅保留年周期 Fourier 一阶特征：`cal_yearly_sin_1`, `cal_yearly_cos_1`。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _strip_brackets(text: str) -> str:
    """去除中英文括号中的说明文本。"""

    out = re.sub(r"（[^）]*）", "", str(text))
    out = re.sub(r"\([^)]*\)", "", out)
    return out


def _to_ts(year: int, month: int, day: int) -> pd.Timestamp:
    """构造标准化日期（00:00:00）。"""

    return pd.Timestamp(year=year, month=month, day=day).normalize()


def _parse_date_token(token: str, fallback_year: int, fallback_month: Optional[int] = None) -> pd.Timestamp:
    """解析日期 token（支持 YYYY年M月D日、M月D日、D日）。"""

    cleaned = _strip_brackets(token)
    m = re.search(r"(?:(\d{4})年)?(?:(\d{1,2})月)?(\d{1,2})日", cleaned)
    if not m:
        raise ValueError(f"无法解析日期 token: {token}")

    y_raw, mo_raw, d_raw = m.groups()
    year = int(y_raw) if y_raw else fallback_year
    month = int(mo_raw) if mo_raw else fallback_month
    day = int(d_raw)
    if month is None:
        raise ValueError(f"日期 token 缺少月份: {token}")
    return _to_ts(year, month, day)


def parse_holiday_csv(csv_source: str | Path | pd.DataFrame, years: Optional[List[int]] = None) -> pd.DataFrame:
    """从结构化 holiday.csv 解析节假日规则。"""

    if isinstance(csv_source, pd.DataFrame):
        raw = csv_source.copy()
    else:
        raw = pd.read_csv(csv_source)

    if "节日名称" not in raw.columns:
        raise KeyError("holiday.csv 缺少 `节日名称` 列")

    year_cols = [c for c in raw.columns if str(c).isdigit()]
    parsed_years = sorted(int(c) for c in year_cols)
    if years is None:
        years = parsed_years

    adjust_row = raw.loc[raw["节日名称"].astype(str).str.strip().eq("调休")]
    adjusted_by_year: Dict[int, List[pd.Timestamp]] = {}
    for y in years:
        col = str(y)
        if adjust_row.empty or col not in raw.columns:
            adjusted_by_year[y] = []
            continue
        adjust_text = str(adjust_row.iloc[0][col]).strip()
        if adjust_text == "" or adjust_text.lower() == "nan":
            adjusted_by_year[y] = []
            continue
        pieces = [p for p in re.split(r"[、,，]", adjust_text) if p]
        adjusted_by_year[y] = sorted({_parse_date_token(p, fallback_year=y) for p in pieces})

    holiday_rows = raw.loc[~raw["节日名称"].astype(str).str.strip().eq("调休")].copy()
    records: List[dict] = []

    for y in years:
        col = str(y)
        if col not in holiday_rows.columns:
            continue

        for _, r in holiday_rows.iterrows():
            name = str(r["节日名称"]).strip()
            cell = str(r[col]).strip()
            if cell == "" or cell.lower() == "nan":
                continue

            cell_clean = _strip_brackets(cell).replace("—", "–").replace("-", "–").replace("至", "–")
            if "–" in cell_clean:
                left, right = [s.strip() for s in cell_clean.split("–", 1)]
                start = _parse_date_token(left, fallback_year=y)
                end = _parse_date_token(right, fallback_year=start.year, fallback_month=start.month)
            else:
                start = _parse_date_token(cell_clean, fallback_year=y)
                end = start

            records.append(
                {
                    "rule_year": y,
                    "holiday_name": name,
                    "holiday_start": start,
                    "holiday_end": end,
                    "adjusted_workdays": adjusted_by_year.get(y, []),
                }
            )

    return pd.DataFrame(records)


def build_holiday_calendar(
    parsed_rules: pd.DataFrame,
    start_date: str | pd.Timestamp = "2020-01-01",
    end_date: str | pd.Timestamp = "2025-12-31",
) -> pd.DataFrame:
    """根据解析规则生成完整按天 holiday calendar。"""

    start_ts = pd.to_datetime(start_date).normalize()
    end_ts = pd.to_datetime(end_date).normalize()

    cal = pd.DataFrame({"date": pd.date_range(start_ts, end_ts, freq="D")})
    cal["year"] = cal["date"].dt.year
    cal["holiday_name"] = pd.NA
    cal["is_holiday"] = 0
    cal["is_adjusted_workday"] = 0

    rules_df = parsed_rules.copy()
    rules_df["holiday_start"] = pd.to_datetime(rules_df["holiday_start"]).dt.normalize()
    rules_df["holiday_end"] = pd.to_datetime(rules_df["holiday_end"]).dt.normalize()

    for _, row in rules_df.iterrows():
        mask = cal["date"].between(row["holiday_start"], row["holiday_end"])
        cal.loc[mask, "is_holiday"] = 1

        new_name = str(row["holiday_name"])
        target_idx = cal.index[mask]
        existing = cal.loc[target_idx, "holiday_name"].astype("string").fillna("")
        merged = existing.where(existing.eq(""), existing + "|" + new_name)
        merged = merged.where(~existing.eq(new_name), existing)
        merged = merged.where(~existing.eq(""), new_name)
        cal.loc[target_idx, "holiday_name"] = merged.values

        workdays = row["adjusted_workdays"]
        if not isinstance(workdays, list):
            workdays = [] if pd.isna(workdays) else list(workdays)
        for wd in workdays:
            wd_ts = pd.to_datetime(wd).normalize()
            cal.loc[cal["date"].eq(wd_ts), "is_adjusted_workday"] = 1

    is_weekend = cal["date"].dt.dayofweek >= 5
    cal["day_type"] = np.where(is_weekend, "rest_day", "workday")
    cal.loc[cal["is_holiday"].eq(1), "day_type"] = "holiday"
    cal.loc[cal["is_adjusted_workday"].eq(1), "day_type"] = "workday"

    overlap = cal["is_holiday"].eq(1) & cal["is_adjusted_workday"].eq(1)
    cal.loc[overlap, "is_holiday"] = 0
    cal.loc[overlap, "holiday_name"] = pd.NA

    return cal.sort_values("date").reset_index(drop=True)


def build_holiday_calendar_from_csv(
    csv_path: str | Path | pd.DataFrame,
    start_date: str | pd.Timestamp = "2020-01-01",
    end_date: str | pd.Timestamp = "2025-12-31",
    years: Optional[List[int]] = None,
) -> pd.DataFrame:
    """从 holiday.csv 直接生成按天 holiday calendar。"""

    parsed_rules = parse_holiday_csv(csv_source=csv_path, years=years)
    return build_holiday_calendar(parsed_rules=parsed_rules, start_date=start_date, end_date=end_date)


def build_calendar_features(
    merged_dataset: pd.DataFrame,
    holiday_calendar: pd.DataFrame,
    fourier_origin: str | pd.Timestamp = "2020-01-01",
) -> pd.DataFrame:
    """根据 merged_dataset.date00 生成日级日历特征（仅保留年周期 Fourier 一阶）。

    Parameters
    ----------
    fourier_origin : str | pd.Timestamp, default \"2020-01-01\"
        年周期 Fourier 的固定时间原点。使用固定原点可避免训练/验证/测试
        分开跑特征时出现整体相位偏移。
    """

    src = merged_dataset.copy()
    if "date00" not in src.columns:
        raise KeyError("merged_dataset 缺少 date00 列")

    dates = pd.DataFrame({"date00": pd.to_datetime(src["date00"], errors="coerce").dt.normalize()})
    dates = dates.dropna().drop_duplicates().sort_values("date00").reset_index(drop=True)

    cal = holiday_calendar.copy()
    cal["date"] = pd.to_datetime(cal["date"]).dt.normalize()

    feat = dates.merge(cal, left_on="date00", right_on="date", how="left", validate="one_to_one")
    feat = feat.drop(columns=["date"])

    feat["cal_year"] = feat["date00"].dt.year
    feat["cal_month"] = feat["date00"].dt.month
    feat["cal_quarter"] = feat["date00"].dt.quarter
    feat["cal_dayofweek"] = feat["date00"].dt.dayofweek
    feat["cal_dayofyear"] = feat["date00"].dt.dayofyear

    feat["cal_is_weekend"] = (feat["cal_dayofweek"] >= 5).astype(int)
    feat["cal_is_holiday"] = feat["is_holiday"].fillna(0).astype(int)
    feat["cal_is_adjusted_workday"] = feat["is_adjusted_workday"].fillna(0).astype(int)
    feat["cal_is_rest_day"] = feat["day_type"].eq("rest_day").fillna(False).astype(int)
    holiday_name = feat["holiday_name"].fillna("").astype("string")
    feat["cal_holiday_name"] = pd.Categorical(holiday_name)

    # 年周期 Fourier（一阶），使用固定时间原点，避免子集单独计算时相位漂移
    origin_ts = pd.to_datetime(fourier_origin).normalize()
    t = (feat["date00"] - origin_ts).dt.days.astype(float)
    angle = 2.0 * np.pi * t / 365.25
    feat["cal_yearly_sin_1"] = np.sin(angle)
    feat["cal_yearly_cos_1"] = np.cos(angle)

    keep_cols = [
        "date00",
        "cal_year",
        "cal_month",
        "cal_quarter",
        "cal_dayofweek",
        "cal_dayofyear",
        "cal_is_weekend",
        "cal_is_holiday",
        "cal_is_adjusted_workday",
        "cal_is_rest_day",
        "cal_holiday_name",
        "cal_yearly_sin_1",
        "cal_yearly_cos_1",
    ]
    return feat[keep_cols].sort_values("date00").reset_index(drop=True)


def build_calendar_features_from_csv(
    merged_dataset: pd.DataFrame,
    holiday_csv_path: str | Path | pd.DataFrame = "holiday.csv",
    start_date: str | pd.Timestamp = "2020-01-01",
    end_date: str | pd.Timestamp = "2025-12-31",
    fourier_origin: str | pd.Timestamp = "2020-01-01",
    years: Optional[List[int]] = None,
) -> pd.DataFrame:
    """一体化入口：`merged_dataset + holiday.csv -> 日级日历特征表`。"""

    holiday_calendar = build_holiday_calendar_from_csv(
        csv_path=holiday_csv_path, start_date=start_date, end_date=end_date, years=years
    )
    return build_calendar_features(
        merged_dataset=merged_dataset,
        holiday_calendar=holiday_calendar,
        fourier_origin=fourier_origin,
    )


def validate_calendar_outputs(
    merged_dataset: pd.DataFrame,
    holiday_calendar: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> Dict[str, bool]:
    """执行基础一致性验证。"""

    checks: Dict[str, bool] = {}
    checks["feature_date_unique"] = feature_df["date00"].is_unique

    valid_types = {"holiday", "workday", "rest_day"}
    checks["day_type_values_valid"] = holiday_calendar["day_type"].isin(valid_types).all()

    one_hot_sum = (
        holiday_calendar["day_type"].eq("holiday").astype(int)
        + holiday_calendar["day_type"].eq("workday").astype(int)
        + holiday_calendar["day_type"].eq("rest_day").astype(int)
    )
    checks["day_type_mutually_exclusive_complete"] = one_hot_sum.eq(1).all()

    overlap = holiday_calendar["is_holiday"].eq(1) & holiday_calendar["is_adjusted_workday"].eq(1)
    checks["holiday_adjusted_consistent"] = (~overlap).all()

    src_dates = pd.to_datetime(merged_dataset["date00"], errors="coerce").dt.normalize().dropna().unique()
    feat_dates = pd.to_datetime(feature_df["date00"], errors="coerce").dt.normalize().dropna().unique()
    checks["feature_covers_all_merged_dates"] = set(src_dates).issubset(set(feat_dates))

    return checks


def minimal_example() -> None:
    """最小示例：基于 holiday.csv 演示完整流程。"""

    holiday_cal = build_holiday_calendar_from_csv("holiday.csv", "2020-01-01", "2025-12-31")

    merged_dataset = pd.DataFrame(
        {
            "datetime": pd.date_range("2023-09-25 00:00:00", "2023-10-10 23:00:00", freq="h"),
            "district": "福田区",
            "PM25": 35.0,
        }
    )
    merged_dataset["date00"] = pd.to_datetime(merged_dataset["datetime"]).dt.floor("D")

    features = build_calendar_features(merged_dataset=merged_dataset, holiday_calendar=holiday_cal)
    checks = validate_calendar_outputs(merged_dataset, holiday_cal, features)

    print("[features]", features.shape)
    print(features.head(5))
    print("\n[validation]")
    for k, v in checks.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    minimal_example()
