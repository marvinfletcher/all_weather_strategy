"""技术信号：按大类资产规则计算月度趋势分数。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _discrete_sign(value: float) -> float:
    if value > 0:
        return 1.0
    if value < 0:
        return -1.0
    return 0.0


def _price_to_monthly(price: pd.Series) -> tuple[pd.Series, pd.Series]:
    series = price.dropna().sort_index()
    monthly_close = series.resample("ME").last()
    monthly_ret = monthly_close.pct_change(fill_method=None)

    period_index = monthly_close.index.to_period("M")
    monthly_close = monthly_close.copy()
    monthly_close.index = period_index
    monthly_ret = monthly_ret.copy()
    monthly_ret.index = period_index
    return monthly_close, monthly_ret


def _ewm_llt(monthly_price: pd.Series, gamma: float) -> pd.Series:
    return monthly_price.ewm(alpha=gamma, adjust=False).mean()


def _trend_score_equity(
    monthly_price: pd.Series,
    period: pd.Period,
    llt_gamma: float,
) -> float:
    if len(monthly_price) < 15:
        return 0.0

    llt = _ewm_llt(monthly_price, llt_gamma)
    llt_ret = llt.pct_change(fill_method=None)
    if period not in llt_ret.index or (period - 1) not in llt_ret.index:
        return 0.0
    if pd.isna(llt_ret.loc[period]) or pd.isna(llt_ret.loc[period - 1]):
        return 0.0

    short_avg = (float(llt_ret.loc[period]) + float(llt_ret.loc[period - 1])) / 2.0
    long_vals = []
    for offset in range(2, 13):
        prev_period = period - offset
        if prev_period not in llt_ret.index or pd.isna(llt_ret.loc[prev_period]):
            return 0.0
        long_vals.append(float(llt_ret.loc[prev_period]))
    long_avg = sum(long_vals) / len(long_vals)
    return _discrete_sign(short_avg - long_avg)


def _trend_score_bond(monthly_ret: pd.Series, period: pd.Period) -> float:
    if period not in monthly_ret.index or (period - 1) not in monthly_ret.index:
        return 0.0
    if pd.isna(monthly_ret.loc[period]) or pd.isna(monthly_ret.loc[period - 1]):
        return 0.0

    short_avg = (float(monthly_ret.loc[period]) + float(monthly_ret.loc[period - 1])) / 2.0
    long_vals = []
    for offset in range(2, 13):
        prev_period = period - offset
        if prev_period not in monthly_ret.index or pd.isna(monthly_ret.loc[prev_period]):
            return 0.0
        long_vals.append(float(monthly_ret.loc[prev_period]))
    long_avg = sum(long_vals) / len(long_vals)
    return _discrete_sign(short_avg - long_avg)


def _trend_score_gold_energy(monthly_ret: pd.Series, period: pd.Period) -> float:
    recent_vals = []
    for offset in range(0, 6):
        prev_period = period - offset
        if prev_period not in monthly_ret.index or pd.isna(monthly_ret.loc[prev_period]):
            return 0.0
        recent_vals.append(float(monthly_ret.loc[prev_period]))
    return _discrete_sign(sum(recent_vals) / len(recent_vals))


def _trend_score_soybean_metal(monthly_ret: pd.Series, period: pd.Period) -> float:
    if period not in monthly_ret.index or pd.isna(monthly_ret.loc[period]):
        return 0.0

    current = float(monthly_ret.loc[period])
    prev_vals = []
    for offset in range(1, 13):
        prev_period = period - offset
        if prev_period not in monthly_ret.index or pd.isna(monthly_ret.loc[prev_period]):
            return 0.0
        prev_vals.append(float(monthly_ret.loc[prev_period]))
    return _discrete_sign(current - sum(prev_vals) / len(prev_vals))


def _trend_score_for_asset_class(
    asset_class: str,
    monthly_price: pd.Series,
    monthly_ret: pd.Series,
    period: pd.Period,
    llt_gamma: float,
) -> float:
    asset_class = asset_class.lower().strip()
    if asset_class == "cash":
        return 0.0
    if asset_class == "equity":
        return _trend_score_equity(monthly_price, period, llt_gamma)
    if asset_class == "bond":
        return _trend_score_bond(monthly_ret, period)
    if asset_class in {"gold", "energy"}:
        return _trend_score_gold_energy(monthly_ret, period)
    if asset_class in {"soybean", "metal"}:
        return _trend_score_soybean_metal(monthly_ret, period)
    return 0.0


def compute_technical_scores(
    daily_price: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    universe: list[str],
    asset_class_map: dict[str, str],
    llt_gamma: float = 0.7,
    score_clip: float = 1.0,
) -> pd.DataFrame:
    """按资产类别计算技术分数，输出区间为 [-score_clip, score_clip]。"""
    price = daily_price.sort_index().copy()
    price.columns = price.columns.map(str)

    monthly_cache: dict[str, tuple[pd.Series, pd.Series]] = {}
    for asset in universe:
        asset_key = str(asset)
        if asset_key not in price.columns:
            continue
        monthly_cache[asset_key] = _price_to_monthly(price[asset_key])

    scores = pd.DataFrame(0.0, index=rebalance_dates, columns=universe)
    for date in rebalance_dates:
        period = pd.Timestamp(date).to_period("M")
        for asset in universe:
            asset_key = str(asset)
            if asset_key not in monthly_cache:
                continue
            asset_class = asset_class_map.get(asset_key)
            if asset_class is None:
                continue
            monthly_price, monthly_ret = monthly_cache[asset_key]
            raw = _trend_score_for_asset_class(
                asset_class,
                monthly_price,
                monthly_ret,
                period,
                llt_gamma,
            )
            scores.loc[date, asset] = float(np.clip(raw, -score_clip, score_clip))

    return scores
