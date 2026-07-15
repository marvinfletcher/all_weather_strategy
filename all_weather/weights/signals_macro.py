"""宏观信号：按规则把月频因子映射成资产分数。"""

from __future__ import annotations

from typing import Any

import pandas as pd


def parse_coef_and_months(
    impact_entry: Any,
    rule: dict,
    defaults: dict | None,
) -> tuple[float, int]:
    """解析 impact 中的 coef 与 trend_months。"""
    defaults = defaults or {}
    default_n = int(defaults.get("trend_months", 3))
    rule_n = rule.get("trend_months")
    fallback_n = int(rule_n) if rule_n is not None else default_n

    if isinstance(impact_entry, dict):
        coef = float(impact_entry["coef"])
        n_raw = impact_entry.get("trend_months", fallback_n)
        return coef, int(n_raw)
    return float(impact_entry), fallback_n


def max_trend_months_from_rules(
    rules: list[dict],
    defaults: dict | None,
) -> int:
    """规则里用到的最长宏观回看月数。"""
    defaults = defaults or {}
    max_n = int(defaults.get("trend_months", 3))
    for rule in rules:
        for entry in rule.get("impact", {}).values():
            _, n_months = parse_coef_and_months(entry, rule, defaults)
            max_n = max(max_n, n_months)
    return max_n


def macro_score_upper_bound(
    asset: str,
    rules: list[dict],
    defaults: dict | None,
) -> float:
    """资产宏观分数绝对值上界，用于后续归一化。"""
    total = 0.0
    for rule in rules:
        impact = rule.get("impact", {})
        if asset not in impact:
            continue
        coef, _ = parse_coef_and_months(impact[asset], rule, defaults)
        total += abs(float(coef))
    return total


def _to_period_frame(macro_monthly: pd.DataFrame) -> pd.DataFrame:
    frame = macro_monthly.sort_index().copy()
    if not isinstance(frame.index, pd.PeriodIndex):
        frame.index = pd.PeriodIndex(pd.to_datetime(frame.index).to_period("M"))
    return frame


def _trend_sign_monthly(
    series: pd.Series,
    period: pd.Period,
    trend_months: int,
) -> float:
    """N=1 看环比，N>=2 看当月相对前 N 月均值的符号。"""
    s = series.dropna().sort_index()
    if period not in s.index:
        return 0.0

    x_t = float(s.loc[period])
    n_months = int(trend_months)
    if n_months <= 0:
        return 0.0

    if n_months == 1:
        prev_period = period - 1
        if prev_period not in s.index:
            return 0.0
        diff = x_t - float(s.loc[prev_period])
    else:
        prev_vals = []
        for offset in range(1, n_months + 1):
            prev_period = period - offset
            if prev_period not in s.index:
                return 0.0
            prev_vals.append(float(s.loc[prev_period]))
        diff = x_t - sum(prev_vals) / len(prev_vals)

    if diff > 0:
        return 1.0
    if diff < 0:
        return -1.0
    return 0.0


def compute_macro_scores(
    macro_monthly: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    rules: list[dict],
    defaults: dict | None = None,
) -> pd.DataFrame:
    """按宏观规则为每个调仓日、每个资产计算累计分数。"""
    defaults = defaults or {}
    required_cols = {rule["factor"] for rule in rules}
    missing = required_cols - set(macro_monthly.columns)
    if missing:
        raise ValueError("宏观宽表缺少因子列：" + ", ".join(sorted(missing)))

    macro_period = _to_period_frame(macro_monthly)
    assets: set[str] = set()
    for rule in rules:
        assets.update(rule.get("impact", {}).keys())
    scores = pd.DataFrame(0.0, index=rebalance_dates, columns=sorted(assets))

    factor_series = {
        factor: pd.to_numeric(macro_period[factor], errors="coerce")
        for factor in required_cols
    }

    for date in rebalance_dates:
        period = pd.Timestamp(date).to_period("M")
        for rule in rules:
            factor = rule["factor"]
            series = factor_series[factor]
            for asset, entry in rule.get("impact", {}).items():
                coef, n_months = parse_coef_and_months(entry, rule, defaults)
                sign = _trend_sign_monthly(series, period, n_months)
                scores.loc[date, asset] += sign * coef

    return scores
