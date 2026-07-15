"""战术调整：价格趋势分数。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strategic import normalize_weights


def trend_score(monthly_ret: pd.DataFrame, fast: int = 2, slow: int = 12) -> pd.DataFrame:
    """近期动量 vs 中期历史均值的符号分数。"""
    fast_ret = monthly_ret.rolling(fast).mean()
    slow_ret = monthly_ret.shift(fast).rolling(max(slow - fast, 1)).mean()
    score = np.sign(fast_ret - slow_ret)
    return score.replace(0, np.nan).ffill().fillna(0.0)


def apply_tactical_tilt(
    center: pd.Series,
    score_row: pd.Series,
    asset_config: pd.DataFrame,
    min_weight: float = 0.0,
    max_weight: float = 0.85,
) -> pd.Series:
    """在中枢权重上叠加趋势分数 * tilt_step，再归一化。"""
    step = asset_config.set_index("asset")["tilt_step"].reindex(center.index).fillna(0.0)
    raw = center + score_row.reindex(center.index).fillna(0.0) * step
    return normalize_weights(raw, min_weight=min_weight, max_weight=max_weight)
