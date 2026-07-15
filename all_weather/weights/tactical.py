"""战术调整：仿照 gf_asset_alloc 的宏观/技术总分调权。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .signals_macro import macro_score_upper_bound


def apply_score_tilt(
    base_weights: pd.Series,
    macro_score: pd.Series,
    tech_score: pd.Series,
    tactical_adjust_caps: dict[str, float],
    cash_asset: str,
    macro_rules: list[dict] | None = None,
    macro_defaults: dict | None = None,
    tech_score_clip_for_norm: float = 1.0,
) -> pd.Series:
    """在战略权重基础上，按 gf_asset_alloc 同款规则调整风险资产并让现金吸收差额。"""
    weights = base_weights.astype(float).copy()
    if cash_asset not in weights.index:
        raise ValueError(f"战术配置中的 cash_asset 不在资产列表内：{cash_asset}")

    macro_rules = macro_rules or []
    risk_assets = [asset for asset in weights.index if asset != cash_asset]
    out: dict[str, float] = {}

    for asset in risk_assets:
        cap = float(tactical_adjust_caps.get(asset, 0.0))
        base = float(weights.get(asset, 0.0))
        macro_value = float(macro_score.get(asset, 0.0))
        tech_value = float(tech_score.get(asset, 0.0))
        total = macro_value + tech_value
        max_abs = max(
            macro_score_upper_bound(asset, macro_rules, macro_defaults)
            + float(tech_score_clip_for_norm),
            1e-9,
        )
        norm_score = float(np.clip(total / max_abs, -1.0, 1.0))
        out[asset] = max(0.0, base + norm_score * cap)

    risk_sum = sum(out.values())
    if risk_sum > 1.0:
        scale = 1.0 / risk_sum
        out = {asset: value * scale for asset, value in out.items()}
        cash_weight = 0.0
    else:
        cash_weight = max(0.0, 1.0 - risk_sum)

    adjusted = pd.Series(out, dtype=float)
    adjusted[cash_asset] = cash_weight
    return adjusted.reindex(weights.index).fillna(0.0)
