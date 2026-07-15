"""战略权重：固定、逆波动、风险平价。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _coerce_bounds(
    index: pd.Index,
    value: float | pd.Series | None,
    default: float,
) -> pd.Series:
    if value is None:
        return pd.Series(float(default), index=index, dtype=float)
    if np.isscalar(value):
        return pd.Series(float(value), index=index, dtype=float)
    return pd.Series(value, dtype=float).reindex(index).fillna(float(default))


def project_weights_to_bounds(
    w: pd.Series,
    target_sum: float = 1.0,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    lower_bounds: pd.Series | None = None,
    upper_bounds: pd.Series | None = None,
    tol: float = 1e-10,
) -> pd.Series:
    """投影到带上下限的权重单纯形：sum=target_sum 且每个资产落在各自边界内。"""
    weights = w.astype(float).fillna(0.0)
    lower = _coerce_bounds(weights.index, lower_bounds, min_weight)
    upper = _coerce_bounds(weights.index, upper_bounds, max_weight)
    target_sum = float(target_sum)

    if (lower > upper + tol).any():
        bad = lower[lower > upper + tol].index.tolist()
        raise ValueError(f"部分资产的最小权重大于最大权重：{bad}")

    lower_sum = float(lower.sum())
    upper_sum = float(upper.sum())
    if lower_sum > target_sum + tol:
        raise ValueError(
            f"最小权重之和 {lower_sum:.6f} 大于目标总权重 {target_sum:.6f}，约束不可行。"
        )
    if upper_sum < target_sum - tol:
        raise ValueError(
            f"最大权重之和 {upper_sum:.6f} 小于目标总权重 {target_sum:.6f}，约束不可行。"
        )

    result = weights.clip(lower=lower, upper=upper)

    def _increase(delta: float) -> float:
        remaining = float(delta)
        for _ in range(len(result) + 5):
            if remaining <= tol:
                break
            room = upper - result
            free = room > tol
            if not free.any():
                break
            pref = (weights[free] - lower[free]).clip(lower=0.0)
            if float(pref.sum()) <= tol:
                pref = room[free]
            step = remaining * pref / float(pref.sum())
            step = np.minimum(step, room[free])
            result.loc[free] = result.loc[free] + step
            remaining -= float(step.sum())
        return remaining

    def _decrease(delta: float) -> float:
        remaining = float(delta)
        for _ in range(len(result) + 5):
            if remaining <= tol:
                break
            reducible = result - lower
            free = reducible > tol
            if not free.any():
                break
            pref = reducible[free]
            step = remaining * pref / float(pref.sum())
            step = np.minimum(step, reducible[free])
            result.loc[free] = result.loc[free] - step
            remaining -= float(step.sum())
        return remaining

    total = float(result.sum())
    if total < target_sum - tol:
        leftover = _increase(target_sum - total)
        if leftover > tol:
            room = upper - result
            asset = room.idxmax()
            if float(room.loc[asset]) < leftover - tol:
                raise ValueError("无法在最大权重约束内补足总权重到目标值。")
            result.loc[asset] += leftover
    elif total > target_sum + tol:
        leftover = _decrease(total - target_sum)
        if leftover > tol:
            reducible = result - lower
            asset = reducible.idxmax()
            if float(reducible.loc[asset]) < leftover - tol:
                raise ValueError("无法在最小权重约束内把总权重降回目标值。")
            result.loc[asset] -= leftover

    result = result.clip(lower=lower, upper=upper)
    total = float(result.sum())
    if abs(total - target_sum) > 1e-8:
        residual = target_sum - total
        slack = (upper - result) if residual > 0 else (result - lower)
        asset = slack.idxmax()
        if float(slack.loc[asset]) < abs(residual) - tol:
            raise ValueError("权重投影后仍无法满足目标总权重。")
        result.loc[asset] += residual

    final_sum = float(result.sum())
    if final_sum <= tol:
        raise ValueError("投影后的权重总和非正，无法归一化。")
    if abs(final_sum - target_sum) > 1e-8:
        result = result * (target_sum / final_sum)
    return result


def normalize_weights(
    w: pd.Series,
    target_sum: float = 1.0,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> pd.Series:
    """投影到 [min, max] 区间并保持总和为 target_sum。"""
    if len(w) == 0:
        return w.astype(float)
    total = float(pd.Series(w, dtype=float).fillna(0.0).sum())
    if total <= 0:
        w = pd.Series(float(target_sum) / len(w), index=w.index, dtype=float)
    return project_weights_to_bounds(
        w,
        target_sum=target_sum,
        min_weight=min_weight,
        max_weight=max_weight,
    )


def fixed_center_weights(asset_config: pd.DataFrame) -> pd.Series:
    """按 asset_config.center_weight 给出战略中枢权重并归一化。"""
    w = asset_config.set_index("asset")["center_weight"]
    return normalize_weights(w)


def inverse_vol_weights(ret_window: pd.DataFrame, vol_floor: float = 1e-6) -> pd.Series:
    """逆波动率权重。"""
    vol = ret_window.std().clip(lower=vol_floor)
    return normalize_weights(1 / vol)


def risk_contribution(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """各资产对组合方差的风险贡献。"""
    port_var = float(weights @ cov @ weights)
    if port_var <= 0:
        return np.ones_like(weights) / len(weights)
    marginal = cov @ weights
    return weights * marginal / port_var


def risk_parity_weights(
    ret_window: pd.DataFrame,
    max_iter: int = 3000,
    step: float = 0.05,
    tol: float = 1e-6,
) -> pd.Series:
    """迭代法近似等风险贡献权重，不依赖额外优化库。"""
    clean = ret_window.dropna(axis=1, how="all").fillna(0.0)
    cols = clean.columns
    n = len(cols)
    if n == 0:
        raise ValueError("收益率窗口为空，无法计算风险平价权重。")
    if len(clean) < 2:
        return pd.Series(np.ones(n) / n, index=cols)
    cov = clean.cov().values + np.eye(n) * 1e-8
    if not np.isfinite(cov).all():
        return pd.Series(np.ones(n) / n, index=cols)
    w = np.ones(n) / n
    target = np.ones(n) / n
    for _ in range(max_iter):
        rc = risk_contribution(w, cov)
        error = rc - target
        if np.max(np.abs(error)) < tol:
            break
        w = w * np.exp(-step * error)
        w = w / w.sum()
    return pd.Series(w, index=cols)


def split_risk_cash_weights(
    asset_config: pd.DataFrame,
    cash_asset: str | None,
) -> tuple[pd.Series, float]:
    """拆分风险资产中枢权重与现金基准权重。"""
    center = fixed_center_weights(asset_config)
    if not cash_asset:
        return center, 0.0
    if cash_asset not in center.index:
        raise ValueError(f"cash_asset `{cash_asset}` 不在资产配置中。")

    cash_weight = float(center.loc[cash_asset])
    risk_center = center.drop(index=cash_asset)
    if risk_center.empty:
        raise ValueError("至少需要一个非现金风险资产。")
    return normalize_weights(risk_center), cash_weight


def build_final_weight_bounds(
    assets: pd.Index,
    cash_asset: str | None,
    min_weight: float,
    max_weight: float,
) -> tuple[pd.Series, pd.Series]:
    """构建最终组合投影用的上下限：风险资产受限，现金只保留残差边界。"""
    lower = pd.Series(float(min_weight), index=assets, dtype=float)
    upper = pd.Series(float(max_weight), index=assets, dtype=float)
    if cash_asset and cash_asset in assets:
        lower.loc[cash_asset] = 0.0
        upper.loc[cash_asset] = 1.0
    return lower, upper
