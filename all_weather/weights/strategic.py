"""战略权重：固定、逆波动、风险平价。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_weights(
    w: pd.Series,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> pd.Series:
    """裁剪到 [min, max] 后归一化到和为 1。"""
    w = w.astype(float).clip(lower=min_weight, upper=max_weight)
    total = w.sum()
    if total <= 0:
        return pd.Series(1 / len(w), index=w.index)
    return w / total


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
    cov = clean.cov().values + np.eye(n) * 1e-8
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
