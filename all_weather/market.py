"""行情标准化与收益率计算。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def to_framework_market_data(price: pd.DataFrame, asset_config: pd.DataFrame) -> pd.DataFrame:
    """宽表价格 -> 回测框架标准长表 (datetime, code, open/high/low/close, volume)。

    price: 宽表，列名为资产名称（asset），索引为日期。
    asset_config: 至少包含 `asset` 与 `order_book_id` 两列。
    """
    code_map = asset_config.set_index("asset")["order_book_id"].to_dict()
    long_price = (
        price.rename(columns=code_map)
        .stack(dropna=True)
        .rename("close_price")
        .reset_index()
        .rename(columns={"date": "datetime", "order_book_id": "code"})
    )
    long_price["datetime"] = pd.to_datetime(long_price["datetime"])
    long_price["code"] = long_price["code"].astype(str)
    long_price["open_price"] = long_price["close_price"]
    long_price["high_price"] = long_price["close_price"]
    long_price["low_price"] = long_price["close_price"]
    long_price["volume"] = 0
    return long_price[
        ["datetime", "code", "open_price", "high_price", "low_price", "close_price", "volume"]
    ]


def to_period_price(price: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    """按调仓频率重采样为周期收盘价。"""
    return price.resample(freq).last().dropna(how="all")


def calc_returns(price: pd.DataFrame) -> pd.DataFrame:
    """ pct_change 收益率，剔除 inf。"""
    return price.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")
