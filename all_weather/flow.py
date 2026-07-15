"""目标权重表生成 + 交易流水生成。"""

from __future__ import annotations

import pandas as pd

from .weights.strategic import (
    fixed_center_weights,
    inverse_vol_weights,
    normalize_weights,
    risk_parity_weights,
)
from .weights.tactical import apply_tactical_tilt, trend_score


FLOW_COLUMNS = [
    "买卖日期",
    "证券代码",
    "买卖数量",
    "买卖权重",
    "买卖价格",
    "买卖收益",
    "保证金比例",
    "买卖方向",
    "盈利方向",
    "交易费用",
    "交易批次",
]

DIRECTION_BUY = "买入"
DIRECTION_SELL = "卖出"
DIRECTION_TRANSFER_IN = "划入"


def build_weight_schedule(
    monthly_ret: pd.DataFrame,
    asset_config: pd.DataFrame,
    strategy: str = "fixed_plus_tilt",
    lookback: int = 36,
    trend_fast: int = 2,
    trend_slow: int = 12,
    min_weight: float = 0.0,
    max_weight: float = 0.85,
) -> pd.DataFrame:
    """按调仓日生成目标权重表 (index=signal_date, columns=asset)。"""
    center = fixed_center_weights(asset_config).reindex(monthly_ret.columns).fillna(0.0)
    score_panel = trend_score(monthly_ret, fast=trend_fast, slow=trend_slow).reindex(
        columns=monthly_ret.columns
    ).fillna(0.0)
    weights = []

    for dt in monthly_ret.index:
        hist = monthly_ret.loc[:dt].tail(lookback)
        if strategy == "fixed":
            w = center.copy()
        elif strategy == "inverse_vol":
            w = inverse_vol_weights(hist).reindex(monthly_ret.columns).fillna(0.0)
        elif strategy == "risk_parity":
            w = risk_parity_weights(hist).reindex(monthly_ret.columns).fillna(0.0)
            w = normalize_weights(w, min_weight=min_weight, max_weight=max_weight)
        elif strategy == "fixed_plus_tilt":
            w = apply_tactical_tilt(
                center,
                score_panel.loc[dt],
                asset_config,
                min_weight=min_weight,
                max_weight=max_weight,
            )
        else:
            raise ValueError(f"未知策略：{strategy}")
        weights.append(w.rename(dt))

    return pd.DataFrame(weights).reindex(columns=monthly_ret.columns).fillna(0.0)


def price_lookup_from_market_data(market_data: pd.DataFrame) -> pd.DataFrame:
    """长表行情 -> 宽表收盘价 (index=datetime, columns=code)。"""
    return market_data.pivot(index="datetime", columns="code", values="close_price").sort_index().ffill()


def align_trade_date(signal_date: pd.Timestamp, trade_calendar: pd.Index) -> pd.Timestamp:
    """把信号日对齐到 >= signal_date 的最近交易日。"""
    pos = trade_calendar.searchsorted(pd.Timestamp(signal_date), side="left")
    if pos >= len(trade_calendar):
        raise ValueError(f"调仓日期 {signal_date.date()} 超出行情范围。")
    return trade_calendar[pos]


def truncate_weights_to_market(
    target_weights: pd.DataFrame,
    trade_calendar: pd.Index,
) -> pd.DataFrame:
    """截断无法对齐到有效交易日的调仓信号（超出行情上界）。"""
    if target_weights.empty or trade_calendar.empty:
        return target_weights

    signal_dates = pd.DatetimeIndex(target_weights.index)
    pos = trade_calendar.searchsorted(signal_dates, side="left")
    truncated = target_weights[pos < len(trade_calendar)]
    dropped = len(target_weights) - len(truncated)
    if dropped:
        print(
            f"已截断 {dropped} 条超出行情范围 "
            f"[{trade_calendar[0].date()}, {trade_calendar[-1].date()}] 的调仓信号。"
        )
    return truncated


def make_flow_data_from_weights(
    target_weights: pd.DataFrame,
    market_data: pd.DataFrame,
    asset_config: pd.DataFrame,
    initial_cash: float = 10_000_000,
    fee_rate: float = 0.0002,
) -> pd.DataFrame:
    """目标权重表 -> 回测框架交易流水（中文列名）。

    - 首条流水为 `划入`，建立初始资金。
    - 权重上升生成 `买入`，权重下降生成 `卖出`。
    - `买卖权重` 使用权重差，实际成交数量由 run_portfolio(weight_type="asset") 换算。
    - 标的上市晚于 start_date 等导致某调仓日价格缺失时，自动忽略该标的本次调仓
      （不生成流水、previous 保持不变），对应资金暂留现金；待价格可用后按 delta 自动建仓。
    """
    price_panel = price_lookup_from_market_data(market_data)
    trade_calendar = price_panel.index
    target_weights = truncate_weights_to_market(target_weights, trade_calendar)
    code_map = asset_config.set_index("asset")["order_book_id"].to_dict()

    rows = [
        {
            "买卖日期": trade_calendar[0],
            "证券代码": "CASH",
            "买卖数量": float(initial_cash),
            "买卖权重": 0.0,
            "买卖价格": 1.0,
            "买卖收益": 0.0,
            "保证金比例": 1.0,
            "买卖方向": DIRECTION_TRANSFER_IN,
            "盈利方向": 1,
            "交易费用": 0.0,
            "交易批次": 0,
        }
    ]

    previous = pd.Series(0.0, index=target_weights.columns)
    batch = 1
    # 记录已提示过"数据缺失"的标的，避免每个调仓日重复打印
    warned_missing: set[str] = set()
    for signal_date, target in target_weights.iterrows():
        trade_date = align_trade_date(signal_date, trade_calendar)
        target = normalize_weights(target.reindex(previous.index).fillna(0.0))

        # 判定该交易日可交易的标的：code 存在且 trade_date 价格有效（非 NaN、> 0）。
        # 标的上市晚于 start_date 时，早期 trade_date 价格为 NaN（ffill 无法填充前导 NaN），
        # 此处自动忽略这些标的——不为其生成流水，其目标权重对应的资金暂时留在现金，
        # 待该标的价格可用后，相对 previous（保持不变）的 delta 会自动生成建仓流水。
        valid = pd.Series(False, index=target.index)
        for asset in target.index:
            code = code_map.get(asset)
            if code is None or code not in price_panel.columns:
                continue
            px = price_panel.loc[trade_date, code]
            if pd.notna(px) and float(px) > 0:
                valid[asset] = True
            elif asset not in warned_missing:
                warned_missing.add(asset)
                print(
                    f"【提示】标的 {asset} ({code}) 在 {trade_date.date()} 无有效价格，"
                    f"本次调仓忽略，待数据可用后自动建仓。"
                )

        delta = target - previous
        # 仅对可交易且 delta 非零的标的学生成流水；不可交易标的不生成买卖流水。
        sells = delta[valid & (delta < -1e-8)].sort_values()
        buys = delta[valid & (delta > 1e-8)].sort_values(ascending=False)
        for direction, part in [(DIRECTION_SELL, sells), (DIRECTION_BUY, buys)]:
            for asset, diff in part.items():
                code = code_map[asset]
                price = float(price_panel.loc[trade_date, code])
                weight = abs(float(diff))
                rows.append(
                    {
                        "买卖日期": trade_date,
                        "证券代码": code,
                        "买卖数量": 0.0,
                        "买卖权重": weight,
                        "买卖价格": price,
                        "买卖收益": 0.0,
                        "保证金比例": 1.0,
                        "买卖方向": direction,
                        "盈利方向": 1,
                        "交易费用": initial_cash * weight * fee_rate,
                        "交易批次": batch,
                    }
                )
        # previous 仅更新可交易标的；不可交易标的 previous 保持不变，
        # 下次其可用时 delta = target - previous(旧值) 即为待建仓权重差。
        previous[valid] = target[valid]
        batch += 1

    flow_data = pd.DataFrame(rows, columns=FLOW_COLUMNS)
    flow_data = flow_data.sort_values(
        ["买卖日期", "交易批次", "买卖方向", "证券代码"]
    ).reset_index(drop=True)
    return flow_data
