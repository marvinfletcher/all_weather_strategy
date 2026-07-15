"""端到端流水线编排。"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .backtest import run_portfolio, run_performance
from .config import BacktestConfig
from .data.rq_provider import RQDataProvider
from .flow import build_weight_schedule, make_flow_data_from_weights
from .market import calc_returns, to_framework_market_data, to_period_price
from .performance import plot_nav_and_drawdown, simple_performance_stats


def run_backtest(cfg: BacktestConfig, provider: Optional[RQDataProvider] = None) -> dict:
    """执行完整流水线：取数 -> 行情标准化 -> 收益率 -> 权重 -> 流水 -> 回测 -> 绩效。

    provider 可外部传入以复用 RQ 连接（notebook 场景）。None 时按 RQ_LICENSE 环境变量新建。
    """
    if provider is None:
        provider = RQDataProvider()

    order_book_ids = [a.order_book_id for a in cfg.assets]
    daily_price = provider.get_price_panel(
        order_book_ids, cfg.start_date, cfg.end_date_str, cfg.price_field
    )
    name_map = {a.order_book_id: a.asset for a in cfg.assets}
    daily_price = daily_price.rename(columns=name_map).dropna(how="all").ffill()
    # 资产缺失校验由 RQDataProvider.get_price_panel 负责（CLI 与 notebook 共用）

    market_data = to_framework_market_data(daily_price, cfg.asset_df)
    monthly_price = to_period_price(daily_price, cfg.rebalance_freq)
    monthly_ret = calc_returns(monthly_price)

    weight_schedule = build_weight_schedule(
        monthly_ret,
        cfg.asset_df,
        strategy=cfg.strategy,
        lookback=cfg.lookback,
        trend_fast=cfg.trend_fast,
        trend_slow=cfg.trend_slow,
        min_weight=cfg.weight_bounds.min,
        max_weight=cfg.weight_bounds.max,
    )
    flow_data = make_flow_data_from_weights(
        weight_schedule,
        market_data,
        cfg.asset_df,
        initial_cash=cfg.initial_cash,
        fee_rate=cfg.trade_fee_rate,
    )

    results = run_portfolio(
        market_data=market_data,
        flow_data=flow_data,
        position_data=None,
        margin=cfg.backtest.margin,
        allow_loan=cfg.backtest.allow_loan,
        weight_type=cfg.backtest.weight_type,
    )
    # run_portfolio 单策略输入时返回 tuple；批量输入返回 dict
    if isinstance(results, tuple):
        backtest_result, position_result, flow_data_new, _ = results
        results_map = {"strategy_0": {
            "backtest_result": backtest_result,
            "position_result": position_result,
            "flow_data_new": flow_data_new,
        }}
    else:
        results_map = results

    return {
        "cfg": cfg,
        "daily_price": daily_price,
        "market_data": market_data,
        "monthly_ret": monthly_ret,
        "weight_schedule": weight_schedule,
        "flow_data": flow_data,
        "results": results_map,
    }


def analyze_results(
    run_output: dict,
    strategy_key: Optional[str] = None,
) -> dict:
    """对 run_backtest 的输出做绩效分析。

    优先使用 backtest.run_performance（需要 plotly/rqdatac）；不可用时回退到
    simple_performance_stats + matplotlib 绘图。
    """
    results_map = run_output["results"]
    if strategy_key is None:
        strategy_key = next(iter(results_map))
    bt_result = results_map[strategy_key]["backtest_result"]
    cfg = run_output["cfg"]

    if run_performance is not None:
        try:
            analysis = run_performance(
                bt_result,
                label=strategy_key,
                benchmark_code=cfg.performance.benchmark_code,
                free_rate=cfg.performance.free_rate,
                plot_periodic=cfg.performance.plot,
                heatmap_periodic=cfg.performance.plot,
                draw_npv=cfg.performance.plot,
                draw_asset=cfg.performance.plot,
                pretty_print_kpi=True,
            )
            return {"engine": "run_performance", "analysis": analysis, "backtest_result": bt_result}
        except Exception as exc:
            print(f"run_performance 执行失败，回退到 simple_performance_stats。原因：{type(exc).__name__}: {exc}")

    stats = simple_performance_stats(bt_result)
    print(stats.to_string())
    if cfg.performance.plot:
        plot_nav_and_drawdown(bt_result, label=strategy_key)
    return {"engine": "simple", "stats": stats, "backtest_result": bt_result}
