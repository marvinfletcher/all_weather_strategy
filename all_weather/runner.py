"""端到端流水线编排。

这里把“本地 CSV -> 信号 -> 权重 -> 流水 -> 回测结果”串成一条稳定主链，
CLI 和 Notebook/脚本调试时都应优先复用这里，而不是各自复制流程。
"""

from __future__ import annotations

from dataclasses import replace
import pandas as pd
from typing import Optional

from .backtest import run_portfolio, run_performance
from .config import BacktestConfig
from .data_loader import load_factor_panel_from_csv, load_price_panel_from_csv
from .flow import align_trade_date, build_weight_schedule, make_flow_data_from_weights
from .market import calc_returns, to_framework_market_data, to_period_price
from .weights.signals_macro import compute_macro_scores, max_trend_months_from_rules
from .weights.signals_technical import compute_technical_scores


def _load_daily_price(cfg: BacktestConfig) -> pd.DataFrame:
    order_book_ids = [a.order_book_id for a in cfg.assets]
    csv_path = cfg.data.resolve_csv_path(cfg.project_root)
    return load_price_panel_from_csv(
        csv_path,
        order_book_ids,
        start_date=cfg.start_date,
        end_date=cfg.end_date_str,
    )


def _load_factor_data(cfg: BacktestConfig) -> pd.DataFrame:
    factor_csv_path = cfg.data.resolve_factor_csv_path(cfg.project_root)
    return load_factor_panel_from_csv(
        factor_csv_path,
        start_date=cfg.start_date,
        end_date=cfg.end_date_str,
    )


def _resolve_signal_dates(
    monthly_ret: pd.DataFrame,
    factor_data: pd.DataFrame,
    cfg: BacktestConfig,
) -> pd.DatetimeIndex:
    """确定允许生成信号的日期区间。

    宏观规则通常需要若干个月历史窗口，如果回测一开始就直接打分，
    很容易因为同比、滞后或趋势窗口不足而得到不稳定信号。
    这里统一把 signal_dates 向后截到“宏观数据已具备足够预热长度”的位置。
    """
    signal_dates = pd.DatetimeIndex(monthly_ret.index)
    if not cfg.tactical.enabled or not cfg.macro_rules:
        return signal_dates

    max_n = max_trend_months_from_rules(cfg.macro_rules, cfg.macro_defaults)
    factor_index = factor_data.sort_index().index
    if len(factor_index) > max_n:
        start_date = pd.Timestamp(factor_index[max_n])
        signal_dates = signal_dates[signal_dates >= start_date]
    return signal_dates


def _build_local_benchmark_npv(
    raw_daily_price: pd.DataFrame,
    benchmark_code: str | None,
    start_date: pd.Timestamp | None = None,
) -> pd.Series | None:
    if not benchmark_code:
        return None

    code = str(benchmark_code).strip()
    if code not in raw_daily_price.columns.astype(str):
        return None

    close = raw_daily_price[code].dropna().astype(float).sort_index()
    if start_date is not None:
        close = close.loc[close.index >= pd.Timestamp(start_date)]
    if len(close) < 2:
        return None

    benchmark_npv = (1.0 + close.pct_change(fill_method=None).fillna(0.0)).cumprod()
    benchmark_npv.iloc[0] = 1.0
    benchmark_npv.name = code
    return benchmark_npv


def _resolve_effective_start_date(
    weight_schedule: pd.DataFrame,
    market_data: pd.DataFrame,
) -> pd.Timestamp:
    """仿照 gf_asset_alloc：正式回测从首个有效调仓对应的交易日开始。"""
    if weight_schedule.empty:
        raise ValueError("目标权重表为空，无法确定正式回测起点。")

    trade_calendar = pd.DatetimeIndex(
        pd.to_datetime(market_data["datetime"]).drop_duplicates().sort_values()
    )
    if trade_calendar.empty:
        raise ValueError("行情为空，无法确定正式回测起点。")

    first_signal_date = pd.Timestamp(weight_schedule.index.min())
    return align_trade_date(first_signal_date, trade_calendar)


def _prepare_backtest_inputs(cfg: BacktestConfig) -> dict:
    """准备与 strategy 无关的共享输入。"""
    raw_daily_price = _load_daily_price(cfg)
    factor_data = _load_factor_data(cfg)
    name_map = {a.order_book_id: a.asset for a in cfg.assets}
    # 后续权重、信号、交易流水都以“资产中文名”为列名协作，
    # 只在最终落到回测流水时再映射回 order_book_id。
    daily_price = raw_daily_price.rename(columns=name_map).dropna(how="all").ffill()

    market_data = to_framework_market_data(daily_price, cfg.asset_df)
    monthly_price = to_period_price(daily_price, cfg.rebalance_freq)
    monthly_ret_all = calc_returns(monthly_price)
    signal_dates = _resolve_signal_dates(monthly_ret_all, factor_data, cfg)
    monthly_ret = monthly_ret_all.reindex(signal_dates).dropna(how="all")

    macro_scores = None
    tech_scores = None
    if cfg.tactical.enabled and not monthly_ret.empty:
        macro_scores = compute_macro_scores(
            factor_data,
            monthly_ret.index,
            cfg.macro_rules,
            defaults=cfg.macro_defaults,
        )
        asset_class_map = dict(cfg.technical.asset_class)
        tech_scores = compute_technical_scores(
            daily_price,
            monthly_ret.index,
            list(monthly_ret.columns),
            asset_class_map=asset_class_map,
            llt_gamma=cfg.technical.llt_gamma,
            score_clip=cfg.technical.trend_score_clip,
        )

    return {
        "raw_daily_price": raw_daily_price,
        "daily_price": daily_price,
        "factor_data": factor_data,
        "market_data": market_data,
        "monthly_ret": monthly_ret,
        "macro_scores": macro_scores,
        "tech_scores": tech_scores,
    }


def _run_backtest_for_strategy(
    cfg: BacktestConfig,
    prepared: dict,
    *,
    strategy: str | None = None,
) -> dict:
    strategy_name = strategy or cfg.strategy
    raw_daily_price = prepared["raw_daily_price"]
    market_data_full = prepared["market_data"]
    monthly_ret = prepared["monthly_ret"]
    macro_scores = prepared["macro_scores"]
    tech_scores = prepared["tech_scores"]

    weight_schedule = build_weight_schedule(
        monthly_ret,
        cfg.asset_df,
        strategy=strategy_name,
        lookback=cfg.lookback,
        min_weight=cfg.weight_bounds.min,
        max_weight=cfg.weight_bounds.max,
        tactical_enabled=cfg.tactical.enabled,
        tactical_adjust_caps=cfg.tactical.tactical_adjust_caps,
        cash_asset=cfg.tactical.cash_asset,
        macro_scores=macro_scores,
        tech_scores=tech_scores,
        macro_rules=cfg.macro_rules,
        macro_defaults=cfg.macro_defaults,
        tech_score_clip_for_norm=cfg.tactical.tech_score_clip_for_norm,
    )
    # 正式净值区间不强行从 start_date 起画平线，而是从首个有效调仓真正落地的交易日开始。
    effective_start_date = _resolve_effective_start_date(weight_schedule, market_data_full)
    market_data = market_data_full.loc[
        pd.to_datetime(market_data_full["datetime"]) >= effective_start_date
    ].copy()

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
        "strategy": strategy_name,
        "raw_daily_price": raw_daily_price,
        "daily_price": prepared["daily_price"],
        "factor_data": prepared["factor_data"],
        "market_data": market_data,
        "effective_start_date": effective_start_date,
        "effective_end_date": pd.to_datetime(market_data["datetime"]).max(),
        "monthly_ret": monthly_ret,
        "macro_scores": macro_scores,
        "tech_scores": tech_scores,
        "weight_schedule": weight_schedule,
        "flow_data": flow_data,
        "local_benchmark_npv": _build_local_benchmark_npv(
            raw_daily_price,
            cfg.performance.benchmark_code,
            start_date=effective_start_date,
        ),
        "results": results_map,
    }


def run_backtest(cfg: BacktestConfig) -> dict:
    """执行完整流水线：读取本地 CSV -> 行情标准化 -> 收益率 -> 权重 -> 流水 -> 回测 -> 绩效。"""
    prepared = _prepare_backtest_inputs(cfg)
    return _run_backtest_for_strategy(cfg, prepared)


def run_backtests(cfg: BacktestConfig, strategies: list[str]) -> dict:
    """一次准备共享数据，顺序执行多种 strategy 并返回汇总结果。"""
    strategies_clean: list[str] = []
    seen: set[str] = set()
    for strategy in strategies:
        name = str(strategy).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        strategies_clean.append(name)

    if not strategies_clean:
        raise ValueError("strategies 不能为空。")

    # 多策略对比时只准备一次价格、因子、收益率和信号，避免重复 I/O 和重复打分。
    prepared = _prepare_backtest_inputs(cfg)
    runs: dict[str, dict] = {}
    for strategy in strategies_clean:
        strategy_cfg = replace(cfg, strategy=strategy)
        runs[strategy] = _run_backtest_for_strategy(
            strategy_cfg,
            prepared,
            strategy=strategy,
        )

    effective_start_date = min(
        pd.Timestamp(run_output["effective_start_date"])
        for run_output in runs.values()
    )
    effective_end_date = max(
        pd.Timestamp(run_output["effective_end_date"])
        for run_output in runs.values()
    )

    return {
        "cfg": cfg,
        "strategies": strategies_clean,
        "raw_daily_price": prepared["raw_daily_price"],
        "daily_price": prepared["daily_price"],
        "factor_data": prepared["factor_data"],
        "market_data": prepared["market_data"],
        "monthly_ret": prepared["monthly_ret"],
        "macro_scores": prepared["macro_scores"],
        "tech_scores": prepared["tech_scores"],
        "effective_start_date": effective_start_date,
        "effective_end_date": effective_end_date,
        "local_benchmark_npv": _build_local_benchmark_npv(
            prepared["raw_daily_price"],
            cfg.performance.benchmark_code,
            start_date=effective_start_date,
        ),
        "runs": runs,
    }


def analyze_results(
    run_output: dict,
    strategy_key: Optional[str] = None,
    *,
    draw_plots: bool | None = None,
    plot_periodic: bool | None = None,
    heatmap_periodic: bool | None = None,
    pretty_print_kpi: bool = True,
    show_titles: bool = True,
) -> dict:
    """对 run_backtest 的输出做绩效分析。"""
    results_map = run_output["results"]
    if strategy_key is None:
        strategy_key = next(iter(results_map))
    bt_result = results_map[strategy_key]["backtest_result"]
    cfg = run_output["cfg"]
    plot_enabled = cfg.performance.plot if draw_plots is None else bool(draw_plots)
    periodic_enabled = plot_enabled if plot_periodic is None else bool(plot_periodic)
    heatmap_enabled = False if heatmap_periodic is None else bool(heatmap_periodic)
    label = run_output.get("strategy") or strategy_key

    analysis = run_performance(
        bt_result,
        label=label,
        benchmark_code=cfg.performance.benchmark_code,
        benchmark_npv=run_output.get("local_benchmark_npv"),
        free_rate=cfg.performance.free_rate,
        plot_periodic=periodic_enabled,
        heatmap_periodic=heatmap_enabled,
        draw_npv=plot_enabled,
        draw_asset=plot_enabled,
        pretty_print_kpi=pretty_print_kpi,
        show_titles=show_titles,
    )
    return {"engine": "run_performance", "analysis": analysis, "backtest_result": bt_result}
