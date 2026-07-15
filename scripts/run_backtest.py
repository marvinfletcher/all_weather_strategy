"""全天候资产配置策略 - CLI 入口。

用法:
    python scripts/run_backtest.py --config config/default.yaml --strategy 0
    python scripts/run_backtest.py --config config/default.yaml --strategy 1
    python scripts/run_backtest.py --config config/default.yaml --strategy 2
    python scripts/run_backtest.py --config config/default.yaml --strategy all
    python scripts/run_backtest.py --config config/default.yaml --no-plot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# 保证从项目根目录运行或从任意目录运行都能定位到 all_weather 包
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from all_weather.config import load_config
from all_weather.runner import analyze_results, run_backtest, run_backtests


STRATEGY_CODE_MAP = {
    "0": "fixed",
    "1": "inverse_vol",
    "2": "risk_parity",
}
STRATEGY_LABEL_MAP = {
    "fixed": "fixed",
    "inverse_vol": "inverse_vol",
    "risk_parity": "risk_parity",
}


def _fmt_pct(value: object) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value):.2f}%"


def _fmt_num(value: object) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value):.2f}"


def _render_echarts_if_enabled(run_output: dict, analysis: dict) -> None:
    cfg = run_output["cfg"]
    if not cfg.performance.plot:
        return

    try:
        from all_weather.backtest.echarts_viz import draw_single_strategy_dashboard_echarts
    except Exception as exc:
        print(f"ECharts 模块加载失败，跳过绩效仪表页。原因：{type(exc).__name__}: {exc}")
        return

    try:
        draw_single_strategy_dashboard_echarts(
            backtest_result=analysis["backtest_result"],
            label=run_output.get("strategy") or cfg.strategy,
            benchmark_code=cfg.performance.benchmark_code,
            benchmark_npv=run_output.get("local_benchmark_npv"),
            start_date=cfg.start_date,
            end_date=cfg.end_date_str,
            kpi=analysis["analysis"].get("kpi"),
            show=True,
        )
    except Exception as exc:
        print(f"ECharts 绩效仪表页生成失败，已跳过。原因：{type(exc).__name__}: {exc}")


def _resolve_requested_strategies(strategy_arg: str) -> list[str]:
    value = str(strategy_arg).strip().lower()
    if value == "all":
        return list(STRATEGY_CODE_MAP.values())
    if value not in STRATEGY_CODE_MAP:
        raise ValueError("`--strategy` 只能取 0 / 1 / 2 / all。")
    return [STRATEGY_CODE_MAP[value]]


def _render_strategy_comparison_if_enabled(
    batch_output: dict,
    report_bundle: dict,
) -> None:
    cfg = batch_output["cfg"]
    if not cfg.performance.plot:
        return

    try:
        from all_weather.backtest.echarts_viz import draw_multi_strategy_dashboard_echarts
    except Exception as exc:
        print(f"多策略 ECharts 模块加载失败，跳过绩效仪表页。原因：{type(exc).__name__}: {exc}")
        return

    strategy_reports: list[dict] = []
    for strategy in batch_output["strategies"]:
        run_output = batch_output["runs"][strategy]
        analysis = report_bundle["analyses"][strategy]
        strategy_reports.append(
            {
                "key": strategy,
                "label": STRATEGY_LABEL_MAP.get(strategy, strategy),
                "backtest_result": analysis["backtest_result"],
                "benchmark_code": cfg.performance.benchmark_code,
                "benchmark_npv": run_output.get("local_benchmark_npv"),
                "start_date": pd.Timestamp(run_output["effective_start_date"]).strftime("%Y-%m-%d"),
                "end_date": pd.Timestamp(run_output["effective_end_date"]).strftime("%Y-%m-%d"),
                "kpi": analysis["analysis"].get("kpi"),
            }
        )

    try:
        draw_multi_strategy_dashboard_echarts(
            strategy_reports=strategy_reports,
            summary_rows=report_bundle["summary_rows"],
            benchmark_npv=batch_output.get("local_benchmark_npv"),
            benchmark_code=cfg.performance.benchmark_code,
            show=True,
        )
    except Exception as exc:
        print(f"多策略 ECharts 仪表页生成失败，已跳过。原因：{type(exc).__name__}: {exc}")


def _print_multi_strategy_summary(batch_output: dict) -> dict:
    rows: list[dict] = []
    summary_rows: list[dict] = []
    analyses: dict[str, dict] = {}
    for strategy in batch_output["strategies"]:
        run_output = batch_output["runs"][strategy]
        print(f"\n[{STRATEGY_LABEL_MAP.get(strategy, strategy)}]")
        effective_start = pd.Timestamp(run_output["effective_start_date"]).strftime("%Y-%m-%d")
        effective_end = pd.Timestamp(run_output["effective_end_date"]).strftime("%Y-%m-%d")
        print(f"正式回测区间: {effective_start} ~ {effective_end}")

        analysis = analyze_results(
            run_output,
            draw_plots=False,
            plot_periodic=False,
            heatmap_periodic=False,
            pretty_print_kpi=True,
            show_titles=False,
        )
        analyses[strategy] = analysis
        kpi = dict(analysis["analysis"]["kpi"])
        kpi["strategy"] = strategy
        rows.append(kpi)
        summary_rows.append(
            {
                "策略": STRATEGY_LABEL_MAP.get(strategy, strategy),
                "正式开始": effective_start,
                "正式结束": effective_end,
                "总收益率": _fmt_pct(kpi.get("总收益率_pct")),
                "年化收益率": _fmt_pct(kpi.get("年化收益率_pct")),
                "最大回撤": _fmt_pct(kpi.get("最大回撤率_pct")),
                "年化波动率": _fmt_pct(kpi.get("年化波动率_pct")),
                "Sharpe": _fmt_num(kpi.get("Sharpe")),
                "Calmar": _fmt_num(kpi.get("Calmar")),
            }
        )

        latest_weight = run_output["weight_schedule"].iloc[-1].sort_values(ascending=False)
        print("最新一期目标权重:")
        print(latest_weight.to_frame("weight").to_string())

    summary = pd.DataFrame()
    if rows:
        summary = pd.DataFrame(rows).set_index("strategy")
        print("\n多策略绩效汇总:")
        print(summary.to_string())
    return {"analyses": analyses, "summary": summary, "summary_rows": summary_rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="全天候资产配置策略回测")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "default.yaml"),
        help="YAML 配置文件路径",
    )
    parser.add_argument("--no-plot", action="store_true", help="禁用绘图")
    parser.add_argument(
        "--strategy",
        default="0",
        help="策略选择：0=fixed，1=inverse_vol，2=risk_parity，all=三种一起跑",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, validate_strategy=False)
    if args.no_plot:
        cfg.performance.plot = False
    strategies = _resolve_requested_strategies(args.strategy)

    print(f"加载配置: {args.config}")
    if len(strategies) == 1:
        print(f"策略: {strategies[0]} | 资产数: {len(cfg.assets)} | 区间: {cfg.start_date} ~ {cfg.end_date_str}")
    else:
        print(f"策略: {', '.join(strategies)} | 资产数: {len(cfg.assets)} | 区间: {cfg.start_date} ~ {cfg.end_date_str}")
    print(f"本地数据: {cfg.data.resolve_csv_path(cfg.project_root)}")

    if len(strategies) > 1:
        batch_output = run_backtests(cfg, strategies)
        overall_start = pd.Timestamp(batch_output["effective_start_date"]).strftime("%Y-%m-%d")
        overall_end = pd.Timestamp(batch_output["effective_end_date"]).strftime("%Y-%m-%d")
        print(f"综合正式回测区间: {overall_start} ~ {overall_end}")
        report_bundle = _print_multi_strategy_summary(batch_output)
        _render_strategy_comparison_if_enabled(batch_output, report_bundle)
        return 0

    cfg.strategy = strategies[0]
    run_output = run_backtest(cfg)
    effective_start = pd.Timestamp(run_output["effective_start_date"]).strftime("%Y-%m-%d")
    effective_end = pd.Timestamp(run_output["effective_end_date"]).strftime("%Y-%m-%d")
    print(f"正式回测区间: {effective_start} ~ {effective_end}")
    analysis = analyze_results(
        run_output,
        draw_plots=False,
        plot_periodic=False,
        heatmap_periodic=False,
        pretty_print_kpi=True,
        show_titles=False,
    )
    _render_echarts_if_enabled(run_output, analysis)

    # 输出最新一期权重
    latest_weight = run_output["weight_schedule"].iloc[-1].sort_values(ascending=False)
    print("\n最新一期目标权重:")
    print(latest_weight.to_frame("weight").to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
