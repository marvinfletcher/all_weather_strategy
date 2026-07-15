"""全天候资产配置策略 - CLI 入口。

用法:
    python scripts/run_backtest.py --config config/default.yaml
    python scripts/run_backtest.py --config config/default.yaml --no-plot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 保证从项目根目录运行或从任意目录运行都能定位到 all_weather 包
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from all_weather.config import load_config
from all_weather.runner import analyze_results, run_backtest


def _render_echarts_if_enabled(run_output: dict, analysis: dict) -> None:
    cfg = run_output["cfg"]
    if not cfg.performance.plot:
        return

    try:
        from all_weather.backtest.echarts_viz import draw_combined_timeseries_echarts
    except Exception as exc:
        print(f"ECharts 模块加载失败，跳过联动图。原因：{type(exc).__name__}: {exc}")
        return

    try:
        draw_combined_timeseries_echarts(
            backtest_result=analysis["backtest_result"],
            label=cfg.strategy,
            benchmark_code=cfg.performance.benchmark_code,
            start_date=cfg.start_date,
            end_date=cfg.end_date_str,
            show=True,
        )
    except ModuleNotFoundError as exc:
        print(f"未安装 pyecharts，跳过 ECharts 联动图。原因：{exc}")
    except Exception as exc:
        print(f"ECharts 联动图生成失败，已跳过。原因：{type(exc).__name__}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="全天候资产配置策略回测")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "default.yaml"),
        help="YAML 配置文件路径",
    )
    parser.add_argument("--no-plot", action="store_true", help="禁用绘图")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.no_plot:
        cfg.performance.plot = False

    print(f"加载配置: {args.config}")
    print(f"策略: {cfg.strategy} | 资产数: {len(cfg.assets)} | 区间: {cfg.start_date} ~ {cfg.end_date_str}")

    run_output = run_backtest(cfg)
    analysis = analyze_results(run_output)
    _render_echarts_if_enabled(run_output, analysis)

    # 输出最新一期权重
    latest_weight = run_output["weight_schedule"].iloc[-1].sort_values(ascending=False)
    print("\n最新一期目标权重:")
    print(latest_weight.to_frame("weight").to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
