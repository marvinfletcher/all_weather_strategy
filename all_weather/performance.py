"""内置绩效分析（当 backtest.run_performance 不可用时的回退方案）。"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
except ModuleNotFoundError:  # pragma: no cover
    plt = None


def simple_performance_stats(backtest_result: pd.DataFrame, periods_per_year: int = 252) -> pd.Series:
    """基础 KPI：累计/年化收益、年化波动、夏普、最大回撤。"""
    df = backtest_result.copy()
    nav = df["npv"].dropna().astype(float)
    if len(nav) < 2:
        return pd.Series({
            "累计收益": np.nan,
            "年化收益": np.nan,
            "年化波动": np.nan,
            "夏普比率": np.nan,
            "最大回撤": np.nan,
        })
    ret = nav.pct_change().dropna()
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    ann_ret = (1 + total_ret) ** (periods_per_year / max(len(nav), 1)) - 1
    ann_vol = ret.std() * np.sqrt(periods_per_year)
    drawdown = nav / nav.cummax() - 1
    return pd.Series(
        {
            "累计收益": total_ret,
            "年化收益": ann_ret,
            "年化波动": ann_vol,
            "夏普比率": ann_ret / ann_vol if ann_vol > 0 else np.nan,
            "最大回撤": drawdown.min(),
        }
    )


def plot_nav_and_drawdown(backtest_result: pd.DataFrame, label: str = "策略") -> None:
    """用 matplotlib 画净值与回撤。"""
    if plt is None:
        print("matplotlib 不可用，跳过净值/回撤绘图。")
        return
    nav = backtest_result["npv"].dropna().astype(float)
    if nav.empty:
        print("净值序列为空，跳过绘图。")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(nav.index, nav.values, lw=2, label=label)
    axes[0].set_title(f"净值曲线 - {label}")
    axes[0].set_ylabel("净值")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    drawdown = nav / nav.cummax() - 1
    axes[1].plot(drawdown.index, drawdown.values, lw=1.5, color="red")
    axes[1].set_title("回撤")
    axes[1].set_ylabel("回撤")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
