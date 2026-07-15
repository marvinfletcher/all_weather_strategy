"""ECharts-based interactive visualizations for backtest results.

This module is intentionally separate from performance_analysis.py.  It is used
when Plotly's shared-axis hover is not enough and a visible linked axis pointer
across stacked time-series panels is required.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any
import webbrowser

import numpy as np
import pandas as pd

from .performance_analysis import (
    _extract_column_series,
    _extract_npv_series,
    _fetch_benchmark_npv,
)


def _require_pyecharts():
    try:
        from pyecharts import options as opts
        from pyecharts.charts import Grid, Line
        from pyecharts.globals import ThemeType
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "未安装 pyecharts。请先执行 `pip install pyecharts`，"
            "或在 requirements.txt 环境中安装依赖。"
        ) from exc
    return opts, Grid, Line, ThemeType


def _to_date_axis(index: pd.Index) -> list[str]:
    return [pd.Timestamp(v).strftime("%Y-%m-%d") for v in index]


def _to_float_list(series: pd.Series, digits: int = 6) -> list[float | None]:
    values: list[float | None] = []
    for value in series.astype(float).values:
        if pd.isna(value) or np.isinf(value):
            values.append(None)
        else:
            values.append(round(float(value), digits))
    return values


def _line_chart(
    title: str,
    x_axis: list[str],
    y_axis_name: str,
    *,
    opts: Any,
    Line: Any,
    height: int | None = None,
) -> Any:
    chart = Line()
    chart.add_xaxis(x_axis)
    chart.set_global_opts(
        title_opts=opts.TitleOpts(title=title, pos_left="center", title_textstyle_opts=opts.TextStyleOpts(font_size=14)),
        xaxis_opts=opts.AxisOpts(
            type_="category",
            boundary_gap=False,
            axislabel_opts=opts.LabelOpts(is_show=True),
            axisline_opts=opts.AxisLineOpts(is_on_zero=False),
        ),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            name=y_axis_name,
            # scale=True,
            splitline_opts=opts.SplitLineOpts(is_show=True, linestyle_opts=opts.LineStyleOpts(opacity=0.35)),
        ),
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
        axispointer_opts=opts.AxisPointerOpts(
            is_show=True,
            link=[{"xAxisIndex": "all"}],
            label=opts.LabelOpts(background_color="#777"),
        ),
        datazoom_opts=[
            opts.DataZoomOpts(type_="inside", xaxis_index=[0, 1, 2], range_start=0, range_end=100),
            opts.DataZoomOpts(type_="slider", xaxis_index=[0, 1, 2], pos_bottom="1%", range_start=0, range_end=100),
        ],
        legend_opts=opts.LegendOpts(is_show=False),
    )
    if height is not None:
        chart.options["height"] = height
    return chart


def build_combined_timeseries_echarts(
    backtest_result: pd.DataFrame,
    label: str = "",
    benchmark_code: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> Any:
    """Build a pyecharts Grid with linked axis pointer across three panels.

    Panels:
    1. Strategy net value and optional benchmark net value.
    2. Drawdown.
    3. Total asset, cash, and security value.
    """
    opts, Grid, Line, ThemeType = _require_pyecharts()

    npv = _extract_npv_series(backtest_result)
    if len(npv) < 2:
        raise ValueError("净值序列长度不足，无法绘制 ECharts 联动时序图。")

    if start_date is None:
        start_date = npv.index.min().strftime("%Y-%m-%d")
    if end_date is None:
        end_date = npv.index.max().strftime("%Y-%m-%d")
    mask = (npv.index >= pd.Timestamp(start_date)) & (npv.index <= pd.Timestamp(end_date))
    npv = npv[mask]
    if len(npv) < 2:
        raise ValueError("指定区间内净值序列长度不足，无法绘制 ECharts 联动时序图。")

    x_axis = _to_date_axis(npv.index)
    drawdown = (npv / npv.cummax() - 1.0) * 100.0
    cash = _extract_column_series(backtest_result, "cash")
    security = _extract_column_series(backtest_result, "security")
    if cash is not None:
        cash = cash.reindex(npv.index).ffill().fillna(0.0)
    if security is not None:
        security = security.reindex(npv.index).ffill().fillna(0.0)
    benchmark_npv = _fetch_benchmark_npv(benchmark_code, npv.index, start_date, end_date)

    strategy_name = label or "策略"
    strategy_return = npv - 1.0
    nav_legend = [f"{strategy_name}收益率"]
    nav = _line_chart("收益率曲线", x_axis, "收益率", opts=opts, Line=Line)
    nav.add_yaxis(
        f"{strategy_name}收益率",
        _to_float_list(strategy_return, digits=4),
        is_symbol_show=False,
        linestyle_opts=opts.LineStyleOpts(width=2, color="#f59e0b"),
        label_opts=opts.LabelOpts(is_show=False),
    )
    if benchmark_npv is not None:
        benchmark_aligned = benchmark_npv.reindex(npv.index).ffill()
        benchmark_return = benchmark_aligned - 1.0
        nav_legend.append(f"基准收益率({benchmark_code})")
        nav.add_yaxis(
            f"基准收益率({benchmark_code})",
            _to_float_list(benchmark_return, digits=4),
            is_symbol_show=False,
            linestyle_opts=opts.LineStyleOpts(width=1.5, color="#0ea5e9"),
            label_opts=opts.LabelOpts(is_show=False),
        )
        excess_return = (strategy_return - benchmark_return).dropna()
        if len(excess_return) > 0:
            nav_legend.append("超额收益率")
            nav.add_yaxis(
                "超额收益率",
                _to_float_list(excess_return.reindex(npv.index).ffill(), digits=4),
                is_symbol_show=False,
                linestyle_opts=opts.LineStyleOpts(width=1.8, color="#facc15"),
                label_opts=opts.LabelOpts(is_show=False),
            )

    dd = _line_chart("回撤分析", x_axis, "回撤(%)", opts=opts, Line=Line)
    dd.add_yaxis(
        "回撤",
        _to_float_list(drawdown, digits=4),
        is_symbol_show=False,
        linestyle_opts=opts.LineStyleOpts(width=1.5, color="#dc2626"),
        areastyle_opts=opts.AreaStyleOpts(opacity=0.25, color="#dc2626"),
        label_opts=opts.LabelOpts(is_show=False),
    )

    asset = _line_chart("资产价值分布", x_axis, "资产价值", opts=opts, Line=Line)
    if cash is not None and security is not None:
        total = cash + security
        asset_legend = ["total asset", "cash", "security"]
        asset.add_yaxis(
            "total asset",
            _to_float_list(total, digits=2),
            is_symbol_show=False,
            linestyle_opts=opts.LineStyleOpts(width=1.5, color="#2563eb"),
            areastyle_opts=opts.AreaStyleOpts(opacity=0.16, color="#2563eb"),
            label_opts=opts.LabelOpts(is_show=False),
        )
        asset.add_yaxis(
            "cash",
            _to_float_list(cash, digits=2),
            is_symbol_show=False,
            linestyle_opts=opts.LineStyleOpts(width=1.2, color="#16a34a"),
            areastyle_opts=opts.AreaStyleOpts(opacity=0.14, color="#16a34a"),
            label_opts=opts.LabelOpts(is_show=False),
        )
        asset.add_yaxis(
            "security",
            _to_float_list(security, digits=2),
            is_symbol_show=False,
            linestyle_opts=opts.LineStyleOpts(width=1.2, color="#ea580c"),
            areastyle_opts=opts.AreaStyleOpts(opacity=0.14, color="#ea580c"),
            label_opts=opts.LabelOpts(is_show=False),
        )
    else:
        asset_legend = ["资产价值"]
        asset.add_yaxis(
            "资产价值",
            _to_float_list(npv, digits=6),
            is_symbol_show=False,
            linestyle_opts=opts.LineStyleOpts(width=1.5, color="#2563eb"),
            areastyle_opts=opts.AreaStyleOpts(opacity=0.16, color="#2563eb"),
            label_opts=opts.LabelOpts(is_show=False),
        )

    title = "收益率 / 回撤 / 资产价值 ECharts 联动图"
    if label:
        title += f" - {label}"
    if benchmark_code and benchmark_npv is not None:
        title += f" | 基准: {benchmark_code}"

    grid = Grid(
        init_opts=opts.InitOpts(
            width="1200px",
            height="900px",
            theme=ThemeType.LIGHT,
            bg_color="#f8fafc",
            page_title=title,
            animation_opts=opts.AnimationOpts(animation=False),
        )
    )
    grid.add(
        nav,
        grid_opts=opts.GridOpts(
            pos_left="8%", pos_right="6%", pos_top="12%", height="20%", background_color="#ffffff"
        ),
    )
    grid.add(
        dd,
        grid_opts=opts.GridOpts(
            pos_left="8%", pos_right="6%", pos_top="42%", height="17%", background_color="#ffffff"
        ),
    )
    grid.add(
        asset,
        grid_opts=opts.GridOpts(
            pos_left="8%", pos_right="6%", pos_top="70%", height="17%", background_color="#ffffff"
        ),
    )

    # Grid 合并多个 Line 后，以最终 options 为准；这里显式写入全局联动配置。
    grid.options["backgroundColor"] = "#f8fafc"
    grid.options["title"] = [
        {"text": title, "left": "center", "top": "1%", "textStyle": {"fontSize": 16}},
        {"text": "收益率曲线", "left": "8%", "top": "6%", "textStyle": {"fontSize": 13}},
        {"text": "回撤分析", "left": "8%", "top": "36%", "textStyle": {"fontSize": 13}},
        {"text": "资产价值分布", "left": "8%", "top": "64%", "textStyle": {"fontSize": 13}},
    ]
    grid.options["legend"] = [
        {
            "data": nav_legend,
            "top": "8%",
            "left": "center",
            "orient": "horizontal",
            "itemGap": 18,
        },
        {
            "data": ["回撤"],
            "top": "38%",
            "left": "center",
            "orient": "horizontal",
            "itemGap": 18,
        },
        {
            "data": asset_legend,
            "top": "66%",
            "left": "center",
            "orient": "horizontal",
            "itemGap": 18,
        },
    ]
    grid.options["tooltip"] = {"trigger": "axis", "axisPointer": {"type": "cross"}}
    grid.options["axisPointer"] = {"link": [{"xAxisIndex": "all"}], "label": {"backgroundColor": "#777"}}
    grid.options["dataZoom"] = [
        {"type": "inside", "xAxisIndex": [0, 1, 2], "start": 0, "end": 100},
        {"type": "slider", "xAxisIndex": [0, 1, 2], "bottom": "1%", "start": 0, "end": 100},
    ]
    return grid


def draw_combined_timeseries_echarts(
    backtest_result: pd.DataFrame,
    label: str = "",
    benchmark_code: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    *,
    output_path: str | Path | None = None,
    notebook: bool = False,
    show: bool = True,
    return_chart: bool = False,
) -> Any:
    """Render the linked ECharts time-series chart.

    Parameters
    ----------
    output_path:
        HTML output path. If omitted and ``show=True`` outside notebook, the
        chart is rendered to a temporary HTML file and opened in the browser.
    notebook:
        Display inline in Jupyter via ``render_notebook``.
    show:
        Whether to display/render immediately.
    return_chart:
        Return the underlying pyecharts Grid object for further customization.
    """
    chart = build_combined_timeseries_echarts(
        backtest_result=backtest_result,
        label=label,
        benchmark_code=benchmark_code,
        start_date=start_date,
        end_date=end_date,
    )

    rendered_path: str | None = None
    if show and notebook:
        try:
            from IPython.display import display

            display(chart.render_notebook())
        except Exception as exc:  # pragma: no cover
            print(f"ECharts notebook 内嵌显示失败，改为 HTML 输出。原因：{type(exc).__name__}: {exc}")
            notebook = False

    if show and not notebook:
        if output_path is None:
            output_path = (
                Path(tempfile.gettempdir())
                / "all_weather_combined_timeseries_echarts.html"
            )
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        rendered_path = chart.render(str(output))
        opened = webbrowser.open(output.resolve().as_uri())
        if opened:
            print(f"ECharts 联动图已在浏览器中打开：{rendered_path}")
        else:
            print(f"ECharts 联动图已输出：{rendered_path}")
    elif output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        rendered_path = chart.render(str(output))

    if return_chart:
        return chart
    return rendered_path
