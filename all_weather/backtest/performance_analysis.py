from .portfolio_backtest import run_portfolio  # noqa: F401  保留以兼容原调用
import pandas as pd
import math
from datetime import datetime
import numpy as np

# plotly 可选：缺失时绘图步骤自动跳过
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ModuleNotFoundError:  # pragma: no cover
    go = None
    make_subplots = None

try:
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
except ModuleNotFoundError:  # pragma: no cover
    plt = None

# 外部依赖全部延迟/回退：rqdatac 仅在需要基准对比时按需 import；
# performance.metrics 缺失时使用本地等价实现；ddb_data 在本项目不再使用。


def _calc_annualized_return(total_return: float, periods: int, periods_per_year: int = 252) -> float:
    """本地实现的年化收益率，与原 performance.metrics.calc_annualized_return 等价。

    total_return: 累计收益率（如 0.2 表示 +20%）
    periods: 区间内数据点数量
    periods_per_year: 一年内的周期数（默认 252 个交易日）
    """
    if periods <= 0:
        return 0.0
    return (1.0 + total_return) ** (periods_per_year / periods) - 1.0


try:
    from performance.metrics import calc_annualized_return  # type: ignore
except Exception:  # pragma: no cover - 外部包缺失
    calc_annualized_return = _calc_annualized_return


def _get_rq():
    """延迟导入 rqdatac，未安装时返回 None。调用方需自行处理。"""
    try:
        import rqdatac as rq
        return rq
    except ModuleNotFoundError:
        return None


def draw_npv_diagram(
    backtest_result,
    start_date=None,
    end_date=None,
    label="",
    benchmark_code=None,
    *,
    show: bool = True,
    return_metrics: bool = False,
    return_fig: bool = False,
):
    """
    绘制净值曲线图，展示策略净值、基准净值以及超额收益
    
    参数:
        backtest_result (pd.DataFrame): 回测结果数据，包含updatetime和npv列（updatetime可以是列或索引）
        start_date (str, optional): 回测开始日期，格式为'YYYY-MM-DD'，默认为None，None时从backtest_result的updatetime取最早日期
        end_date (str, optional): 回测结束日期，格式为'YYYY-MM-DD'，默认为None，None时从backtest_result的updatetime取最新日期
        label (str, optional): 策略标签名称，默认为空字符串
        benchmark_code (str, optional): 对比基准代码，6位数字（如'000300'），默认为None。None时不进行基准对比分析
    
    返回:
        - 默认返回 None（并直接显示 plotly 图表）
        - 若 return_metrics=True：返回净值/基准/超额等关键指标 dict
        - 若 return_fig=True：返回 plotly Figure
        - 若两者都为 True：返回 (metrics, fig)
    """
    # 兼容处理：updatetime可能是列或索引
    # run_portfolio 返回的 backtest_result 通常将 updatetime 设置为索引
    if 'updatetime' in backtest_result.columns:
        updatetime_series = pd.to_datetime(backtest_result['updatetime'])
    elif backtest_result.index.name == 'updatetime':
        # updatetime 作为索引名称，转换为 Series
        updatetime_series = pd.Series(pd.to_datetime(backtest_result.index), index=backtest_result.index)
    elif isinstance(backtest_result.index, pd.DatetimeIndex):
        # 索引是 DatetimeIndex（即使名称不是 updatetime），转换为 Series
        updatetime_series = pd.Series(pd.to_datetime(backtest_result.index), index=backtest_result.index)
    elif len(backtest_result) > 0 and pd.api.types.is_datetime64_any_dtype(backtest_result.index):
        # 索引是 datetime 类型（兼容处理），转换为 Series
        updatetime_series = pd.Series(pd.to_datetime(backtest_result.index), index=backtest_result.index)
    else:
        raise ValueError(f"backtest_result必须包含'updatetime'列或索引。当前列: {list(backtest_result.columns)}, 索引名: {backtest_result.index.name}, 索引类型: {type(backtest_result.index)}")
    
    # 确保 updatetime_series 是 Series 类型（不是 DatetimeIndex），以便使用 .iloc
    if isinstance(updatetime_series, pd.DatetimeIndex):
        updatetime_series = pd.Series(updatetime_series, index=updatetime_series)
    
    # 如果start_date或end_date为None，从backtest_result的updatetime取最早和最新日期
    if start_date is None:
        start_date = updatetime_series.min().strftime('%Y-%m-%d')
    if end_date is None:
        end_date = updatetime_series.max().strftime('%Y-%m-%d')
    
    if go is None:
        print("警告：未安装 plotly，跳过净值曲线绘制。")
        return None

    # 画出净值曲线图
    fig = go.Figure()

    # 仅在做基准对比时，添加参考线（净值=1）
    if benchmark_code is not None and str(benchmark_code).strip() != "":
        fig.add_trace(
            go.Scatter(
                x=updatetime_series,
                y=[1] * len(backtest_result),
                line=dict(color="gray"),
                showlegend=False,
            )
        )
    
    # 计算策略收益指标
    strategy_total_return = backtest_result.iloc[-1]['npv'] - 1
    # 使用数据点数量作为周期数，与关键绩效计算保持一致
    strategy_annual_return = calc_annualized_return(strategy_total_return, len(backtest_result), 252)
    
    # 如果有基准代码，获取基准数据并计算收益
    benchmark_data = None
    benchmark_total_return = None
    benchmark_annual_return = None
    excess_total_return = None
    excess_annual_return = None
    rq_code = None
    raw_code = None
    
    if benchmark_code is not None and benchmark_code != '':
        # 输入的基准代码可能来自其他平台（如 SH000300 / 000300.SH / 000300.XSHG 等）
        # 规则更新：benchmark_code 对比基准 **都是指数**，默认按 rq 指数后缀补全为 .XSHG
        # （若输入显式指明 SZ，则使用 .XSHE）
        raw_code = str(benchmark_code).strip()

        import re

        def _normalize_index_code_to_rq(code: str) -> str:
            s = code.strip().upper()

            # 已是 rq 标准
            m = re.match(r"^(?P<num>\d{6})\.(?P<suf>XSHG|XSHE)$", s)
            if m:
                return f"{m.group('num')}.{m.group('suf')}"

            # Wind/Tushare：SH000300 / SZ399001
            m = re.match(r"^(?P<ex>SH|SZ)(?P<num>\d{6})$", s)
            if m:
                suf = "XSHG" if m.group("ex") == "SH" else "XSHE"
                return f"{m.group('num')}.{suf}"

            # 000300.SH / 399001.SZ
            m = re.match(r"^(?P<num>\d{6})\.(?P<ex>SH|SZ)$", s)
            if m:
                suf = "XSHG" if m.group("ex") == "SH" else "XSHE"
                return f"{m.group('num')}.{suf}"

            # 纯6位：默认认为指数，补 .XSHG
            m = re.match(r"^(?P<num>\d{6})$", s)
            if m:
                return f"{m.group('num')}.XSHG"

            return code

        rq = _get_rq()
        if rq is None:
            print("警告：未安装 rqdatac，跳过基准对比。")
            benchmark_data = None
            rq_code = None
        else:
            rq_code = _normalize_index_code_to_rq(raw_code)

            # 兜底：如果仍无法规范化，再尝试 rq.id_convert
            if rq_code == raw_code:
                try:
                    converted = rq.id_convert(raw_code, to=None)
                    if isinstance(converted, list):
                        rq_code = converted[0] if len(converted) > 0 else raw_code
                    elif isinstance(converted, pd.Series):
                        rq_code = converted.iloc[0] if len(converted) > 0 else raw_code
                    elif converted:
                        rq_code = converted
                except Exception:
                    pass

            if rq_code == raw_code:
                print(f"警告：基准指数代码未能转换为 rq 标准，将直接使用: {raw_code}")

            try:
                # 使用rq接口获取基准的行情
                # 基于策略数据的日期范围获取基准数据，前后多取几天以确保覆盖
                strategy_start = updatetime_series.iloc[0]
                strategy_end = updatetime_series.iloc[-1]
                # 前后各多取5个交易日，确保覆盖
                benchmark_start = (pd.to_datetime(strategy_start) - pd.Timedelta(days=10)).strftime('%Y-%m-%d')
                benchmark_end = (pd.to_datetime(strategy_end) + pd.Timedelta(days=10)).strftime('%Y-%m-%d')

                benchmark_index = rq.get_price(
                    order_book_ids=rq_code,
                    start_date=benchmark_start,
                    end_date=benchmark_end,
                    frequency='1d',
                    fields=['open', 'close']
                )

                if len(benchmark_index) > 0:
                    benchmark_index = benchmark_index.reset_index()
                    benchmark_index.rename(columns={'date': 'time'}, inplace=True)
                    benchmark_index['time'] = pd.to_datetime(benchmark_index['time'])
                    benchmark_index.index = benchmark_index['time']

                    # 基于策略数据的日期对齐基准数据
                    # 为策略数据的每个日期找到对应的基准收盘价（使用前向填充）
                    strategy_dates = pd.to_datetime(updatetime_series)
                    benchmark_close_aligned = []
                    benchmark_dates_aligned = []

                    # 创建基准数据的收盘价序列（按日期排序）
                    benchmark_close_series = benchmark_index['close'].sort_index()

                    # 为策略数据的每个日期查找对应的基准收盘价
                    prev_benchmark_close = None
                    for strategy_date in strategy_dates:
                        # 查找基准数据中该日期或之前最近的日期
                        benchmark_dates_before = benchmark_close_series.index[benchmark_close_series.index <= strategy_date]
                        if len(benchmark_dates_before) > 0:
                            # 使用该日期或之前最近的基准收盘价
                            closest_date = benchmark_dates_before[-1]
                            prev_benchmark_close = benchmark_close_series.loc[closest_date]
                            benchmark_close_aligned.append(prev_benchmark_close)
                            benchmark_dates_aligned.append(strategy_date)
                        elif prev_benchmark_close is not None:
                            # 如果找不到，使用前一个有效值（前向填充）
                            benchmark_close_aligned.append(prev_benchmark_close)
                            benchmark_dates_aligned.append(strategy_date)

                    if len(benchmark_close_aligned) > 0:
                        # 为基准计算净值数据（使用收盘价的日收益率）
                        npv_bench = 1
                        npv_series = []
                        prev_close = None
                        for i, (bench_date, bench_close) in enumerate(zip(benchmark_dates_aligned, benchmark_close_aligned)):
                            if prev_close is not None and bench_close is not None:
                                # 使用前一日收盘价和当日收盘价计算日收益率
                                daily_return = (bench_close - prev_close) / prev_close
                                npv_bench = npv_bench * (1 + daily_return)
                            npv_series.append(npv_bench)
                            prev_close = bench_close

                        # 创建对齐后的基准数据DataFrame
                        benchmark_index_aligned = pd.DataFrame({
                            'time': benchmark_dates_aligned,
                            'close': benchmark_close_aligned,
                            'npv': npv_series
                        })
                        benchmark_index_aligned.index = benchmark_index_aligned['time']
                        benchmark_data = benchmark_index_aligned

                        # 计算基准收益指标
                        benchmark_total_return = benchmark_index_aligned.iloc[-1]['npv'] - 1
                        # 使用数据点数量作为周期数，与关键绩效计算保持一致
                        benchmark_annual_return = calc_annualized_return(benchmark_total_return, len(benchmark_index_aligned), 252)

                        # 计算超额收益
                        excess_total_return = strategy_total_return - benchmark_total_return
                        excess_annual_return = strategy_annual_return - benchmark_annual_return

                        # 添加基准净值曲线（使用对齐后的数据）
                        if go is not None:
                            fig.add_trace(go.Scatter(x=benchmark_index_aligned['time'], y=benchmark_index_aligned['npv'],
                                                     name=f'基准({raw_code})',
                                                     line=dict(color='deepskyblue'),
                                                     hovertemplate=
                                                     "time: %{x}<br>" +
                                                     "NPV: %{y}<br>"))

                        # 添加超额收益曲线（时间已对齐，直接计算）
                        excess_npv = []
                        excess_time = []
                        for i, strategy_date in enumerate(strategy_dates):
                            if i < len(benchmark_index_aligned):
                                strategy_npv = backtest_result.iloc[i]['npv']
                                benchmark_npv = benchmark_index_aligned.iloc[i]['npv']
                                excess_npv.append(1 + strategy_npv - benchmark_npv)
                                excess_time.append(strategy_date)

                        if len(excess_npv) > 0 and go is not None:
                            fig.add_trace(
                                go.Scatter(x=excess_time, y=excess_npv,
                                           name='超额收益',
                                           line=dict(color='yellow'),
                                           hovertemplate=
                                           "time: %{x}<br>" +
                                           "超额收益: %{hovertext}<br>",
                                           hovertext=[f'{((npv - 1) * 100):.3f}%' for npv in excess_npv])
                            )
                    else:
                        print(f'警告：无法对齐基准数据到策略日期范围')
                        benchmark_data = None
            except Exception as e:
                print(f'警告：获取基准数据失败 ({rq_code}): {e}')
                benchmark_data = None
    
    # 添加策略净值曲线
    fig.add_trace(go.Scatter(x=updatetime_series, y=backtest_result['npv'],
                             name=label if label else '策略',
                             line=dict(color='orange'),
                             hovertemplate=
                             "time: %{x}<br>" +
                             "NPV: %{y}<br>"))

    # 构建图例文本，显示收益指标
    annotation_text = f'策略总收益: {strategy_total_return*100:.2f}%<br>'
    annotation_text += f'策略年化收益: {strategy_annual_return*100:.2f}%'
    
    if benchmark_data is not None and benchmark_total_return is not None:
        annotation_text += f'<br><br>基准总收益: {benchmark_total_return*100:.2f}%<br>'
        annotation_text += f'基准年化收益: {benchmark_annual_return*100:.2f}%<br>'
        annotation_text += f'<br>超额总收益: {excess_total_return*100:.2f}%<br>'
        annotation_text += f'超额年化收益: {excess_annual_return*100:.2f}%'
    
    # 添加文本注释显示收益指标
    fig.add_annotation(
        text=annotation_text,
        xref="paper", yref="paper",
        # 放到绘图区右侧，避免遮挡净值曲线
        # 右侧上方通常是图例区域，这里下移避免遮挡 legend/label
        x=1.02, y=0.70,
        xanchor="left", yanchor="top",
        align="left",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=12),
        showarrow=False
    )

    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    fig.update_layout(
        autosize=False,
        width=1200,
        height=700,
        title='净值曲线',
        # 固定图例到右侧上方（与收益注释上下分离）
        legend=dict(x=1.02, y=0.98, xanchor="left", yanchor="top"),
        # 给右侧注释留出空间，避免被裁剪
        margin=dict(l=60, r=280, t=60, b=60),
    )
    if show:
        fig.show()

    metrics = {
        "strategy_total_return": float(strategy_total_return),
        "strategy_annual_return": float(strategy_annual_return),
        "benchmark_raw_code": raw_code,
        "benchmark_rq_code": rq_code,
        "benchmark_total_return": None if benchmark_total_return is None else float(benchmark_total_return),
        "benchmark_annual_return": None if benchmark_annual_return is None else float(benchmark_annual_return),
        "excess_total_return": None if excess_total_return is None else float(excess_total_return),
        "excess_annual_return": None if excess_annual_return is None else float(excess_annual_return),
    }

    if return_metrics and return_fig:
        return metrics, fig
    if return_metrics:
        return metrics
    if return_fig:
        return fig
    return None

def draw_asset_values(backtest_result, *, show: bool = True, return_fig: bool = False):
    """
    绘制资产价值分布图，展示现金、证券持仓和总资产的变化情况
    
    参数:
        backtest_result (pd.DataFrame): 回测结果数据，包含updatetime、cash和security列（updatetime可以是列或索引）
        show (bool): 是否显示图表，默认为True
        return_fig (bool): 是否返回plotly Figure对象，默认为False
    
    返回:
        - 默认返回 None（并直接显示 plotly 图表）
        - 若 return_fig=True：返回 plotly Figure
    """
    # 兼容处理：updatetime可能是列或索引
    # run_portfolio 返回的 backtest_result 通常将 updatetime 设置为索引
    if 'updatetime' in backtest_result.columns:
        time = pd.to_datetime(backtest_result['updatetime'])
    elif backtest_result.index.name == 'updatetime':
        # updatetime 作为索引名称
        time = pd.to_datetime(backtest_result.index)
    elif isinstance(backtest_result.index, pd.DatetimeIndex):
        # 索引是 DatetimeIndex（即使名称不是 updatetime）
        time = pd.to_datetime(backtest_result.index)
    elif len(backtest_result) > 0 and pd.api.types.is_datetime64_any_dtype(backtest_result.index):
        # 索引是 datetime 类型（兼容处理）
        time = pd.to_datetime(backtest_result.index)
    else:
        raise ValueError(f"backtest_result必须包含'updatetime'列或索引。当前列: {list(backtest_result.columns)}, 索引名: {backtest_result.index.name}, 索引类型: {type(backtest_result.index)}")
    cash_flow = backtest_result['cash']
    security_holding_value = backtest_result['security']
    cash_usage = backtest_result['security'] / (backtest_result['cash'] + backtest_result['security']) * 100

    print(f'平均资金利用率：{cash_usage.mean()}')

    if go is None:
        print("警告：未安装 plotly，跳过资产价值分布图。")
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=time, y=security_holding_value + cash_flow, fill='tonexty',
                   name='total asset', hovertext=list(map(lambda t: f'{t:.2f}%', cash_usage)), hovertemplate=
                   "time: %{x}<br>" +
                   "Asset Value: %{y}<br>" +
                   "资金利用率: %{hovertext}<br>"))  # fill down to xaxis
    fig.add_trace(go.Scatter(x=time, y=cash_flow, fill='tozeroy',
                             name='cash', hovertext=list(map(lambda t: f'{t:.2f}%', cash_usage)), hovertemplate=
                             "time: %{x}<br>" +
                             "Cash Value: %{y}<br>" +
                             "资金利用率: %{hovertext}<br>"))  # fill down to xaxis
    fig.add_trace(go.Scatter(x=time, y=security_holding_value, fill='tozeroy',
                             name='security', hovertext=list(map(lambda t: f'{t:.2f}%', cash_usage)), hovertemplate=
                             "time: %{x}<br>" +
                             "Security Value: %{y}<br>" +
                             "资金利用率: %{hovertext}<br>"))  # fill down to xaxis
    
    fig.update_layout(
        autosize=False,
        width=1200,
        height=600,
        title='资产价值分布',
        xaxis_title='时间',
        yaxis_title='资产价值',
        legend=dict(x=1.02, y=0.98, xanchor="left", yanchor="top"),
        margin=dict(l=60, r=200, t=60, b=60),
    )
    
    if show:
        fig.show()
    
    if return_fig:
        return fig
    return None


def _extract_npv_series(backtest_result: pd.DataFrame) -> pd.Series:
    """从 backtest_result 提取按时间排序的 npv 序列（兼容 updatetime 为列或索引）。"""
    df = backtest_result
    if "updatetime" in df.columns:
        idx = pd.to_datetime(df["updatetime"])
    elif df.index.name == "updatetime" or isinstance(df.index, pd.DatetimeIndex) or (
        len(df) > 0 and pd.api.types.is_datetime64_any_dtype(df.index)
    ):
        idx = pd.to_datetime(df.index)
    else:
        raise ValueError(
            f"backtest_result必须包含'updatetime'列或索引。当前列: {list(df.columns)}, "
            f"索引名: {df.index.name}, 索引类型: {type(df.index)}"
        )
    if "npv" not in df.columns:
        if "yields" in df.columns:
            npv = 1.0 + df["yields"].astype(float)
        else:
            raise ValueError("backtest_result 缺少必要列：npv（或 yields）")
    else:
        npv = df["npv"].astype(float)
    s = pd.Series(npv.values, index=idx).sort_index().dropna()
    return s[~s.index.duplicated(keep="last")]


def draw_drawdown_diagram(
    backtest_result,
    label: str = "",
    *,
    show: bool = True,
    return_fig: bool = False,
):
    """用 plotly 绘制回撤曲线（nav/nav.cummax()-1，填充到 0）。"""
    if go is None:
        print("警告：未安装 plotly，跳过回撤分析图。")
        return None
    npv = _extract_npv_series(backtest_result)
    if len(npv) < 2:
        print("净值序列长度不足，跳过回撤图。")
        return None
    drawdown = npv / npv.cummax() - 1.0

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values * 100.0,
            mode="lines",
            line=dict(color="#d62728", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(214,39,40,0.25)",
            name="回撤",
            hovertemplate="time: %{x}<br>回撤: %{y:.2f}%<br>",
        )
    )
    max_dd = float(drawdown.min() * 100.0)
    max_dd_date = drawdown.idxmin()
    fig.add_annotation(
        text=f"最大回撤: {max_dd:.2f}% ({max_dd_date.strftime('%Y-%m-%d')})",
        xref="paper", yref="paper", x=0.98, y=0.02, xanchor="right", yanchor="bottom",
        showarrow=False, bgcolor="rgba(255,255,255,0.8)", bordercolor="black", borderwidth=1,
        font=dict(size=12),
    )
    fig.update_layout(
        autosize=False, width=1200, height=420,
        title=f"回撤分析{(' - ' + label) if label else ''}",
        xaxis_title="时间", yaxis_title="回撤（%）",
        margin=dict(l=60, r=60, t=60, b=60),
    )
    if show:
        fig.show()
    if return_fig:
        return fig
    return None


def _fetch_benchmark_npv(benchmark_code, strategy_index, start_date, end_date):
    """紧凑版基准净值获取：返回对齐到 strategy_index 的基准净值序列，失败返回 None。"""
    if not benchmark_code:
        return None
    rq = _get_rq()
    if rq is None:
        return None
    import re
    raw = str(benchmark_code).strip()
    s = raw.upper()
    m = re.match(r"^(\d{6})\.(XSHG|XSHE)$", s)
    if m:
        code = s
    else:
        m = re.match(r"^(SH|SZ)(\d{6})$", s)
        if m:
            code = f"{m.group(2)}.{'XSHG' if m.group(1) == 'SH' else 'XSHE'}"
        elif re.match(r"^\d{6}$", s):
            code = s + ".XSHG"
        else:
            code = raw
    try:
        bp = rq.get_price(code, start_date=start_date, end_date=end_date, frequency="1d", fields="close")
        if bp is None or len(bp) == 0:
            return None
        close_raw = bp["close"] if "close" in bp.columns else bp.iloc[:, 0]
        if isinstance(close_raw, pd.DataFrame):
            close_raw = close_raw.iloc[:, 0]

        if isinstance(bp.index, pd.MultiIndex):
            date_index = None
            for level in range(bp.index.nlevels):
                parsed = pd.to_datetime(bp.index.get_level_values(level), errors="coerce")
                if parsed.notna().any():
                    date_index = parsed
                    break
            if date_index is None:
                raise ValueError("RQ 基准行情 MultiIndex 中未找到日期层")
            close = pd.Series(close_raw.values, index=date_index)
        else:
            date_index = pd.to_datetime(close_raw.index, errors="coerce")
            if not date_index.notna().any():
                raise ValueError("RQ 基准行情索引无法解析为日期")
            close = pd.Series(close_raw.values, index=date_index)

        close = close[close.index.notna()].sort_index()
        close = close.groupby(close.index).last().ffill()
        aligned = close.reindex(close.index.union(strategy_index)).sort_index().ffill().reindex(strategy_index)
        bnav = (1 + aligned.pct_change().fillna(0.0)).cumprod()
        bnav.iloc[0] = 1.0
        return bnav
    except Exception as exc:
        print(f"警告：基准 {code} 获取失败，联动图中跳过基准。原因：{type(exc).__name__}")
        return None


def _extract_column_series(backtest_result, col):
    """从 backtest_result 提取指定列的时序（兼容 updatetime 为列或索引）。"""
    df = backtest_result
    if "updatetime" in df.columns:
        idx = pd.to_datetime(df["updatetime"])
    elif df.index.name == "updatetime" or isinstance(df.index, pd.DatetimeIndex) or (
        len(df) > 0 and pd.api.types.is_datetime64_any_dtype(df.index)
    ):
        idx = pd.to_datetime(df.index)
    else:
        return None
    if col not in df.columns:
        return None
    s = pd.Series(df[col].astype(float).values, index=idx).sort_index()
    return s[~s.index.duplicated(keep="last")]


def draw_combined_timeseries(
    backtest_result,
    label: str = "",
    benchmark_code: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    *,
    show: bool = True,
    return_fig: bool = False,
):
    """净值 / 回撤 / 资产价值 三图纵向排列，共享时间轴，鼠标联动定位同一时间。

    依赖 plotly 的 shared_xaxes + hovermode='x' + spike lines 实现联动十字线。
    现有 draw_npv_diagram / draw_drawdown_diagram / draw_asset_values 三个独立函数保留不变。
    """
    if go is None or make_subplots is None:
        print("警告：未安装 plotly，跳过联动时序图。")
        return None
    npv = _extract_npv_series(backtest_result)
    if len(npv) < 2:
        print("净值序列长度不足，跳过联动时序图。")
        return None

    if start_date is None:
        start_date = npv.index.min().strftime("%Y-%m-%d")
    if end_date is None:
        end_date = npv.index.max().strftime("%Y-%m-%d")
    mask = (npv.index >= pd.Timestamp(start_date)) & (npv.index <= pd.Timestamp(end_date))
    npv = npv[mask]
    if len(npv) < 2:
        print("区间内净值不足，跳过联动时序图。")
        return None
    drawdown = (npv / npv.cummax() - 1.0) * 100.0

    cash = _extract_column_series(backtest_result, "cash")
    security = _extract_column_series(backtest_result, "security")
    if cash is not None:
        cash = cash.reindex(npv.index).ffill().fillna(0.0)
    if security is not None:
        security = security.reindex(npv.index).ffill().fillna(0.0)

    bench_npv = _fetch_benchmark_npv(benchmark_code, npv.index, start_date, end_date)

    title_main = "净值 / 回撤 / 资产价值 联动时序图"
    if label:
        title_main += f" - {label}"
    base = f" | 基准: {benchmark_code}" if (benchmark_code and bench_npv is not None) else ""
    title_main += base

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=("净值曲线", "回撤分析", "资产价值分布"),
    )

    # 子图1：净值（+基准）
    fig.add_trace(
        go.Scatter(
            x=npv.index, y=npv.values, mode="lines", line=dict(color="orange", width=2),
            name=label or "策略", hovertemplate="time: %{x}<br>NPV: %{y:.4f}<br>",
        ), row=1, col=1,
    )
    if bench_npv is not None:
        fig.add_trace(
            go.Scatter(
                x=bench_npv.index, y=bench_npv.values, mode="lines",
                line=dict(color="deepskyblue", width=1.5), name=f"基准({benchmark_code})",
                hovertemplate="time: %{x}<br>基准NPV: %{y:.4f}<br>",
            ), row=1, col=1,
        )

    # 子图2：回撤
    fig.add_trace(
        go.Scatter(
            x=drawdown.index, y=drawdown.values, mode="lines",
            line=dict(color="#d62728", width=1.5), fill="tozeroy", fillcolor="rgba(214,39,40,0.25)",
            name="回撤", hovertemplate="time: %{x}<br>回撤: %{y:.2f}%<br>",
        ), row=2, col=1,
    )

    # 子图3：资产价值分布
    if cash is not None and security is not None:
        total = cash + security
        cash_usage = (security / total.replace(0, np.nan) * 100).fillna(0.0)
        fig.add_trace(
            go.Scatter(
                x=total.index, y=total.values, mode="lines", fill="tonexty",
                line=dict(color="#1f77b4", width=1), name="total asset",
                hovertemplate="time: %{x}<br>Asset: %{y:,.0f}<br>资金利用率: %{customdata:.2f}%<br>",
                customdata=cash_usage.values,
            ), row=3, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=cash.index, y=cash.values, mode="lines", fill="tozeroy",
                line=dict(color="#2ca02c", width=1), name="cash",
                hovertemplate="time: %{x}<br>Cash: %{y:,.0f}<br>",
            ), row=3, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=security.index, y=security.values, mode="lines", fill="tozeroy",
                line=dict(color="#ff7f0e", width=1), name="security",
                hovertemplate="time: %{x}<br>Security: %{y:,.0f}<br>",
            ), row=3, col=1,
        )

    # shared_xaxes=True：hovermode='x' 会把同一 x 处的 hover 标签同步到三个子图。
    # matching 轴默认隐藏上层刻度且 spike 不渲染，故强制三图都显示刻度，
    # 并在每个子图都开启 spike（底部轴必定渲染；上层轴为 best-effort）。
    fig.update_layout(
        hovermode="x",
        dragmode="zoom",
        autosize=False,
        width=1200,
        height=1100,
        title=dict(text=title_main, x=0.5, xanchor="center", y=0.985),
        legend=dict(x=1.02, y=0.98, xanchor="left", yanchor="top"),
        margin=dict(l=60, r=200, t=110, b=80),
    )
    for r in (1, 2, 3):
        fig.update_xaxes(
            showspikes=True, spikethickness=2, spikecolor="black",
            spikesnap="hovered data", spikemode="across",
            showline=True, showticklabels=True, row=r, col=1,
        )
    fig.update_yaxes(title_text="净值", row=1, col=1)
    fig.update_yaxes(title_text="回撤（%）", row=2, col=1)
    fig.update_yaxes(title_text="资产价值", row=3, col=1)
    # 共享时间轴时仅保留底部 x 轴标题，避免子图之间文字重叠。
    fig.update_xaxes(title_text=None, row=1, col=1)
    fig.update_xaxes(title_text=None, row=2, col=1)
    fig.update_xaxes(title_text="时间", row=3, col=1)

    if show:
        fig.show()
    if return_fig:
        return fig
    return None


def plot_periodic_returns_plotly(
    backtest_result,
    periods: tuple[str, ...] = ("year", "quarter", "month"),
    *,
    show: bool = True,
    return_fig: bool = False,
):
    """用 plotly 在一张图的垂直子图中绘制 year/quarter/month 周期收益柱状图。

    返回 dict[period -> DataFrame]，与 analyze_periodic_returns 的返回一致。
    """
    if go is None or make_subplots is None:
        print("警告：未安装 plotly，跳过周期收益柱状图。")
        return {}
    periods = [str(p).strip().lower() for p in periods if p]
    periods = [p for p in periods if p in {"year", "quarter", "month"}]
    if not periods:
        return {}

    tables = {}
    for p in periods:
        tables[p] = analyze_periodic_returns(backtest_result, period=p, plot=False, heatmap=False)

    per_name = {"year": "年度", "quarter": "季度", "month": "月度"}
    titles = [f"{per_name[p]}收益（%）" for p in periods]
    fig = make_subplots(
        rows=len(periods), cols=1, shared_xaxes=False, vertical_spacing=0.10, subplot_titles=titles
    )
    for i, p in enumerate(periods, start=1):
        res = tables[p]
        y = res["period_return_pct"].values
        colors = np.where(y >= 0, "#d62728", "#2ca02c")  # 红涨绿跌
        fig.add_trace(
            go.Bar(
                x=res.index.astype(str),
                y=y,
                marker_color=colors,
                text=[f"{v:.2f}%" for v in y],
                textposition="outside",
                name=f"{per_name[p]}收益",
                hovertemplate="周期: %{x}<br>收益: %{y:.2f}%<br>",
            ),
            row=i, col=1,
        )
        fig.add_hline(y=0, line=dict(color="black", width=0.8), row=i, col=1)
        fig.update_yaxes(title_text="收益（%）", row=i, col=1)

    fig.update_layout(
        autosize=False, width=1200, height=280 * len(periods) + 80,
        title="周期收益统计", showlegend=False,
        margin=dict(l=60, r=60, t=60, b=60),
    )
    if show:
        fig.show()
    if return_fig:
        return fig
    return tables


def plot_periodic_heatmap_plotly(
    backtest_result,
    period: str = "month",
    *,
    show: bool = True,
    return_fig: bool = False,
):
    """用 plotly 单独绘制周期收益热力图（month 或 quarter）。"""
    if go is None:
        print("警告：未安装 plotly，跳过收益热力图。")
        return None
    period = str(period).strip().lower()
    if period not in {"month", "quarter"}:
        print(f"热力图仅支持 month/quarter，当前 period={period}，跳过。")
        return None
    res = analyze_periodic_returns(backtest_result, period=period, plot=False, heatmap=False)
    per_idx = res.index  # PeriodIndex 字符串
    # 重建 PeriodIndex 以取 year/month/quarter
    per_obj = pd.PeriodIndex(per_idx, freq=period[0].upper() if period == "month" else "Q")
    if period == "month":
        years = per_obj.year
        cols = per_obj.month
        col_labels = [str(i) for i in range(1, 13)]
        hm_title = "月度收益热力图（%）"
    else:
        years = per_obj.year
        cols = per_obj.quarter
        col_labels = [f"Q{i}" for i in range(1, 5)]
        hm_title = "季度收益热力图（%）"

    hm = pd.DataFrame(
        {"year": years, "col": cols, "val": res["period_return_pct"].values}
    ).pivot(index="year", columns="col", values="val")
    if period == "month":
        hm = hm.reindex(columns=list(range(1, 13)))
    else:
        hm = hm.reindex(columns=list(range(1, 5)))
    hm = hm.sort_index()

    z = hm.values
    # 标注文本
    text = [[(f"{v:.1f}" if pd.notna(v) else "") for v in row] for row in z]
    # 用 NaN 屏蔽缺失格（plotly heatmap 用 colorscale，缺失需特殊处理）
    z_masked = np.where(pd.isna(z), np.nan, z)

    fig = go.Figure(
        data=go.Heatmap(
            z=z_masked,
            x=col_labels,
            y=[str(y) for y in hm.index],
            colorscale="RdYlGn_r",
            colorbar=dict(title="收益（%）"),
            text=text,
            texttemplate="%{text}",
            hovertemplate="年: %{y}<br>列: %{x}<br>收益: %{z:.2f}%<br>",
            zmid=0,
        )
    )
    fig.update_layout(
        autosize=False, width=1200, height=max(320, 60 * len(hm.index) + 120),
        title=hm_title,
        xaxis_title=("月份" if period == "month" else "季度"),
        yaxis_title="年份",
        margin=dict(l=60, r=60, t=60, b=60),
    )
    if show:
        fig.show()
    if return_fig:
        return fig
    return None


def analyze_periodic_returns(
    backtest_result: pd.DataFrame,
    period: str = "month",
    plot: bool = True,
    heatmap: bool = True,
    figsize: tuple = (14, 4),
    title: str | None = None,
) -> pd.DataFrame:
    """
    周期收益分析：按年/季度/月统计收益，并进行可视化。

    参数:
        backtest_result (pd.DataFrame): 回测结果，需包含:
            - updatetime: datetime-like
            - npv: 净值序列（从 1 开始累计）
            若缺少 npv 但存在 yields（累计收益），会自动用 1+yields 作为 npv。
        period (str): 周期粒度，可选 "year" / "quarter" / "month"
        plot (bool): 是否绘制周期收益柱状图
        heatmap (bool): 是否绘制热力图（仅 month/quarter 有意义）
        figsize (tuple): 图尺寸
        title (str|None): 图标题，None 时自动生成

    返回:
        pd.DataFrame: 周期收益结果表（包含开始/结束日期、起止净值与周期收益）
    """
    if not isinstance(backtest_result, pd.DataFrame):
        raise TypeError("backtest_result 必须是 pd.DataFrame")
    
    # 兼容处理：updatetime可能是列或索引
    df = backtest_result.copy()
    if "updatetime" in df.columns:
        # 如果updatetime已经是列，直接使用
        df["updatetime"] = pd.to_datetime(df["updatetime"])
        df = df.sort_values("updatetime")
    elif df.index.name == 'updatetime' or isinstance(df.index, pd.DatetimeIndex) or (len(df) > 0 and pd.api.types.is_datetime64_any_dtype(df.index)):
        # 如果updatetime是索引，先重置索引（避免名称冲突），然后创建列
        df = df.reset_index(drop=False)  # reset_index会保留索引作为列
        # 如果索引名称是'updatetime'，reset_index后会自动创建'updatetime'列
        # 如果索引没有名称但是DatetimeIndex，reset_index后会创建'index'列，需要重命名
        if 'index' in df.columns and df.index.name != 'updatetime':
            df.rename(columns={'index': 'updatetime'}, inplace=True)
        elif df.index.name != 'updatetime' and 'updatetime' not in df.columns:
            # 如果reset_index后没有updatetime列，从索引创建
            df["updatetime"] = pd.to_datetime(df.index)
        df["updatetime"] = pd.to_datetime(df["updatetime"])
        df = df.sort_values("updatetime")
    else:
        raise ValueError(f"backtest_result必须包含'updatetime'列或索引。当前列: {list(df.columns)}, 索引名: {df.index.name}, 索引类型: {type(df.index)}")

    if "npv" not in df.columns:
        # 兼容旧字段：yields 为累计收益（例如 0.2 表示 +20%）
        if "yields" in df.columns:
            df["npv"] = 1.0 + df["yields"].astype(float)
        else:
            raise ValueError("backtest_result 缺少必要列：npv（或 yields）")

    npv = (
        df.set_index("updatetime")["npv"]
        .astype(float)
        .sort_index()
        .dropna()
    )
    # 去除重复索引，保留最后一个（通常代表该时点最新状态）
    npv = npv[~npv.index.duplicated(keep="last")]
    if len(npv) < 2:
        raise ValueError("净值序列长度不足，无法做周期收益分析（至少需要 2 个时点）")

    period = str(period).strip().lower()
    if period not in {"year", "quarter", "month"}:
        raise ValueError('period 仅支持 "year" / "quarter" / "month"')

    # 用 Period 分组，能拿到每个周期内真实的首末交易日
    if period == "year":
        per = npv.index.to_period("Y")
        per_name = "年度"
    elif period == "quarter":
        per = npv.index.to_period("Q")
        per_name = "季度"
    else:
        per = npv.index.to_period("M")
        per_name = "月度"

    grouped = npv.groupby(per)
    npv_end = grouped.last()
    period_start_date = grouped.apply(lambda s: s.index.min())
    period_end_date = grouped.apply(lambda s: s.index.max())

    period_ret = npv_end.pct_change()
    # 第一段周期收益：用第一天净值作为起点
    period_ret.iloc[0] = npv_end.iloc[0] / npv.iloc[0] - 1.0

    npv_start = npv_end.shift(1)
    npv_start.iloc[0] = npv.iloc[0]

    result = pd.DataFrame(
        {
            "start_date": pd.to_datetime(period_start_date.values),
            "end_date": pd.to_datetime(period_end_date.values),
            "npv_start": npv_start.values,
            "npv_end": npv_end.values,
            "period_return": period_ret.values,
        },
        index=npv_end.index.astype(str),
    )
    result.index.name = "period"
    result["period_return_pct"] = result["period_return"] * 100.0

    # 可视化
    if plot:
        if plt is None:
            print("警告：未安装 matplotlib，跳过周期收益柱状图。")
        else:
            plot_title = title if title is not None else f"{per_name}收益（%）"
            y = result["period_return_pct"].values
            colors = np.where(y >= 0, "#d62728", "#2ca02c")  # 红涨绿跌（符合国内习惯）
            plt.figure(figsize=figsize)
            bars = plt.bar(result.index.astype(str), y, color=colors, alpha=0.9)
            plt.axhline(0, color="black", linewidth=0.8)
            plt.title(plot_title)
            plt.ylabel("收益（%）")
            plt.xticks(rotation=45, ha="right")

            # 添加数字标签
            for i, (bar, value) in enumerate(zip(bars, y)):
                # 根据数值正负决定标签位置（正数在上方，负数在下方）
                height = bar.get_height()
                # 计算标签位置：正数在柱子上方，负数在柱子下方
                if height >= 0:
                    label_y = height + max(abs(height) * 0.05, 0.5)  # 至少距离0.5个单位
                    va = 'bottom'
                else:
                    label_y = height - max(abs(height) * 0.05, 0.5)  # 至少距离0.5个单位
                    va = 'top'

                plt.text(bar.get_x() + bar.get_width() / 2, label_y,
                        f'{value:.2f}%',
                        ha='center', va=va,
                        fontsize=13, fontweight='bold',
                        color='black')

            plt.tight_layout()
            plt.show()

    if heatmap and period in {"month", "quarter"}:
        per_idx = npv_end.index  # PeriodIndex
        if period == "month":
            years = per_idx.year
            cols = per_idx.month
            col_labels = [str(i) for i in range(1, 13)]
            hm_title = title if title is not None else "月度收益热力图（%）"
        else:
            years = per_idx.year
            cols = per_idx.quarter
            col_labels = [f"Q{i}" for i in range(1, 5)]
            hm_title = title if title is not None else "季度收益热力图（%）"

        hm = pd.DataFrame(
            {"year": years, "col": cols, "val": result["period_return_pct"].values}
        ).pivot(index="year", columns="col", values="val")
        # 保证列顺序
        if period == "month":
            hm = hm.reindex(columns=list(range(1, 13)))
        else:
            hm = hm.reindex(columns=list(range(1, 5)))

        plt.figure(figsize=(12, max(3, 0.5 * len(hm.index))))
        im = plt.imshow(hm.values, aspect="auto", cmap="RdYlGn_r", interpolation="nearest")
        plt.colorbar(im, label="收益（%）")
        plt.title(hm_title)
        plt.yticks(range(len(hm.index)), hm.index.astype(str))
        plt.xticks(range(len(hm.columns)), col_labels)

        # 标注数值（控制密度，避免太拥挤）
        for i in range(hm.shape[0]):
            for j in range(hm.shape[1]):
                v = hm.values[i, j]
                if pd.notna(v):
                    plt.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8)
        plt.tight_layout()
        plt.show()

    return result


def analyze_key_performance(
    backtest_result: pd.DataFrame,
    free_rate: float = 2.0,
    pretty_print: bool = True,
    title: str = "关键绩效指标",
) -> dict:
    """
    关键绩效分析：只输入 backtest_result，计算关键绩效指标并返回。

    参数:
        backtest_result (pd.DataFrame): `run_portfolio` 的回测结果表。通常至少包含：
            - npv: 净值
            - yields: 累计收益（npv-1）
            - drawdownrate_max: 最大回撤率（0~1）
            可选包含：
            - winning_rate: 胜率（0~1）
            - pnl_ratio: 盈亏比
        free_rate (float): 无风险利率，单位：%，用于 Sharpe 计算

    返回:
        dict: 关键绩效指标字段对象（数值型，单位见字段名）
    """
    if not isinstance(backtest_result, pd.DataFrame):
        raise TypeError("backtest_result 必须是 pd.DataFrame")
    if len(backtest_result) == 0:
        raise ValueError("backtest_result 为空，无法计算关键绩效指标")

    df = backtest_result.copy()
    # 兼容：如果缺少 npv 但有 yields，则补齐
    if "npv" not in df.columns and "yields" in df.columns:
        df["npv"] = 1.0 + df["yields"].astype(float)
    if "yields" not in df.columns and "npv" in df.columns:
        df["yields"] = df["npv"].astype(float) - 1.0

    vol = get_volatility(df)
    sp, y_all, y_yr = get_sharpe(free_rate, df)

    last = df.iloc[-1]
    dd_pct = float(last.get("drawdownrate_max", np.nan) * 100) if "drawdownrate_max" in df.columns else np.nan
    calmar = (float(y_yr) / dd_pct) if dd_pct and dd_pct != 0 else np.nan

    kpi = {
        "总收益率_pct": float(y_all * 100),
        "年化收益率_pct": float(y_yr),
        "最大回撤率_pct": float(dd_pct),
        "Sharpe": float(sp),
        "Calmar": float(calmar),
        "volatility": float(vol),
        "年化波动率_pct": float(vol * 100),  # 年化波动率转换为百分比
    }

    # 可选字段：若不存在则不强制要求
    if "winning_rate" in df.columns:
        kpi["胜率_pct"] = float(last.get("winning_rate", np.nan) * 100)
    if "pnl_ratio" in df.columns:
        kpi["盈亏比"] = float(last.get("pnl_ratio", np.nan))

    def _round2(v):
        try:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return v
            return round(float(v), 2)
        except Exception:
            return v

    kpi = {k: _round2(v) for k, v in kpi.items()}

    def _infer_date_range(_df: pd.DataFrame) -> tuple[str | None, str | None]:
        # 优先用 index（run_portfolio 的 backtest_result 通常用行情时间做 index）
        try:
            idx = pd.to_datetime(_df.index, errors="coerce")
            idx = idx[~pd.isna(idx)]
            if len(idx) > 0:
                return idx.min().strftime("%Y-%m-%d"), idx.max().strftime("%Y-%m-%d")
        except Exception:
            pass
        # 其次用 updatetime 列
        if "updatetime" in _df.columns:
            try:
                s = pd.to_datetime(_df["updatetime"], errors="coerce").dropna()
                if len(s) > 0:
                    return s.min().strftime("%Y-%m-%d"), s.max().strftime("%Y-%m-%d")
            except Exception:
                pass
        return None, None

    def _pretty_print_kpi(_kpi: dict, _df: pd.DataFrame) -> None:
        start, end = _infer_date_range(_df)
        header = title.strip() if title else "关键绩效指标"
        table_width = 56
        line = "=" * table_width
        sub = "-" * table_width

        # 处理中文等宽字符对齐：按“显示宽度”计算 padding
        import unicodedata

        def _disp_len(s: str) -> int:
            n = 0
            for ch in s:
                n += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
            return n

        def _ljust_disp(s: str, width: int, fill: str = " ") -> str:
            pad = max(0, width - _disp_len(s))
            return s + fill * pad

        def _rjust_disp(s: str, width: int, fill: str = " ") -> str:
            pad = max(0, width - _disp_len(s))
            return fill * pad + s

        # 布局：指标名在左，单位贴最右，数值在单位左侧右对齐（整体“右对齐”观感更好）
        label_w = 16
        unit_w = 2  # 例如 "%" 或空

        def fmt(label: str, value, unit: str = "") -> str:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                v = "--"
            else:
                v = f"{value:.2f}" if isinstance(value, (int, float, np.floating)) else str(value)
            left = _ljust_disp(label, label_w)
            unit_field = _rjust_disp(unit or "", unit_w)
            # value 区域占据剩余宽度（保证 value + 单位整体靠右）
            value_w = table_width - _disp_len(left) - 1 - _disp_len(unit_field)
            value_field = _rjust_disp(v, value_w)
            return (left + value_field + " " + unit_field)

        rows: list[str] = []
        # 按常用阅读顺序输出
        rows.append(fmt("总收益率", _kpi.get("总收益率_pct"), "%"))
        rows.append(fmt("年化收益率", _kpi.get("年化收益率_pct"), "%"))
        rows.append(fmt("最大回撤率", _kpi.get("最大回撤率_pct"), "%"))
        rows.append(fmt("年化波动率", _kpi.get("年化波动率_pct"), "%"))
        if "胜率_pct" in _kpi:
            rows.append(fmt("胜率", _kpi.get("胜率_pct"), "%"))
        rows.append(fmt("Sharpe", _kpi.get("Sharpe"), ""))
        rows.append(fmt("Calmar", _kpi.get("Calmar"), ""))
        if "盈亏比" in _kpi:
            rows.append(fmt("盈亏比", _kpi.get("盈亏比"), ""))

        print(line)
        # 居中也按显示宽度更美观：简单按 table_width 近似处理
        print(f"{header}".center(table_width))
        if start and end:
            print(sub)
            print(f"区间: {start} ~ {end}    无风险利率: {free_rate:.2f}%")
        print(sub)
        for r in rows:
            print(r)
        print(line)

    if pretty_print:
        _pretty_print_kpi(kpi, df)

    return kpi


def run_performance(
    backtest_result: pd.DataFrame,
    *,
    label: str = "",
    benchmark_code: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    free_rate: float = 2.0,
    periodic_periods: tuple[str, ...] = ("year", "quarter", "month"),
    plot_periodic: bool = True,
    heatmap_periodic: bool = True,
    draw_npv: bool = True,
    draw_asset: bool = True,
    pretty_print_kpi: bool = True,
    show_titles: bool = True,
) -> dict:
    """
    一键执行回测结果的常用分析流程（对 notebook 里的调用做统一封装）。

    默认执行内容：
    - 关键绩效指标：`analyze_key_performance`
    - 联动时序图：`draw_combined_timeseries`（净值/回撤/资产价值 三图纵向共享时间轴，鼠标联动）
    - 周期收益柱状图：`plot_periodic_returns_plotly`（year/quarter/month 垂直子图）
    - 收益热力图：`plot_periodic_heatmap_plotly`（plotly，月度/季度）

    参数:
        backtest_result (pd.DataFrame): `run_portfolio` 的回测结果表
        label (str): 策略名称（用于净值图例）
        benchmark_code (str|None): 基准指数代码（例如 "000300" / "000300.SH" / "SH000300" 等）
        start_date/end_date (str|None): 基准拉取区间，None 时从回测表推断
        free_rate (float): 无风险利率（%），用于 Sharpe/Calmar 等
        periodic_periods (tuple[str,...]): 周期粒度集合，可包含 "year"/"quarter"/"month"
        plot_periodic/heatmap_periodic (bool): 周期收益柱状图/热力图是否绘制
        draw_npv/draw_asset (bool): 任一为 True 即绘制联动时序图（两者合并为一张图）
        pretty_print_kpi (bool): KPI 是否在控制台美观打印
        show_titles (bool): 是否在每一步分析前打印章节标题（更像报告）

    返回:
        dict: 结构化结果（kpi、periodic_returns 等），便于后续写报告/落库
    """
    if not isinstance(backtest_result, pd.DataFrame):
        raise TypeError("backtest_result 必须是 pd.DataFrame")
    if len(backtest_result) == 0:
        raise ValueError("backtest_result 为空，无法执行业绩分析")

    def _print_title(title: str, *, sep: str = "=", width: int = 72) -> None:
        if not show_titles:
            return
        t = str(title).strip()
        line = sep * max(40, int(width))
        print("\n" + line)
        print(t)
        print(line)

    result: dict = {}

    # 1) 关键绩效（放在最前，作为报告摘要）
    _print_title("【1】关键绩效指标（KPI 汇总）")
    result["kpi"] = analyze_key_performance(
        backtest_result,
        free_rate=free_rate,
        pretty_print=pretty_print_kpi,
    )

    # 2) 联动时序图：净值 / 回撤 / 资产价值 三图纵向共享时间轴，鼠标联动定位同一时间
    if draw_npv or draw_asset:
        base = f" | 基准: {benchmark_code}" if benchmark_code else ""
        _print_title(f"【2】联动时序图（策略: {label or '未命名策略'}{base}）")
        draw_combined_timeseries(
            backtest_result=backtest_result,
            label=label,
            benchmark_code=benchmark_code,
            start_date=start_date,
            end_date=end_date,
        )

    # 5) 周期收益柱状图（year/quarter/month 垂直子图）+ 6) 收益热力图
    periodic_out: dict[str, pd.DataFrame] = {}
    if periodic_periods:
        if isinstance(periodic_periods, str):
            periodic_periods = (periodic_periods,)
        periods_clean = []
        for p in periodic_periods:
            if p is None:
                continue
            period = str(p).strip().lower()
            if period and period in {"year", "quarter", "month"}:
                periods_clean.append(period)
        if periods_clean:
            _print_title("【3】周期收益统计（年度 / 季度 / 月度）")
            # 先用无绘图模式拿到结果表，再用 plotly 一次性画垂直子图
            for period in periods_clean:
                periodic_out[period] = analyze_periodic_returns(
                    backtest_result, period=period, plot=False, heatmap=False
                )
            if plot_periodic:
                plot_periodic_returns_plotly(backtest_result, periods=tuple(periods_clean))

            _print_title("【4】收益热力图")
            if heatmap_periodic:
                # 月度热力图（最常用）；若配置含 quarter 也补一张季度热力图
                heat_periods = [p for p in periods_clean if p in {"month", "quarter"}]
                for p in heat_periods:
                    plot_periodic_heatmap_plotly(backtest_result, period=p)
    result["periodic_returns"] = periodic_out

    return result


def get_volatility(backtest):
    '''计算年化波动率，按照简单收益率计算，参数为回测报告backtest'''
    npv = backtest['npv'].values
    day_yield = np.zeros(len(npv))
    day_yield[0] = npv[0] - 1
    for inx in range(1, len(npv)):
        day_yield[inx] = npv[inx] / npv[inx - 1] - 1
    volatility = np.std(day_yield) * 252 ** 0.5
    return volatility


def get_sharpe(free_rate: float, backtest):
    '''计算夏普比：(年化收益-无风险利率)/年化波动率
    free_rate:无风险利率, 单位：%
    backtest:回测报告
    '''
    all_yield = backtest.iloc[-1]['yields']
    year_yield = calc_annualized_return(all_yield, len(backtest), 252) * 100  # 年化收益率：%
    volatility = get_volatility(backtest) * 100  # 单位统一：%
    sharpe = (year_yield - free_rate) / volatility
    return sharpe, all_yield, year_yield
