"""Standalone ECharts HTML dashboards for backtest results."""

from __future__ import annotations

from html import escape
import json
from pathlib import Path
import re
import tempfile
from typing import Any
import webbrowser

import numpy as np
import pandas as pd

from .performance_analysis import (
    _align_benchmark_npv,
    _extract_column_series,
    _extract_npv_series,
    _fetch_benchmark_npv,
    analyze_periodic_returns,
)


ECHARTS_CDN = "https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z_-]+", "-", str(text).strip()).strip("-").lower()
    return cleaned or "strategy"


def _safe_float(value: Any, digits: int = 4) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(number) or np.isinf(number):
        return None
    return round(number, digits)


def _to_date_axis(index: pd.Index) -> list[str]:
    return [pd.Timestamp(value).strftime("%Y-%m-%d") for value in index]


def _to_float_list(series: pd.Series, digits: int = 4) -> list[float | None]:
    values: list[float | None] = []
    for value in series.astype(float).tolist():
        values.append(_safe_float(value, digits=digits))
    return values


def _format_pct(value: Any) -> str:
    number = _safe_float(value, digits=2)
    return "--" if number is None else f"{number:.2f}%"


def _render_table(
    rows: list[dict[str, Any]],
    *,
    table_class: str = "report-table",
    scroll_class: str = "table-scroll",
) -> str:
    if not rows:
        return "<div class=\"table-empty\">暂无数据</div>"

    columns = list(rows[0].keys())
    head = "".join(f"<th>{escape(str(column))}</th>" for column in columns)
    body_rows: list[str] = []
    for row in rows:
        cells = "".join(
            f"<td>{escape(str(row.get(column, '--')))}</td>" for column in columns
        )
        body_rows.append(f"<tr>{cells}</tr>")
    body = "".join(body_rows)
    return (
        f"<div class=\"{escape(scroll_class)}\">"
        f"<table class=\"{escape(table_class)}\">"
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{body}</tbody>"
        "</table>"
        "</div>"
    )


def _resolve_benchmark_npv(
    benchmark_code: str | None,
    strategy_index: pd.Index,
    benchmark_npv: pd.Series | None,
    start_date: str | None,
    end_date: str | None,
) -> pd.Series | None:
    aligned = _align_benchmark_npv(benchmark_npv, strategy_index)
    if aligned is not None:
        return aligned
    return _fetch_benchmark_npv(benchmark_code, strategy_index, start_date, end_date)


def _slice_backtest_result(
    backtest_result: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    df = backtest_result.copy()
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    if "updatetime" in df.columns:
        time_index = pd.to_datetime(df["updatetime"], errors="coerce")
        mask = time_index.notna() & (time_index >= start_ts) & (time_index <= end_ts)
        return df.loc[mask].copy()

    if isinstance(df.index, pd.DatetimeIndex) or (
        len(df.index) > 0 and pd.api.types.is_datetime64_any_dtype(df.index)
    ):
        time_index = pd.to_datetime(df.index, errors="coerce")
        mask = (~pd.isna(time_index)) & (time_index >= start_ts) & (time_index <= end_ts)
        return df.loc[mask].copy()

    return df


def _build_annual_rows(backtest_result: pd.DataFrame) -> list[dict[str, str]]:
    try:
        year_df = analyze_periodic_returns(
            backtest_result,
            period="year",
            plot=False,
            heatmap=False,
        )
    except Exception:
        return []

    rows: list[dict[str, str]] = []
    for period, row in year_df.iterrows():
        start_date = pd.Timestamp(row["start_date"]).strftime("%Y-%m-%d")
        end_date = pd.Timestamp(row["end_date"]).strftime("%Y-%m-%d")
        rows.append(
            {
                "年份": str(period),
                "开始日期": start_date,
                "结束日期": end_date,
                "年度收益率": _format_pct(row.get("period_return_pct")),
            }
        )
    return rows


def _build_detail_payload(
    *,
    key: str,
    label: str,
    backtest_result: pd.DataFrame,
    benchmark_code: str | None = None,
    benchmark_npv: pd.Series | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    kpi: dict[str, Any] | None = None,
) -> dict[str, Any]:
    npv = _extract_npv_series(backtest_result)
    if len(npv) < 2:
        raise ValueError("净值序列长度不足，无法生成 ECharts 仪表页。")

    if start_date is None:
        start_date = npv.index.min().strftime("%Y-%m-%d")
    if end_date is None:
        end_date = npv.index.max().strftime("%Y-%m-%d")

    mask = (npv.index >= pd.Timestamp(start_date)) & (npv.index <= pd.Timestamp(end_date))
    npv = npv.loc[mask]
    if len(npv) < 2:
        raise ValueError("指定区间内净值序列长度不足，无法生成 ECharts 仪表页。")

    benchmark = _resolve_benchmark_npv(
        benchmark_code=benchmark_code,
        strategy_index=npv.index,
        benchmark_npv=benchmark_npv,
        start_date=start_date,
        end_date=end_date,
    )
    if benchmark is not None:
        benchmark = benchmark.reindex(npv.index).ffill()

    strategy_return_pct = (npv - 1.0) * 100.0
    drawdown_pct = (npv / npv.cummax() - 1.0) * 100.0

    cash = _extract_column_series(backtest_result, "cash")
    security = _extract_column_series(backtest_result, "security")
    if cash is not None:
        cash = cash.reindex(npv.index).ffill().fillna(0.0)
    if security is not None:
        security = security.reindex(npv.index).ffill().fillna(0.0)

    return_series = [
        {
            "name": f"{label}收益率",
            "values": _to_float_list(strategy_return_pct, digits=2),
            "color": "#f59e0b",
            "width": 2.4,
            "line_type": "solid",
        }
    ]
    if benchmark is not None:
        benchmark_return_pct = (benchmark - 1.0) * 100.0
        return_series.append(
            {
                "name": f"基准收益率({benchmark_code})",
                "values": _to_float_list(benchmark_return_pct, digits=2),
                "color": "#0ea5e9",
                "width": 1.8,
                "line_type": "solid",
            }
        )
        excess_return_pct = strategy_return_pct - benchmark_return_pct
        return_series.append(
            {
                "name": "超额收益率",
                "values": _to_float_list(excess_return_pct, digits=2),
                "color": "#facc15",
                "width": 1.8,
                "line_type": "dashed",
            }
        )

    asset_series: list[dict[str, Any]]
    if cash is not None and security is not None:
        total = cash + security
        asset_series = [
            {
                "name": "total asset",
                "values": _to_float_list(total, digits=2),
                "color": "#2563eb",
                "width": 1.8,
                "line_type": "solid",
                "area_opacity": 0.15,
            },
            {
                "name": "cash",
                "values": _to_float_list(cash, digits=2),
                "color": "#16a34a",
                "width": 1.4,
                "line_type": "solid",
                "area_opacity": 0.10,
            },
            {
                "name": "security",
                "values": _to_float_list(security, digits=2),
                "color": "#ea580c",
                "width": 1.4,
                "line_type": "solid",
                "area_opacity": 0.10,
            },
        ]
    else:
        asset_series = [
            {
                "name": "资产净值代理",
                "values": _to_float_list(npv, digits=4),
                "color": "#2563eb",
                "width": 1.8,
                "line_type": "solid",
                "area_opacity": 0.12,
            }
        ]

    start_text = pd.Timestamp(npv.index.min()).strftime("%Y-%m-%d")
    end_text = pd.Timestamp(npv.index.max()).strftime("%Y-%m-%d")
    annual_rows = _build_annual_rows(_slice_backtest_result(backtest_result, start_text, end_text))
    dom_id = _slugify(key)

    return {
        "key": key,
        "dom_id": dom_id,
        "label": label,
        "annual_rows": annual_rows,
        "chart": {
            "dates": _to_date_axis(npv.index),
            "title": f"{label} 收益率 / 回撤 / 资产价值联动图",
            "return_series": return_series,
            "drawdown_series": [
                {
                    "name": "回撤",
                    "values": _to_float_list(drawdown_pct, digits=2),
                    "color": "#dc2626",
                    "width": 1.6,
                    "line_type": "solid",
                    "area_opacity": 0.22,
                }
            ],
            "asset_series": asset_series,
        },
    }


def _build_overview_chart(
    strategy_reports: list[dict[str, Any]],
    benchmark_npv: pd.Series | None = None,
    benchmark_code: str | None = None,
) -> dict[str, Any]:
    nav_map: dict[str, pd.Series] = {}
    aligned_index = pd.DatetimeIndex([])
    for report in strategy_reports:
        label = str(report["label"])
        npv = _extract_npv_series(report["backtest_result"])
        if len(npv) < 2:
            continue
        npv = npv.dropna().astype(float).sort_index()
        nav_map[label] = npv
        aligned_index = aligned_index.union(npv.index)

    if aligned_index.empty or not nav_map:
        raise ValueError("多策略净值为空，无法生成总览图。")

    aligned_index = pd.DatetimeIndex(aligned_index).sort_values()
    palette = ["#2563eb", "#f59e0b", "#16a34a", "#dc2626", "#7c3aed", "#0891b2"]

    overview_series: list[dict[str, Any]] = []
    for idx, (label, npv) in enumerate(nav_map.items()):
        aligned = npv.reindex(aligned_index).ffill()
        overview_series.append(
            {
                "name": label,
                "values": _to_float_list(aligned, digits=4),
                "color": palette[idx % len(palette)],
                "width": 2.2,
                "line_type": "solid",
            }
        )

    if benchmark_npv is not None:
        benchmark = pd.Series(benchmark_npv.copy())
        benchmark.index = pd.to_datetime(benchmark.index, errors="coerce")
        benchmark = benchmark[benchmark.index.notna()].sort_index().dropna().astype(float)
        if not benchmark.empty:
            benchmark = benchmark.reindex(benchmark.index.union(aligned_index)).sort_index().ffill().reindex(aligned_index)
            overview_series.append(
                {
                    "name": f"基准({benchmark_code})" if benchmark_code else "基准",
                    "values": _to_float_list(benchmark, digits=4),
                    "color": "#64748b",
                    "width": 1.8,
                    "line_type": "dashed",
                }
            )

    return {
        "title": "多策略净值对比图",
        "dates": _to_date_axis(aligned_index),
        "series": overview_series,
    }


def _build_single_dashboard_html(detail: dict[str, Any]) -> str:
    annual_table_html = _render_table(
        detail["annual_rows"],
        table_class="report-table report-table-annual",
        scroll_class="table-scroll table-scroll-annual",
    )
    detail_section = (
        f"<section class=\"screen active\" id=\"detail-screen-{detail['dom_id']}\" "
        f"data-detail-dom-id=\"{detail['dom_id']}\">"
        "<div class=\"screen-head\">"
        f"<h2>{escape(detail['label'])}</h2>"
        "</div>"
        "<div class=\"detail-layout\">"
        f"<div class=\"card chart-card\"><div class=\"chart\" id=\"detail-chart-{detail['dom_id']}\"></div></div>"
        "<div class=\"card table-card\">"
        "<h3>年度收益表</h3>"
        f"{annual_table_html}"
        "</div>"
        "</div>"
        "</section>"
    )

    chart_payload = {
        "mode": "single",
        "details": [
            {
                "dom_id": detail["dom_id"],
                "label": detail["label"],
                "chart": detail["chart"],
            }
        ],
    }
    return _wrap_dashboard_html(
        title=f"{detail['label']} 绩效图表",
        body_markup=detail_section,
        chart_payload=chart_payload,
    )


def _build_multi_dashboard_html(
    *,
    overview_chart: dict[str, Any],
    details: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> str:
    button_markup = "".join(
        (
            f"<button class=\"nav-button\" type=\"button\" "
            f"onclick=\"showDetail('{detail['dom_id']}')\">"
            f"{escape(detail['label'])}"
            "</button>"
        )
        for detail in details
    )
    summary_table_html = _render_table(summary_rows)

    overview_section = (
        "<section class=\"screen active\" id=\"overview-screen\">"
        "<div class=\"screen-head\">"
        "<h2>多策略总览</h2>"
        "</div>"
        "<div class=\"overview-layout\">"
        "<div class=\"card chart-card\">"
        "<div class=\"chart chart-overview\" id=\"overview-chart\"></div>"
        "</div>"
        "<div class=\"card nav-card\">"
        "<h3>策略详情</h3>"
        "<p>点击按钮进入各策略的收益率 / 回撤 / 资产价值联动图。</p>"
        f"<div class=\"nav-button-group\">{button_markup}</div>"
        "</div>"
        "</div>"
        "<div class=\"card table-card\">"
        "<h3>多策略绩效汇总表</h3>"
        f"{summary_table_html}"
        "</div>"
        "</section>"
    )

    detail_sections: list[str] = []
    for detail in details:
        annual_table_html = _render_table(
            detail["annual_rows"],
            table_class="report-table report-table-annual",
            scroll_class="table-scroll table-scroll-annual",
        )
        detail_sections.append(
            f"<section class=\"screen\" id=\"detail-screen-{detail['dom_id']}\" "
            f"data-detail-dom-id=\"{detail['dom_id']}\">"
            "<div class=\"detail-topbar\">"
            "<button class=\"back-button\" type=\"button\" onclick=\"showOverview()\">返回总览</button>"
            "<div>"
            f"<h2>{escape(detail['label'])}</h2>"
            "</div>"
            "</div>"
            "<div class=\"detail-layout\">"
            f"<div class=\"card chart-card\"><div class=\"chart\" id=\"detail-chart-{detail['dom_id']}\"></div></div>"
            "<div class=\"card table-card\">"
            "<h3>年度收益表</h3>"
            f"{annual_table_html}"
            "</div>"
            "</div>"
            "</section>"
        )

    chart_payload = {
        "mode": "multi",
        "overview": overview_chart,
        "details": [
            {
                "dom_id": detail["dom_id"],
                "label": detail["label"],
                "chart": detail["chart"],
            }
            for detail in details
        ],
    }
    return _wrap_dashboard_html(
        title="多策略绩效图表",
        body_markup=overview_section + "".join(detail_sections),
        chart_payload=chart_payload,
    )


def _wrap_dashboard_html(
    *,
    title: str,
    body_markup: str,
    chart_payload: dict[str, Any],
) -> str:
    payload_json = json.dumps(chart_payload, ensure_ascii=False).replace("</", "<\\/")
    html = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>__PAGE_TITLE__</title>
  <script src="__ECHARTS_CDN__"></script>
  <style>
    :root {
      --bg: #f3f7fb;
      --card: #ffffff;
      --text: #0f172a;
      --muted: #475569;
      --line: #dbe5f0;
      --accent: #2563eb;
      --accent-soft: #dbeafe;
      --shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
      --radius: 22px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "PingFang SC", "Microsoft YaHei", "Helvetica Neue", Arial, sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(37, 99, 235, 0.10), transparent 28%),
        radial-gradient(circle at top right, rgba(245, 158, 11, 0.10), transparent 24%),
        linear-gradient(180deg, #f8fbff 0%, var(--bg) 100%);
    }
    .page {
      max-width: 1480px;
      margin: 0 auto;
      padding: 28px 24px 40px;
    }
    .hero {
      margin-bottom: 24px;
      padding: 28px 30px;
      border-radius: 28px;
      background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%);
      color: #ffffff;
      box-shadow: var(--shadow);
    }
    .hero h1 {
      margin: 0 0 10px;
      font-size: 30px;
      line-height: 1.2;
    }
    .hero p {
      margin: 0;
      color: rgba(255, 255, 255, 0.82);
      font-size: 15px;
      line-height: 1.6;
    }
    .screen {
      display: none;
      animation: fadeIn 180ms ease;
    }
    .screen.active { display: block; }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .screen-head,
    .detail-topbar {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 18px;
    }
    .screen-head h2,
    .detail-topbar h2 {
      margin: 0 0 6px;
      font-size: 24px;
    }
    .screen-head p,
    .detail-topbar p {
      margin: 0;
      color: var(--muted);
      line-height: 1.6;
    }
    .card {
      border-radius: var(--radius);
      background: var(--card);
      border: 1px solid rgba(219, 229, 240, 0.85);
      box-shadow: var(--shadow);
    }
    .chart-card,
    .table-card,
    .nav-card {
      padding: 18px 18px 16px;
    }
    .overview-layout {
      display: grid;
      grid-template-columns: minmax(0, 1.8fr) minmax(280px, 0.8fr);
      gap: 18px;
      margin-bottom: 18px;
    }
    .detail-layout {
      display: grid;
      grid-template-columns: minmax(0, 1.55fr) minmax(360px, 1.05fr);
      gap: 18px;
    }
    .chart {
      width: 100%;
      min-height: 760px;
    }
    .chart-overview {
      min-height: 620px;
    }
    .nav-card h3,
    .table-card h3 {
      margin: 2px 0 10px;
      font-size: 18px;
    }
    .nav-card p {
      margin: 0 0 14px;
      color: var(--muted);
      line-height: 1.6;
    }
    .nav-button-group {
      display: grid;
      gap: 12px;
    }
    .nav-button,
    .back-button {
      appearance: none;
      border: 0;
      cursor: pointer;
      border-radius: 14px;
      padding: 13px 16px;
      font-size: 15px;
      font-weight: 600;
      transition: transform 160ms ease, box-shadow 160ms ease, background 160ms ease;
    }
    .nav-button {
      color: #0f172a;
      background: linear-gradient(135deg, #e0ecff 0%, #fff7ed 100%);
      border: 1px solid rgba(37, 99, 235, 0.16);
      text-align: left;
    }
    .nav-button:hover,
    .back-button:hover {
      transform: translateY(-1px);
      box-shadow: 0 12px 22px rgba(37, 99, 235, 0.12);
    }
    .back-button {
      color: #ffffff;
      background: linear-gradient(135deg, #1d4ed8 0%, #0f172a 100%);
      padding-left: 18px;
      padding-right: 18px;
      white-space: nowrap;
    }
    .table-scroll {
      overflow-x: auto;
    }
    .table-scroll-annual {
      overflow-x: visible;
    }
    .report-table {
      width: 100%;
      border-collapse: collapse;
      min-width: 640px;
    }
    .report-table th,
    .report-table td {
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      white-space: nowrap;
      font-size: 14px;
    }
    .report-table th {
      position: sticky;
      top: 0;
      z-index: 1;
      background: #eef5ff;
      color: #0f172a;
      font-weight: 700;
    }
    .report-table-annual {
      min-width: 0;
      table-layout: fixed;
    }
    .report-table-annual th,
    .report-table-annual td {
      padding: 10px 8px;
      font-size: 13px;
      white-space: normal;
      word-break: break-word;
    }
    .report-table-annual th:nth-child(1),
    .report-table-annual td:nth-child(1) {
      width: 18%;
    }
    .report-table-annual th:nth-child(2),
    .report-table-annual td:nth-child(2),
    .report-table-annual th:nth-child(3),
    .report-table-annual td:nth-child(3) {
      width: 31%;
    }
    .report-table-annual th:nth-child(4),
    .report-table-annual td:nth-child(4) {
      width: 20%;
    }
    .report-table tbody tr:nth-child(even) td {
      background: #fbfdff;
    }
    .table-empty {
      padding: 18px 0;
      color: var(--muted);
    }
    @media (max-width: 1100px) {
      .overview-layout,
      .detail-layout {
        grid-template-columns: 1fr;
      }
      .chart,
      .chart-overview {
        min-height: 620px;
      }
    }
    @media (max-width: 720px) {
      .page {
        padding: 18px 14px 28px;
      }
      .hero {
        padding: 24px 20px;
      }
      .hero h1 {
        font-size: 24px;
      }
      .screen-head,
      .detail-topbar {
        flex-direction: column;
      }
      .report-table {
        min-width: 520px;
      }
      .report-table-annual {
        min-width: 0;
      }
      .chart,
      .chart-overview {
        min-height: 540px;
      }
    }
  </style>
</head>
<body>
  <div class="page">
    <div class="hero">
      <h1>__PAGE_TITLE__</h1>
    </div>
    __BODY_MARKUP__
  </div>
  <script>
    const DASHBOARD = __CHART_PAYLOAD__;
    const chartCache = {};

    function detailByDomId(domId) {
      return (DASHBOARD.details || []).find((item) => item.dom_id === domId);
    }

    function getChart(chartId) {
      if (!chartCache[chartId]) {
        const el = document.getElementById(chartId);
        if (!el) return null;
        chartCache[chartId] = echarts.init(el, null, { renderer: 'canvas' });
      }
      return chartCache[chartId];
    }

    function fmtNumber(value, digits = 2) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return '--';
      return Number(value).toLocaleString('zh-CN', {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      });
    }

    function tooltipFormatter(params) {
      if (!params || !params.length) return '';
      const axisLabel = params[0].axisValueLabel || params[0].axisValue || '';
      const lines = [axisLabel];
      params.forEach((item) => {
        const value = Array.isArray(item.value) ? item.value[1] : item.value;
        const isPercent = item.seriesName.includes('收益率') || item.seriesName === '回撤';
        lines.push(`${item.marker}${item.seriesName}: ${fmtNumber(value, isPercent ? 2 : 2)}${isPercent ? '%' : ''}`);
      });
      return lines.join('<br/>');
    }

    function buildOverviewOption(overview) {
      return {
        animation: false,
        backgroundColor: '#ffffff',
        title: {
          text: overview.title,
          left: 'center',
          top: 8,
          textStyle: { fontSize: 18, fontWeight: 700 },
        },
        legend: {
          top: 42,
          left: 'center',
        },
        tooltip: {
          trigger: 'axis',
          axisPointer: { type: 'cross' },
          formatter: tooltipFormatter,
        },
        grid: {
          left: 64,
          right: 24,
          top: 90,
          bottom: 78,
        },
        xAxis: {
          type: 'category',
          boundaryGap: false,
          data: overview.dates,
        },
        yAxis: {
          type: 'value',
          name: '净值',
          scale: true,
          axisLabel: {
            formatter: function (value) { return fmtNumber(value, 2); },
          },
          splitLine: {
            lineStyle: { color: 'rgba(148, 163, 184, 0.25)' },
          },
        },
        dataZoom: [
          { type: 'inside', start: 0, end: 100 },
          { type: 'slider', start: 0, end: 100, bottom: 18 },
        ],
        series: overview.series.map((series) => ({
          name: series.name,
          type: 'line',
          data: series.values,
          showSymbol: false,
          smooth: false,
          lineStyle: {
            width: series.width || 2,
            color: series.color,
            type: series.line_type || 'solid',
          },
          emphasis: { focus: 'series' },
        })),
      };
    }

    function buildDetailOption(detail) {
      const returnSeries = (detail.chart.return_series || []).map((series) => ({
        name: series.name,
        type: 'line',
        xAxisIndex: 0,
        yAxisIndex: 0,
        data: series.values,
        showSymbol: false,
        lineStyle: {
          width: series.width || 2,
          color: series.color,
          type: series.line_type || 'solid',
        },
      }));
      const drawdownSeries = (detail.chart.drawdown_series || []).map((series) => ({
        name: series.name,
        type: 'line',
        xAxisIndex: 1,
        yAxisIndex: 1,
        data: series.values,
        showSymbol: false,
        lineStyle: {
          width: series.width || 1.5,
          color: series.color,
          type: series.line_type || 'solid',
        },
        areaStyle: {
          opacity: series.area_opacity || 0.18,
          color: series.color,
        },
      }));
      const assetSeries = (detail.chart.asset_series || []).map((series) => ({
        name: series.name,
        type: 'line',
        xAxisIndex: 2,
        yAxisIndex: 2,
        data: series.values,
        showSymbol: false,
        lineStyle: {
          width: series.width || 1.5,
          color: series.color,
          type: series.line_type || 'solid',
        },
        areaStyle: series.area_opacity ? {
          opacity: series.area_opacity,
          color: series.color,
        } : undefined,
      }));

      return {
        animation: false,
        backgroundColor: '#ffffff',
        title: [
          {
            text: detail.chart.title,
            left: 'center',
            top: 8,
            textStyle: { fontSize: 18, fontWeight: 700 },
          },
          { text: '收益率', left: 64, top: 48, textStyle: { fontSize: 13, fontWeight: 600 } },
          { text: '回撤', left: 64, top: '38%', textStyle: { fontSize: 13, fontWeight: 600 } },
          { text: '资产价值', left: 64, top: '64%', textStyle: { fontSize: 13, fontWeight: 600 } },
        ],
        legend: [
          {
            data: returnSeries.map((series) => series.name),
            top: 40,
            left: 'center',
          },
          {
            data: drawdownSeries.map((series) => series.name),
            top: '34%',
            left: 'center',
          },
          {
            data: assetSeries.map((series) => series.name),
            top: '60%',
            left: 'center',
          },
        ],
        tooltip: {
          trigger: 'axis',
          axisPointer: { type: 'cross' },
          formatter: tooltipFormatter,
        },
        axisPointer: {
          link: [{ xAxisIndex: 'all' }],
          label: { backgroundColor: '#64748b' },
        },
        grid: [
          { left: 64, right: 28, top: 84, height: '18%' },
          { left: 64, right: 28, top: '42%', height: '14%' },
          { left: 64, right: 28, top: '68%', height: '16%' },
        ],
        xAxis: [
          {
            gridIndex: 0,
            type: 'category',
            boundaryGap: false,
            data: detail.chart.dates,
            axisLabel: { show: false },
          },
          {
            gridIndex: 1,
            type: 'category',
            boundaryGap: false,
            data: detail.chart.dates,
            axisLabel: { show: false },
          },
          {
            gridIndex: 2,
            type: 'category',
            boundaryGap: false,
            data: detail.chart.dates,
          },
        ],
        yAxis: [
          {
            gridIndex: 0,
            type: 'value',
            name: '收益率(%)',
            scale: true,
            axisLabel: { formatter: function (value) { return fmtNumber(value, 0) + '%'; } },
            splitLine: { lineStyle: { color: 'rgba(148, 163, 184, 0.22)' } },
          },
          {
            gridIndex: 1,
            type: 'value',
            name: '回撤(%)',
            scale: true,
            axisLabel: { formatter: function (value) { return fmtNumber(value, 0) + '%'; } },
            splitLine: { lineStyle: { color: 'rgba(148, 163, 184, 0.18)' } },
          },
          {
            gridIndex: 2,
            type: 'value',
            name: '资产价值',
            scale: true,
            axisLabel: { formatter: function (value) { return fmtNumber(value, 0); } },
            splitLine: { lineStyle: { color: 'rgba(148, 163, 184, 0.18)' } },
          },
        ],
        dataZoom: [
          { type: 'inside', xAxisIndex: [0, 1, 2], start: 0, end: 100 },
          { type: 'slider', xAxisIndex: [0, 1, 2], start: 0, end: 100, bottom: 18 },
        ],
        series: [...returnSeries, ...drawdownSeries, ...assetSeries],
      };
    }

    function renderOverviewChart() {
      if (!DASHBOARD.overview) return;
      const chart = getChart('overview-chart');
      if (!chart) return;
      chart.setOption(buildOverviewOption(DASHBOARD.overview), true);
      chart.resize();
    }

    function renderDetailChart(domId) {
      const detail = detailByDomId(domId);
      if (!detail) return;
      const chart = getChart(`detail-chart-${domId}`);
      if (!chart) return;
      chart.setOption(buildDetailOption(detail), true);
      chart.resize();
    }

    function activateScreen(screenId) {
      document.querySelectorAll('.screen').forEach((screen) => {
        screen.classList.remove('active');
      });
      const target = document.getElementById(screenId);
      if (!target) return;
      target.classList.add('active');
      const detailDomId = target.dataset.detailDomId;
      window.setTimeout(() => {
        if (screenId === 'overview-screen') {
          renderOverviewChart();
        } else if (detailDomId) {
          renderDetailChart(detailDomId);
        }
      }, 0);
    }

    function showDetail(domId) {
      activateScreen(`detail-screen-${domId}`);
    }

    function showOverview() {
      activateScreen('overview-screen');
    }

    window.showDetail = showDetail;
    window.showOverview = showOverview;

    window.addEventListener('resize', () => {
      Object.values(chartCache).forEach((chart) => {
        if (chart) chart.resize();
      });
    });

    document.addEventListener('DOMContentLoaded', () => {
      if (DASHBOARD.mode === 'multi') {
        renderOverviewChart();
      } else if ((DASHBOARD.details || []).length > 0) {
        renderDetailChart(DASHBOARD.details[0].dom_id);
      }
    });
  </script>
</body>
</html>
"""
    html = html.replace("__PAGE_TITLE__", escape(title))
    html = html.replace("__BODY_MARKUP__", body_markup)
    html = html.replace("__CHART_PAYLOAD__", payload_json)
    html = html.replace("__ECHARTS_CDN__", ECHARTS_CDN)
    return html


def _render_dashboard(
    html: str,
    *,
    output_path: str | Path | None,
    default_filename: str,
    show: bool,
) -> str:
    output = Path(output_path) if output_path is not None else Path(tempfile.gettempdir()) / default_filename
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")

    rendered_path = str(output.resolve())
    if show:
        opened = webbrowser.open(output.resolve().as_uri())
        if opened:
            print(f"绩效图表页已在浏览器中打开：{rendered_path}")
        else:
            print(f"绩效图表页已输出：{rendered_path}")
    return rendered_path


def draw_single_strategy_dashboard_echarts(
    *,
    backtest_result: pd.DataFrame,
    label: str,
    benchmark_code: str | None = None,
    benchmark_npv: pd.Series | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    kpi: dict[str, Any] | None = None,
    output_path: str | Path | None = None,
    show: bool = True,
) -> str:
    detail = _build_detail_payload(
        key=label,
        label=label,
        backtest_result=backtest_result,
        benchmark_code=benchmark_code,
        benchmark_npv=benchmark_npv,
        start_date=start_date,
        end_date=end_date,
        kpi=kpi,
    )
    html = _build_single_dashboard_html(detail)
    return _render_dashboard(
        html,
        output_path=output_path,
        default_filename=f"all_weather_{detail['dom_id']}_dashboard.html",
        show=show,
    )


def draw_multi_strategy_dashboard_echarts(
    *,
    strategy_reports: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    benchmark_npv: pd.Series | None = None,
    benchmark_code: str | None = None,
    output_path: str | Path | None = None,
    show: bool = True,
) -> str:
    if not strategy_reports:
        raise ValueError("strategy_reports 不能为空。")

    details = [
        _build_detail_payload(
            key=str(report["key"]),
            label=str(report["label"]),
            backtest_result=report["backtest_result"],
            benchmark_code=report.get("benchmark_code", benchmark_code),
            benchmark_npv=report.get("benchmark_npv"),
            start_date=report.get("start_date"),
            end_date=report.get("end_date"),
            kpi=report.get("kpi"),
        )
        for report in strategy_reports
    ]
    overview_chart = _build_overview_chart(
        strategy_reports=strategy_reports,
        benchmark_npv=benchmark_npv,
        benchmark_code=benchmark_code,
    )
    html = _build_multi_dashboard_html(
        overview_chart=overview_chart,
        details=details,
        summary_rows=summary_rows,
    )
    return _render_dashboard(
        html,
        output_path=output_path,
        default_filename="all_weather_multi_strategy_dashboard.html",
        show=show,
    )
