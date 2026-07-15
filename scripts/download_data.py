"""下载全天候策略所需的 RQ 日频价格并缓存到本地 CSV。

用法:
    python scripts/download_data.py --config config/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from all_weather.config import load_config

# 宏观规则里存在同比、滞后一期、趋势窗口等需求，
# 因此下载因子时会额外向前多取一段历史，避免正式回测起点附近信号缺失。
FACTOR_WARMUP_MONTHS = 24


def _load_rq_license() -> str:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        raise RuntimeError("未找到项目根目录 .env 文件，请先配置 RQ_LICENSE。")

    for line in env_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text.startswith("RQ_LICENSE="):
            return text.split("=", 1)[1].strip().strip("'").strip('"')
    raise RuntimeError("`.env` 中未找到 RQ_LICENSE，请按 RQ_LICENSE='你的license' 格式填写。")


def _init_rq():
    try:
        import rqdatac
    except ImportError as exc:  # pragma: no cover - 环境依赖
        raise ImportError("当前环境未安装 rqdatac。请先安装并确认 RQ 数据权限。") from exc

    license_value = _load_rq_license()
    try:
        rqdatac.init(
            "license",
            license_value,
            ("rqdatad-pro.ricequant.com", 16011),
        )
    except TypeError:
        pass
    except Exception as exc:
        raise RuntimeError(
            "米筐连接失败。请检查项目根目录 .env 中的 RQ_LICENSE 是否正确。"
        ) from exc
    return rqdatac


def _download_price_panel(
    rqdatac,
    order_book_ids: list[str],
    start_date: str,
    end_date: str,
    field: str,
) -> pd.DataFrame:
    raw = rqdatac.get_price(
        order_book_ids,
        start_date=start_date,
        end_date=end_date,
        frequency="1d",
        fields=field,
        adjust_type="pre",
        expect_df=True,
    )
    if raw is None or len(raw) == 0:
        raise ValueError("RQ 未返回价格数据。请检查代码、日期、权限和字段名。")

    if isinstance(raw.index, pd.MultiIndex):
        names = list(raw.index.names)
        id_level = "order_book_id" if "order_book_id" in names else names[1]
        price = (
            raw[field].unstack(id_level)
            if field in raw.columns
            else raw.iloc[:, 0].unstack(id_level)
        )
        price.index = pd.to_datetime(price.index)
        price = price.sort_index()
    elif isinstance(raw, pd.DataFrame):
        price = raw.copy()
        price.index = pd.to_datetime(price.index)
        price = price.sort_index()
    else:
        raise TypeError(f"未识别的 RQ 返回格式：{type(raw)}")

    returned_ids = set(price.columns.astype(str))
    missing = [oid for oid in order_book_ids if str(oid) not in returned_ids]
    if missing:
        raise ValueError(
            "RQ 未返回以下 order_book_id 的价格数据（代码无效或无数据权限）：\n  - "
            + "\n  - ".join(missing)
            + "\n请检查代码格式及 RQ 权限后重试。"
        )
    return price


def _make_factor_download_start_date(start_date: str) -> str:
    warmup_start = pd.Timestamp(start_date) - pd.DateOffset(months=FACTOR_WARMUP_MONTHS)
    return warmup_start.strftime("%Y-%m-%d")


def _clip_factor_panel_to_range(
    factor: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    clipped = factor.loc[(factor.index >= start_ts) & (factor.index <= end_ts)].copy()
    if clipped.empty:
        raise ValueError(
            f"因子表在区间 {start_date} ~ {end_date} 内为空。"
            "请检查日期设置、数据权限或是否需要更长历史回看。"
        )
    return clipped


def _download_china_factor_panel(
    rqdatac,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    raw = rqdatac.econ.get_factors(
        factors=[
            "制造业采购经理指数PMI_当月",
            "居民消费价格指数CPI_当月同比(上年同月=100)",
            "社会融资规模存量_同比增速_月末数",
            "中债国债到期收益率曲线:10年:日",
        ],
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
    )
    if raw is None or len(raw) == 0:
        raise ValueError("RQ 未返回中国因子数据。请检查宏观因子权限。")

    df = raw.reset_index()
    for col in ["info_date", "start_date", "end_date", "rice_create_tm"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    factor_map = {
        "中债国债到期收益率曲线:10年:日": "cn10y_yield",
        "居民消费价格指数CPI_当月同比(上年同月=100)": "cn_cpi_yoy_index",
        "社会融资规模存量_同比增速_月末数": "cn_social_financing_yoy_pct",
        "制造业采购经理指数PMI_当月": "cn_pmi",
    }
    df["factor_name"] = df["factor"].map(factor_map)
    df = df[df["factor_name"].notna()].copy()

    daily = df[df["factor_name"].isin(["cn10y_yield"])].copy()
    daily["month"] = daily["end_date"].dt.to_period("M")
    daily_monthly = (
        daily.sort_values(["factor_name", "end_date"])
        .groupby(["month", "factor_name"], as_index=False)
        .tail(1)
    )[["month", "factor_name", "value"]]

    monthly = df[
        df["factor_name"].isin(
            [
                "cn_cpi_yoy_index",
                "cn_social_financing_yoy_pct",
                "cn_pmi",
            ]
        )
    ].copy()
    monthly["month"] = monthly["end_date"].dt.to_period("M")
    monthly_monthly = (
        monthly.sort_values(["factor_name", "end_date", "info_date"])
        .groupby(["month", "factor_name"], as_index=False)
        .tail(1)
    )[["month", "factor_name", "value"]]

    wide = (
        pd.concat([daily_monthly, monthly_monthly], ignore_index=True)
        .pivot(index="month", columns="factor_name", values="value")
        .sort_index()
    )
    wide.index = wide.index.to_timestamp("M")
    wide.index.name = "date"
    wide = wide.reindex(
        columns=[
            "cn10y_yield",
            "cn_cpi_yoy_index",
            "cn_social_financing_yoy_pct",
            "cn_pmi",
        ]
    )

    china_factor = wide.copy()
    # 月频宏观指标按“已知上一期”参与调仓，避免当期数据在月内被提前使用。
    for col in ["cn_cpi_yoy_index", "cn_social_financing_yoy_pct", "cn_pmi"]:
        china_factor[f"{col}_lag1"] = china_factor[col].shift(1)

    return china_factor[
        [
            "cn10y_yield",
            "cn_cpi_yoy_index_lag1",
            "cn_social_financing_yoy_pct_lag1",
            "cn_pmi_lag1",
        ]
    ].dropna()


def _download_us_factor_panel(start_date: str, end_date: str) -> pd.DataFrame:
    try:
        from pandas_datareader import data as web
    except ImportError as exc:  # pragma: no cover - 环境依赖
        raise ImportError(
            "当前环境未安装 pandas_datareader，无法下载美国宏观因子。"
        ) from exc

    series = {
        "DTWEXBGS": "usd_index",
        "DGS10": "us10y",
        "M2SL": "us_m2",
        "CPIAUCSL": "us_cpi",
        "M1SL": "us_m1",
        "DCOILWTICO": "wti",
    }
    df = web.DataReader(
        list(series.keys()),
        "fred",
        pd.Timestamp(start_date).to_pydatetime(),
        pd.Timestamp(end_date).to_pydatetime(),
    ).rename(columns=series)

    monthly = df.resample("M").last()
    monthly["us_m2_yoy_pct"] = monthly["us_m2"].pct_change(12, fill_method=None) * 100
    monthly["us_cpi_yoy_pct"] = monthly["us_cpi"].pct_change(12, fill_method=None) * 100
    monthly["us_m1_yoy_pct"] = monthly["us_m1"].pct_change(12, fill_method=None) * 100

    for col in ["us_m2_yoy_pct", "us_cpi_yoy_pct", "us_m1_yoy_pct"]:
        monthly[f"{col}_lag1"] = monthly[col].shift(1)

    monthly["us_cpi_yoy_pct_lag1"] = monthly["us_cpi_yoy_pct_lag1"].ffill()
    monthly = monthly.dropna()

    return monthly[
        [
            "usd_index",
            "us10y",
            "us_m2_yoy_pct_lag1",
            "us_cpi_yoy_pct_lag1",
            "us_m1_yoy_pct_lag1",
            "wti",
        ]
    ].copy()


def _download_factor_panel(
    rqdatac,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    factor_start_date = _make_factor_download_start_date(start_date)
    china_factor = _download_china_factor_panel(rqdatac, factor_start_date, end_date)
    us_factor = _download_us_factor_panel(factor_start_date, end_date)
    all_factors = china_factor.merge(us_factor, left_index=True, right_index=True)
    all_factors["us10y_cn10y_diff"] = all_factors["us10y"] - all_factors["cn10y_yield"]

    if all_factors.empty:
        raise ValueError("因子表为空。请检查日期区间和数据权限。")

    all_factors.index = pd.to_datetime(all_factors.index)
    all_factors = all_factors.sort_index()
    all_factors.index.name = "date"
    return _clip_factor_panel_to_range(all_factors, start_date, end_date)


def _save_price_panel_to_csv(price: pd.DataFrame, csv_path: Path) -> Path:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    output = price.copy()
    output.index = pd.to_datetime(output.index)
    output = output.sort_index()
    output.index.name = "date"
    output.columns = output.columns.map(str)
    output.to_csv(csv_path, encoding="utf-8-sig")
    return csv_path


def _save_factor_panel_to_csv(factor: pd.DataFrame, csv_path: Path) -> Path:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    output = factor.copy()
    output.index = pd.to_datetime(output.index)
    output = output.sort_index()
    output.index.name = "date"
    output.to_csv(csv_path, encoding="utf-8-sig")
    return csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description="下载全天候策略价格数据到本地 CSV")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "default.yaml"),
        help="YAML 配置文件路径",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    csv_path = cfg.data.resolve_csv_path(cfg.project_root)
    factor_csv_path = cfg.data.resolve_factor_csv_path(cfg.project_root)
    order_book_ids = [asset.order_book_id for asset in cfg.assets]

    print(f"加载配置: {args.config}")
    print(f"下载区间: {cfg.start_date} ~ {cfg.end_date_str}")
    print(f"资产数: {len(order_book_ids)}")
    print(f"价格输出: {csv_path}")
    print(f"因子输出: {factor_csv_path}")

    rqdatac = _init_rq()
    daily_price = _download_price_panel(
        rqdatac,
        order_book_ids,
        cfg.start_date,
        cfg.end_date_str,
        cfg.price_field,
    )
    all_factors = _download_factor_panel(rqdatac, cfg.start_date, cfg.end_date_str)

    saved_price_path = _save_price_panel_to_csv(daily_price, csv_path)
    saved_factor_path = _save_factor_panel_to_csv(all_factors, factor_csv_path)

    print("\n下载完成:")
    print(f"  daily_price 行数: {len(daily_price)}")
    print(f"  daily_price 列数: {len(daily_price.columns)}")
    print(f"  daily_price 保存到: {saved_price_path}")
    print(f"  all_factors 行数: {len(all_factors)}")
    print(f"  all_factors 列数: {len(all_factors.columns)}")
    print(f"  all_factors 保存到: {saved_factor_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
