"""本地 CSV 数据读取。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_price_panel_from_csv(
    csv_path: str | Path,
    order_book_ids: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """从本地 CSV 读取宽表价格数据。

    约定格式：
    - index: date
    - columns: order_book_id
    - values: 价格字段（如 close）

    这里故意做严格校验：
    - 缺少配置资产列时直接报错
    - 不默默补列，也不默默删资产
    这样可以尽早发现“下载脚本没有把资产下全”这类问题。
    """
    path = Path(csv_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"未找到本地价格文件：{path}\n"
            "请先运行 `python scripts/download_data.py --config ...` 生成 CSV。"
        )

    price = pd.read_csv(path, index_col=0, parse_dates=True)
    if price.empty:
        raise ValueError(f"本地价格文件为空：{path}")

    price.index = pd.to_datetime(price.index, errors="coerce")
    price = price.loc[price.index.notna()].sort_index()
    price = price.loc[~price.index.duplicated(keep="last")]
    price.columns = price.columns.map(str)
    price = price.apply(pd.to_numeric, errors="coerce")

    required_columns = [str(order_book_id) for order_book_id in order_book_ids]
    missing = [order_book_id for order_book_id in required_columns if order_book_id not in price.columns]
    if missing:
        raise ValueError(
            "本地 CSV 缺少以下 order_book_id 列：\n  - "
            + "\n  - ".join(missing)
            + f"\n当前文件：{path}"
        )

    price = price.loc[:, required_columns]
    if start_date:
        price = price.loc[price.index >= pd.Timestamp(start_date)]
    if end_date:
        price = price.loc[price.index <= pd.Timestamp(end_date)]

    if price.empty:
        raise ValueError(
            f"本地 CSV 在区间 {start_date} ~ {end_date} 内无可用数据：{path}"
        )
    return price


def load_factor_panel_from_csv(
    csv_path: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """从本地 CSV 读取月频宏观因子宽表。

    因子文件默认由 scripts/download_data.py 生成，因此这里保持只读，
    不承担任何补数、下载或回写职责。
    """
    path = Path(csv_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"未找到本地因子文件：{path}\n"
            "请先运行 `python scripts/download_data.py --config ...` 生成 all_factors.csv。"
        )

    factor = pd.read_csv(path, index_col=0, parse_dates=True)
    if factor.empty:
        raise ValueError(f"本地因子文件为空：{path}")

    factor.index = pd.to_datetime(factor.index, errors="coerce")
    factor = factor.loc[factor.index.notna()].sort_index()
    factor = factor.loc[~factor.index.duplicated(keep="last")]
    factor.columns = factor.columns.map(str)
    factor = factor.apply(pd.to_numeric, errors="coerce")

    if start_date:
        factor = factor.loc[factor.index >= pd.Timestamp(start_date)]
    if end_date:
        factor = factor.loc[factor.index <= pd.Timestamp(end_date)]

    if factor.empty:
        raise ValueError(
            f"本地因子文件在区间 {start_date} ~ {end_date} 内无可用数据：{path}"
        )
    return factor
