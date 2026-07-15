"""RQ 数据源封装。唯一直接依赖 rqdatac 的位置。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class RQDataProvider:
    """从 Ricequant 拉取日频价格面板。

    凭证固定从项目根目录 `.env` 的 `RQ_LICENSE='...'` 读取。
    """

    def __init__(self):
        try:
            import rqdatac
        except ImportError as exc:  # pragma: no cover - 环境依赖
            raise ImportError("当前环境未安装 rqdatac。请先安装并确认 RQ 数据权限。") from exc

        self.rqdatac = rqdatac

        env_path = Path(__file__).resolve().parents[2] / ".env"
        if not env_path.exists():
            raise RuntimeError("未找到项目根目录 .env 文件，请先配置 RQ_LICENSE。")

        license_value = None
        for line in env_path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if text.startswith("RQ_LICENSE="):
                license_value = text.split("=", 1)[1].strip().strip("'")
                break

        if license_value is None:
            raise RuntimeError("`.env` 中未找到 RQ_LICENSE，请按 RQ_LICENSE='你的license' 格式填写。")

        try:
            rqdatac.init(
                "license",
                license_value,
                ("rqdatad-pro.ricequant.com", 16011),
            )
        except TypeError:
            # 已初始化过的情况
            pass
        except Exception as exc:
            raise RuntimeError(
                "米筐连接失败。请检查项目根目录 .env 中的 RQ_LICENSE 是否正确。"
            ) from exc

    def get_price_panel(
        self,
        order_book_ids: list[str],
        start_date: str,
        end_date: str,
        field: str = "close",
    ) -> pd.DataFrame:
        """拉取日频价格，返回宽表 (index=date, columns=order_book_id)。

        若 RQ 对部分 order_book_id 未返回数据（代码无效或无权限），抛 ValueError
        列出缺失的 order_book_id，避免静默丢弃资产导致回测资产数少于配置。
        """
        raw = self.rqdatac.get_price(
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
            price.index = pd.to_datetime(price.index)
            price = price.sort_index()
        else:
            raise TypeError(f"未识别的 RQ 返回格式：{type(raw)}")

        # 显式校验：rqdatac 对无效 order_book_id 只返回有效部分数据（带 UserWarning），
        # 不会抛错。此处比对请求与返回，缺失时立即报错，避免下游静默丢失资产。
        returned_ids = set(price.columns.astype(str))
        missing = [oid for oid in order_book_ids if str(oid) not in returned_ids]
        if missing:
            raise ValueError(
                "RQ 未返回以下 order_book_id 的价格数据（代码无效或无数据权限）：\n  - "
                + "\n  - ".join(missing)
                + "\n请检查代码格式及 RQ 权限后重试。"
            )
        return price
