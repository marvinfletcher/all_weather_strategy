from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

VALID_STRATEGIES = frozenset({"fixed", "inverse_vol", "risk_parity", "fixed_plus_tilt"})


@dataclass
class AssetSpec:
    asset: str
    order_book_id: str
    asset_class: str
    center_weight: float
    tilt_step: float = 0.0


@dataclass
class BacktestEngineConfig:
    margin: bool = False
    allow_loan: bool = False
    weight_type: str = "asset"


@dataclass
class PerformanceConfig:
    benchmark_code: str | None = "000300.XSHG"
    free_rate: float = 2.0
    plot: bool = True


@dataclass
class WeightBounds:
    min: float = 0.0
    max: float = 0.85


@dataclass
class BacktestConfig:
    start_date: str
    end_date: str | None
    rebalance_freq: str
    price_field: str
    initial_cash: float
    trade_fee_rate: float
    strategy: str
    lookback: int
    trend_fast: int
    trend_slow: int
    weight_bounds: WeightBounds
    backtest: BacktestEngineConfig
    performance: PerformanceConfig
    assets: list[AssetSpec] = field(default_factory=list)

    @property
    def end_date_str(self) -> str:
        if self.end_date:
            return self.end_date
        return pd.Timestamp.today().strftime("%Y-%m-%d")

    @property
    def asset_df(self) -> pd.DataFrame:
        return pd.DataFrame([a.__dict__ for a in self.assets])


def _require_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"配置项 `{key}` 必须是对象。")
    return value


def load_config(path: str | Path) -> BacktestConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    assets_raw = raw.get("assets") or []
    if not assets_raw:
        raise ValueError("配置中 `assets` 不能为空。")

    assets = [AssetSpec(**item) for item in assets_raw]
    order_book_ids = [a.order_book_id for a in assets]
    if len(set(order_book_ids)) != len(order_book_ids):
        raise ValueError("assets 中存在重复的 order_book_id。")

    center_sum = sum(a.center_weight for a in assets)
    if center_sum <= 0:
        raise ValueError("assets 的 center_weight 之和必须大于 0。")

    strategy = str(raw.get("strategy", "fixed_plus_tilt"))
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"未知 strategy: {strategy}，可选 {sorted(VALID_STRATEGIES)}")

    bounds_raw = _require_mapping(raw, "weight_bounds") if "weight_bounds" in raw else {"min": 0.0, "max": 0.85}
    backtest_raw = _require_mapping(raw, "backtest") if "backtest" in raw else {}
    perf_raw = _require_mapping(raw, "performance") if "performance" in raw else {}

    return BacktestConfig(
        start_date=str(raw["start_date"]),
        end_date=raw.get("end_date"),
        rebalance_freq=str(raw.get("rebalance_freq", "M")),
        price_field=str(raw.get("price_field", "close")),
        initial_cash=float(raw.get("initial_cash", 10_000_000)),
        trade_fee_rate=float(raw.get("trade_fee_rate", 0.0002)),
        strategy=strategy,
        lookback=int(raw.get("lookback", 36)),
        trend_fast=int(raw.get("trend_fast", 2)),
        trend_slow=int(raw.get("trend_slow", 12)),
        weight_bounds=WeightBounds(
            min=float(bounds_raw.get("min", 0.0)),
            max=float(bounds_raw.get("max", 0.85)),
        ),
        backtest=BacktestEngineConfig(
            margin=bool(backtest_raw.get("margin", False)),
            allow_loan=bool(backtest_raw.get("allow_loan", False)),
            weight_type=str(backtest_raw.get("weight_type", "asset")),
        ),
        performance=PerformanceConfig(
            benchmark_code=perf_raw.get("benchmark_code"),
            free_rate=float(perf_raw.get("free_rate", 2.0)),
            plot=bool(perf_raw.get("plot", True)),
        ),
        assets=assets,
    )
