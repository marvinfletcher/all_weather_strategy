from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

VALID_STRATEGIES = frozenset({"fixed", "inverse_vol", "risk_parity"})
VALID_DATA_SOURCES = frozenset({"csv"})
DEFAULT_FACTOR_CSV = "all_factors.csv"


@dataclass
class AssetSpec:
    asset: str
    order_book_id: str
    asset_class: str
    center_weight: float


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
class DataConfig:
    source: str = "csv"
    csv_path: str = "data/daily_price.csv"
    factor_csv_path: str | None = None

    def resolve_csv_path(self, project_root: Path) -> Path:
        path = Path(self.csv_path).expanduser()
        if path.is_absolute():
            return path.resolve()
        return (project_root / path).resolve()

    def resolve_factor_csv_path(self, project_root: Path) -> Path:
        if self.factor_csv_path:
            path = Path(self.factor_csv_path).expanduser()
            if path.is_absolute():
                return path.resolve()
            return (project_root / path).resolve()
        return self.resolve_csv_path(project_root).with_name(DEFAULT_FACTOR_CSV)


@dataclass
class WeightBounds:
    min: float = 0.01
    max: float = 0.85


@dataclass
class TacticalConfig:
    enabled: bool = True
    cash_asset: str | None = None
    tactical_adjust_caps: dict[str, float] = field(default_factory=dict)
    tech_score_clip_for_norm: float = 1.0


@dataclass
class TechnicalConfig:
    asset_class: dict[str, str] = field(default_factory=dict)
    llt_gamma: float = 0.7
    trend_score_clip: float = 1.0


@dataclass
class BacktestConfig:
    config_path: Path = field(repr=False)
    start_date: str
    end_date: str | None
    rebalance_freq: str
    price_field: str
    data: DataConfig
    initial_cash: float
    trade_fee_rate: float
    strategy: str
    lookback: int
    weight_bounds: WeightBounds
    tactical: TacticalConfig
    macro_defaults: dict[str, Any]
    macro_rules: list[dict[str, Any]]
    technical: TechnicalConfig
    backtest: BacktestEngineConfig
    performance: PerformanceConfig
    assets: list[AssetSpec] = field(default_factory=list)

    @property
    def end_date_str(self) -> str:
        if self.end_date:
            return self.end_date
        return pd.Timestamp.today().strftime("%Y-%m-%d")

    @property
    def project_root(self) -> Path:
        config_dir = self.config_path.parent
        if config_dir.name == "config":
            return config_dir.parent
        return config_dir

    @property
    def asset_df(self) -> pd.DataFrame:
        return pd.DataFrame([a.__dict__ for a in self.assets])


def _require_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"配置项 `{key}` 必须是对象。")
    return value


def _infer_cash_asset(assets: list[AssetSpec]) -> str | None:
    for asset in assets:
        if "货币" in asset.asset_class:
            return asset.asset
    return assets[-1].asset if assets else None


def load_config(
    path: str | Path,
    *,
    validate_strategy: bool = True,
) -> BacktestConfig:
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

    strategy = str(raw.get("strategy", "risk_parity"))
    if strategy not in VALID_STRATEGIES:
        if not validate_strategy:
            strategy = "risk_parity"
        else:
            raise ValueError(f"未知 strategy: {strategy}，可选 {sorted(VALID_STRATEGIES)}")

    data_raw = _require_mapping(raw, "data") if "data" in raw else {}
    data_source = str(data_raw.get("source", "csv"))
    if data_source not in VALID_DATA_SOURCES:
        raise ValueError(f"未知 data.source: {data_source}，可选 {sorted(VALID_DATA_SOURCES)}")

    bounds_raw = _require_mapping(raw, "weight_bounds") if "weight_bounds" in raw else {"min": 0.01, "max": 0.85}
    tactical_raw = _require_mapping(raw, "tactical") if "tactical" in raw else {}
    macro_defaults_raw = _require_mapping(raw, "macro_defaults") if "macro_defaults" in raw else {}
    backtest_raw = _require_mapping(raw, "backtest") if "backtest" in raw else {}
    technical_raw = _require_mapping(raw, "technical") if "technical" in raw else {}
    perf_raw = _require_mapping(raw, "performance") if "performance" in raw else {}
    technical_asset_class = technical_raw.get("asset_class", {})
    if technical_asset_class and not isinstance(technical_asset_class, dict):
        raise ValueError("配置项 `technical.asset_class` 必须是对象。")
    macro_rules = raw.get("macro_rules") or []
    if not isinstance(macro_rules, list):
        raise ValueError("配置项 `macro_rules` 必须是列表。")

    cash_asset = tactical_raw.get("cash_asset") or _infer_cash_asset(assets)

    return BacktestConfig(
        config_path=config_path,
        start_date=str(raw["start_date"]),
        end_date=raw.get("end_date"),
        rebalance_freq=str(raw.get("rebalance_freq", "M")),
        price_field=str(raw.get("price_field", "close")),
        data=DataConfig(
            source=data_source,
            csv_path=str(data_raw.get("csv_path", "data/daily_price.csv")),
            factor_csv_path=(
                str(data_raw["factor_csv_path"])
                if data_raw.get("factor_csv_path") is not None
                else None
            ),
        ),
        initial_cash=float(raw.get("initial_cash", 10_000_000)),
        trade_fee_rate=float(raw.get("trade_fee_rate", 0.0002)),
        strategy=strategy,
        lookback=int(raw.get("lookback", 36)),
        weight_bounds=WeightBounds(
            min=float(bounds_raw.get("min", 0.01)),
            max=float(bounds_raw.get("max", 0.85)),
        ),
        tactical=TacticalConfig(
            enabled=bool(tactical_raw.get("enabled", True)),
            cash_asset=str(cash_asset) if cash_asset is not None else None,
            tactical_adjust_caps={
                str(key): float(value)
                for key, value in (tactical_raw.get("tactical_adjust_caps") or {}).items()
            },
            tech_score_clip_for_norm=float(
                tactical_raw.get("tech_score_clip_for_norm", 1.0)
            ),
        ),
        macro_defaults=dict(macro_defaults_raw),
        macro_rules=[dict(rule) for rule in macro_rules],
        technical=TechnicalConfig(
            asset_class={str(key): str(value) for key, value in technical_asset_class.items()},
            llt_gamma=float(technical_raw.get("llt_gamma", 0.7)),
            trend_score_clip=float(technical_raw.get("trend_score_clip", 1.0)),
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
