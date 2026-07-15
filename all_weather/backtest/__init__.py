"""回测执行层（复制自 gf_alpha.backtest，已隔离外部依赖）。"""

from .portfolio_backtest import run_portfolio  # noqa: F401

try:
    from .performance_analysis import run_performance  # noqa: F401
except Exception as exc:  # pragma: no cover - 外部依赖缺失时回退
    run_performance = None
    print(
        f"performance_analysis 不可用，将回退到 simple_performance_stats。"
        f"原因：{type(exc).__name__}: {exc}"
    )
