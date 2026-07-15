# 全天候资产配置策略

基于 RQData 数据源的模块化全天候多资产配置研究框架。从 `全天候资产配置_RQ基础框架.ipynb` 重构而来，支持 YAML 驱动配置、多资产输入、风险平价权重分配，回测框架已复制到本项目内可独立定制。

## 目录结构

```
all_weather_strategy/
├── config/default.yaml              # 参数 + 资产池
├── all_weather/                     # 项目包
│   ├── config.py                    # YAML 加载 + 数据类
│   ├── data/rq_provider.py          # RQDataProvider（读 RQ_LICENSE 环境变量）
│   ├── market.py                    # 行情标准化 / 收益率
│   ├── weights/                     # 战略权重 + 战术调整
│   ├── flow.py                      # 目标权重 -> 交易流水
│   ├── backtest/                    # 复制自 gf_alpha.backtest（已隔离外部依赖）
│   ├── performance.py               # 内置回退绩效分析
│   └── runner.py                    # 端到端流水线编排
└── scripts/
    ├── run_backtest.py              # CLI 入口
    └── run_backtest.ipynb           # Notebook 分步调试入口
```

## 安装

```bash
cd all_weather_strategy
pip install -r requirements.txt
```

## RQ 凭证配置

凭证固定从项目根目录 `.env` 读取，格式为 `RQ_LICENSE='你的license'`，YAML 不存凭证，避免泄露。

```bash
# 复制 .env.example 为 .env 并填入真实 license
cp .env.example .env
```

## 使用

### CLI 一键运行

```bash
python scripts/run_backtest.py --config config/default.yaml
```

默认会输出常规绩效图；若环境已安装 `pyecharts`，还会额外把 ECharts 联动时序图渲染到临时 HTML 并自动在浏览器中打开。

### Notebook 分步调试

打开 `scripts/run_backtest.ipynb`，按单元格依次执行。最贵的"取数"步骤只跑一次，调整策略参数或流水逻辑时只需重跑对应单元格，`daily_price` / `market_data` 不重复拉取 RQ 数据。

## 支持的策略

| strategy | 说明 |
|---|---|
| `fixed` | 固定中枢权重（来自 YAML center_weight） |
| `inverse_vol` | 逆波动率权重 |
| `risk_parity` | 风险平价权重（迭代法等风险贡献） |
| `fixed_plus_tilt` | 固定中枢 + 趋势战术调整（默认） |

## 扩展点

- 数据层：替换 `data/rq_provider.py` 即可换数据源
- 战略权重：在 `weights/strategic.py` 增加 Black-Litterman、目标波动等
- 战术调整：在 `weights/tactical.py` 加入宏观因子（PMI、CPI、利率等）
- 回测框架：`backtest/portfolio_backtest.py` 已在项目内，可自由定制
- 绩效分析：`backtest/performance_analysis.py` 已隔离外部依赖，不可用时自动回退到 `performance.py`
