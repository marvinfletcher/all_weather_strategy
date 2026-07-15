# 全天候资产配置策略

基于 RQData 数据源的模块化全天候多资产配置研究框架。从 `全天候资产配置_RQ基础框架.ipynb` 重构而来，支持 YAML 驱动配置、多资产输入、风险平价权重分配，回测框架已复制到本项目内可独立定制。

当前主流程已经固定为：

`scripts/download_data.py` -> `data/daily_price.csv` / `data/all_factors.csv` -> `scripts/run_backtest.py` -> `all_weather.runner` -> `all_weather.flow` -> `all_weather.backtest`

## 目录结构

```
all_weather_strategy/
├── config/default.yaml              # 参数 + 资产池
├── data/                            # download_data.py 生成的价格/宏观因子 CSV
├── all_weather/                     # 项目包
│   ├── config.py                    # YAML 加载 + 数据类
│   ├── data_loader.py               # 本地 CSV 行情/因子读取
│   ├── market.py                    # 行情标准化 / 收益率
│   ├── weights/                     # 战略权重 + 战术调整
│   │   ├── strategic.py             # fixed / inverse_vol / risk_parity
│   │   ├── signals_macro.py         # 宏观因子打分
│   │   ├── signals_technical.py     # 技术趋势打分
│   │   └── tactical.py              # 总分 -> 实际调权
│   ├── flow.py                      # 目标权重 -> 交易流水
│   ├── backtest/                    # 复制自 gf_alpha.backtest（已隔离外部依赖）
│   └── runner.py                    # 端到端流水线编排
└── scripts/
    ├── download_data.py             # 单独下载 RQ 数据并保存 CSV
    └── run_backtest.py              # CLI 入口
```

## 安装

```bash
cd all_weather_strategy
pip install -r requirements.txt
```

## RQ 凭证配置

凭证固定从项目根目录 `.env` 读取，格式为 `RQ_LICENSE='你的license'`。`scripts/download_data.py` 会访问 RQ 下载资产价格与中国宏观因子，并通过 FRED 下载美国宏观因子。

```bash
# 复制 .env.example 为 .env 并填入真实 license
cp .env.example .env
```

## 使用

### 1. 先下载本地价格 CSV

```bash
python scripts/download_data.py --config config/default.yaml
```

默认会按 `config/default.yaml` 里的 `data.csv_path` 保存到 `data/daily_price.csv`，并在同目录额外生成一份 `all_factors.csv`。

### 2. 再运行回测

```bash
python scripts/run_backtest.py --config config/default.yaml --strategy 0
```

此时资产价格会从 `daily_price.csv` 读取，宏观因子会从 `all_factors.csv` 读取，不再每次回测都重新拉取数据。当前实际运行方式以命令行 `--strategy` 为准，常用写法就是下面四种：

```bash
python scripts/run_backtest.py --config config/default.yaml --strategy 0
python scripts/run_backtest.py --config config/default.yaml --strategy 1
python scripts/run_backtest.py --config config/default.yaml --strategy 2
python scripts/run_backtest.py --config config/default.yaml --strategy all
```

映射关系：

- `--strategy 0` -> `fixed`
- `--strategy 1` -> `inverse_vol`
- `--strategy 2` -> `risk_parity`
- `--strategy all` -> 三种策略一起运行，并生成“总览页 + 三个策略详情按钮”的 ECharts 仪表页

如果不显式传 `--strategy`，当前脚本默认等价于 `--strategy 0`，也就是 `fixed`。

单策略时，会直接弹出一个 ECharts 仪表页，包含“收益率 / 回撤 / 资产价值”联动图和各年份年度收益表。多策略模式会先弹出总览页：上方是三策略净值对比图，下方是多策略绩效汇总表，右侧三个按钮分别进入 `fixed / inverse_vol / risk_parity` 的单独详情页，详情页里同样包含联动图、年度收益表和返回按钮。

## 执行链路

### 下载阶段

`scripts/download_data.py` 做三件事：

1. 从 `.env` 读取 `RQ_LICENSE`
2. 下载配置资产的日频价格宽表
3. 下载中国 + 美国宏观因子，并整理成月频 `all_factors.csv`

这里会额外预留一段宏观预热区间，所以因子下载起点通常会早于回测 `start_date`。这样做是为了让同比、趋势窗口、滞后一期等规则在正式回测开始前就能算出有效信号。

### 回测阶段

`scripts/run_backtest.py` 会把命令行参数传给 `all_weather.runner.run_backtest()` 或 `run_backtests()`，核心步骤是：

1. 从本地 CSV 读取 `daily_price` 与 `all_factors`
2. 把价格从宽表整理成回测框架使用的长表 `market_data`
3. 把日频价格重采样成月频价格，再转成月收益率
4. 用 `weights/strategic.py` 生成战略权重
5. 用 `signals_macro.py + signals_technical.py + tactical.py` 生成战术调权
6. 用 `flow.py` 把目标权重表变成交易流水
7. 用 `backtest/portfolio_backtest.py` 跑净值，再由 `performance_analysis.py` 做绩效分析

### 一个容易忽略但很关键的行为

如果某个资产在早期调仓日还没有有效价格，当前实现不会报错，也不会把这部分权重强行分给别的资产，而是：

- 本次调仓忽略该资产
- 对应资金暂时留在现金
- 等该资产后续首次有有效价格时，再按 `target - previous` 自动补建仓

这也是 `LOGIC.md` 和 `all_weather/flow.py` 里反复强调的真实交易语义。

## 配置重点与阅读顺序

`config/default.yaml` 里最值得先看的不是全部参数，而是这几组：

- `data`: 本地价格与因子 CSV 的路径
- `assets`: 资产池、代码、资产类别、中枢权重
- `tactical`: 现金资产是谁、每个风险资产最多允许偏离中枢多少
- `macro_rules`: 每个宏观因子影响哪些资产、方向正负、趋势窗口多长
- `technical`: 技术打分按什么资产类别走规则
- `performance`: 基准代码、无风险利率、是否绘图

如果你接下来要继续读代码，最快顺序通常是：

1. `README.md`
2. `config/default.yaml`
3. `scripts/run_backtest.py`
4. `all_weather/runner.py`
5. `all_weather/flow.py`
6. `all_weather/weights/strategic.py`
7. `all_weather/weights/tactical.py`
8. `LOGIC.md`

其中最值得盯住的三张中间表是：

- `monthly_ret`
- `weight_schedule`
- `flow_data`

这三张表基本就把“信号 -> 权重 -> 交易”的主链串起来了。

## 支持的策略

| strategy | 说明 |
|---|---|
| `fixed` | 固定比例战略权重，再叠加宏观+技术战术调整 |
| `inverse_vol` | 逆波动率战略权重，再叠加宏观+技术战术调整 |
| `risk_parity` | 风险平价战略权重，再叠加宏观+技术战术调整 |

## 扩展点

- 数据层：回测阶段统一通过 `data_loader.py` 读取本地价格/因子 CSV；下载逻辑集中在 `scripts/download_data.py`
- 战略权重：在 `weights/strategic.py` 增加 Black-Litterman、目标波动等
- 宏观信号：在 `weights/signals_macro.py` 调整因子规则、趋势窗口和影响方向
- 技术信号：在 `weights/signals_technical.py` 调整各资产类别的趋势打分方法
- 战术调权：在 `weights/tactical.py` 调整总分到权重的映射、带宽和现金吸收逻辑；当前默认仿照 `gf_asset_alloc`，按 `base_weight + normalized_score * cap` 调整风险资产，现金吸收残差
- 回测框架：`backtest/portfolio_backtest.py` 已在项目内，可自由定制
- 绩效分析：统一使用 `backtest/performance_analysis.py`
