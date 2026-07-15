# 全天候策略初始化与调仓逻辑

本文档记录当前代码中策略初始化、目标权重生成、交易流水生成和中间调仓的真实逻辑，便于后续检查和迭代。

## 1. 初始化配置与数据

配置入口为 `config/default.yaml`。当前默认逻辑：

- 起始日：`2021-01-01`
- 结束日：`2025-12-31`
- 调仓频率：月度 `M`
- 策略：由命令行 `--strategy` 决定；`0=fixed`，`1=inverse_vol`，`2=risk_parity`，`all` 为三种一起运行
- 初始资金：`10,000,000`
- 交易费率：`0.0002`
- 权重上下限：`0.01 ~ 0.85`
- 回测权重类型：`weight_type: "asset"`
- 资产池：配置中的全部 `assets`
- 战术层：默认启用 `宏观分数 + 技术分数 + 现金残差调节`

当前最常用的实际运行命令是：

```bash
python scripts/run_backtest.py --config config/default.yaml --strategy 0
python scripts/run_backtest.py --config config/default.yaml --strategy 1
python scripts/run_backtest.py --config config/default.yaml --strategy 2
python scripts/run_backtest.py --config config/default.yaml --strategy all
```

如果不显式传 `--strategy`，脚本默认等价于 `--strategy 0`。

`runner.run_backtest` 的主流程：

1. 读取配置资产的 `order_book_id`。
2. 从 `data/daily_price.csv` 读取本地缓存价格。
3. 将价格列名从 `order_book_id` 改成资产名称。
4. 将日频价格转换为回测框架需要的长表行情 `market_data`。
5. 将日频价格按 `rebalance_freq` 重采样为周期价格。
6. 基于周期价格计算收益率。
7. 生成目标权重表 `weight_schedule`。
8. 取第一期有效调仓信号对应的实际交易日，作为正式回测起点。
9. 将行情裁到这个正式起点之后，再把目标权重表转换成交易流水 `flow_data`。
10. 调用复制过来的回测框架 `run_portfolio` 执行回测。

本地 CSV 读取时会显式校验配置资产是否齐全。如果某个 `order_book_id` 缺列，直接抛错，避免静默少跑资产。

## 1.1 数据下载与缓存

新增独立脚本 `scripts/download_data.py`：

1. 读取配置中的 `start_date`、`end_date`、`price_field`、`assets` 和 `data.csv_path`。
2. 在脚本内部直接连接米筐并下载宽表价格。
3. 同时下载中国宏观因子与 FRED 美国宏观因子。
4. 将结果分别保存为 `daily_price.csv` 与 `all_factors.csv`（或配置中的其它路径）。

因此，日常使用的推荐顺序变成：

1. 先运行 `python scripts/download_data.py --config config/default.yaml`
2. 再运行 `python scripts/run_backtest.py --config config/default.yaml --strategy 0/1/2/all`

这样回测阶段不再重复访问 RQ 下载同一份资产价格数据。

## 1.2 文件职责与阅读顺序

如果你是顺着当前代码继续改，最值得先看的文件一般是：

1. `config/default.yaml`
2. `scripts/run_backtest.py`
3. `all_weather/runner.py`
4. `all_weather/flow.py`
5. `all_weather/weights/strategic.py`
6. `all_weather/weights/signals_macro.py`
7. `all_weather/weights/signals_technical.py`
8. `all_weather/weights/tactical.py`

对应职责大致是：

- `config/default.yaml`：资产池、参数、宏观规则、技术规则
- `scripts/run_backtest.py`：CLI 入口，负责解释 `--strategy 0/1/2/all`
- `all_weather/runner.py`：把本地 CSV、信号、权重、流水、回测串起来
- `all_weather/flow.py`：目标权重如何落成真实交易流水
- `weights/strategic.py`：`fixed / inverse_vol / risk_parity`
- `weights/signals_macro.py`：宏观因子如何映射成资产分数
- `weights/signals_technical.py`：价格趋势如何映射成技术分数
- `weights/tactical.py`：宏观分数 + 技术分数如何变成实际调权

## 2. 目标权重生成

目标权重表由 `flow.build_weight_schedule` 生成：

- index：调仓信号日，来自月度收益率的日期索引
- columns：资产名称
- value：该信号日的目标权重

当前支持三类策略：

- `fixed`：固定比例战略权重
- `inverse_vol`：逆波动率战略权重
- `risk_parity`：风险平价战略权重

若命令行未显式传入 `--strategy`，默认按 `0`，也就是 `fixed` 运行。

当前绩效可视化也统一改为 ECharts HTML 仪表页：

- 单策略：展示收益率 / 回撤 / 资产价值联动图，并附年度收益表
- 多策略：先展示净值对比图和多策略绩效汇总表，右侧按钮进入各策略详情页，详情页带返回按钮

### 三类战略权重

`fixed` 直接使用 `center_weight` 归一化后的中枢权重。

`inverse_vol` 使用最近 `lookback` 期收益率窗口的波动率倒数作为权重。

`risk_parity` 使用最近 `lookback` 期收益率窗口：

1. 删除全为空的资产列。
2. 缺失收益填 0。
3. 计算协方差矩阵，并加一个很小的对角扰动。
4. 从等权开始迭代。
5. 每轮计算风险贡献。
6. 通过指数更新降低风险贡献偏离。
7. 收敛后返回近似等风险贡献权重。
8. 再投影到满足“每个资产上下限 + 总和为 1”的权重集合里。

### 宏观 + 技术战术调整逻辑

当 `tactical.enabled: true` 时，`fixed / inverse_vol / risk_parity` 三种战略权重都会继续叠加战术层：

1. 从 `data/all_factors.csv` 读取宏观宽表。
2. `weights/signals_macro.py` 按 `macro_rules` 逐条计算资产宏观分数。
3. `weights/signals_technical.py` 按资产类别计算技术趋势分数。
4. `weights/tactical.py` 将 `(macro_score + tech_score)` 按理论上界归一化到 `[-1, 1]`。
5. 归一化分数乘以 `tactical_adjust_caps`，得到每个风险资产相对战略权重的增减幅度。
6. 货币资产权重不单独打分，而是作为残差：

```python
w_cash = 1 - sum(w_risk_assets)
```

这样风险资产被调高时，现金自动下降；风险资产被调低时，现金自动上升。

当前版本按 `gf_asset_alloc` 同款方式处理：

- 风险资产目标权重：`base_weight + normalized_score * tactical_adjust_cap`
- 若某个风险资产被调到负数，则直接截到 `0`
- 若风险资产合计超过 `1`，则按比例整体缩放回 `1`
- 现金资产权重始终取 `1 - sum(risk_assets)`

也就是说，`fixed / inverse_vol / risk_parity` 三种战略权重的区别保留，但战术层的加减仓与现金残差行为统一对齐到 `gf_asset_alloc`。

### 正式回测起点

当前版本这里也对齐 `gf_asset_alloc`：

1. 先用历史价格和宏观数据生成完整的信号/目标权重。
2. 找到第一期有效目标权重对应的实际交易日。
3. 正式净值曲线从这个交易日开始。

因此，配置里的 `start_date` 更像“原始数据读取起点 + 信号预热起点”，不一定等于最终展示的净值起点。如果宏观规则需要先预热 12 个月，或者第一期调仓日要顺延到下一个交易日，前面那段时间不会再以“净值一直等于 1” 的形式出现在正式回测结果里。

## 3. 策略初始化流水

交易流水由 `flow.make_flow_data_from_weights` 生成。

第一条流水永远是现金划入：

```python
{
    "买卖日期": trade_calendar[0],
    "证券代码": "CASH",
    "买卖数量": initial_cash,
    "买卖权重": 0.0,
    "买卖价格": 1.0,
    "买卖方向": "划入",
    "交易批次": 0,
}
```

然后初始化上一期已执行目标权重：

```python
previous = pd.Series(0.0, index=target_weights.columns)
```

因此第一次有效调仓时：

```python
delta = target - previous
```

等价于：

```python
delta = 初始目标权重 - 0
```

所有可交易且目标权重大于 0 的资产都会生成买入流水。

## 4. 中间调仓逻辑

每个目标权重信号日都会对齐到实际交易日：

```python
trade_date = align_trade_date(signal_date, trade_calendar)
```

对齐规则：

- 使用 `>= signal_date` 的最近一个交易日
- 如果信号日超出行情范围，则该信号会被截断或抛错

每个调仓日会重新归一化目标权重：

```python
target = normalize_weights(target.reindex(previous.index).fillna(0.0))
```

随后逐个资产检查该交易日是否可交易：

- 资产代码存在
- 行情表中有该代码
- `trade_date` 当天价格非空
- 价格大于 0

只有满足这些条件的资产才会参与本次调仓。

调仓差额：

```python
delta = target - previous
```

交易方向：

- `delta > 0`：买入
- `delta < 0`：卖出
- `abs(delta) <= 1e-8`：忽略

流水中的 `买卖权重` 为：

```python
abs(delta)
```

交易费用为：

```python
initial_cash * abs(delta) * fee_rate
```

当前流水生成顺序：

1. 先生成卖出流水
2. 再生成买入流水
3. 同一批次内按方向和证券代码排序

## 5. 晚上市或中途可用标的处理

这是当前流水逻辑的关键约束。

如果某个资产配置在资产池中，但上市晚于回测开始日，早期调仓日没有有效价格：

- 不报错
- 不生成买入流水
- 不生成卖出流水
- 该资产的目标权重对应资金暂时留在现金
- 该资产的 `previous` 不更新
- 只在第一次遇到缺失价格时打印提示

核心原因是：

```python
previous[valid] = target[valid]
```

这里只更新当期可交易资产。不可交易资产的 `previous` 保持旧值。

因此，当该资产未来第一次有有效价格时：

```python
delta = target - previous
```

会自动形成待建仓权重差，并生成买入流水。

换句话说，当前逻辑不是把晚上市资产的权重分配给其它资产，而是让这部分目标权重暂留现金，等资产可交易后自动建仓。

## 6. 伪代码

```python
# 初始化
cash = initial_cash
previous_weight = {asset: 0 for asset in assets}

# 每个调仓信号日
for signal_date in monthly_returns.index:
    trade_date = next_trade_date(signal_date)

    target_weight = calc_target_weight(signal_date)
    target_weight = normalize(target_weight)

    valid_assets = [
        asset for asset in assets
        if price(asset, trade_date) is not None
        and price(asset, trade_date) > 0
    ]

    delta = target_weight - previous_weight

    for asset in valid_assets:
        if delta[asset] < 0:
            generate_sell_flow(asset, abs(delta[asset]))
        elif delta[asset] > 0:
            generate_buy_flow(asset, abs(delta[asset]))

    # 只更新可交易资产
    previous_weight[valid_assets] = target_weight[valid_assets]

    # 不可交易资产 previous 不动
    # 等未来可交易时，会自动补建仓
```

## 7. 当前逻辑的风险点

### Risk 1：晚上市资产的目标权重暂留现金，可能导致早期组合风险低于目标

如果多个资产在回测早期都不可交易，组合实际仓位会低于目标权重，现金比例被动升高，早期收益和回撤会受到现金拖累。

缓解方式：

- 当前逻辑已经显式打印第一次缺失价格提示。
- 如果未来希望“可交易资产之间重新分配晚上市资产权重”，需要新增一个配置开关，例如 `redistribute_missing_assets: true`，不能静默改变当前行为。

### Risk 2：交易费用按 `initial_cash * 权重差` 估算，不随组合净值变化

当前流水生成阶段不知道回测执行时的实时净值，因此交易费用固定按初始资金估算。这在长期回测中可能低估或高估实际费用。

缓解方式：

- 当前逻辑保持简单、可复现。
- 如果后续需要更精确费用，应把费用计算移入回测撮合阶段，用调仓日实时资产净值计算。

### Risk 3：`previous` 表示“上次成功执行的目标权重”，不等于真实漂移后的持仓权重

资产价格变化会让真实持仓权重偏离 `previous`。当前调仓流水以目标权重差生成，依赖回测框架的 `weight_type="asset"` 按权重指令调整。

缓解方式：

- 当前逻辑适合目标权重型回测。
- 如果未来需要精确按调仓日前真实市值再平衡，应在回测结果/持仓状态中读取真实权重后生成交易差额。
