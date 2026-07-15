# 全天候策略初始化与调仓逻辑

本文档记录当前代码中策略初始化、目标权重生成、交易流水生成和中间调仓的真实逻辑，便于后续检查和迭代。

## 1. 初始化配置与数据

配置入口为 `config/default.yaml`。当前默认逻辑：

- 起始日：`2021-01-01`
- 结束日：`null`，运行时使用当天日期
- 调仓频率：月度 `M`
- 策略：`fixed_plus_tilt`
- 初始资金：`10,000,000`
- 交易费率：`0.0002`
- 权重上下限：`0.0 ~ 0.85`
- 回测权重类型：`weight_type: "asset"`
- 资产池：配置中的全部 `assets`

`runner.run_backtest` 的主流程：

1. 读取配置资产的 `order_book_id`。
2. 通过 `RQDataProvider.get_price_panel` 拉取日频价格。
3. 将价格列名从 `order_book_id` 改成资产名称。
4. 将日频价格转换为回测框架需要的长表行情 `market_data`。
5. 将日频价格按 `rebalance_freq` 重采样为周期价格。
6. 基于周期价格计算收益率。
7. 生成目标权重表 `weight_schedule`。
8. 将目标权重表转换成交易流水 `flow_data`。
9. 调用复制过来的回测框架 `run_portfolio` 执行回测。

`RQDataProvider.get_price_panel` 会显式校验 RQ 是否返回了所有配置资产。如果某个 `order_book_id` 完全未返回，直接抛错，避免静默少跑资产。

## 2. 目标权重生成

目标权重表由 `flow.build_weight_schedule` 生成：

- index：调仓信号日，来自月度收益率的日期索引
- columns：资产名称
- value：该信号日的目标权重

当前支持四类策略：

- `fixed`：固定中枢权重
- `inverse_vol`：逆波动率权重
- `risk_parity`：风险平价权重
- `fixed_plus_tilt`：固定中枢权重 + 趋势战术偏移

当前默认策略为 `fixed_plus_tilt`。

### fixed_plus_tilt 逻辑

1. 从配置中的 `center_weight` 读取战略中枢权重。
2. 对中枢权重归一化，使权重和为 1。
3. 用月度收益计算趋势分数：
   - `fast_ret = monthly_ret.rolling(trend_fast).mean()`
   - `slow_ret = monthly_ret.shift(trend_fast).rolling(trend_slow - trend_fast).mean()`
   - `score = sign(fast_ret - slow_ret)`
4. 分数为：
   - `+1`：近期强于中期历史均值
   - `-1`：近期弱于中期历史均值
   - `0`：无明确信号或数据不足
5. 对每个资产计算原始目标权重：

```python
raw_weight = center_weight + trend_score * tilt_step
```

6. 将原始权重裁剪到 `[min_weight, max_weight]`。
7. 再次归一化到权重和为 1。

### risk_parity 逻辑

`risk_parity` 使用最近 `lookback` 期收益率窗口：

1. 删除全为空的资产列。
2. 缺失收益填 0。
3. 计算协方差矩阵，并加一个很小的对角扰动。
4. 从等权开始迭代。
5. 每轮计算风险贡献。
6. 通过指数更新降低风险贡献偏离。
7. 收敛后返回近似等风险贡献权重。
8. 再按 `[min_weight, max_weight]` 裁剪并归一化。

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
