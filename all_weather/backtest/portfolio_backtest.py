# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 09:33:20 2022

@author: GF
"""
#%%导入和全局变量
from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Dict, Union, Any
from enum import Enum  # 枚举类
from datetime import datetime
from copy import deepcopy
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

TABLE_FLOW_DATA = {
    "买卖日期": "updatetime",
    "证券代码": "code",
    "买卖数量": "volume",
    "买卖权重": "weight",  # 对于股票，买入权重为可用现金的权重，卖出权重为可用仓位的权重
    "买卖价格": "price",
    "买卖收益": "pnl",
    "保证金比例":"margin",
    "买卖方向": "direction",
    "盈利方向": "offset",
    "交易费用": "fees",
    "交易批次": "batch"
}
TABLE_FLOW_DATA_CH = dict([val, key] for key, val in TABLE_FLOW_DATA.items())

TABLE_POSITION_DATA = {
    "调整日期": "updatetime",
    "证券代码": "code",
    "持仓数量": "position",
    "持仓方向": "direction",
    "成本价格": "price",
    "是否融资融券": "is_margin",
    "是否收益计提": "is_pnl",#类似买回购这些收益计提自定义方法
}

OFFSET_DICT={
    1:"多",
    -1:"空"
}

class Direction(Enum):
    """
    交易流水方向
    """
    BUY = "买入"
    SELL = "卖出"
    TRANSFER_IN = "划入"
    TRANSFER_OUT = "划出"


@dataclass
class PermanceObj:
    """
    业绩结果对象
    """

    updatetime: datetime = None  # 更新时间

    # 权益
    asset: float = 0.  # 最新权益：资金划转、买入、卖出
    asset_init: float = 0.  # 初始权益：资金划转
    asset_max: float = 0.  # 历史最大权益：行情
    cash: float = 0.  # 可用现金（钱）：资金划转、买入、卖出
    security: float = 0.  # 持仓证券市值（券）、买入、卖出
    holdweight: float = 0.  # 资金占用率：资金划转、买入、卖出

    # 盈亏
    pnl_real_accum: float = 0.  # 累计已实现盈亏：卖出

    pnl_float: float = 0.  # 浮动盈亏：行情
    pnl_all: float = 0.  # 总盈亏=已实现盈亏+浮动盈亏-费用：买入、卖出

    winning_accum: float = 0.  # 盈利交易累计：卖出
    losing_accum: float = 0.  # 亏损交易累计：卖出
    winning_num: int = 0  # 盈利交易累计次数
    losing_num: int = 0  # 亏损交易累计次数
    winning_average: float = 0.  # 平均每笔盈利：卖出
    losing_average: float = 0.  # 平均每笔亏损：卖出

    # 交易费用
    fee_accum: float = 0.  # 总交易手续费：买入、卖出
    total_num: int = 0  # 总交易次数：买入、卖出
    close_num: int = 0  # 总交易笔数：卖出算完成一笔交易

    # 关键绩效
    npv: float = 1.  # 净值：行情
    yields: float = 0.  # 收益率:行情
    drawdown_max: float = 0.  # 最大回撤：行情
    drawdownrate_max: float = 0.  # 最大回撤率：行情
    winning_rate: float = 0.  # 胜率：卖出
    pnl_ratio: float = 0.  # 盈亏比：卖出

    #仓位
    position:int=0 #组合持有资产类别有多少种
    position_detial:str="" #组合持仓的明细



def data_standard(market_data: pd.DataFrame, flow_data: pd.DataFrame, position_data: pd.DataFrame):
    """
    对交易流水和行情数据进行标准化处理
    market_data:DataFrame,行情数据文件，多层级索引三维数据，时间作为第一层级索引，代码作为第二层级索引，行情指标为列
    flow_data:DataFrame,交易流水数据文件
    position_data:DataFrame,最新持仓数据

    返回：预处理后的行情数据、交易流水、持仓数据
    """
    # 行情数据的标准化处理
    market_data = _standardize_market_data(market_data)

    # 交易流水的数据标准化处理
    flow_data = _standardize_flow_data(flow_data)

    # 持仓数据标准化处理
    position_dict = _standardize_position_data(position_data)

    # 检查行情是否覆盖交易流水范围
    _check_market_flow_range(market_data, flow_data)

    return market_data, flow_data, position_dict


def _standardize_market_data(market_data: pd.DataFrame):
    """
    行情数据标准化：输出为(datetime, code)的MultiIndex
    """
    # 检查是否已经设置了datetime和code的多级索引
    is_indexed = False
    if isinstance(market_data.index, pd.MultiIndex) and market_data.index.nlevels == 2:
        # 已经是2层多级索引，认为已经设置好了
        is_indexed = True
        # 确保第一层（datetime）是datetime类型，第二层（code）是字符串类型
        level0_values = market_data.index.get_level_values(0)
        level1_values = market_data.index.get_level_values(1)
        
        need_reindex = False
        new_level0 = level0_values
        new_level1 = level1_values
        
        # 检查第一层是否需要转换为datetime
        if not pd.api.types.is_datetime64_any_dtype(level0_values):
            new_level0 = pd.to_datetime(level0_values)
            need_reindex = True
        
        # 检查第二层是否需要转换为字符串
        if not pd.api.types.is_string_dtype(level1_values):
            new_level1 = level1_values.astype(str)
            need_reindex = True
        
        # 如果需要重新构建索引
        if need_reindex:
            # 使用from_arrays重新构建MultiIndex，保持原有顺序
            market_data.index = pd.MultiIndex.from_arrays([new_level0, new_level1])
        
        # 对索引进行排序
        market_data.sort_index(inplace=True)
    
    # 如果没有设置好索引，进行标准化处理
    if not is_indexed:
        # 检查是否有datetime和code列
        if "datetime" not in market_data.columns or "code" not in market_data.columns:
            raise ValueError("market_data必须包含'datetime'和'code'列，或者已经设置了(datetime, code)的多级索引")
        
        market_data["datetime"] = pd.to_datetime(market_data["datetime"])
        market_data.sort_values("datetime", inplace=True)
        market_data["code"] = market_data["code"].astype(str)
        market_data.set_index(["datetime", "code"], drop=True, inplace=True)

    return market_data


def _standardize_flow_data(flow_data: pd.DataFrame):
    """
    交易流水标准化：重命名中文列并设置 updatetime 索引
    """
    flow_data.rename(columns=TABLE_FLOW_DATA, inplace=True)
    flow_data["updatetime"] = pd.to_datetime(flow_data["updatetime"])
    flow_data["volume"] = flow_data["volume"].apply(lambda x: float(x))
    flow_data["price"] = flow_data["price"].apply(lambda x: float(x))
    flow_data.sort_values("updatetime", inplace=True)
    flow_data.set_index(["updatetime"], drop=True, inplace=True)
    return flow_data


def _standardize_position_data(position_data: pd.DataFrame):
    """
    持仓数据标准化：返回字典形式持仓
    """
    if position_data is not None:
        position_data.rename(columns=TABLE_POSITION_DATA, inplace=True)
        position_data["code"] = position_data["code"].astype(str)
        position_data["is_margin"] = position_data["is_margin"].fillna("")
        position_data["updatetime"] = pd.to_datetime(position_data["updatetime"])
        position_data.set_index(["code"], drop=True, inplace=True)
        position_dict = position_data.to_dict("index")
    else:
        position_dict = {}
    return position_dict


def _check_market_flow_range(market_data: pd.DataFrame, flow_data: pd.DataFrame):
    """
    检查行情数据时间范围是否覆盖流水时间范围
    """
    # 检查行情数据的时间范围是否覆盖交易流水时间范围
    # 注意：交易流水时间使用左标签（在该时点先处理流水，再做行情驱动更新）
    # 框架要求行情至少覆盖【买入/卖出】相关流水的时间范围；
    # 对于首条资金划入（初始资金）等资金划转流水，行情不必覆盖该时间点。
    trade_flow = flow_data[flow_data["direction"].isin([Direction.BUY.value, Direction.SELL.value])]
    if len(trade_flow) > 0:
        flow_start, flow_end = trade_flow.index.min(), trade_flow.index.max()
    else:
        # 纯资金划转等场景：退化为原逻辑
        flow_start, flow_end = flow_data.index[0], flow_data.index[-1]
    market_start, market_end = market_data.index[0][0], market_data.index[-1][0]
    if market_start > flow_start or market_end < flow_end:
        print("行情数据的时间范围没有覆盖买卖流水的时间范围，请补充足够的行情数据:", market_start, flow_start, market_end, flow_end)

        # raise Exception("行情数据的时间范围没有覆盖交易流水的时间范围，请补充足够的行情数据")
    return


def _build_market_cache(market_data: pd.DataFrame):
    """
    预构建行情缓存，避免循环中重复 loc/to_dict
    """
    all_market_dates = list(market_data.index.levels[0])
    market_dict_cache = {}
    for datetime_market in all_market_dates:
        sub = market_data.loc[datetime_market]
        if isinstance(sub, pd.Series):
            code = sub.index[0]
            market_dict_cache[datetime_market] = {str(code): {'close_price': float(sub.iloc[0])}}
        else:
            market_dict_cache[datetime_market] = sub.to_dict("index")
    return all_market_dates, market_dict_cache


def _prepare_first_transfer_flow(backtest_obj: PermanceObj, flow_data: pd.DataFrame, flow_data_new_list: list):
    """
    处理首条资金划入/划出流水，并剔除已处理流水
    """
    if len(flow_data) > 0:
        first_flow = flow_data.iloc[0]
        first_flow_time = flow_data.index[0]
        if first_flow.get("direction") in (Direction.TRANSFER_IN.value, Direction.TRANSFER_OUT.value):
            backtest_obj = cash_transfer(backtest_obj, first_flow)
            backtest_obj.updatetime = first_flow_time
            first_flow_dict = first_flow.to_dict()
            first_flow_dict["updatetime"] = first_flow_time
            flow_data_new_list.append(first_flow_dict)
            flow_data = flow_data.iloc[1:]
        else:
            print("【Warning】交易流水第一条不是资金划入/划出，将按原逻辑在行情遍历中处理。")
    return backtest_obj, flow_data


def _finalize_portfolio_outputs(flow_data_new_list: list, backtest_result_list: list, position_result_list: list):
    """
    将列表结果一次性转换为DataFrame
    """
    if flow_data_new_list:
        flow_data_new = pd.DataFrame(flow_data_new_list)
        flow_data_new.set_index("updatetime", inplace=True, drop=False)
        flow_data_new.rename(columns=TABLE_FLOW_DATA_CH, inplace=True)
    else:
        flow_data_new = pd.DataFrame()

    if backtest_result_list:
        backtest_result = pd.DataFrame(backtest_result_list)
        backtest_result.set_index("updatetime", inplace=True)
    else:
        backtest_result = pd.DataFrame()

    if position_result_list:
        position_result = pd.concat(position_result_list, ignore_index=False)
    else:
        position_result = pd.DataFrame()
    return backtest_result, position_result, flow_data_new


def _run_portfolio_core(
    all_market_dates: list,
    market_dict_cache: Dict,
    flow_data: pd.DataFrame,
    position_dict: Dict,
    backtest_his: Dict = None,
    margin=False,
    allow_loan=True,
    weight_type='cash',
    verbose=True,
    progress_title="回测"
):
    """
    单组合回测内核（输入需已标准化）
    """
    flow_data_new_list = []
    backtest_result_list = []
    position_result_list = []

    backtest_obj = PermanceObj()
    if backtest_his:
        backtest_obj.__dict__.update(backtest_his)

    backtest_obj, flow_data = _prepare_first_transfer_flow(backtest_obj, flow_data, flow_data_new_list)
    if flow_data_new_list:
        backtest_result_list.append({**backtest_obj.__dict__, "updatetime": backtest_obj.updatetime})

    total_days = len(all_market_dates)
    flow_index_values = flow_data.index.values
    flow_cursor = 0
    price_latest_available = {}

    if verbose:
        print(f"\n开始{progress_title}，共 {total_days} 个交易日")
        print("=" * 80)
    progress_interval = max(1, min(total_days // 10, 50))

    for idx, datetime_market in enumerate(all_market_dates):
        if verbose and (idx == 0 or (idx + 1) % progress_interval == 0 or idx == total_days - 1):
            progress = (idx + 1) / total_days * 100
            print(
                f"{progress_title}进度: [{idx + 1}/{total_days}] {progress:.1f}% | "
                f"当前日期: {datetime_market.strftime('%Y-%m-%d')} | "
                f"净值: {backtest_obj.npv:.4f} | 权益: {backtest_obj.asset:,.2f}"
            )

        market_dict = market_dict_cache[datetime_market]
        next_cursor = np.searchsorted(flow_index_values, np.datetime64(datetime_market), side="right")
        flow_data_current = flow_data.iloc[flow_cursor:next_cursor]
        flow_cursor = next_cursor

        last_backtest_obj_asset = backtest_obj.asset
        last_backtest_obj_npv = backtest_obj.npv

        if len(flow_data_current) > 0:
            for batch, group in flow_data_current.groupby("batch"):
                _ = batch
                weight_asset = backtest_obj.asset
                weight_cash = backtest_obj.cash
                weight_position = position_dict
                for datatetime_flow, row in group.iterrows():
                    if weight_type == 'cash':
                        row = weight_to_volume(weight_cash, weight_position, row)
                    if weight_type == 'asset':
                        row = weight_to_volume(weight_asset, weight_position, row)
                    row_dict = row.to_dict()
                    row_dict["updatetime"] = datatetime_flow
                    flow_data_new_list.append(row_dict)
                    backtest_obj, position_dict = on_flow(
                        datatetime_flow, backtest_obj, row, position_dict, margin, allow_loan=allow_loan
                    )

        last_backtest_obj = PermanceObj()
        last_backtest_obj.asset = last_backtest_obj_asset
        last_backtest_obj.npv = last_backtest_obj_npv
        backtest_obj = on_market(
            datetime_market, market_dict, last_backtest_obj, backtest_obj, position_dict, price_latest_available, margin
        )

        if len(flow_data_current) > 0:
            direction_values = flow_data_current["direction"].values
            if Direction.BUY.value in direction_values or Direction.SELL.value in direction_values:
                position_df = pd.DataFrame.from_dict(position_dict, orient="index")
                position_df["code"] = position_df.index
                position_result_list.append(position_df)

        backtest_result_list.append({**backtest_obj.__dict__, "updatetime": datetime_market})

    if verbose:
        print("=" * 80)
        print(f"{progress_title}完成！")
        print(f"回测期间: {all_market_dates[0].strftime('%Y-%m-%d')} 至 {all_market_dates[-1].strftime('%Y-%m-%d')}")
        print(f"总交易日数: {total_days}")
        print(f"最终净值: {backtest_obj.npv:.4f}")
        print(f"累计收益率: {backtest_obj.yields*100:.2f}%")
        print(f"最终权益: {backtest_obj.asset:,.2f} 元")
        print("=" * 80)

    backtest_result, position_result, flow_data_new = _finalize_portfolio_outputs(
        flow_data_new_list, backtest_result_list, position_result_list
    )
    return backtest_result, position_result, flow_data_new, backtest_obj


def fee_cut(backtest_obj: PermanceObj, flow_dict: Dict):
    """
    买入卖出操作扣减手续费
    """
    # 手续费计算到总盈亏里面
    fee = flow_dict.get("fees", 0.)  # 交易费用
    backtest_obj.fee_accum += fee
    backtest_obj.pnl_all -= fee
    backtest_obj.cash -= fee
    return backtest_obj


def cash_transfer(backtest_obj: PermanceObj, flow_dict: Dict):
    """
    根据交易流水的资金划转更新最新绩效对象
    """
    if flow_dict.get("direction") == Direction.TRANSFER_IN.value:  # 入金操作
        # 更新可用现金、权益、资金利用率
        backtest_obj.cash += flow_dict.get("volume", 0)
        backtest_obj.asset += flow_dict.get("volume", 0)
        backtest_obj.holdweight = 1 - backtest_obj.cash / backtest_obj.asset
        backtest_obj.asset_init += flow_dict.get("volume", 0)
    elif flow_dict.get("direction") == Direction.TRANSFER_OUT.value:  # 出金操作
        # 更新可用现金、权益、资金利用率
        backtest_obj.cash -= flow_dict.get("volume", 0)
        backtest_obj.asset -= flow_dict.get("volume", 0)
        backtest_obj.holdweight = 1 - backtest_obj.cash / backtest_obj.asset
        backtest_obj.asset_init -= flow_dict.get("volume", 0)

    backtest_obj.asset_max = max(backtest_obj.asset_max, backtest_obj.asset)

    return backtest_obj


def weight_to_volume(weight_cash: float, weight_position: dict, flow_dict: Dict):
    """
    权重转换成交易数量(向下取整)
    保持原始逻辑：flow_dict可以是Series或dict，都支持.get()方法
    指数等高价格标的(>1000)使用1手，否则使用100手，避免 volume=0 导致无法买入
    """
    # 如果设置了交易权重，则转换成数量
    if flow_dict.get("weight", 0) > 0:
        weight = flow_dict.get("weight", 0)
        price = float(flow_dict.get("price", 0) or 0)
        if price <= 0:
            return flow_dict
        # 指数等高价格标的用1手，ETF/股票用100手，避免 int(x/100)*100=0 导致无法买入
        lot_size = 1 if price > 5000 else 100
        if flow_dict.get("direction") == Direction.BUY.value:  # 买入操作
            flow_dict["volume"] = int(weight_cash * weight / price / lot_size) * lot_size
        elif flow_dict.get("direction") == Direction.SELL.value:  # 卖出操作
            code_pos = flow_dict["code"] + "@" + OFFSET_DICT[flow_dict.get("offset", 1)]
            try:
                flow_dict["volume"] = int(weight_position[code_pos]["position"] * weight / lot_size) * lot_size
            except:
                print(f'【Warning】卖出{code_pos}时无多单持仓，因此不执行卖出。原因可能为现金额不足而未创建多单仓位')

    return flow_dict


def win_lost_compare(pnl_real, backtest_obj: PermanceObj):
    """
    卖出完成一笔交易的盈亏对比
    """
    if pnl_real > 0:
        backtest_obj.winning_accum += abs(pnl_real)
        backtest_obj.winning_num += 1
    else:
        backtest_obj.losing_accum += abs(pnl_real)
        backtest_obj.losing_num += 1

    backtest_obj.winning_average = backtest_obj.winning_accum / backtest_obj.winning_num if backtest_obj.winning_num > 0 else 0
    backtest_obj.losing_average = backtest_obj.losing_accum / backtest_obj.losing_num if backtest_obj.losing_num > 0 else 0
    backtest_obj.winning_rate = backtest_obj.winning_num / backtest_obj.close_num
    backtest_obj.pnl_ratio = backtest_obj.winning_average / backtest_obj.losing_average if backtest_obj.losing_average > 0 else 0

    return backtest_obj


def stock_buy(backtest_obj: PermanceObj, flow_dict: Dict, position_dict: Dict, updatetime: datetime, allow_loan: bool):
    """
    根据交易流水的股票买入操作记录更新最新绩效对象
    """
    code = flow_dict.get("code")
    if flow_dict.get("direction") == Direction.BUY.value:  # 买入操作

        # 更新持券市值、可用现金
        # print(flow_dict.get("volume", 0),type(flow_dict.get("volume", 0)))
        # print(flow_dict.get("price", 0),type(flow_dict.get("price", 0)))
        value = flow_dict.get("volume", 0) * flow_dict.get("price", 0.)* flow_dict.get("margin", 1.) # 市值

        if not allow_loan and backtest_obj.cash < value:
            return backtest_obj, position_dict

        backtest_obj.cash -= value
        backtest_obj.security += value

        # if str(updatetime.date())=="2019-02-15":
        #     print("买入",backtest_obj.security,backtest_obj.asset)
        backtest_obj.holdweight = backtest_obj.security / backtest_obj.asset


        # 扣减手续费
        backtest_obj = fee_cut(backtest_obj, flow_dict)

        # 买入算一次交易操作
        backtest_obj.total_num += 1

        # 记录持仓（优化：直接使用in检查字典键，不需要调用keys()）
        code_position = code + "@" + OFFSET_DICT[flow_dict.get("offset", 1)]
        if code_position in position_dict:  # 已有仓位

            old_position = position_dict[code_position]["position"]
            old_price = position_dict[code_position]["price"]
            new_price = flow_dict.get("price", 0.)
            new_volume = flow_dict.get("volume", 0)
            if new_volume>0:
                avg_price = (old_position * old_price + new_volume * new_price) / (old_position + new_volume)
                position_dict[code_position]["position"] += new_volume
                position_dict[code_position]["price"] = avg_price
            position_dict[code_position]["updatetime"] = updatetime

        else:  # 新持仓
            new_position = {
                "updatetime": updatetime,
                "position": flow_dict.get("volume", 0),
                "direction":flow_dict.get("offset",1),
                "price": flow_dict.get("price", 0.),
                "is_margin": "",
                "is_pnl":1 if flow_dict.get("pnl",0)>0 else 0
            }
            position_dict[code+"@"+OFFSET_DICT[flow_dict.get("offset",1)]] = new_position

    return backtest_obj, position_dict


def stock_sell(
    backtest_obj: PermanceObj,
    flow_dict: Dict,
    position_dict: Dict,
    updatetime: datetime,
    margin,
    allow_loan: bool = True,
):
    """
    根据交易流水的股票卖出操作记录更新最新绩效对象
    """
    code = flow_dict.get("code")
    code_position = code + "@" + OFFSET_DICT.get(flow_dict.get("offset", 1),'多')
    if flow_dict.get("direction") == Direction.SELL.value and \
            code_position in position_dict:  # 卖出操作（优化：直接使用in检查字典键）

        # 检查可平量是否足够
        pos_vol = position_dict[code_position]["position"]
        sell_vol = flow_dict.get("volume", 0)
        
        if pos_vol > 0 and pos_vol < sell_vol:
            print(f"【Warning】卖出数量大于持仓量，可平仓量不足，仅卖出剩余持仓。持仓={pos_vol} 卖出申请={sell_vol} @ {updatetime}")
            sell_vol = pos_vol
        elif pos_vol == 0 and pos_vol < sell_vol:
            # allow_loan=False 时，上游买入可能因现金不足未成交，对应卖出不应再强平，否则与持仓不一致
            if not allow_loan:
                print(
                    f"【Warning】禁止融资模式下可平量不足，跳过卖出：{code_position} 持仓={pos_vol} 卖出申请={sell_vol} @ {updatetime}"
                )
                return backtest_obj, position_dict
            raise Exception("卖出数量大于持仓量，可平仓量不足，请检查！")

        # 更新持券市值、可用现金
        if flow_dict.get("margin", 1.)<1:
            value = flow_dict.get("volume", 0) * position_dict[code_position]["price"]* flow_dict.get("margin", 1.)  # 市值
        else:
            value = flow_dict.get("volume", 0) * flow_dict.get("price", 0.)* flow_dict.get("margin", 1.)  # 市值
        backtest_obj.cash += value
        if margin:
            backtest_obj.security-=value

        # if str(updatetime.date())=="2019-02-15":
        #     print("卖出",flow_dict.get("volume", 0),value,backtest_obj.security,backtest_obj.asset)
        backtest_obj.holdweight = backtest_obj.security / backtest_obj.asset


        # 计算已实现盈亏
        cost = position_dict[code_position]["price"]
        pnl_real = (flow_dict["price"] - cost) * flow_dict["volume"]*flow_dict.get("offset",1)
        if flow_dict.get("pnl",0)>0:
            pnl_real+=float(flow_dict.get("pnl"))
        backtest_obj.pnl_real_accum += pnl_real
        backtest_obj.pnl_all += pnl_real

        # 扣减手续费
        backtest_obj = fee_cut(backtest_obj, flow_dict)

        # 卖出算一次交易操作、完成一笔交易
        backtest_obj.total_num += 1
        backtest_obj.close_num += 1

        # 每笔交易的盈亏对比
        backtest_obj = win_lost_compare(pnl_real, backtest_obj)

        # 记录持仓
        new_volume = flow_dict.get("volume", 0)
        new_position = position_dict[code_position]["position"] - new_volume
        position_dict[code_position]["updatetime"] = updatetime
        if new_position > 0:
            position_dict[code_position]["position"] -= new_volume
        else:
            position_dict[code_position]["position"]=0
            # del position_dict[code_position]

    return backtest_obj, position_dict


def on_flow(updatetime: datetime, backtest_obj: PermanceObj, \
            flow_dict: Dict, position_dict: Dict,margin, allow_loan: bool):
    """
    根据一条交易流水更新最新组合绩效和持仓情况
    updatetime:datetime,更新时间
    backtest_obj：PermanceObj，最新组合绩效的对象
    flow_dict：dict，交易流水表的一行记录
    position_dict:dict,组合最新的持仓记录

    返回：更新后的最新组合绩效对象和最新持仓记录
    """
    # print("流水更新----------------------")
    # print(updatetime,flow_dict,backtest_obj,position_dict)

    # 更新资金划转操作
    backtest_obj = cash_transfer(backtest_obj, flow_dict)

    # 更新股票的买入操作
    backtest_obj, position_dict = stock_buy(backtest_obj, flow_dict, position_dict, updatetime, allow_loan=allow_loan)

    # 更新股票的卖出操作
    backtest_obj, position_dict = stock_sell(backtest_obj, flow_dict, position_dict, updatetime, margin, allow_loan=allow_loan)

    return backtest_obj, position_dict


def on_market(updatetime: datetime, market_dict: Dict, \
              last_backtest_obj: PermanceObj, backtest_obj: PermanceObj, position_dict: Dict, price_latest_available: Dict,margin):
    """
    根据一条交易流水更新最新组合绩效和持仓情况
    updatetime:datetime,更新时间
    market_dict:dict，当前时间行情
    last_backtest_obj：PermanceObj，更新交易操作前的绩效对象
    backtest_obj：PermanceObj，最新组合绩效的对象
    position_dict:dict,组合最新的持仓记录

    返回：更新后的最新组合绩效对象
    """
    # print("行情更新----------------------")
    # print(updatetime,market_dict,last_backtest_obj,backtest_obj,position_dict)

    # 遍历仓位计算浮动盈亏（优化：缓存字符串分割结果）
    pnl_float = 0
    # 预先获取market_dict的keys集合，避免重复查找（性能优化）
    market_codes = set(market_dict.keys())
    
    for code_pos, pos in position_dict.items():

        #收益计提自定义的证券跳过行情驱动
        if pos.get("is_pnl",0)==1:
            continue

        # 优化：使用partition替代split，只分割一次（性能优化）
        code = code_pos.partition("@")[0]
        if code in market_codes:
            row = market_dict[code]
            price_latest_available[code] = row["close_price"] if isinstance(row, dict) else float(row)
        else:
            if code not in price_latest_available:
                # 使用成本价作为估值回退，避免持仓被忽略导致净值异常或后期净值不变
                fallback = pos.get("price", 0)
                if fallback is None or (isinstance(fallback, (int, float)) and fallback <= 0):
                    continue
                price_latest_available[code] = float(fallback)
            # print(f'missing code {code} at date {updatetime}, trying using latest available price ...')

        #更新持仓的市场价格
        pos["price_market"]=price_latest_available[code]
        position_dict[code_pos]=pos

        if pos["position"]>0:
            pnl_float += (price_latest_available[code] - pos["price"]) * pos["position"]*pos.get("direction",1)
        # if str(updatetime)[:10]=="2023-03-13":
        #     print(code,"浮动盈亏计算",price_latest_available[code],pos["price"],pos.get("direction",1),(price_latest_available[code] - pos["price"]) * pos["position"]*pos.get("direction",1))
        #     print(pnl_float)



    backtest_obj.pnl_float = pnl_float
    backtest_obj.pnl_all = backtest_obj.pnl_real_accum + backtest_obj.pnl_float - backtest_obj.fee_accum

    # 最新权益=期初权益+总盈亏
    # if str(updatetime)[:10] == "2023-03-13":
    #     print('最新权益',backtest_obj.asset_init,backtest_obj.pnl_all,backtest_obj.cash)

    backtest_obj.asset = backtest_obj.asset_init + backtest_obj.pnl_all
    if margin:
        backtest_obj.cash = backtest_obj.asset - backtest_obj.security
        backtest_obj.holdweight=backtest_obj.security/backtest_obj.asset
    else:
        backtest_obj.security = backtest_obj.asset - backtest_obj.cash
        backtest_obj.holdweight=backtest_obj.security/backtest_obj.asset

    backtest_obj.asset_max = max(backtest_obj.asset, backtest_obj.asset_max)

    # 计算收益-风险等关键绩效
    if last_backtest_obj.asset > 0:
        backtest_obj.npv *= backtest_obj.asset / last_backtest_obj.asset
    else:
        backtest_obj.npv *= backtest_obj.asset / backtest_obj.asset_max
    backtest_obj.yields = backtest_obj.npv - 1
    backtest_obj.drawdown_max = max(backtest_obj.drawdown_max, backtest_obj.asset_max - backtest_obj.asset)
    backtest_obj.drawdownrate_max = max(backtest_obj.drawdownrate_max, \
                                        (backtest_obj.asset_max - backtest_obj.asset) / backtest_obj.asset_max)

    #更新持仓（优化：使用生成器表达式和sum替代列表推导式）
    backtest_obj.position = sum(1 for pos in position_dict.values() if pos.get("position", 0) > 0)
    backtest_obj.position_detial = str({code: pos for code, pos in position_dict.items() if pos.get("position", 0) > 0})

    # 更新时间
    backtest_obj.updatetime = updatetime
    return backtest_obj


def run_portfolio(market_data: pd.DataFrame, flow_data: pd.DataFrame, position_data: pd.DataFrame=None, \
                  backtest_his: Dict = {},margin=False, allow_loan=True, weight_type='cash'):
    """
    运行投资组合
    market_data:DataFrame,行情数据文件，多层级索引三维数据，时间作为第一层级索引，代码作为第二层级索引，行情指标为列
    flow_data:DataFrame,交易流水数据文件
    position_data:DataFrame,最新持仓数据
    backtest_dict:Dict,组合回测的历史记录
    allow_loan: boolean, 是否允许借款买入，如填入False则在现金不足时不执行买入操作

    """

    # 支持批量flow_data（list/tuple/dict）：一次行情回放，多组合并行回测
    if isinstance(flow_data, (list, tuple, dict)):
        return run_portfolio_batch(
            market_data=market_data,
            flow_data_batch=flow_data,
            position_data_batch=position_data,
            backtest_his_batch=backtest_his,
            margin=margin,
            allow_loan=allow_loan,
            weight_type=weight_type
        )

    # 复制输入数据，避免修改原始对象（使用浅拷贝，因为后续会重新索引）
    market_data = market_data.copy(deep=False)
    flow_data = flow_data.copy(deep=False)
    if position_data is not None:
        position_data = position_data.copy(deep=False)

    # 数据预处理
    market_data, flow_data, position_dict = data_standard(market_data, flow_data, position_data)
    all_market_dates, market_dict_cache = _build_market_cache(market_data)
    return _run_portfolio_core(
        all_market_dates=all_market_dates,
        market_dict_cache=market_dict_cache,
        flow_data=flow_data,
        position_dict=position_dict,
        backtest_his=backtest_his,
        margin=margin,
        allow_loan=allow_loan,
        weight_type=weight_type,
        verbose=True,
        progress_title="回测"
    )


def run_portfolio_batch(
    market_data: pd.DataFrame,
    flow_data_batch: Union[Dict[Any, pd.DataFrame], list, tuple],
    position_data_batch: Union[Dict[Any, pd.DataFrame], list, tuple, pd.DataFrame, None] = None,
    backtest_his_batch: Union[Dict[Any, Dict], list, tuple, Dict, None] = None,
    margin=False,
    allow_loan=True,
    weight_type='cash'
):
    """
    批量回测：多份flow_data共享同一份market_data，只回放一次行情时间轴。
    返回:
        Dict[strategy_id, Dict[str, Any]]
        每个strategy_id下包含:
            backtest_result, position_result, flow_data_new, backtest_obj
    """
    market_data = market_data.copy(deep=False)
    market_data = _standardize_market_data(market_data)
    all_market_dates, market_dict_cache = _build_market_cache(market_data)

    if isinstance(flow_data_batch, dict):
        strategy_items = list(flow_data_batch.items())
    else:
        strategy_items = [(f"strategy_{i}", df) for i, df in enumerate(flow_data_batch)]

    def _get_item_by_key(container, key, idx, default=None):
        if container is None:
            return default
        if isinstance(container, dict):
            if key in container:
                return container[key]
            return default
        if isinstance(container, (list, tuple)):
            if idx < len(container):
                return container[idx]
            return default
        return container

    states = {}
    for idx, (strategy_id, flow_df_raw) in enumerate(strategy_items):
        flow_df = flow_df_raw.copy(deep=False)
        position_df = _get_item_by_key(position_data_batch, strategy_id, idx, default=None)
        if position_df is not None:
            position_df = position_df.copy(deep=False)
        backtest_his = _get_item_by_key(backtest_his_batch, strategy_id, idx, default={})
        if backtest_his is None:
            backtest_his = {}

        flow_df = _standardize_flow_data(flow_df)
        position_dict = _standardize_position_data(position_df)
        _check_market_flow_range(market_data, flow_df)

        backtest_obj = PermanceObj()
        backtest_obj.__dict__.update(backtest_his)
        flow_data_new_list = []
        backtest_result_list = []
        position_result_list = []

        backtest_obj, flow_df = _prepare_first_transfer_flow(backtest_obj, flow_df, flow_data_new_list)
        if flow_data_new_list:
            backtest_result_list.append({**backtest_obj.__dict__, "updatetime": backtest_obj.updatetime})

        states[strategy_id] = {
            "flow_data": flow_df,
            "flow_index_values": flow_df.index.values,
            "flow_cursor": 0,
            "position_dict": position_dict,
            "price_latest_available": {},
            "flow_data_new_list": flow_data_new_list,
            "backtest_result_list": backtest_result_list,
            "position_result_list": position_result_list,
            "backtest_obj": backtest_obj
        }

    total_days = len(all_market_dates)
    total_strategy = len(states)
    print(f"\n开始批量回测，共 {total_strategy} 个策略，{total_days} 个交易日（行情只回放一次）")
    print("=" * 80)
    progress_interval = max(1, min(total_days // 10, 50))

    for day_idx, datetime_market in enumerate(all_market_dates):
        if day_idx == 0 or (day_idx + 1) % progress_interval == 0 or day_idx == total_days - 1:
            progress = (day_idx + 1) / total_days * 100
            print(
                f"批量回测进度: [{day_idx + 1}/{total_days}] {progress:.1f}% | "
                f"当前日期: {datetime_market.strftime('%Y-%m-%d')}"
            )
        market_dict = market_dict_cache[datetime_market]

        for strategy_id, state in states.items():
            flow_data = state["flow_data"]
            flow_index_values = state["flow_index_values"]
            flow_cursor = state["flow_cursor"]
            next_cursor = np.searchsorted(flow_index_values, np.datetime64(datetime_market), side="right")
            flow_data_current = flow_data.iloc[flow_cursor:next_cursor]
            state["flow_cursor"] = next_cursor

            backtest_obj = state["backtest_obj"]
            position_dict = state["position_dict"]

            last_backtest_obj_asset = backtest_obj.asset
            last_backtest_obj_npv = backtest_obj.npv

            if len(flow_data_current) > 0:
                for batch, group in flow_data_current.groupby("batch"):
                    _ = batch
                    weight_asset = backtest_obj.asset
                    weight_cash = backtest_obj.cash
                    weight_position = position_dict
                    for datatetime_flow, row in group.iterrows():
                        if weight_type == 'cash':
                            row = weight_to_volume(weight_cash, weight_position, row)
                        if weight_type == 'asset':
                            row = weight_to_volume(weight_asset, weight_position, row)
                        row_dict = row.to_dict()
                        row_dict["updatetime"] = datatetime_flow
                        state["flow_data_new_list"].append(row_dict)
                        backtest_obj, position_dict = on_flow(
                            datatetime_flow, backtest_obj, row, position_dict, margin, allow_loan=allow_loan
                        )

            last_backtest_obj = PermanceObj()
            last_backtest_obj.asset = last_backtest_obj_asset
            last_backtest_obj.npv = last_backtest_obj_npv
            backtest_obj = on_market(
                datetime_market,
                market_dict,
                last_backtest_obj,
                backtest_obj,
                position_dict,
                state["price_latest_available"],
                margin
            )

            if len(flow_data_current) > 0:
                direction_values = flow_data_current["direction"].values
                if Direction.BUY.value in direction_values or Direction.SELL.value in direction_values:
                    position_df = pd.DataFrame.from_dict(position_dict, orient="index")
                    position_df["code"] = position_df.index
                    state["position_result_list"].append(position_df)

            state["backtest_result_list"].append({**backtest_obj.__dict__, "updatetime": datetime_market})
            state["backtest_obj"] = backtest_obj
            state["position_dict"] = position_dict

    print("=" * 80)
    print("批量回测完成！")
    print(f"回测期间: {all_market_dates[0].strftime('%Y-%m-%d')} 至 {all_market_dates[-1].strftime('%Y-%m-%d')}")
    print(f"总交易日数: {total_days}")
    print(f"策略数量: {total_strategy}")
    print("=" * 80)

    results = {}
    for strategy_id, state in states.items():
        backtest_result, position_result, flow_data_new = _finalize_portfolio_outputs(
            state["flow_data_new_list"], state["backtest_result_list"], state["position_result_list"]
        )
        results[strategy_id] = {
            "backtest_result": backtest_result,
            "position_result": position_result,
            "flow_data_new": flow_data_new,
            "backtest_obj": state["backtest_obj"]
        }
    return results
    


# %%
if __name__ == '__main__':

    # 交易流水数据处理:行情时间使用左标签，对应交易流水开始时间
    flow_data = pd.read_excel("backtest_data/FlowData.xlsx")

    # 行情数据读取测试
    market_data = pd.read_csv("backtest_data/MarketData.csv")

    # 持仓数据读取
    position_data = pd.read_excel("backtest_data/PositionData.xlsx")

    backtest_result, position_result,flow_data_new, backtest_obj = run_portfolio(market_data, flow_data, position_data)



# %%
