"""
简单执行引擎

实现基础的订单执行逻辑，包含市场冲击和滑点模型
"""

import numpy as np
from typing import Tuple

from rl_trading_framework.core.base import BaseExecutionEngine
from rl_trading_framework.core.types import (
    OrderInfo,
    MarketData,
    OrderStatus,
    OrderType,
    OrderSide,
)


class SimpleExecutionEngine(BaseExecutionEngine):
    """
    简单执行引擎

    输入：
        - order: 订单信息
        - market_data: 当前市场数据
        - liquidity: 流动性系数

    输出：
        - executed_quantity: 成交数量
        - execution_price: 成交价格
        - updated_order: 更新后的订单

    市场冲击模型：
        permanent_impact = k1 * (quantity / avg_volume)^gamma
        temporary_impact = k2 * (quantity / avg_volume)^delta

    滑点模型：
        slippage = base_slippage * sqrt(quantity / avg_volume) * volatility
    """

    def __init__(
        self,
        permanent_impact_coef: float = 0.1,
        temporary_impact_coef: float = 0.5,
        impact_exponent: float = 0.6,
        base_slippage: float = 0.0001,
        fill_probability: float = 1.0,
    ):
        """
        初始化执行引擎

        Args:
            permanent_impact_coef: 永久冲击系数 (k1)
            temporary_impact_coef: 临时冲击系数 (k2)
            impact_exponent: 冲击指数 (gamma, delta)
            base_slippage: 基础滑点
            fill_probability: 成交概率（限价单使用）
        """
        self.permanent_impact_coef = permanent_impact_coef
        self.temporary_impact_coef = temporary_impact_coef
        self.impact_exponent = impact_exponent
        self.base_slippage = base_slippage
        self.fill_probability = fill_probability

    def execute_order(
        self,
        order: OrderInfo,
        market_data: MarketData,
        liquidity: float = 1.0,
    ) -> Tuple[float, float, OrderInfo]:
        """
        执行订单

        Args:
            order: 订单信息
            market_data: 当前市场数据
            liquidity: 流动性系数（0-1，越高流动性越好）

        Returns:
            executed_quantity: 实际成交数量
            execution_price: 平均成交价格
            updated_order: 更新后的订单信息
        """
        if order.is_finished:
            return 0.0, 0.0, order

        # 计算参考价格
        reference_price = self._get_reference_price(market_data, order.side)

        # 根据订单类型执行
        if order.order_type == OrderType.MARKET:
            executed_qty, exec_price = self._execute_market_order(
                order, market_data, reference_price, liquidity
            )
        elif order.order_type == OrderType.LIMIT:
            executed_qty, exec_price = self._execute_limit_order(
                order, market_data, reference_price, liquidity
            )
        else:
            # 其他类型（TWAP, VWAP等）简化为市价单处理
            executed_qty, exec_price = self._execute_market_order(
                order, market_data, reference_price, liquidity
            )

        # 更新订单状态
        order.filled_quantity += executed_qty
        if executed_qty > 0:
            # 更新平均成交价
            total_value = order.avg_fill_price * (order.filled_quantity - executed_qty) + exec_price * executed_qty
            order.avg_fill_price = total_value / order.filled_quantity

        # 更新订单状态
        if order.filled_quantity >= order.quantity * 0.999:  # 99.9%视为完全成交
            order.status = OrderStatus.FILLED
        elif order.filled_quantity > 0:
            order.status = OrderStatus.PARTIALLY_FILLED
        else:
            order.status = OrderStatus.PENDING

        return executed_qty, exec_price, order

    def _execute_market_order(
        self,
        order: OrderInfo,
        market_data: MarketData,
        reference_price: float,
        liquidity: float,
    ) -> Tuple[float, float]:
        """执行市价单"""
        quantity = order.remaining_quantity

        # 计算市场冲击
        market_impact = self.calculate_market_impact(
            quantity,
            market_data,
            order.side.value,
        )

        # 计算滑点
        slippage = self.calculate_slippage(
            order.order_type.value,
            quantity,
            market_data,
        )

        # 综合价格影响（考虑流动性）
        total_impact = (market_impact + slippage) / liquidity

        # 计算成交价格
        if order.side == OrderSide.BUY:
            execution_price = reference_price * (1 + total_impact)
        else:
            execution_price = reference_price * (1 - total_impact)

        # 市价单全部成交
        executed_quantity = quantity

        return executed_quantity, execution_price

    def _execute_limit_order(
        self,
        order: OrderInfo,
        market_data: MarketData,
        reference_price: float,
        liquidity: float,
    ) -> Tuple[float, float]:
        """执行限价单"""
        if order.price is None:
            raise ValueError("限价单必须指定价格")

        quantity = order.remaining_quantity

        # 检查限价单是否能够成交
        can_fill = False
        if order.side == OrderSide.BUY:
            # 买单：限价 >= 卖价时可能成交
            can_fill = order.price >= reference_price
        else:
            # 卖单：限价 <= 买价时可能成交
            can_fill = order.price <= reference_price

        if not can_fill:
            return 0.0, 0.0  # 无法成交

        # 考虑成交概率
        if np.random.random() > self.fill_probability:
            return 0.0, 0.0

        # 部分成交（模拟限价单可能不能立即全部成交）
        fill_ratio = np.random.uniform(0.5, 1.0) * liquidity
        executed_quantity = quantity * fill_ratio

        # 成交价格为限价或更好的价格
        if order.side == OrderSide.BUY:
            # 买单可能以低于限价的价格成交
            execution_price = min(order.price, reference_price)
        else:
            # 卖单可能以高于限价的价格成交
            execution_price = max(order.price, reference_price)

        return executed_quantity, execution_price

    def calculate_market_impact(
        self,
        quantity: float,
        market_data: MarketData,
        side: str,
    ) -> float:
        """
        计算市场冲击

        使用平方根模型：impact = k * (quantity / volume)^gamma

        Args:
            quantity: 交易数量
            market_data: 市场数据
            side: 交易方向

        Returns:
            市场冲击（价格百分比）
        """
        if market_data.volume < 1e-8:
            return 0.01  # 默认1%冲击

        # 相对交易量
        relative_size = quantity / market_data.volume

        # 永久冲击
        permanent_impact = self.permanent_impact_coef * np.power(relative_size, self.impact_exponent)

        # 临时冲击
        temporary_impact = self.temporary_impact_coef * np.power(relative_size, self.impact_exponent)

        # 总冲击
        total_impact = permanent_impact + temporary_impact

        return float(total_impact)

    def calculate_slippage(
        self,
        order_type: str,
        quantity: float,
        market_data: MarketData,
    ) -> float:
        """
        计算滑点

        Args:
            order_type: 订单类型
            quantity: 交易数量
            market_data: 市场数据

        Returns:
            滑点（价格百分比）
        """
        if market_data.volume < 1e-8:
            return self.base_slippage

        # 计算价格波动率
        if market_data.high > market_data.low:
            volatility = (market_data.high - market_data.low) / market_data.close
        else:
            volatility = 0.001  # 默认波动率

        # 相对交易量
        relative_size = quantity / market_data.volume

        # 滑点与数量和波动率相关
        slippage = self.base_slippage * np.sqrt(relative_size) * (1 + volatility * 10)

        # 限价单滑点更小
        if order_type == "limit":
            slippage *= 0.5

        return float(slippage)

    def _get_reference_price(self, market_data: MarketData, side: OrderSide) -> float:
        """获取参考价格"""
        if side == OrderSide.BUY:
            # 买单使用卖价（ask）
            return market_data.ask_price if market_data.ask_price else market_data.close
        else:
            # 卖单使用买价（bid）
            return market_data.bid_price if market_data.bid_price else market_data.close
