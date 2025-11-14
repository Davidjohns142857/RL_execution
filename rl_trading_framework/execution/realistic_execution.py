"""
真实执行引擎

实现更真实的市场微观结构，包括订单簿、部分成交、价格影响等
"""

import numpy as np
from typing import Tuple, Dict, List
from collections import deque

from rl_trading_framework.execution.simple_execution import SimpleExecutionEngine
from rl_trading_framework.core.types import (
    OrderInfo,
    MarketData,
    OrderStatus,
    OrderType,
    OrderSide,
)


class RealisticExecutionEngine(SimpleExecutionEngine):
    """
    真实执行引擎

    在SimpleExecutionEngine基础上增加：
    - 订单簿深度模拟
    - 市场订单的部分成交
    - 价格影响的时间衰减
    - 流动性恢复机制

    额外输入：
        - order_book_depth: 订单簿深度（价格层数）
        - liquidity_recovery_rate: 流动性恢复速率

    市场微观结构特性：
        1. 订单簿有限深度
        2. 大单需要多层价格成交
        3. 市场冲击会随时间衰减
    """

    def __init__(
        self,
        permanent_impact_coef: float = 0.1,
        temporary_impact_coef: float = 0.5,
        impact_exponent: float = 0.6,
        base_slippage: float = 0.0001,
        fill_probability: float = 0.8,
        order_book_depth: int = 5,
        liquidity_recovery_rate: float = 0.1,
        adverse_selection_prob: float = 0.05,
    ):
        """
        初始化真实执行引擎

        Args:
            permanent_impact_coef: 永久冲击系数
            temporary_impact_coef: 临时冲击系数
            impact_exponent: 冲击指数
            base_slippage: 基础滑点
            fill_probability: 限价单成交概率
            order_book_depth: 订单簿深度
            liquidity_recovery_rate: 流动性恢复速率
            adverse_selection_prob: 逆向选择概率（限价单）
        """
        super().__init__(
            permanent_impact_coef,
            temporary_impact_coef,
            impact_exponent,
            base_slippage,
            fill_probability,
        )
        self.order_book_depth = order_book_depth
        self.liquidity_recovery_rate = liquidity_recovery_rate
        self.adverse_selection_prob = adverse_selection_prob

        # 市场状态
        self.price_impact_history: deque = deque(maxlen=100)
        self.recent_trades: List[Dict] = []

    def execute_order(
        self,
        order: OrderInfo,
        market_data: MarketData,
        liquidity: float = 1.0,
    ) -> Tuple[float, float, OrderInfo]:
        """
        执行订单（更真实的模拟）

        考虑：
        - 订单簿深度限制
        - 部分成交
        - 价格影响累积
        """
        if order.is_finished:
            return 0.0, 0.0, order

        # 生成模拟订单簿
        order_book = self._generate_order_book(market_data, liquidity)

        # 根据订单类型执行
        if order.order_type == OrderType.MARKET:
            executed_qty, exec_price = self._execute_market_order_realistic(
                order, market_data, order_book, liquidity
            )
        elif order.order_type == OrderType.LIMIT:
            executed_qty, exec_price = self._execute_limit_order_realistic(
                order, market_data, order_book, liquidity
            )
        else:
            executed_qty, exec_price = self._execute_market_order_realistic(
                order, market_data, order_book, liquidity
            )

        # 更新订单状态
        if executed_qty > 0:
            order.filled_quantity += executed_qty
            total_value = order.avg_fill_price * (order.filled_quantity - executed_qty) + exec_price * executed_qty
            order.avg_fill_price = total_value / order.filled_quantity

            # 记录交易
            self.recent_trades.append({
                'quantity': executed_qty,
                'price': exec_price,
                'side': order.side,
                'impact': abs(exec_price - market_data.close) / market_data.close,
            })

        # 更新订单状态标志
        if order.filled_quantity >= order.quantity * 0.999:
            order.status = OrderStatus.FILLED
        elif order.filled_quantity > 0:
            order.status = OrderStatus.PARTIALLY_FILLED

        return executed_qty, exec_price, order

    def _generate_order_book(
        self,
        market_data: MarketData,
        liquidity: float,
    ) -> Dict[str, List[Tuple[float, float]]]:
        """
        生成模拟订单簿

        Returns:
            {'bids': [(price, quantity), ...], 'asks': [(price, quantity), ...]}
        """
        mid_price = market_data.close
        spread = market_data.close * 0.001  # 0.1% spread

        # 估计每层的量（基于总成交量）
        avg_layer_volume = market_data.volume / (self.order_book_depth * 2) * liquidity

        bids = []
        asks = []

        for i in range(self.order_book_depth):
            # 价格随着层级递减/递增，量也递减
            level_factor = 1.0 - i * 0.1  # 每层递减10%

            # 买盘
            bid_price = mid_price - spread / 2 - i * spread * 0.5
            bid_volume = avg_layer_volume * level_factor * np.random.uniform(0.8, 1.2)
            bids.append((bid_price, bid_volume))

            # 卖盘
            ask_price = mid_price + spread / 2 + i * spread * 0.5
            ask_volume = avg_layer_volume * level_factor * np.random.uniform(0.8, 1.2)
            asks.append((ask_price, ask_volume))

        return {'bids': bids, 'asks': asks}

    def _execute_market_order_realistic(
        self,
        order: OrderInfo,
        market_data: MarketData,
        order_book: Dict,
        liquidity: float,
    ) -> Tuple[float, float]:
        """
        真实的市价单执行（考虑订单簿深度）
        """
        remaining_qty = order.remaining_quantity
        total_cost = 0.0
        total_qty = 0.0

        # 选择对应的订单簿一侧
        book_side = order_book['asks'] if order.side == OrderSide.BUY else order_book['bids']

        # 逐层成交
        for price, available_qty in book_side:
            if remaining_qty <= 0:
                break

            # 这一层能成交的量
            fill_qty = min(remaining_qty, available_qty)

            # 添加额外的市场冲击
            impact = self.calculate_market_impact(fill_qty, market_data, order.side.value)
            actual_price = price * (1 + impact if order.side == OrderSide.BUY else 1 - impact)

            total_cost += fill_qty * actual_price
            total_qty += fill_qty
            remaining_qty -= fill_qty

        if total_qty > 0:
            avg_price = total_cost / total_qty
        else:
            # 如果订单簿流动性不足，使用市场价加大滑点
            avg_price = market_data.close * (1.01 if order.side == OrderSide.BUY else 0.99)
            total_qty = order.remaining_quantity * 0.5  # 只能成交一半

        return total_qty, avg_price

    def _execute_limit_order_realistic(
        self,
        order: OrderInfo,
        market_data: MarketData,
        order_book: Dict,
        liquidity: float,
    ) -> Tuple[float, float]:
        """
        真实的限价单执行
        """
        if order.price is None:
            raise ValueError("限价单必须指定价格")

        # 检查是否能立即成交（aggressive limit order）
        book_side = order_book['asks'] if order.side == OrderSide.BUY else order_book['bids']
        best_price, best_qty = book_side[0]

        can_fill_immediately = False
        if order.side == OrderSide.BUY:
            can_fill_immediately = order.price >= best_price
        else:
            can_fill_immediately = order.price <= best_price

        if not can_fill_immediately:
            # 限价单不能立即成交，模拟等待成交
            # 考虑被动成交的概率
            if np.random.random() < self.fill_probability * liquidity:
                # 部分成交
                fill_ratio = np.random.uniform(0.3, 0.8)
                executed_qty = order.remaining_quantity * fill_ratio
                # 成交价格为限价
                return executed_qty, order.price
            else:
                return 0.0, 0.0

        # 可以立即成交的情况
        remaining_qty = order.remaining_quantity
        total_cost = 0.0
        total_qty = 0.0

        for price, available_qty in book_side:
            # 检查价格是否在限价范围内
            if order.side == OrderSide.BUY and price > order.price:
                break
            if order.side == OrderSide.SELL and price < order.price:
                break

            fill_qty = min(remaining_qty, available_qty)
            total_cost += fill_qty * price
            total_qty += fill_qty
            remaining_qty -= fill_qty

            if remaining_qty <= 0:
                break

        # 考虑逆向选择风险
        if np.random.random() < self.adverse_selection_prob:
            # 模拟信息不对称导致的不利价格
            penalty = 0.001  # 0.1%的不利价格
            if order.side == OrderSide.BUY:
                total_cost *= (1 + penalty)
            else:
                total_cost *= (1 - penalty)

        if total_qty > 0:
            avg_price = total_cost / total_qty
        else:
            return 0.0, 0.0

        return total_qty, avg_price

    def get_cumulative_impact(self) -> float:
        """
        获取累积市场冲击

        Returns:
            累积冲击值
        """
        if len(self.recent_trades) == 0:
            return 0.0

        # 计算最近交易的加权冲击（考虑时间衰减）
        total_impact = 0.0
        total_weight = 0.0

        for i, trade in enumerate(reversed(self.recent_trades[-10:])):
            # 时间衰减权重
            decay_factor = np.exp(-i * self.liquidity_recovery_rate)
            weight = trade['quantity'] * decay_factor
            total_impact += trade['impact'] * weight
            total_weight += weight

        return total_impact / total_weight if total_weight > 0 else 0.0
