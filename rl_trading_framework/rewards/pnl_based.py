"""
基于PnL的奖励函数

直接使用盈亏（Profit and Loss）作为奖励
"""

import numpy as np

from rl_trading_framework.core.base import BaseReward
from rl_trading_framework.core.types import State, Action


class PnLBasedReward(BaseReward):
    """
    基于PnL的奖励函数

    奖励 = 当前step的PnL变化

    输入：
        - state: 当前状态
        - action: 执行的动作
        - next_state: 下一状态
        - executed_quantity: 成交数量
        - execution_price: 成交价格

    输出：
        - reward: PnL变化

    适用场景：
        - 做市策略
        - 套利策略
        - 需要关注最终PnL的场景
    """

    def __init__(
        self,
        inventory_penalty: float = 0.01,
        transaction_cost_bps: float = 1.0,
        normalize: bool = True,
    ):
        """
        初始化PnL奖励函数

        Args:
            inventory_penalty: 持仓惩罚系数（鼓励平仓）
            transaction_cost_bps: 交易成本（basis points, 1bp = 0.01%）
            normalize: 是否标准化奖励
        """
        self.inventory_penalty = inventory_penalty
        self.transaction_cost_bps = transaction_cost_bps
        self.normalize = normalize

        # 上一步的总价值
        self._prev_portfolio_value = None

    def calculate(
        self,
        state: State,
        action: Action,
        next_state: State,
        executed_quantity: float,
        execution_price: float,
    ) -> float:
        """
        计算PnL奖励

        Args:
            state: 当前状态
            action: 执行的动作
            next_state: 下一状态
            executed_quantity: 实际执行数量
            execution_price: 执行价格

        Returns:
            PnL变化
        """
        # 计算当前投资组合价值
        current_market_price = next_state.observation.market_data.close
        current_portfolio_value = (
            next_state.observation.cash +
            next_state.observation.position * current_market_price
        )

        # 初始化
        if self._prev_portfolio_value is None:
            prev_market_price = state.observation.market_data.close
            self._prev_portfolio_value = (
                state.observation.cash +
                state.observation.position * prev_market_price
            )

        # PnL变化
        pnl_change = current_portfolio_value - self._prev_portfolio_value

        # 交易成本
        transaction_cost = 0.0
        if executed_quantity > 1e-8:
            transaction_cost = (
                executed_quantity * execution_price * self.transaction_cost_bps / 10000.0
            )

        # 持仓惩罚（鼓励完成目标）
        inventory_penalty = 0.0
        if next_state.observation.target_quantity > 1e-8:
            # 惩罚未完成的目标
            remaining_ratio = next_state.observation.target_quantity / (
                next_state.observation.executed_quantity + next_state.observation.target_quantity
            )
            inventory_penalty = -self.inventory_penalty * remaining_ratio

        # 总奖励
        reward = pnl_change - transaction_cost + inventory_penalty

        # 标准化
        if self.normalize and self._prev_portfolio_value > 0:
            reward = reward / self._prev_portfolio_value * 100  # 转换为百分比

        # 更新上一步价值
        self._prev_portfolio_value = current_portfolio_value

        return float(reward)

    def reset(self):
        """重置状态"""
        self._prev_portfolio_value = None
