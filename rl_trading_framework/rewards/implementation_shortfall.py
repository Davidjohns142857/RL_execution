"""
Implementation Shortfall奖励函数

基于执行缺口（Implementation Shortfall）的奖励函数，
衡量实际执行价格与决策时刻价格之间的差异。
"""

import numpy as np
from typing import Optional

from rl_trading_framework.core.base import BaseReward
from rl_trading_framework.core.types import State, Action, OrderSide


class ImplementationShortfallReward(BaseReward):
    """
    Implementation Shortfall奖励函数

    奖励 = -(实际成交价格 - 参考价格) * 成交数量 / 参考价格

    输入：
        - state: 当前状态
        - action: 执行的动作
        - next_state: 下一状态
        - executed_quantity: 成交数量
        - execution_price: 成交价格

    输出：
        - reward: 奖励值（标量）

    计算公式：
        对于买单：reward = -(execution_price - arrival_price) / arrival_price * quantity
        对于卖单：reward = (execution_price - arrival_price) / arrival_price * quantity

    奖励组成：
        1. 价格偏差惩罚：执行价格偏离参考价格的程度
        2. 时间惩罚：未在规定时间内完成的惩罚
        3. 完成奖励：完成目标的额外奖励
    """

    def __init__(
        self,
        arrival_price: Optional[float] = None,
        time_penalty_weight: float = 0.001,
        completion_bonus: float = 1.0,
        risk_aversion: float = 0.1,
        normalize: bool = True,
    ):
        """
        初始化IS奖励函数

        Args:
            arrival_price: 参考价格（决策时刻的价格）
            time_penalty_weight: 时间惩罚权重
            completion_bonus: 完成奖励
            risk_aversion: 风险厌恶系数（惩罚价格波动）
            normalize: 是否标准化奖励
        """
        self.arrival_price = arrival_price
        self.time_penalty_weight = time_penalty_weight
        self.completion_bonus = completion_bonus
        self.risk_aversion = risk_aversion
        self.normalize = normalize

        # 记录初始价格
        self._initial_price = None

    def calculate(
        self,
        state: State,
        action: Action,
        next_state: State,
        executed_quantity: float,
        execution_price: float,
    ) -> float:
        """
        计算奖励

        Args:
            state: 当前状态
            action: 执行的动作
            next_state: 下一状态
            executed_quantity: 实际执行数量
            execution_price: 执行价格

        Returns:
            奖励值
        """
        # 设置初始价格（arrival price）
        if self._initial_price is None:
            self._initial_price = state.observation.market_data.close
            if self.arrival_price is None:
                self.arrival_price = self._initial_price

        current_price = next_state.observation.market_data.close

        # 1. 执行成本（Implementation Shortfall）
        if executed_quantity > 1e-8:
            # 计算相对执行成本
            price_diff = execution_price - self.arrival_price

            # 根据交易方向调整
            # 买单：execution_price越低越好
            # 卖单：execution_price越高越好
            # 这里假设是买单，如需要支持卖单需要传入side信息
            execution_cost = -abs(price_diff) / self.arrival_price * executed_quantity

            # 标准化
            if self.normalize:
                execution_cost = execution_cost / max(state.observation.target_quantity, 1.0)
        else:
            execution_cost = 0.0

        # 2. 时间惩罚（鼓励及时完成）
        time_penalty = -self.time_penalty_weight * (1.0 / max(next_state.observation.time_remaining, 1))

        # 3. 完成奖励
        completion_reward = 0.0
        if next_state.done:
            completion_rate = next_state.observation.executed_quantity / (
                next_state.observation.executed_quantity + next_state.observation.target_quantity
            )
            if completion_rate > 0.95:  # 95%以上完成
                completion_reward = self.completion_bonus
            else:
                # 未完成惩罚
                completion_reward = -self.completion_bonus * (1.0 - completion_rate)

        # 4. 市场冲击惩罚
        market_impact_penalty = -self.risk_aversion * next_state.observation.market_impact

        # 总奖励
        total_reward = (
            execution_cost +
            time_penalty +
            completion_reward +
            market_impact_penalty
        )

        return float(total_reward)

    def reset(self):
        """重置奖励函数状态"""
        self._initial_price = None
        if self.arrival_price is not None:
            # 如果设置了固定arrival_price，保持不变
            pass
