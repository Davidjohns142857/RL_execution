"""
复合奖励函数

组合多个奖励函数，支持自定义权重
"""

from typing import List, Tuple

from rl_trading_framework.core.base import BaseReward
from rl_trading_framework.core.types import State, Action


class CompositeReward(BaseReward):
    """
    复合奖励函数

    将多个奖励函数按权重组合

    输入：
        - reward_functions: 奖励函数列表，每个元素为(奖励函数, 权重)

    输出：
        - reward: 加权组合后的奖励

    示例：
        >>> is_reward = ImplementationShortfallReward()
        >>> pnl_reward = PnLBasedReward()
        >>> composite = CompositeReward([
        ...     (is_reward, 0.7),
        ...     (pnl_reward, 0.3),
        ... ])
    """

    def __init__(self, reward_functions: List[Tuple[BaseReward, float]]):
        """
        初始化复合奖励函数

        Args:
            reward_functions: [(奖励函数, 权重), ...]
        """
        self.reward_functions = reward_functions

        # 验证权重和
        total_weight = sum(weight for _, weight in reward_functions)
        if abs(total_weight - 1.0) > 1e-6:
            print(f"警告：权重和为 {total_weight}，不等于1.0")

    def calculate(
        self,
        state: State,
        action: Action,
        next_state: State,
        executed_quantity: float,
        execution_price: float,
    ) -> float:
        """
        计算复合奖励

        Args:
            state: 当前状态
            action: 执行的动作
            next_state: 下一状态
            executed_quantity: 实际执行数量
            execution_price: 执行价格

        Returns:
            加权组合的奖励
        """
        total_reward = 0.0

        for reward_fn, weight in self.reward_functions:
            reward = reward_fn.calculate(
                state,
                action,
                next_state,
                executed_quantity,
                execution_price,
            )
            total_reward += reward * weight

        return total_reward

    def reset(self):
        """重置所有子奖励函数"""
        for reward_fn, _ in self.reward_functions:
            if hasattr(reward_fn, 'reset'):
                reward_fn.reset()

    def add_reward(self, reward_fn: BaseReward, weight: float):
        """
        添加新的奖励函数

        Args:
            reward_fn: 奖励函数
            weight: 权重
        """
        self.reward_functions.append((reward_fn, weight))

    def get_individual_rewards(
        self,
        state: State,
        action: Action,
        next_state: State,
        executed_quantity: float,
        execution_price: float,
    ) -> dict:
        """
        获取各个子奖励的值（用于分析）

        Returns:
            {奖励函数名: 奖励值}
        """
        rewards = {}

        for reward_fn, weight in self.reward_functions:
            reward = reward_fn.calculate(
                state,
                action,
                next_state,
                executed_quantity,
                execution_price,
            )
            name = reward_fn.__class__.__name__
            rewards[name] = {
                'value': reward,
                'weighted_value': reward * weight,
                'weight': weight,
            }

        return rewards
