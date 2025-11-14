"""
自定义组件示例

演示如何创建自定义的环境、奖励函数、策略等组件
"""

import numpy as np
from typing import Tuple, Dict, Any

from rl_trading_framework.core.base import BaseReward
from rl_trading_framework.core.types import State, Action


class CustomReward(BaseReward):
    """
    自定义奖励函数示例

    结合多个因素：
    - 执行价格
    - 市场冲击
    - 时间效率
    - 风险控制
    """

    def __init__(
        self,
        price_weight: float = 0.5,
        impact_weight: float = 0.3,
        time_weight: float = 0.1,
        risk_weight: float = 0.1,
    ):
        """
        初始化自定义奖励函数

        Args:
            price_weight: 价格因素权重
            impact_weight: 市场冲击权重
            time_weight: 时间效率权重
            risk_weight: 风险控制权重
        """
        self.price_weight = price_weight
        self.impact_weight = impact_weight
        self.time_weight = time_weight
        self.risk_weight = risk_weight

        self._arrival_price = None
        self._price_history = []

    def calculate(
        self,
        state: State,
        action: Action,
        next_state: State,
        executed_quantity: float,
        execution_price: float,
    ) -> float:
        """
        计算自定义奖励

        Args:
            state: 当前状态
            action: 动作
            next_state: 下一状态
            executed_quantity: 执行数量
            execution_price: 执行价格

        Returns:
            奖励值
        """
        # 记录初始价格
        if self._arrival_price is None:
            self._arrival_price = state.observation.market_data.close

        current_price = next_state.observation.market_data.close
        self._price_history.append(current_price)

        reward = 0.0

        # 1. 价格因素：相对参考价格的偏差
        if executed_quantity > 1e-8:
            price_deviation = -(execution_price - self._arrival_price) / self._arrival_price
            reward += self.price_weight * price_deviation * executed_quantity

        # 2. 市场冲击惩罚
        market_impact = next_state.observation.market_impact
        reward -= self.impact_weight * market_impact

        # 3. 时间效率：鼓励按时完成
        if next_state.observation.time_remaining > 0:
            time_efficiency = next_state.observation.executed_quantity / (
                next_state.observation.executed_quantity + next_state.observation.target_quantity
            )
            reward += self.time_weight * time_efficiency

        # 4. 风险控制：惩罚价格波动
        if len(self._price_history) > 2:
            price_volatility = np.std(self._price_history[-10:])
            reward -= self.risk_weight * price_volatility / self._arrival_price

        return float(reward)

    def reset(self):
        """重置状态"""
        self._arrival_price = None
        self._price_history = []


def demonstrate_custom_reward():
    """演示自定义奖励函数的使用"""
    print("="*60)
    print("自定义组件示例")
    print("="*60)

    # 创建自定义奖励函数
    custom_reward = CustomReward(
        price_weight=0.5,
        impact_weight=0.3,
        time_weight=0.1,
        risk_weight=0.1,
    )

    print("\n自定义奖励函数已创建:")
    print(f"  价格权重: {custom_reward.price_weight}")
    print(f"  市场冲击权重: {custom_reward.impact_weight}")
    print(f"  时间效率权重: {custom_reward.time_weight}")
    print(f"  风险控制权重: {custom_reward.risk_weight}")

    print("\n使用方法:")
    print("  1. 创建环境时，将custom_reward作为reward_function参数传入")
    print("  2. 环境会自动调用calculate()方法计算奖励")
    print("  3. 可以根据需要调整各个权重")

    print("\n示例代码:")
    print("""
    from rl_trading_framework.environments.trading_env import TradingEnvironment

    env = TradingEnvironment(
        data_loader=data_loader,
        execution_engine=execution_engine,
        reward_function=custom_reward,  # 使用自定义奖励函数
        target_quantity=10000,
        max_steps=50,
    )
    """)


if __name__ == "__main__":
    demonstrate_custom_reward()
