"""
RL Trading Framework
====================

一个模块化、可扩展的强化学习交易执行框架

主要模块：
- core: 核心基础类和接口
- environments: 交易环境模拟
- agents: RL智能体实现
- policies: 策略网络（DQN, PPO, A3C等）
- data: 数据加载和预处理
- execution: 订单执行和滑点模拟
- rewards: 奖励函数
- trainers: 训练流程管理
- evaluators: 性能评估和回测
"""

__version__ = "0.1.0"
__author__ = "RL Trading Framework Team"

from rl_trading_framework.core.base import (
    BaseEnvironment,
    BaseAgent,
    BasePolicy,
    BaseReward,
)

__all__ = [
    "BaseEnvironment",
    "BaseAgent",
    "BasePolicy",
    "BaseReward",
]
