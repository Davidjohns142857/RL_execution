"""
基础抽象类定义

本模块定义了所有核心组件的抽象基类，确保模块的可替换性和一致性。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from rl_trading_framework.core.types import (
    State,
    Action,
    Observation,
    OrderInfo,
    MarketData,
    TransitionBatch,
    EpisodeMetrics,
)


class BaseEnvironment(ABC):
    """
    交易环境基类

    定义了强化学习环境的标准接口，兼容OpenAI Gym风格。

    主要职责：
    1. 模拟交易市场环境
    2. 处理智能体的动作
    3. 计算奖励
    4. 返回新的观察状态
    """

    @abstractmethod
    def reset(self, **kwargs) -> Observation:
        """
        重置环境到初始状态

        Args:
            **kwargs: 重置参数（如随机种子、起始时间等）

        Returns:
            初始观察状态

        Example:
            >>> env = TradingEnvironment()
            >>> obs = env.reset(seed=42)
        """
        pass

    @abstractmethod
    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        执行一步动作

        Args:
            action: 智能体的动作

        Returns:
            observation: 新的观察状态
            reward: 奖励值
            done: 是否结束
            info: 额外信息字典

        Example:
            >>> action = Action(trade_ratio=0.1)
            >>> obs, reward, done, info = env.step(action)
        """
        pass

    @abstractmethod
    def get_observation(self) -> Observation:
        """
        获取当前观察状态

        Returns:
            当前观察
        """
        pass

    @property
    @abstractmethod
    def observation_space_shape(self) -> Tuple[int, ...]:
        """
        观察空间形状

        Returns:
            观察空间的shape（例如 (feature_dim,) 或 (height, width, channels)）
        """
        pass

    @property
    @abstractmethod
    def action_space_shape(self) -> Tuple[int, ...]:
        """
        动作空间形状

        Returns:
            动作空间的shape
        """
        pass


class BaseAgent(ABC):
    """
    智能体基类

    定义了RL智能体的标准接口。

    主要职责：
    1. 根据观察选择动作
    2. 学习和更新策略
    3. 管理经验回放
    """

    @abstractmethod
    def select_action(self, observation: Observation, training: bool = True) -> Action:
        """
        根据观察选择动作

        Args:
            observation: 当前观察
            training: 是否处于训练模式（训练模式可能包含探索）

        Returns:
            选择的动作

        Example:
            >>> agent = DQNAgent(...)
            >>> action = agent.select_action(obs, training=True)
        """
        pass

    @abstractmethod
    def update(self, batch: TransitionBatch) -> Dict[str, float]:
        """
        使用经验批次更新策略

        Args:
            batch: 经验回放批次

        Returns:
            训练指标字典（如loss、q_value等）

        Example:
            >>> metrics = agent.update(batch)
            >>> print(f"Loss: {metrics['loss']}")
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        保存模型

        Args:
            path: 保存路径

        Example:
            >>> agent.save("models/agent_epoch_100.pth")
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        加载模型

        Args:
            path: 模型路径

        Example:
            >>> agent.load("models/best_agent.pth")
        """
        pass


class BasePolicy(ABC):
    """
    策略网络基类

    定义了策略网络的标准接口。

    主要职责：
    1. 将观察映射到动作或动作分布
    2. 支持不同的网络架构（MLP、CNN、RNN等）
    """

    @abstractmethod
    def forward(self, observation: np.ndarray) -> np.ndarray:
        """
        前向传播

        Args:
            observation: 观察数组 shape: (batch_size, obs_dim) 或 (obs_dim,)

        Returns:
            策略输出（动作、Q值或动作分布参数）

        Example:
            >>> policy = MLPPolicy(obs_dim=50, action_dim=4)
            >>> output = policy.forward(obs_array)
        """
        pass

    @abstractmethod
    def get_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        获取动作

        Args:
            observation: 观察数组
            deterministic: 是否使用确定性策略

        Returns:
            动作数组

        Example:
            >>> action = policy.get_action(obs, deterministic=True)
        """
        pass

    @abstractmethod
    def update_parameters(self, loss: float) -> None:
        """
        更新网络参数

        Args:
            loss: 损失值

        Example:
            >>> policy.update_parameters(td_loss)
        """
        pass


class BaseReward(ABC):
    """
    奖励函数基类

    定义了奖励计算的标准接口。

    主要职责：
    1. 根据状态、动作和结果计算奖励
    2. 支持不同的奖励函数设计
    """

    @abstractmethod
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

        Example:
            >>> reward_fn = ImplementationShortfallReward()
            >>> reward = reward_fn.calculate(state, action, next_state, qty, price)
        """
        pass


class BaseDataLoader(ABC):
    """
    数据加载器基类

    定义了数据加载的标准接口。

    主要职责：
    1. 加载历史市场数据
    2. 数据预处理和标准化
    3. 支持不同的数据源
    """

    @abstractmethod
    def load_data(
        self,
        symbol: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        **kwargs
    ) -> List[MarketData]:
        """
        加载市场数据

        Args:
            symbol: 交易标的代码
            start_time: 开始时间（Unix时间戳）
            end_time: 结束时间（Unix时间戳）
            **kwargs: 其他参数

        Returns:
            市场数据列表

        Example:
            >>> loader = CSVDataLoader("data/")
            >>> data = loader.load_data("AAPL", start_time=1609459200)
        """
        pass

    @abstractmethod
    def preprocess(self, data: List[MarketData]) -> List[MarketData]:
        """
        预处理数据

        Args:
            data: 原始市场数据

        Returns:
            预处理后的数据

        Example:
            >>> processed_data = loader.preprocess(raw_data)
        """
        pass


class BaseExecutionEngine(ABC):
    """
    执行引擎基类

    定义了订单执行的标准接口。

    主要职责：
    1. 模拟订单执行
    2. 计算市场冲击和滑点
    3. 管理订单状态
    """

    @abstractmethod
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
            liquidity: 流动性系数（0-1之间）

        Returns:
            executed_quantity: 实际成交数量
            execution_price: 成交价格
            updated_order: 更新后的订单信息

        Example:
            >>> engine = SimpleExecutionEngine()
            >>> qty, price, order = engine.execute_order(order, market_data)
        """
        pass

    @abstractmethod
    def calculate_market_impact(
        self,
        quantity: float,
        market_data: MarketData,
        side: str,
    ) -> float:
        """
        计算市场冲击

        Args:
            quantity: 交易数量
            market_data: 市场数据
            side: 交易方向（"buy" 或 "sell"）

        Returns:
            市场冲击（价格百分比变化）

        Example:
            >>> impact = engine.calculate_market_impact(1000, market_data, "buy")
        """
        pass

    @abstractmethod
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

        Example:
            >>> slippage = engine.calculate_slippage("market", 1000, market_data)
        """
        pass


class BaseTrainer(ABC):
    """
    训练器基类

    定义了训练流程的标准接口。

    主要职责：
    1. 管理训练循环
    2. 收集和存储经验
    3. 记录训练指标
    """

    @abstractmethod
    def train(
        self,
        num_episodes: int,
        **kwargs
    ) -> Dict[str, List[float]]:
        """
        训练智能体

        Args:
            num_episodes: 训练episode数量
            **kwargs: 其他训练参数

        Returns:
            训练历史字典（包含各种指标的列表）

        Example:
            >>> trainer = RLTrainer(env, agent)
            >>> history = trainer.train(num_episodes=1000)
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        num_episodes: int,
        **kwargs
    ) -> EpisodeMetrics:
        """
        评估智能体

        Args:
            num_episodes: 评估episode数量
            **kwargs: 其他评估参数

        Returns:
            评估指标

        Example:
            >>> metrics = trainer.evaluate(num_episodes=100)
        """
        pass


class BaseEvaluator(ABC):
    """
    评估器基类

    定义了性能评估的标准接口。

    主要职责：
    1. 评估交易策略性能
    2. 计算各种评估指标
    3. 生成评估报告
    """

    @abstractmethod
    def evaluate_episode(
        self,
        agent: BaseAgent,
        env: BaseEnvironment,
        **kwargs
    ) -> EpisodeMetrics:
        """
        评估单个episode

        Args:
            agent: 智能体
            env: 环境
            **kwargs: 其他参数

        Returns:
            episode评估指标

        Example:
            >>> evaluator = TradingEvaluator()
            >>> metrics = evaluator.evaluate_episode(agent, env)
        """
        pass

    @abstractmethod
    def generate_report(
        self,
        metrics_list: List[EpisodeMetrics],
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        生成评估报告

        Args:
            metrics_list: 多个episode的指标列表
            output_path: 报告输出路径（可选）

        Returns:
            汇总统计字典

        Example:
            >>> report = evaluator.generate_report(metrics_list, "reports/eval.json")
        """
        pass
