"""
交易执行环境实现

本模块实现了一个完整的交易执行环境，模拟在限定时间内执行大额订单的场景。
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from copy import deepcopy

from rl_trading_framework.core.base import BaseEnvironment, BaseReward, BaseDataLoader, BaseExecutionEngine
from rl_trading_framework.core.types import (
    State,
    Action,
    Observation,
    MarketData,
    OrderInfo,
    OrderSide,
    OrderType,
    OrderStatus,
)


class TradingEnvironment(BaseEnvironment):
    """
    交易执行环境

    模拟在有限时间内执行大额订单的场景，智能体需要在市场冲击和时间风险之间平衡。

    输入：
        - data_loader: 数据加载器
        - execution_engine: 执行引擎
        - reward_function: 奖励函数
        - target_quantity: 目标交易数量（总量）
        - max_steps: 最大执行步数
        - initial_cash: 初始现金
        - side: 交易方向（"buy" 或 "sell"）

    输出：
        - observation: 观察状态（Observation对象）
        - reward: 奖励值（float）
        - done: 是否结束（bool）
        - info: 额外信息（dict）

    关键数据流：
        1. reset() -> 初始化环境 -> 返回初始observation
        2. step(action) -> 执行动作 -> 更新状态 -> 计算奖励 -> 返回(obs, reward, done, info)
    """

    def __init__(
        self,
        data_loader: BaseDataLoader,
        execution_engine: BaseExecutionEngine,
        reward_function: BaseReward,
        target_quantity: float,
        max_steps: int = 100,
        initial_cash: float = 1000000.0,
        side: str = "buy",
        symbol: str = "ASSET",
        window_size: int = 20,
    ):
        """
        初始化交易环境

        Args:
            data_loader: 数据加载器
            execution_engine: 订单执行引擎
            reward_function: 奖励函数
            target_quantity: 目标交易总量
            max_steps: 最大执行步数
            initial_cash: 初始现金
            side: 交易方向（"buy" 或 "sell"）
            symbol: 交易标的
            window_size: 历史数据窗口大小
        """
        self.data_loader = data_loader
        self.execution_engine = execution_engine
        self.reward_function = reward_function
        self.target_quantity = target_quantity
        self.max_steps = max_steps
        self.initial_cash = initial_cash
        self.side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        self.symbol = symbol
        self.window_size = window_size

        # 环境状态
        self.current_step = 0
        self.market_data_list: List[MarketData] = []
        self.current_data_idx = 0
        self.state: Optional[State] = None

        # 执行状态
        self.executed_quantity = 0.0
        self.total_cost = 0.0  # 总成本
        self.avg_execution_price = 0.0
        self.position = 0.0
        self.cash = initial_cash

        # 历史记录
        self.price_history: List[float] = []
        self.volume_history: List[float] = []
        self.action_history: List[Action] = []
        self.execution_history: List[Dict[str, Any]] = []

    def reset(self, **kwargs) -> Observation:
        """
        重置环境

        Args:
            **kwargs: 可选参数
                - data: 自定义市场数据列表
                - start_idx: 起始数据索引
                - seed: 随机种子

        Returns:
            初始观察状态
        """
        # 设置随机种子
        if 'seed' in kwargs:
            np.random.seed(kwargs['seed'])

        # 加载或使用提供的数据
        if 'data' in kwargs:
            self.market_data_list = kwargs['data']
        else:
            # 从data_loader加载数据
            self.market_data_list = self.data_loader.load_data(
                self.symbol,
                **kwargs
            )

        if len(self.market_data_list) < self.max_steps + self.window_size:
            raise ValueError(
                f"市场数据不足：需要至少 {self.max_steps + self.window_size} 条数据，"
                f"实际只有 {len(self.market_data_list)} 条"
            )

        # 重置状态
        self.current_step = 0
        self.current_data_idx = kwargs.get('start_idx', self.window_size)
        self.executed_quantity = 0.0
        self.total_cost = 0.0
        self.avg_execution_price = 0.0
        self.position = 0.0 if self.side == OrderSide.BUY else self.target_quantity
        self.cash = self.initial_cash

        # 重置历史
        self.price_history = []
        self.volume_history = []
        self.action_history = []
        self.execution_history = []

        # 初始化历史价格窗口
        for i in range(self.window_size):
            idx = self.current_data_idx - self.window_size + i
            self.price_history.append(self.market_data_list[idx].close)
            self.volume_history.append(self.market_data_list[idx].volume)

        # 创建初始状态
        self.state = self._build_state()

        return self.state.observation

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        执行一步

        Args:
            action: 智能体动作

        Returns:
            observation: 新的观察
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        if self.state is None:
            raise RuntimeError("环境未初始化，请先调用 reset()")

        # 保存当前状态用于奖励计算
        prev_state = deepcopy(self.state)

        # 获取当前市场数据
        current_market_data = self.market_data_list[self.current_data_idx]

        # 计算要执行的数量
        remaining_quantity = self.target_quantity - self.executed_quantity
        trade_quantity = action.trade_ratio * remaining_quantity

        executed_qty = 0.0
        execution_price = 0.0

        # 如果有实际交易
        if trade_quantity > 1e-6:  # 避免极小交易
            # 创建订单
            order = OrderInfo(
                order_id=f"order_{self.current_step}",
                symbol=self.symbol,
                side=self.side,
                order_type=action.order_type,
                quantity=trade_quantity,
                price=current_market_data.close * (1 + action.limit_price_offset) if action.order_type == OrderType.LIMIT else None,
                timestamp=current_market_data.timestamp,
            )

            # 执行订单
            executed_qty, execution_price, updated_order = self.execution_engine.execute_order(
                order,
                current_market_data,
                liquidity=1.0,  # 可以根据市场状态调整
            )

            # 更新执行状态
            if executed_qty > 0:
                # 更新总成本和平均价格
                self.total_cost += executed_qty * execution_price
                self.executed_quantity += executed_qty
                self.avg_execution_price = self.total_cost / self.executed_quantity

                # 更新持仓和现金
                if self.side == OrderSide.BUY:
                    self.position += executed_qty
                    self.cash -= executed_qty * execution_price
                else:
                    self.position -= executed_qty
                    self.cash += executed_qty * execution_price

                # 记录执行历史
                self.execution_history.append({
                    'step': self.current_step,
                    'quantity': executed_qty,
                    'price': execution_price,
                    'timestamp': current_market_data.timestamp,
                    'market_price': current_market_data.close,
                })

        # 记录动作
        self.action_history.append(action)

        # 前进到下一个时间步
        self.current_step += 1
        self.current_data_idx += 1

        # 更新历史
        self.price_history.append(current_market_data.close)
        self.volume_history.append(current_market_data.volume)

        # 检查是否结束
        done = (
            self.current_step >= self.max_steps or
            self.current_data_idx >= len(self.market_data_list) or
            self.executed_quantity >= self.target_quantity * 0.999  # 99.9%完成视为完成
        )

        # 构建新状态
        self.state = self._build_state()
        self.state.done = done

        # 计算奖励
        reward = self.reward_function.calculate(
            prev_state,
            action,
            self.state,
            executed_qty,
            execution_price if executed_qty > 0 else current_market_data.close,
        )

        # 构建info
        info = {
            'step': self.current_step,
            'executed_quantity': executed_qty,
            'execution_price': execution_price,
            'total_executed': self.executed_quantity,
            'avg_execution_price': self.avg_execution_price,
            'completion_rate': self.executed_quantity / self.target_quantity,
            'remaining_steps': self.max_steps - self.current_step,
            'market_price': current_market_data.close,
        }

        return self.state.observation, reward, done, info

    def get_observation(self) -> Observation:
        """获取当前观察"""
        if self.state is None:
            raise RuntimeError("环境未初始化")
        return self.state.observation

    def _build_state(self) -> State:
        """构建当前状态"""
        current_market_data = self.market_data_list[self.current_data_idx]

        # 构建观察
        observation = Observation(
            market_data=current_market_data,
            historical_prices=np.array(self.price_history[-self.window_size:]),
            historical_volumes=np.array(self.volume_history[-self.window_size:]),
            position=self.position,
            cash=self.cash,
            target_quantity=self.target_quantity - self.executed_quantity,
            time_remaining=self.max_steps - self.current_step,
            executed_quantity=self.executed_quantity,
            avg_execution_price=self.avg_execution_price,
            market_impact=self._estimate_market_impact(),
        )

        # 构建完整状态
        state = State(
            observation=observation,
            true_volatility=self._calculate_volatility(),
            liquidity=1.0,  # 简化处理
            step=self.current_step,
            done=False,
        )

        return state

    def _estimate_market_impact(self) -> float:
        """估计当前市场冲击"""
        if len(self.execution_history) == 0:
            return 0.0

        # 简单估计：最近几次执行的平均冲击
        recent_executions = self.execution_history[-5:]
        impacts = [
            abs(exec['price'] - exec['market_price']) / exec['market_price']
            for exec in recent_executions
        ]
        return np.mean(impacts) if impacts else 0.0

    def _calculate_volatility(self) -> float:
        """计算价格波动率"""
        if len(self.price_history) < 2:
            return 0.0

        prices = np.array(self.price_history[-self.window_size:])
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns)

    @property
    def observation_space_shape(self) -> Tuple[int, ...]:
        """观察空间形状"""
        # 基础特征数 + 历史价格窗口
        base_features = 11  # 从Observation.to_array()
        return (base_features + self.window_size,)

    @property
    def action_space_shape(self) -> Tuple[int, ...]:
        """动作空间形状"""
        # trade_ratio, order_type, limit_price_offset, urgency
        return (4,)

    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        渲染环境（可选）

        Args:
            mode: 渲染模式

        Returns:
            渲染结果（根据mode不同）
        """
        if mode == 'human':
            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"Executed: {self.executed_quantity:.2f}/{self.target_quantity:.2f} "
                  f"({self.executed_quantity/self.target_quantity*100:.1f}%)")
            print(f"Avg Price: {self.avg_execution_price:.4f}")
            print(f"Current Price: {self.market_data_list[self.current_data_idx].close:.4f}")
            print(f"Cash: {self.cash:.2f}, Position: {self.position:.2f}")
            print("-" * 50)
        return None

    def get_metrics(self) -> Dict[str, float]:
        """
        获取当前episode的指标

        Returns:
            指标字典
        """
        if len(self.execution_history) == 0:
            return {}

        # 计算VWAP
        market_vwap = np.average(
            [md.close for md in self.market_data_list[:self.current_data_idx]],
            weights=[md.volume for md in self.market_data_list[:self.current_data_idx]],
        )

        # 计算执行缺口（Implementation Shortfall）
        arrival_price = self.market_data_list[self.window_size].close
        if self.side == OrderSide.BUY:
            shortfall = (self.avg_execution_price - arrival_price) * self.executed_quantity
        else:
            shortfall = (arrival_price - self.avg_execution_price) * self.executed_quantity

        metrics = {
            'completion_rate': self.executed_quantity / self.target_quantity,
            'avg_execution_price': self.avg_execution_price,
            'vwap': market_vwap,
            'vwap_slippage': (self.avg_execution_price - market_vwap) / market_vwap,
            'implementation_shortfall': shortfall,
            'total_cost': self.total_cost,
            'num_executions': len(self.execution_history),
        }

        return metrics
