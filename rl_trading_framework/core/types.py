"""
核心数据类型定义

本模块定义了框架中使用的所有核心数据类型，确保类型安全和清晰的接口定义。
"""

from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    TWAP = "twap"  # Time-Weighted Average Price
    VWAP = "vwap"  # Volume-Weighted Average Price
    POV = "pov"    # Percentage of Volume


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class MarketData:
    """
    市场数据

    支持从简单的OHLCV数据到完整的Level-2高频数据（10档行情）

    Attributes:
        timestamp: 时间戳（Unix时间戳，单位：秒）
        open: 开盘价
        high: 最高价
        low: 最低价
        close: 收盘价（或最新价last_prc）
        volume: 成交量

        # Level-1 数据（可选）
        bid_price: 买一价（可选）
        ask_price: 卖一价（可选）
        bid_volume: 买一量（可选）
        ask_volume: 卖一量（可选）
        vwap: 成交量加权平均价（可选）

        # Level-2 高频数据（可选）
        bid_prices: 买盘价格列表（10档，可选）
        bid_volumes: 买盘数量列表（10档，可选）
        ask_prices: 卖盘价格列表（10档，可选）
        ask_volumes: 卖盘数量列表（10档，可选）

        # 额外市场信息（可选）
        prev_close: 昨收价（可选）
        turnover: 成交额（可选）
        num_trades: 成交笔数（可选）
        high_limited: 涨停价（可选）
        low_limited: 跌停价（可选）
        weighted_bid_price: 加权买价（可选）
        weighted_ask_price: 加权卖价（可选）
        total_bid_volume: 买盘总量（可选）
        total_ask_volume: 卖盘总量（可选）
        trading_status: 交易状态（可选，83=开市前，67=集合竞价，84=连续交易，69=闭市，66=休市）

        # 元数据（可选）
        symbol: 股票代码（可选）
        datetime_str: 日期时间字符串（可选）
        exchtime: 交易所时间戳微秒（可选）
        localtime: 本地时间戳微秒（可选）
    """
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float

    # Level-1
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    bid_volume: Optional[float] = None
    ask_volume: Optional[float] = None
    vwap: Optional[float] = None

    # Level-2 (10档行情)
    bid_prices: Optional[List[float]] = None
    bid_volumes: Optional[List[float]] = None
    ask_prices: Optional[List[float]] = None
    ask_volumes: Optional[List[float]] = None

    # 额外信息
    prev_close: Optional[float] = None
    turnover: Optional[float] = None
    num_trades: Optional[int] = None
    high_limited: Optional[float] = None
    low_limited: Optional[float] = None
    weighted_bid_price: Optional[float] = None
    weighted_ask_price: Optional[float] = None
    total_bid_volume: Optional[float] = None
    total_ask_volume: Optional[float] = None
    trading_status: Optional[int] = None

    # 元数据
    symbol: Optional[str] = None
    datetime_str: Optional[str] = None
    exchtime: Optional[int] = None
    localtime: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'bid_price': self.bid_price,
            'ask_price': self.ask_price,
            'bid_volume': self.bid_volume,
            'ask_volume': self.ask_volume,
            'vwap': self.vwap,
            'bid_prices': self.bid_prices,
            'bid_volumes': self.bid_volumes,
            'ask_prices': self.ask_prices,
            'ask_volumes': self.ask_volumes,
            'prev_close': self.prev_close,
            'turnover': self.turnover,
            'num_trades': self.num_trades,
            'high_limited': self.high_limited,
            'low_limited': self.low_limited,
            'weighted_bid_price': self.weighted_bid_price,
            'weighted_ask_price': self.weighted_ask_price,
            'total_bid_volume': self.total_bid_volume,
            'total_ask_volume': self.total_ask_volume,
            'trading_status': self.trading_status,
            'symbol': self.symbol,
            'datetime_str': self.datetime_str,
            'exchtime': self.exchtime,
            'localtime': self.localtime,
        }

    def get_mid_price(self) -> float:
        """获取中间价（买一卖一的平均）"""
        if self.bid_price is not None and self.ask_price is not None:
            return (self.bid_price + self.ask_price) / 2.0
        return self.close

    def get_spread(self) -> float:
        """获取买卖价差"""
        if self.bid_price is not None and self.ask_price is not None:
            return self.ask_price - self.bid_price
        return 0.0

    def get_spread_bps(self) -> float:
        """获取买卖价差（基点，万分之一）"""
        spread = self.get_spread()
        mid = self.get_mid_price()
        if mid > 0:
            return (spread / mid) * 10000
        return 0.0


@dataclass
class OrderInfo:
    """
    订单信息

    Attributes:
        order_id: 订单ID
        symbol: 交易标的
        side: 订单方向（买/卖）
        order_type: 订单类型（市价/限价等）
        quantity: 订单数量（总量）
        price: 订单价格（限价单使用）
        filled_quantity: 已成交数量
        avg_fill_price: 平均成交价格
        status: 订单状态
        timestamp: 订单创建时间
        params: 额外参数（如TWAP时长、POV比例等）
    """
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    timestamp: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def remaining_quantity(self) -> float:
        """剩余未成交数量"""
        return self.quantity - self.filled_quantity

    @property
    def is_finished(self) -> bool:
        """订单是否已完成（成交或取消）"""
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]


@dataclass
class Action:
    """
    智能体动作

    对于交易执行任务，动作通常包括：
    - 本时间步要交易的数量或比例
    - 订单类型选择
    - 其他执行参数

    Attributes:
        trade_ratio: 交易比例（0-1之间，表示剩余目标数量的百分比）
        order_type: 订单类型
        limit_price_offset: 限价单相对市场价格的偏移（百分比）
        urgency: 紧急程度（0-1之间，影响执行激进程度）
        params: 其他自定义参数
    """
    trade_ratio: float = 0.0  # 0表示不交易，1表示全部交易
    order_type: OrderType = OrderType.MARKET
    limit_price_offset: float = 0.0  # 例如0.001表示高于市价0.1%
    urgency: float = 0.5  # 紧急程度
    params: Dict[str, Any] = field(default_factory=dict)

    def to_array(self) -> np.ndarray:
        """转换为numpy数组（用于某些RL算法）"""
        return np.array([
            self.trade_ratio,
            self.order_type.value if isinstance(self.order_type, Enum) else 0,
            self.limit_price_offset,
            self.urgency,
        ])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Action':
        """从numpy数组创建Action对象"""
        return cls(
            trade_ratio=float(arr[0]),
            order_type=OrderType.MARKET,  # 简化处理
            limit_price_offset=float(arr[2]) if len(arr) > 2 else 0.0,
            urgency=float(arr[3]) if len(arr) > 3 else 0.5,
        )


@dataclass
class Observation:
    """
    环境观察

    包含智能体观察到的所有信息，用于决策

    Attributes:
        market_data: 当前市场数据
        historical_prices: 历史价格序列
        historical_volumes: 历史成交量序列
        position: 当前持仓
        cash: 当前现金
        target_quantity: 目标交易数量（剩余）
        time_remaining: 剩余时间步数
        executed_quantity: 已执行数量
        avg_execution_price: 平均执行价格
        market_impact: 市场冲击估计
        extra_features: 额外特征（如技术指标、订单簿特征等）
    """
    market_data: MarketData
    historical_prices: np.ndarray = field(default_factory=lambda: np.array([]))
    historical_volumes: np.ndarray = field(default_factory=lambda: np.array([]))
    position: float = 0.0
    cash: float = 0.0
    target_quantity: float = 0.0
    time_remaining: int = 0
    executed_quantity: float = 0.0
    avg_execution_price: float = 0.0
    market_impact: float = 0.0
    extra_features: Dict[str, np.ndarray] = field(default_factory=dict)

    def to_array(self) -> np.ndarray:
        """
        转换为numpy数组（用于神经网络输入）

        Returns:
            flatten的特征向量
        """
        features = [
            # 当前市场状态
            self.market_data.close,
            self.market_data.volume,
            self.market_data.high - self.market_data.low,  # 价格波动

            # 历史价格统计
            np.mean(self.historical_prices) if len(self.historical_prices) > 0 else 0.0,
            np.std(self.historical_prices) if len(self.historical_prices) > 0 else 0.0,

            # 仓位和目标
            self.position,
            self.target_quantity,
            self.executed_quantity / max(self.target_quantity, 1e-8),  # 执行进度

            # 时间和价格
            self.time_remaining,
            self.avg_execution_price,
            self.market_impact,
        ]

        # 添加历史价格窗口
        if len(self.historical_prices) > 0:
            features.extend(self.historical_prices[-10:].tolist())  # 最近10个价格
        else:
            features.extend([0.0] * 10)

        return np.array(features, dtype=np.float32)

    def get_feature_dim(self) -> int:
        """获取特征维度"""
        return len(self.to_array())


@dataclass
class State:
    """
    环境状态（内部状态，比Observation更完整）

    State包含环境的完整状态，而Observation是智能体能观察到的部分

    Attributes:
        observation: 智能体可观察的状态
        true_volatility: 真实波动率（不一定对智能体可见）
        liquidity: 流动性状态
        order_book: 订单簿（可选）
        pending_orders: 待成交订单列表
        step: 当前时间步
        done: episode是否结束
        info: 其他信息
    """
    observation: Observation
    true_volatility: float = 0.0
    liquidity: float = 1.0
    order_book: Optional[Dict[str, Any]] = None
    pending_orders: List[OrderInfo] = field(default_factory=list)
    step: int = 0
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransitionBatch:
    """
    经验回放批次

    用于训练RL算法的经验数据批次

    Attributes:
        states: 状态批次
        actions: 动作批次
        rewards: 奖励批次
        next_states: 下一状态批次
        dones: 终止标志批次
    """
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray

    def __len__(self) -> int:
        return len(self.states)


@dataclass
class EpisodeMetrics:
    """
    Episode评估指标

    Attributes:
        total_reward: 总奖励
        execution_shortfall: 执行缺口（Implementation Shortfall）
        vwap_slippage: 相对VWAP的滑点
        market_impact: 总市场冲击
        completion_rate: 完成率
        avg_execution_price: 平均执行价格
        total_cost: 总成本
        sharpe_ratio: 夏普比率（可选）
        steps: 总步数
        extra_metrics: 额外指标
    """
    total_reward: float = 0.0
    execution_shortfall: float = 0.0
    vwap_slippage: float = 0.0
    market_impact: float = 0.0
    completion_rate: float = 0.0
    avg_execution_price: float = 0.0
    total_cost: float = 0.0
    sharpe_ratio: Optional[float] = None
    steps: int = 0
    extra_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'total_reward': self.total_reward,
            'execution_shortfall': self.execution_shortfall,
            'vwap_slippage': self.vwap_slippage,
            'market_impact': self.market_impact,
            'completion_rate': self.completion_rate,
            'avg_execution_price': self.avg_execution_price,
            'total_cost': self.total_cost,
            'sharpe_ratio': self.sharpe_ratio,
            'steps': self.steps,
            **self.extra_metrics,
        }
