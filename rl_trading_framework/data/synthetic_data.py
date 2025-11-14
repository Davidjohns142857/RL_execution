"""
合成数据生成器

用于生成模拟的市场数据，便于测试和训练
"""

import numpy as np
from typing import List, Optional
from datetime import datetime, timedelta

from rl_trading_framework.core.base import BaseDataLoader
from rl_trading_framework.core.types import MarketData


class SyntheticDataGenerator(BaseDataLoader):
    """
    合成数据生成器

    使用几何布朗运动（GBM）生成模拟价格数据

    输入参数：
        - initial_price: 初始价格
        - mu: 漂移率（年化收益率）
        - sigma: 波动率（年化）
        - dt: 时间步长（秒）
        - add_microstructure_noise: 是否添加微观结构噪声

    输出：
        - List[MarketData]: 生成的市场数据列表

    数学模型：
        dS = μS dt + σS dW
        其中 W 是维纳过程
    """

    def __init__(
        self,
        initial_price: float = 100.0,
        mu: float = 0.0,  # 年化漂移率
        sigma: float = 0.2,  # 年化波动率
        dt: float = 60.0,  # 时间步长（秒）
        add_microstructure_noise: bool = True,
    ):
        """
        初始化合成数据生成器

        Args:
            initial_price: 初始价格
            mu: 漂移率（年化）
            sigma: 波动率（年化）
            dt: 时间步长（秒）
            add_microstructure_noise: 是否添加微观结构噪声
        """
        self.initial_price = initial_price
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.add_microstructure_noise = add_microstructure_noise

    def load_data(
        self,
        symbol: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        **kwargs
    ) -> List[MarketData]:
        """
        生成合成市场数据

        Args:
            symbol: 交易标的（仅用于标识）
            start_time: 开始时间（Unix时间戳）
            end_time: 结束时间（Unix时间戳）
            **kwargs: 其他参数
                - num_steps: 生成的数据点数量（如果未指定start_time和end_time）
                - seed: 随机种子

        Returns:
            生成的市场数据列表
        """
        # 设置随机种子
        if 'seed' in kwargs:
            np.random.seed(kwargs['seed'])

        # 确定时间范围
        if start_time is None:
            start_time = datetime.now().timestamp()

        if end_time is None:
            num_steps = kwargs.get('num_steps', 1000)
            end_time = start_time + num_steps * self.dt
        else:
            num_steps = int((end_time - start_time) / self.dt)

        # 生成价格路径（几何布朗运动）
        prices = self._generate_gbm_path(num_steps)

        # 生成市场数据
        market_data_list = []
        current_time = start_time

        for i in range(num_steps):
            # 基础价格
            base_price = prices[i]

            # 生成OHLC
            if self.add_microstructure_noise:
                # 添加日内波动
                intraday_volatility = self.sigma * base_price * 0.01  # 1%的日内波动
                high = base_price + abs(np.random.normal(0, intraday_volatility))
                low = base_price - abs(np.random.normal(0, intraday_volatility))
                open_price = base_price + np.random.normal(0, intraday_volatility * 0.5)
                close_price = base_price + np.random.normal(0, intraday_volatility * 0.5)

                # 确保 high >= close, open >= low
                high = max(high, open_price, close_price)
                low = min(low, open_price, close_price)
            else:
                open_price = base_price
                close_price = base_price
                high = base_price
                low = base_price

            # 生成成交量（对数正态分布）
            avg_volume = 1000000
            volume = np.random.lognormal(np.log(avg_volume), 0.5)

            # 生成买卖价差
            spread = base_price * 0.001  # 0.1% 价差
            bid_price = close_price - spread / 2
            ask_price = close_price + spread / 2

            # 计算VWAP（简化为close价格加上小的随机扰动）
            vwap = close_price * (1 + np.random.normal(0, 0.0001))

            market_data = MarketData(
                timestamp=current_time,
                open=float(open_price),
                high=float(high),
                low=float(low),
                close=float(close_price),
                volume=float(volume),
                bid_price=float(bid_price),
                ask_price=float(ask_price),
                bid_volume=float(volume * np.random.uniform(0.3, 0.7)),
                ask_volume=float(volume * np.random.uniform(0.3, 0.7)),
                vwap=float(vwap),
            )

            market_data_list.append(market_data)
            current_time += self.dt

        return market_data_list

    def _generate_gbm_path(self, num_steps: int) -> np.ndarray:
        """
        生成几何布朗运动价格路径

        Args:
            num_steps: 步数

        Returns:
            价格数组
        """
        # 转换为每步的参数
        dt_years = self.dt / (365.25 * 24 * 3600)  # 转换为年
        mu_dt = self.mu * dt_years
        sigma_dt = self.sigma * np.sqrt(dt_years)

        # 生成随机收益
        returns = np.random.normal(mu_dt, sigma_dt, num_steps)

        # 计算价格路径
        price_ratios = np.exp(returns)
        prices = self.initial_price * np.cumprod(price_ratios)

        return prices

    def preprocess(self, data: List[MarketData]) -> List[MarketData]:
        """
        预处理数据（合成数据通常不需要预处理）

        Args:
            data: 市场数据

        Returns:
            原样返回
        """
        return data

    def generate_trend_data(
        self,
        num_steps: int = 1000,
        trend: str = "up",
        trend_strength: float = 0.1,
        **kwargs
    ) -> List[MarketData]:
        """
        生成带有特定趋势的数据

        Args:
            num_steps: 数据点数量
            trend: 趋势类型（"up", "down", "sideways"）
            trend_strength: 趋势强度
            **kwargs: 其他参数

        Returns:
            市场数据列表
        """
        # 调整mu以产生趋势
        original_mu = self.mu
        if trend == "up":
            self.mu = trend_strength
        elif trend == "down":
            self.mu = -trend_strength
        else:  # sideways
            self.mu = 0.0

        # 生成数据
        data = self.load_data("SYNTHETIC", num_steps=num_steps, **kwargs)

        # 恢复原始mu
        self.mu = original_mu

        return data
