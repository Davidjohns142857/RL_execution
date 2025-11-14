"""
CSV数据加载器

从CSV文件加载历史市场数据
"""

import os
import pandas as pd
from typing import List, Optional
import numpy as np

from rl_trading_framework.core.base import BaseDataLoader
from rl_trading_framework.core.types import MarketData


class CSVDataLoader(BaseDataLoader):
    """
    CSV数据加载器

    输入：
        - data_dir: CSV数据文件目录
        - normalize: 是否标准化数据

    输出：
        - List[MarketData]: 市场数据列表

    CSV格式要求：
        必须包含列：timestamp, open, high, low, close, volume
        可选列：bid_price, ask_price, bid_volume, ask_volume, vwap
    """

    def __init__(self, data_dir: str, normalize: bool = False):
        """
        初始化CSV数据加载器

        Args:
            data_dir: CSV文件所在目录
            normalize: 是否标准化数据
        """
        self.data_dir = data_dir
        self.normalize = normalize
        self._cache = {}  # 数据缓存

    def load_data(
        self,
        symbol: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        **kwargs
    ) -> List[MarketData]:
        """
        从CSV文件加载数据

        Args:
            symbol: 交易标的代码（对应CSV文件名，如 "AAPL.csv"）
            start_time: 开始时间（Unix时间戳）
            end_time: 结束时间（Unix时间戳）
            **kwargs: 其他参数

        Returns:
            市场数据列表
        """
        # 构建文件路径
        file_path = os.path.join(self.data_dir, f"{symbol}.csv")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")

        # 检查缓存
        cache_key = f"{symbol}_{start_time}_{end_time}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 读取CSV
        df = pd.read_csv(file_path)

        # 验证必需的列
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"CSV文件缺少必需的列: {missing_columns}")

        # 过滤时间范围
        if start_time is not None:
            df = df[df['timestamp'] >= start_time]
        if end_time is not None:
            df = df[df['timestamp'] <= end_time]

        # 转换为MarketData对象
        market_data_list = []
        for _, row in df.iterrows():
            market_data = MarketData(
                timestamp=float(row['timestamp']),
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row['volume']),
                bid_price=float(row['bid_price']) if 'bid_price' in row and pd.notna(row['bid_price']) else None,
                ask_price=float(row['ask_price']) if 'ask_price' in row and pd.notna(row['ask_price']) else None,
                bid_volume=float(row['bid_volume']) if 'bid_volume' in row and pd.notna(row['bid_volume']) else None,
                ask_volume=float(row['ask_volume']) if 'ask_volume' in row and pd.notna(row['ask_volume']) else None,
                vwap=float(row['vwap']) if 'vwap' in row and pd.notna(row['vwap']) else None,
            )
            market_data_list.append(market_data)

        # 预处理
        market_data_list = self.preprocess(market_data_list)

        # 缓存
        self._cache[cache_key] = market_data_list

        return market_data_list

    def preprocess(self, data: List[MarketData]) -> List[MarketData]:
        """
        预处理数据

        Args:
            data: 原始市场数据

        Returns:
            预处理后的数据
        """
        if not self.normalize:
            return data

        # 提取价格和成交量
        prices = np.array([d.close for d in data])
        volumes = np.array([d.volume for d in data])

        # 标准化（使用第一个价格作为基准）
        price_base = prices[0]

        normalized_data = []
        for i, d in enumerate(data):
            scale = price_base
            normalized_md = MarketData(
                timestamp=d.timestamp,
                open=d.open / scale,
                high=d.high / scale,
                low=d.low / scale,
                close=d.close / scale,
                volume=d.volume,  # 成交量不标准化
                bid_price=d.bid_price / scale if d.bid_price else None,
                ask_price=d.ask_price / scale if d.ask_price else None,
                bid_volume=d.bid_volume,
                ask_volume=d.ask_volume,
                vwap=d.vwap / scale if d.vwap else None,
            )
            normalized_data.append(normalized_md)

        return normalized_data

    def save_to_csv(self, data: List[MarketData], symbol: str) -> None:
        """
        保存数据到CSV文件

        Args:
            data: 市场数据列表
            symbol: 交易标的代码
        """
        file_path = os.path.join(self.data_dir, f"{symbol}.csv")

        # 转换为DataFrame
        records = [d.to_dict() for d in data]
        df = pd.DataFrame(records)

        # 保存
        df.to_csv(file_path, index=False)
        print(f"数据已保存到: {file_path}")
