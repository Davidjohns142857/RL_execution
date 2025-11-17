"""
历史真实数据加载器

支持从CSV文件加载真实市场数据，包括：
- market_data: Level-2行情数据（10档买卖盘）
- transaction: 逐笔成交数据（可选）
- order: 委托订单数据（可选）

数据格式：
- 文件名格式: {symbol}_{date}_market_data.csv
- 支持中国A股Level-2高频数据格式
"""

import os
from typing import List, Optional, Dict, Any
from datetime import datetime
import numpy as np
import pandas as pd

from rl_trading_framework.core.base import BaseDataLoader
from rl_trading_framework.core.types import MarketData


class HistoricalDataLoader(BaseDataLoader):
    """
    历史真实数据加载器

    输入：
        - data_dir: 数据目录路径
        - symbol: 股票代码（例如 "000001"）
        - date: 日期（例如 "20170105"）
        - data_type: 数据类型 ("market_data", "transaction", "order")

    输出：
        - List[MarketData]: 市场数据列表

    数据字段映射：
        CSV字段 -> MarketData字段
        - datetime -> timestamp (转换为Unix时间戳)
        - last_prc -> close (最新价)
        - open -> open
        - high -> high
        - low -> low
        - volume -> volume
        - bid_prc1 -> bid_price (买一价)
        - ask_prc1 -> ask_price (卖一价)
        - bid_vol1 -> bid_volume (买一量)
        - ask_vol1 -> ask_volume (卖一量)
        - bid_prc1-10 -> bid_prices (10档买盘价格)
        - bid_vol1-10 -> bid_volumes (10档买盘数量)
        - ask_prc1-10 -> ask_prices (10档卖盘价格)
        - ask_vol1-10 -> ask_volumes (10档卖盘数量)
        - 其他字段映射见代码

    使用示例：
        >>> loader = HistoricalDataLoader(
        ...     data_dir="/path/to/data/market_data",
        ...     default_symbol="000001",
        ...     default_date="20170105"
        ... )
        >>> data = loader.load_data("000001", date="20170105", num_steps=1000)
        >>> print(f"加载了 {len(data)} 条数据")
    """

    def __init__(
        self,
        data_dir: str,
        default_symbol: str = "000001",
        default_date: Optional[str] = None,
        trading_hours_only: bool = True,
        subsample_freq: Optional[int] = None,
    ):
        """
        初始化历史数据加载器

        Args:
            data_dir: 数据文件夹路径
            default_symbol: 默认股票代码
            default_date: 默认日期（格式：YYYYMMDD）
            trading_hours_only: 是否只加载交易时段数据（过滤开市前、闭市后）
            subsample_freq: 子采样频率（例如：10表示每10条取1条）
        """
        self.data_dir = data_dir
        self.default_symbol = default_symbol
        self.default_date = default_date
        self.trading_hours_only = trading_hours_only
        self.subsample_freq = subsample_freq

        # 缓存已加载的数据
        self._cache: Dict[str, List[MarketData]] = {}

    def load_data(
        self,
        symbol: Optional[str] = None,
        date: Optional[str] = None,
        num_steps: Optional[int] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        **kwargs
    ) -> List[MarketData]:
        """
        加载市场数据

        Args:
            symbol: 股票代码（默认使用default_symbol）
            date: 日期 YYYYMMDD 格式（默认使用default_date）
            num_steps: 返回的数据条数（从start_time开始）
            start_time: 起始时间 "HH:MM:SS" 格式
            end_time: 结束时间 "HH:MM:SS" 格式
            **kwargs: 其他参数

        Returns:
            MarketData列表

        Raises:
            FileNotFoundError: 如果数据文件不存在
            ValueError: 如果参数无效
        """
        symbol = symbol or self.default_symbol
        date = date or self.default_date

        if symbol is None or date is None:
            raise ValueError("必须提供symbol和date参数")

        # 生成缓存键
        cache_key = f"{symbol}_{date}"

        # 检查缓存
        if cache_key in self._cache:
            data = self._cache[cache_key]
        else:
            # 加载数据
            data = self._load_from_csv(symbol, date)
            self._cache[cache_key] = data

        # 应用过滤
        filtered_data = self._filter_data(
            data,
            start_time=start_time,
            end_time=end_time,
            num_steps=num_steps
        )

        return filtered_data

    def _load_from_csv(self, symbol: str, date: str) -> List[MarketData]:
        """
        从CSV文件加载数据

        Args:
            symbol: 股票代码
            date: 日期

        Returns:
            MarketData列表
        """
        # 构建文件路径
        filename = f"{symbol}_{date}_market_data.csv"
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"数据文件不存在: {filepath}\n"
                f"请确保数据文件在 {self.data_dir} 目录下，"
                f"文件名格式为 {{symbol}}_{{date}}_market_data.csv"
            )

        # 读取CSV
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            raise RuntimeError(f"读取CSV文件失败: {filepath}, 错误: {e}")

        # 数据验证
        required_columns = ['datetime', 'last_prc', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"CSV文件缺少必要列: {missing_columns}")

        # 转换为MarketData列表
        market_data_list = []

        for idx, row in df.iterrows():
            # 过滤交易时段（如果需要）
            if self.trading_hours_only:
                status = row.get('status', 84) % 256
                # 只保留连续交易时段 (84)
                if status != 84:
                    continue

            # 子采样（如果需要）
            if self.subsample_freq is not None:
                if idx % self.subsample_freq != 0:
                    continue

            market_data = self._row_to_market_data(row, symbol)
            market_data_list.append(market_data)

        if len(market_data_list) == 0:
            raise ValueError(f"没有有效数据: {filepath}")

        return market_data_list

    def _row_to_market_data(self, row: pd.Series, symbol: str) -> MarketData:
        """
        将DataFrame行转换为MarketData对象

        Args:
            row: DataFrame行
            symbol: 股票代码

        Returns:
            MarketData对象
        """
        # 解析时间戳
        datetime_str = str(row['datetime'])
        try:
            dt = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
            timestamp = dt.timestamp()
        except:
            # 如果解析失败，使用exchtime（微秒转秒）
            if 'exchtime' in row and pd.notna(row['exchtime']):
                timestamp = row['exchtime'] / 1e6
            else:
                timestamp = 0.0

        # 提取10档买卖盘
        bid_prices = []
        bid_volumes = []
        ask_prices = []
        ask_volumes = []

        for i in range(1, 11):
            bid_prc_col = f'bid_prc{i}'
            bid_vol_col = f'bid_vol{i}'
            ask_prc_col = f'ask_prc{i}'
            ask_vol_col = f'ask_vol{i}'

            if bid_prc_col in row and pd.notna(row[bid_prc_col]):
                bid_prices.append(float(row[bid_prc_col]))
            else:
                bid_prices.append(0.0)

            if bid_vol_col in row and pd.notna(row[bid_vol_col]):
                bid_volumes.append(float(row[bid_vol_col]))
            else:
                bid_volumes.append(0.0)

            if ask_prc_col in row and pd.notna(row[ask_prc_col]):
                ask_prices.append(float(row[ask_prc_col]))
            else:
                ask_prices.append(0.0)

            if ask_vol_col in row and pd.notna(row[ask_vol_col]):
                ask_volumes.append(float(row[ask_vol_col]))
            else:
                ask_volumes.append(0.0)

        # 安全获取值的辅助函数
        def safe_get(key, default=None):
            if key in row and pd.notna(row[key]):
                return row[key]
            return default

        # 创建MarketData对象
        market_data = MarketData(
            # 基础OHLCV
            timestamp=timestamp,
            open=float(safe_get('open', safe_get('last_prc', 0))),
            high=float(safe_get('high', safe_get('last_prc', 0))),
            low=float(safe_get('low', safe_get('last_prc', 0))),
            close=float(safe_get('last_prc', 0)),
            volume=float(safe_get('volume', 0)),

            # Level-1 买一卖一
            bid_price=float(safe_get('bid_prc1', 0)) if safe_get('bid_prc1') else None,
            ask_price=float(safe_get('ask_prc1', 0)) if safe_get('ask_prc1') else None,
            bid_volume=float(safe_get('bid_vol1', 0)) if safe_get('bid_vol1') else None,
            ask_volume=float(safe_get('ask_vol1', 0)) if safe_get('ask_vol1') else None,

            # Level-2 10档行情
            bid_prices=bid_prices if any(p > 0 for p in bid_prices) else None,
            bid_volumes=bid_volumes if any(v > 0 for v in bid_volumes) else None,
            ask_prices=ask_prices if any(p > 0 for p in ask_prices) else None,
            ask_volumes=ask_volumes if any(v > 0 for v in ask_volumes) else None,

            # 额外信息
            prev_close=float(safe_get('prev_close')) if safe_get('prev_close') else None,
            turnover=float(safe_get('turnover')) if safe_get('turnover') else None,
            num_trades=int(safe_get('num_trades')) if safe_get('num_trades') else None,
            high_limited=float(safe_get('high_limited')) if safe_get('high_limited') else None,
            low_limited=float(safe_get('low_limited')) if safe_get('low_limited') else None,
            weighted_bid_price=float(safe_get('weighted_bid_prc')) if safe_get('weighted_bid_prc') else None,
            weighted_ask_price=float(safe_get('weighted_ask_prc')) if safe_get('weighted_ask_prc') else None,
            total_bid_volume=float(safe_get('total_bid_vol')) if safe_get('total_bid_vol') else None,
            total_ask_volume=float(safe_get('total_ask_vol')) if safe_get('total_ask_vol') else None,
            trading_status=int(safe_get('status', 84)) % 256 if safe_get('status') else None,

            # 元数据
            symbol=symbol,
            datetime_str=datetime_str,
            exchtime=int(safe_get('exchtime')) if safe_get('exchtime') else None,
            localtime=int(safe_get('localtime')) if safe_get('localtime') else None,
        )

        return market_data

    def _filter_data(
        self,
        data: List[MarketData],
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        num_steps: Optional[int] = None,
    ) -> List[MarketData]:
        """
        过滤数据

        Args:
            data: 原始数据
            start_time: 起始时间 "HH:MM:SS"
            end_time: 结束时间 "HH:MM:SS"
            num_steps: 返回的数据条数

        Returns:
            过滤后的数据
        """
        filtered = data

        # 时间范围过滤
        if start_time or end_time:
            filtered = []
            for md in data:
                if md.datetime_str is None:
                    continue

                time_str = md.datetime_str.split(' ')[-1]  # 提取时间部分

                if start_time and time_str < start_time:
                    continue
                if end_time and time_str > end_time:
                    continue

                filtered.append(md)

        # 限制数量
        if num_steps is not None:
            filtered = filtered[:num_steps]

        return filtered

    def get_available_dates(self, symbol: Optional[str] = None) -> List[str]:
        """
        获取可用的日期列表

        Args:
            symbol: 股票代码

        Returns:
            日期列表（格式：YYYYMMDD）
        """
        symbol = symbol or self.default_symbol
        dates = []

        if not os.path.exists(self.data_dir):
            return dates

        # 扫描目录查找文件
        pattern = f"{symbol}_"
        for filename in os.listdir(self.data_dir):
            if filename.startswith(pattern) and filename.endswith('_market_data.csv'):
                # 提取日期部分
                parts = filename.replace('.csv', '').split('_')
                if len(parts) >= 2:
                    date = parts[1]
                    dates.append(date)

        return sorted(dates)

    def get_data_info(self, symbol: Optional[str] = None, date: Optional[str] = None) -> Dict[str, Any]:
        """
        获取数据信息

        Args:
            symbol: 股票代码
            date: 日期

        Returns:
            数据信息字典
        """
        symbol = symbol or self.default_symbol
        date = date or self.default_date

        if symbol is None or date is None:
            return {}

        try:
            data = self.load_data(symbol, date)

            if len(data) == 0:
                return {}

            return {
                'symbol': symbol,
                'date': date,
                'num_records': len(data),
                'start_time': data[0].datetime_str,
                'end_time': data[-1].datetime_str,
                'price_range': (
                    min(md.low for md in data),
                    max(md.high for md in data),
                ),
                'total_volume': sum(md.volume for md in data if md.volume),
                'has_level2': data[0].bid_prices is not None,
            }
        except Exception as e:
            return {'error': str(e)}

    def preprocess(self, data: List[MarketData]) -> List[MarketData]:
        """
        预处理数据

        Args:
            data: 原始数据列表

        Returns:
            预处理后的数据列表
        """
        # 默认不做额外预处理，因为已经在load_data中处理了
        # 如果需要额外的预处理，可以在这里添加
        return data
