"""
数据工具模块

提供数据验证、预处理、统计分析等功能
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from rl_trading_framework.core.types import MarketData


class DataValidator:
    """
    数据验证器

    用于验证MarketData的完整性和质量
    """

    @staticmethod
    def validate_market_data(data: MarketData) -> Dict[str, Any]:
        """
        验证单条MarketData

        Args:
            data: MarketData对象

        Returns:
            验证结果字典
        """
        issues = []
        warnings = []

        # 1. 检查必要字段
        if data.close <= 0:
            issues.append("close价格必须大于0")

        if data.volume < 0:
            issues.append("成交量不能为负")

        # 2. 检查OHLC逻辑
        if data.high < data.low:
            issues.append(f"最高价({data.high})不能低于最低价({data.low})")

        if data.close > data.high or data.close < data.low:
            warnings.append(f"收盘价({data.close})超出最高({data.high})最低({data.low})范围")

        if data.open > data.high or data.open < data.low:
            warnings.append(f"开盘价({data.open})超出最高最低范围")

        # 3. 检查买卖盘
        if data.bid_price is not None and data.ask_price is not None:
            if data.bid_price > data.ask_price:
                issues.append(
                    f"买一价({data.bid_price})不能高于卖一价({data.ask_price})"
                )

            # 检查价差是否合理（不超过5%）
            spread_pct = (data.ask_price - data.bid_price) / data.bid_price * 100
            if spread_pct > 5:
                warnings.append(f"买卖价差过大: {spread_pct:.2f}%")

        # 4. 检查10档行情
        if data.bid_prices is not None and len(data.bid_prices) > 0:
            # 检查买盘价格递减
            for i in range(len(data.bid_prices) - 1):
                if data.bid_prices[i] > 0 and data.bid_prices[i+1] > 0:
                    if data.bid_prices[i] < data.bid_prices[i+1]:
                        warnings.append(
                            f"买盘价格应递减: bid_prc{i+1}={data.bid_prices[i]} "
                            f"> bid_prc{i+2}={data.bid_prices[i+1]}"
                        )

        if data.ask_prices is not None and len(data.ask_prices) > 0:
            # 检查卖盘价格递增
            for i in range(len(data.ask_prices) - 1):
                if data.ask_prices[i] > 0 and data.ask_prices[i+1] > 0:
                    if data.ask_prices[i] > data.ask_prices[i+1]:
                        warnings.append(
                            f"卖盘价格应递增: ask_prc{i+1}={data.ask_prices[i]} "
                            f"< ask_prc{i+2}={data.ask_prices[i+1]}"
                        )

        # 5. 检查涨跌停
        if data.high_limited is not None and data.close > data.high_limited:
            warnings.append(f"价格({data.close})超过涨停价({data.high_limited})")

        if data.low_limited is not None and data.close < data.low_limited:
            warnings.append(f"价格({data.close})低于跌停价({data.low_limited})")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
        }

    @staticmethod
    def validate_data_list(data_list: List[MarketData]) -> Dict[str, Any]:
        """
        验证MarketData列表

        Args:
            data_list: MarketData列表

        Returns:
            验证结果统计
        """
        if len(data_list) == 0:
            return {
                'valid': False,
                'error': '数据列表为空',
            }

        total_issues = 0
        total_warnings = 0
        invalid_records = []

        for idx, data in enumerate(data_list):
            result = DataValidator.validate_market_data(data)
            if not result['valid']:
                invalid_records.append((idx, result['issues']))
                total_issues += len(result['issues'])

            total_warnings += len(result['warnings'])

        # 检查时间序列
        time_issues = []
        for i in range(len(data_list) - 1):
            if data_list[i].timestamp >= data_list[i+1].timestamp:
                time_issues.append(f"时间戳非递增: index {i} -> {i+1}")

        return {
            'valid': total_issues == 0 and len(time_issues) == 0,
            'total_records': len(data_list),
            'invalid_records': len(invalid_records),
            'total_issues': total_issues,
            'total_warnings': total_warnings,
            'time_issues': time_issues,
            'invalid_details': invalid_records[:10],  # 只返回前10个
        }


class DataAnalyzer:
    """
    数据分析器

    提供数据统计和分析功能
    """

    @staticmethod
    def get_basic_stats(data_list: List[MarketData]) -> Dict[str, Any]:
        """
        获取基本统计信息

        Args:
            data_list: MarketData列表

        Returns:
            统计信息字典
        """
        if len(data_list) == 0:
            return {}

        prices = [md.close for md in data_list]
        volumes = [md.volume for md in data_list if md.volume is not None]

        # 计算收益率
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                ret = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(ret)

        stats = {
            'num_records': len(data_list),
            'time_range': {
                'start': data_list[0].datetime_str if data_list[0].datetime_str else data_list[0].timestamp,
                'end': data_list[-1].datetime_str if data_list[-1].datetime_str else data_list[-1].timestamp,
            },
            'price': {
                'mean': np.mean(prices),
                'std': np.std(prices),
                'min': np.min(prices),
                'max': np.max(prices),
                'start': prices[0],
                'end': prices[-1],
                'change': (prices[-1] - prices[0]) / prices[0] * 100,  # 百分比
            },
            'volume': {
                'mean': np.mean(volumes) if volumes else 0,
                'std': np.std(volumes) if volumes else 0,
                'total': np.sum(volumes) if volumes else 0,
            },
            'returns': {
                'mean': np.mean(returns) if returns else 0,
                'std': np.std(returns) if returns else 0,
                'min': np.min(returns) if returns else 0,
                'max': np.max(returns) if returns else 0,
            },
        }

        # Level-2 统计
        if data_list[0].bid_prices is not None:
            all_spreads = []
            for md in data_list:
                if md.bid_price and md.ask_price:
                    spread_bps = md.get_spread_bps()
                    all_spreads.append(spread_bps)

            if all_spreads:
                stats['spread'] = {
                    'mean_bps': np.mean(all_spreads),
                    'std_bps': np.std(all_spreads),
                    'min_bps': np.min(all_spreads),
                    'max_bps': np.max(all_spreads),
                }

        return stats

    @staticmethod
    def detect_anomalies(data_list: List[MarketData], threshold: float = 3.0) -> List[Tuple[int, str]]:
        """
        检测异常数据点

        Args:
            data_list: MarketData列表
            threshold: 异常检测阈值（标准差倍数）

        Returns:
            异常列表 [(索引, 描述)]
        """
        anomalies = []

        if len(data_list) < 10:
            return anomalies

        # 1. 价格跳变检测
        returns = []
        for i in range(1, len(data_list)):
            if data_list[i-1].close > 0:
                ret = (data_list[i].close - data_list[i-1].close) / data_list[i-1].close
                returns.append((i, ret))

        if returns:
            ret_values = [r[1] for r in returns]
            mean_ret = np.mean(ret_values)
            std_ret = np.std(ret_values)

            for idx, ret in returns:
                if abs(ret - mean_ret) > threshold * std_ret:
                    anomalies.append((
                        idx,
                        f"价格异常跳变: {ret*100:.2f}% (阈值: {threshold}σ)"
                    ))

        # 2. 成交量异常检测
        volumes = [(i, md.volume) for i, md in enumerate(data_list) if md.volume]
        if volumes:
            vol_values = [v[1] for v in volumes]
            mean_vol = np.mean(vol_values)
            std_vol = np.std(vol_values)

            for idx, vol in volumes:
                if abs(vol - mean_vol) > threshold * std_vol:
                    anomalies.append((
                        idx,
                        f"成交量异常: {vol:.0f} (均值: {mean_vol:.0f})"
                    ))

        return anomalies

    @staticmethod
    def get_trading_hours_stats(data_list: List[MarketData]) -> Dict[str, Any]:
        """
        获取交易时段统计

        Args:
            data_list: MarketData列表

        Returns:
            时段统计信息
        """
        # 按交易状态分组
        status_counts = {}
        status_names = {
            83: '开市前',
            67: '集合竞价',
            84: '连续交易',
            69: '闭市',
            66: '休市',
        }

        for md in data_list:
            if md.trading_status is not None:
                status = md.trading_status
                if status not in status_counts:
                    status_counts[status] = 0
                status_counts[status] += 1

        # 格式化输出
        status_stats = {}
        for status, count in status_counts.items():
            name = status_names.get(status, f'未知({status})')
            status_stats[name] = {
                'count': count,
                'percentage': count / len(data_list) * 100,
            }

        return {
            'total_records': len(data_list),
            'status_breakdown': status_stats,
        }


class DataPreprocessor:
    """
    数据预处理器

    提供数据清洗、重采样、特征工程等功能
    """

    @staticmethod
    def filter_trading_hours(
        data_list: List[MarketData],
        status_filter: List[int] = None
    ) -> List[MarketData]:
        """
        过滤交易时段

        Args:
            data_list: 原始数据
            status_filter: 保留的交易状态列表（默认：[84] 连续交易）

        Returns:
            过滤后的数据
        """
        if status_filter is None:
            status_filter = [84]  # 默认只保留连续交易时段

        filtered = []
        for md in data_list:
            if md.trading_status in status_filter:
                filtered.append(md)

        return filtered

    @staticmethod
    def resample(
        data_list: List[MarketData],
        freq: int
    ) -> List[MarketData]:
        """
        重采样（降采样）

        Args:
            data_list: 原始数据
            freq: 采样频率（每N条取1条）

        Returns:
            重采样后的数据
        """
        return [data_list[i] for i in range(0, len(data_list), freq)]

    @staticmethod
    def fill_missing_values(data_list: List[MarketData]) -> List[MarketData]:
        """
        填充缺失值

        Args:
            data_list: 原始数据

        Returns:
            填充后的数据
        """
        # 简单实现：用前一个值填充
        filled = []

        for i, md in enumerate(data_list):
            # 如果价格为0或缺失，用前一个值
            if i > 0 and (md.close == 0 or md.close is None):
                # 创建副本并填充
                from copy import deepcopy
                md_copy = deepcopy(md)
                md_copy.close = filled[-1].close
                md_copy.open = filled[-1].open
                md_copy.high = filled[-1].high
                md_copy.low = filled[-1].low
                filled.append(md_copy)
            else:
                filled.append(md)

        return filled

    @staticmethod
    def add_technical_features(data_list: List[MarketData]) -> List[MarketData]:
        """
        添加技术指标特征（存储在extra_features中）

        暂未实现，留待扩展

        Args:
            data_list: 原始数据

        Returns:
            添加特征后的数据
        """
        # TODO: 添加MA, EMA, RSI, MACD等技术指标
        return data_list


def print_data_summary(data_list: List[MarketData]) -> None:
    """
    打印数据摘要

    Args:
        data_list: MarketData列表
    """
    print("=" * 60)
    print("数据摘要")
    print("=" * 60)

    # 验证
    validation = DataValidator.validate_data_list(data_list)
    print(f"\n✓ 数据验证: {'通过' if validation['valid'] else '失败'}")
    print(f"  - 总记录数: {validation['total_records']}")
    print(f"  - 无效记录: {validation['invalid_records']}")
    print(f"  - 问题数: {validation['total_issues']}")
    print(f"  - 警告数: {validation['total_warnings']}")

    # 统计
    stats = DataAnalyzer.get_basic_stats(data_list)
    if stats:
        print(f"\n✓ 时间范围: {stats['time_range']['start']} ~ {stats['time_range']['end']}")
        print(f"\n✓ 价格统计:")
        print(f"  - 起始/结束: {stats['price']['start']:.4f} / {stats['price']['end']:.4f}")
        print(f"  - 涨跌幅: {stats['price']['change']:.2f}%")
        print(f"  - 最低/最高: {stats['price']['min']:.4f} / {stats['price']['max']:.4f}")
        print(f"  - 均值±标准差: {stats['price']['mean']:.4f} ± {stats['price']['std']:.4f}")

        print(f"\n✓ 成交量统计:")
        print(f"  - 总量: {stats['volume']['total']:.0f}")
        print(f"  - 均值±标准差: {stats['volume']['mean']:.0f} ± {stats['volume']['std']:.0f}")

        print(f"\n✓ 收益率统计:")
        print(f"  - 均值: {stats['returns']['mean']*100:.4f}%")
        print(f"  - 波动率: {stats['returns']['std']*100:.4f}%")
        print(f"  - 最小/最大: {stats['returns']['min']*100:.2f}% / {stats['returns']['max']*100:.2f}%")

        if 'spread' in stats:
            print(f"\n✓ 买卖价差统计:")
            print(f"  - 均值: {stats['spread']['mean_bps']:.2f} bps")
            print(f"  - 最小/最大: {stats['spread']['min_bps']:.2f} / {stats['spread']['max_bps']:.2f} bps")

    # 异常检测
    anomalies = DataAnalyzer.detect_anomalies(data_list)
    if anomalies:
        print(f"\n⚠ 检测到 {len(anomalies)} 个异常数据点:")
        for idx, desc in anomalies[:5]:  # 只显示前5个
            print(f"  - Index {idx}: {desc}")
        if len(anomalies) > 5:
            print(f"  ... 还有 {len(anomalies) - 5} 个异常")

    # Level-2 信息
    if data_list[0].bid_prices is not None:
        print(f"\n✓ Level-2数据: 是 (10档行情)")
    else:
        print(f"\n✓ Level-2数据: 否")

    print("=" * 60)
