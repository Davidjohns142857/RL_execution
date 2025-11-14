"""
指标计算工具
"""

import numpy as np
from typing import List


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    计算夏普比率

    Args:
        returns: 收益率数组
        risk_free_rate: 无风险利率（年化）
        periods_per_year: 每年的期数（日频为252）

    Returns:
        夏普比率
    """
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - risk_free_rate / periods_per_year
    if np.std(excess_returns) == 0:
        return 0.0

    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods_per_year)
    return float(sharpe)


def calculate_max_drawdown(prices: np.ndarray) -> float:
    """
    计算最大回撤

    Args:
        prices: 价格序列

    Returns:
        最大回撤（百分比）
    """
    if len(prices) == 0:
        return 0.0

    cumulative_max = np.maximum.accumulate(prices)
    drawdowns = (prices - cumulative_max) / cumulative_max
    max_dd = np.min(drawdowns)

    return float(max_dd)


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    计算索提诺比率（仅考虑下行风险）

    Args:
        returns: 收益率数组
        risk_free_rate: 无风险利率
        periods_per_year: 每年期数

    Returns:
        索提诺比率
    """
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - risk_free_rate / periods_per_year

    # 只考虑负收益
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0:
        return float('inf')

    downside_std = np.std(downside_returns)
    if downside_std == 0:
        return 0.0

    sortino = np.mean(excess_returns) / downside_std * np.sqrt(periods_per_year)
    return float(sortino)


def calculate_calmar_ratio(
    returns: np.ndarray,
    prices: np.ndarray,
    periods_per_year: int = 252,
) -> float:
    """
    计算卡玛比率（收益率/最大回撤）

    Args:
        returns: 收益率数组
        prices: 价格序列
        periods_per_year: 每年期数

    Returns:
        卡玛比率
    """
    if len(returns) == 0 or len(prices) == 0:
        return 0.0

    annual_return = np.mean(returns) * periods_per_year
    max_dd = abs(calculate_max_drawdown(prices))

    if max_dd == 0:
        return 0.0

    calmar = annual_return / max_dd
    return float(calmar)
