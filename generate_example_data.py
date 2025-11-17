"""
生成示例Level-2市场数据CSV文件

模拟中国A股的10档行情数据
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_market_data_csv(
    symbol="000001",
    date="20170105",
    num_records=500,
    output_dir="example_data/market_data"
):
    """
    生成示例market_data CSV文件

    Args:
        symbol: 股票代码
        date: 日期 YYYYMMDD
        num_records: 记录数
        output_dir: 输出目录
    """

    # 基础参数
    base_price = 10.0
    prev_close = 9.95

    # 生成时间序列（交易时段：09:30-15:00）
    start_time = datetime.strptime(f"{date} 09:30:00", "%Y%m%d %H:%M:%S")

    # 交易时段分布
    # 上午：09:30-11:30 (120分钟)
    # 下午：13:00-15:00 (120分钟)
    morning_records = num_records // 2
    afternoon_records = num_records - morning_records

    times = []
    # 上午
    for i in range(morning_records):
        t = start_time + timedelta(seconds=i * 120.0 / morning_records * 60)
        times.append(t)
    # 下午
    afternoon_start = datetime.strptime(f"{date} 13:00:00", "%Y%m%d %H:%M:%S")
    for i in range(afternoon_records):
        t = afternoon_start + timedelta(seconds=i * 120.0 / afternoon_records * 60)
        times.append(t)

    # 生成价格序列（GBM）
    returns = np.random.normal(0.0001, 0.002, num_records)
    price_path = base_price * np.cumprod(1 + returns)

    # 计算OHLC
    opens = price_path.copy()
    highs = price_path * (1 + np.abs(np.random.normal(0, 0.001, num_records)))
    lows = price_path * (1 - np.abs(np.random.normal(0, 0.001, num_records)))
    closes = price_path

    # 涨跌停价（10%）
    high_limited = prev_close * 1.10
    low_limited = prev_close * 0.90

    # 生成数据
    data = []

    cumulative_volume = 0
    cumulative_turnover = 0.0
    cumulative_trades = 0

    for i, t in enumerate(times):
        # 基础信息
        datetime_str = t.strftime("%Y-%m-%d %H:%M:%S")
        date_str = date
        time_str = t.strftime("%H%M%S") + "000"  # 加3位毫秒

        # 时间戳（微秒）
        timestamp = int(t.timestamp())
        exchtime = timestamp * 1000000  # 转微秒
        localtime = exchtime + np.random.randint(-1000, 1000)  # 略微延迟

        # 价格
        last_prc = closes[i]
        open_prc = opens[i]
        high_prc = highs[i]
        low_prc = lows[i]

        # 确保价格在涨跌停范围内
        last_prc = np.clip(last_prc, low_limited, high_limited)
        high_prc = np.clip(high_prc, low_limited, high_limited)
        low_prc = np.clip(low_prc, low_limited, high_limited)

        # 交易状态（84=连续交易）
        if t.hour < 9 or (t.hour == 9 and t.minute < 30):
            status = 83  # 开市前
        elif t.hour >= 15:
            status = 69  # 闭市
        else:
            status = 84  # 连续交易

        # 成交量和成交额
        volume_increment = np.random.randint(10000, 100000)
        cumulative_volume += volume_increment
        turnover_increment = volume_increment * last_prc
        cumulative_turnover += turnover_increment
        cumulative_trades += np.random.randint(10, 100)

        # 生成10档买卖盘
        # 买盘：价格递减
        bid_prc1 = last_prc - 0.01
        bid_prices = [bid_prc1 - i * 0.01 for i in range(10)]
        bid_volumes = [np.random.randint(1000, 10000) for _ in range(10)]

        # 卖盘：价格递增
        ask_prc1 = last_prc + 0.01
        ask_prices = [ask_prc1 + i * 0.01 for i in range(10)]
        ask_volumes = [np.random.randint(1000, 10000) for _ in range(10)]

        # 加权价格
        total_bid_vol = sum(bid_volumes)
        total_ask_vol = sum(ask_volumes)
        weighted_bid_prc = sum(p * v for p, v in zip(bid_prices, bid_volumes)) / total_bid_vol
        weighted_ask_prc = sum(p * v for p, v in zip(ask_prices, ask_volumes)) / total_ask_vol

        # 构建记录
        record = {
            'symbol': symbol,
            'datetime': datetime_str,
            'date': date_str,
            'time': time_str,
            'exchtime': exchtime,
            'localtime': localtime,
            'last_prc': round(last_prc, 2),
            'status': status,
            'prev_close': round(prev_close, 2),
            'open': round(open_prc, 2),
            'high': round(high_prc, 2),
            'low': round(low_prc, 2),
            'high_limited': round(high_limited, 2),
            'low_limited': round(low_limited, 2),
            'volume': cumulative_volume,
            'turnover': round(cumulative_turnover, 2),
            'num_trades': cumulative_trades,
            'weighted_ask_prc': round(weighted_ask_prc, 2),
            'weighted_bid_prc': round(weighted_bid_prc, 2),
            'total_ask_vol': total_ask_vol,
            'total_bid_vol': total_bid_vol,
        }

        # 添加10档买卖盘
        for j in range(10):
            record[f'ask_prc{j+1}'] = round(ask_prices[j], 2)
            record[f'ask_vol{j+1}'] = ask_volumes[j]
            record[f'bid_prc{j+1}'] = round(bid_prices[j], 2)
            record[f'bid_vol{j+1}'] = bid_volumes[j]

        # num字段（无特殊含义）
        record['num'] = 0

        data.append(record)

    # 创建DataFrame
    df = pd.DataFrame(data)

    # 保存CSV
    import os
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{symbol}_{date}_market_data.csv"
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)

    print(f"✓ 生成示例数据文件: {filepath}")
    print(f"  - 记录数: {len(df)}")
    print(f"  - 时间范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
    print(f"  - 价格范围: {df['last_prc'].min():.2f} ~ {df['last_prc'].max():.2f}")
    print(f"  - 总成交量: {df['volume'].iloc[-1]:,}")
    print(f"  - 总成交额: {df['turnover'].iloc[-1]:,.2f}")

    return filepath


if __name__ == "__main__":
    # 生成示例数据
    filepath = generate_market_data_csv(
        symbol="000001",
        date="20170105",
        num_records=500,
        output_dir="example_data/market_data"
    )

    print(f"\n示例数据已生成！")
    print(f"文件路径: {filepath}")
    print(f"\n现在可以运行测试:")
    print(f"  python examples/test_historical_data.py")
