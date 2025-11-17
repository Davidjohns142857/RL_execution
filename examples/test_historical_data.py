"""
真实历史数据加载测试示例

展示如何：
1. 加载真实的Level-2市场数据
2. 验证数据质量
3. 查看数据统计
4. 使用真实数据进行简单回测

使用前请设置数据路径！
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl_trading_framework.data.historical_data import HistoricalDataLoader
from rl_trading_framework.data.data_utils import (
    DataValidator,
    DataAnalyzer,
    DataPreprocessor,
    print_data_summary
)


def test_data_loading():
    """测试数据加载功能"""
    print("\n" + "=" * 60)
    print("测试1: 数据加载")
    print("=" * 60)

    # ==================== 重要：设置你的数据路径 ====================
    # 请将下面的路径改为你的实际数据路径
    DATA_DIR = "/path/to/your/data/market_data"  # 修改为你的数据路径
    SYMBOL = "000001"  # 股票代码
    DATE = "20170105"  # 日期 YYYYMMDD格式

    # 检查路径是否存在
    if not os.path.exists(DATA_DIR):
        print(f"\n⚠️  数据路径不存在: {DATA_DIR}")
        print("\n请按以下步骤操作：")
        print("1. 编辑此文件，修改 DATA_DIR 为你的真实数据路径")
        print("2. 确保数据文件格式为: {symbol}_{date}_market_data.csv")
        print("3. 例如: 000001_20170105_market_data.csv")
        print("\n示例数据路径设置:")
        print("  DATA_DIR = '/home/user/market_data'")
        print("  SYMBOL = '000001'")
        print("  DATE = '20170105'")
        print("\n运行前请确保:")
        print("  - 数据文件存在于 {DATA_DIR}/{symbol}_{date}_market_data.csv")
        print("  - CSV文件包含必要列: datetime, last_prc, volume")
        return None

    try:
        # 创建数据加载器
        print(f"\n正在加载数据...")
        print(f"  - 数据目录: {DATA_DIR}")
        print(f"  - 股票代码: {SYMBOL}")
        print(f"  - 日期: {DATE}")

        loader = HistoricalDataLoader(
            data_dir=DATA_DIR,
            default_symbol=SYMBOL,
            default_date=DATE,
            trading_hours_only=True,  # 只加载交易时段数据
            subsample_freq=None,  # 不降采样
        )

        # 加载数据
        data = loader.load_data(
            symbol=SYMBOL,
            date=DATE,
            # num_steps=1000,  # 可选：只加载前1000条
        )

        print(f"\n✓ 成功加载 {len(data)} 条数据")

        # 显示前3条数据
        print(f"\n前3条数据示例:")
        for i, md in enumerate(data[:3]):
            print(f"\n  [{i}] 时间: {md.datetime_str}")
            print(f"      价格: {md.close:.4f}, 成交量: {md.volume:.0f}")
            print(f"      买一: {md.bid_price:.4f} x {md.bid_volume:.0f}")
            print(f"      卖一: {md.ask_price:.4f} x {md.ask_volume:.0f}")
            if md.bid_prices:
                print(f"      10档买盘: {[f'{p:.4f}' for p in md.bid_prices[:5]]}")
                print(f"      10档卖盘: {[f'{p:.4f}' for p in md.ask_prices[:5]]}")

        return data

    except FileNotFoundError as e:
        print(f"\n❌ 文件未找到: {e}")
        print("\n请检查:")
        print(f"  1. 数据文件是否存在: {DATA_DIR}/{SYMBOL}_{DATE}_market_data.csv")
        print(f"  2. 文件名格式是否正确")
        return None

    except Exception as e:
        print(f"\n❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_data_validation(data):
    """测试数据验证"""
    if data is None:
        print("\n⚠️  跳过验证测试（数据未加载）")
        return

    print("\n" + "=" * 60)
    print("测试2: 数据验证")
    print("=" * 60)

    # 验证单条数据
    print("\n验证第一条数据:")
    result = DataValidator.validate_market_data(data[0])
    print(f"  - 有效: {result['valid']}")
    if result['issues']:
        print(f"  - 问题: {result['issues']}")
    if result['warnings']:
        print(f"  - 警告: {result['warnings']}")

    # 验证整个数据集
    print("\n验证整个数据集:")
    result = DataValidator.validate_data_list(data)
    print(f"  - 总记录: {result['total_records']}")
    print(f"  - 无效记录: {result['invalid_records']}")
    print(f"  - 问题数: {result['total_issues']}")
    print(f"  - 警告数: {result['total_warnings']}")

    if result['time_issues']:
        print(f"  - 时间问题: {len(result['time_issues'])} 个")

    if result['valid']:
        print("\n✓ 数据验证通过！")
    else:
        print("\n⚠️  发现数据问题，请检查")


def test_data_analysis(data):
    """测试数据分析"""
    if data is None:
        print("\n⚠️  跳过分析测试（数据未加载）")
        return

    print("\n" + "=" * 60)
    print("测试3: 数据统计分析")
    print("=" * 60)

    # 使用内置的打印函数
    print_data_summary(data)

    # 异常检测
    print("\n检测异常数据点...")
    anomalies = DataAnalyzer.detect_anomalies(data, threshold=3.0)
    if anomalies:
        print(f"  发现 {len(anomalies)} 个异常数据点（前5个）:")
        for idx, desc in anomalies[:5]:
            print(f"    - Index {idx}: {desc}")
    else:
        print("  ✓ 未发现异常数据点")

    # 交易时段统计
    print("\n交易时段统计:")
    hours_stats = DataAnalyzer.get_trading_hours_stats(data)
    for status_name, stats in hours_stats['status_breakdown'].items():
        print(f"  - {status_name}: {stats['count']} 条 ({stats['percentage']:.1f}%)")


def test_data_preprocessing(data):
    """测试数据预处理"""
    if data is None:
        print("\n⚠️  跳过预处理测试（数据未加载）")
        return

    print("\n" + "=" * 60)
    print("测试4: 数据预处理")
    print("=" * 60)

    # 重采样
    print("\n降采样测试（每10条取1条）:")
    resampled = DataPreprocessor.resample(data, freq=10)
    print(f"  - 原始数据: {len(data)} 条")
    print(f"  - 降采样后: {len(resampled)} 条")
    print(f"  - 压缩比: {len(data)/len(resampled):.1f}x")

    # 过滤交易时段
    print("\n过滤交易时段（只保留连续交易）:")
    filtered = DataPreprocessor.filter_trading_hours(data, status_filter=[84])
    print(f"  - 原始数据: {len(data)} 条")
    print(f"  - 过滤后: {len(filtered)} 条")


def test_simple_backtest(data):
    """测试简单回测"""
    if data is None:
        print("\n⚠️  跳过回测测试（数据未加载）")
        return

    print("\n" + "=" * 60)
    print("测试5: 简单回测示例")
    print("=" * 60)

    from rl_trading_framework.environments.trading_env import TradingEnvironment
    from rl_trading_framework.execution.simple_execution import SimpleExecutionEngine
    from rl_trading_framework.rewards.implementation_shortfall import ImplementationShortfallReward
    from rl_trading_framework.core.types import Action

    # 创建简单的数据加载器包装
    class SimpleDataLoader:
        def __init__(self, data):
            self.data = data

        def load_data(self, symbol, **kwargs):
            return self.data

    # 创建环境
    print("\n创建交易环境...")
    env = TradingEnvironment(
        data_loader=SimpleDataLoader(data),
        execution_engine=SimpleExecutionEngine(),
        reward_function=ImplementationShortfallReward(),
        target_quantity=10000.0,  # 目标交易10000股
        max_steps=min(20, len(data) // 2),  # 最多20步
        initial_cash=1000000.0,
        side="buy",
        window_size=10,
    )

    # 重置环境
    print("重置环境...")
    obs = env.reset()

    # 运行简单的均匀执行策略
    print("\n运行均匀执行策略（TWAP）...")
    total_reward = 0.0
    step = 0

    while step < env.max_steps:
        # 均匀执行：每步执行1/剩余步数
        remaining_steps = env.max_steps - step
        trade_ratio = 1.0 / remaining_steps if remaining_steps > 0 else 0.0

        action = Action(
            trade_ratio=trade_ratio,
            urgency=0.5,
        )

        obs, reward, done, info = env.step(action)
        total_reward += reward
        step += 1

        if step <= 3 or done:  # 只打印前3步和最后一步
            print(f"  步骤 {step}:")
            print(f"    - 执行数量: {info['executed_quantity']:.0f}")
            print(f"    - 执行价格: {info['execution_price']:.4f}")
            print(f"    - 完成率: {info['completion_rate']*100:.1f}%")
            print(f"    - 奖励: {reward:.6f}")

        if done:
            break

    # 显示最终结果
    print(f"\n回测结果:")
    print(f"  - 总步数: {step}")
    print(f"  - 总奖励: {total_reward:.6f}")
    metrics = env.get_metrics()
    print(f"  - 完成率: {metrics['completion_rate']*100:.1f}%")
    print(f"  - 平均执行价格: {metrics['avg_execution_price']:.4f}")
    print(f"  - VWAP: {metrics['vwap']:.4f}")
    print(f"  - VWAP滑点: {metrics['vwap_slippage']*10000:.2f} bps")
    print(f"  - Implementation Shortfall: {metrics['implementation_shortfall']:.2f}")


def main():
    """主函数"""
    print("=" * 60)
    print("真实历史数据加载测试")
    print("=" * 60)

    # 测试1: 加载数据
    data = test_data_loading()

    if data is None:
        print("\n" + "=" * 60)
        print("测试终止：请先配置数据路径")
        print("=" * 60)
        print("\n配置步骤:")
        print("1. 编辑本文件，找到 DATA_DIR 变量")
        print("2. 设置为你的真实数据路径")
        print("3. 设置 SYMBOL 和 DATE")
        print("4. 重新运行测试")
        return

    # 测试2: 验证
    test_data_validation(data)

    # 测试3: 分析
    test_data_analysis(data)

    # 测试4: 预处理
    test_data_preprocessing(data)

    # 测试5: 简单回测
    test_simple_backtest(data)

    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)
    print("\n下一步:")
    print("1. 检查数据质量和统计信息")
    print("2. 如果数据正常，可以开始训练RL模型")
    print("3. 参考 example_real_data_training.py 进行完整训练")


if __name__ == "__main__":
    main()
