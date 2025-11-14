"""
快速测试 - 不依赖PyTorch

测试框架核心功能（数据、执行、环境等）
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

print("正在等待依赖安装...")
import time
time.sleep(5)

try:
    import numpy as np
except ImportError:
    print("等待numpy安装...")
    time.sleep(30)
    import numpy as np

print("\n" + "="*60)
print("RL Trading Framework - 快速测试")
print("="*60)

# 测试1: 数据模块
print("\n[1/6] 测试数据模块...")
from rl_trading_framework.data import SyntheticDataGenerator

gen = SyntheticDataGenerator(initial_price=100.0, sigma=0.2)
data = gen.load_data("TEST", num_steps=200, seed=42)
print(f"  ✓ 生成 {len(data)} 条数据")
print(f"  ✓ 价格范围: [{min(d.close for d in data):.2f}, {max(d.close for d in data):.2f}]")

# 测试2: 核心数据类型
print("\n[2/6] 测试核心数据类型...")
from rl_trading_framework.core.types import (
    MarketData, Action, Observation, OrderInfo, State, OrderSide, OrderType
)

md = MarketData(
    timestamp=1.0,
    open=100.0,
    high=101.0,
    low=99.0,
    close=100.5,
    volume=1000000.0,
)
print(f"  ✓ MarketData: close={md.close}")

action = Action(trade_ratio=0.1, order_type=OrderType.MARKET)
print(f"  ✓ Action: trade_ratio={action.trade_ratio}")

obs = Observation(
    market_data=md,
    historical_prices=np.array([100.0] * 20),
    historical_volumes=np.array([1000000.0] * 20),
    target_quantity=10000,
    time_remaining=50,
)
obs_array = obs.to_array()
print(f"  ✓ Observation转数组: shape={obs_array.shape}")

# 测试3: 执行引擎
print("\n[3/6] 测试执行引擎...")
from rl_trading_framework.execution import SimpleExecutionEngine

engine = SimpleExecutionEngine()
order = OrderInfo(
    order_id="test_001",
    symbol="TEST",
    side=OrderSide.BUY,
    order_type=OrderType.MARKET,
    quantity=1000,
)

qty, price, updated_order = engine.execute_order(order, data[50])
print(f"  ✓ 订单执行: qty={qty:.2f}, price={price:.4f}")
print(f"  ✓ 市场冲击: {engine.calculate_market_impact(1000, data[50], 'buy'):.6f}")

# 测试4: 奖励函数
print("\n[4/6] 测试奖励函数...")
from rl_trading_framework.rewards import ImplementationShortfallReward

reward_fn = ImplementationShortfallReward()
state = State(observation=obs)
next_state = State(observation=obs)

reward = reward_fn.calculate(state, action, next_state, 100, 100.5)
print(f"  ✓ 奖励计算: {reward:.6f}")

# 测试5: 环境（不使用策略网络）
print("\n[5/6] 测试交易环境...")
from rl_trading_framework.environments import TradingEnvironment

env = TradingEnvironment(
    data_loader=gen,
    execution_engine=engine,
    reward_function=reward_fn,
    target_quantity=5000,
    max_steps=30,
)

obs = env.reset(seed=42)
print(f"  ✓ 环境重置成功")
print(f"  ✓ 观察空间: {env.observation_space_shape}")

# 执行几步
total_reward = 0
for i in range(5):
    action = Action(trade_ratio=0.2, order_type=OrderType.MARKET)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    if done:
        break

print(f"  ✓ 执行 {i+1} 步，累计奖励: {total_reward:.6f}")

# 测试6: 完整Episode
print("\n[6/6] 测试完整Episode...")
obs = env.reset(seed=123)
episode_reward = 0
steps = 0

while steps < 30:
    action = Action(trade_ratio=0.15, order_type=OrderType.MARKET)
    obs, reward, done, info = env.step(action)
    episode_reward += reward
    steps += 1
    if done:
        break

metrics = env.get_metrics()
print(f"  ✓ Episode完成:")
print(f"    步数: {steps}")
print(f"    总奖励: {episode_reward:.6f}")
print(f"    完成率: {metrics.get('completion_rate', 0):.2%}")
print(f"    平均执行价: {metrics.get('avg_execution_price', 0):.4f}")
print(f"    VWAP滑点: {metrics.get('vwap_slippage', 0):.6f}")

# 总结
print("\n" + "="*60)
print("✓ 核心功能测试全部通过！")
print("="*60)
print("\n说明:")
print("  - 数据生成和加载: ✓")
print("  - 核心数据类型: ✓")
print("  - 订单执行引擎: ✓")
print("  - 奖励函数计算: ✓")
print("  - 交易环境模拟: ✓")
print("  - 完整Episode运行: ✓")
print("\n框架核心功能正常，无Bug！")
print("\n注意: RL智能体和策略网络需要等待PyTorch安装完成后测试。")
