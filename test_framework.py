"""
框架快速测试脚本

验证所有模块是否正常工作
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np

print("="*60)
print("RL Trading Framework 测试")
print("="*60)

# 1. 测试数据模块
print("\n[1/9] 测试数据模块...")
from rl_trading_framework.data import SyntheticDataGenerator

data_gen = SyntheticDataGenerator(initial_price=100.0, sigma=0.2)
data = data_gen.load_data("TEST", num_steps=200, seed=42)
print(f"  ✓ 生成了 {len(data)} 条市场数据")
print(f"  ✓ 第一条数据: close={data[0].close:.2f}, volume={data[0].volume:.0f}")

# 2. 测试执行引擎
print("\n[2/9] 测试执行引擎...")
from rl_trading_framework.execution import SimpleExecutionEngine
from rl_trading_framework.core.types import OrderInfo, OrderSide, OrderType

engine = SimpleExecutionEngine()
order = OrderInfo(
    order_id="test_001",
    symbol="TEST",
    side=OrderSide.BUY,
    order_type=OrderType.MARKET,
    quantity=1000,
)
executed_qty, exec_price, updated_order = engine.execute_order(order, data[0])
print(f"  ✓ 订单执行成功: 数量={executed_qty:.2f}, 价格={exec_price:.2f}")

# 3. 测试奖励函数
print("\n[3/9] 测试奖励函数...")
from rl_trading_framework.rewards import ImplementationShortfallReward
from rl_trading_framework.core.types import State, Action, Observation

reward_fn = ImplementationShortfallReward()
obs = Observation(
    market_data=data[0],
    historical_prices=np.array([d.close for d in data[:20]]),
    target_quantity=5000,
    time_remaining=50,
)
state = State(observation=obs)
action = Action(trade_ratio=0.1)
next_state = State(observation=obs)

reward = reward_fn.calculate(state, action, next_state, 100, 100.5)
print(f"  ✓ 奖励计算成功: reward={reward:.4f}")

# 4. 测试环境
print("\n[4/9] 测试交易环境...")
from rl_trading_framework.environments import TradingEnvironment

env = TradingEnvironment(
    data_loader=data_gen,
    execution_engine=engine,
    reward_function=reward_fn,
    target_quantity=10000,
    max_steps=50,
)
obs = env.reset(seed=42)
print(f"  ✓ 环境重置成功")
print(f"  ✓ 观察空间维度: {env.observation_space_shape}")
print(f"  ✓ 动作空间维度: {env.action_space_shape}")

# 5. 测试策略网络
print("\n[5/9] 测试策略网络...")
from rl_trading_framework.policies import MLPPolicy

obs_dim = env.observation_space_shape[0]
action_dim = env.action_space_shape[0]

policy = MLPPolicy(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=(64, 64))
obs_array = obs.to_array()
action_array = policy.forward(obs_array)
print(f"  ✓ 策略网络创建成功")
print(f"  ✓ 输入维度: {obs_array.shape}, 输出维度: {action_array.shape}")

# 6. 测试智能体
print("\n[6/9] 测试智能体...")
from rl_trading_framework.agents import DQNAgent

agent = DQNAgent(policy=policy, buffer_size=1000, batch_size=32)
action = agent.select_action(obs, training=True)
print(f"  ✓ 智能体创建成功")
print(f"  ✓ 动作选择成功: trade_ratio={action.trade_ratio:.3f}")

# 7. 测试环境交互
print("\n[7/9] 测试环境交互...")
next_obs, reward, done, info = env.step(action)
print(f"  ✓ 环境step成功")
print(f"  ✓ reward={reward:.4f}, done={done}")
print(f"  ✓ info: {info}")

# 8. 测试训练器
print("\n[8/9] 测试训练器...")
from rl_trading_framework.trainers import RLTrainer

trainer = RLTrainer(env=env, agent=agent, save_dir="test_models")
print(f"  ✓ 训练器创建成功")

# 运行几个episode测试
print("  ✓ 运行3个训练episodes...")
history = trainer.train(num_episodes=3)
print(f"  ✓ 训练完成，奖励: {history['episode_rewards']}")

# 9. 测试评估器
print("\n[9/9] 测试评估器...")
from rl_trading_framework.evaluators import TradingEvaluator

evaluator = TradingEvaluator()
metrics = evaluator.evaluate_episode(agent, env)
print(f"  ✓ 评估完成")
print(f"  ✓ 总奖励: {metrics.total_reward:.4f}")
print(f"  ✓ 完成率: {metrics.completion_rate:.2%}")

# 完成
print("\n" + "="*60)
print("✓ 所有测试通过！框架工作正常。")
print("="*60)
print("\n下一步:")
print("  1. 运行完整训练示例: python rl_trading_framework/examples/simple_training.py")
print("  2. 查看README.md了解详细使用方法")
print("  3. 查看docs/MODULE_DOCUMENTATION.md了解模块详情")
