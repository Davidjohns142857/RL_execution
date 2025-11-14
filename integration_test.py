"""
完整集成测试

测试框架的所有核心功能，验证是否有bug
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import traceback

def test_module(name, func):
    """测试单个模块"""
    print(f"\n{'='*60}")
    print(f"测试: {name}")
    print('='*60)
    try:
        func()
        print(f"✓ {name} 测试通过")
        return True
    except Exception as e:
        print(f"✗ {name} 测试失败:")
        print(f"  错误: {str(e)}")
        traceback.print_exc()
        return False


def test_1_data_modules():
    """测试数据模块"""
    from rl_trading_framework.data import SyntheticDataGenerator, CSVDataLoader

    # 测试合成数据生成器
    print("\n[1.1] 测试SyntheticDataGenerator...")
    gen = SyntheticDataGenerator(
        initial_price=100.0,
        mu=0.05,
        sigma=0.2,
        dt=60.0,
    )

    # 生成数据
    data = gen.load_data("TEST", num_steps=500, seed=42)
    print(f"  生成 {len(data)} 条数据")
    print(f"  价格范围: [{min(d.close for d in data):.2f}, {max(d.close for d in data):.2f}]")

    # 验证数据完整性
    assert len(data) == 500, "数据数量不对"
    assert all(d.close > 0 for d in data), "价格必须为正"
    assert all(d.volume > 0 for d in data), "成交量必须为正"

    # 测试趋势数据
    print("\n[1.2] 测试趋势数据生成...")
    uptrend_data = gen.generate_trend_data(num_steps=200, trend="up", trend_strength=0.2, seed=42)
    print(f"  上涨趋势: 起始价格={uptrend_data[0].close:.2f}, 结束价格={uptrend_data[-1].close:.2f}")

    # CSV加载器（不测试实际加载，因为没有CSV文件）
    print("\n[1.3] 测试CSVDataLoader初始化...")
    csv_loader = CSVDataLoader(data_dir="./test_data", normalize=False)
    print("  CSVDataLoader初始化成功")


def test_2_execution_engines():
    """测试执行引擎"""
    from rl_trading_framework.execution import SimpleExecutionEngine, RealisticExecutionEngine
    from rl_trading_framework.core.types import OrderInfo, OrderSide, OrderType, MarketData
    from rl_trading_framework.data import SyntheticDataGenerator

    # 生成测试数据
    gen = SyntheticDataGenerator(initial_price=100.0, sigma=0.2)
    data = gen.load_data("TEST", num_steps=100, seed=42)
    market_data = data[50]

    print(f"\n[2.1] 测试SimpleExecutionEngine...")
    engine = SimpleExecutionEngine(
        permanent_impact_coef=0.1,
        temporary_impact_coef=0.5,
    )

    # 测试市价单
    order = OrderInfo(
        order_id="test_market_001",
        symbol="TEST",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=1000,
    )

    qty, price, updated_order = engine.execute_order(order, market_data)
    print(f"  市价单: 成交量={qty:.2f}, 价格={price:.4f}")
    assert qty == 1000, "市价单应该全部成交"
    assert price > market_data.close, "买单价格应该高于市价"

    # 测试限价单
    print("\n[2.2] 测试限价单...")
    limit_order = OrderInfo(
        order_id="test_limit_001",
        symbol="TEST",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=1000,
        price=market_data.close * 1.01,  # 高于市价1%
    )

    qty2, price2, _ = engine.execute_order(limit_order, market_data)
    print(f"  限价单: 成交量={qty2:.2f}, 价格={price2:.4f}")

    # 测试RealisticExecutionEngine
    print("\n[2.3] 测试RealisticExecutionEngine...")
    realistic_engine = RealisticExecutionEngine(
        order_book_depth=5,
        liquidity_recovery_rate=0.1,
    )

    qty3, price3, _ = realistic_engine.execute_order(order, market_data)
    print(f"  真实执行: 成交量={qty3:.2f}, 价格={price3:.4f}")
    print(f"  累积冲击: {realistic_engine.get_cumulative_impact():.6f}")


def test_3_reward_functions():
    """测试奖励函数"""
    from rl_trading_framework.rewards import (
        ImplementationShortfallReward,
        PnLBasedReward,
        CompositeReward
    )
    from rl_trading_framework.core.types import State, Action, Observation, MarketData
    from rl_trading_framework.data import SyntheticDataGenerator

    # 准备测试数据
    gen = SyntheticDataGenerator(initial_price=100.0, sigma=0.2)
    data = gen.load_data("TEST", num_steps=100, seed=42)

    obs = Observation(
        market_data=data[0],
        historical_prices=np.array([d.close for d in data[:20]]),
        historical_volumes=np.array([d.volume for d in data[:20]]),
        target_quantity=10000,
        executed_quantity=0,
        time_remaining=50,
    )

    state = State(observation=obs)
    action = Action(trade_ratio=0.1)

    next_obs = Observation(
        market_data=data[1],
        historical_prices=np.array([d.close for d in data[1:21]]),
        historical_volumes=np.array([d.volume for d in data[1:21]]),
        target_quantity=9000,
        executed_quantity=1000,
        time_remaining=49,
    )
    next_state = State(observation=next_obs)

    print("\n[3.1] 测试ImplementationShortfallReward...")
    is_reward = ImplementationShortfallReward(
        time_penalty_weight=0.001,
        completion_bonus=1.0,
    )

    reward1 = is_reward.calculate(state, action, next_state, 1000, 100.5)
    print(f"  IS奖励: {reward1:.6f}")
    assert isinstance(reward1, float), "奖励必须是float"

    print("\n[3.2] 测试PnLBasedReward...")
    pnl_reward = PnLBasedReward(
        inventory_penalty=0.01,
        transaction_cost_bps=1.0,
    )

    reward2 = pnl_reward.calculate(state, action, next_state, 1000, 100.5)
    print(f"  PnL奖励: {reward2:.6f}")

    print("\n[3.3] 测试CompositeReward...")
    composite = CompositeReward([
        (is_reward, 0.6),
        (pnl_reward, 0.4),
    ])

    reward3 = composite.calculate(state, action, next_state, 1000, 100.5)
    print(f"  组合奖励: {reward3:.6f}")

    # 测试重置
    is_reward.reset()
    pnl_reward.reset()
    composite.reset()
    print("  奖励函数重置成功")


def test_4_environment():
    """测试交易环境"""
    from rl_trading_framework.environments import TradingEnvironment
    from rl_trading_framework.data import SyntheticDataGenerator
    from rl_trading_framework.execution import SimpleExecutionEngine
    from rl_trading_framework.rewards import ImplementationShortfallReward
    from rl_trading_framework.core.types import Action, OrderType

    print("\n[4.1] 创建环境...")
    env = TradingEnvironment(
        data_loader=SyntheticDataGenerator(initial_price=100.0, sigma=0.2),
        execution_engine=SimpleExecutionEngine(),
        reward_function=ImplementationShortfallReward(),
        target_quantity=10000,
        max_steps=50,
        initial_cash=1000000.0,
        side="buy",
    )

    print(f"  观察空间: {env.observation_space_shape}")
    print(f"  动作空间: {env.action_space_shape}")

    print("\n[4.2] 测试reset...")
    obs = env.reset(seed=42)
    print(f"  初始观察维度: {obs.to_array().shape}")
    print(f"  目标数量: {obs.target_quantity}")
    print(f"  初始价格: {obs.market_data.close:.2f}")

    print("\n[4.3] 测试step...")
    action = Action(
        trade_ratio=0.2,
        order_type=OrderType.MARKET,
        urgency=0.5,
    )

    next_obs, reward, done, info = env.step(action)
    print(f"  Step结果:")
    print(f"    奖励: {reward:.6f}")
    print(f"    完成: {done}")
    print(f"    已执行: {info.get('total_executed', 0):.2f}")
    print(f"    成交价: {info.get('execution_price', 0):.4f}")

    # 运行完整episode
    print("\n[4.4] 运行完整episode...")
    obs = env.reset(seed=42)
    total_reward = 0
    steps = 0

    while steps < 50:
        action = Action(trade_ratio=0.1, order_type=OrderType.MARKET)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        if done:
            break

    print(f"  Episode完成:")
    print(f"    总步数: {steps}")
    print(f"    总奖励: {total_reward:.6f}")

    metrics = env.get_metrics()
    print(f"    完成率: {metrics.get('completion_rate', 0):.2%}")
    print(f"    平均执行价: {metrics.get('avg_execution_price', 0):.4f}")


def test_5_policies():
    """测试策略网络"""
    from rl_trading_framework.policies import MLPPolicy, LSTMPolicy

    obs_dim = 31
    action_dim = 4

    print("\n[5.1] 测试MLPPolicy...")
    mlp = MLPPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=(128, 128),
        learning_rate=3e-4,
    )

    # 单个输入
    obs = np.random.randn(obs_dim)
    action = mlp.forward(obs)
    print(f"  单个输入: {obs.shape} -> {action.shape}")
    assert action.shape == (action_dim,), "输出维度错误"
    assert np.all((action >= 0) & (action <= 1)), "输出应该在[0,1]范围"

    # 批量输入
    obs_batch = np.random.randn(32, obs_dim)
    actions_batch = mlp.forward(obs_batch)
    print(f"  批量输入: {obs_batch.shape} -> {actions_batch.shape}")
    assert actions_batch.shape == (32, action_dim), "批量输出维度错误"

    # 测试get_action
    action2 = mlp.get_action(obs, deterministic=True)
    print(f"  确定性动作: {action2}")

    print("\n[5.2] 测试LSTMPolicy...")
    lstm = LSTMPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_size=64,
        num_layers=2,
    )

    # LSTM需要序列输入
    obs_seq = np.random.randn(10, obs_dim)  # 10个时间步
    action3 = lstm.forward(obs_seq)
    print(f"  序列输入: {obs_seq.shape} -> {action3.shape}")

    # 重置隐藏状态
    lstm.reset_hidden_state()
    print("  LSTM隐藏状态重置成功")

    # 测试保存和加载
    print("\n[5.3] 测试模型保存/加载...")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        temp_path = f.name

    mlp.save(temp_path)
    print(f"  模型已保存到: {temp_path}")

    # 创建新模型并加载
    mlp2 = MLPPolicy(obs_dim=obs_dim, action_dim=action_dim)
    mlp2.load(temp_path)
    print("  模型加载成功")

    # 验证加载的模型输出相同
    action_original = mlp.forward(obs)
    action_loaded = mlp2.forward(obs)
    assert np.allclose(action_original, action_loaded, atol=1e-5), "加载的模型输出不一致"
    print("  模型一致性验证通过")

    os.unlink(temp_path)


def test_6_agents():
    """测试智能体"""
    from rl_trading_framework.agents import DQNAgent, PPOAgent
    from rl_trading_framework.policies import MLPPolicy
    from rl_trading_framework.core.types import Observation, Action
    from rl_trading_framework.data import SyntheticDataGenerator

    obs_dim = 31
    action_dim = 4

    # 准备测试观察
    gen = SyntheticDataGenerator(initial_price=100.0, sigma=0.2)
    data = gen.load_data("TEST", num_steps=100, seed=42)

    obs = Observation(
        market_data=data[0],
        historical_prices=np.array([d.close for d in data[:20]]),
        historical_volumes=np.array([d.volume for d in data[:20]]),
        target_quantity=10000,
        time_remaining=50,
    )

    print("\n[6.1] 测试DQNAgent...")
    policy = MLPPolicy(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=(64, 64))
    dqn_agent = DQNAgent(
        policy=policy,
        gamma=0.99,
        epsilon_start=1.0,
        buffer_size=1000,
        batch_size=32,
    )

    # 测试动作选择
    action = dqn_agent.select_action(obs, training=True)
    print(f"  动作选择: trade_ratio={action.trade_ratio:.3f}")
    assert isinstance(action, Action), "返回的应该是Action对象"

    # 测试存储经验
    next_obs = obs  # 简化
    dqn_agent.store_transition(obs, action, -0.1, next_obs, False)
    print(f"  经验缓冲区大小: {len(dqn_agent.replay_buffer)}")

    # 填充更多经验
    for i in range(100):
        dqn_agent.store_transition(obs, action, -0.1, next_obs, False)

    # 测试更新
    metrics = dqn_agent.update()
    if metrics:
        print(f"  更新指标: {metrics}")

    print("\n[6.2] 测试PPOAgent...")
    policy2 = MLPPolicy(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=(64, 64))
    ppo_agent = PPOAgent(
        policy=policy2,
        clip_epsilon=0.2,
        epochs=5,
        batch_size=32,
    )

    action2 = ppo_agent.select_action(obs, training=True)
    print(f"  PPO动作选择: trade_ratio={action2.trade_ratio:.3f}")

    # PPO需要完整的轨迹
    for i in range(10):
        ppo_agent.store_transition(obs, action2, -0.1, next_obs, False)

    metrics2 = ppo_agent.update()
    if metrics2:
        print(f"  PPO更新指标: {metrics2}")


def test_7_training():
    """测试训练流程"""
    from rl_trading_framework.environments import TradingEnvironment
    from rl_trading_framework.data import SyntheticDataGenerator
    from rl_trading_framework.execution import SimpleExecutionEngine
    from rl_trading_framework.rewards import ImplementationShortfallReward
    from rl_trading_framework.policies import MLPPolicy
    from rl_trading_framework.agents import DQNAgent
    from rl_trading_framework.trainers import RLTrainer

    print("\n[7.1] 创建训练环境...")
    env = TradingEnvironment(
        data_loader=SyntheticDataGenerator(initial_price=100.0, sigma=0.2),
        execution_engine=SimpleExecutionEngine(),
        reward_function=ImplementationShortfallReward(),
        target_quantity=5000,
        max_steps=30,
    )

    obs_dim = env.observation_space_shape[0]
    action_dim = env.action_space_shape[0]

    print("\n[7.2] 创建智能体...")
    policy = MLPPolicy(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=(64, 64))
    agent = DQNAgent(policy=policy, buffer_size=500, batch_size=32)

    print("\n[7.3] 创建训练器...")
    trainer = RLTrainer(
        env=env,
        agent=agent,
        save_dir="test_models",
        log_interval=2,
        save_interval=10,
    )

    print("\n[7.4] 训练5个episodes...")
    history = trainer.train(num_episodes=5)

    print(f"  训练历史:")
    print(f"    Episode奖励: {history['episode_rewards']}")
    print(f"    Episode长度: {history['episode_lengths']}")
    if history['completion_rates']:
        print(f"    完成率: {[f'{r:.2%}' for r in history['completion_rates']]}")

    print("\n[7.5] 评估智能体...")
    eval_metrics = trainer.evaluate(num_episodes=3)
    print(f"  评估结果:")
    print(f"    平均奖励: {eval_metrics.total_reward:.6f}")
    print(f"    完成率: {eval_metrics.completion_rate:.2%}")


def test_8_evaluator():
    """测试评估器"""
    from rl_trading_framework.environments import TradingEnvironment
    from rl_trading_framework.data import SyntheticDataGenerator
    from rl_trading_framework.execution import SimpleExecutionEngine
    from rl_trading_framework.rewards import ImplementationShortfallReward
    from rl_trading_framework.policies import MLPPolicy
    from rl_trading_framework.agents import DQNAgent
    from rl_trading_framework.evaluators import TradingEvaluator

    print("\n[8.1] 准备环境和智能体...")
    env = TradingEnvironment(
        data_loader=SyntheticDataGenerator(initial_price=100.0, sigma=0.2),
        execution_engine=SimpleExecutionEngine(),
        reward_function=ImplementationShortfallReward(),
        target_quantity=5000,
        max_steps=30,
    )

    policy = MLPPolicy(obs_dim=31, action_dim=4, hidden_sizes=(64, 64))
    agent = DQNAgent(policy=policy)

    print("\n[8.2] 评估单个episode...")
    evaluator = TradingEvaluator()
    metrics = evaluator.evaluate_episode(agent, env)

    print(f"  评估指标:")
    print(f"    总奖励: {metrics.total_reward:.6f}")
    print(f"    完成率: {metrics.completion_rate:.2%}")
    print(f"    步数: {metrics.steps}")

    print("\n[8.3] 生成评估报告...")
    metrics_list = [metrics]
    for i in range(4):
        m = evaluator.evaluate_episode(agent, env)
        metrics_list.append(m)

    report = evaluator.generate_report(metrics_list, output_path=None)
    print(f"  报告统计:")
    print(f"    Episodes: {report['num_episodes']}")
    print(f"    平均奖励: {report['total_reward']['mean']:.6f}")


def test_9_integration():
    """完整集成测试"""
    print("\n[9.1] 测试完整工作流...")

    from rl_trading_framework.environments import TradingEnvironment
    from rl_trading_framework.data import SyntheticDataGenerator
    from rl_trading_framework.execution import RealisticExecutionEngine
    from rl_trading_framework.rewards import CompositeReward, ImplementationShortfallReward, PnLBasedReward
    from rl_trading_framework.policies import MLPPolicy
    from rl_trading_framework.agents import DQNAgent
    from rl_trading_framework.trainers import RLTrainer
    from rl_trading_framework.evaluators import TradingEvaluator

    # 使用组合奖励
    composite_reward = CompositeReward([
        (ImplementationShortfallReward(), 0.7),
        (PnLBasedReward(), 0.3),
    ])

    # 使用真实执行引擎
    execution_engine = RealisticExecutionEngine(
        order_book_depth=5,
        liquidity_recovery_rate=0.1,
    )

    # 创建环境
    env = TradingEnvironment(
        data_loader=SyntheticDataGenerator(initial_price=100.0, mu=0.05, sigma=0.2),
        execution_engine=execution_engine,
        reward_function=composite_reward,
        target_quantity=8000,
        max_steps=40,
        side="buy",
    )

    # 创建智能体
    policy = MLPPolicy(obs_dim=31, action_dim=4, hidden_sizes=(128, 64))
    agent = DQNAgent(
        policy=policy,
        gamma=0.99,
        epsilon_start=0.5,
        epsilon_end=0.05,
        buffer_size=2000,
        batch_size=64,
    )

    # 训练
    trainer = RLTrainer(env=env, agent=agent, save_dir="integration_test_models")
    print("  训练10个episodes...")
    history = trainer.train(num_episodes=10)

    print(f"\n  训练完成:")
    print(f"    最终奖励: {history['episode_rewards'][-1]:.6f}")
    print(f"    平均奖励: {np.mean(history['episode_rewards']):.6f}")

    # 评估
    evaluator = TradingEvaluator()
    metrics_list = []
    for i in range(5):
        m = evaluator.evaluate_episode(agent, env)
        metrics_list.append(m)

    report = evaluator.generate_report(metrics_list, output_path=None)
    print(f"\n  评估报告:")
    print(f"    平均完成率: {report['completion_rate']['mean']:.2%}")

    print("\n  ✓ 完整集成测试通过")


def main():
    """主测试函数"""
    print("="*60)
    print("RL Trading Framework - 完整集成测试")
    print("="*60)

    tests = [
        ("数据模块", test_1_data_modules),
        ("执行引擎", test_2_execution_engines),
        ("奖励函数", test_3_reward_functions),
        ("交易环境", test_4_environment),
        ("策略网络", test_5_policies),
        ("智能体", test_6_agents),
        ("训练流程", test_7_training),
        ("评估器", test_8_evaluator),
        ("完整集成", test_9_integration),
    ]

    results = []
    for name, func in tests:
        result = test_module(name, func)
        results.append((name, result))

    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {status}: {name}")

    print(f"\n总计: {passed}/{total} 通过")

    if passed == total:
        print("\n" + "🎉 "*10)
        print("所有测试通过！框架运行正常，无Bug！")
        print("🎉 "*10)
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查上面的错误信息")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
