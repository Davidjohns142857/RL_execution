"""
简单训练示例

演示如何使用框架训练一个基础的交易执行智能体
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from rl_trading_framework.environments.trading_env import TradingEnvironment
from rl_trading_framework.data.synthetic_data import SyntheticDataGenerator
from rl_trading_framework.execution.simple_execution import SimpleExecutionEngine
from rl_trading_framework.rewards.implementation_shortfall import ImplementationShortfallReward
from rl_trading_framework.policies.mlp_policy import MLPPolicy
from rl_trading_framework.agents.dqn_agent import DQNAgent
from rl_trading_framework.trainers.rl_trainer import RLTrainer
from rl_trading_framework.evaluators.trading_evaluator import TradingEvaluator


def main():
    """主函数"""
    print("="*60)
    print("RL Trading Framework - 简单训练示例")
    print("="*60)

    # 1. 创建数据加载器
    print("\n步骤 1: 创建合成数据生成器...")
    data_generator = SyntheticDataGenerator(
        initial_price=100.0,
        mu=0.0,  # 无漂移
        sigma=0.2,  # 20%年化波动率
        dt=60.0,  # 1分钟数据
    )

    # 2. 创建执行引擎
    print("步骤 2: 创建执行引擎...")
    execution_engine = SimpleExecutionEngine(
        permanent_impact_coef=0.1,
        temporary_impact_coef=0.5,
    )

    # 3. 创建奖励函数
    print("步骤 3: 创建奖励函数...")
    reward_function = ImplementationShortfallReward(
        time_penalty_weight=0.001,
        completion_bonus=1.0,
    )

    # 4. 创建交易环境
    print("步骤 4: 创建交易环境...")
    env = TradingEnvironment(
        data_loader=data_generator,
        execution_engine=execution_engine,
        reward_function=reward_function,
        target_quantity=10000,  # 目标交易10000股
        max_steps=50,  # 最多50步
        initial_cash=1000000.0,  # 初始100万现金
        side="buy",  # 买入
        symbol="SYNTHETIC",
        window_size=20,
    )

    # 5. 创建策略网络
    print("步骤 5: 创建策略网络...")
    obs_dim = env.observation_space_shape[0]
    action_dim = env.action_space_shape[0]

    policy = MLPPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=(128, 128),
        learning_rate=3e-4,
    )

    # 6. 创建智能体
    print("步骤 6: 创建DQN智能体...")
    agent = DQNAgent(
        policy=policy,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=64,
    )

    # 7. 创建训练器
    print("步骤 7: 创建训练器...")
    trainer = RLTrainer(
        env=env,
        agent=agent,
        save_dir="models/simple_example",
        log_interval=10,
        save_interval=50,
    )

    # 8. 开始训练
    print("\n步骤 8: 开始训练...\n")
    history = trainer.train(
        num_episodes=100,  # 训练100个episodes
    )

    # 9. 评估
    print("\n步骤 9: 评估训练好的智能体...")
    evaluator = TradingEvaluator()

    # 评估多个episodes
    eval_metrics_list = []
    for i in range(10):
        metrics = evaluator.evaluate_episode(agent, env)
        eval_metrics_list.append(metrics)

    # 生成评估报告
    report = evaluator.generate_report(
        eval_metrics_list,
        output_path="models/simple_example/evaluation_report.json"
    )

    # 10. 绘制训练曲线
    print("\n步骤 10: 绘制训练曲线...")
    trainer.plot_training_history(
        save_path="models/simple_example/training_curves.png"
    )

    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print(f"模型保存在: models/simple_example/")
    print(f"评估报告: models/simple_example/evaluation_report.json")
    print(f"训练曲线: models/simple_example/training_curves.png")


if __name__ == "__main__":
    main()
