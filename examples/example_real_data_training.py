"""
使用真实历史数据进行RL训练的完整示例

展示如何：
1. 加载真实Level-2市场数据
2. 配置训练环境
3. 训练强化学习智能体（支持无PyTorch的简单策略）
4. 评估训练结果

注意：本示例包含两种模式：
- 模式1: 不依赖PyTorch，使用简单启发式策略
- 模式2: 使用PyTorch训练RL模型（需要安装PyTorch）
"""

import sys
import os
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl_trading_framework.data.historical_data import HistoricalDataLoader
from rl_trading_framework.data.data_utils import print_data_summary
from rl_trading_framework.environments.trading_env import TradingEnvironment
from rl_trading_framework.execution.simple_execution import SimpleExecutionEngine
from rl_trading_framework.rewards.implementation_shortfall import ImplementationShortfallReward
from rl_trading_framework.core.types import Action


# ==================== 配置参数 ====================
DATA_DIR = "/path/to/your/data/market_data"  # 修改为你的数据路径
SYMBOL = "000001"
DATE = "20170105"

# 训练参数
TARGET_QUANTITY = 10000.0  # 目标交易股数
MAX_STEPS = 20  # 每个episode最大步数
NUM_EPISODES = 10  # 训练episode数
INITIAL_CASH = 1000000.0  # 初始资金


class SimpleDataLoader:
    """简单数据加载器包装"""
    def __init__(self, data):
        self.data = data

    def load_data(self, symbol, **kwargs):
        return self.data


class HeuristicPolicy:
    """
    启发式策略（不依赖PyTorch）

    实现几种经典的执行策略：
    - TWAP (Time-Weighted Average Price): 均匀分配
    - VWAP-like: 根据历史成交量分配
    - Adaptive: 根据市场状态自适应调整
    """

    def __init__(self, policy_type="twap"):
        """
        Args:
            policy_type: 策略类型 ("twap", "vwap", "adaptive")
        """
        self.policy_type = policy_type

    def select_action(self, observation, env):
        """
        选择动作

        Args:
            observation: 环境观察
            env: 环境对象

        Returns:
            Action对象
        """
        if self.policy_type == "twap":
            return self._twap_action(env)
        elif self.policy_type == "vwap":
            return self._vwap_action(observation, env)
        elif self.policy_type == "adaptive":
            return self._adaptive_action(observation, env)
        else:
            raise ValueError(f"未知策略类型: {self.policy_type}")

    def _twap_action(self, env):
        """TWAP策略：均匀执行"""
        remaining_steps = env.max_steps - env.current_step
        if remaining_steps > 0:
            trade_ratio = 1.0 / remaining_steps
        else:
            trade_ratio = 0.0

        return Action(trade_ratio=trade_ratio, urgency=0.5)

    def _vwap_action(self, observation, env):
        """VWAP策略：根据历史成交量比例执行"""
        # 简化实现：使用当前成交量与历史平均的比值
        current_volume = observation.market_data.volume
        avg_volume = np.mean(observation.historical_volumes) if len(observation.historical_volumes) > 0 else 1.0

        # 根据成交量调整执行比例
        volume_ratio = current_volume / max(avg_volume, 1.0)
        base_ratio = 1.0 / max(env.max_steps - env.current_step, 1)
        trade_ratio = base_ratio * np.clip(volume_ratio, 0.5, 2.0)
        trade_ratio = np.clip(trade_ratio, 0.0, 1.0)

        return Action(trade_ratio=trade_ratio, urgency=0.5)

    def _adaptive_action(self, observation, env):
        """自适应策略：根据市场状态和执行进度调整"""
        remaining_steps = env.max_steps - env.current_step
        if remaining_steps == 0:
            return Action(trade_ratio=0.0)

        # 基础比例（TWAP）
        base_ratio = 1.0 / remaining_steps

        # 根据执行进度调整
        completion_rate = observation.executed_quantity / max(env.target_quantity, 1.0)
        time_progress = env.current_step / env.max_steps

        # 如果执行落后，加速
        if completion_rate < time_progress - 0.1:
            urgency = 0.8
            trade_ratio = base_ratio * 1.5
        # 如果执行超前，减速
        elif completion_rate > time_progress + 0.1:
            urgency = 0.3
            trade_ratio = base_ratio * 0.7
        else:
            urgency = 0.5
            trade_ratio = base_ratio

        # 根据价差调整
        spread_bps = observation.market_data.get_spread_bps()
        if spread_bps > 50:  # 价差过大，降低紧迫性
            urgency *= 0.8

        trade_ratio = np.clip(trade_ratio, 0.0, 1.0)

        return Action(trade_ratio=trade_ratio, urgency=urgency)


def load_and_validate_data():
    """加载并验证数据"""
    print("=" * 60)
    print("步骤1: 加载并验证数据")
    print("=" * 60)

    if not os.path.exists(DATA_DIR):
        print(f"\n❌ 数据路径不存在: {DATA_DIR}")
        print("\n请修改脚本顶部的配置参数：")
        print("  DATA_DIR = '/your/data/path'")
        print("  SYMBOL = '000001'")
        print("  DATE = '20170105'")
        return None

    try:
        loader = HistoricalDataLoader(
            data_dir=DATA_DIR,
            default_symbol=SYMBOL,
            default_date=DATE,
            trading_hours_only=True,
        )

        data = loader.load_data(symbol=SYMBOL, date=DATE)
        print(f"\n✓ 成功加载 {len(data)} 条数据")

        # 显示数据摘要
        print_data_summary(data)

        return data

    except Exception as e:
        print(f"\n❌ 加载数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_environment(data):
    """创建训练环境"""
    print("\n" + "=" * 60)
    print("步骤2: 创建训练环境")
    print("=" * 60)

    env = TradingEnvironment(
        data_loader=SimpleDataLoader(data),
        execution_engine=SimpleExecutionEngine(
            permanent_impact_coef=0.01,
            temporary_impact_coef=0.005,
        ),
        reward_function=ImplementationShortfallReward(
            arrival_price_weight=1.0,
            vwap_weight=0.5,
        ),
        target_quantity=TARGET_QUANTITY,
        max_steps=MAX_STEPS,
        initial_cash=INITIAL_CASH,
        side="buy",
        window_size=20,
    )

    print(f"\n✓ 环境配置:")
    print(f"  - 目标数量: {TARGET_QUANTITY:.0f} 股")
    print(f"  - 最大步数: {MAX_STEPS}")
    print(f"  - 观察空间维度: {env.observation_space_shape}")
    print(f"  - 动作空间维度: {env.action_space_shape}")

    return env


def train_with_heuristic(env, policy_type="adaptive"):
    """使用启发式策略进行训练/评估"""
    print("\n" + "=" * 60)
    print(f"步骤3: 运行启发式策略 - {policy_type.upper()}")
    print("=" * 60)

    policy = HeuristicPolicy(policy_type=policy_type)

    episode_rewards = []
    episode_metrics = []

    for episode in range(NUM_EPISODES):
        obs = env.reset(seed=episode)
        total_reward = 0.0
        done = False

        while not done:
            # 选择动作
            action = policy.select_action(obs, env)

            # 执行动作
            obs, reward, done, info = env.step(action)
            total_reward += reward

        # 记录结果
        metrics = env.get_metrics()
        episode_rewards.append(total_reward)
        episode_metrics.append(metrics)

        # 显示进度
        if (episode + 1) % max(1, NUM_EPISODES // 5) == 0 or episode == 0:
            print(f"\nEpisode {episode + 1}/{NUM_EPISODES}:")
            print(f"  - 总奖励: {total_reward:.6f}")
            print(f"  - 完成率: {metrics['completion_rate']*100:.1f}%")
            print(f"  - 平均执行价: {metrics['avg_execution_price']:.4f}")
            print(f"  - VWAP滑点: {metrics['vwap_slippage']*10000:.2f} bps")

    return episode_rewards, episode_metrics


def evaluate_results(rewards, metrics, policy_name):
    """评估训练结果"""
    print("\n" + "=" * 60)
    print(f"步骤4: 评估结果 - {policy_name}")
    print("=" * 60)

    # 奖励统计
    print(f"\n奖励统计:")
    print(f"  - 平均: {np.mean(rewards):.6f}")
    print(f"  - 标准差: {np.std(rewards):.6f}")
    print(f"  - 最小: {np.min(rewards):.6f}")
    print(f"  - 最大: {np.max(rewards):.6f}")

    # 执行指标统计
    completion_rates = [m['completion_rate'] for m in metrics]
    vwap_slippages = [m['vwap_slippage'] for m in metrics]
    avg_prices = [m['avg_execution_price'] for m in metrics]

    print(f"\n完成率:")
    print(f"  - 平均: {np.mean(completion_rates)*100:.2f}%")
    print(f"  - 最小: {np.min(completion_rates)*100:.2f}%")
    print(f"  - 最大: {np.max(completion_rates)*100:.2f}%")

    print(f"\nVWAP滑点 (bps):")
    print(f"  - 平均: {np.mean(vwap_slippages)*10000:.2f}")
    print(f"  - 标准差: {np.std(vwap_slippages)*10000:.2f}")

    print(f"\n平均执行价:")
    print(f"  - 均值: {np.mean(avg_prices):.4f}")
    print(f"  - 标准差: {np.std(avg_prices):.4f}")


def compare_strategies(env):
    """比较不同策略"""
    print("\n" + "=" * 60)
    print("步骤5: 策略对比")
    print("=" * 60)

    strategies = ["twap", "vwap", "adaptive"]
    results = {}

    for strategy in strategies:
        print(f"\n测试策略: {strategy.upper()}")
        print("-" * 40)
        rewards, metrics = train_with_heuristic(env, policy_type=strategy)
        results[strategy] = {
            'rewards': rewards,
            'metrics': metrics,
            'avg_reward': np.mean(rewards),
            'avg_completion': np.mean([m['completion_rate'] for m in metrics]),
            'avg_slippage': np.mean([m['vwap_slippage'] for m in metrics]),
        }

    # 对比结果
    print("\n" + "=" * 60)
    print("策略对比结果")
    print("=" * 60)
    print(f"\n{'策略':<10} {'平均奖励':<15} {'平均完成率':<15} {'平均滑点(bps)':<15}")
    print("-" * 60)
    for strategy, result in results.items():
        print(f"{strategy.upper():<10} {result['avg_reward']:<15.6f} "
              f"{result['avg_completion']*100:<15.2f} "
              f"{result['avg_slippage']*10000:<15.2f}")

    # 找出最佳策略
    best_strategy = max(results.items(), key=lambda x: x[1]['avg_reward'])
    print(f"\n✓ 最佳策略: {best_strategy[0].upper()}")
    print(f"  - 平均奖励: {best_strategy[1]['avg_reward']:.6f}")


def main():
    """主函数"""
    print("=" * 60)
    print("使用真实历史数据进行RL训练")
    print("=" * 60)

    # 步骤1: 加载数据
    data = load_and_validate_data()
    if data is None:
        return

    # 步骤2: 创建环境
    env = create_environment(data)

    # 步骤3-5: 训练和评估（使用启发式策略）
    print("\n" + "=" * 60)
    print("使用启发式策略进行评估")
    print("（不需要PyTorch，可直接运行）")
    print("=" * 60)

    # 单策略训练
    rewards, metrics = train_with_heuristic(env, policy_type="adaptive")
    evaluate_results(rewards, metrics, "Adaptive Policy")

    # 策略对比
    compare_strategies(env)

    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)

    print("\n下一步:")
    print("1. 如果安装了PyTorch，可以训练真正的RL模型")
    print("2. 调整环境参数（target_quantity, max_steps等）")
    print("3. 尝试不同的奖励函数")
    print("4. 使用不同日期的数据进行测试")

    # PyTorch模式提示
    try:
        import torch
        print("\n" + "=" * 60)
        print("检测到PyTorch，可以使用RL算法训练")
        print("=" * 60)
        print("\n参考 comprehensive_test.py 中的训练代码")
        print("或查看文档了解如何使用DQN/PPO等算法")
    except ImportError:
        print("\n" + "=" * 60)
        print("未安装PyTorch")
        print("=" * 60)
        print("\n如需使用RL算法（DQN/PPO等），请安装PyTorch:")
        print("  pip install torch")


if __name__ == "__main__":
    main()
