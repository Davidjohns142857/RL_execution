"""
强化学习交易执行训练演示

展示：
1. 训练过程中的学习曲线
2. 策略性能的改进
3. 详细的执行轨迹
4. 不同策略的对比
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl_trading_framework.data.historical_data import HistoricalDataLoader
from rl_trading_framework.environments.trading_env import TradingEnvironment
from rl_trading_framework.execution.simple_execution import SimpleExecutionEngine
from rl_trading_framework.rewards.implementation_shortfall import ImplementationShortfallReward
from rl_trading_framework.core.types import Action

# 尝试导入matplotlib，如果失败就跳过可视化
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("注意: matplotlib未安装，将跳过图形生成")


# 配置
DATA_DIR = os.path.join(os.path.dirname(__file__), "example_data", "market_data")
SYMBOL = "000001"
DATE = "20170105"
TARGET_QUANTITY = 10000.0
MAX_STEPS = 20
NUM_EPISODES = 50  # 增加训练轮数
INITIAL_CASH = 1000000.0


class SimpleDataLoader:
    def __init__(self, data):
        self.data = data
    def load_data(self, symbol, **kwargs):
        return self.data


class LearningPolicy:
    """
    简单的学习策略

    使用Q-learning思想，逐步学习最优的执行比例
    """
    def __init__(self, num_states=5, num_actions=10, learning_rate=0.1,
                 epsilon=0.3, gamma=0.95):
        """
        Args:
            num_states: 状态离散化数量（根据完成进度）
            num_actions: 动作离散化数量（执行比例）
            learning_rate: 学习率
            epsilon: 探索率
            gamma: 折扣因子
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.lr = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma

        # Q表初始化
        self.q_table = np.zeros((num_states, num_actions))

        # 动作空间：从激进到保守
        self.action_ratios = np.linspace(0.3, 1.0, num_actions)

        # 记录学习历史
        self.episode_rewards = []
        self.episode_completions = []

    def get_state(self, env):
        """根据执行进度获取状态"""
        completion_rate = env.executed_quantity / env.target_quantity
        time_progress = env.current_step / env.max_steps

        # 状态定义：基于完成率和时间进度的差异
        progress_diff = completion_rate - time_progress

        # 离散化状态
        if progress_diff < -0.2:
            state = 0  # 严重落后
        elif progress_diff < -0.05:
            state = 1  # 轻微落后
        elif progress_diff < 0.05:
            state = 2  # 进度正常
        elif progress_diff < 0.2:
            state = 3  # 轻微超前
        else:
            state = 4  # 严重超前

        return state

    def select_action(self, env, training=True):
        """选择动作（epsilon-greedy）"""
        state = self.get_state(env)

        # 探索 vs 利用
        if training and np.random.random() < self.epsilon:
            # 探索：随机选择
            action_idx = np.random.randint(0, self.num_actions)
        else:
            # 利用：选择Q值最大的动作
            action_idx = np.argmax(self.q_table[state])

        # 计算实际执行比例
        remaining_steps = env.max_steps - env.current_step
        base_ratio = 1.0 / max(remaining_steps, 1)
        trade_ratio = base_ratio * self.action_ratios[action_idx]
        trade_ratio = np.clip(trade_ratio, 0.0, 1.0)

        return Action(trade_ratio=trade_ratio, urgency=0.5), state, action_idx

    def update(self, state, action_idx, reward, next_state, done):
        """更新Q表"""
        if done:
            # 终止状态
            target = reward
        else:
            # Q-learning更新
            target = reward + self.gamma * np.max(self.q_table[next_state])

        # 更新Q值
        self.q_table[state, action_idx] += self.lr * (target - self.q_table[state, action_idx])

    def train_episode(self, env, episode_num):
        """训练一个episode"""
        obs = env.reset(seed=episode_num)
        total_reward = 0.0
        done = False

        trajectory = []

        while not done:
            # 选择动作
            action, state, action_idx = self.select_action(env, training=True)

            # 执行动作
            next_obs, reward, done, info = env.step(action)
            total_reward += reward

            # 记录轨迹
            trajectory.append({
                'state': state,
                'action_idx': action_idx,
                'trade_ratio': action.trade_ratio,
                'reward': reward,
                'executed_qty': info['executed_quantity'],
                'price': info['execution_price'],
                'completion': info['completion_rate'],
            })

            # 获取下一个状态
            next_state = self.get_state(env) if not done else state

            # 更新Q表
            self.update(state, action_idx, reward, next_state, done)

            obs = next_obs

        # 记录episode结果
        metrics = env.get_metrics()
        self.episode_rewards.append(total_reward)
        self.episode_completions.append(metrics['completion_rate'])

        return total_reward, metrics, trajectory


def train_and_visualize():
    """训练并可视化结果"""
    print("=" * 80)
    print("强化学习交易执行训练演示")
    print("=" * 80)

    # 加载数据
    print("\n[1/4] 加载数据...")
    loader = HistoricalDataLoader(
        data_dir=DATA_DIR,
        default_symbol=SYMBOL,
        default_date=DATE,
        trading_hours_only=True,
    )
    data = loader.load_data(SYMBOL, DATE)
    print(f"  ✓ 加载了 {len(data)} 条Level-2数据")

    # 创建环境
    print("\n[2/4] 创建训练环境...")
    env = TradingEnvironment(
        data_loader=SimpleDataLoader(data),
        execution_engine=SimpleExecutionEngine(
            permanent_impact_coef=0.01,
            temporary_impact_coef=0.005,
        ),
        reward_function=ImplementationShortfallReward(
            time_penalty_weight=0.001,
            completion_bonus=1.0,
        ),
        target_quantity=TARGET_QUANTITY,
        max_steps=MAX_STEPS,
        initial_cash=INITIAL_CASH,
        side="buy",
        window_size=20,
    )
    print(f"  ✓ 目标: 买入 {TARGET_QUANTITY:.0f} 股")
    print(f"  ✓ 执行步数: {MAX_STEPS} 步")

    # 创建学习策略
    print("\n[3/4] 开始训练...")
    policy = LearningPolicy(
        num_states=5,
        num_actions=10,
        learning_rate=0.1,
        epsilon=0.3,
        gamma=0.95
    )

    # 训练循环
    best_reward = -float('inf')
    best_episode = 0
    best_trajectory = None

    for episode in range(NUM_EPISODES):
        total_reward, metrics, trajectory = policy.train_episode(env, episode)

        if total_reward > best_reward:
            best_reward = total_reward
            best_episode = episode
            best_trajectory = trajectory

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(policy.episode_rewards[-10:])
            avg_completion = np.mean(policy.episode_completions[-10:])
            print(f"  Episode {episode+1:3d}: "
                  f"Avg奖励={avg_reward:.6f}, "
                  f"Avg完成率={avg_completion*100:.1f}%, "
                  f"最佳奖励={best_reward:.6f}")

    print(f"\n  ✓ 训练完成！最佳Episode: {best_episode+1}, 奖励: {best_reward:.6f}")

    # 分析结果
    print("\n[4/4] 分析训练结果...")

    # 计算统计数据
    rewards = policy.episode_rewards
    completions = policy.episode_completions

    print(f"\n  训练统计:")
    print(f"    奖励 - 平均: {np.mean(rewards):.6f}, 标准差: {np.std(rewards):.6f}")
    print(f"    奖励 - 最小: {np.min(rewards):.6f}, 最大: {np.max(rewards):.6f}")
    print(f"    完成率 - 平均: {np.mean(completions)*100:.2f}%")
    print(f"    完成率 - 最小: {np.min(completions)*100:.2f}%, 最大: {np.max(completions)*100:.2f}%")

    # 学习曲线分析
    window = 10
    if len(rewards) >= window:
        smoothed_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
        initial_perf = np.mean(rewards[:10])
        final_perf = np.mean(rewards[-10:])
        improvement = (final_perf - initial_perf) / abs(initial_perf) * 100
        print(f"\n  学习改进:")
        print(f"    初期表现 (前10轮): {initial_perf:.6f}")
        print(f"    后期表现 (后10轮): {final_perf:.6f}")
        print(f"    改进幅度: {improvement:+.2f}%")

    # 显示最佳执行轨迹
    print(f"\n  最佳执行轨迹 (Episode {best_episode+1}):")
    print(f"    {'步骤':<6} {'状态':<8} {'动作':<8} {'执行量':<10} {'价格':<10} {'完成率':<10} {'奖励':<12}")
    print(f"    {'-'*70}")
    for i, step in enumerate(best_trajectory[:5]):  # 显示前5步
        state_name = ['严重落后', '轻微落后', '正常', '轻微超前', '严重超前'][step['state']]
        print(f"    {i+1:<6} {state_name:<8} "
              f"{step['trade_ratio']:.3f}    "
              f"{step['executed_qty']:<10.0f} "
              f"{step['price']:<10.4f} "
              f"{step['completion']*100:<10.1f} "
              f"{step['reward']:<12.6f}")
    if len(best_trajectory) > 5:
        print(f"    ... (省略 {len(best_trajectory)-5} 步)")

    # 显示学习到的Q表
    print(f"\n  学习到的Q表 (数值越大表示该状态-动作对越优):")
    print(f"    {'状态':<12} " + " ".join([f"A{i:<7}" for i in range(policy.num_actions)]))
    print(f"    {'-'*100}")
    state_names = ['严重落后', '轻微落后', '正常', '轻微超前', '严重超前']
    for s in range(policy.num_states):
        q_values = " ".join([f"{policy.q_table[s, a]:7.3f}" for a in range(policy.num_actions)])
        best_action = np.argmax(policy.q_table[s])
        print(f"    {state_names[s]:<12} {q_values}  (最佳: A{best_action})")

    # 生成可视化（如果matplotlib可用）
    if HAS_MATPLOTLIB:
        print(f"\n  生成训练曲线...")
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. 奖励曲线
        ax = axes[0, 0]
        ax.plot(rewards, alpha=0.3, label='Episode奖励')
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(rewards)), smoothed, linewidth=2, label=f'{window}轮平滑')
        ax.axhline(y=best_reward, color='r', linestyle='--', label=f'最佳: {best_reward:.4f}')
        ax.set_xlabel('Episode')
        ax.set_ylabel('总奖励')
        ax.set_title('学习曲线 - 奖励变化')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. 完成率曲线
        ax = axes[0, 1]
        ax.plot(np.array(completions) * 100, alpha=0.3, label='Episode完成率')
        if len(completions) >= window:
            smoothed = np.convolve(completions, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(completions)), smoothed * 100, linewidth=2, label=f'{window}轮平滑')
        ax.axhline(y=100, color='g', linestyle='--', label='100%完成')
        ax.set_xlabel('Episode')
        ax.set_ylabel('完成率 (%)')
        ax.set_title('学习曲线 - 完成率变化')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Q值热力图
        ax = axes[1, 0]
        im = ax.imshow(policy.q_table, cmap='RdYlGn', aspect='auto')
        ax.set_xticks(range(policy.num_actions))
        ax.set_xticklabels([f'A{i}' for i in range(policy.num_actions)])
        ax.set_yticks(range(policy.num_states))
        ax.set_yticklabels(['严重落后', '轻微落后', '正常', '轻微超前', '严重超前'])
        ax.set_xlabel('动作 (执行激进程度)')
        ax.set_ylabel('状态 (执行进度)')
        ax.set_title('学习到的Q值表')
        plt.colorbar(im, ax=ax)

        # 4. 最佳执行轨迹
        ax = axes[1, 1]
        steps = list(range(1, len(best_trajectory) + 1))
        completions_traj = [t['completion'] * 100 for t in best_trajectory]
        ax.plot(steps, completions_traj, marker='o', linewidth=2, label='完成进度')
        ax.axline((0, 0), (MAX_STEPS, 100), color='gray', linestyle='--', alpha=0.5, label='理想均匀执行')
        ax.set_xlabel('执行步数')
        ax.set_ylabel('完成率 (%)')
        ax.set_title(f'最佳执行轨迹 (Episode {best_episode+1})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, MAX_STEPS)
        ax.set_ylim(0, 105)

        plt.tight_layout()
        output_file = 'training_results.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  ✓ 训练曲线已保存到: {output_file}")
    else:
        print(f"\n  (跳过图形生成，matplotlib未安装)")

    print("\n" + "=" * 80)
    print("训练完成！")
    print("=" * 80)

    return policy, env


if __name__ == "__main__":
    policy, env = train_and_visualize()
