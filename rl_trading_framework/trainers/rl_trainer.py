"""
强化学习训练器

管理训练循环、日志记录和模型保存
"""

import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm
import json

from rl_trading_framework.core.base import BaseTrainer, BaseAgent, BaseEnvironment
from rl_trading_framework.core.types import EpisodeMetrics


class RLTrainer(BaseTrainer):
    """
    RL训练器

    输入：
        - env: 训练环境
        - agent: RL智能体
        - num_episodes: 训练episode数量

    输出：
        - history: 训练历史字典

    功能：
        1. 管理训练循环
        2. 收集训练指标
        3. 保存模型检查点
        4. 日志记录
    """

    def __init__(
        self,
        env: BaseEnvironment,
        agent: BaseAgent,
        save_dir: str = "models",
        log_interval: int = 10,
        save_interval: int = 100,
    ):
        """
        初始化训练器

        Args:
            env: 训练环境
            agent: 智能体
            save_dir: 模型保存目录
            log_interval: 日志打印间隔
            save_interval: 模型保存间隔
        """
        self.env = env
        self.agent = agent
        self.save_dir = save_dir
        self.log_interval = log_interval
        self.save_interval = save_interval

        # 训练历史
        self.history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'completion_rates': [],
            'avg_execution_prices': [],
            'losses': [],
        }

    def train(
        self,
        num_episodes: int,
        max_steps_per_episode: Optional[int] = None,
        **kwargs
    ) -> Dict[str, List[float]]:
        """
        训练智能体

        Args:
            num_episodes: 训练episode数量
            max_steps_per_episode: 每个episode最大步数
            **kwargs: 其他参数

        Returns:
            训练历史
        """
        print(f"开始训练 {num_episodes} episodes...")

        for episode in tqdm(range(num_episodes), desc="训练进度"):
            # 重置环境
            obs = self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False

            # Episode循环
            while not done:
                # 选择动作
                action = self.agent.select_action(obs, training=True)

                # 执行动作
                next_obs, reward, done, info = self.env.step(action)

                # 存储经验
                if hasattr(self.agent, 'store_transition'):
                    self.agent.store_transition(obs, action, reward, next_obs, done)

                # 更新智能体
                if hasattr(self.agent, 'update'):
                    metrics = self.agent.update()
                    if 'loss' in metrics:
                        self.history['losses'].append(metrics['loss'])

                # 累积奖励
                episode_reward += reward
                episode_length += 1

                # 更新观察
                obs = next_obs

                # 检查最大步数
                if max_steps_per_episode and episode_length >= max_steps_per_episode:
                    break

            # 记录episode指标
            self.history['episode_rewards'].append(episode_reward)
            self.history['episode_lengths'].append(episode_length)

            # 获取环境指标
            if hasattr(self.env, 'get_metrics'):
                env_metrics = self.env.get_metrics()
                if 'completion_rate' in env_metrics:
                    self.history['completion_rates'].append(env_metrics['completion_rate'])
                if 'avg_execution_price' in env_metrics:
                    self.history['avg_execution_prices'].append(env_metrics['avg_execution_price'])

            # 打印日志
            if (episode + 1) % self.log_interval == 0:
                self._log_progress(episode + 1)

            # 保存模型
            if (episode + 1) % self.save_interval == 0:
                self._save_checkpoint(episode + 1)

        print("训练完成！")
        return self.history

    def evaluate(
        self,
        num_episodes: int,
        deterministic: bool = True,
        **kwargs
    ) -> EpisodeMetrics:
        """
        评估智能体

        Args:
            num_episodes: 评估episode数量
            deterministic: 是否使用确定性策略
            **kwargs: 其他参数

        Returns:
            评估指标
        """
        print(f"评估 {num_episodes} episodes...")

        episode_rewards = []
        completion_rates = []
        execution_prices = []

        for episode in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0.0
            done = False

            while not done:
                # 选择动作（不探索）
                action = self.agent.select_action(obs, training=False)

                # 执行动作
                next_obs, reward, done, info = self.env.step(action)

                episode_reward += reward
                obs = next_obs

            episode_rewards.append(episode_reward)

            # 收集指标
            if hasattr(self.env, 'get_metrics'):
                metrics = self.env.get_metrics()
                completion_rates.append(metrics.get('completion_rate', 0.0))
                execution_prices.append(metrics.get('avg_execution_price', 0.0))

        # 汇总指标
        eval_metrics = EpisodeMetrics(
            total_reward=np.mean(episode_rewards),
            completion_rate=np.mean(completion_rates) if completion_rates else 0.0,
            avg_execution_price=np.mean(execution_prices) if execution_prices else 0.0,
            steps=int(np.mean(self.history['episode_lengths'][-num_episodes:])) if self.history['episode_lengths'] else 0,
        )

        print(f"\n评估结果:")
        print(f"  平均奖励: {eval_metrics.total_reward:.4f}")
        print(f"  完成率: {eval_metrics.completion_rate:.2%}")
        print(f"  平均执行价格: {eval_metrics.avg_execution_price:.4f}")

        return eval_metrics

    def _log_progress(self, episode: int):
        """打印训练进度"""
        recent_rewards = self.history['episode_rewards'][-self.log_interval:]
        recent_lengths = self.history['episode_lengths'][-self.log_interval:]

        avg_reward = np.mean(recent_rewards)
        avg_length = np.mean(recent_lengths)

        print(f"\nEpisode {episode}:")
        print(f"  平均奖励: {avg_reward:.4f}")
        print(f"  平均步数: {avg_length:.1f}")

        if self.history['completion_rates']:
            recent_completion = self.history['completion_rates'][-self.log_interval:]
            print(f"  平均完成率: {np.mean(recent_completion):.2%}")

        if self.history['losses']:
            recent_loss = self.history['losses'][-100:]
            print(f"  平均损失: {np.mean(recent_loss):.4f}")

    def _save_checkpoint(self, episode: int):
        """保存检查点"""
        import os
        os.makedirs(self.save_dir, exist_ok=True)

        # 保存模型
        model_path = os.path.join(self.save_dir, f"agent_episode_{episode}.pth")
        self.agent.save(model_path)
        print(f"模型已保存: {model_path}")

        # 保存训练历史
        history_path = os.path.join(self.save_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def plot_training_history(self, save_path: Optional[str] = None):
        """
        绘制训练曲线

        Args:
            save_path: 图片保存路径
        """
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # 奖励曲线
            axes[0, 0].plot(self.history['episode_rewards'])
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')

            # 完成率
            if self.history['completion_rates']:
                axes[0, 1].plot(self.history['completion_rates'])
                axes[0, 1].set_title('Completion Rate')
                axes[0, 1].set_xlabel('Episode')
                axes[0, 1].set_ylabel('Rate')

            # 损失
            if self.history['losses']:
                axes[1, 0].plot(self.history['losses'])
                axes[1, 0].set_title('Training Loss')
                axes[1, 0].set_xlabel('Update Step')
                axes[1, 0].set_ylabel('Loss')

            # Episode长度
            axes[1, 1].plot(self.history['episode_lengths'])
            axes[1, 1].set_title('Episode Length')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Steps')

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path)
                print(f"训练曲线已保存: {save_path}")
            else:
                plt.show()

        except ImportError:
            print("需要安装matplotlib才能绘制图表")
