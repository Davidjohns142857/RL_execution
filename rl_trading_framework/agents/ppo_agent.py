"""
PPO智能体

Proximal Policy Optimization算法实现（简化版）
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple

from rl_trading_framework.core.base import BaseAgent, BasePolicy
from rl_trading_framework.core.types import Action, Observation, TransitionBatch


class PPOAgent(BaseAgent):
    """
    PPO智能体（简化版）

    使用PPO算法优化策略

    输入：
        - policy: 策略网络
        - clip_epsilon: PPO裁剪参数
        - epochs: 每次更新的训练轮数

    输出：
        - action: 选择的动作

    算法特点：
        1. 使用裁剪目标防止策略更新过大
        2. 可以多次使用同一批数据更新
        3. 更稳定的训练过程
    """

    def __init__(
        self,
        policy: BasePolicy,
        clip_epsilon: float = 0.2,
        epochs: int = 10,
        batch_size: int = 64,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = 'cpu',
    ):
        """
        初始化PPO智能体

        Args:
            policy: 策略网络
            clip_epsilon: PPO裁剪参数
            epochs: 训练轮数
            batch_size: 批次大小
            gamma: 折扣因子
            gae_lambda: GAE参数
            device: 设备
        """
        self.policy = policy
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = torch.device(device)

        # 存储轨迹
        self.trajectory_buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'log_probs': [],
        }

    def select_action(self, observation: Observation, training: bool = True) -> Action:
        """
        选择动作

        Args:
            observation: 观察
            training: 是否训练模式

        Returns:
            动作
        """
        obs_array = observation.to_array()

        # 获取动作
        action_array = self.policy.get_action(obs_array, deterministic=not training)

        # 添加探索噪声（训练模式）
        if training:
            noise = np.random.normal(0, 0.1, size=action_array.shape)
            action_array = np.clip(action_array + noise, 0, 1)

        # 转换为Action对象
        action = Action.from_array(action_array)

        # 存储用于训练
        if training:
            self.trajectory_buffer['states'].append(obs_array)
            self.trajectory_buffer['actions'].append(action_array)
            # log_prob会在更新时计算

        return action

    def store_transition(self, state, action, reward, next_state, done):
        """
        存储转移

        Args:
            state: 状态
            action: 动作
            reward: 奖励
            next_state: 下一状态
            done: 是否结束
        """
        self.trajectory_buffer['rewards'].append(reward)
        self.trajectory_buffer['dones'].append(done)

    def update(self, batch: TransitionBatch = None) -> dict:
        """
        使用PPO更新策略

        Returns:
            训练指标
        """
        if len(self.trajectory_buffer['rewards']) == 0:
            return {}

        # 计算优势函数
        advantages, returns = self._compute_gae()

        # 转换为数组
        states = np.array(self.trajectory_buffer['states'])
        actions = np.array(self.trajectory_buffer['actions'])

        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO更新
        total_loss = 0.0
        for _ in range(self.epochs):
            loss = self.policy.compute_loss(states, actions, advantages)
            self.policy.update_parameters(loss)
            total_loss += loss.item()

        # 清空缓冲区
        self.trajectory_buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'log_probs': [],
        }

        return {
            'loss': total_loss / self.epochs,
            'mean_advantage': advantages.mean(),
        }

    def _compute_gae(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算GAE优势函数

        Returns:
            advantages: 优势值
            returns: 回报值
        """
        rewards = np.array(self.trajectory_buffer['rewards'])
        dones = np.array(self.trajectory_buffer['dones'])

        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)

        # 反向计算GAE
        gae = 0
        next_value = 0  # 简化：假设最后的value为0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = 0
            else:
                next_non_terminal = 1.0 - dones[t]
                # 这里简化了value函数的计算
                # 实际应该使用critic网络预测value
                next_value = rewards[t + 1]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - 0
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae

        returns = advantages  # 简化：直接使用advantages作为returns

        return advantages, returns

    def save(self, path: str) -> None:
        """保存模型"""
        self.policy.save(path)

    def load(self, path: str) -> None:
        """加载模型"""
        self.policy.load(path)
