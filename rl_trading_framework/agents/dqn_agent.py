"""
DQN智能体

Deep Q-Network算法实现
"""

import numpy as np
import torch
import torch.nn as nn
from collections import deque
import random
from typing import Optional

from rl_trading_framework.core.base import BaseAgent, BasePolicy
from rl_trading_framework.core.types import Action, Observation, TransitionBatch


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """添加经验"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """采样批次"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent(BaseAgent):
    """
    DQN智能体

    使用Deep Q-Network学习价值函数

    输入：
        - policy: Q网络（策略网络）
        - gamma: 折扣因子
        - epsilon: 探索率
        - buffer_size: 经验回放缓冲区大小

    输出：
        - action: 选择的动作

    算法流程：
        1. 使用ε-greedy策略选择动作
        2. 存储经验到回放缓冲区
        3. 从缓冲区采样批次更新Q网络
        4. 周期性更新目标网络
    """

    def __init__(
        self,
        policy: BasePolicy,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 10,
        device: str = 'cpu',
    ):
        """
        初始化DQN智能体

        Args:
            policy: Q网络
            gamma: 折扣因子
            epsilon_start: 初始探索率
            epsilon_end: 最终探索率
            epsilon_decay: 探索率衰减
            buffer_size: 回放缓冲区大小
            batch_size: 批次大小
            target_update_freq: 目标网络更新频率
            device: 设备
        """
        self.policy = policy
        self.target_policy = self._create_target_network(policy)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.update_counter = 0

    def _create_target_network(self, policy):
        """创建目标网络（深拷贝）"""
        # 简化实现：这里假设policy有network属性
        # 实际使用时需要根据具体的policy类型调整
        import copy
        return copy.deepcopy(policy)

    def select_action(self, observation: Observation, training: bool = True) -> Action:
        """
        选择动作（ε-greedy）

        Args:
            observation: 观察
            training: 是否训练模式

        Returns:
            动作
        """
        # 转换观察为数组
        obs_array = observation.to_array()

        # ε-greedy策略
        if training and np.random.random() < self.epsilon:
            # 随机探索
            trade_ratio = np.random.uniform(0, 1)
            action_array = np.array([trade_ratio, 0, 0, 0.5])
        else:
            # 利用策略
            action_array = self.policy.get_action(obs_array, deterministic=not training)

        # 转换为Action对象
        action = Action.from_array(action_array)

        return action

    def update(self, batch: Optional[TransitionBatch] = None) -> dict:
        """
        更新Q网络

        Args:
            batch: 经验批次（如果为None，从回放缓冲区采样）

        Returns:
            训练指标
        """
        if batch is None:
            # 从回放缓冲区采样
            if len(self.replay_buffer) < self.batch_size:
                return {}

            states, actions, rewards, next_states, dones = self.replay_buffer.sample(
                self.batch_size
            )
        else:
            states = batch.states
            actions = batch.actions
            rewards = batch.rewards
            next_states = batch.next_states
            dones = batch.dones

        # 转换为tensor
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # 当前Q值
        current_q = self.policy.network(states_t)

        # 目标Q值
        with torch.no_grad():
            next_q = self.target_policy.network(next_states_t)
            # 取最大Q值对应的动作
            max_next_q = next_q.max(dim=1)[0]
            target_q = rewards_t + self.gamma * max_next_q * (1 - dones_t)

        # 计算TD loss
        # 这里简化处理，假设actions是索引
        # 实际使用时需要根据action的类型调整
        loss = nn.MSELoss()(current_q.mean(dim=1), target_q)

        # 更新网络
        self.policy.update_parameters(loss)

        # 更新目标网络
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self._update_target_network()

        # 衰减探索率
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return {
            'loss': loss.item(),
            'epsilon': self.epsilon,
            'q_value': current_q.mean().item(),
        }

    def _update_target_network(self):
        """更新目标网络"""
        self.target_policy.network.load_state_dict(self.policy.network.state_dict())

    def store_transition(self, state, action, reward, next_state, done):
        """
        存储经验

        Args:
            state: 当前状态
            action: 动作
            reward: 奖励
            next_state: 下一状态
            done: 是否结束
        """
        # 转换为数组
        if hasattr(state, 'to_array'):
            state = state.to_array()
        if hasattr(next_state, 'to_array'):
            next_state = next_state.to_array()
        if hasattr(action, 'to_array'):
            action = action.to_array()

        self.replay_buffer.push(state, action, reward, next_state, done)

    def save(self, path: str) -> None:
        """保存模型"""
        self.policy.save(path)

    def load(self, path: str) -> None:
        """加载模型"""
        self.policy.load(path)
        self._update_target_network()
