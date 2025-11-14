"""
LSTM策略网络

使用LSTM处理时序数据
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional

from rl_trading_framework.core.base import BasePolicy


class LSTMPolicy(BasePolicy):
    """
    LSTM策略网络

    适用于需要记忆历史信息的场景

    输入：
        - observation: 观察序列 (seq_len, obs_dim) 或 (batch, seq_len, obs_dim)

    输出：
        - action: 动作向量 (action_dim,) 或 (batch, action_dim)

    网络结构：
        Input -> LSTM -> FC layers -> Output
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        fc_sizes: Tuple[int, ...] = (128,),
        learning_rate: float = 3e-4,
        device: str = 'cpu',
    ):
        """
        初始化LSTM策略网络

        Args:
            obs_dim: 每个时间步的观察维度
            action_dim: 动作维度
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            fc_sizes: 全连接层大小
            learning_rate: 学习率
            device: 设备
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = torch.device(device)

        # 构建网络
        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # 全连接层
        fc_layers = []
        input_size = hidden_size
        for fc_size in fc_sizes:
            fc_layers.append(nn.Linear(input_size, fc_size))
            fc_layers.append(nn.ReLU())
            input_size = fc_size

        fc_layers.append(nn.Linear(input_size, action_dim))
        fc_layers.append(nn.Sigmoid())

        self.fc = nn.Sequential(*fc_layers)

        # 移动到设备
        self.lstm.to(self.device)
        self.fc.to(self.device)

        # 优化器
        self.optimizer = optim.Adam(
            list(self.lstm.parameters()) + list(self.fc.parameters()),
            lr=learning_rate
        )

        # 隐藏状态
        self.hidden_state = None

    def forward(self, observation: np.ndarray) -> np.ndarray:
        """
        前向传播

        Args:
            observation: 观察 shape: (seq_len, obs_dim) or (batch, seq_len, obs_dim)

        Returns:
            动作输出
        """
        # 处理输入形状
        is_batch = len(observation.shape) == 3
        if not is_batch:
            # (seq_len, obs_dim) -> (1, seq_len, obs_dim)
            observation = observation[np.newaxis, :, :]

        obs_tensor = torch.FloatTensor(observation).to(self.device)

        # LSTM前向传播
        with torch.no_grad():
            lstm_out, self.hidden_state = self.lstm(obs_tensor, self.hidden_state)

            # 取最后一个时间步的输出
            last_output = lstm_out[:, -1, :]

            # 通过全连接层
            action_tensor = self.fc(last_output)

        action = action_tensor.cpu().numpy()

        if not is_batch:
            action = action[0]

        return action

    def get_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        获取动作

        Args:
            observation: 观察
            deterministic: 是否确定性

        Returns:
            动作
        """
        action = self.forward(observation)

        if not deterministic:
            noise = np.random.normal(0, 0.1, size=action.shape)
            action = np.clip(action + noise, 0, 1)

        return action

    def update_parameters(self, loss: torch.Tensor) -> None:
        """
        更新参数

        Args:
            loss: 损失
        """
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.lstm.parameters()) + list(self.fc.parameters()),
            max_norm=1.0
        )
        self.optimizer.step()

    def reset_hidden_state(self):
        """重置隐藏状态"""
        self.hidden_state = None

    def save(self, path: str) -> None:
        """保存模型"""
        torch.save({
            'lstm_state_dict': self.lstm.state_dict(),
            'fc_state_dict': self.fc.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
        }, path)

    def load(self, path: str) -> None:
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.lstm.load_state_dict(checkpoint['lstm_state_dict'])
        self.fc.load_state_dict(checkpoint['fc_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
