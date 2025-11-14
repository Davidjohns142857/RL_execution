"""
多层感知机（MLP）策略网络

使用全连接神经网络实现策略
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional

from rl_trading_framework.core.base import BasePolicy


class MLPPolicy(BasePolicy):
    """
    MLP策略网络

    输入：
        - observation: 观察向量 (obs_dim,)

    输出：
        - action: 动作向量 (action_dim,) 或动作分布参数

    网络结构：
        Input -> Hidden1 -> Hidden2 -> ... -> Output
        使用ReLU激活函数，最后一层使用Sigmoid/Tanh

    参数：
        - obs_dim: 观察空间维度
        - action_dim: 动作空间维度
        - hidden_sizes: 隐藏层大小列表
        - activation: 激活函数
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        activation: str = 'relu',
        learning_rate: float = 3e-4,
        device: str = 'cpu',
    ):
        """
        初始化MLP策略网络

        Args:
            obs_dim: 观察维度
            action_dim: 动作维度
            hidden_sizes: 隐藏层大小
            activation: 激活函数 ('relu', 'tanh')
            learning_rate: 学习率
            device: 设备 ('cpu' or 'cuda')
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_sizes = hidden_sizes
        self.device = torch.device(device)

        # 构建网络
        self.network = self._build_network(activation)
        self.network.to(self.device)

        # 优化器
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

    def _build_network(self, activation: str) -> nn.Module:
        """
        构建神经网络

        Args:
            activation: 激活函数类型

        Returns:
            PyTorch网络模块
        """
        # 选择激活函数
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'tanh':
            act_fn = nn.Tanh
        else:
            raise ValueError(f"不支持的激活函数: {activation}")

        # 构建层
        layers = []
        input_size = self.obs_dim

        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(act_fn())
            input_size = hidden_size

        # 输出层
        layers.append(nn.Linear(input_size, self.action_dim))
        layers.append(nn.Sigmoid())  # 输出0-1之间的值

        return nn.Sequential(*layers)

    def forward(self, observation: np.ndarray) -> np.ndarray:
        """
        前向传播

        Args:
            observation: 观察数组 shape: (batch_size, obs_dim) or (obs_dim,)

        Returns:
            动作输出 shape: (batch_size, action_dim) or (action_dim,)
        """
        # 转换为tensor
        is_batch = len(observation.shape) > 1
        if not is_batch:
            observation = observation[np.newaxis, :]

        obs_tensor = torch.FloatTensor(observation).to(self.device)

        # 前向传播
        with torch.no_grad():
            action_tensor = self.network(obs_tensor)

        # 转换回numpy
        action = action_tensor.cpu().numpy()

        if not is_batch:
            action = action[0]

        return action

    def get_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        获取动作

        Args:
            observation: 观察
            deterministic: 是否使用确定性策略

        Returns:
            动作数组
        """
        action = self.forward(observation)

        # 如果非确定性，添加探索噪声
        if not deterministic:
            noise = np.random.normal(0, 0.1, size=action.shape)
            action = np.clip(action + noise, 0, 1)

        return action

    def update_parameters(self, loss: torch.Tensor) -> None:
        """
        更新网络参数

        Args:
            loss: 损失张量
        """
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()

    def compute_loss(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> torch.Tensor:
        """
        计算策略损失（用于策略梯度算法）

        Args:
            observations: 观察批次
            actions: 动作批次
            advantages: 优势值批次

        Returns:
            策略损失
        """
        obs_tensor = torch.FloatTensor(observations).to(self.device)
        action_tensor = torch.FloatTensor(actions).to(self.device)
        adv_tensor = torch.FloatTensor(advantages).to(self.device)

        # 前向传播获取预测动作
        pred_actions = self.network(obs_tensor)

        # 计算MSE损失（简化版策略梯度）
        loss = torch.mean((pred_actions - action_tensor) ** 2 * adv_tensor.unsqueeze(1))

        return loss

    def save(self, path: str) -> None:
        """
        保存模型

        Args:
            path: 保存路径
        """
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'hidden_sizes': self.hidden_sizes,
        }, path)

    def load(self, path: str) -> None:
        """
        加载模型

        Args:
            path: 模型路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
