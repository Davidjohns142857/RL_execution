# 快速开始指南

## 安装

### 1. 克隆仓库
```bash
git clone <repository-url>
cd RL_execution
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

## 验证安装

运行测试脚本验证所有模块正常工作：

```bash
python test_framework.py
```

如果看到 "✓ 所有测试通过！" 说明安装成功。

## 第一个训练任务

### 方式1：运行示例脚本

```bash
python rl_trading_framework/examples/simple_training.py
```

这将：
1. 生成合成市场数据
2. 创建交易环境
3. 训练DQN智能体（100 episodes）
4. 评估性能
5. 保存模型和训练曲线

结果保存在 `models/simple_example/` 目录。

### 方式2：自己编写代码

创建文件 `my_training.py`:

```python
from rl_trading_framework.environments import TradingEnvironment
from rl_trading_framework.data import SyntheticDataGenerator
from rl_trading_framework.execution import SimpleExecutionEngine
from rl_trading_framework.rewards import ImplementationShortfallReward
from rl_trading_framework.policies import MLPPolicy
from rl_trading_framework.agents import DQNAgent
from rl_trading_framework.trainers import RLTrainer

# 1. 创建组件
data_gen = SyntheticDataGenerator(initial_price=100.0, sigma=0.2)
execution_engine = SimpleExecutionEngine()
reward_fn = ImplementationShortfallReward()

# 2. 创建环境
env = TradingEnvironment(
    data_loader=data_gen,
    execution_engine=execution_engine,
    reward_function=reward_fn,
    target_quantity=10000,
    max_steps=50,
)

# 3. 创建智能体
obs_dim = env.observation_space_shape[0]
action_dim = env.action_space_shape[0]
policy = MLPPolicy(obs_dim=obs_dim, action_dim=action_dim)
agent = DQNAgent(policy=policy)

# 4. 训练
trainer = RLTrainer(env=env, agent=agent)
history = trainer.train(num_episodes=100)

# 5. 保存模型
agent.save("my_agent.pth")
```

运行：
```bash
python my_training.py
```

## 使用真实数据

### 准备CSV数据

创建目录并准备数据：
```bash
mkdir -p data
```

CSV文件格式（例如 `data/AAPL.csv`）：
```csv
timestamp,open,high,low,close,volume
1609459200,100.0,101.0,99.5,100.5,1000000
1609459260,100.5,101.2,100.0,100.8,1200000
...
```

### 使用CSV数据训练

```python
from rl_trading_framework.data import CSVDataLoader

# 加载数据
data_loader = CSVDataLoader(data_dir="./data")

# 创建环境（其他代码相同）
env = TradingEnvironment(
    data_loader=data_loader,  # 使用CSV数据
    execution_engine=execution_engine,
    reward_function=reward_fn,
    target_quantity=5000,
    max_steps=100,
)

# ... 继续训练
```

## 自定义组件

### 自定义奖励函数

```python
from rl_trading_framework.core.base import BaseReward
from rl_trading_framework.core.types import State, Action

class MyReward(BaseReward):
    def calculate(self, state, action, next_state,
                  executed_quantity, execution_price):
        # 你的奖励逻辑
        reward = ...
        return reward

# 使用
my_reward = MyReward()
env = TradingEnvironment(..., reward_function=my_reward)
```

### 组合多个奖励

```python
from rl_trading_framework.rewards import (
    ImplementationShortfallReward,
    PnLBasedReward,
    CompositeReward
)

composite_reward = CompositeReward([
    (ImplementationShortfallReward(), 0.7),
    (PnLBasedReward(), 0.3),
])

env = TradingEnvironment(..., reward_function=composite_reward)
```

## 评估模型

```python
from rl_trading_framework.evaluators import TradingEvaluator

# 加载训练好的模型
agent.load("my_agent.pth")

# 评估
evaluator = TradingEvaluator()
metrics_list = []

for i in range(10):
    metrics = evaluator.evaluate_episode(agent, env)
    metrics_list.append(metrics)

# 生成报告
report = evaluator.generate_report(
    metrics_list,
    output_path="evaluation_report.json"
)
```

## 切换RL算法

### 使用PPO代替DQN

```python
from rl_trading_framework.agents import PPOAgent

# 创建PPO智能体
agent = PPOAgent(
    policy=policy,
    clip_epsilon=0.2,
    epochs=10,
)

# 训练（代码完全相同）
trainer = RLTrainer(env=env, agent=agent)
history = trainer.train(num_episodes=100)
```

### 使用LSTM策略

```python
from rl_trading_framework.policies import LSTMPolicy

# LSTM策略网络
policy = LSTMPolicy(
    obs_dim=obs_dim,
    action_dim=action_dim,
    hidden_size=128,
    num_layers=2,
)

agent = DQNAgent(policy=policy)
# ... 继续训练
```

## 可视化训练过程

训练完成后：

```python
# 绘制训练曲线
trainer.plot_training_history(save_path="training_curves.png")
```

或者手动绘制：

```python
import matplotlib.pyplot as plt

plt.plot(history['episode_rewards'])
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Progress')
plt.savefig('rewards.png')
```

## GPU加速

如果有CUDA GPU：

```python
# 创建策略时指定device
policy = MLPPolicy(
    obs_dim=obs_dim,
    action_dim=action_dim,
    device='cuda'  # 使用GPU
)

agent = DQNAgent(policy=policy, device='cuda')
```

## 调整超参数

### 环境参数

```python
env = TradingEnvironment(
    data_loader=data_gen,
    execution_engine=execution_engine,
    reward_function=reward_fn,
    target_quantity=20000,      # 增加目标数量
    max_steps=100,              # 增加时间步数
    window_size=30,             # 更长的历史窗口
)
```

### 网络参数

```python
policy = MLPPolicy(
    obs_dim=obs_dim,
    action_dim=action_dim,
    hidden_sizes=(256, 256, 128),  # 更深的网络
    learning_rate=1e-4,             # 更小的学习率
)
```

### 智能体参数

```python
agent = DQNAgent(
    policy=policy,
    gamma=0.99,              # 折扣因子
    epsilon_start=1.0,       # 初始探索率
    epsilon_end=0.01,        # 最终探索率
    epsilon_decay=0.995,     # 衰减速度
    buffer_size=200000,      # 更大的回放缓冲
    batch_size=128,          # 更大的批次
    target_update_freq=20,   # 目标网络更新频率
)
```

## 常见问题

### Q: 训练很慢怎么办？
A:
1. 使用GPU (`device='cuda'`)
2. 减少episodes数量先测试
3. 增大batch_size
4. 使用更小的网络

### Q: 奖励一直是负数正常吗？
A: 正常。Implementation Shortfall奖励通常是负数（成本）。关注奖励是否在增长（变得less negative）。

### Q: 如何知道训练效果好不好？
A:
1. 查看completion_rate是否接近1.0
2. 对比不同策略的vwap_slippage
3. 运行evaluator生成详细报告

### Q: 可以用于实盘交易吗？
A: 本框架主要用于研究和开发。实盘需要：
1. 接入真实交易API
2. 实现风险管理
3. 充分回测验证
4. 遵守监管要求

## 下一步

1. **阅读完整文档**: [README.md](README.md)
2. **深入了解模块**: [docs/MODULE_DOCUMENTATION.md](docs/MODULE_DOCUMENTATION.md)
3. **查看高级示例**: `rl_trading_framework/examples/`
4. **尝试自定义组件**: 继承基类实现你的想法

## 获取帮助

- 查看代码注释（非常详细）
- 阅读模块文档
- 参考示例代码
- 提交Issue

祝你使用愉快！ 🚀
