# RL Trading Framework

一个模块化、可扩展的强化学习交易执行框架

## 📋 目录

- [简介](#简介)
- [核心特性](#核心特性)
- [框架架构](#框架架构)
- [快速开始](#快速开始)
- [模块说明](#模块说明)
- [使用示例](#使用示例)
- [扩展开发](#扩展开发)
- [项目结构](#项目结构)

---

## 简介

RL Trading Framework 是一个专为交易执行任务设计的强化学习框架。它提供了完整的模块化组件，支持：

- 📊 **多种数据源**：CSV文件、合成数据生成器
- 🎯 **灵活的交易环境**：支持买/卖、限价/市价等多种订单类型
- 🧠 **多种RL算法**：DQN、PPO等主流算法
- 💰 **可定制的奖励函数**：Implementation Shortfall、PnL等
- 📈 **完整的训练评估流程**：训练器、评估器、可视化

## 核心特性

### ✨ 高度模块化

每个组件都是独立的、可替换的：
- **环境（Environment）**：模拟交易市场
- **智能体（Agent）**：执行决策的RL算法
- **策略网络（Policy）**：MLP、LSTM等网络架构
- **数据加载器（DataLoader）**：支持多种数据源
- **执行引擎（ExecutionEngine）**：模拟真实订单执行
- **奖励函数（Reward）**：可定制的奖励计算

### 🔧 易于扩展

所有组件都继承自抽象基类，遵循统一的接口规范，方便添加新功能。

### 📝 清晰的文档

- 每个模块都有详细的输入/输出说明
- 关键数据格式都有明确定义
- 丰富的代码注释和使用示例

---

## 框架架构

```
RL Trading Framework
│
├── Core (核心模块)
│   ├── types.py          # 数据类型定义
│   └── base.py           # 抽象基类
│
├── Environments (交易环境)
│   └── TradingEnvironment # 交易执行环境
│
├── Data (数据模块)
│   ├── CSVDataLoader     # CSV数据加载
│   └── SyntheticData     # 合成数据生成
│
├── Execution (执行引擎)
│   ├── SimpleExecution   # 简单执行模型
│   └── RealisticExecution # 真实执行模型
│
├── Rewards (奖励函数)
│   ├── ImplementationShortfall
│   ├── PnLBased
│   └── CompositeReward
│
├── Policies (策略网络)
│   ├── MLPPolicy         # 多层感知机
│   └── LSTMPolicy        # LSTM网络
│
├── Agents (智能体)
│   ├── DQNAgent         # DQN算法
│   └── PPOAgent         # PPO算法
│
├── Trainers (训练器)
│   └── RLTrainer        # 训练管理
│
└── Evaluators (评估器)
    └── TradingEvaluator # 性能评估
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行简单示例

```bash
python rl_trading_framework/examples/simple_training.py
```

### 3. 基础使用代码

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
policy = MLPPolicy(obs_dim=31, action_dim=4)
agent = DQNAgent(policy=policy)

# 4. 训练
trainer = RLTrainer(env=env, agent=agent)
history = trainer.train(num_episodes=100)
```

---

## 模块说明

### 1. 核心数据类型 (core/types.py)

#### MarketData
市场数据结构
```python
@dataclass
class MarketData:
    timestamp: float      # 时间戳
    open: float          # 开盘价
    high: float          # 最高价
    low: float           # 最低价
    close: float         # 收盘价
    volume: float        # 成交量
    bid_price: Optional[float]  # 买一价
    ask_price: Optional[float]  # 卖一价
```

#### Action
智能体动作
```python
@dataclass
class Action:
    trade_ratio: float          # 交易比例 (0-1)
    order_type: OrderType       # 订单类型
    limit_price_offset: float   # 限价偏移
    urgency: float             # 紧急程度
```

#### Observation
环境观察
```python
@dataclass
class Observation:
    market_data: MarketData           # 当前市场数据
    historical_prices: np.ndarray     # 历史价格
    position: float                   # 当前持仓
    target_quantity: float            # 目标数量
    time_remaining: int               # 剩余时间
    executed_quantity: float          # 已执行数量
    # ... 更多字段
```

### 2. 环境模块 (environments/)

#### TradingEnvironment

**输入参数：**
- `data_loader`: 数据加载器
- `execution_engine`: 执行引擎
- `reward_function`: 奖励函数
- `target_quantity`: 目标交易数量
- `max_steps`: 最大步数
- `side`: 交易方向 ("buy" / "sell")

**主要方法：**
```python
obs = env.reset()                           # 重置环境
obs, reward, done, info = env.step(action)  # 执行一步
metrics = env.get_metrics()                 # 获取指标
```

**输出：**
- `observation`: Observation对象
- `reward`: float
- `done`: bool
- `info`: dict

### 3. 数据模块 (data/)

#### CSVDataLoader
从CSV文件加载历史数据

**CSV格式要求：**
```csv
timestamp,open,high,low,close,volume
1609459200,100.0,101.0,99.5,100.5,1000000
...
```

#### SyntheticDataGenerator
生成模拟数据（几何布朗运动）

**参数：**
- `initial_price`: 初始价格
- `mu`: 漂移率（年化）
- `sigma`: 波动率（年化）
- `dt`: 时间步长（秒）

### 4. 执行引擎 (execution/)

#### SimpleExecutionEngine

**市场冲击模型：**
```
impact = k * (quantity / volume)^γ
```

**输入：**
- `order`: OrderInfo
- `market_data`: MarketData
- `liquidity`: float

**输出：**
- `executed_quantity`: float
- `execution_price`: float
- `updated_order`: OrderInfo

#### RealisticExecutionEngine

增强特性：
- 订单簿深度模拟
- 部分成交
- 流动性恢复机制

### 5. 奖励函数 (rewards/)

#### ImplementationShortfallReward

基于执行缺口的奖励

**计算公式：**
```
reward = -(execution_price - arrival_price) / arrival_price
         - time_penalty
         + completion_bonus
```

#### PnLBasedReward

基于盈亏的奖励

**计算公式：**
```
reward = portfolio_value_change
         - transaction_cost
         - inventory_penalty
```

#### CompositeReward

组合多个奖励函数

**使用示例：**
```python
composite = CompositeReward([
    (ImplementationShortfallReward(), 0.7),
    (PnLBasedReward(), 0.3),
])
```

### 6. 策略网络 (policies/)

#### MLPPolicy

**网络结构：**
```
Input(obs_dim) -> FC(256) -> ReLU -> FC(256) -> ReLU -> FC(action_dim) -> Sigmoid
```

**输入：** observation array (batch_size, obs_dim)
**输出：** action array (batch_size, action_dim)

#### LSTMPolicy

**网络结构：**
```
Input -> LSTM(hidden_size, num_layers) -> FC layers -> Output
```

适用于需要记忆历史信息的场景

### 7. 智能体 (agents/)

#### DQNAgent

**核心参数：**
- `gamma`: 折扣因子 (0.99)
- `epsilon`: 探索率 (1.0 -> 0.01)
- `buffer_size`: 回放缓冲区大小
- `batch_size`: 训练批次大小

#### PPOAgent

**核心参数：**
- `clip_epsilon`: PPO裁剪参数 (0.2)
- `epochs`: 训练轮数
- `gae_lambda`: GAE参数 (0.95)

### 8. 训练器 (trainers/)

#### RLTrainer

**主要方法：**
```python
history = trainer.train(num_episodes=100)   # 训练
metrics = trainer.evaluate(num_episodes=10) # 评估
trainer.plot_training_history()             # 绘图
```

**输出指标：**
- episode_rewards
- episode_lengths
- completion_rates
- avg_execution_prices
- losses

### 9. 评估器 (evaluators/)

#### TradingEvaluator

**评估指标：**
- Total Reward
- Implementation Shortfall
- VWAP Slippage
- Completion Rate
- Sharpe Ratio

**使用示例：**
```python
evaluator = TradingEvaluator()
metrics = evaluator.evaluate_episode(agent, env)
report = evaluator.generate_report(metrics_list)
```

---

## 使用示例

### 示例1：使用CSV数据训练

```python
from rl_trading_framework.data import CSVDataLoader

# 加载真实数据
data_loader = CSVDataLoader(data_dir="./data")
data = data_loader.load_data("AAPL", start_time=1609459200)

# 创建环境
env = TradingEnvironment(
    data_loader=data_loader,
    execution_engine=SimpleExecutionEngine(),
    reward_function=ImplementationShortfallReward(),
    target_quantity=5000,
    max_steps=100,
)

# ... 继续训练
```

### 示例2：自定义奖励函数

```python
from rl_trading_framework.core.base import BaseReward

class MyReward(BaseReward):
    def calculate(self, state, action, next_state,
                  executed_quantity, execution_price):
        # 自定义奖励逻辑
        reward = ...
        return reward

# 使用自定义奖励
my_reward = MyReward()
env = TradingEnvironment(..., reward_function=my_reward)
```

### 示例3：模型保存和加载

```python
# 训练后保存
agent.save("models/my_agent.pth")

# 加载模型
agent.load("models/my_agent.pth")

# 评估
evaluator = TradingEvaluator()
metrics = evaluator.evaluate_episode(agent, env)
```

---

## 扩展开发

### 添加新的RL算法

1. 继承 `BaseAgent` 和 `BasePolicy`
2. 实现必需的方法：
   - `select_action()`
   - `update()`
   - `save()` / `load()`

### 添加新的执行模型

1. 继承 `BaseExecutionEngine`
2. 实现：
   - `execute_order()`
   - `calculate_market_impact()`
   - `calculate_slippage()`

### 添加新的环境

1. 继承 `BaseEnvironment`
2. 实现：
   - `reset()`
   - `step()`
   - `get_observation()`

---

## 项目结构

```
RL_execution/
├── rl_trading_framework/
│   ├── core/               # 核心基础模块
│   │   ├── __init__.py
│   │   ├── base.py        # 抽象基类
│   │   └── types.py       # 数据类型定义
│   │
│   ├── environments/       # 交易环境
│   │   ├── __init__.py
│   │   └── trading_env.py
│   │
│   ├── data/              # 数据加载
│   │   ├── __init__.py
│   │   ├── csv_loader.py
│   │   └── synthetic_data.py
│   │
│   ├── execution/         # 执行引擎
│   │   ├── __init__.py
│   │   ├── simple_execution.py
│   │   └── realistic_execution.py
│   │
│   ├── rewards/           # 奖励函数
│   │   ├── __init__.py
│   │   ├── implementation_shortfall.py
│   │   ├── pnl_based.py
│   │   └── composite_reward.py
│   │
│   ├── policies/          # 策略网络
│   │   ├── __init__.py
│   │   ├── mlp_policy.py
│   │   └── rnn_policy.py
│   │
│   ├── agents/            # RL智能体
│   │   ├── __init__.py
│   │   ├── dqn_agent.py
│   │   └── ppo_agent.py
│   │
│   ├── trainers/          # 训练器
│   │   ├── __init__.py
│   │   └── rl_trainer.py
│   │
│   ├── evaluators/        # 评估器
│   │   ├── __init__.py
│   │   └── trading_evaluator.py
│   │
│   ├── utils/             # 工具函数
│   │   ├── __init__.py
│   │   ├── logger.py
│   │   └── metrics.py
│   │
│   ├── examples/          # 示例代码
│   │   ├── simple_training.py
│   │   └── custom_components.py
│   │
│   └── configs/           # 配置文件
│
├── requirements.txt       # 依赖列表
└── README.md             # 本文档
```

---

## 关键数据格式总结

### 观察空间 (Observation)

维度：`base_features(11) + window_size(20) = 31`

特征组成：
1. 当前市场状态：close, volume, price_range
2. 历史统计：mean, std
3. 仓位信息：position, target_quantity, executed_ratio
4. 时间：time_remaining
5. 价格：avg_execution_price, market_impact
6. 历史价格窗口：最近N个价格

### 动作空间 (Action)

维度：4

组成：
1. `trade_ratio`: [0, 1] 本步交易比例
2. `order_type`: 订单类型编码
3. `limit_price_offset`: 限价偏移
4. `urgency`: [0, 1] 紧急程度

---

## 常见问题

### Q: 如何选择合适的RL算法？

- **DQN**: 适合离散或可离散化的动作空间
- **PPO**: 适合连续动作空间，训练更稳定

### Q: 如何调整奖励函数？

使用 `CompositeReward` 组合多个奖励函数，调整权重。

### Q: 数据格式有什么要求？

CSV数据必须包含：timestamp, open, high, low, close, volume

### Q: 如何加速训练？

1. 使用GPU：设置 `device='cuda'`
2. 增大batch_size
3. 使用并行环境（future work）

---

## 贡献指南

欢迎提交Issue和Pull Request！

## 许可证

MIT License

---

## 联系方式

- 项目主页：[GitHub Repository]
- 文档：本README及代码注释
- 问题反馈：GitHub Issues

---

**Happy Trading! 📈🚀**
