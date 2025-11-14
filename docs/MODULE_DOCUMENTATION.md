# 模块详细文档

## 目录

1. [数据流概览](#数据流概览)
2. [核心数据类型](#核心数据类型)
3. [模块详细说明](#模块详细说明)
4. [典型使用流程](#典型使用流程)

---

## 数据流概览

```
┌─────────────────────────────────────────────────────────────┐
│                      训练/评估流程                            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────┐   ┌──────────────┐   ┌─────────────────┐
│   DataLoader     │──▶│ Environment  │◀─▶│     Agent       │
│  加载市场数据      │   │  模拟交易环境  │   │  选择动作/学习  │
└──────────────────┘   └──────────────┘   └─────────────────┘
                            │  ▲
                            ▼  │
                       ┌─────────────┐
                       │ExecutionEngine│
                       │  执行订单     │
                       └─────────────┘
                            │  ▲
                            ▼  │
                       ┌─────────────┐
                       │RewardFunction│
                       │  计算奖励    │
                       └─────────────┘
```

---

## 核心数据类型

### 1. MarketData - 市场数据

**用途：** 表示单个时间点的市场状态

**字段说明：**

| 字段名 | 类型 | 说明 | 必需 |
|--------|------|------|------|
| timestamp | float | Unix时间戳（秒） | ✓ |
| open | float | 开盘价 | ✓ |
| high | float | 最高价 | ✓ |
| low | float | 最低价 | ✓ |
| close | float | 收盘价 | ✓ |
| volume | float | 成交量 | ✓ |
| bid_price | float | 买一价 | ✗ |
| ask_price | float | 卖一价 | ✗ |
| bid_volume | float | 买一量 | ✗ |
| ask_volume | float | 卖一量 | ✗ |
| vwap | float | 成交量加权平均价 | ✗ |

**示例：**
```python
market_data = MarketData(
    timestamp=1609459200.0,
    open=100.0,
    high=101.5,
    low=99.5,
    close=100.8,
    volume=1000000.0,
    bid_price=100.75,
    ask_price=100.85,
)
```

---

### 2. Action - 智能体动作

**用途：** 表示智能体在一个时间步的决策

**字段说明：**

| 字段名 | 类型 | 取值范围 | 说明 |
|--------|------|----------|------|
| trade_ratio | float | [0, 1] | 本步交易剩余目标的比例 |
| order_type | OrderType | enum | 订单类型（MARKET/LIMIT等） |
| limit_price_offset | float | [-1, 1] | 限价相对市价的偏移（%） |
| urgency | float | [0, 1] | 执行紧急程度 |

**含义解释：**

- `trade_ratio = 0.1`: 交易剩余目标数量的10%
- `trade_ratio = 0.0`: 本步不交易
- `trade_ratio = 1.0`: 立即交易所有剩余数量
- `limit_price_offset = 0.001`: 限价比市价高0.1%（买单）

**数组表示：**
```python
action_array = [trade_ratio, order_type_code, limit_price_offset, urgency]
# 例如: [0.1, 0, 0.001, 0.5]
```

---

### 3. Observation - 环境观察

**用途：** 智能体观察到的环境状态

**核心字段：**

| 分类 | 字段 | 类型 | 说明 |
|------|------|------|------|
| **市场数据** | market_data | MarketData | 当前市场状态 |
| **历史数据** | historical_prices | ndarray | 历史价格序列 |
| | historical_volumes | ndarray | 历史成交量序列 |
| **仓位信息** | position | float | 当前持仓数量 |
| | cash | float | 可用现金 |
| | target_quantity | float | 剩余目标数量 |
| **执行状态** | executed_quantity | float | 已执行数量 |
| | avg_execution_price | float | 平均执行价格 |
| | market_impact | float | 当前市场冲击估计 |
| **时间** | time_remaining | int | 剩余时间步数 |

**转换为数组：**
```python
obs_array = observation.to_array()
# Shape: (31,) = 11个基础特征 + 20个历史价格
```

**数组组成：**
```
[0]: close price
[1]: volume
[2]: price range (high-low)
[3-4]: historical price mean, std
[5]: position
[6]: target_quantity
[7]: execution progress (executed/target)
[8]: time_remaining
[9]: avg_execution_price
[10]: market_impact
[11-30]: last 20 historical prices
```

---

### 4. OrderInfo - 订单信息

**用途：** 表示一个订单的完整信息

**字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| order_id | str | 订单唯一标识 |
| symbol | str | 交易标的 |
| side | OrderSide | BUY/SELL |
| order_type | OrderType | MARKET/LIMIT/TWAP等 |
| quantity | float | 订单总量 |
| price | float | 限价（限价单） |
| filled_quantity | float | 已成交数量 |
| avg_fill_price | float | 平均成交价 |
| status | OrderStatus | 订单状态 |
| timestamp | float | 创建时间 |
| params | dict | 额外参数 |

**状态转换：**
```
PENDING -> PARTIALLY_FILLED -> FILLED
          └-> CANCELLED
          └-> REJECTED
```

---

### 5. State - 完整状态

**用途：** 环境的完整内部状态（比Observation更全面）

**字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| observation | Observation | 智能体可见的观察 |
| true_volatility | float | 真实波动率（不可见） |
| liquidity | float | 流动性状态 |
| order_book | dict | 订单簿（可选） |
| pending_orders | list | 待成交订单列表 |
| step | int | 当前步数 |
| done | bool | 是否结束 |
| info | dict | 其他信息 |

**Observation vs State:**
- Observation: 智能体能观察到的信息（部分可观测）
- State: 环境的完整状态（可能包含隐藏信息）

---

## 模块详细说明

### 1. Environment 模块

#### TradingEnvironment

**初始化参数：**

```python
env = TradingEnvironment(
    data_loader: BaseDataLoader,        # 数据加载器
    execution_engine: BaseExecutionEngine,  # 执行引擎
    reward_function: BaseReward,        # 奖励函数
    target_quantity: float,             # 目标交易量
    max_steps: int = 100,               # 最大步数
    initial_cash: float = 1000000.0,    # 初始现金
    side: str = "buy",                  # 交易方向
    symbol: str = "ASSET",              # 标的代码
    window_size: int = 20,              # 历史窗口大小
)
```

**核心方法：**

1. **reset() -> Observation**
   - 输入：可选的kwargs（如seed, start_idx）
   - 输出：初始观察
   - 作用：重置环境到初始状态

   ```python
   obs = env.reset(seed=42, start_idx=100)
   ```

2. **step(action) -> Tuple[Observation, float, bool, dict]**
   - 输入：Action对象
   - 输出：
     - observation: 新的观察
     - reward: float
     - done: bool
     - info: dict

   ```python
   obs, reward, done, info = env.step(action)
   ```

3. **get_metrics() -> dict**
   - 输入：无
   - 输出：当前episode的评估指标

   ```python
   metrics = env.get_metrics()
   # {
   #     'completion_rate': 0.95,
   #     'avg_execution_price': 100.5,
   #     'vwap': 100.3,
   #     'vwap_slippage': 0.002,
   #     'implementation_shortfall': 50.0,
   # }
   ```

**数据流：**

```
reset() 流程:
1. 加载市场数据 (data_loader.load_data)
2. 初始化历史价格窗口
3. 构建初始State
4. 返回observation

step() 流程:
1. 计算交易数量 (action.trade_ratio * remaining)
2. 创建订单 OrderInfo
3. 执行订单 (execution_engine.execute_order)
4. 更新持仓和现金
5. 计算奖励 (reward_function.calculate)
6. 构建新State
7. 返回 (obs, reward, done, info)
```

---

### 2. Data 模块

#### CSVDataLoader

**初始化：**
```python
loader = CSVDataLoader(
    data_dir: str,          # CSV文件目录
    normalize: bool = False # 是否标准化
)
```

**load_data() 方法：**

输入：
```python
data = loader.load_data(
    symbol: str,                    # 标的代码（文件名）
    start_time: Optional[float],    # 开始时间戳
    end_time: Optional[float],      # 结束时间戳
)
```

输出：
```python
List[MarketData]  # 市场数据列表
```

**CSV文件格式：**
```csv
timestamp,open,high,low,close,volume,bid_price,ask_price
1609459200,100.0,101.0,99.5,100.5,1000000,100.4,100.6
1609459260,100.5,101.2,100.0,100.8,1200000,100.7,100.9
...
```

#### SyntheticDataGenerator

**初始化：**
```python
generator = SyntheticDataGenerator(
    initial_price: float = 100.0,   # 初始价格
    mu: float = 0.0,                # 漂移率（年化）
    sigma: float = 0.2,             # 波动率（年化）
    dt: float = 60.0,               # 时间步长（秒）
    add_microstructure_noise: bool = True,
)
```

**数学模型：**
```
价格演化（几何布朗运动）：
dS = μS dt + σS dW

离散化：
S(t+1) = S(t) * exp((μ - σ²/2)dt + σ√dt * Z)
其中 Z ~ N(0,1)
```

**生成数据：**
```python
data = generator.load_data(
    symbol="SYNTHETIC",
    num_steps=1000,    # 生成1000个数据点
    seed=42,           # 随机种子
)
```

---

### 3. Execution 模块

#### SimpleExecutionEngine

**execute_order() 方法：**

输入：
```python
executed_qty, execution_price, updated_order = engine.execute_order(
    order: OrderInfo,        # 订单信息
    market_data: MarketData, # 当前市场数据
    liquidity: float = 1.0,  # 流动性系数 [0,1]
)
```

输出：
```python
(
    executed_quantity: float,  # 实际成交数量
    execution_price: float,    # 成交价格
    updated_order: OrderInfo,  # 更新后的订单
)
```

**市场冲击模型：**
```python
# 永久冲击
permanent_impact = k1 * (quantity / volume)^γ

# 临时冲击
temporary_impact = k2 * (quantity / volume)^δ

# 总冲击
total_impact = permanent_impact + temporary_impact
```

**滑点模型：**
```python
slippage = base_slippage * √(quantity/volume) * (1 + 10*volatility)
```

**执行价格计算：**
```python
# 买单
execution_price = reference_price * (1 + total_impact + slippage)

# 卖单
execution_price = reference_price * (1 - total_impact - slippage)
```

#### RealisticExecutionEngine

**增强特性：**

1. **订单簿模拟**
   ```python
   order_book = {
       'bids': [(price1, volume1), (price2, volume2), ...],
       'asks': [(price1, volume1), (price2, volume2), ...],
   }
   ```

2. **逐层成交**
   - 大额订单需要跨多个价格层成交
   - 每层有不同的可用流动性

3. **流动性恢复**
   - 市场冲击随时间衰减
   - 恢复速率可配置

---

### 4. Reward 模块

#### ImplementationShortfallReward

**calculate() 方法：**

输入：
```python
reward = reward_fn.calculate(
    state: State,              # 当前状态
    action: Action,            # 执行的动作
    next_state: State,         # 下一状态
    executed_quantity: float,  # 成交数量
    execution_price: float,    # 成交价格
)
```

输出：
```python
float  # 奖励值
```

**计算公式：**
```python
# 1. 执行成本
execution_cost = -(execution_price - arrival_price) / arrival_price

# 2. 时间惩罚
time_penalty = -time_weight / max(time_remaining, 1)

# 3. 完成奖励
if done and completion_rate > 0.95:
    completion_reward = completion_bonus
else:
    completion_reward = -completion_bonus * (1 - completion_rate)

# 4. 市场冲击惩罚
impact_penalty = -risk_aversion * market_impact

# 总奖励
reward = execution_cost + time_penalty + completion_reward + impact_penalty
```

#### PnLBasedReward

**计算公式：**
```python
# 投资组合价值
portfolio_value = cash + position * current_price

# PnL变化
pnl_change = portfolio_value - prev_portfolio_value

# 交易成本
transaction_cost = quantity * price * cost_bps / 10000

# 持仓惩罚
inventory_penalty = -penalty_coef * (remaining / target)

# 总奖励
reward = pnl_change - transaction_cost + inventory_penalty
```

---

### 5. Policy 模块

#### MLPPolicy

**网络结构：**
```
Input(obs_dim=31)
    ↓
Linear(31, 256) + ReLU
    ↓
Linear(256, 256) + ReLU
    ↓
Linear(256, 4) + Sigmoid
    ↓
Output(action_dim=4)
```

**forward() 方法：**

输入：
```python
# 单个observation
obs = np.array([...])  # shape: (31,)
action = policy.forward(obs)  # shape: (4,)

# 批次
obs_batch = np.array([...])  # shape: (batch_size, 31)
actions = policy.forward(obs_batch)  # shape: (batch_size, 4)
```

输出：
```python
np.ndarray  # shape: (action_dim,) or (batch_size, action_dim)
# 值域: [0, 1] (经过Sigmoid)
```

#### LSTMPolicy

**网络结构：**
```
Input(seq_len, obs_dim)
    ↓
LSTM(obs_dim, hidden_size, num_layers)
    ↓
取最后时间步输出
    ↓
Linear(hidden_size, 128) + ReLU
    ↓
Linear(128, action_dim) + Sigmoid
    ↓
Output(action_dim)
```

**适用场景：**
- 需要记忆长期依赖
- 时序特征重要
- 价格趋势预测

---

### 6. Agent 模块

#### DQNAgent

**核心算法：**
```
1. ε-greedy探索
   if random() < ε:
       action = random_action()
   else:
       action = argmax Q(s, a)

2. 经验回放
   buffer.push(s, a, r, s', done)

3. Q值更新
   target = r + γ * max Q'(s', a')
   loss = MSE(Q(s,a), target)

4. 目标网络更新
   Q'_params = Q_params (每N步)
```

**select_action() 方法：**

输入：
```python
action = agent.select_action(
    observation: Observation,
    training: bool = True,
)
```

输出：
```python
Action  # Action对象
```

**update() 方法：**

输入：
```python
metrics = agent.update(
    batch: Optional[TransitionBatch] = None
)
# 如果batch为None，从replay buffer采样
```

输出：
```python
dict  # 训练指标
# {
#     'loss': 0.123,
#     'epsilon': 0.5,
#     'q_value': 10.5,
# }
```

---

### 7. Trainer 模块

#### RLTrainer

**train() 方法：**

输入：
```python
history = trainer.train(
    num_episodes: int,                    # episode数量
    max_steps_per_episode: Optional[int], # 每episode最大步数
)
```

输出：
```python
dict  # 训练历史
# {
#     'episode_rewards': [r1, r2, ...],
#     'episode_lengths': [l1, l2, ...],
#     'completion_rates': [c1, c2, ...],
#     'avg_execution_prices': [p1, p2, ...],
#     'losses': [loss1, loss2, ...],
# }
```

**训练循环伪代码：**
```python
for episode in range(num_episodes):
    obs = env.reset()
    done = False
    episode_reward = 0

    while not done:
        # 1. 选择动作
        action = agent.select_action(obs, training=True)

        # 2. 执行动作
        next_obs, reward, done, info = env.step(action)

        # 3. 存储经验
        agent.store_transition(obs, action, reward, next_obs, done)

        # 4. 更新策略
        metrics = agent.update()

        # 5. 累积奖励
        episode_reward += reward
        obs = next_obs

    # 6. 记录指标
    history['episode_rewards'].append(episode_reward)
```

**evaluate() 方法：**

输入：
```python
metrics = trainer.evaluate(
    num_episodes: int,
    deterministic: bool = True,
)
```

输出：
```python
EpisodeMetrics  # 评估指标汇总
```

---

### 8. Evaluator 模块

#### TradingEvaluator

**evaluate_episode() 方法：**

输入：
```python
metrics = evaluator.evaluate_episode(
    agent: BaseAgent,
    env: BaseEnvironment,
    render: bool = False,
)
```

输出：
```python
EpisodeMetrics  # 单个episode的完整指标
# 包含：
# - total_reward
# - execution_shortfall
# - vwap_slippage
# - completion_rate
# - avg_execution_price
# - sharpe_ratio
# - ...
```

**generate_report() 方法：**

输入：
```python
report = evaluator.generate_report(
    metrics_list: List[EpisodeMetrics],
    output_path: Optional[str] = "report.json",
)
```

输出：
```python
dict  # 汇总统计报告
# {
#     'num_episodes': 10,
#     'total_reward': {
#         'mean': 100.5,
#         'std': 10.2,
#         'min': 80.0,
#         'max': 120.0,
#     },
#     'completion_rate': {...},
#     'vwap_slippage': {...},
#     ...
# }
```

---

## 典型使用流程

### 完整训练流程

```python
# 1. 准备数据
data_gen = SyntheticDataGenerator(initial_price=100.0, sigma=0.2)

# 2. 配置组件
execution_engine = SimpleExecutionEngine()
reward_fn = ImplementationShortfallReward()

# 3. 创建环境
env = TradingEnvironment(
    data_loader=data_gen,
    execution_engine=execution_engine,
    reward_function=reward_fn,
    target_quantity=10000,
    max_steps=50,
)

# 4. 创建策略和智能体
obs_dim = env.observation_space_shape[0]  # 31
action_dim = env.action_space_shape[0]    # 4

policy = MLPPolicy(obs_dim=obs_dim, action_dim=action_dim)
agent = DQNAgent(policy=policy)

# 5. 训练
trainer = RLTrainer(env=env, agent=agent)
history = trainer.train(num_episodes=100)

# 6. 评估
evaluator = TradingEvaluator()
metrics_list = []
for _ in range(10):
    metrics = evaluator.evaluate_episode(agent, env)
    metrics_list.append(metrics)

report = evaluator.generate_report(metrics_list)

# 7. 保存模型
agent.save("models/trained_agent.pth")
```

### 数据流追踪

```
Episode开始:
├── env.reset()
│   ├── data_loader.load_data() -> List[MarketData]
│   ├── 初始化历史窗口
│   └── 返回 Observation
│
├── 循环: while not done
│   │
│   ├── agent.select_action(obs) -> Action
│   │   ├── obs.to_array() -> np.ndarray(31,)
│   │   ├── policy.forward() -> np.ndarray(4,)
│   │   └── Action.from_array() -> Action
│   │
│   ├── env.step(action) -> (obs', reward, done, info)
│   │   ├── 计算交易数量
│   │   ├── 创建 OrderInfo
│   │   ├── execution_engine.execute_order()
│   │   │   ├── 输入: OrderInfo, MarketData
│   │   │   ├── 计算市场冲击和滑点
│   │   │   └── 输出: (qty, price, order)
│   │   ├── 更新持仓和现金
│   │   ├── reward_fn.calculate() -> reward
│   │   └── 构建新 State
│   │
│   ├── agent.store_transition(s, a, r, s', done)
│   │   └── replay_buffer.push()
│   │
│   └── agent.update() -> metrics
│       ├── replay_buffer.sample() -> batch
│       ├── 计算TD loss
│       ├── policy.update_parameters(loss)
│       └── 更新target network
│
└── Episode结束
    ├── env.get_metrics() -> dict
    └── 记录到history
```

---

## 性能优化建议

### 1. 数据加载优化

```python
# 使用缓存
loader = CSVDataLoader(data_dir="./data")
# 第一次加载会读取文件
data1 = loader.load_data("AAPL")
# 第二次直接从缓存读取
data2 = loader.load_data("AAPL")  # 快速
```

### 2. 批量处理

```python
# 策略网络支持批量前向传播
obs_batch = np.stack([obs1, obs2, ...])  # (batch_size, obs_dim)
actions_batch = policy.forward(obs_batch)  # 一次处理多个
```

### 3. GPU加速

```python
# 创建policy和agent时指定device
policy = MLPPolicy(obs_dim=31, action_dim=4, device='cuda')
agent = DQNAgent(policy=policy, device='cuda')
```

### 4. 经验回放优化

```python
# 适当调整buffer大小和batch size
agent = DQNAgent(
    policy=policy,
    buffer_size=100000,  # 根据内存调整
    batch_size=256,      # 增大batch提高效率
)
```

---

## 调试技巧

### 1. 观察环境状态

```python
obs = env.reset()
print(f"Observation shape: {obs.to_array().shape}")
print(f"Market price: {obs.market_data.close}")
print(f"Target quantity: {obs.target_quantity}")
```

### 2. 检查动作合理性

```python
action = agent.select_action(obs)
print(f"Trade ratio: {action.trade_ratio}")
print(f"Trade quantity: {action.trade_ratio * obs.target_quantity}")
```

### 3. 监控奖励计算

```python
# 在reward function中添加日志
def calculate(self, state, action, next_state, qty, price):
    reward = ...
    print(f"Reward components: cost={cost}, penalty={penalty}")
    return reward
```

### 4. 可视化训练曲线

```python
trainer.plot_training_history(save_path="training.png")
```

---

## 常见错误和解决方案

### 1. 观察维度不匹配

**错误：** `RuntimeError: size mismatch`

**原因：** policy的obs_dim与实际observation维度不符

**解决：**
```python
obs_dim = env.observation_space_shape[0]  # 使用环境提供的维度
policy = MLPPolicy(obs_dim=obs_dim, action_dim=4)
```

### 2. 数据不足

**错误：** `ValueError: 市场数据不足`

**原因：** 数据长度 < max_steps + window_size

**解决：**
```python
# 生成足够的数据
data = generator.load_data(num_steps=max_steps + window_size + 100)
```

### 3. 奖励爆炸/消失

**症状：** 奖励值过大或接近0

**解决：**
```python
# 使用标准化
reward_fn = ImplementationShortfallReward(normalize=True)

# 或调整奖励权重
reward_fn = CompositeReward([
    (reward1, 0.5),
    (reward2, 0.5),
])
```

---

## 总结

本文档详细说明了框架中各模块的：
- 输入/输出格式
- 数据类型定义
- 计算公式
- 使用示例
- 数据流转

使用时请参考相应章节，遇到问题可查阅调试技巧部分。
