# 真实历史数据使用指南

## 概述

本框架已完整支持真实Level-2高频市场数据，包括中国A股的10档买卖盘口数据。

## 支持的数据格式

### Market Data CSV 格式

文件命名: `{symbol}_{date}_market_data.csv`

例如: `000001_20170105_market_data.csv`

### 必要字段

| 字段 | 说明 | 示例 |
|------|------|------|
| `symbol` | 股票代码 | 000001 |
| `datetime` | 时间戳 | 2017-01-05 09:30:01 |
| `last_prc` | 最新价格 | 10.25 |
| `volume` | 累计成交量（股） | 1000000 |
| `open` | 开盘价 | 10.20 |
| `high` | 最高价 | 10.30 |
| `low` | 最低价 | 10.15 |

### 可选字段（Level-2数据）

| 字段 | 说明 |
|------|------|
| `bid_prc1-10` | 买一至买十价格 |
| `bid_vol1-10` | 买一至买十数量 |
| `ask_prc1-10` | 卖一至卖十价格 |
| `ask_vol1-10` | 卖一至卖十数量 |
| `prev_close` | 昨收价 |
| `turnover` | 成交额 |
| `num_trades` | 成交笔数 |
| `high_limited` | 涨停价 |
| `low_limited` | 跌停价 |
| `weighted_bid_prc` | 加权买价 |
| `weighted_ask_prc` | 加权卖价 |
| `total_bid_vol` | 买盘总量 |
| `total_ask_vol` | 卖盘总量 |
| `status` | 交易状态 |
| `exchtime` | 交易所时间戳（微秒） |
| `localtime` | 本地时间戳（微秒） |

### 交易状态说明

| 状态码 | 说明 |
|--------|------|
| 83 | 开市前 |
| 67 | 集合竞价 |
| 84 | 连续交易 |
| 69 | 闭市 |
| 66 | 休市 |

## 快速开始

### 1. 准备数据

将你的market_data CSV文件放在一个目录下，例如：

```
/home/user/market_data/
├── 000001_20170105_market_data.csv
├── 000001_20170106_market_data.csv
└── 000002_20170105_market_data.csv
```

### 2. 测试数据加载

编辑 `examples/test_historical_data.py`，设置数据路径：

```python
DATA_DIR = "/home/user/market_data"  # 你的数据目录
SYMBOL = "000001"                     # 股票代码
DATE = "20170105"                     # 日期
```

运行测试：

```bash
python examples/test_historical_data.py
```

测试将执行以下检查：
- ✓ 数据加载
- ✓ 数据验证
- ✓ 数据统计分析
- ✓ 数据预处理
- ✓ 简单回测

### 3. 运行完整训练

编辑 `examples/example_real_data_training.py`，设置相同的数据路径，然后运行：

```bash
python examples/example_real_data_training.py
```

这将：
1. 加载并验证数据
2. 创建训练环境
3. 使用启发式策略（TWAP/VWAP/Adaptive）进行评估
4. 对比不同策略的表现

## API 使用

### 基础用法

```python
from rl_trading_framework.data.historical_data import HistoricalDataLoader

# 创建数据加载器
loader = HistoricalDataLoader(
    data_dir="/path/to/data",
    default_symbol="000001",
    default_date="20170105",
    trading_hours_only=True,  # 只加载交易时段
    subsample_freq=None,       # 不降采样
)

# 加载数据
data = loader.load_data(
    symbol="000001",
    date="20170105",
    num_steps=1000,           # 可选：只加载前1000条
    start_time="09:30:00",    # 可选：起始时间
    end_time="15:00:00",      # 可选：结束时间
)

print(f"加载了 {len(data)} 条数据")
```

### 数据验证

```python
from rl_trading_framework.data.data_utils import (
    DataValidator,
    DataAnalyzer,
    print_data_summary
)

# 快速查看数据摘要
print_data_summary(data)

# 详细验证
result = DataValidator.validate_data_list(data)
print(f"数据有效: {result['valid']}")
print(f"无效记录: {result['invalid_records']}")

# 统计分析
stats = DataAnalyzer.get_basic_stats(data)
print(f"平均价格: {stats['price']['mean']:.4f}")
print(f"价格波动率: {stats['returns']['std']*100:.2f}%")

# 异常检测
anomalies = DataAnalyzer.detect_anomalies(data, threshold=3.0)
print(f"发现 {len(anomalies)} 个异常数据点")
```

### 数据预处理

```python
from rl_trading_framework.data.data_utils import DataPreprocessor

# 只保留连续交易时段
filtered_data = DataPreprocessor.filter_trading_hours(
    data,
    status_filter=[84]  # 84 = 连续交易
)

# 降采样（每10条取1条）
resampled_data = DataPreprocessor.resample(data, freq=10)

# 填充缺失值
filled_data = DataPreprocessor.fill_missing_values(data)
```

### 在训练中使用

```python
from rl_trading_framework.environments.trading_env import TradingEnvironment
from rl_trading_framework.execution.simple_execution import SimpleExecutionEngine
from rl_trading_framework.rewards.implementation_shortfall import ImplementationShortfallReward

# 简单包装器
class SimpleDataLoader:
    def __init__(self, data):
        self.data = data

    def load_data(self, symbol, **kwargs):
        return self.data

# 创建环境
env = TradingEnvironment(
    data_loader=SimpleDataLoader(data),
    execution_engine=SimpleExecutionEngine(),
    reward_function=ImplementationShortfallReward(),
    target_quantity=10000.0,
    max_steps=20,
    initial_cash=1000000.0,
    side="buy",
)

# 运行episode
obs = env.reset()
done = False
total_reward = 0.0

while not done:
    # 你的策略逻辑
    action = your_policy.select_action(obs)

    # 执行动作
    obs, reward, done, info = env.step(action)
    total_reward += reward

# 获取指标
metrics = env.get_metrics()
print(f"完成率: {metrics['completion_rate']*100:.1f}%")
print(f"VWAP滑点: {metrics['vwap_slippage']*10000:.2f} bps")
```

## 数据质量检查

### 检查清单

运行测试脚本后，确认以下内容：

- [ ] 数据加载成功（无FileNotFoundError）
- [ ] 数据验证通过（valid=True）
- [ ] 时间戳单调递增
- [ ] 价格数据合理（OHLC逻辑正确）
- [ ] 买卖盘价格合理（bid < ask）
- [ ] 10档行情正确（价格递增/递减）
- [ ] 无异常数据点或异常可解释

### 常见问题

**Q: FileNotFoundError: 数据文件不存在**

A: 检查：
1. 文件路径是否正确
2. 文件命名格式是否为 `{symbol}_{date}_market_data.csv`
3. 文件权限是否可读

**Q: 数据验证失败（买一价 > 卖一价）**

A: 这可能是数据质量问题，检查：
1. 原始CSV中的bid_prc1和ask_prc1列
2. 是否有缺失值或异常值
3. 考虑使用数据清洗功能

**Q: 价格跳变过大**

A: Level-2数据可能包含集合竞价等特殊时段，建议：
1. 使用 `trading_hours_only=True` 只加载连续交易时段
2. 使用 `filter_trading_hours` 过滤特定交易状态
3. 检查是否有停牌、涨跌停等情况

**Q: 内存占用过大**

A: 对于大文件，建议：
1. 使用 `num_steps` 参数限制加载条数
2. 使用 `subsample_freq` 降采样
3. 分批加载和处理

## 高级功能

### 多日期数据

```python
# 获取可用日期列表
dates = loader.get_available_dates("000001")
print(f"可用日期: {dates}")

# 循环加载多个日期
for date in dates:
    data = loader.load_data(symbol="000001", date=date)
    # 处理数据...
```

### 自定义数据字段

如果你的CSV有额外字段，可以通过 `MarketData.extra_features` 字段存储：

```python
# 在 HistoricalDataLoader._row_to_market_data 中添加：
extra_features = {}
if 'custom_field' in row:
    extra_features['custom_field'] = row['custom_field']

market_data = MarketData(
    ...,
    # extra_features=extra_features  # 当前版本Observation支持
)
```

### Level-2 特征提取

```python
# 使用10档行情数据
md = data[0]

if md.bid_prices is not None:
    # 订单簿深度
    total_bid_depth = sum(md.bid_volumes)
    total_ask_depth = sum(md.ask_volumes)

    # 订单簿不平衡
    imbalance = (total_bid_depth - total_ask_depth) / (total_bid_depth + total_ask_depth)

    # 加权中间价
    weighted_mid = (md.weighted_bid_price + md.weighted_ask_price) / 2

    # 多档价差
    spread_5 = md.ask_prices[4] - md.bid_prices[4]  # 第5档价差
```

## 性能优化

### 数据缓存

`HistoricalDataLoader` 自动缓存已加载的数据：

```python
# 第一次加载（从CSV读取）
data1 = loader.load_data("000001", "20170105")  # 慢

# 第二次加载（从缓存读取）
data2 = loader.load_data("000001", "20170105")  # 快
```

### 降采样建议

根据你的训练需求选择合适的采样频率：

- **高频训练**（tick级）: `subsample_freq=1`（不降采样）
- **1秒数据**: `subsample_freq=10`（假设原始是100ms）
- **5秒数据**: `subsample_freq=50`
- **1分钟数据**: `subsample_freq=600`

## 后续扩展

### Transaction 和 Order 数据支持

当前版本主要支持market_data，未来可以扩展：

1. **Transaction数据**：用于更精确的成交模拟
2. **Order数据**：用于订单簿重构
3. **多标的交易**：同时交易多个股票

### 贡献数据加载器

如果你有其他数据源（如美股、期货等），欢迎贡献新的DataLoader！

参考 `HistoricalDataLoader` 的实现，继承 `BaseDataLoader` 并实现 `load_data` 方法。

## 联系与反馈

如有问题或建议，请：
1. 检查本文档和示例代码
2. 查看 `MODULE_DOCUMENTATION.md` 了解详细的模块说明
3. 提交Issue或Pull Request

祝训练顺利！🚀
