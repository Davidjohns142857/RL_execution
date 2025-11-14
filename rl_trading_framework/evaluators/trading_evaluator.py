"""
交易评估器

评估交易策略性能并生成详细报告
"""

import numpy as np
from typing import List, Dict, Any, Optional
import json

from rl_trading_framework.core.base import BaseEvaluator, BaseAgent, BaseEnvironment
from rl_trading_framework.core.types import EpisodeMetrics


class TradingEvaluator(BaseEvaluator):
    """
    交易评估器

    输入：
        - agent: 智能体
        - env: 环境

    输出：
        - metrics: 评估指标
        - report: 详细报告

    评估指标：
        - Implementation Shortfall
        - VWAP Slippage
        - Completion Rate
        - Sharpe Ratio
        - 等
    """

    def __init__(self):
        """初始化评估器"""
        pass

    def evaluate_episode(
        self,
        agent: BaseAgent,
        env: BaseEnvironment,
        render: bool = False,
        **kwargs
    ) -> EpisodeMetrics:
        """
        评估单个episode

        Args:
            agent: 智能体
            env: 环境
            render: 是否渲染
            **kwargs: 其他参数

        Returns:
            episode指标
        """
        obs = env.reset(**kwargs)
        total_reward = 0.0
        steps = 0
        done = False

        # 记录执行历史
        execution_prices = []
        market_prices = []

        while not done:
            # 选择动作
            action = agent.select_action(obs, training=False)

            # 执行
            next_obs, reward, done, info = env.step(action)

            total_reward += reward
            steps += 1

            # 记录价格
            if info.get('execution_price', 0) > 0:
                execution_prices.append(info['execution_price'])
            market_prices.append(info.get('market_price', 0))

            obs = next_obs

            if render:
                env.render()

        # 获取环境指标
        env_metrics = env.get_metrics() if hasattr(env, 'get_metrics') else {}

        # 计算夏普比率（简化版）
        sharpe_ratio = None
        if len(execution_prices) > 1 and len(market_prices) > 1:
            returns = np.diff(market_prices) / market_prices[:-1]
            if np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)

        # 构建指标
        metrics = EpisodeMetrics(
            total_reward=total_reward,
            execution_shortfall=env_metrics.get('implementation_shortfall', 0.0),
            vwap_slippage=env_metrics.get('vwap_slippage', 0.0),
            completion_rate=env_metrics.get('completion_rate', 0.0),
            avg_execution_price=env_metrics.get('avg_execution_price', 0.0),
            total_cost=env_metrics.get('total_cost', 0.0),
            sharpe_ratio=sharpe_ratio,
            steps=steps,
            extra_metrics={
                'num_executions': env_metrics.get('num_executions', 0),
            }
        )

        return metrics

    def generate_report(
        self,
        metrics_list: List[EpisodeMetrics],
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        生成评估报告

        Args:
            metrics_list: episode指标列表
            output_path: 报告保存路径

        Returns:
            汇总统计
        """
        if not metrics_list:
            return {}

        # 提取各项指标
        rewards = [m.total_reward for m in metrics_list]
        completion_rates = [m.completion_rate for m in metrics_list]
        vwap_slippages = [m.vwap_slippage for m in metrics_list]
        shortfalls = [m.execution_shortfall for m in metrics_list]
        sharpe_ratios = [m.sharpe_ratio for m in metrics_list if m.sharpe_ratio is not None]

        # 汇总统计
        report = {
            'num_episodes': len(metrics_list),
            'total_reward': {
                'mean': float(np.mean(rewards)),
                'std': float(np.std(rewards)),
                'min': float(np.min(rewards)),
                'max': float(np.max(rewards)),
            },
            'completion_rate': {
                'mean': float(np.mean(completion_rates)),
                'std': float(np.std(completion_rates)),
                'min': float(np.min(completion_rates)),
                'max': float(np.max(completion_rates)),
            },
            'vwap_slippage': {
                'mean': float(np.mean(vwap_slippages)),
                'std': float(np.std(vwap_slippages)),
                'min': float(np.min(vwap_slippages)),
                'max': float(np.max(vwap_slippages)),
            },
            'implementation_shortfall': {
                'mean': float(np.mean(shortfalls)),
                'std': float(np.std(shortfalls)),
            },
        }

        if sharpe_ratios:
            report['sharpe_ratio'] = {
                'mean': float(np.mean(sharpe_ratios)),
                'std': float(np.std(sharpe_ratios)),
            }

        # 保存报告
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"评估报告已保存: {output_path}")

        # 打印摘要
        self._print_summary(report)

        return report

    def _print_summary(self, report: Dict[str, Any]):
        """打印报告摘要"""
        print("\n" + "="*50)
        print("评估报告摘要")
        print("="*50)
        print(f"\n总Episode数: {report['num_episodes']}")

        print(f"\n总奖励:")
        print(f"  均值: {report['total_reward']['mean']:.4f}")
        print(f"  标准差: {report['total_reward']['std']:.4f}")
        print(f"  范围: [{report['total_reward']['min']:.4f}, {report['total_reward']['max']:.4f}]")

        print(f"\n完成率:")
        print(f"  均值: {report['completion_rate']['mean']:.2%}")
        print(f"  标准差: {report['completion_rate']['std']:.2%}")

        print(f"\nVWAP滑点:")
        print(f"  均值: {report['vwap_slippage']['mean']:.4f}")
        print(f"  标准差: {report['vwap_slippage']['std']:.4f}")

        print(f"\nImplementation Shortfall:")
        print(f"  均值: {report['implementation_shortfall']['mean']:.4f}")

        if 'sharpe_ratio' in report:
            print(f"\nSharpe Ratio:")
            print(f"  均值: {report['sharpe_ratio']['mean']:.4f}")

        print("="*50 + "\n")

    def compare_strategies(
        self,
        strategies: Dict[str, List[EpisodeMetrics]],
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        比较多个策略

        Args:
            strategies: {策略名: 指标列表}
            output_path: 对比报告保存路径

        Returns:
            对比结果
        """
        comparison = {}

        for name, metrics_list in strategies.items():
            report = self.generate_report(metrics_list)
            comparison[name] = report

        # 打印对比
        print("\n" + "="*50)
        print("策略对比")
        print("="*50)

        for name in strategies.keys():
            print(f"\n{name}:")
            print(f"  平均奖励: {comparison[name]['total_reward']['mean']:.4f}")
            print(f"  平均完成率: {comparison[name]['completion_rate']['mean']:.2%}")
            print(f"  平均VWAP滑点: {comparison[name]['vwap_slippage']['mean']:.4f}")

        # 保存对比报告
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(comparison, f, indent=2)
            print(f"\n对比报告已保存: {output_path}")

        return comparison
