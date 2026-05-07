"""
完整因子研究与回测主程序（简化版）

运行完整的量化因子研究流程:
1. 数据准备
2. 候选因子生成
3. 因子评估
4. 因子筛选
5. 回测验证
6. 报告生成
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime
import json

from src.data.data_provider import MockDataProvider
from src.factors.auto.enhanced_generator import EnhancedFactorGenerator
from src.factors.auto.factor_evaluator import FactorEvaluator
from src.core.backtest import CrossSectionBacktestEngine
from src.factors.auto.factor_candidate import FactorCandidate
from src.factors.auto.factor_gatekeeper import FactorGatekeeper


def run_full_pipeline():
    """运行完整的因子研究流水线"""

    print("="*60)
    print("MyQuant AI量化选股系统 - 因子研究与回测")
    print("="*60)

    print("\n[Step 1] 初始化数据...")
    provider = MockDataProvider(n_stocks=30, n_days=200)
    features = provider.get_all_features()
    returns = provider.get_returns()

    print(f"  - 数据范围: {features.index.get_level_values(0).min().date()} 至 {features.index.get_level_values(0).max().date()}")
    print(f"  - 股票数量: {len(features.index.get_level_values(1).unique())}")

    print("\n[Step 2] 生成候选因子...")
    generator = EnhancedFactorGenerator()
    factors = generator.generate_all_factors(features, generate_neutral=False)
    print(f"  - 生成候选因子: {len(factors)} 个")

    print("\n[Step 3] 评估候选因子...")
    evaluator = FactorEvaluator()

    evaluated_factors = {}
    evaluation_results = []

    factor_items = list(factors.items())[:30]

    for i, (name, factor_data) in enumerate(factor_items):
        try:
            eval_result = evaluator.evaluate_single(factor_data, returns)

            evaluated_factors[name] = {
                'factor_data': factor_data,
                'evaluation': eval_result
            }

            evaluation_results.append({
                'factor_name': name,
                'rank_ic_mean': eval_result['rank_ic']['mean'],
                'icir': eval_result.get('icir', 0),
                'coverage': eval_result.get('coverage', 0)
            })

            if (i + 1) % 10 == 0:
                print(f"  - 已评估 {i+1}/{len(factor_items)} 个因子...")

        except Exception as e:
            print(f"  - 评估 {name} 失败: {e}")

    eval_df = pd.DataFrame(evaluation_results)
    print(f"  - 成功评估: {len(eval_df)} 个因子")
    if len(eval_df) > 0:
        print(f"  - 平均 RankIC: {eval_df['rank_ic_mean'].mean():.4f}")

    print("\n[Step 4] 筛选有效因子...")

    valid_factors = [
        name for name, data in evaluated_factors.items()
        if abs(data['evaluation']['rank_ic']['mean']) > 0.01
    ][:10]

    print(f"  - 筛选后有效因子: {len(valid_factors)} 个")

    if valid_factors:
        top_5 = sorted(
            [(name, evaluated_factors[name]['evaluation']['rank_ic']['mean'])
             for name in valid_factors],
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]

        print("  - Top 5 因子:")
        for name, ic in top_5:
            print(f"    {name}: IC={ic:.4f}")

    print("\n[Step 5] 回测验证...")
    if valid_factors:
        engine = CrossSectionBacktestEngine()

        best_factor_name = valid_factors[0]
        best_factor = evaluated_factors[best_factor_name]['factor_data']

        print(f"  - 回测因子: {best_factor_name}")
        bt_result = engine.run(best_factor, returns, n_groups=10, long_short=True)

        print(f"  - 总收益: {bt_result['total_return']*100:.2f}%")
        print(f"  - Sharpe: {bt_result['sharpe_ratio']:.2f}")
        print(f"  - 最大回撤: {bt_result['max_drawdown']*100:.2f}%")

        group_returns = bt_result['group_returns']
        if group_returns:
            print("  - 分组收益:")
            for g, ret in sorted(group_returns.items()):
                print(f"    Group {g}: {ret*100:.2f}%")
    else:
        bt_result = None

    print("\n[Step 6] 因子入库审核...")
    gatekeeper = FactorGatekeeper()

    approved_factors = []
    for name in valid_factors[:5]:
        eval_result = evaluated_factors[name]['evaluation']

        from src.factors.auto.factor_candidate import FactorCandidate
        candidate = FactorCandidate(
            expression=name,
            name=name,
            description=f'Auto generated factor: {name}',
            source='auto'
        )
        candidate.set_evaluation_results(eval_result)

        approval = gatekeeper.approve_or_reject(candidate)

        if approval['approved']:
            approved_factors.append(name)
            print(f"  - 审核通过: {name}")

    print(f"  - 审核通过因子: {len(approved_factors)} 个")

    print("\n" + "="*60)
    print("研究流程完成!")
    print("="*60)

    print(f"\n结果摘要:")
    print(f"  - 候选因子: {len(factors)} 个")
    print(f"  - 评估通过: {len(evaluated_factors)} 个")
    print(f"  - 筛选有效: {len(valid_factors)} 个")
    print(f"  - 入库审核: {len(approved_factors)} 个")

    if valid_factors:
        best_factor = valid_factors[0]
        print(f"\n最佳因子: {best_factor}")
        print(f"  - RankIC: {evaluated_factors[best_factor]['evaluation']['rank_ic']['mean']:.4f}")

    os.makedirs('research_results', exist_ok=True)

    result_file = 'research_results/pipeline_result.json'
    result = {
        'timestamp': datetime.now().isoformat(),
        'factors_generated': len(factors),
        'factors_evaluated': len(evaluated_factors),
        'factors_selected': len(valid_factors),
        'factors_approved': len(approved_factors),
        'best_factor': valid_factors[0] if valid_factors else None,
        'backtest': bt_result
    }

    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n结果已保存至: {result_file}")

    return result


if __name__ == '__main__':
    result = run_full_pipeline()
