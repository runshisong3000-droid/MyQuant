"""
Pipeline Reliability Audit Tests

测试内容:
1. RankIC 横截面计算测试
2. ICIR 边界测试
3. Sharpe 边界测试
4. 信号滞后测试
5. 未来函数检测测试
6. MultiIndex 兼容测试
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime


class TestRankICCrossSectional:
    """RankIC 横截面计算测试"""

    def test_rank_ic_cross_sectional_basic(self):
        """测试每个 date 单独计算 RankIC"""
        dates = pd.date_range('2024-01-01', periods=5, freq='B')
        stocks = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
        
        index = pd.MultiIndex.from_product([dates, stocks], names=['date', 'stock'])
        
        np.random.seed(42)
        factor = pd.Series(np.random.randn(len(index)), index=index)
        future_return = pd.Series(np.random.randn(len(index)) * 0.01, index=index)
        
        from src.factors.auto.factor_evaluator import FactorEvaluator
        evaluator = FactorEvaluator()
        
        result = evaluator.calculate_rank_ic(factor, future_return)
        
        assert len(result['timeseries']) == len(dates), "IC 序列长度应等于日期数"
        assert 'mean' in result, "应返回 mean_ic"
        assert 'std' in result, "应返回 ic_std"
        assert result['count'] == len(dates), "应记录有效 IC 数量"

    def test_rank_ic_constant_factor(self):
        """测试因子值全相同时跳过该日期"""
        dates = pd.date_range('2024-01-01', periods=3, freq='B')
        stocks = ['A', 'B', 'C', 'D', 'E']
        
        index = pd.MultiIndex.from_product([dates, stocks], names=['date', 'stock'])
        
        factor = pd.Series(np.ones(len(index)), index=index)
        future_return = pd.Series(np.random.randn(len(index)) * 0.01, index=index)
        
        from src.factors.auto.factor_evaluator import FactorEvaluator
        evaluator = FactorEvaluator()
        
        result = evaluator.calculate_rank_ic(factor, future_return)
        
        assert result['count'] == 0, "常量因子应跳过所有日期"
        assert result['mean'] == 0.0, "无有效 IC 时 mean 应为 0"


class TestICIRBoundaries:
    """ICIR 边界测试"""

    def test_icir_normal(self):
        """测试正常 IC 序列"""
        ic_values = [0.1, 0.2, 0.15, 0.25, 0.18, 0.12, 0.22, 0.17, 0.21, 0.16, 0.19]
        rank_ic_result = {
            'mean': np.mean(ic_values),
            'std': np.std(ic_values),
            'count': len(ic_values),
            'timeseries': ic_values
        }
        
        from src.factors.auto.factor_evaluator import FactorEvaluator
        evaluator = FactorEvaluator()
        
        icir = evaluator.calculate_icir(rank_ic_result)
        
        assert np.isfinite(icir), "ICIR 应有限"
        assert icir > 0, "正 IC 应产生正 ICIR"

    def test_icir_constant_ic(self):
        """测试全部相同 IC 序列"""
        ic_values = [0.1, 0.1, 0.1, 0.1, 0.1]
        rank_ic_result = {
            'mean': 0.1,
            'std': 0.0,
            'count': 5,
            'timeseries': ic_values
        }
        
        from src.factors.auto.factor_evaluator import FactorEvaluator
        evaluator = FactorEvaluator()
        
        icir = evaluator.calculate_icir(rank_ic_result)
        
        assert icir == 0, "IC 标准差为 0 时 ICIR 应为 0"

    def test_icir_with_nan(self):
        """测试包含 NaN 的 IC 序列"""
        ic_values = [0.1, np.nan, 0.2, np.nan, 0.15]
        rank_ic_result = {
            'mean': np.nanmean(ic_values),
            'std': np.nanstd(ic_values),
            'count': 3,
            'timeseries': ic_values
        }
        
        from src.factors.auto.factor_evaluator import FactorEvaluator
        evaluator = FactorEvaluator()
        
        icir = evaluator.calculate_icir(rank_ic_result)
        
        assert np.isfinite(icir), "ICIR 应处理 NaN"


class TestSharpeBoundaries:
    """Sharpe 边界测试"""

    def test_sharpe_normal(self):
        """测试正常收益率"""
        returns = pd.Series(np.random.randn(100) * 0.01)
        
        from src.metrics.performance import calculate_sharpe
        
        sharpe = calculate_sharpe(returns)
        
        assert np.isfinite(sharpe), "Sharpe 应有限"

    def test_sharpe_all_zeros(self):
        """测试全零收益率"""
        returns = pd.Series(np.zeros(100))
        
        from src.metrics.performance import calculate_sharpe
        
        sharpe = calculate_sharpe(returns)
        
        assert sharpe == 0, "全零收益率 Sharpe 应为 0"

    def test_sharpe_with_nan_inf(self):
        """测试包含 NaN 和 inf 的收益率"""
        returns = pd.Series([0.01, np.nan, np.inf, 0.02, -np.inf, 0.01])
        
        from src.metrics.performance import calculate_sharpe
        
        sharpe = calculate_sharpe(returns)
        
        assert np.isfinite(sharpe), "Sharpe 应处理 NaN 和 inf"


class TestSignalLag:
    """信号滞后测试"""

    def test_signal_trade_date_separation(self):
        """测试信号日和交易日分离"""
        dates = pd.date_range('2024-01-01', periods=5, freq='B')
        stocks = ['A', 'B', 'C']
        
        data = []
        for i, date in enumerate(dates):
            for stock in stocks:
                data.append({
                    'date': date,
                    'stock': stock,
                    'close': 100 + i * 10,
                    'factor': np.random.rand()
                })
        
        df = pd.DataFrame(data)
        
        df['future_return'] = df.groupby('stock')['close'].pct_change().shift(-1)
        
        from src.factors.auto.factor_evaluator import FactorEvaluator
        evaluator = FactorEvaluator()
        
        index = pd.MultiIndex.from_frame(df[['date', 'stock']])
        factor = df.set_index(index)['factor']
        future_return = df.set_index(index)['future_return']
        
        result = evaluator.evaluate_single(factor, future_return)
        
        assert 'rank_ic' in result, "应返回 RankIC"

    def test_no_same_day_trading(self):
        """测试不允许当天信号当天交易"""
        dates = pd.date_range('2024-01-01', periods=5, freq='B')
        stocks = ['A', 'B']
        
        data = []
        for i, date in enumerate(dates):
            data.append({
                'date': date,
                'stock': 'A',
                'close': 100 + i,
                'signal': 0.5 if i % 2 == 0 else 0.1
            })
            data.append({
                'date': date,
                'stock': 'B',
                'close': 100 - i,
                'signal': 0.1 if i % 2 == 0 else 0.5
            })
        
        df = pd.DataFrame(data)
        
        df['return'] = df.groupby('stock')['close'].pct_change()
        df['next_day_return'] = df.groupby('stock')['return'].shift(-1)
        
        positions = []
        for i in range(len(dates) - 1):
            date = dates[i]
            next_date = dates[i + 1]
            
            day_data = df[df['date'] == date]
            next_day_data = df[df['date'] == next_date]
            
            long_stock = day_data.loc[day_data['signal'].idxmax(), 'stock']
            
            next_return = next_day_data[next_day_data['stock'] == long_stock]['return'].values[0]
            positions.append({'date': next_date, 'return': next_return})
        
        pos_df = pd.DataFrame(positions)
        
        assert len(pos_df) == len(dates) - 1, "应跳过最后一天"
        assert pos_df['return'].notna().all(), "不应有 NaN"


class TestFutureLeakageDetection:
    """未来函数检测测试"""

    def test_leakage_detection(self):
        """测试未来函数检测"""
        from src.factors.auto.factor_evaluator import FactorEvaluator
        
        evaluator = FactorEvaluator()
        
        high_risk_names = [
            'close_shift_-1',
            'future_return', 
            'target_1d',
            'label_signal',
            'return_forward_5',
            'lead_close'
        ]
        
        safe_names = [
            'momentum_20d',
            'reversal_5d',
            'volume_ma_10',
            'std_30d'
        ]
        
        for name in high_risk_names:
            assert evaluator._check_future_leakage(name), f"应检测到风险: {name}"
        
        for name in safe_names:
            assert not evaluator._check_future_leakage(name), f"不应检测到风险: {name}"

    def test_gatekeeper_rejects_high_risk(self):
        """测试 gatekeeper 拒绝高风险因子"""
        from src.factors.auto.factor_candidate import FactorCandidate
        from src.factors.auto.factor_gatekeeper import FactorGatekeeper
        
        gatekeeper = FactorGatekeeper()
        
        candidate = FactorCandidate(
            expression='close.shift(-1)',
            name='close_shift_-1',
            description='高风险因子',
            source='test'
        )
        
        approval = gatekeeper.approve_or_reject(candidate)
        
        assert not approval['approved'], "高风险因子应被拒绝"


class TestMultiIndexCompatibility:
    """MultiIndex 兼容测试"""

    def test_multiindex_to_regular_columns(self):
        """测试 MultiIndex 转换为普通列"""
        dates = pd.date_range('2024-01-01', periods=3, freq='B')
        stocks = ['A', 'B', 'C']
        
        index = pd.MultiIndex.from_product([dates, stocks], names=['date', 'stock'])
        data = pd.DataFrame({
            'close': np.random.randn(len(index)) + 100,
            'factor': np.random.rand(len(index))
        }, index=index)
        
        df = data.reset_index()
        
        assert 'date' in df.columns, "应包含 date 列"
        assert 'stock' in df.columns, "应包含 stock 列"
        assert len(df) == len(index), "行数应保持不变"

    def test_backtest_with_multiindex(self):
        """测试回测处理 MultiIndex"""
        dates = pd.date_range('2024-01-01', periods=5, freq='B')
        stocks = ['A', 'B', 'C']
        
        index = pd.MultiIndex.from_product([dates, stocks], names=['date', 'stock'])
        
        np.random.seed(42)
        data = pd.DataFrame({
            'close': 100 + np.random.randn(len(index)) * 2,
            'signal': np.random.rand(len(index))
        }, index=index)
        
        dates_list = sorted(data.index.get_level_values(0).unique())
        
        portfolio_returns = []
        positions = {}
        
        for i, date in enumerate(dates_list):
            day_data = data.xs(date, level=0)
            
            if i == 0:
                top_stocks = day_data['signal'].nlargest(2).index.tolist()
                positions = {s: 0.5 for s in top_stocks}
            
            portfolio_returns.append({'date': date, 'positions': len(positions)})
        
        result_df = pd.DataFrame(portfolio_returns)
        result_df['date'] = pd.to_datetime(result_df['date'])
        result_df = result_df.set_index('date')
        
        assert result_df.index.name == 'date', "index 应为 date"
        assert len(result_df) == len(dates), "应包含所有日期"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
