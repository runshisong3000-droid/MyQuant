import unittest
import pandas as pd
import numpy as np
from src.metrics.risk import RiskMetrics


class TestRiskMetrics(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 100)
        
        self.portfolio = pd.DataFrame({
            'total': 1000000 * (1 + returns).cumprod()
        }, index=dates)
        
        self.signals = pd.DataFrame({
            'signal': [0, 1, 0, -1, 0, 1, 0, -1, 0, 1]
        })
    
    def test_total_return(self):
        metrics = RiskMetrics.evaluate(self.portfolio)
        self.assertIsInstance(metrics['total_return'], float)
        self.assertTrue(-1 <= metrics['total_return'] <= 3)
    
    def test_sharpe_ratio(self):
        returns = self.portfolio['total'].pct_change().dropna()
        sharpe = RiskMetrics.calculate_sharpe_ratio(returns)
        self.assertIsInstance(sharpe, float)
    
    def test_max_drawdown(self):
        max_dd = RiskMetrics.calculate_max_drawdown(self.portfolio)
        self.assertIsInstance(max_dd, float)
        self.assertTrue(max_dd >= 0)
    
    def test_sortino_ratio(self):
        returns = self.portfolio['total'].pct_change().dropna()
        sortino = RiskMetrics.calculate_sortino_ratio(returns)
        self.assertIsInstance(sortino, float)
    
    def test_volatility(self):
        returns = self.portfolio['total'].pct_change().dropna()
        vol = RiskMetrics.calculate_volatility(returns)
        self.assertIsInstance(vol, float)
        self.assertTrue(vol >= 0)
    
    def test_profit_factor(self):
        pf = RiskMetrics.calculate_profit_factor(self.portfolio)
        self.assertIsInstance(pf, (float, int))
        self.assertTrue(pf >= 0)


if __name__ == '__main__':
    unittest.main()