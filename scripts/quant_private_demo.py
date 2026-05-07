#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MyQuant 量化私募架构演示脚本

展示真正接近中国量化私募的AI选股系统架构：
    1. Ranking Model - 截面排序模型
    2. Portfolio Optimization - 组合优化
    3. Risk Engine - 风控引擎
    4. Research Agent - 自我迭代研究
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from datetime import datetime
from src.utils.logger import get_logger
from src.utils.run_manager import RunManager
from src.strategy.ai_stock_picking import (
    RankingModel,
    EnsembleRankingModel,
    CrossSectionalTrainer,
    ResearchAgent,
    FactorGenerator
)
from src.core import (
    PortfolioOptimizer,
    ConstrainedOptimizer,
    MeanVarianceOptimizer,
    DrawdownControl
)


def generate_sample_cross_sectional_data(n_dates=100, n_stocks=50, n_features=30):
    """生成截面数据"""
    dates = pd.date_range('2023-01-01', periods=n_dates, freq='D')
    stocks = [f'{i:06d}.SH' for i in range(1, n_stocks + 1)]
    
    features = []
    returns = []
    
    np.random.seed(42)
    
    for date in dates:
        date_features = np.random.randn(n_stocks, n_features)
        date_returns = np.random.randn(n_stocks) * 0.01
        
        features.append(pd.DataFrame(date_features, index=stocks))
        returns.append(pd.Series(date_returns, index=stocks))
    
    features_df = pd.concat(features, keys=dates, names=['date', 'stock'])
    returns_df = pd.concat(returns, keys=dates, names=['date', 'stock'])
    
    return features_df, returns_df


def test_ranking_model():
    """测试排序模型"""
    logger.info("Testing Ranking Model...")
    
    features, returns = generate_sample_cross_sectional_data()
    
    all_dates = features.index.get_level_values(0).unique()
    train_dates = all_dates[:int(len(all_dates) * 0.7)]
    test_dates = all_dates[int(len(all_dates) * 0.7):]
    
    X_train = features.loc[train_dates].values
    y_train = returns.loc[train_dates].values
    
    X_test = features.loc[test_dates].values
    y_test = returns.loc[test_dates].values
    
    model = RankingModel(model_type='lgbm')
    model.fit(X_train, y_train)
    
    scores = model.predict(X_test)
    ic = model.evaluate_ic(X_test, y_test)
    
    logger.info(f"Ranking Model IC: {ic:.4f}")
    
    ranks = model.predict_rank(X_test)
    logger.info(f"Generated ranks for {len(ranks)} samples")
    
    return model


def test_portfolio_optimizer():
    """测试组合优化器"""
    logger.info("\nTesting Portfolio Optimizer...")
    
    n_assets = 20
    np.random.seed(42)
    
    expected_returns = np.random.uniform(-0.01, 0.02, n_assets)
    
    cov_matrix = np.random.randn(n_assets, n_assets) * 0.01
    cov_matrix = cov_matrix @ cov_matrix.T
    
    optimizer = MeanVarianceOptimizer(returns=expected_returns, cov_matrix=cov_matrix, lambda_reg=1.0)
    
    weights = optimizer.optimize()
    
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = portfolio_return / portfolio_risk
    
    logger.info(f"Expected Return: {portfolio_return:.4f}")
    logger.info(f"Portfolio Risk: {portfolio_risk:.4f}")
    logger.info(f"Sharpe Ratio: {sharpe:.4f}")
    logger.info(f"Number of positions: {np.sum(weights > 0.001)}")
    
    return optimizer


def test_risk_engine():
    """测试风控引擎"""
    logger.info("\nTesting Risk Engine...")
    
    n_stocks = 50
    n_factors = 9
    
    np.random.seed(42)
    factor_data = pd.DataFrame(np.random.randn(n_stocks, n_factors))
    factor_cov_matrix = np.eye(n_factors) * 0.01
    
    weights = np.random.rand(n_stocks)
    weights = weights / weights.sum()
    
    factor_exposure = np.dot(weights.T, factor_data.values)
    factor_risk = np.sqrt(np.dot(factor_exposure, np.dot(factor_cov_matrix, factor_exposure)))
    
    logger.info(f"Factor Risk: {factor_risk:.4f}")
    logger.info(f"Factor Exposures: {factor_exposure[:3].round(4)}...")
    
    drawdown_control = DrawdownControl(max_drawdown=0.1)
    drawdown_control.update(0.95)
    is_exceeded, current_dd = drawdown_control.check_drawdown()
    logger.info(f"Drawdown exceeded: {is_exceeded}, Current DD: {current_dd:.4f}")
    
    return drawdown_control


def test_research_agent():
    """测试研究Agent"""
    logger.info("\nTesting Research Agent...")
    
    factor_generator = FactorGenerator()
    formulas = factor_generator.generate_factor_formulas(10)
    
    logger.info(f"Generated {len(formulas)} factor formulas")
    for i, formula in enumerate(formulas[:3]):
        logger.info(f"  {i+1}. {formula}")
    
    hypotheses = factor_generator.create_hypotheses(5)
    logger.info(f"Created {len(hypotheses)} research hypotheses")
    
    return factor_generator


def main():
    global logger
    
    run_manager = RunManager(module_name='quant_private_demo')
    run_manager.start_run()
    run_id = run_manager.run_id
    
    logger = get_logger('quant_private_demo', run_id)
    
    logger.info("="*60)
    logger.info("MyQuant - 中国量化私募架构演示")
    logger.info("="*60)
    
    try:
        test_ranking_model()
        test_portfolio_optimizer()
        test_risk_engine()
        test_research_agent()
        
        logger.info("="*60)
        logger.info("量化私募架构演示完成!")
        logger.info("="*60)
        
        run_manager.end_run(status='completed')
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}", exc_info=True)
        run_manager.end_run(status='failed')


if __name__ == '__main__':
    main()