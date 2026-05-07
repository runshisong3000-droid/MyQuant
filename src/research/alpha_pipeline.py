"""
Alpha研究流水线 - Alpha Pipeline

核心功能:
    - 数据加载
    - 因子计算
    - 因子筛选
    - 因子正交化
    - 模型训练
    - 回测验证
    - 结果分析
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import os


class AlphaPipeline:
    """
    Alpha研究流水线
    
    完整的Alpha因子研究流程：
    1. 数据准备阶段
    2. 因子生成阶段
    3. 因子筛选阶段
    4. 因子组合阶段
    5. 模型训练阶段
    6. 回测验证阶段
    7. 结果分析阶段
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.data = {}
        self.factors = {}
        self.selected_factors = []
        self.model = None
        self.backtest_results = None
        self.metrics = {}
        
        self._setup_default_config()
    
    def _setup_default_config(self):
        """设置默认配置"""
        defaults = {
            'data': {
                'start_date': '2018-01-01',
                'end_date': datetime.now().strftime('%Y-%m-%d'),
                'universe': 'all',
                'frequency': 'daily'
            },
            'factors': {
                'categories': ['technical', 'fundamental'],
                'max_factors': 100
            },
            'selection': {
                'min_ic': 0.03,
                'min_ir': 0.3,
                'max_correlation': 0.8,
                'max_factors': 30
            },
            'model': {
                'type': 'linear',
                'params': {}
            },
            'backtest': {
                'initial_capital': 1000000,
                'transaction_cost': 0.001,
                'rebalance_freq': 'weekly'
            }
        }
        
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
            else:
                self.config[key].update(defaults.get(key, {}))
    
    def load_data(self, data_loader):
        """
        加载数据
        
        Args:
            data_loader: 数据加载器对象
        """
        print("Loading data...")
        
        start_date = self.config['data']['start_date']
        end_date = self.config['data']['end_date']
        
        self.data['prices'] = data_loader.get_price_data(start_date, end_date)
        self.data['fundamentals'] = data_loader.get_fundamental_data(start_date, end_date)
        self.data['returns'] = data_loader.get_returns(start_date, end_date)
        
        print(f"Data loaded: {len(self.data['prices'])} dates, {len(self.data['prices'].columns)} stocks")
    
    def generate_factors(self, factor_engine):
        """
        生成因子
        
        Args:
            factor_engine: 因子引擎对象
        """
        print("Generating factors...")
        
        categories = self.config['factors']['categories']
        max_factors = self.config['factors']['max_factors']
        
        all_factors = []
        
        for category in categories:
            factors = factor_engine.generate_category_factors(
                self.data['prices'],
                self.data['fundamentals'],
                category=category
            )
            all_factors.extend(factors)
        
        self.factors = {f['name']: f['data'] for f in all_factors[:max_factors]}
        
        print(f"Generated {len(self.factors)} factors")
    
    def screen_factors(self, factor_tester):
        """
        筛选因子
        
        Args:
            factor_tester: 因子测试器对象
        """
        print("Screening factors...")
        
        min_ic = self.config['selection']['min_ic']
        min_ir = self.config['selection']['min_ir']
        max_correlation = self.config['selection']['max_correlation']
        max_factors = self.config['selection']['max_factors']
        
        factor_metrics = {}
        
        for factor_name, factor_data in self.factors.items():
            try:
                test = factor_tester(factor_data, self.data['returns'])
                ic_summary = test.calculate_ic_summary()
                
                if ic_summary['abs_mean_ic'] >= min_ic and abs(ic_summary['ir']) >= min_ir:
                    factor_metrics[factor_name] = {
                        'ic': ic_summary['mean_ic'],
                        'ir': ic_summary['ir'],
                        'ic_std': ic_summary['std_ic'],
                        'positive_ic_ratio': ic_summary['positive_ic_ratio']
                    }
            except Exception as e:
                print(f"Error testing {factor_name}: {e}")
        
        sorted_factors = sorted(
            factor_metrics.items(),
            key=lambda x: abs(x[1]['ir']),
            reverse=True
        )
        
        selected = []
        for factor_name, metrics in sorted_factors:
            if len(selected) >= max_factors:
                break
            
            is_correlated = False
            for selected_name in selected:
                corr = self._calculate_factor_correlation(factor_name, selected_name)
                if abs(corr) > max_correlation:
                    is_correlated = True
                    break
            
            if not is_correlated:
                selected.append(factor_name)
        
        self.selected_factors = selected
        self.metrics['factor_selection'] = {f: factor_metrics[f] for f in selected}
        
        print(f"Selected {len(self.selected_factors)} factors")
    
    def _calculate_factor_correlation(self, factor1: str, factor2: str) -> float:
        """计算两个因子的相关性"""
        f1 = self.factors[factor1].dropna()
        f2 = self.factors[factor2].dropna()
        
        common = f1.index.intersection(f2.index)
        if len(common) > 0:
            return np.corrcoef(f1.loc[common], f2.loc[common])[0, 1]
        return 0.0
    
    def orthogonalize_factors(self):
        """正交化因子"""
        print("Orthogonalizing factors...")
        
        if len(self.selected_factors) < 2:
            return
        
        factor_data = pd.DataFrame({
            f: self.factors[f] for f in self.selected_factors
        })
        
        from sklearn.decomposition import PCA
        pca = PCA(n_components=len(self.selected_factors))
        orthogonalized = pca.fit_transform(factor_data.fillna(0))
        
        for i, factor_name in enumerate(self.selected_factors):
            self.factors[f'{factor_name}_ortho'] = pd.Series(
                orthogonalized[:, i],
                index=factor_data.index,
                name=f'{factor_name}_ortho'
            )
        
        print("Factors orthogonalized")
    
    def train_model(self, model_trainer):
        """
        训练模型
        
        Args:
            model_trainer: 模型训练器对象
        """
        print("Training model...")
        
        features = pd.DataFrame({
            f: self.factors[f] for f in self.selected_factors
        })
        
        self.model = model_trainer.train(
            X=features,
            y=self.data['returns'],
            model_type=self.config['model']['type'],
            params=self.config['model']['params']
        )
        
        print("Model trained")
    
    def run_backtest(self, backtest_engine):
        """
        运行回测
        
        Args:
            backtest_engine: 回测引擎对象
        """
        print("Running backtest...")
        
        self.backtest_results = backtest_engine.run(
            model=self.model,
            factors=self.selected_factors,
            factor_data=self.factors,
            prices=self.data['prices'],
            config=self.config['backtest']
        )
        
        self.metrics['backtest'] = {
            'total_return': self.backtest_results.get('total_return'),
            'annualized_return': self.backtest_results.get('annualized_return'),
            'sharpe_ratio': self.backtest_results.get('sharpe_ratio'),
            'max_drawdown': self.backtest_results.get('max_drawdown'),
            'win_rate': self.backtest_results.get('win_rate')
        }
        
        print(f"Backtest completed: Sharpe={self.metrics['backtest']['sharpe_ratio']:.2f}")
    
    def analyze_results(self, analyzer):
        """
        分析结果
        
        Args:
            analyzer: 分析器对象
        """
        print("Analyzing results...")
        
        self.metrics['attribution'] = analyzer.analyze(
            backtest_results=self.backtest_results,
            factors=self.selected_factors,
            factor_data=self.factors
        )
        
        print("Results analyzed")
    
    def run(self, data_loader, factor_engine, factor_tester, model_trainer, backtest_engine, analyzer):
        """
        运行完整的Alpha研究流程
        
        Args:
            data_loader: 数据加载器
            factor_engine: 因子引擎
            factor_tester: 因子测试器
            model_trainer: 模型训练器
            backtest_engine: 回测引擎
            analyzer: 分析器
        """
        print("=" * 60)
        print("Starting Alpha Pipeline")
        print("=" * 60)
        
        self.load_data(data_loader)
        self.generate_factors(factor_engine)
        self.screen_factors(factor_tester)
        self.orthogonalize_factors()
        self.train_model(model_trainer)
        self.run_backtest(backtest_engine)
        self.analyze_results(analyzer)
        
        print("=" * 60)
        print("Alpha Pipeline completed")
        print("=" * 60)
        
        return self.metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """获取流程摘要"""
        return {
            'config': self.config,
            'n_factors_generated': len(self.factors),
            'n_factors_selected': len(self.selected_factors),
            'selected_factors': self.selected_factors,
            'metrics': self.metrics
        }


class PipelineStage:
    """
    流水线阶段基类
    
    定义流水线的各个阶段
    """

    def __init__(self, name: str):
        self.name = name
        self.status = 'pending'
        self.output = None
    
    def run(self, input_data: Any) -> Any:
        """运行阶段"""
        self.status = 'running'
        try:
            self.output = self._execute(input_data)
            self.status = 'completed'
            return self.output
        except Exception as e:
            self.status = 'failed'
            raise e
    
    def _execute(self, input_data: Any) -> Any:
        """执行阶段逻辑"""
        raise NotImplementedError


class PipelineRunner:
    """
    流水线运行器
    
    管理和执行多个流水线阶段
    """

    def __init__(self, stages: List[PipelineStage]):
        self.stages = stages
        self.results = {}
    
    def run(self, initial_data: Any = None) -> Dict[str, Any]:
        """运行流水线"""
        data = initial_data
        
        for stage in self.stages:
            print(f"Running stage: {stage.name}")
            data = stage.run(data)
            self.results[stage.name] = data
        
        return self.results
    
    def get_status(self) -> Dict[str, str]:
        """获取各阶段状态"""
        return {stage.name: stage.status for stage in self.stages}