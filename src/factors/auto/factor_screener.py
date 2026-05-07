"""
因子筛选器 - Factor Screener

核心功能:
    - LightGBM feature importance筛选
    - RankIC筛选
    - 相关性过滤
    - 稳定性过滤
    - 综合评分
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import warnings


class FactorScreener:
    """
    因子筛选器
    
    自动筛选有效因子：
    1. LightGBM feature importance
    2. RankIC筛选
    3. 相关性过滤
    4. 稳定性过滤
    5. 综合评分
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.factor_scores = {}
        self.selected_factors = []
        self.rejected_factors = []
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'min_rank_ic': 0.02,
            'min_icir': 0.3,
            'max_correlation': 0.7,
            'min_stability': 0.5,
            'min_importance': 0.01,
            'min_coverage': 0.8,
            'max_factors': 50,
            'weight_ic': 0.4,
            'weight_icir': 0.3,
            'weight_importance': 0.2,
            'weight_stability': 0.1
        }
    
    def screen_factors(
        self,
        factors: Dict[str, pd.Series],
        returns: pd.Series,
        market_cap: Optional[pd.Series] = None,
        industry: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        筛选因子
        
        Args:
            factors: 因子字典
            returns: 收益率数据
            market_cap: 市值数据（可选）
            industry: 行业数据（可选）
            
        Returns:
            筛选结果
        """
        print(f"Screening {len(factors)} factors...")
        
        all_scores = []
        
        for name, factor_data in factors.items():
            try:
                scores = self._calculate_factor_scores(
                    name, factor_data, returns, market_cap, industry
                )
                all_scores.append(scores)
            except Exception as e:
                print(f"Error screening {name}: {e}")
        
        if not all_scores:
            return {'selected': [], 'rejected': [], 'scores': {}}
        
        scores_df = pd.DataFrame(all_scores)
        scores_df = scores_df.set_index('factor_name')
        
        scores_df['composite_score'] = self._calculate_composite_score(scores_df)
        
        scores_df = scores_df.sort_values('composite_score', ascending=False)
        
        self.factor_scores = scores_df.to_dict('index')
        
        max_factors = self.config['max_factors']
        selected = []
        rejected = []
        
        for name, row in scores_df.iterrows():
            if len(selected) < max_factors:
                if row['composite_score'] > 0:
                    selected.append(name)
            else:
                rejected.append(name)
        
        self.selected_factors = selected
        self.rejected_factors = rejected
        
        return {
            'selected': selected,
            'rejected': rejected,
            'scores': self.factor_scores,
            'summary': self._generate_summary(scores_df)
        }
    
    def _calculate_factor_scores(
        self,
        name: str,
        factor_data: pd.Series,
        returns: pd.Series,
        market_cap: Optional[pd.Series],
        industry: Optional[pd.Series]
    ) -> Dict[str, float]:
        """计算单个因子的各项评分"""
        scores = {
            'factor_name': name,
            'rank_ic': 0.0,
            'icir': 0.0,
            'importance': 0.0,
            'stability': 0.0,
            'coverage': 0.0,
            'max_correlation': 0.0,
            'industry_neutral_ic': 0.0,
            'market_cap_neutral_ic': 0.0
        }
        
        from src.factors.auto.factor_evaluator import FactorEvaluator
        evaluator = FactorEvaluator()
        
        try:
            eval_result = evaluator.evaluate_single(factor_data, returns, industry, market_cap)
            
            scores['rank_ic'] = abs(eval_result['rank_ic']['mean'])
            scores['icir'] = eval_result.get('icir', 0.0)
            scores['coverage'] = eval_result.get('coverage', 0.0)
            
            if 'industry_neutral' in eval_result:
                scores['industry_neutral_ic'] = abs(eval_result['industry_neutral']['mean'])
            
            if 'market_cap_neutral' in eval_result:
                scores['market_cap_neutral_ic'] = abs(eval_result['market_cap_neutral']['mean'])
            
            ic_timeseries = eval_result['rank_ic'].get('timeseries', [])
            if len(ic_timeseries) > 0:
                positive_ratio = np.mean([ic > 0 for ic in ic_timeseries])
                scores['stability'] = positive_ratio
            
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
        
        scores['importance'] = self._calculate_importance(factor_data, returns)
        
        return scores
    
    def _calculate_importance(self, factor_data: pd.Series, returns: pd.Series) -> float:
        """计算因子重要性（基于与收益的相关性）"""
        try:
            common = factor_data.dropna().index.intersection(returns.dropna().index)
            if len(common) > 100:
                corr = abs(np.corrcoef(factor_data.loc[common], returns.loc[common])[0, 1])
                return min(corr, 1.0)
        except:
            pass
        return 0.0
    
    def _calculate_composite_score(self, scores_df: pd.DataFrame) -> pd.Series:
        """计算综合评分"""
        w_ic = self.config['weight_ic']
        w_icir = self.config['weight_icir']
        w_importance = self.config['weight_importance']
        w_stability = self.config['weight_stability']
        
        rank_ic_norm = scores_df['rank_ic'] / (scores_df['rank_ic'].max() + 1e-8)
        icir_norm = scores_df['icir'] / (scores_df['icir'].max() + 1e-8)
        importance_norm = scores_df['importance'] / (scores_df['importance'].max() + 1e-8)
        stability_norm = scores_df['stability']
        
        composite = (
            w_ic * rank_ic_norm +
            w_icir * icir_norm +
            w_importance * importance_norm +
            w_stability * stability_norm
        )
        
        return composite
    
    def _generate_summary(self, scores_df: pd.DataFrame) -> Dict[str, Any]:
        """生成筛选摘要"""
        return {
            'total_factors': len(scores_df),
            'selected_count': len(self.selected_factors),
            'rejected_count': len(self.rejected_factors),
            'mean_composite_score': scores_df['composite_score'].mean(),
            'top_5_factors': scores_df.nlargest(5, 'composite_score')[['rank_ic', 'icir', 'composite_score']].to_dict('index'),
            'score_distribution': scores_df['composite_score'].describe().to_dict()
        }


class LightGBMImportanceScreener:
    """
    LightGBM特征重要性筛选
    
    使用LightGBM模型评估因子重要性
    """

    def __init__(self):
        self.model = None
        self.feature_importance = {}
    
    def calculate_importance(
        self,
        factors: Dict[str, pd.Series],
        returns: pd.Series,
        n_estimators: int = 100,
        max_depth: int = 5
    ) -> Dict[str, float]:
        """
        计算因子重要性
        
        Args:
            factors: 因子字典
            returns: 收益率数据
            n_estimators: 树数量
            max_depth: 最大深度
            
        Returns:
            因子重要性字典
        """
        try:
            import lightgbm as lgb
        except ImportError:
            print("LightGBM not installed, using correlation-based importance")
            return self._calculate_correlation_importance(factors, returns)
        
        feature_matrix = []
        stock_returns = []
        valid_stocks = []
        
        for stock in returns.index.get_level_values(1).unique():
            try:
                stock_factor_data = []
                for factor_name, factor_data in factors.items():
                    if isinstance(factor_data.index, pd.MultiIndex):
                        factor_val = factor_data.xs(stock, level=1).mean()
                    else:
                        factor_val = factor_data.get(stock, np.nan)
                    
                    if not np.isnan(factor_val):
                        stock_factor_data.append(factor_val)
                    else:
                        stock_factor_data.append(0.0)
                
                stock_return = returns.xs(stock, level=1).mean() if isinstance(returns.index, pd.MultiIndex) else returns.get(stock, np.nan)
                
                if not np.isnan(stock_return):
                    feature_matrix.append(stock_factor_data)
                    stock_returns.append(stock_return)
                    valid_stocks.append(stock)
            
            except Exception as e:
                continue
        
        if len(feature_matrix) < 100:
            return self._calculate_correlation_importance(factors, returns)
        
        X = np.array(feature_matrix)
        y = np.array(stock_returns)
        feature_names = list(factors.keys())
        
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.05,
            random_state=42,
            verbose=-1
        )
        
        model.fit(X, y)
        
        importances = model.feature_importances_
        
        self.feature_importance = {
            name: float(imp) for name, imp in zip(feature_names, importances)
        }
        
        self.model = model
        
        total = sum(self.feature_importance.values())
        if total > 0:
            self.feature_importance = {
                k: v / total for k, v in self.feature_importance.items()
            }
        
        return self.feature_importance
    
    def _calculate_correlation_importance(
        self,
        factors: Dict[str, pd.Series],
        returns: pd.Series
    ) -> Dict[str, float]:
        """基于相关性计算重要性"""
        importances = {}
        
        for name, factor_data in factors.items():
            try:
                common = factor_data.dropna().index.intersection(returns.dropna().index)
                if len(common) > 100:
                    corr = abs(np.corrcoef(factor_data.loc[common], returns.loc[common])[0, 1])
                    importances[name] = min(corr, 1.0)
                else:
                    importances[name] = 0.0
            except:
                importances[name] = 0.0
        
        total = sum(importances.values())
        if total > 0:
            importances = {k: v / total for k, v in importances.items()}
        
        return importances
    
    def get_top_factors(self, n: int = 20) -> List[Tuple[str, float]]:
        """获取最重要的N个因子"""
        sorted_factors = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_factors[:n]


class CorrelationFilter:
    """
    相关性过滤器
    
    去除高相关因子
    """

    def __init__(self, max_correlation: float = 0.7):
        self.max_correlation = max_correlation
        self.correlation_matrix = None
    
    def filter(
        self,
        factors: Dict[str, pd.Series],
        returns: pd.Series = None
    ) -> List[str]:
        """
        过滤高相关因子
        
        Args:
            factors: 因子字典
            returns: 收益率数据（可选）
            
        Returns:
            保留的因子列表
        """
        factor_names = list(factors.keys())
        
        if len(factor_names) == 0:
            return []
        
        corr_matrix = np.zeros((len(factor_names), len(factor_names)))
        
        for i, name1 in enumerate(factor_names):
            for j, name2 in enumerate(factor_names):
                if i == j:
                    corr_matrix[i, j] = 1.0
                elif j > i:
                    try:
                        common = factors[name1].dropna().index.intersection(
                            factors[name2].dropna().index
                        )
                        if len(common) > 100:
                            corr = np.corrcoef(
                                factors[name1].loc[common],
                                factors[name2].loc[common]
                            )[0, 1]
                            if np.isnan(corr):
                                corr = 0.0
                            corr_matrix[i, j] = abs(corr)
                            corr_matrix[j, i] = abs(corr)
                    except:
                        corr_matrix[i, j] = 0.0
                        corr_matrix[j, i] = 0.0
        
        self.correlation_matrix = pd.DataFrame(
            corr_matrix,
            index=factor_names,
            columns=factor_names
        )
        
        importance_scores = {}
        for name in factor_names:
            if returns is not None:
                try:
                    common = factors[name].dropna().index.intersection(returns.dropna().index)
                    if len(common) > 100:
                        corr = abs(np.corrcoef(factors[name].loc[common], returns.loc[common])[0, 1])
                        importance_scores[name] = corr
                    else:
                        importance_scores[name] = 0.0
                except:
                    importance_scores[name] = 0.0
            else:
                importance_scores[name] = 1.0
        
        selected = []
        rejected = []
        
        sorted_factors = sorted(
            importance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for name, score in sorted_factors:
            if name in rejected:
                continue
            
            selected.append(name)
            
            for other_name in factor_names:
                if other_name != name and other_name not in rejected:
                    if self.correlation_matrix.loc[name, other_name] > self.max_correlation:
                        rejected.append(other_name)
        
        return selected
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """获取相关性矩阵"""
        return self.correlation_matrix


class StabilityFilter:
    """
    稳定性过滤器
    
    筛选IC稳定的因子
    """

    def __init__(self, min_positive_ratio: float = 0.5):
        self.min_positive_ratio = min_positive_ratio
    
    def calculate_stability(
        self,
        factor_data: pd.Series,
        returns: pd.Series
    ) -> float:
        """
        计算因子稳定性
        
        Args:
            factor_data: 因子数据
            returns: 收益率数据
            
        Returns:
            稳定性分数（0-1）
        """
        from src.factors.auto.factor_evaluator import FactorEvaluator
        evaluator = FactorEvaluator()
        
        ic_result = evaluator.calculate_rank_ic(factor_data, returns)
        ic_timeseries = ic_result.get('timeseries', [])
        
        if len(ic_timeseries) == 0:
            return 0.0
        
        positive_ratio = np.mean([ic > 0 for ic in ic_timeseries])
        
        ic_std = np.std(ic_timeseries)
        ic_mean = np.abs(np.mean(ic_timeseries))
        
        if ic_std > 0:
            ir = ic_mean / ic_std
        else:
            ir = 0.0
        
        stability = positive_ratio * 0.5 + min(ir / 2, 1.0) * 0.5
        
        return stability
    
    def filter(
        self,
        factors: Dict[str, pd.Series],
        returns: pd.Series,
        min_stability: float = 0.5
    ) -> List[str]:
        """
        过滤不稳定因子
        
        Args:
            factors: 因子字典
            returns: 收益率数据
            min_stability: 最小稳定性
            
        Returns:
            保留的因子列表
        """
        selected = []
        
        for name, factor_data in factors.items():
            try:
                stability = self.calculate_stability(factor_data, returns)
                if stability >= min_stability:
                    selected.append(name)
            except Exception as e:
                print(f"Error calculating stability for {name}: {e}")
        
        return selected


class ComprehensiveScreener:
    """
    综合筛选器
    
    组合多种筛选方法
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.factor_screener = FactorScreener(config)
        self.lgbm_screener = LightGBMImportanceScreener()
        self.corr_filter = CorrelationFilter(
            max_correlation=config.get('max_correlation', 0.7) if config else 0.7
        )
        self.stability_filter = StabilityFilter(
            min_positive_ratio=config.get('min_stability', 0.5) if config else 0.5
        )
    
    def screen(
        self,
        factors: Dict[str, pd.Series],
        returns: pd.Series,
        market_cap: Optional[pd.Series] = None,
        industry: Optional[pd.Series] = None,
        use_lgbm: bool = True
    ) -> Dict[str, Any]:
        """
        综合筛选
        
        Args:
            factors: 因子字典
            returns: 收益率数据
            market_cap: 市值数据
            industry: 行业数据
            use_lgbm: 是否使用LightGBM
            
        Returns:
            筛选结果
        """
        print(f"Starting comprehensive screening of {len(factors)} factors...")
        
        if use_lgbm:
            print("Step 1: Calculating LightGBM importance...")
            lgbm_importance = self.lgbm_screener.calculate_importance(
                factors, returns
            )
            
            lgbm_selected = [
                name for name, imp in lgbm_importance.items()
                if imp > 0.01
            ]
            
            factors = {k: factors[k] for k in lgbm_selected if k in factors}
            print(f"  LightGBM selected {len(factors)} factors")
        
        print("Step 2: Screening by metrics...")
        screener_result = self.factor_screener.screen_factors(
            factors, returns, market_cap, industry
        )
        
        print("Step 3: Filtering by correlation...")
        selected_after_corr = self.corr_filter.filter(
            {k: factors[k] for k in screener_result['selected']}, returns
        )
        print(f"  After correlation filter: {len(selected_after_corr)} factors")
        
        print("Step 4: Filtering by stability...")
        selected_after_stability = self.stability_filter.filter(
            {k: factors[k] for k in selected_after_corr},
            returns,
            min_stability=0.5
        )
        print(f"  After stability filter: {len(selected_after_stability)} factors")
        
        return {
            'selected': selected_after_stability,
            'scores': self.factor_screener.factor_scores,
            'lgbm_importance': self.lgbm_screener.feature_importance if use_lgbm else {},
            'correlation_matrix': self.corr_filter.get_correlation_matrix(),
            'summary': {
                'initial_count': len(factors) if not use_lgbm else len(lgbm_importance),
                'after_metrics': len(screener_result['selected']),
                'after_correlation': len(selected_after_corr),
                'after_stability': len(selected_after_stability),
                'final_count': len(selected_after_stability)
            }
        }
    
    def get_report(self) -> pd.DataFrame:
        """生成筛选报告"""
        if not self.factor_screener.factor_scores:
            return pd.DataFrame()
        
        report_data = []
        
        for name, scores in self.factor_screener.factor_scores.items():
            row = {'factor_name': name}
            row.update(scores)
            report_data.append(row)
        
        return pd.DataFrame(report_data).sort_values('composite_score', ascending=False)
