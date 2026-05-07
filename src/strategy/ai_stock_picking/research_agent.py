"""
AI Research Agent - 自我迭代研究Agent

核心功能:
    - 自动发现新的alpha因子
    - 自我迭代研究循环
    - 自动生成研究假设
    - 自动回测和评估
    - 自动淘汰无效因子
    - 保留有效alpha
    - 自动生成研究报告

循环流程:
    1. 阅读市场数据
    2. 提出市场假设
    3. 生成新因子（使用AutoFactorPipeline）
    4. 回测
    5. 评估Sharpe/IC/IR
    6. 淘汰失败因子
    7. 保留有效因子
    8. 组合优化
    9. 生成研究报告
    10. 继续研究

这是未来AI量化的核心架构。
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import json
import os


class ResearchHypothesis:
    """研究假设"""

    def __init__(
        self,
        hypothesis_id: str,
        description: str,
        factor_formula: str,
        expected_sign: str = 'positive',
        confidence: float = 0.5,
        category: str = 'unknown'
    ):
        self.hypothesis_id = hypothesis_id
        self.description = description
        self.factor_formula = factor_formula
        self.expected_sign = expected_sign
        self.confidence = confidence
        self.category = category
        self.status = 'pending'
        self.results = None
        self.evaluation_details = None
    
    def to_dict(self):
        """转换为字典"""
        return {
            'hypothesis_id': self.hypothesis_id,
            'description': self.description,
            'factor_formula': self.factor_formula,
            'expected_sign': self.expected_sign,
            'confidence': self.confidence,
            'category': self.category,
            'status': self.status,
            'results': self.results,
            'evaluation_details': self.evaluation_details
        }


class MarketHypothesisGenerator:
    """
    市场假设生成器
    
    自动生成有意义的研究假设：
    - 基于市场规律的假设
    - 基于因子组合的假设
    - 基于行业分析的假设
    """

    def __init__(self):
        self.hypothesis_templates = [
            {
                'template': '高{factor}的股票未来收益高于低{factor}的股票',
                'factors': ['动量', '成交量', '换手率', '波动率', 'ROE', '毛利率'],
                'category': 'momentum'
            },
            {
                'template': '{factor1}与{factor2}的比值可以预测股票收益',
                'factors1': ['收盘价', '成交量', '换手率', '市值'],
                'factors2': ['开盘价', '成交额', '波动率', '市盈率'],
                'category': 'ratio'
            },
            {
                'template': '{period}周期内的{indicator}变化与未来收益正相关',
                'periods': ['短期', '中期', '长期'],
                'indicators': ['价格', '成交量', '资金流向', '情绪指标'],
                'category': 'time_series'
            },
            {
                'template': '行业{industry}中，{factor}高的股票表现更好',
                'industries': ['消费', '科技', '金融', '医药', '制造'],
                'factors': ['ROE', '增长速度', '现金流', '研发投入'],
                'category': 'industry'
            },
            {
                'template': '反转效应：近期表现{direction}的股票未来会{opposite}',
                'directions': ['强劲', '疲软'],
                'opposites': ['回调', '反弹'],
                'category': 'reversal'
            }
        ]
        
        self.factor_descriptions = {
            'momentum': '价格持续上涨的股票未来继续上涨的概率更高',
            'reversal': '短期内剧烈波动后会向均值回归',
            'volatility': '高波动股票蕴含更高风险补偿',
            'liquidity': '流动性好的股票更容易产生alpha',
            'quality': '高质量财务指标预示更好的未来表现',
            'value': '估值合理的股票长期表现更优'
        }
    
    def generate_hypotheses(self, n_hypotheses: int = 20) -> List[ResearchHypothesis]:
        """生成研究假设"""
        hypotheses = []
        
        for i in range(n_hypotheses):
            template = np.random.choice(self.hypothesis_templates)
            
            if 'factor' in template:
                factor = np.random.choice(template['factors'])
                description = template['template'].format(factor=factor)
                category = template['category']
                formula = self._generate_formula_from_description(description)
            
            elif 'factor1' in template:
                factor1 = np.random.choice(template['factors1'])
                factor2 = np.random.choice(template['factors2'])
                description = template['template'].format(factor1=factor1, factor2=factor2)
                category = template['category']
                formula = self._generate_formula_from_description(description)
            
            elif 'period' in template:
                period = np.random.choice(template['periods'])
                indicator = np.random.choice(template['indicators'])
                description = template['template'].format(period=period, indicator=indicator)
                category = template['category']
                formula = self._generate_formula_from_description(description)
            
            elif 'industry' in template:
                industry = np.random.choice(template['industries'])
                factor = np.random.choice(template['factors'])
                description = template['template'].format(industry=industry, factor=factor)
                category = template['category']
                formula = self._generate_formula_from_description(description)
            
            elif 'direction' in template:
                direction = np.random.choice(template['directions'])
                opposite = np.random.choice(template['opposites'])
                description = template['template'].format(direction=direction, opposite=opposite)
                category = template['category']
                formula = self._generate_formula_from_description(description)
            
            else:
                description = f"研究假设 {i+1}"
                category = 'unknown'
                formula = 'close'
            
            hypothesis = ResearchHypothesis(
                hypothesis_id=f'hyp_{datetime.now().strftime("%Y%m%d")}_{i:03d}',
                description=description,
                factor_formula=formula,
                expected_sign=np.random.choice(['positive', 'negative']),
                confidence=np.random.uniform(0.3, 0.7),
                category=category
            )
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _generate_formula_from_description(self, description: str) -> str:
        """从描述生成因子公式"""
        if '动量' in description:
            return 'close.pct_change(20)'
        elif '成交量' in description:
            return 'volume / volume.rolling(20).mean()'
        elif '换手率' in description:
            return 'turnover / turnover.rolling(20).mean()'
        elif '波动率' in description:
            return 'close.pct_change().rolling(20).std()'
        elif 'ROE' in description:
            return 'roic'
        elif '价格' in description:
            return 'close'
        elif '反转' in description:
            return '-close.pct_change(5)'
        elif '资金流向' in description:
            return 'amount / amount.rolling(20).mean()'
        else:
            return 'close.pct_change(20)'


class FactorFormulaGenerator:
    """
    因子公式生成器
    
    自动生成因子公式：
    - 基础公式
    - 组合公式
    - 变换公式
    """

    def __init__(self):
        self.operators = [
            ('rank', 1, lambda x, w: f'rank({x})'),
            ('zscore', 1, lambda x, w: f'zscore({x})'),
            ('log', 1, lambda x, w: f'np.log({x})'),
            ('abs', 1, lambda x, w: f'np.abs({x})'),
            ('delta', 2, lambda x, w: f'{x}.diff({w})'),
            ('rolling_mean', 2, lambda x, w: f'{x}.rolling({w}).mean()'),
            ('rolling_std', 2, lambda x, w: f'{x}.rolling({w}).std()'),
            ('rolling_max', 2, lambda x, w: f'{x}.rolling({w}).max()'),
            ('rolling_min', 2, lambda x, w: f'{x}.rolling({w}).min()'),
            ('rolling_corr', 3, lambda x, w, y: f'{x}.rolling({w}).corr({y})')
        ]
        
        self.fields = ['close', 'open', 'high', 'low', 'volume', 'amount', 'turnover']
        self.windows = [5, 10, 20, 60]
    
    def generate_formula(self, complexity: int = 2) -> str:
        """生成单个公式"""
        if complexity == 1:
            return np.random.choice(self.fields)
        
        operator = np.random.choice(self.operators)
        op_name, n_args, op_func = operator
        
        if n_args == 1:
            arg = self.generate_formula(complexity - 1)
            return op_func(arg, None)
        
        elif n_args == 2:
            arg = np.random.choice(self.fields)
            window = np.random.choice(self.windows)
            return op_func(arg, window)
        
        else:
            arg1 = np.random.choice(self.fields)
            arg2 = np.random.choice([f for f in self.fields if f != arg1])
            window = np.random.choice(self.windows)
            return op_func(arg1, window, arg2)
    
    def generate_batch(self, n: int = 50, complexity_range: Tuple[int, int] = (1, 3)) -> List[str]:
        """批量生成公式"""
        formulas = set()
        
        while len(formulas) < n:
            complexity = np.random.randint(complexity_range[0], complexity_range[1] + 1)
            formula = self.generate_formula(complexity)
            formulas.add(formula)
        
        return list(formulas)


class ResearchReportGenerator:
    """
    研究报告生成器
    
    自动生成研究报告：
    - 因子表现总结
    - IC/IR统计
    - 分组收益分析
    - 多空收益分析
    - 换手率分析
    - 稳定性分析
    - 建议和结论
    """

    def __init__(self):
        pass
    
    def generate_report(
        self,
        agent: 'ResearchAgent',
        filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """生成研究报告"""
        report = {
            'report_id': f'report_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'generated_at': datetime.now().isoformat(),
            'sections': []
        }
        
        report['sections'].append(self._generate_summary_section(agent))
        report['sections'].append(self._generate_factor_section(agent))
        report['sections'].append(self._generate_backtest_section(agent))
        report['sections'].append(self._generate_recommendations_section(agent))
        
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report
    
    def _generate_summary_section(self, agent: 'ResearchAgent') -> Dict[str, Any]:
        """生成摘要章节"""
        valid_count = len(agent.valid_factors)
        total_count = len(agent.hypotheses)
        
        return {
            'title': '研究摘要',
            'content': {
                'total_hypotheses': total_count,
                'valid_factors': valid_count,
                'invalid_factors': len(agent.invalid_factors),
                'discovery_rate': valid_count / max(total_count, 1) * 100,
                'report_period': datetime.now().strftime('%Y年%m月%d日'),
                'research_cycles': len([e for e in agent.research_log if 'cycle' in e['message'].lower()])
            }
        }
    
    def _generate_factor_section(self, agent: 'ResearchAgent') -> Dict[str, Any]:
        """生成因子分析章节"""
        if not agent.valid_factors:
            return {
                'title': '因子分析',
                'content': {'message': '暂无有效因子'}
            }
        
        factor_stats = []
        
        for hypothesis in agent.valid_factors:
            results = hypothesis.results or {}
            details = hypothesis.evaluation_details or {}
            
            factor_stats.append({
                'name': hypothesis.hypothesis_id,
                'description': hypothesis.description,
                'formula': hypothesis.factor_formula,
                'category': hypothesis.category,
                'ic': results.get('avg_ic'),
                'icir': results.get('ir'),
                'coverage': details.get('coverage'),
                'turnover': details.get('turnover'),
                'status': hypothesis.status
            })
        
        return {
            'title': '因子分析',
            'content': {
                'factors': factor_stats,
                'top_factors': sorted(factor_stats, key=lambda x: abs(x['ic'] or 0), reverse=True)[:5]
            }
        }
    
    def _generate_backtest_section(self, agent: 'ResearchAgent') -> Dict[str, Any]:
        """生成回测分析章节"""
        return {
            'title': '回测分析',
            'content': {
                'message': '回测结果需要通过backtest_engine运行后生成',
                'last_backtest': agent.research_log[-1] if agent.research_log else None
            }
        }
    
    def _generate_recommendations_section(self, agent: 'ResearchAgent') -> Dict[str, Any]:
        """生成建议章节"""
        recommendations = []
        
        if len(agent.valid_factors) == 0:
            recommendations.append('当前没有发现有效因子，建议调整筛选参数或增加候选因子数量')
        
        if len(agent.valid_factors) < 5:
            recommendations.append('有效因子数量较少，建议继续运行研究循环')
        
        if len(agent.valid_factors) >= 5:
            recommendations.append('已发现多个有效因子，建议进行组合测试和回测验证')
        
        recommendations.append('建议定期运行研究循环以发现新的alpha因子')
        
        return {
            'title': '研究建议',
            'content': {
                'recommendations': recommendations,
                'next_steps': [
                    '1. 运行回测验证因子有效性',
                    '2. 优化因子权重',
                    '3. 考虑行业中性化处理',
                    '4. 监控因子表现稳定性',
                    '5. 继续迭代发现新因子'
                ]
            }
        }
    
    def print_report(self, report: Dict[str, Any]):
        """打印报告"""
        print(f"\n{'='*60}")
        print(f"研究报告 - {report['generated_at']}")
        print(f"{'='*60}")
        
        for section in report['sections']:
            print(f"\n【{section['title']}】")
            self._print_section_content(section['content'])
        
        print(f"\n{'='*60}\n")
    
    def _print_section_content(self, content: Dict[str, Any], indent: int = 0):
        """递归打印章节内容"""
        for key, value in content.items():
            if isinstance(value, dict):
                print('  ' * indent + f"{key}:")
                self._print_section_content(value, indent + 1)
            elif isinstance(value, list):
                print('  ' * indent + f"{key}:")
                for item in value:
                    if isinstance(item, dict):
                        self._print_section_content(item, indent + 1)
                    else:
                        print('  ' * (indent + 1) + f"- {item}")
            else:
                print('  ' * indent + f"{key}: {value}")


class ResearchAgent:
    """
    AI研究Agent
    
    核心能力:
    - 自动阅读和分析市场数据
    - 生成研究假设
    - 测试因子有效性
    - 自我迭代改进
    - 接入AutoFactorPipeline自动学习因子
    - 自动生成研究报告
    """

    def __init__(
        self,
        data_provider=None,
        backtest_engine=None,
        risk_engine=None,
        results_dir: str = 'research_results'
    ):
        self.data_provider = data_provider
        self.backtest_engine = backtest_engine
        self.risk_engine = risk_engine
        self.results_dir = results_dir
        
        self.hypothesis_generator = MarketHypothesisGenerator()
        self.formula_generator = FactorFormulaGenerator()
        self.report_generator = ResearchReportGenerator()
        
        self.hypotheses = []
        self.valid_factors = []
        self.invalid_factors = []
        self.research_log = []
        
        self.auto_factor_pipeline = None
        
        os.makedirs(results_dir, exist_ok=True)
    
    def initialize_auto_factor_pipeline(self, config: Optional[Dict[str, Any]] = None):
        """初始化自动因子流水线"""
        from src.factors.auto import AutoFactorPipelineBuilder
        
        self.auto_factor_pipeline = AutoFactorPipelineBuilder.build(config)
        self.log("Auto Factor Pipeline initialized")
    
    def read_market_data(self) -> Dict[str, Any]:
        """阅读市场数据"""
        if self.data_provider is None:
            return {}
        
        return {
            'stocks': self.data_provider.get_stock_list(),
            'features': self.data_provider.get_all_features(),
            'returns': self.data_provider.get_returns(),
            'market_cap': self.data_provider.get_market_cap(),
            'industry': self.data_provider.get_industry()
        }
    
    def generate_hypotheses(self, n_hypotheses: int = 20):
        """生成研究假设"""
        new_hypotheses = self.hypothesis_generator.generate_hypotheses(n_hypotheses)
        self.hypotheses.extend(new_hypotheses)
        
        self.log(f"Generated {len(new_hypotheses)} new hypotheses")
    
    def generate_factor_formulas(self, n_formulas: int = 50) -> List[str]:
        """生成因子公式"""
        return self.formula_generator.generate_batch(n_formulas)
    
    def run_auto_factor_discovery(
        self,
        data: pd.DataFrame,
        returns: pd.Series,
        n_expressions: int = 50,
        use_genetic: bool = False,
        use_neural: bool = False
    ) -> Dict[str, Any]:
        """使用自动因子流水线发现新因子"""
        if self.auto_factor_pipeline is None:
            self.initialize_auto_factor_pipeline()
        
        self.log(f"Starting auto factor discovery with {n_expressions} candidates")
        
        result = self.auto_factor_pipeline.run(
            data=data,
            returns=returns,
            n_expressions=n_expressions,
            use_genetic=use_genetic,
            use_neural=use_neural
        )
        
        for factor_id in result['approved']:
            factor_info = next(
                (f for f in result['approved_details'] if f['factor_id'] == factor_id),
                None
            )
            
            if factor_info:
                hypothesis = ResearchHypothesis(
                    hypothesis_id=factor_id,
                    description=f"Auto-discovered factor: {factor_info['expression']}",
                    factor_formula=factor_info['expression'],
                    expected_sign='positive',
                    confidence=0.8,
                    category='auto_discovered'
                )
                hypothesis.status = 'auto_discovered'
                hypothesis.results = {
                    'success': True,
                    'avg_ic': factor_info['evaluation_results'].get('rank_ic', {}).get('mean'),
                    'ir': factor_info['evaluation_results'].get('icir'),
                    'n_dates': factor_info['evaluation_results'].get('rank_ic', {}).get('count')
                }
                hypothesis.evaluation_details = {
                    'coverage': factor_info['metadata'].get('coverage'),
                    'turnover': factor_info['evaluation_results'].get('turnover'),
                    'max_correlation': factor_info['evaluation_results'].get('max_correlation')
                }
                
                self.valid_factors.append(hypothesis)
        
        self.log(f"Auto factor discovery completed: {len(result['approved'])} approved factors")
        
        return result
    
    def evaluate_hypothesis(self, hypothesis: ResearchHypothesis) -> Dict[str, float]:
        """评估单个假设"""
        try:
            factor_data = self._compute_factor(hypothesis.factor_formula)
            
            if factor_data is None or factor_data.empty:
                hypothesis.status = 'failed'
                hypothesis.results = {'success': False, 'error': 'Failed to compute factor'}
                return hypothesis.results
            
            ic_values = []
            dates = factor_data.index.get_level_values(0).unique()
            
            for date in dates:
                factor_vals = factor_data.loc[date].values
                returns = self._get_returns_for_date(date)
                
                valid_mask = ~np.isnan(factor_vals) & ~np.isnan(returns)
                if np.sum(valid_mask) > 10:
                    ic = np.corrcoef(factor_vals[valid_mask], returns[valid_mask])[0, 1]
                    if not np.isnan(ic):
                        ic_values.append(ic)
            
            if ic_values:
                avg_ic = np.mean(ic_values)
                ic_std = np.std(ic_values)
                ir = avg_ic / ic_std if ic_std > 0 else 0
                
                results = {
                    'avg_ic': avg_ic,
                    'ic_std': ic_std,
                    'ir': ir,
                    'n_dates': len(ic_values),
                    'positive_ratio': np.mean([ic > 0 for ic in ic_values]),
                    'success': True
                }
            else:
                results = {'success': False, 'error': 'No valid IC values'}
            
            hypothesis.results = results
            hypothesis.status = 'evaluated'
            
            return results
            
        except Exception as e:
            hypothesis.status = 'failed'
            hypothesis.results = {'success': False, 'error': str(e)}
            return {'success': False, 'error': str(e)}
    
    def _compute_factor(self, formula: str) -> Optional[pd.Series]:
        """计算因子值"""
        if self.data_provider is None:
            return None
        
        features = self.data_provider.get_all_features()
        
        try:
            window_matches = [int(w) for w in formula if w.isdigit()]
            window = window_matches[0] if window_matches else 20
            
            if 'rolling_mean' in formula:
                if 'close' in formula:
                    return features['close'].groupby(level=1).rolling(window).mean()
                elif 'volume' in formula:
                    return features['volume'].groupby(level=1).rolling(window).mean()
            
            elif 'rolling_std' in formula:
                return features['close'].groupby(level=1).rolling(window).std()
            
            elif 'rolling_max' in formula:
                high = features['high'].groupby(level=1).rolling(window).max()
                low = features['low'].groupby(level=1).rolling(window).min()
                return high - low
            
            elif 'rolling_corr' in formula:
                return features[['close', 'volume']].groupby(level=1).rolling(window).corr().unstack()['close']['volume']
            
            elif '(close - open)' in formula:
                return (features['close'] - features['open']) / features['open']
            
            elif 'pct_change' in formula:
                return features['close'].groupby(level=1).pct_change(window)
            
            else:
                return features['close'].groupby(level=1).pct_change(window)
        
        except Exception as e:
            print(f"Error computing factor: {e}")
            return None
    
    def _get_returns_for_date(self, date) -> np.ndarray:
        """获取指定日期的收益率"""
        if self.data_provider is None:
            return np.array([])
        
        try:
            return self.data_provider.get_returns_for_date(date)
        except:
            return np.array([])
    
    def filter_valid_factors(self, min_ic: float = 0.03, min_ir: float = 0.5):
        """过滤有效因子"""
        for hypothesis in self.hypotheses:
            if hypothesis.status == 'evaluated' and hypothesis.results.get('success', False):
                ic = hypothesis.results.get('avg_ic', 0)
                ir = hypothesis.results.get('ir', 0)
                
                if abs(ic) >= min_ic and abs(ir) >= min_ir:
                    self.valid_factors.append(hypothesis)
                    self.log(f"Valid factor found: {hypothesis.hypothesis_id}, IC={ic:.4f}, IR={ir:.4f}")
                else:
                    self.invalid_factors.append(hypothesis)
    
    def run_backtest(self, factor_weights: Optional[Dict[str, float]] = None):
        """运行回测"""
        if self.backtest_engine is None:
            self.log("No backtest engine available")
            return None
        
        if factor_weights is None:
            factor_weights = {h.hypothesis_id: 1.0 for h in self.valid_factors}
        
        result = self.backtest_engine.run(
            factors=self.valid_factors,
            weights=factor_weights
        )
        
        self.log(f"Backtest completed: Sharpe={result.get('sharpe_ratio', 0):.4f}, MaxDD={result.get('max_drawdown', 0):.4f}")
        
        return result
    
    def optimize_weights(self) -> Dict[str, float]:
        """优化因子权重"""
        if not self.valid_factors:
            return {}
        
        ir_values = []
        for h in self.valid_factors:
            ir = h.results.get('ir', 0) if h.results else 0
            ir_values.append(abs(ir))
        
        if sum(ir_values) == 0:
            return {h.hypothesis_id: 1.0 / len(self.valid_factors) for h in self.valid_factors}
        
        ir_values = np.array(ir_values)
        weights = ir_values / np.sum(ir_values)
        
        return {h.hypothesis_id: w for h, w in zip(self.valid_factors, weights)}
    
    def generate_report(self, filename: Optional[str] = None) -> Dict[str, Any]:
        """生成研究报告"""
        report = self.report_generator.generate_report(self, filename)
        self.log(f"Research report generated")
        return report
    
    def print_report(self):
        """打印研究报告"""
        report = self.report_generator.generate_report(self)
        self.report_generator.print_report(report)
    
    def log(self, message: str):
        """记录研究日志"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'n_hypotheses': len(self.hypotheses),
            'n_valid_factors': len(self.valid_factors),
            'n_invalid_factors': len(self.invalid_factors)
        }
        
        self.research_log.append(entry)
        
        log_file = os.path.join(self.results_dir, 'research_log.json')
        with open(log_file, 'w') as f:
            json.dump(self.research_log, f, indent=2)
    
    def save_results(self):
        """保存研究结果"""
        results = {
            'valid_factors': [h.to_dict() for h in self.valid_factors],
            'invalid_factors': [h.to_dict() for h in self.invalid_factors],
            'research_log': self.research_log,
            'summary': {
                'total_hypotheses': len(self.hypotheses),
                'valid_count': len(self.valid_factors),
                'invalid_count': len(self.invalid_factors),
                'valid_ratio': len(self.valid_factors) / max(len(self.hypotheses), 1),
                'timestamp': datetime.now().isoformat()
            }
        }
        
        result_file = os.path.join(self.results_dir, f'research_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.log(f"Results saved to {result_file}")
    
    def run_research_cycle(self, n_hypotheses: int = 20, use_auto_factor: bool = True):
        """运行完整的研究循环"""
        self.log("Starting research cycle...")
        
        self.log("Step 1: Reading market data")
        market_data = self.read_market_data()
        
        if use_auto_factor and self.auto_factor_pipeline is not None:
            self.log("Step 2: Running auto factor discovery")
            data = market_data.get('features', pd.DataFrame())
            returns = market_data.get('returns', pd.Series())
            
            if not data.empty and not returns.empty:
                self.run_auto_factor_discovery(
                    data=data,
                    returns=returns,
                    n_expressions=50,
                    use_genetic=False,
                    use_neural=False
                )
        
        self.log("Step 3: Generating hypotheses")
        self.generate_hypotheses(n_hypotheses)
        
        self.log("Step 4: Evaluating hypotheses")
        for hypothesis in self.hypotheses:
            if hypothesis.status == 'pending':
                self.evaluate_hypothesis(hypothesis)
        
        self.log("Step 5: Filtering valid factors")
        self.filter_valid_factors()
        
        self.log("Step 6: Optimizing weights")
        weights = self.optimize_weights()
        
        self.log("Step 7: Running backtest")
        backtest_result = self.run_backtest(weights)
        
        self.log("Step 8: Generating research report")
        report = self.generate_report()
        
        self.log("Step 9: Saving results")
        self.save_results()
        
        self.log("Research cycle completed")
        
        return {
            'valid_factors': self.valid_factors,
            'weights': weights,
            'backtest_result': backtest_result,
            'report': report
        }


class AutoResearchPipeline:
    """
    自动研究管道
    
    持续运行研究循环，不断发现新alpha
    """

    def __init__(self, research_agent: ResearchAgent):
        self.research_agent = research_agent
        self.is_running = False
        self.cycle_count = 0
    
    def start(self, n_cycles: int = 10, n_hypotheses_per_cycle: int = 20):
        """启动自动研究"""
        self.is_running = True
        
        while self.is_running and self.cycle_count < n_cycles:
            print(f"\n=== Research Cycle {self.cycle_count + 1}/{n_cycles} ===")
            
            try:
                result = self.research_agent.run_research_cycle(
                    n_hypotheses=n_hypotheses_per_cycle,
                    use_auto_factor=True
                )
                
                n_valid = len(result['valid_factors'])
                print(f"Cycle {self.cycle_count + 1} completed: {n_valid} valid factors found")
                
                if n_valid == 0:
                    print("No valid factors found, reducing threshold...")
                    self.research_agent.filter_valid_factors(min_ic=0.02, min_ir=0.3)
                
                self.research_agent.print_report()
                
            except Exception as e:
                print(f"Cycle {self.cycle_count + 1} failed: {str(e)}")
            
            self.cycle_count += 1
        
        self.is_running = False
        print("\n=== Auto Research Completed ===")
        print(f"Total cycles: {self.cycle_count}")
        print(f"Total valid factors: {len(self.research_agent.valid_factors)}")
    
    def stop(self):
        """停止自动研究"""
        self.is_running = False
