"""
Dashboard Data Loader - 数据读取层

功能:
    - 安全读取 markdown 报告
    - 解析报告中的表格和数据
    - 读取 parquet 和 json 文件
    - 返回友好错误信息
    - 支持按 profile 读取数据
    - 不联网、不训练、不修改文件
"""

import os
import json
import pandas as pd
import yaml

class DataLoader:
    def __init__(self, profile_name=None):
        self.reports_dir = os.path.join(os.path.dirname(__file__), '..', 'reports')
        self.data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        self.dashboard_dir = os.path.join(self.data_dir, 'dashboard')
        self.profile_name = profile_name or 'research_lite'
        
    def set_profile(self, profile_name):
        """设置当前 profile"""
        self.profile_name = profile_name
    
    def get_current_profile(self):
        """获取当前 profile"""
        return self.profile_name
    
    def get_profile_config(self, profile_name=None):
        """获取 profile 配置"""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'compute_profile.yaml')
        if not os.path.exists(config_path):
            return None
        
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            profiles = config.get('profiles', {})
            if profile_name is None:
                profile_name = self.profile_name
            
            return profiles.get(profile_name)
        except Exception:
            return None
    
    def get_profile_dashboard_dir(self):
        """获取 profile-specific dashboard 目录"""
        return os.path.join(self.dashboard_dir, 'profiles', self.profile_name)
    
    def get_available_profiles(self):
        """获取可用的 profiles"""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'compute_profile.yaml')
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            profiles = config.get('profiles', {})
            return list(profiles.keys())
        
        return ['student_laptop', 'research_lite', 'research_medium_trial', 'research_medium']
    
    def profile_has_artifacts(self):
        """检查当前 profile 是否有 artifacts"""
        profile_dir = self.get_profile_dashboard_dir()
        if not os.path.exists(profile_dir):
            return False
        
        files = os.listdir(profile_dir)
        parquet_files = [f for f in files if f.endswith('.parquet')]
        json_files = [f for f in files if f.endswith('.json')]
        
        return len(parquet_files) > 0 or len(json_files) > 0
    
    def get_profile_manifest(self):
        """获取 profile manifest"""
        manifest_path = os.path.join(self.get_profile_dashboard_dir(), 'profile_manifest.json')
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return None
        return None
    
    def safe_file_exists(self, filepath):
        """检查文件是否存在"""
        try:
            return os.path.exists(filepath)
        except Exception:
            return False
    
    def load_markdown_report(self, filename):
        """加载 markdown 报告文件"""
        filepath = os.path.join(self.reports_dir, filename)
        if not self.safe_file_exists(filepath):
            return None, f"文件不存在: {filename}"
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            return content, None
        except Exception as e:
            return None, f"读取失败: {str(e)}"
    
    def load_json(self, filename, directory='reports'):
        """加载 json 文件"""
        if directory == 'dashboard':
            filepath = os.path.join(self.get_profile_dashboard_dir(), filename)
        else:
            filepath = os.path.join(self.reports_dir, filename)
            
        if not self.safe_file_exists(filepath):
            return None, f"文件不存在: {filename}"
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data, None
        except Exception as e:
            return None, f"读取失败: {str(e)}"
    
    def load_parquet(self, relative_path):
        """加载 parquet 文件"""
        filepath = os.path.join(self.data_dir, relative_path)
        if not self.safe_file_exists(filepath):
            return None, f"文件不存在: {relative_path}"
        
        try:
            df = pd.read_parquet(filepath)
            return df, None
        except Exception as e:
            return None, f"读取失败: {str(e)}"
    
    def load_dashboard_parquet(self, filename):
        """加载 profile-specific dashboard 目录下的 parquet 文件"""
        filepath = os.path.join(self.get_profile_dashboard_dir(), filename)
        if not self.safe_file_exists(filepath):
            return None, f"文件不存在: {filename}"
        
        try:
            df = pd.read_parquet(filepath)
            return df, None
        except Exception as e:
            return None, f"读取失败: {str(e)}"
    
    def parse_markdown_table(self, md_content, table_name=None):
        """解析 markdown 表格"""
        if not md_content:
            return None
        
        lines = md_content.split('\n')
        table_start = 0
        
        if table_name:
            for i, line in enumerate(lines):
                if table_name.lower() in line.lower():
                    for j in range(i, min(i+10, len(lines))):
                        if lines[j].strip().startswith('|'):
                            table_start = j
                            break
                    break
        
        table_lines = []
        in_table = False
        
        for i in range(table_start, len(lines)):
            line = lines[i].strip()
            if line.startswith('|'):
                table_lines.append(line)
                in_table = True
            elif in_table and line == '' and len(table_lines) > 2:
                break
        
        if len(table_lines) < 3:
            return None
        
        try:
            header_line = table_lines[0]
            headers = [h.strip() for h in header_line.split('|')[1:-1]]
            
            data = []
            for line in table_lines[2:]:
                row = [cell.strip() for cell in line.split('|')[1:-1]]
                if len(row) == len(headers):
                    data.append(row)
            
            return pd.DataFrame(data, columns=headers)
        except Exception:
            return None
    
    def extract_section(self, md_content, section_name):
        """提取 markdown 中的章节"""
        if not md_content:
            return None
        
        lines = md_content.split('\n')
        start_idx = None
        end_idx = None
        
        for i, line in enumerate(lines):
            if line.startswith('## ') and section_name.lower() in line.lower():
                start_idx = i + 1
                continue
            if start_idx is not None and line.startswith('## '):
                end_idx = i
                break
        
        if start_idx is None:
            return None
        
        end_idx = end_idx if end_idx else len(lines)
        section_lines = lines[start_idx:end_idx]
        
        return '\n'.join(section_lines).strip()
    
    def parse_research_lite_report(self):
        """解析 research_lite_report.md"""
        content, error = self.load_markdown_report('research_lite_report.md')
        if error:
            return None, error
        
        result = {
            'data_overview': self.parse_markdown_table(content, 'Data Overview'),
            'formula_factors': self.parse_markdown_table(content, 'Formula Factors'),
            'neural_factors': self.parse_markdown_table(content, 'Neural Factors'),
            'leakage_check': self.parse_markdown_table(content, 'Leakage Check'),
            'raw_content': content
        }
        return result, None
    
    def parse_profile_report(self):
        """解析当前 profile 的报告"""
        report_name = f'{self.profile_name}_report.md'
        content, error = self.load_markdown_report(report_name)
        if error:
            return None, error
        
        result = {
            'data_overview': self.parse_markdown_table(content, 'Data Overview'),
            'formula_factors': self.parse_markdown_table(content, 'Formula Factors'),
            'neural_factors': self.parse_markdown_table(content, 'Neural Factors'),
            'leakage_check': self.parse_markdown_table(content, 'Leakage Check'),
            'raw_content': content
        }
        return result, None
    
    def parse_student_laptop_report(self):
        """解析 student_laptop_report.md"""
        content, error = self.load_markdown_report('student_laptop_report.md')
        if error:
            return None, error
        
        result = {
            'performance_metrics': self.parse_markdown_table(content, 'Performance Metrics'),
            'audit_checklist': self.parse_markdown_table(content, 'Audit Checklist'),
            'raw_content': content
        }
        return result, None
    
    def parse_neural_factor_report(self):
        """解析 neural_factor_report.md"""
        content, error = self.load_markdown_report('neural_factor_report.md')
        if error:
            return None, error
        
        result = {
            'neural_factor_evaluation': self.parse_markdown_table(content, 'Neural Factor Evaluation'),
            'leakage_check': self.parse_markdown_table(content, 'Leakage Check'),
            'raw_content': content
        }
        return result, None
    
    def parse_encoder_comparison_report(self):
        """解析 neural_encoder_comparison.md"""
        content, error = self.load_markdown_report('neural_encoder_comparison.md')
        if error:
            return None, error
        
        result = {
            'comparison_metrics': self.parse_markdown_table(content, 'Comparison Metrics'),
            'mlp_factors': self.parse_markdown_table(content, 'MLP'),
            'cnn_factors': self.parse_markdown_table(content, 'CNN'),
            'transformer_factors': self.parse_markdown_table(content, 'TRANSFORMER'),
            'leakage_check': self.parse_markdown_table(content, 'Leakage Check'),
            'raw_content': content
        }
        return result, None
    
    def load_neural_factors_parquet(self):
        """加载 neural_factors.parquet"""
        return self.load_dashboard_parquet('neural_factors.parquet')
    
    def load_research_lite_prices(self):
        """加载 research_lite_prices.parquet"""
        return self.load_parquet('processed/research_lite_prices.parquet')
    
    def load_neural_factor_metadata(self):
        """加载 neural_factor_metadata.json"""
        return self.load_json('neural_factor_metadata.json')
    
    def load_dashboard_manifest(self):
        """加载 profile_manifest.json"""
        return self.load_json('profile_manifest.json', directory='dashboard')
    
    def load_equity_curve(self):
        """加载 equity_curve.parquet"""
        return self.load_dashboard_parquet('equity_curve.parquet')
    
    def load_drawdown_curve(self):
        """加载 drawdown_curve.parquet"""
        return self.load_dashboard_parquet('drawdown_curve.parquet')
    
    def load_backtest_summary(self):
        """加载 backtest_summary.json"""
        return self.load_json('backtest_summary.json', directory='dashboard')
    
    def load_factor_ic_series(self):
        """加载 factor_ic_series.parquet"""
        return self.load_dashboard_parquet('factor_ic_series.parquet')
    
    def load_factor_summary(self):
        """加载 factor_summary.parquet"""
        return self.load_dashboard_parquet('factor_summary.parquet')
    
    def load_factor_correlation(self):
        """加载 factor_correlation.parquet"""
        return self.load_dashboard_parquet('factor_correlation.parquet')
    
    def load_neural_factors_dashboard(self):
        """加载 dashboard 目录下的 neural_factors.parquet"""
        return self.load_dashboard_parquet('neural_factors.parquet')
    
    def load_neural_factor_summary(self):
        """加载 neural_factor_summary.parquet"""
        return self.load_dashboard_parquet('neural_factor_summary.parquet')
    
    def load_formula_factor_metadata(self):
        """加载 formula_factor_metadata.json"""
        return self.load_json('formula_factor_metadata.json', directory='dashboard')
    
    def load_formula_factors_panel(self):
        """加载 formula_factors.parquet"""
        return self.load_dashboard_parquet('formula_factors.parquet')
    
    def load_encoder_comparison_data(self):
        """加载 encoder_comparison.parquet"""
        return self.load_dashboard_parquet('encoder_comparison.parquet')
    
    def load_reliability_status(self):
        """加载 reliability_status.json (优先)，否则返回默认状态"""
        status, error = self.load_json('reliability_status.json', directory='dashboard')
        if error:
            return self._get_default_reliability_status()
        
        checks = []
        for key, value in status.items():
            if key in ['real_data', 'simulated_data_forbidden', 'future_leakage_check',
                       'target_alignment', 'signal_trade_lag', 'rankic_cross_sectional',
                       'icir_safe_handling', 'sharpe_safe_handling', 'multiindex_alignment',
                       'scaler_fit_scope', 'suspended_stock_filter', 'limit_up_down_filter',
                       'st_filter', 'out_of_sample_validation', 'paper_trading']:
                checks.append({
                    'item': key.replace('_', ' ').title(),
                    'status': value,
                    'details': self._get_status_details(key)
                })
        return checks
    
    def _get_default_reliability_status(self):
        """获取默认可信度状态"""
        return [
            {'item': 'Real Data', 'status': 'OK', 'details': 'Using AkShare real data'},
            {'item': 'Simulated Data Forbidden', 'status': 'OK', 'details': 'Not using simulated data'},
            {'item': 'Future Leakage Check', 'status': 'WARN', 'details': 'Not verified'},
            {'item': 'Target Alignment', 'status': 'OK', 'details': 'signal_date < target_date'},
            {'item': 'Signal-Trade Lag', 'status': 'OK', 'details': 't -> t+1 trading'},
            {'item': 'RankIC Cross-Sectional', 'status': 'OK', 'details': 'Calculated per date'},
            {'item': 'ICIR Safe Handling', 'status': 'OK', 'details': 'Handles edge cases'},
            {'item': 'Sharpe Safe Handling', 'status': 'OK', 'details': 'Handles NaN/inf'},
            {'item': 'MultiIndex Alignment', 'status': 'OK', 'details': 'Properly formatted'},
            {'item': 'Scaler Fit Scope', 'status': 'OK', 'details': 'Train only'},
            {'item': 'Suspended Stock Filter', 'status': 'TODO', 'details': 'Not implemented'},
            {'item': 'Limit-up/down Filter', 'status': 'TODO', 'details': 'Not implemented'},
            {'item': 'ST Filter', 'status': 'TODO', 'details': 'Not implemented'},
            {'item': 'Out-of-Sample Validation', 'status': 'WARN', 'details': 'Needs more time'},
            {'item': 'Paper Trading', 'status': 'TODO', 'details': 'Not implemented'}
        ]
    
    def _get_status_details(self, key):
        """获取状态详情"""
        details_map = {
            'real_data': 'Using AkShare real data',
            'simulated_data_forbidden': 'Not using simulated data',
            'future_leakage_check': 'All checks passed',
            'target_alignment': 'signal_date < target_date',
            'signal_trade_lag': 't -> t+1 trading',
            'rankic_cross_sectional': 'Calculated per date',
            'icir_safe_handling': 'Handles edge cases',
            'sharpe_safe_handling': 'Handles NaN/inf',
            'multiindex_alignment': 'Properly formatted',
            'scaler_fit_scope': 'Train only',
            'suspended_stock_filter': 'Not implemented',
            'limit_up_down_filter': 'Not implemented',
            'st_filter': 'Not implemented',
            'out_of_sample_validation': 'Needs more time',
            'paper_trading': 'Not implemented'
        }
        return details_map.get(key, '')
    
    def get_available_reports(self):
        """获取可用报告列表"""
        reports = [
            'student_laptop_report.md',
            'research_lite_report.md',
            'research_medium_trial_report.md',
            'research_medium_report.md',
            'neural_factor_report.md',
            'neural_encoder_comparison.md',
            'data_source_reliability_report.md'
        ]
        
        result = []
        for report in reports:
            path = os.path.join(self.reports_dir, report)
            exists = self.safe_file_exists(path)
            result.append({
                'name': report,
                'exists': exists
            })
        return result
    
    def get_pipeline_status(self):
        """获取 pipeline 状态"""
        manifest = self.get_profile_manifest()
        if manifest and 'pipeline_status' in manifest:
            status = []
            for name, status_val in manifest['pipeline_status'].items():
                desc_map = {
                    'student_laptop_pipeline': '基础研究',
                    'neural_factor_pipeline': '神经因子',
                    'research_lite_pipeline': '可信度审计',
                    'research_medium_trial_pipeline': '中等样本试跑',
                    'research_medium_pipeline': '中等样本验证',
                    'neural_encoder_comparison': '编码器对比'
                }
                status.append({
                    'name': name,
                    'status': status_val.upper(),
                    'desc': desc_map.get(name, name)
                })
            return status
        
        research_lite, _ = self.parse_research_lite_report()
        encoder_compare, _ = self.parse_encoder_comparison_report()
        
        return [
            {'name': 'student_laptop_pipeline', 'status': 'OK', 'desc': '基础研究'},
            {'name': 'neural_factor_pipeline', 'status': 'OK', 'desc': '神经因子'},
            {'name': 'research_lite_pipeline', 'status': 'OK' if research_lite else 'UNKNOWN', 'desc': '可信度审计'},
            {'name': 'neural_encoder_comparison', 'status': 'OK' if encoder_compare else 'UNKNOWN', 'desc': '编码器对比'}
        ]
    
    def get_dashboard_artifacts_status(self):
        """获取 dashboard artifacts 状态"""
        manifest = self.get_profile_manifest()
        if manifest and 'artifacts' in manifest:
            return manifest['artifacts']
        
        profile_dir = self.get_profile_dashboard_dir()
        if os.path.exists(profile_dir):
            files = os.listdir(profile_dir)
            artifacts = {}
            for f in files:
                artifacts[f] = {'exists': True}
            return artifacts
        
        default_artifacts = {
            'equity_curve.parquet': {'exists': False},
            'drawdown_curve.parquet': {'exists': False},
            'backtest_summary.json': {'exists': False},
            'factor_ic_series.parquet': {'exists': False},
            'factor_summary.parquet': {'exists': False},
            'factor_correlation.parquet': {'exists': False},
            'neural_factors.parquet': {'exists': False},
            'neural_factor_summary.parquet': {'exists': False},
            'encoder_comparison.parquet': {'exists': False},
            'reliability_status.json': {'exists': True},
            'trading_constraint_summary.parquet': {'exists': False},
            'tradable_mask.parquet': {'exists': False},
            'trading_constraint_report.json': {'exists': False},
            'constrained_backtest_summary.json': {'exists': False},
            'constrained_equity_curve.parquet': {'exists': False},
            'constrained_drawdown_curve.parquet': {'exists': False},
            'formula_factors.parquet': {'exists': False},
            'formula_factor_metadata.json': {'exists': False},
            'profile_manifest.json': {'exists': False}
        }
        return default_artifacts
    
    def load_trading_constraint_summary(self):
        """加载 trading_constraint_summary.parquet"""
        return self.load_dashboard_parquet('trading_constraint_summary.parquet')
    
    def load_tradable_mask(self):
        """加载 tradable_mask.parquet"""
        return self.load_dashboard_parquet('tradable_mask.parquet')
    
    def load_trading_constraint_report(self):
        """加载 trading_constraint_report.json"""
        return self.load_json('trading_constraint_report.json', directory='dashboard')
    
    def load_constrained_backtest_summary(self):
        """加载 constrained_backtest_summary.json"""
        return self.load_json('constrained_backtest_summary.json', directory='dashboard')
    
    def load_constrained_equity_curve(self):
        """加载 constrained_equity_curve.parquet"""
        return self.load_dashboard_parquet('constrained_equity_curve.parquet')
    
    def load_constrained_drawdown_curve(self):
        """加载 constrained_drawdown_curve.parquet"""
        return self.load_dashboard_parquet('constrained_drawdown_curve.parquet')
    
    def parse_trading_constraints_report(self):
        """解析 trading_constraints_report.md"""
        content, error = self.load_markdown_report('trading_constraints_report.md')
        if error:
            return None, error
        
        result = {
            'data_availability': self.parse_markdown_table(content, 'Data Availability'),
            'constraint_rules': self.parse_markdown_table(content, 'Constraint Summary'),
            'constraint_summary': self.parse_markdown_table(content, 'Constraint Summary'),
            'backtest_impact': self.parse_markdown_table(content, 'Backtest Impact'),
            'raw_content': content
        }
        return result, None
    
    def get_profile_info(self):
        """获取当前 profile 的信息"""
        manifest = self.get_profile_manifest()
        if manifest:
            return {
                'profile': manifest.get('profile', self.profile_name),
                'stock_count_target': manifest.get('stock_count_target'),
                'stock_count_actual': manifest.get('stock_count_actual'),
                'history_months_target': manifest.get('history_months_target'),
                'date_start': manifest.get('date_start'),
                'date_end': manifest.get('date_end'),
                'can_use_for_live_trading': manifest.get('can_use_for_live_trading', False)
            }
        
        return {
            'profile': self.profile_name,
            'stock_count_target': None,
            'stock_count_actual': None,
            'history_months_target': None,
            'date_start': None,
            'date_end': None,
            'can_use_for_live_trading': False
        }

loader = DataLoader()
