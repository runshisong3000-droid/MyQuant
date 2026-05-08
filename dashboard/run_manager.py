"""
Run Manager - 安全运行器

功能:
    - 只允许运行白名单脚本
    - 禁止任意 shell 命令
    - 记录运行历史
    - 保存运行日志
    - 防止重复运行
"""

import os
import sys
import json
import time
import subprocess
import uuid
from datetime import datetime
from typing import Dict, List, Optional

class RunManager:
    ALLOWED_SCRIPTS = [
        'scripts/run_student_laptop_pipeline.py',
        'scripts/run_research_lite_pipeline.py',
        'scripts/run_neural_factor_pipeline.py',
        'scripts/run_neural_encoder_comparison.py',
        'scripts/run_oos_validation_pipeline.py',
        'scripts/run_trading_constraints_pipeline.py'
    ]
    
    def __init__(self):
        self.project_root = os.path.dirname(os.path.dirname(__file__))
        self.logs_dir = os.path.join(self.project_root, 'logs', 'dashboard_runs')
        self.run_history_path = os.path.join(self.project_root, 'data', 'dashboard', 'run_history.json')
        
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.run_history_path), exist_ok=True)
    
    def is_script_allowed(self, script_path: str) -> bool:
        """检查脚本是否在白名单中"""
        normalized = os.path.normpath(script_path)
        allowed_normalized = [os.path.normpath(s) for s in self.ALLOWED_SCRIPTS]
        return normalized in allowed_normalized
    
    def generate_run_id(self) -> str:
        """生成唯一的 run_id"""
        return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    
    def get_run_history(self) -> List[Dict]:
        """获取运行历史"""
        if not os.path.exists(self.run_history_path):
            return []
        
        try:
            with open(self.run_history_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    
    def save_run_history(self, history: List[Dict]):
        """保存运行历史"""
        with open(self.run_history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
    
    def get_running_pipelines(self) -> List[str]:
        """获取正在运行的 pipeline"""
        history = self.get_run_history()
        return [h['pipeline_name'] for h in history if h['status'] == 'RUNNING']
    
    def get_pipeline_info(self, script_path: str) -> Dict:
        """获取 pipeline 信息"""
        info = {
            'scripts/run_student_laptop_pipeline.py': {
                'name': 'student_laptop_pipeline',
                'description': '基础研究 pipeline，使用真实 AkShare 数据',
                'estimated_time': '5-10 分钟',
                'input_data': ['AkShare 股票数据'],
                'output_artifacts': ['equity_curve.parquet', 'drawdown_curve.parquet', 'backtest_summary.json'],
                'category': 'backtest'
            },
            'scripts/run_research_lite_pipeline.py': {
                'name': 'research_lite_pipeline',
                'description': '可信度审计 pipeline，验证因子质量',
                'estimated_time': '10-20 分钟',
                'input_data': ['processed prices', 'formula factors'],
                'output_artifacts': ['factor_summary.parquet'],
                'category': 'audit'
            },
            'scripts/run_neural_factor_pipeline.py': {
                'name': 'neural_factor_pipeline',
                'description': '神经因子生成 pipeline，训练 autoencoder',
                'estimated_time': '15-30 分钟',
                'input_data': ['processed prices', 'sequence dataset'],
                'output_artifacts': ['neural_factors.parquet', 'neural_factor_summary.parquet'],
                'category': 'neural'
            },
            'scripts/run_neural_encoder_comparison.py': {
                'name': 'neural_encoder_comparison',
                'description': '编码器对比 pipeline，对比 MLP/CNN/Transformer',
                'estimated_time': '30-60 分钟',
                'input_data': ['processed prices'],
                'output_artifacts': ['encoder_comparison.parquet'],
                'category': 'neural'
            },
            'scripts/run_oos_validation_pipeline.py': {
                'name': 'oos_validation_pipeline',
                'description': '样本外验证 pipeline，对比 formula/neural/formula+neural',
                'estimated_time': '5-10 分钟',
                'input_data': ['factor_summary.parquet', 'neural_factors.parquet', 'prices'],
                'output_artifacts': ['oos_split_info.json', 'oos_feature_comparison.parquet', 'oos_rankic_series.parquet', 'oos_backtest_summary.parquet', 'oos_equity_curves.parquet'],
                'category': 'validation'
            },
            'scripts/run_trading_constraints_pipeline.py': {
                'name': 'trading_constraints_pipeline',
                'description': '交易约束 pipeline，检查 ST/停牌/涨跌停/新股/流动性/容量约束',
                'estimated_time': '5-10 分钟',
                'input_data': ['research_lite_prices.parquet'],
                'output_artifacts': ['trading_constraint_summary.parquet', 'tradable_mask.parquet', 'trading_constraint_report.json', 'constrained_backtest_summary.json', 'constrained_equity_curve.parquet', 'constrained_drawdown_curve.parquet'],
                'category': 'constraint'
            }
        }
        
        for allowed in self.ALLOWED_SCRIPTS:
            if script_path.endswith(allowed):
                return info.get(allowed, {'name': os.path.basename(script_path)})
        
        return {'name': os.path.basename(script_path)}
    
    def run_pipeline(self, script_path: str) -> Dict:
        """安全运行 pipeline"""
        if not self.is_script_allowed(script_path):
            return {
                'status': 'FAILED',
                'error': f"Script not in whitelist: {script_path}"
            }
        
        full_path = os.path.join(self.project_root, script_path)
        if not os.path.exists(full_path):
            return {
                'status': 'FAILED',
                'error': f"Script not found: {script_path}"
            }
        
        pipeline_info = self.get_pipeline_info(script_path)
        pipeline_name = pipeline_info['name']
        
        # 检查是否正在运行
        running_pipelines = self.get_running_pipelines()
        if pipeline_name in running_pipelines:
            return {
                'status': 'FAILED',
                'error': f"Pipeline already running: {pipeline_name}"
            }
        
        run_id = self.generate_run_id()
        log_path = os.path.join(self.logs_dir, f"{run_id}.log")
        
        # 创建运行记录
        run_record = {
            'run_id': run_id,
            'pipeline_name': pipeline_name,
            'script_path': script_path,
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'status': 'RUNNING',
            'return_code': None,
            'log_path': log_path,
            'generated_artifacts': []
        }
        
        # 更新运行历史
        history = self.get_run_history()
        history.insert(0, run_record)
        self.save_run_history(history)
        
        # 运行脚本
        try:
            with open(log_path, 'w', encoding='utf-8') as log_file:
                result = subprocess.run(
                    [sys.executable, full_path],
                    cwd=self.project_root,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True
                )
            
            # 更新记录
            run_record['end_time'] = datetime.now().isoformat()
            run_record['return_code'] = result.returncode
            run_record['status'] = 'SUCCESS' if result.returncode == 0 else 'FAILED'
            
            # 提取生成的 artifacts
            if run_record['status'] == 'SUCCESS':
                run_record['generated_artifacts'] = pipeline_info.get('output_artifacts', [])
            
            self.save_run_history(history)
            
            return run_record
            
        except Exception as e:
            run_record['end_time'] = datetime.now().isoformat()
            run_record['status'] = 'FAILED'
            run_record['error'] = str(e)
            self.save_run_history(history)
            return run_record
    
    def get_log_content(self, run_id: str) -> str:
        """获取运行日志"""
        log_path = os.path.join(self.logs_dir, f"{run_id}.log")
        if not os.path.exists(log_path):
            return "Log file not found"
        
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content[-5000:]  # 只返回最后 5000 字符
        except:
            return "Failed to read log"
    
    def get_latest_run(self, pipeline_name: str) -> Optional[Dict]:
        """获取某个 pipeline 的最近一次运行记录"""
        history = self.get_run_history()
        for record in history:
            if record['pipeline_name'] == pipeline_name:
                return record
        return None


run_manager = RunManager()
