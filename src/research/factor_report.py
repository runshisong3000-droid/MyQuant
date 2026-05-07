"""
因子报告模块 - Factor Report

核心功能:
    - 生成因子分析报告
    - 可视化因子表现
    - 输出HTML/PDF报告
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import os


class FactorReport:
    """
    因子报告生成器
    
    生成专业的因子分析报告，包含：
    1. 因子基本信息
    2. IC分析结果
    3. 分组收益分析
    4. 换手率分析
    5. 衰减分析
    6. 相关性分析
    """

    def __init__(self, factor_name: str):
        self.factor_name = factor_name
        self.report_data = {
            'factor_name': factor_name,
            'generated_at': datetime.now().isoformat(),
            'sections': []
        }
    
    def add_section(self, title: str, content: Dict[str, Any]):
        """添加报告章节"""
        self.report_data['sections'].append({
            'title': title,
            'content': content
        })
    
    def add_factor_info(self, info: Dict[str, Any]):
        """添加因子基本信息"""
        self.add_section('因子基本信息', info)
    
    def add_ic_analysis(self, ic_summary: Dict[str, float], ic_series: pd.Series):
        """添加IC分析"""
        self.add_section('IC分析', {
            'summary': ic_summary,
            'timeseries': ic_series.to_dict()
        })
    
    def add_group_analysis(self, group_summary: Dict[str, float], group_returns: pd.DataFrame):
        """添加分组收益分析"""
        self.add_section('分组收益分析', {
            'summary': group_summary,
            'group_returns': group_returns.to_dict('records')
        })
    
    def add_turnover_analysis(self, turnover_series: pd.Series):
        """添加换手率分析"""
        self.add_section('换手率分析', {
            'mean_turnover': turnover_series.mean(),
            'timeseries': turnover_series.to_dict()
        })
    
    def add_decay_analysis(self, decay_data: pd.DataFrame):
        """添加衰减分析"""
        self.add_section('衰减分析', {
            'decay_curve': decay_data.to_dict('index')
        })
    
    def add_correlation_analysis(self, correlations: Dict[str, float]):
        """添加相关性分析"""
        self.add_section('相关性分析', correlations)
    
    def generate_html(self, output_path: Optional[str] = None) -> str:
        """生成HTML报告"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>因子分析报告 - {self.factor_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #3498db; border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
        .section {{ margin: 20px 0; }}
        .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #ecf0f1; border-radius: 5px; }}
        .metric-name {{ font-weight: bold; color: #7f8c8d; }}
        .metric-value {{ font-size: 24px; color: #2c3e50; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; }}
        .positive {{ color: green; }}
        .negative {{ color: red; }}
    </style>
</head>
<body>
    <h1>因子分析报告</h1>
    <h2>因子名称: {self.factor_name}</h2>
    <p>生成时间: {self.report_data['generated_at']}</p>
"""
        
        for section in self.report_data['sections']:
            html += f"<div class='section'><h2>{section['title']}</h2>"
            
            if section['title'] == '因子基本信息':
                for key, value in section['content'].items():
                    html += f"<p><strong>{key}:</strong> {value}</p>"
            
            elif section['title'] == 'IC分析':
                summary = section['content']['summary']
                html += """
                <div class='metric'>
                    <div class='metric-name'>Mean IC</div>
                    <div class='metric-value {color}'>{value:.4f}</div>
                </div>
                <div class='metric'>
                    <div class='metric-name'>IC Std</div>
                    <div class='metric-value'>{std:.4f}</div>
                </div>
                <div class='metric'>
                    <div class='metric-name'>IR</div>
                    <div class='metric-value {ir_color}'>{ir:.4f}</div>
                </div>
                <div class='metric'>
                    <div class='metric-name'>Positive IC%</div>
                    <div class='metric-value'>{positive:.2%}</div>
                </div>
                """.format(
                    color='positive' if summary['mean_ic'] > 0 else 'negative',
                    value=summary['mean_ic'],
                    std=summary['std_ic'],
                    ir_color='positive' if summary['ir'] > 0 else 'negative',
                    ir=summary['ir'],
                    positive=summary['positive_ic_ratio']
                )
            
            elif section['title'] == '分组收益分析':
                summary = section['content']['summary']
                html += """
                <table>
                    <tr>
                        <th>分组</th>
                        <th>平均收益</th>
                        <th>标准差</th>
                    </tr>
                """
                n_groups = sum(1 for k in summary.keys() if k.startswith('group_') and '_mean' in k)
                for i in range(1, n_groups + 1):
                    html += f"""
                    <tr>
                        <td>Group {i}</td>
                        <td>{summary[f'group_{i}_mean']:.4f}</td>
                        <td>{summary[f'group_{i}_std']:.4f}</td>
                    </tr>
                    """
                html += """
                </table>
                <br>
                <div class='metric'>
                    <div class='metric-name'>Long-Short Mean</div>
                    <div class='metric-value {ls_color}'>{ls:.4f}</div>
                </div>
                <div class='metric'>
                    <div class='metric-name'>Long-Short IR</div>
                    <div class='metric-value {ls_ir_color}'>{ls_ir:.4f}</div>
                </div>
                <div class='metric'>
                    <div class='metric-name'>Win Rate</div>
                    <div class='metric-value'>{win_rate:.2%}</div>
                </div>
                <div class='metric'>
                    <div class='metric-name'>Monotonicity</div>
                    <div class='metric-value {mono_color}'>{mono:.4f}</div>
                </div>
                """.format(
                    ls_color='positive' if summary['long_short_mean'] > 0 else 'negative',
                    ls=summary['long_short_mean'],
                    ls_ir_color='positive' if summary['long_short_ir'] > 0 else 'negative',
                    ls_ir=summary['long_short_ir'],
                    win_rate=summary['long_short_win_rate'],
                    mono_color='positive' if summary['monotonicity'] > 0 else 'negative',
                    mono=summary['monotonicity']
                )
            
            elif section['title'] == '换手率分析':
                html += f"<p><strong>平均换手率:</strong> {section['content']['mean_turnover']:.2%}</p>"
            
            elif section['title'] == '衰减分析':
                decay = section['content']['decay_curve']
                html += """
                <table>
                    <tr><th>滞后</th><th>Mean IC</th><th>IR</th></tr>
                """
                for lag, values in decay.items():
                    html += f"<tr><td>{lag}</td><td>{values['mean_ic']:.4f}</td><td>{values['ir']:.4f}</td></tr>"
                html += "</table>"
            
            elif section['title'] == '相关性分析':
                html += "<table><tr><th>因子</th><th>相关性</th></tr>"
                for factor, corr in section['content'].items():
                    color = 'positive' if corr > 0 else 'negative'
                    html += f"<tr><td>{factor}</td><td class='{color}'>{corr:.4f}</td></tr>"
                html += "</table>"
            
            html += "</div>"
        
        html += "</body></html>"
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
        
        return html
    
    def save_json(self, output_path: str):
        """保存JSON报告"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.report_data, f, indent=2, ensure_ascii=False)


class FactorReportGenerator:
    """
    因子报告生成器工厂
    
    批量生成因子报告
    """

    def __init__(self):
        pass
    
    @staticmethod
    def generate_from_test(factor_test, factor_name: str) -> FactorReport:
        """
        从因子测试结果生成报告
        
        Args:
            factor_test: FactorTest对象
            factor_name: 因子名称
            
        Returns:
            FactorReport对象
        """
        report = FactorReport(factor_name)
        
        report.add_factor_info({
            '测试日期': datetime.now().strftime('%Y-%m-%d'),
            '数据周期': f"{factor_test.dates[0].strftime('%Y-%m-%d')} 至 {factor_test.dates[-1].strftime('%Y-%m-%d')}",
            '股票数量': len(factor_test.factor_data.index.get_level_values(1).unique())
        })
        
        ic_summary = factor_test.calculate_ic_summary()
        ic_series = factor_test.calculate_ic()
        report.add_ic_analysis(ic_summary, ic_series)
        
        group_summary = factor_test.calculate_group_summary()
        group_returns = factor_test.calculate_group_returns()
        report.add_group_analysis(group_summary, group_returns)
        
        turnover_series = factor_test.calculate_turnover()
        report.add_turnover_analysis(turnover_series)
        
        decay_data = factor_test.calculate_decay()
        report.add_decay_analysis(decay_data)
        
        return report
    
    @staticmethod
    def generate_batch(
        factor_tests: Dict[str, 'FactorTest'],
        output_dir: str = 'factor_reports'
    ):
        """
        批量生成因子报告
        
        Args:
            factor_tests: 因子测试字典
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for factor_name, factor_test in factor_tests.items():
            report = FactorReportGenerator.generate_from_test(factor_test, factor_name)
            
            html_path = os.path.join(output_dir, f'{factor_name}_report.html')
            json_path = os.path.join(output_dir, f'{factor_name}_report.json')
            
            report.generate_html(html_path)
            report.save_json(json_path)
            
            print(f"Generated report for {factor_name}")