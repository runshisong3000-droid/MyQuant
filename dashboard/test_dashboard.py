"""
Dashboard 测试脚本 - 验证报告解析功能
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd

def load_report(filepath):
    print("  尝试读取:", filepath)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        print("  读取成功，长度:", len(content))
        return content
    else:
        print("  文件不存在")
        return None

def parse_markdown_table(md_content, table_name=None):
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
        
        df = pd.DataFrame(data, columns=headers)
        return df
    
    except Exception as e:
        print("  解析错误:", str(e))
        return None

# 测试报告加载
reports_dir = os.path.join(os.path.dirname(__file__), '..', 'reports')
print("报告目录:", reports_dir)
print("目录存在:", os.path.exists(reports_dir))

print("\n" + "=" * 60)
print("Dashboard 测试 - 报告解析")
print("=" * 60)

# 加载报告
encoder_path = os.path.join(reports_dir, 'neural_encoder_comparison.md')
research_lite_path = os.path.join(reports_dir, 'research_lite_report.md')

print("\n1. 报告加载测试:")
print("\n  读取 encoder_comparison.md:")
encoder_report = load_report(encoder_path)

print("\n  读取 research_lite_report.md:")
research_lite_report = load_report(research_lite_path)

print("\n   encoder_comparison:", "OK" if encoder_report else "FAIL")
print("   research_lite:", "OK" if research_lite_report else "FAIL")

# 测试表格解析
print("\n2. 表格解析测试:")

encoder_df = parse_markdown_table(encoder_report, "Comparison Metrics")
if encoder_df is not None:
    print("   编码器对比表格: OK")
    print("   数据:")
    print(encoder_df.to_string())
else:
    print("   编码器对比表格: FAIL")

formula_df = parse_markdown_table(research_lite_report, "Formula Factors")
if formula_df is not None:
    print("   公式因子表格: OK")
    print("   行数:", len(formula_df))
else:
    print("   公式因子表格: FAIL")

leakage_df = parse_markdown_table(encoder_report, "Leakage Check")
if leakage_df is not None:
    print("   Leakage Check 表格: OK")
    print("   数据:")
    print(leakage_df.to_string())
else:
    print("   Leakage Check 表格: FAIL")

print("\n3. 数据处理测试:")
if encoder_df is not None:
    encoder_df['Avg RankIC'] = encoder_df['Avg RankIC'].astype(float)
    best_encoder = encoder_df.loc[encoder_df['Avg RankIC'].abs().idxmax()]
    print("   最佳编码器:", best_encoder['Encoder'])
    print("   平均 RankIC:", "{:.4f}".format(best_encoder['Avg RankIC']))

print("\n" + "=" * 60)
print("测试完成!")
print("=" * 60)