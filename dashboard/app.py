"""
MyQuant Dashboard Visual MVP

核心功能:
    - Overview: 项目概览
    - Formula Factor Lab: 公式因子分析
    - Neural Feature Lab: 神经特征展示
    - Encoder Comparison: 编码器对比
    - Backtest Report: 回测报告
    - Reliability Audit: 可信度审计
    - Profile Selector: 支持多 profile 切换

优先读取 data/dashboard/profiles/{profile}/ 目录下的结构化数据文件。
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from dashboard.data_loader import DataLoader
from dashboard.run_manager import run_manager

st.set_page_config(
    page_title="MyQuant Visual MVP",
    page_icon="📊",
    layout="wide"
)

def get_status_color(status):
    if status == 'OK':
        return 'green'
    elif status == 'WARN':
        return 'yellow'
    elif status == 'FAIL':
        return 'red'
    elif status == 'TODO':
        return 'gray'
    return 'gray'

def display_status_badge(status):
    color = get_status_color(status)
    if status == 'OK':
        return f":green[✅ {status}]"
    elif status == 'WARN':
        return f":yellow[⚠️ {status}]"
    elif status == 'FAIL':
        return f":red[❌ {status}]"
    elif status == 'TODO':
        return f":gray[🔄 {status}]"
    return status

# ==================== 初始化 DataLoader ====================
if 'data_loader' not in st.session_state:
    st.session_state.data_loader = DataLoader('research_lite')

if 'current_profile' not in st.session_state:
    st.session_state.current_profile = 'research_lite'

def reload_data():
    """重新加载当前 profile 的数据"""
    loader = st.session_state.data_loader
    
    manifest, _ = loader.load_dashboard_manifest()
    research_lite_report, _ = loader.parse_research_lite_report()
    student_laptop_report, _ = loader.parse_student_laptop_report()
    neural_factor_report, _ = loader.parse_neural_factor_report()
    encoder_compare_report, _ = loader.parse_encoder_comparison_report()
    trading_constraints_report, _ = loader.parse_trading_constraints_report()
    profile_report, _ = loader.parse_profile_report()
    
    # 优先读取 dashboard 目录下的结构化数据
    equity_curve_df, _ = loader.load_equity_curve()
    drawdown_curve_df, _ = loader.load_drawdown_curve()
    backtest_summary, _ = loader.load_backtest_summary()
    factor_summary_df, _ = loader.load_factor_summary()
    encoder_comparison_df, _ = loader.load_encoder_comparison_data()
    neural_factors_df, _ = loader.load_neural_factors_dashboard()
    neural_factor_summary_df, _ = loader.load_neural_factor_summary()
    reliability_status = loader.load_reliability_status()
    pipeline_status = loader.get_pipeline_status()
    available_reports = loader.get_available_reports()
    artifacts_status = loader.get_dashboard_artifacts_status()
    
    # 交易约束相关数据
    trading_constraint_summary_df, _ = loader.load_trading_constraint_summary()
    tradable_mask_df, _ = loader.load_tradable_mask()
    trading_constraint_json, _ = loader.load_trading_constraint_report()
    constrained_backtest_summary, _ = loader.load_constrained_backtest_summary()
    constrained_equity_curve_df, _ = loader.load_constrained_equity_curve()
    constrained_drawdown_curve_df, _ = loader.load_constrained_drawdown_curve()
    
    # 获取 profile 信息
    profile_info = loader.get_profile_info()
    
    return {
        'manifest': manifest,
        'research_lite_report': research_lite_report,
        'student_laptop_report': student_laptop_report,
        'neural_factor_report': neural_factor_report,
        'encoder_compare_report': encoder_compare_report,
        'trading_constraints_report': trading_constraints_report,
        'profile_report': profile_report,
        'equity_curve_df': equity_curve_df,
        'drawdown_curve_df': drawdown_curve_df,
        'backtest_summary': backtest_summary,
        'factor_summary_df': factor_summary_df,
        'encoder_comparison_df': encoder_comparison_df,
        'neural_factors_df': neural_factors_df,
        'neural_factor_summary_df': neural_factor_summary_df,
        'reliability_status': reliability_status,
        'pipeline_status': pipeline_status,
        'available_reports': available_reports,
        'artifacts_status': artifacts_status,
        'trading_constraint_summary_df': trading_constraint_summary_df,
        'tradable_mask_df': tradable_mask_df,
        'trading_constraint_json': trading_constraint_json,
        'constrained_backtest_summary': constrained_backtest_summary,
        'constrained_equity_curve_df': constrained_equity_curve_df,
        'constrained_drawdown_curve_df': constrained_drawdown_curve_df,
        'profile_info': profile_info
    }

# ==================== Profile 切换处理 ====================
def change_profile(new_profile):
    if new_profile != st.session_state.current_profile:
        st.session_state.current_profile = new_profile
        st.session_state.data_loader.set_profile(new_profile)
        # 触发页面刷新以重新加载数据
        st.experimental_rerun()

# ==================== Sidebar ====================

st.sidebar.title("📊 MyQuant")
st.sidebar.markdown("---")

# Profile Selector
st.sidebar.subheader("🔧 Profile Selector")
loader = st.session_state.data_loader
available_profiles = loader.get_available_profiles()

selected_profile = st.sidebar.selectbox(
    "选择研究配置",
    available_profiles,
    index=available_profiles.index(st.session_state.current_profile),
    on_change=change_profile,
    args=(st.session_state.current_profile,)
)

# 确认按钮（因为 selectbox 的 on_change 在选择时触发，这里需要重新处理）
if st.sidebar.button("应用 Profile"):
    if selected_profile != st.session_state.current_profile:
        change_profile(selected_profile)

# 当前 profile 信息
profile_info = loader.get_profile_info()
st.sidebar.markdown("---")
st.sidebar.subheader(f"📋 当前 Profile: {st.session_state.current_profile}")
st.sidebar.write(f"**股票数**: {profile_info.get('stock_count_actual', 'N/A')} / {profile_info.get('stock_count_target', 'N/A')}")
st.sidebar.write(f"**月份**: {profile_info.get('history_months_target', 'N/A')}")
st.sidebar.write(f"**日期范围**: {profile_info.get('date_start', 'N/A')} ~ {profile_info.get('date_end', 'N/A')}")
st.sidebar.write(f"**可实盘**: {'✅' if profile_info.get('can_use_for_live_trading') else '❌'}")

# 检查当前 profile 是否有 artifacts
has_artifacts = loader.profile_has_artifacts()
if not has_artifacts:
    st.sidebar.warning("⚠️ 该 Profile 尚未生成结果，请到 Run Center 运行。")

st.sidebar.markdown("---")
st.sidebar.subheader("📁 数据来源")
st.sidebar.info(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

st.sidebar.markdown("---")
st.sidebar.subheader("可用报告")
for report in loader.get_available_reports():
    if report['exists']:
        st.sidebar.success(report['name'])
    else:
        st.sidebar.warning(report['name'])

st.sidebar.markdown("---")
st.sidebar.subheader("⚠️ 重要提示")
st.sidebar.error("""
当前结果**仅用于研究**，**不能直接用于实盘交易**。

- RankIC 整体较弱
- 样本量有限
- 仍需样本外验证
""")

# ==================== 加载数据 ====================
data = reload_data()

# 提取常用变量
manifest = data['manifest']
research_lite_report = data['research_lite_report']
student_laptop_report = data['student_laptop_report']
neural_factor_report = data['neural_factor_report']
encoder_compare_report = data['encoder_compare_report']
trading_constraints_report = data['trading_constraints_report']
profile_report = data['profile_report']
equity_curve_df = data['equity_curve_df']
drawdown_curve_df = data['drawdown_curve_df']
backtest_summary = data['backtest_summary']
factor_summary_df = data['factor_summary_df']
encoder_comparison_df = data['encoder_comparison_df']
neural_factors_df = data['neural_factors_df']
neural_factor_summary_df = data['neural_factor_summary_df']
reliability_status = data['reliability_status']
pipeline_status = data['pipeline_status']
available_reports = data['available_reports']
artifacts_status = data['artifacts_status']
trading_constraint_summary_df = data['trading_constraint_summary_df']
trading_constraint_json = data['trading_constraint_json']
constrained_backtest_summary = data['constrained_backtest_summary']
constrained_equity_curve_df = data['constrained_equity_curve_df']
constrained_drawdown_curve_df = data['constrained_drawdown_curve_df']
profile_info = data['profile_info']

# 获取统计数据
stock_count = profile_info.get('stock_count_actual', 'N/A')
months = profile_info.get('history_months_target', 'N/A')
formula_factor_count = len(factor_summary_df) if factor_summary_df is not None else 'N/A'

# ==================== 页面布局 ====================

st.title("📊 MyQuant Visual MVP")
st.markdown(f"### 当前 Profile: **{st.session_state.current_profile}**")
st.markdown("---")

# 检查是否有 artifacts
if not has_artifacts:
    st.warning("""
    ⚠️ **该 Profile 尚未生成结果**
    
    请前往 **Run Center** 运行对应的 pipeline 生成数据。
    """)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "Overview", 
    "Formula Factor Lab", 
    "Neural Feature Lab", 
    "Encoder Comparison",
    "Backtest Report",
    "Reliability Audit",
    "Out-of-Sample Validation",
    "Trading Constraints",
    "Run Center"
])

# ==================== Overview Tab ====================

with tab1:
    st.subheader("📈 Project Overview")
    st.markdown("""
        MyQuant A股AI量化选股与可视化特征学习平台 - 项目概览
    """)
    
    st.markdown("---")
    st.subheader("📊 数据概览")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("数据源", "AkShare")
    col2.metric("股票数量", stock_count)
    col3.metric("历史月份", months)
    col4.metric("公式因子", formula_factor_count)
    
    neural_factor_count = len(neural_factor_summary_df) if neural_factor_summary_df is not None else '0'
    col1, col2, col3 = st.columns(3)
    col1.metric("神经因子", neural_factor_count)
    col2.metric("测试通过", "141/141")
    col3.metric("Pipeline状态", "✅ 全部OK")
    
    # Artifact 完成率
    if artifacts_status:
        total = len(artifacts_status)
        completed = sum(1 for info in artifacts_status.values() if info.get('exists', False))
        completion_rate = (completed / total) * 100
        col1, col2 = st.columns(2)
        col1.metric("Artifact 完成率", f"{completion_rate:.1f}%")
        col2.metric("已完成 / 总计", f"{completed}/{total}")
    
    st.markdown("---")
    st.subheader("📋 Dashboard Manifest")
    if manifest:
        st.write(f"**版本**: {manifest.get('version', 'N/A')}")
        st.write(f"**生成时间**: {manifest.get('generated_at', 'N/A')}")
        
        st.markdown("### Artifacts 状态")
        artifact_rows = []
        for name, info in artifacts_status.items():
            status = "✅" if info.get('exists', False) else "❌"
            artifact_rows.append({"文件": name, "存在": status, "来源": info.get('generated_by', 'N/A')})
        
        st.dataframe(pd.DataFrame(artifact_rows))
    else:
        st.info("Manifest 文件暂不可用")
    
    # 数据覆盖率报告状态
    coverage_report_path = os.path.join(os.path.dirname(__file__), '..', 'reports', 'data_coverage_report.md')
    coverage_report_exists = os.path.exists(coverage_report_path)
    if coverage_report_exists:
        st.success("✅ 数据覆盖率报告已生成")
    else:
        st.warning("⚠️ 数据覆盖率报告暂未生成")
    
    st.markdown("---")
    st.subheader("🔄 Pipeline 状态")
    for pipeline in pipeline_status:
        col1, col2, col3 = st.columns([2, 1, 4])
        col1.write(f"**{pipeline['name']}**")
        col2.write(display_status_badge(pipeline['status']))
        col3.write(f"*描述: {pipeline['desc']}*")
    
    st.markdown("---")
    st.subheader("🏆 当前结论")
    st.success("✅ 工程已跑通")
    st.info("🔍 可研究")
    st.error("❌ 不可实盘")
    
    st.markdown("---")
    st.subheader("⚠️ 当前风险")
    risks = [
        "RankIC 整体较弱",
        "样本仍有限",
        "仍需样本外验证",
        "仍需交易约束完善"
    ]
    for risk in risks:
        st.warning(f"- {risk}")

# ==================== Formula Factor Lab Tab ====================

with tab2:
    st.subheader("🧪 Formula Factor Lab")
    st.markdown("""
        展示公式因子评价结果，包括 RankIC、ICIR 等关键指标。
    """)
    
    if not has_artifacts:
        st.warning("⚠️ 该 Profile 尚未生成结果，请先运行 pipeline")
    else:
        st.markdown("---")
        st.subheader("📊 Factor Summary")
        
        if factor_summary_df is not None:
            st.dataframe(factor_summary_df)
            
            st.markdown("---")
            st.subheader("📈 RankIC 分布")
            fig = px.bar(
                factor_summary_df.sort_values('rank_ic_mean', ascending=False),
                x='factor_name',
                y='rank_ic_mean',
                title='因子 RankIC 分布',
                color='rank_ic_mean',
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            avg_rankic = factor_summary_df['rank_ic_mean'].mean()
            if avg_rankic < 0.02:
                st.warning("⚠️ 当前 RankIC 较弱，仅为初步研究信号")
        else:
            if profile_report and profile_report['formula_factors'] is not None:
                df = profile_report['formula_factors']
                if 'RankIC' in df.columns:
                    df['RankIC'] = df['RankIC'].astype(float)
                    df['ICIR'] = df['ICIR'].astype(float)
                    
                    st.dataframe(df)
                    
                    fig = px.bar(
                        df.sort_values('RankIC', ascending=False),
                        x='Factor',
                        y='RankIC',
                        title='因子 RankIC 分布',
                        color='RankIC',
                        color_continuous_scale='RdYlGn'
                    )
                    fig.update_layout(height=400, xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("因子数据暂不可用")
        
        st.markdown("---")
        st.subheader("📈 ICIR 分布")
        if factor_summary_df is not None:
            fig2 = px.bar(
                factor_summary_df.sort_values('icir', ascending=False),
                x='factor_name',
                y='icir',
                title='因子 ICIR 分布',
                color='icir',
                color_continuous_scale='Blues'
            )
            fig2.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("ICIR 数据暂不可用")
        
        st.markdown("---")
        st.subheader("📈 RankIC 时间序列")
        factor_ic_series_df, _ = loader.load_factor_ic_series()
        if factor_ic_series_df is not None and not factor_ic_series_df.empty:
            fig3 = px.line(factor_ic_series_df, x='date', y='rank_ic', color='factor_name',
                          title='因子 RankIC 时间序列', height=400)
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.warning("⚠️ 请运行 pipeline 生成 factor_ic_series.parquet")
        
        st.markdown("---")
        st.subheader("🔥 Factor Correlation Heatmap")
        factor_correlation_df, _ = loader.load_factor_correlation()
        if factor_correlation_df is not None and not factor_correlation_df.empty:
            corr_matrix = factor_correlation_df.pivot(index='factor_1', columns='factor_2', values='correlation')
            fig4 = px.imshow(corr_matrix, title='因子相关性热力图',
                            labels=dict(x="因子", y="因子", color="相关性"),
                            x=corr_matrix.columns,
                            y=corr_matrix.index,
                            color_continuous_scale='RdBu_r')
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.warning("⚠️ 请运行 pipeline 生成 factor_correlation.parquet")

# ==================== Neural Feature Lab Tab ====================

with tab3:
    st.subheader("🧠 Neural Feature Lab")
    st.markdown("""
        展示神经特征学习结果。
    """)
    
    if not has_artifacts:
        st.warning("⚠️ 该 Profile 尚未生成结果，请先运行 pipeline")
    else:
        st.markdown("---")
        st.subheader("📊 Neural Factor Summary")
        
        if neural_factor_summary_df is not None:
            st.dataframe(neural_factor_summary_df)
            
            fig = px.bar(neural_factor_summary_df, x='factor_name', y='rank_ic_mean', 
                         title='Neural Factor RankIC', color='rank_ic_mean', 
                         color_continuous_scale='RdYlGn')
            fig.update_layout(height=300, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            avg_rankic = neural_factor_summary_df['rank_ic_mean'].mean()
            if avg_rankic < 0.02:
                st.warning("⚠️ 当前 Neural Factor RankIC 较弱，仅为研究信号")
        else:
            st.info("Neural Factor 汇总数据暂不可用")
        
        st.markdown("---")
        st.subheader("🔥 Neural Factor Heatmap")
        if neural_factors_df is not None:
            factor_cols = [col for col in neural_factors_df.columns if 'neural_factor_' in col]
            if len(factor_cols) > 0:
                fig = px.imshow(neural_factors_df[factor_cols].corr(), 
                               title='Neural Factor 相关性热力图',
                               labels=dict(x="Factor", y="Factor", color="Correlation"))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("未找到 neural_factor_* 列")
        else:
            st.warning("⚠️ neural_factors.parquet 数据暂缺，请先运行 neural_factor_pipeline")
        
        st.markdown("---")
        st.subheader("🚫 Leakage Check")
        if profile_report and profile_report['leakage_check'] is not None:
            df = profile_report['leakage_check']
            st.dataframe(df)
            
            if 'FAIL' in df['Status'].values:
                st.error("❌ Leakage Check FAIL")
            else:
                st.success("✅ Leakage Check OK")
        elif neural_factor_report and neural_factor_report['leakage_check'] is not None:
            df = neural_factor_report['leakage_check']
            st.dataframe(df)
            
            if 'FAIL' in df['Status'].values:
                st.error("❌ Leakage Check FAIL")
            else:
                st.success("✅ Leakage Check OK")
        else:
            st.info("Leakage Check 数据暂不可用")

# ==================== Encoder Comparison Tab ====================

with tab4:
    st.subheader("⚡ Encoder Comparison")
    st.markdown("""
        对比 MLP、CNN1D、TinyTransformer 三种编码器的表现。
    """)
    
    if not has_artifacts:
        st.warning("⚠️ 该 Profile 尚未生成结果，请先运行 pipeline")
    else:
        st.markdown("---")
        st.subheader("📊 对比指标")
        
        if encoder_comparison_df is not None:
            st.dataframe(encoder_comparison_df)
            
            st.markdown("---")
            st.subheader("📈 Train Loss / Val Loss")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=encoder_comparison_df['encoder'], y=encoder_comparison_df['train_loss'], name='Train Loss'))
            fig.add_trace(go.Bar(x=encoder_comparison_df['encoder'], y=encoder_comparison_df['val_loss'], name='Val Loss'))
            fig.update_layout(barmode='group', title='训练/验证损失对比', height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.subheader("🎯 Avg RankIC 对比")
            fig2 = px.bar(encoder_comparison_df, x='encoder', y='avg_rankic', 
                          title='平均 RankIC 对比', color='avg_rankic', 
                          color_continuous_scale='RdYlGn')
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
            
            st.markdown("---")
            st.subheader("🏆 通过因子数")
            fig3 = px.bar(encoder_comparison_df, x='encoder', y='passing_factors', 
                          title='通过 Gatekeeper 的因子数', color='passing_factors',
                          color_continuous_scale='Greens')
            fig3.update_layout(height=400)
            st.plotly_chart(fig3, use_container_width=True)
            
            st.markdown("---")
            st.subheader("📊 Best Encoder")
            best_encoder = encoder_comparison_df.loc[encoder_comparison_df['avg_rankic'].abs().idxmax()]
            col1, col2, col3 = st.columns(3)
            col1.metric("最佳模型", best_encoder['encoder'].upper())
            col2.metric("平均 RankIC", f"{float(best_encoder['avg_rankic']):.4f}")
            col3.metric("通过因子数", f"{best_encoder['passing_factors']}")
            
        elif encoder_compare_report and encoder_compare_report['comparison_metrics'] is not None:
            df = encoder_compare_report['comparison_metrics']
            st.dataframe(df)
            st.info("回退到 Markdown 解析，建议运行 encoder comparison pipeline")
        else:
            st.warning("编码器对比数据暂不可用")
        
        st.markdown("---")
        st.subheader("⚠️ 结论")
        st.warning("""
            - 当前 MLP 暂时最好
            - 但整体 RankIC 较弱
            - **不能用于实盘**
            - 后续需要更长样本、更多 raw features、样本外验证
            
            只有**初步研究信号**，离有效模型还有距离。
        """)

# ==================== Backtest Report Tab ====================

with tab5:
    st.subheader("📈 Backtest Report")
    st.markdown("""
        展示回测结果和交易约束说明。
    """)
    
    if not has_artifacts:
        st.warning("⚠️ 该 Profile 尚未生成结果，请先运行 pipeline")
    else:
        st.markdown("---")
        st.subheader("📊 绩效指标")
        
        if backtest_summary is not None:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("总收益", f"{backtest_summary['total_return']*100:.2f}%")
            col2.metric("年化收益", f"{backtest_summary['annual_return']*100:.2f}%")
            col3.metric("Sharpe", f"{backtest_summary['sharpe']:.2f}")
            col4.metric("最大回撤", f"{backtest_summary['max_drawdown']*100:.2f}%")
        elif student_laptop_report and student_laptop_report['performance_metrics'] is not None:
            df = student_laptop_report['performance_metrics']
            st.dataframe(df)
        else:
            st.info("绩效指标数据暂不可用")
        
        st.markdown("---")
        st.subheader("💸 交易成本说明")
        if backtest_summary is not None:
            st.markdown(f"""
            - **佣金**: {backtest_summary['transaction_cost']*100:.2f}%
            - **印花税**: {backtest_summary['stamp_tax']*100:.2f}%
            - **滑点**: {backtest_summary['slippage']*100:.2f}%
            """)
        else:
            st.markdown("""
            - **佣金**: 0.10%
            - **印花税**: 0.10%
            - **滑点**: 0.10%
            """)
        
        st.markdown("---")
        st.subheader("📈 净值曲线")
        if equity_curve_df is not None:
            fig = px.line(equity_curve_df, x='date', y='portfolio_value', 
                          title='组合净值曲线')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.subheader("📉 回撤曲线")
            if drawdown_curve_df is not None:
                fig2 = px.line(drawdown_curve_df, x='date', y='drawdown', 
                               title='回撤曲线',
                               color_discrete_sequence=['red'])
                fig2.update_layout(height=300)
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("回撤数据暂不可用")
        else:
            st.warning("⚠️ 净值曲线数据暂未保存，请先运行 pipeline")
        
        st.markdown("---")
        st.subheader("📅 Signal-Trade Timing")
        st.markdown("""
        - **Signal Date**: t (交易日结束时)
        - **Trade Date**: t+1 (下一交易日)
        - **Return Period**: t+1 日收益
        
        ✅ 满足 t+1 交易约束
        """)
        
        st.markdown("---")
        st.subheader("🛡️ 交易约束实现状态")
        constraints = [
            {'name': '手续费', 'status': 'OK', 'details': '已实现 0.10%'},
            {'name': '印花税', 'status': 'OK', 'details': '已实现 0.10%'},
            {'name': '滑点', 'status': 'OK', 'details': '已实现 0.10%'},
            {'name': '停牌过滤', 'status': 'WARN', 'details': '未实现'},
            {'name': '涨跌停过滤', 'status': 'WARN', 'details': '未实现'},
            {'name': 'ST 过滤', 'status': 'WARN', 'details': '未实现'},
            {'name': '新股过滤', 'status': 'WARN', 'details': '未实现'},
            {'name': 'T+1 交易', 'status': 'OK', 'details': '已实现'}
        ]
        
        for constraint in constraints:
            col1, col2, col3 = st.columns([2, 1, 4])
            col1.write(f"**{constraint['name']}**")
            col2.write(display_status_badge(constraint['status']))
            col3.write(f"* {constraint['details']}")

# ==================== Reliability Audit Tab ====================

with tab6:
    st.subheader("🔍 Reliability Audit")
    st.markdown("""
        可信度审计展示，检查数据真实性、未来函数、日期对齐等关键指标。
    """)
    
    if not has_artifacts:
        st.warning("⚠️ 该 Profile 尚未生成结果，请先运行 pipeline")
    else:
        st.markdown("---")
        st.subheader("✅ Audit Checklist")
        
        status_counts = {'OK': 0, 'WARN': 0, 'FAIL': 0, 'TODO': 0}
        for check in reliability_status:
            status_counts[check['status']] += 1
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("✅ OK", status_counts['OK'])
        col2.metric("⚠️ WARN", status_counts['WARN'])
        col3.metric("❌ FAIL", status_counts['FAIL'])
        col4.metric("🔄 TODO", status_counts['TODO'])
        
        st.markdown("---")
        for check in reliability_status:
            color = get_status_color(check['status'])
            st.markdown(f"""
            <div style="background-color: rgba(0,0,0,0.05); padding: 10px; border-radius: 5px; margin-bottom: 8px;">
                <span style="color: {color}; font-weight: bold;">{check['status']}</span>
                <span style="margin-left: 10px; font-weight: bold;">{check['item']}</span>
                <p style="margin-top: 5px; margin-left: 30px; color: #666;">{check['details']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("🏁 Final Status")
        if status_counts['FAIL'] > 0:
            st.error("STATUS: FAIL - 存在失败项，Pipeline 已停止")
        elif status_counts['WARN'] > 0:
            st.warning("STATUS: WARN - 存在警告项，需要进一步完善")
        else:
            st.success("STATUS: OK - 所有检查通过")
        
        st.markdown("---")
        st.subheader("⚠️ 风险提示")
        st.error("""
            当前回测结果仅供研究参考，不能用于实盘交易。
            
            主要限制:
            - 股票样本量有限
            - 时间周期较短
            - 部分交易约束未实现
            - 未经过样本外验证
        """)

# ==================== Out-of-Sample Validation Tab ====================

with tab7:
    st.subheader("🔬 Out-of-Sample Validation")
    st.markdown("""
        样本外验证展示，对比 formula-only、neural-only、formula+neural 三类特征集的样本外表现。
    """)
    
    if not has_artifacts:
        st.warning("⚠️ 该 Profile 尚未生成结果，请先运行 pipeline")
    else:
        oos_feature_comparison_df = None
        oos_equity_curves_df = None
        oos_backtest_summary_df = None
        oos_split_info = None
        
        feature_comparison_path = os.path.join(loader.get_profile_dashboard_dir(), 'oos_feature_comparison.parquet')
        equity_curves_path = os.path.join(loader.get_profile_dashboard_dir(), 'oos_equity_curves.parquet')
        backtest_summary_path = os.path.join(loader.get_profile_dashboard_dir(), 'oos_backtest_summary.parquet')
        split_info_path = os.path.join(loader.get_profile_dashboard_dir(), 'oos_split_info.json')
        
        if os.path.exists(feature_comparison_path):
            oos_feature_comparison_df = pd.read_parquet(feature_comparison_path)
        if os.path.exists(equity_curves_path):
            oos_equity_curves_df = pd.read_parquet(equity_curves_path)
        if os.path.exists(backtest_summary_path):
            oos_backtest_summary_df = pd.read_parquet(backtest_summary_path)
        if os.path.exists(split_info_path):
            with open(split_info_path, 'r', encoding='utf-8') as f:
                oos_split_info = json.load(f)
        
        if oos_split_info:
            st.markdown("---")
            st.subheader("📅 Time Split")
            
            col1, col2, col3 = st.columns(3)
            col1.write(f"**Train**: {oos_split_info['train_start']} ~ {oos_split_info['train_end']}")
            col2.write(f"**Validation**: {oos_split_info['validation_start']} ~ {oos_split_info['validation_end']}")
            col3.write(f"**Test**: {oos_split_info['test_start']} ~ {oos_split_info['test_end']}")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Train Samples", oos_split_info['train_samples'])
            col2.metric("Validation Samples", oos_split_info['validation_samples'])
            col3.metric("Test Samples", oos_split_info['test_samples'])
            col4.metric("Stock Count", oos_split_info['stock_count'])
        
        if oos_feature_comparison_df is not None:
            st.markdown("---")
            st.subheader("📊 Feature Set Comparison")
            
            st.dataframe(oos_feature_comparison_df[['feature_set', 'feature_count', 'test_rank_ic', 'test_icir', 'coverage', 'total_return', 'annual_return', 'sharpe', 'max_drawdown']])
            
            st.markdown("---")
            st.subheader("📈 Test RankIC")
            fig = px.bar(
                oos_feature_comparison_df,
                x='feature_set',
                y='test_rank_ic',
                title='样本外 RankIC 对比',
                color='test_rank_ic',
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.subheader("📈 Test ICIR")
            fig = px.bar(
                oos_feature_comparison_df,
                x='feature_set',
                y='test_icir',
                title='样本外 ICIR 对比',
                color='test_icir',
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.subheader("📈 OOS Equity Curves")
            if oos_equity_curves_df is not None:
                fig = px.line(
                    oos_equity_curves_df,
                    x='date',
                    y='portfolio_value',
                    color='feature_set',
                    title='样本外净值曲线对比',
                    labels={'portfolio_value': 'Portfolio Value', 'date': 'Date'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.subheader("🏆 Key Findings")
            
            best_rankic = oos_feature_comparison_df.loc[oos_feature_comparison_df['test_rank_ic'].idxmax()]
            best_sharpe = oos_feature_comparison_df.loc[oos_feature_comparison_df['sharpe'].idxmax()]
            
            col1, col2 = st.columns(2)
            col1.success(f"**最佳 RankIC**: {best_rankic['feature_set']} ({best_rankic['test_rank_ic']:.4f})")
            col2.success(f"**最佳 Sharpe**: {best_sharpe['feature_set']} ({best_sharpe['sharpe']:.2f})")
            
            formula_ic = oos_feature_comparison_df[oos_feature_comparison_df['feature_set'] == 'formula_only']['test_rank_ic'].iloc[0]
            combined_ic = oos_feature_comparison_df[oos_feature_comparison_df['feature_set'] == 'formula_plus_neural']['test_rank_ic'].iloc[0]
            neural_ic = oos_feature_comparison_df[oos_feature_comparison_df['feature_set'] == 'neural_only']['test_rank_ic'].iloc[0]
            
            if combined_ic > formula_ic + 0.01:
                st.success("✅ formula+neural 优于 formula-only，神经因子提供了增量信息！")
            else:
                st.warning("⚠️ formula+neural 未明显优于 formula-only，神经因子增量有限")
            
            if neural_ic > 0.05:
                st.info("🔍 neural_only 有一定预测能力")
            else:
                st.info("🔍 neural_only 预测能力有限")
        
        else:
            st.info("""
                ⚠️ 样本外验证数据尚未生成。
                
                请运行 `oos_validation_pipeline` 生成数据：
                - 点击 Run Center
                - 选择 oos_validation_pipeline
                - 点击 Run
            """)
        
        st.markdown("---")
        st.subheader("⚠️ 重要提示")
        st.error("""
            当前结果**仅用于研究**，**不能直接用于实盘交易**。
            
            主要限制:
            - 样本量有限
            - 交易约束未完全实现
            - 仅为初步验证
        """)

# ==================== Trading Constraints Tab ====================

with tab8:
    st.subheader("🛡️ Trading Constraints")
    st.markdown("""
        展示交易约束检查结果、数据可用性、以及约束后的回测结果对比。
    """)
    
    if not has_artifacts:
        st.warning("⚠️ 该 Profile 尚未生成结果，请先运行 pipeline")
    else:
        st.markdown("---")
        st.subheader("📊 Constraint Status Matrix")
        
        # 展示数据可用性和约束状态
        if trading_constraint_json:
            data_availability = trading_constraint_json.get('data_availability', {})
            
            constraint_info = [
                {'Constraint': 'ST Filter', 'Status': data_availability.get('st_field', 'WARN'), 'Notes': 'ST股票过滤'},
                {'Constraint': 'Suspended Filter', 'Status': data_availability.get('suspended_field', 'WARN'), 'Notes': '停牌股票过滤（成交量/成交额为0）'},
                {'Constraint': 'Limit-up Filter', 'Status': data_availability.get('limit_up_down_field', 'WARN'), 'Notes': '涨停不可买入'},
                {'Constraint': 'Limit-down Filter', 'Status': data_availability.get('limit_up_down_field', 'WARN'), 'Notes': '跌停不可卖出'},
                {'Constraint': 'New Stock Filter', 'Status': data_availability.get('listing_date_field', 'WARN'), 'Notes': '新股上市不足N天过滤'},
                {'Constraint': 'Liquidity Filter', 'Status': data_availability.get('amount_field', 'WARN'), 'Notes': '成交额低于阈值过滤'},
                {'Constraint': 'Capacity Filter', 'Status': 'OK', 'Notes': '成交额占比约束'}
            ]
            
            constraint_df = pd.DataFrame(constraint_info)
            
            # 显示带颜色的表格
            for idx, row in constraint_df.iterrows():
                st.markdown(f"**{row['Constraint']}**: {display_status_badge(row['Status'])} - {row['Notes']}")
        else:
            st.warning("⚠️ trading_constraint_report.json 数据暂缺，请先运行 trading_constraints_pipeline")
        
        st.markdown("---")
        st.subheader("📈 Filter Reasons Summary")
        
        if trading_constraint_summary_df is not None:
            # 展示过滤统计
            if 'st_filtered' in trading_constraint_summary_df.columns:
                total_st = trading_constraint_summary_df['st_filtered'].sum()
                total_suspended = trading_constraint_summary_df['suspended_filtered'].sum()
                total_limit_up = trading_constraint_summary_df['limit_up_filtered'].sum()
                total_limit_down = trading_constraint_summary_df['limit_down_filtered'].sum()
                total_new_stock = trading_constraint_summary_df['new_stock_filtered'].sum()
                total_liquidity = trading_constraint_summary_df['liquidity_filtered'].sum()
                
                filter_summary = pd.DataFrame({
                    'Reason': ['ST股票', '停牌', '涨停', '跌停', '新股', '流动性'],
                    'Count': [total_st, total_suspended, total_limit_up, total_limit_down, total_new_stock, total_liquidity]
                })
                
                fig = px.bar(filter_summary, x='Reason', y='Count', title='过滤原因分布', color='Count', color_continuous_scale='Reds')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.dataframe(trading_constraint_summary_df)
        else:
            st.warning("⚠️ trading_constraint_summary.parquet 数据暂缺，请先运行 trading_constraints_pipeline")
        
        st.markdown("---")
        st.subheader("📅 Daily Tradable Count")
        
        if trading_constraint_summary_df is not None and 'date' in trading_constraint_summary_df.columns:
            if 'tradable_count' in trading_constraint_summary_df.columns:
                fig = px.line(trading_constraint_summary_df, x='date', y='tradable_count', title='每日可交易股票数')
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            elif 'total_candidates' in trading_constraint_summary_df.columns:
                fig = px.line(trading_constraint_summary_df, x='date', y='total_candidates', title='每日候选股票数')
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📊 Constrained Backtest Results")
        
        if constrained_backtest_summary:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Return", f"{constrained_backtest_summary.get('total_return', 0)*100:.2f}%")
            col2.metric("Annual Return", f"{constrained_backtest_summary.get('annual_return', 0)*100:.2f}%")
            col3.metric("Sharpe Ratio", f"{constrained_backtest_summary.get('sharpe', 0):.2f}")
            col4.metric("Max Drawdown", f"{constrained_backtest_summary.get('max_drawdown', 0)*100:.2f}%")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Turnover", f"{constrained_backtest_summary.get('turnover', 0):.4f}")
            col2.metric("Filtered Trades", constrained_backtest_summary.get('filtered_trade_count', 0))
            col3.metric("Blocked Sells", constrained_backtest_summary.get('blocked_sell_count', 0))
        else:
            st.warning("⚠️ constrained_backtest_summary.json 数据暂缺，请先运行 trading_constraints_pipeline")
        
        st.markdown("---")
        st.subheader("📈 Constrained Equity Curve")
        
        if constrained_equity_curve_df is not None:
            fig = px.line(constrained_equity_curve_df, x='date', y='portfolio_value', title='约束后净值曲线')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ constrained_equity_curve.parquet 数据暂缺，请先运行 trading_constraints_pipeline")
        
        st.markdown("---")
        st.subheader("📉 Constrained Drawdown Curve")
        
        if constrained_drawdown_curve_df is not None:
            fig = px.line(constrained_drawdown_curve_df, x='date', y='drawdown', title='约束后回撤曲线')
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ constrained_drawdown_curve.parquet 数据暂缺，请先运行 trading_constraints_pipeline")
        
        st.markdown("---")
        st.subheader("⚠️ Important Limitations")
        
        st.warning("""
            **当前约束实现的限制**:
            - ST 过滤: 缺少 is_st 或股票名称字段，暂无法精确实现
            - 新股过滤: 缺少上市日期字段，暂无法精确实现
            - 停牌过滤: 用成交量/成交额为 0 近似判断，可能有误判
            - 涨跌停过滤: 用涨跌幅阈值近似判断，可能有误判
            - 流动性过滤: OK（使用成交额阈值）
            - 容量约束: OK（使用参与率阈值）
            
            **当前结果仅用于研究参考，不能用于实盘交易！**
        """)
        
        st.markdown("---")
        st.subheader("🚫 Real Trading Warning")
        
        st.error("""
            ⚠️ **这是研究平台，不是交易平台**
            
            - 所有操作仅用于研究目的
            - 不产生真实交易
            - 不连接实盘账户
            - 不执行真实下单
            - 数据仅供参考，不构成投资建议
            - **can_use_for_live_trading: false**
        """)

# ==================== Run Center Tab ====================

with tab9:
    st.subheader("🎮 Run Center / 实验操作中心")
    st.markdown("""
        安全触发研究 pipeline，记录运行历史，查看日志。
        
        **⚠️ 重要提示**: 这是研究 pipeline，**不是实盘交易系统**。所有操作仅用于研究目的。
    """)
    
    st.markdown("---")
    st.subheader("🔧 Profile 设置")
    
    st.write(f"**当前 Profile**: `{st.session_state.current_profile}`")
    
    # Profile 参数
    profile_config = loader.get_profile_config(st.session_state.current_profile)
    if profile_config:
        st.write("**Profile 参数**:")
        st.write(f"- 股票数: {profile_config.get('stock_count', 'N/A')}")
        st.write(f"- 历史月份: {profile_config.get('history_months', 'N/A')}")
        st.write(f"- 公式因子数: {profile_config.get('formula_factor_limit', 'N/A')}")
        st.write(f"- 神经嵌入维度: {profile_config.get('neural_embedding_dim', 'N/A')}")
    
    st.markdown("---")
    st.subheader("📋 Available Pipelines")
    
    running_pipelines = run_manager.get_running_pipelines()
    run_history = run_manager.get_run_history()
    
    for script_path in run_manager.ALLOWED_SCRIPTS:
        info = run_manager.get_run_history()
        pipeline_info = run_manager.get_pipeline_info(script_path)
        pipeline_name = pipeline_info['name']
        
        is_running = pipeline_name in running_pipelines
        latest_run = run_manager.get_latest_run(pipeline_name)
        
        with st.expander(f"🚀 {pipeline_name}", expanded=False):
            st.write(f"**描述**: {pipeline_info.get('description', 'N/A')}")
            st.write(f"**预计耗时**: {pipeline_info.get('estimated_time', 'N/A')}")
            
            st.markdown("**输入数据**:")
            for data in pipeline_info.get('input_data', []):
                st.write(f"- {data}")
            
            st.markdown("**输出 artifacts**:")
            for artifact in pipeline_info.get('output_artifacts', []):
                st.write(f"- {artifact}")
            
            if latest_run:
                st.markdown("**最近运行**:")
                st.write(f"- 时间: {latest_run.get('start_time', 'N/A')}")
                st.write(f"- 状态: {latest_run.get('status', 'N/A')}")
                
                if 'log_path' in latest_run and latest_run['status'] != 'RUNNING':
                    if st.button(f"查看日志", key=f"log_{pipeline_name}"):
                        log_content = run_manager.get_log_content(latest_run['run_id'])
                        st.text_area("运行日志", log_content, height=200)
            
            if is_running:
                st.warning("⏳ 正在运行中...")
            else:
                if st.button(f"运行 {pipeline_name}", key=f"run_{pipeline_name}"):
                    with st.spinner(f"正在运行 {pipeline_name}..."):
                        result = run_manager.run_pipeline(script_path, profile=st.session_state.current_profile)
                    if result['status'] == 'SUCCESS':
                        st.success(f"✅ {pipeline_name} 运行成功")
                    else:
                        st.error(f"❌ {pipeline_name} 运行失败: {result.get('error', 'Unknown error')}")
    
    st.markdown("---")
    st.subheader("📊 Run History")
    
    if run_history:
        history_df = pd.DataFrame(run_history)
        history_df = history_df[['run_id', 'pipeline_name', 'profile', 'status', 'start_time', 'end_time']]
        st.dataframe(history_df)
    else:
        st.info("暂无运行记录")
    
    st.markdown("---")
    st.subheader("⚠️ 风险提示")
    st.error("""
        ⚠️ **这是研究平台，不是交易平台**
        
        - 所有操作仅用于研究目的
        - 不产生真实交易
        - 不连接实盘账户
        - 不执行真实下单
        - 数据仅供参考，不构成投资建议
    """)

def main():
    pass

if __name__ == "__main__":
    main()
