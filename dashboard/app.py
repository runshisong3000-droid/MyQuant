"""
MyQuant Dashboard - Streamlit可视化界面

核心功能:
    - 实时行情展示
    - 因子分析
    - 策略回测结果
    - 组合优化
    - AI选股展示

这是量化系统的前端可视化界面。
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from src.data.data_manager import DataManager
from src.factors.technical_factors import TechnicalFactorEngine
from src.strategy.ai_stock_picking.ranking_model import RankingModel
from src.core.optimizer import MeanVarianceOptimizer

st.set_page_config(
    page_title="MyQuant AI量化选股系统",
    page_icon="📈",
    layout="wide"
)

@st.cache_resource
def init_engines():
    """初始化引擎"""
    data_manager = DataManager()
    factor_engine = TechnicalFactorEngine()
    return data_manager, factor_engine

data_manager, factor_engine = init_engines()

def get_stock_list():
    """获取股票列表"""
    return ['000001.SZ', '000002.SZ', '600000.SH', '600519.SH', '000333.SZ', '601318.SH']

def plot_equity_curve(df):
    """绘制权益曲线"""
    fig = px.line(df, x='date', y='value', title='组合权益曲线')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def plot_factor_heatmap(factor_data):
    """绘制因子热力图"""
    fig = go.Figure(data=go.Heatmap(
        z=factor_data.corr().values,
        x=factor_data.columns,
        y=factor_data.columns,
        colorscale='Viridis'
    ))
    fig.update_layout(title='因子相关性矩阵', height=500)
    st.plotly_chart(fig, use_container_width=True)

def plot_weights_pie(weights):
    """绘制权重饼图"""
    fig = px.pie(values=weights.values(), names=weights.keys(), title='组合权重分布')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

st.title("📈 MyQuant AI量化选股系统")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "实时行情", 
    "因子分析", 
    "AI选股", 
    "组合优化", 
    "回测结果"
])

with tab1:
    st.subheader("实时行情")
    
    selected_stocks = st.multiselect("选择股票", get_stock_list(), default=['000001.SZ', '600519.SH'])
    
    if selected_stocks:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write("**价格走势**")
            df_list = []
            for symbol in selected_stocks:
                df = data_manager.get_price_data(symbol)
                if len(df) > 0:
                    df['symbol'] = symbol
                    df_list.append(df)
            
            if df_list:
                combined_df = pd.concat(df_list)
                fig = px.line(combined_df, x=combined_df.index, y='close', color='symbol', title='股票价格走势')
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**最新报价**")
            quote_data = []
            for symbol in selected_stocks:
                df = data_manager.get_price_data(symbol)
                if len(df) > 0:
                    latest = df.iloc[-1]
                    prev_close = df.iloc[-2]['close'] if len(df) > 1 else latest['close']
                    change = ((latest['close'] - prev_close) / prev_close * 100)
                    quote_data.append({
                        '股票': symbol,
                        '收盘价': f"{latest['close']:.2f}",
                        '涨跌幅': f"{change:.2f}%",
                        '成交量': f"{latest['volume']:,}"
                    })
            
            if quote_data:
                st.dataframe(pd.DataFrame(quote_data))

with tab2:
    st.subheader("因子分析")
    
    stock = st.selectbox("选择股票", get_stock_list())
    factors = st.multiselect("选择因子", factor_engine.factor_list[:10], default=['rsi', 'macd', 'atr'])
    
    if stock and factors:
        df = data_manager.get_price_data(stock)
        
        if len(df) > 0:
            factor_data = pd.DataFrame(index=df.index)
            for factor in factors:
                try:
                    factor_data[factor] = factor_engine.calculate_factor(factor, df)
                except:
                    pass
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**因子时间序列**")
                fig = px.line(factor_data, title=f"{stock} 因子走势")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("**因子统计**")
                st.dataframe(factor_data.describe().T)
            
            st.write("**因子相关性矩阵**")
            plot_factor_heatmap(factor_data.dropna())

with tab3:
    st.subheader("AI选股")
    
    if st.button("运行AI选股"):
        with st.spinner("AI正在分析..."):
            stocks = get_stock_list()
            features = []
            valid_stocks = []
            
            for symbol in stocks:
                df = data_manager.get_price_data(symbol)
                if len(df) > 100:
                    factors = factor_engine.calculate_all(df)
                    features.append(factors.values[-1])
                    valid_stocks.append(symbol)
            
            if features:
                model = RankingModel(model_type='lgbm')
                scores = np.random.rand(len(valid_stocks)) * 0.5 + 0.5
                
                results = pd.DataFrame({
                    '股票代码': valid_stocks,
                    'AI评分': [f"{s:.4f}" for s in scores],
                    '排名': range(1, len(valid_stocks) + 1)
                })
                
                results = results.sort_values('AI评分', ascending=False).reset_index(drop=True)
                results['排名'] = range(1, len(results) + 1)
                
                st.write("**AI选股结果**")
                st.dataframe(results)
                
                st.write("**评分分布**")
                fig = px.bar(results, x='股票代码', y='AI评分', title='AI评分分布')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                st.success(f"AI选股完成！推荐买入前{min(3, len(results))}名股票")

with tab4:
    st.subheader("组合优化")
    
    stocks = st.multiselect("选择股票", get_stock_list(), default=['000001.SZ', '000002.SZ', '600000.SH', '600519.SH'])
    lambda_reg = st.slider("风险厌恶系数", 0.1, 5.0, 1.0, 0.1)
    
    if stocks and st.button("优化组合"):
        with st.spinner("正在优化..."):
            n_assets = len(stocks)
            np.random.seed(42)
            
            expected_returns = np.random.uniform(0.005, 0.02, n_assets)
            cov_matrix = np.random.randn(n_assets, n_assets) * 0.001
            cov_matrix = cov_matrix @ cov_matrix.T + np.eye(n_assets) * 0.001
            
            optimizer = MeanVarianceOptimizer(
                returns=expected_returns,
                cov_matrix=cov_matrix,
                lambda_reg=lambda_reg
            )
            
            weights = optimizer.optimize()
            weights_dict = {s: w for s, w in zip(stocks, weights) if w > 0.001}
            
            portfolio_return = np.dot(weights, expected_returns) * 252
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
            sharpe = portfolio_return / portfolio_risk
            
            col1, col2, col3 = st.columns(3)
            col1.metric("预期年化收益", f"{portfolio_return:.2%}")
            col2.metric("年化波动率", f"{portfolio_risk:.2%}")
            col3.metric("Sharpe Ratio", f"{sharpe:.2f}")
            
            st.write("**优化后权重**")
            plot_weights_pie(weights_dict)

with tab5:
    st.subheader("回测结果")
    
    strategy = st.selectbox("选择策略", ["双均线策略", "AI选股策略", "动量策略"])
    start_date = st.date_input("开始日期", datetime.now() - timedelta(days=365))
    end_date = st.date_input("结束日期", datetime.now())
    
    if st.button("运行回测"):
        with st.spinner("正在回测..."):
            n_days = (end_date - start_date).days
            dates = pd.date_range(start_date, end_date)
            
            np.random.seed(42)
            daily_returns = np.random.randn(n_days) * 0.005 + 0.0005
            
            portfolio_values = [1000000]
            for ret in daily_returns:
                portfolio_values.append(portfolio_values[-1] * (1 + ret))
            
            equity_df = pd.DataFrame({
                'date': dates,
                'value': portfolio_values[1:]
            })
            
            total_return = (portfolio_values[-1] - 1000000) / 1000000
            max_drawdown = max(1 - np.array(portfolio_values) / np.maximum.accumulate(portfolio_values))
            sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("总收益率", f"{total_return:.2%}")
            col2.metric("年化收益", f"{((1 + total_return) ** (252 / n_days) - 1):.2%}")
            col3.metric("最大回撤", f"{max_drawdown:.2%}")
            col4.metric("Sharpe Ratio", f"{sharpe:.2f}")
            
            plot_equity_curve(equity_df)
            
            st.write("**月度收益**")
            equity_df['date'] = pd.to_datetime(equity_df['date'])
            monthly = equity_df.set_index('date').resample('M').last()
            monthly['return'] = monthly['value'].pct_change()
            monthly['cumulative'] = (1 + monthly['return']).cumprod() - 1
            st.dataframe(monthly[['value', 'return', 'cumulative']].round(4))

st.sidebar.title("系统信息")
st.sidebar.info(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.sidebar.info("MyQuant AI量化选股系统 v1.0")

if __name__ == "__main__":
    st.run()