import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class PlotHelper:
    @staticmethod
    def plot_equity_curve(portfolio: pd.DataFrame, title: str = "Equity Curve"):
        """绘制权益曲线"""
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio['total'], label='Equity', color='blue')
        plt.plot(portfolio['cash'], label='Cash', color='green')
        plt.plot(portfolio['holdings'], label='Holdings', color='red')
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def plot_drawdown(portfolio: pd.DataFrame, title: str = "Drawdown"):
        """绘制回撤曲线"""
        portfolio_value = portfolio['total']
        peak = portfolio_value.cummax()
        drawdown = (portfolio_value - peak) / peak
        
        plt.figure(figsize=(12, 6))
        plt.fill_between(drawdown.index, drawdown, 0, where=drawdown < 0, color='red', alpha=0.3)
        plt.plot(drawdown, color='red')
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def plot_signals(data: pd.DataFrame, signals: pd.DataFrame, title: str = "Price and Signals"):
        """绘制价格和信号"""
        plt.figure(figsize=(12, 6))
        plt.plot(data['close'], label='Price', color='blue')
        
        buy_signals = signals[signals['signal'] == 1]
        sell_signals = signals[signals['signal'] == -1]
        
        plt.scatter(buy_signals.index, data.loc[buy_signals.index, 'close'], 
                   marker='^', color='green', label='Buy Signal', s=100)
        plt.scatter(sell_signals.index, data.loc[sell_signals.index, 'close'], 
                   marker='v', color='red', label='Sell Signal', s=100)
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def plot_returns_distribution(returns: pd.Series, title: str = "Returns Distribution"):
        """绘制收益分布"""
        plt.figure(figsize=(12, 6))
        sns.histplot(returns, kde=True, bins=50)
        
        plt.title(title)
        plt.xlabel('Daily Returns')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def plot_multiple_equity_curves(portfolios: Dict[str, pd.DataFrame], title: str = "Strategy Comparison"):
        """绘制多个权益曲线对比"""
        plt.figure(figsize=(12, 6))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i, (name, portfolio) in enumerate(portfolios.items()):
            plt.plot(portfolio['total'], label=name, color=colors[i % len(colors)])
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def plot_interactive_equity(portfolio: pd.DataFrame):
        """绘制交互式权益曲线"""
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           subplot_titles=('Equity Curve', 'Drawdown'),
                           vertical_spacing=0.1)
        
        fig.add_trace(go.Scatter(x=portfolio.index, y=portfolio['total'], 
                               name='Total Equity', line=dict(color='blue')), row=1, col=1)
        
        portfolio_value = portfolio['total']
        peak = portfolio_value.cummax()
        drawdown = (portfolio_value - peak) / peak
        
        fig.add_trace(go.Scatter(x=portfolio.index, y=drawdown,
                               name='Drawdown', line=dict(color='red')), row=2, col=1)
        
        fig.update_layout(height=600, title_text='Backtest Results')
        fig.show()
    
    @staticmethod
    def plot_correlation_matrix(data: pd.DataFrame, title: str = "Correlation Matrix"):
        """绘制相关矩阵"""
        corr = data.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
        
        plt.title(title)
        plt.show()
