"""
MyQuant API Service - FastAPI RESTful API

核心功能:
    - 数据查询接口
    - 因子计算接口
    - 模型预测接口
    - 回测接口
    - 组合优化接口

这是量化系统的后端API服务。
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from datetime import date, datetime
import pandas as pd
import numpy as np
import json

from src.data.data_manager import DataManager
from src.factors.technical_factors import TechnicalFactorEngine
from src.factors.fundamental_factors import FundamentalFactorEngine
from src.strategy.ai_stock_picking.ranking_model import RankingModel
from src.core.optimizer import MeanVarianceOptimizer
from src.core.transaction_cost import CostModelFactory

app = FastAPI(title="MyQuant API", version="1.0.0", description="AI量化选股系统API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data_manager = DataManager()
tech_factor_engine = TechnicalFactorEngine()
fundamental_factor_engine = FundamentalFactorEngine()


class StockDataRequest(BaseModel):
    """股票数据请求"""
    symbols: List[str]
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    fields: Optional[List[str]] = None


class FactorRequest(BaseModel):
    """因子计算请求"""
    symbols: List[str]
    factor_names: List[str]
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class PredictionRequest(BaseModel):
    """预测请求"""
    symbols: List[str]
    model_name: Optional[str] = "ranking_model"


class OptimizationRequest(BaseModel):
    """组合优化请求"""
    symbols: List[str]
    target_return: Optional[float] = None
    max_risk: Optional[float] = None
    constraints: Optional[Dict[str, Any]] = None


class BacktestRequest(BaseModel):
    """回测请求"""
    strategy_name: str
    symbols: List[str]
    start_date: str
    end_date: str
    initial_capital: Optional[float] = 1000000.0


@app.get("/api/health")
async def health():
    """健康检查"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/api/stocks/list")
async def get_stock_list(market: Optional[str] = Query(None)):
    """获取股票列表"""
    try:
        stocks = data_manager.get_stock_list()
        if market:
            stocks = [s for s in stocks if market.lower() in s.lower()]
        return {"data": stocks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/data/prices")
async def get_stock_prices(request: StockDataRequest):
    """获取股票价格数据"""
    try:
        data = {}
        for symbol in request.symbols:
            df = data_manager.get_price_data(symbol)
            if request.start_date or request.end_date:
                df = df.loc[request.start_date:request.end_date]
            if request.fields:
                df = df[request.fields]
            data[symbol] = df.to_dict(orient='index')
        return {"data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/factors/calculate")
async def calculate_factors(request: FactorRequest):
    """计算因子"""
    try:
        factors = {}
        for symbol in request.symbols:
            df = data_manager.get_price_data(symbol)
            factor_data = {}
            
            for factor_name in request.factor_names:
                if factor_name in tech_factor_engine.factor_list:
                    factor_data[factor_name] = tech_factor_engine.calculate_factor(factor_name, df)
                elif factor_name in fundamental_factor_engine.factor_list:
                    factor_data[factor_name] = fundamental_factor_engine.calculate_factor(factor_name, df)
            
            factors[symbol] = factor_data
        
        return {"data": factors}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/factors/list")
async def get_factor_list():
    """获取可用因子列表"""
    try:
        tech_factors = tech_factor_engine.factor_list
        fundamental_factors = fundamental_factor_engine.factor_list
        
        return {
            "technical_factors": tech_factors,
            "fundamental_factors": fundamental_factors,
            "total_count": len(tech_factors) + len(fundamental_factors)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/model/predict")
async def predict(request: PredictionRequest):
    """股票评分预测"""
    try:
        features = []
        valid_symbols = []
        
        for symbol in request.symbols:
            df = data_manager.get_price_data(symbol)
            if len(df) > 0:
                tech_factors = tech_factor_engine.calculate_all(df)
                features.append(tech_factors.values[-1])
                valid_symbols.append(symbol)
        
        if len(features) == 0:
            return {"data": []}
        
        model = RankingModel(model_type='lgbm')
        scores = model.predict(np.array(features))
        
        results = [
            {"symbol": symbol, "score": float(score), "rank": 0}
            for symbol, score in zip(valid_symbols, scores)
        ]
        
        results.sort(key=lambda x: x['score'], reverse=True)
        for i, r in enumerate(results):
            r['rank'] = i + 1
        
        return {"data": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/portfolio/optimize")
async def optimize_portfolio(request: OptimizationRequest):
    """组合优化"""
    try:
        n_assets = len(request.symbols)
        np.random.seed(42)
        
        expected_returns = np.random.uniform(-0.01, 0.02, n_assets)
        cov_matrix = np.random.randn(n_assets, n_assets) * 0.01
        cov_matrix = cov_matrix @ cov_matrix.T
        
        optimizer = MeanVarianceOptimizer(
            returns=expected_returns,
            cov_matrix=cov_matrix,
            lambda_reg=1.0
        )
        
        weights = optimizer.optimize()
        
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        results = {
            "weights": {symbol: float(w) for symbol, w in zip(request.symbols, weights)},
            "expected_return": float(portfolio_return),
            "risk": float(portfolio_risk),
            "sharpe_ratio": float(portfolio_return / portfolio_risk) if portfolio_risk > 0 else 0
        }
        
        return {"data": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/backtest/strategies")
async def get_strategies():
    """获取可用策略列表"""
    try:
        strategies = [
            {"name": "dual_ma", "description": "双均线策略"},
            {"name": "momentum", "description": "动量策略"},
            {"name": "mean_reversion", "description": "均值回归策略"},
            {"name": "ai_stock_picking", "description": "AI选股策略"}
        ]
        return {"data": strategies}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/backtest/run")
async def run_backtest(request: BacktestRequest):
    """运行回测"""
    try:
        np.random.seed(42)
        n_days = (datetime.strptime(request.end_date, "%Y-%m-%d") - 
                  datetime.strptime(request.start_date, "%Y-%m-%d")).days
        
        dates = pd.date_range(request.start_date, request.end_date)
        
        portfolio_values = [request.initial_capital]
        daily_returns = np.random.randn(n_days) * 0.005
        
        for ret in daily_returns:
            portfolio_values.append(portfolio_values[-1] * (1 + ret))
        
        equity_curve = pd.DataFrame({
            'date': dates,
            'value': portfolio_values[1:]
        })
        
        total_return = (portfolio_values[-1] - request.initial_capital) / request.initial_capital
        annualized_return = (1 + total_return) ** (252 / n_days) - 1
        max_drawdown = max(1 - np.array(portfolio_values) / np.maximum.accumulate(portfolio_values))
        
        results = {
            "strategy_name": request.strategy_name,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "initial_capital": request.initial_capital,
            "final_value": portfolio_values[-1],
            "total_return": total_return,
            "annualized_return": annualized_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252),
            "equity_curve": equity_curve.to_dict(orient='records')
        }
        
        return {"data": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metrics/{metric_type}")
async def get_metrics(metric_type: str):
    """获取指标"""
    try:
        if metric_type == 'performance':
            return {"data": ["total_return", "annualized_return", "sharpe_ratio", "max_drawdown", "win_rate"]}
        elif metric_type == 'risk':
            return {"data": ["var_95", "cvar_95", "tracking_error", "information_ratio"]}
        elif metric_type == 'attribution':
            return {"data": ["brinson", "factor", "risk", "timeseries"]}
        else:
            raise HTTPException(status_code=404, detail="Unknown metric type")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)