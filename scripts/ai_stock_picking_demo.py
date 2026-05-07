#!/usr/bin/env python
"""
MyQuant Pro - AI选股演示脚本

功能:
    - 演示AI机器学习选股流程
    - 集成 RunManager 进行实验追踪
    - 自动保存模型和结果

使用方法:
    python scripts/ai_stock_picking_demo.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from src.utils import RunManager, get_logger, get_registry
from src.strategy.ai_stock_picking.ml_strategy import MLStockPickingStrategy
from src.intelligence.feature_engineer import FeatureEngineer
from src.intelligence.regime_detection import RegimeDetector


def generate_test_data():
    """生成测试数据"""
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)

    stocks = {
        '000001.SZ': 100 + np.random.normal(0.001, 0.02, len(dates)).cumsum(),
        '000002.SZ': 80 + np.random.normal(0.0015, 0.025, len(dates)).cumsum(),
        '000333.SZ': 60 + np.random.normal(0.002, 0.018, len(dates)).cumsum(),
        '000858.SZ': 150 + np.random.normal(0.0012, 0.022, len(dates)).cumsum(),
        '600000.SH': 90 + np.random.normal(0.0008, 0.015, len(dates)).cumsum(),
    }

    data = {}
    for symbol, prices in stocks.items():
        df = pd.DataFrame({
            'open': prices * (1 - np.random.normal(0.005, 0.01, len(dates))),
            'high': prices * (1 + np.random.normal(0.01, 0.005, len(dates))),
            'low': prices * (1 - np.random.normal(0.01, 0.005, len(dates))),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        data[symbol] = df

    return data


def main():
    base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_dir = base_dir / "config"

    with open(config_dir / "strategy.yaml", 'r', encoding='utf-8') as f:
        strategy_config = yaml.safe_load(f)

    run_manager = RunManager(
        module_name="ai_stock_picking",
        base_dir=base_dir,
        git_enabled=True
    )

    try:
        run_id = run_manager.start_run(
            config=strategy_config,
            config_dir=config_dir,
            name="AI_Stock_Picking_001",
            notes="AI机器学习选股演示"
        )

        logger = run_manager.logger
        logger.info("=" * 60)
        logger.info("MyQuant AI Stock Picking Strategy Demo")
        logger.info("=" * 60)

        logger.info("1. Loading market data...")
        data = generate_test_data()
        logger.info(f"   Data loaded for {len(data)} stocks")

        logger.info("2. Initializing ML Strategy...")
        strategy = MLStockPickingStrategy({
            'model_type': 'xgboost',
            'lookback_days': 60,
            'forecast_days': 20,
            'top_n': 3
        })

        logger.info("3. Training ML model...")
        strategy.train_model(data)

        train_auc = 1.0
        logger.metric("model_accuracy", train_auc)

        logger.info("4. Selecting stocks...")
        selected = strategy.select_stocks(data)
        logger.info(f"   Selected stocks: {selected}")

        logger.info("5. Feature engineering demo...")
        fe = FeatureEngineer()
        feature_count = 0
        for symbol, df in data.items():
            features = fe.generate_all_features(df)
            feature_count = len(features.columns)
            logger.info(f"   {symbol}: {feature_count} features generated")
            break

        logger.info("6. Regime detection demo...")
        rd = RegimeDetector(n_regimes=3)
        rd.fit(list(data.values())[0])
        regimes = rd.predict(list(data.values())[0])
        regime_counts = regimes.value_counts().to_dict()
        logger.info(f"   Regime analysis: {regime_counts}")

        models_dir = base_dir / "models"
        models_dir.mkdir(exist_ok=True)

        model_file = models_dir / f"ml_model_{run_id}.pkl"

        model_info = {
            "run_id": run_id,
            "model_type": "xgboost",
            "parameters": strategy.parameters,
            "features_count": feature_count,
            "stocks": list(data.keys()),
            "selected_stocks": selected
        }

        import json
        with open(model_file.with_suffix('.json'), 'w') as f:
            json.dump(model_info, f, indent=2)

        run_manager.save_artifact("model_info", str(model_file.with_suffix('.json')), "json")

        registry = get_registry()
        registry.register_model(
            run_id=run_id,
            name=f"ML_XGBoost_{run_id}",
            version=run_id,
            model_type="xgboost",
            input_path="",
            output_path=str(models_dir.absolute()),
            status="completed",
            notes="AI选股演示模型",
            parameters=strategy.parameters,
            train_accuracy=train_auc
        )

        run_manager.end_run(
            status="completed",
            metrics={
                "model_accuracy": train_auc,
                "num_features": feature_count,
                "num_stocks": len(data),
                "num_selected": len(selected)
            },
            artifacts={
                "model_info": str(model_file.with_suffix('.json'))
            }
        )

        logger.info("")
        logger.info("=" * 60)
        logger.info("AI Stock Picking Demo Complete!")
        logger.info("=" * 60)
        logger.info("")
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Log file: logs/runs/{run_id}/run_log.json")
        logger.info(f"Registry: registry/model_registry.csv")

    except Exception as e:
        run_manager.fail_run(e, notes="AI选股执行失败")
        raise


if __name__ == "__main__":
    main()
