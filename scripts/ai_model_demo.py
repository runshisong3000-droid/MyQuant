#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MyQuant AI模型层演示脚本

功能:
    - 测试深度学习模型（LSTM、Transformer）
    - 测试模型集成功能
    - 测试训练框架
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from datetime import datetime
from src.utils.logger import get_logger
from src.utils.run_manager import RunManager
from src.strategy.ai_stock_picking import (
    LSTMModel,
    GRUModel,
    AttentionLSTM,
    TransformerTimeSeries,
    ModelEnsembleFactory,
    ModelSelector,
    DataProcessor,
    CrossValidator,
    ModelEvaluator,
    TimeSeriesDataset
)
from torch.utils.data import DataLoader


def generate_sample_data(n_samples=1000, n_features=38, seq_length=30):
    """生成模拟时间序列数据"""
    np.random.seed(42)
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.1 > 0).astype(int)
    
    return X, y


def test_nn_models():
    """测试神经网络模型"""
    logger.info("Testing Neural Network Models...")
    
    X, y = generate_sample_data(n_samples=500, n_features=38)
    
    data_processor = DataProcessor()
    X_scaled, _ = data_processor.fit_transform(X, scaler_type='standard')
    
    seq_length = 30
    dataset = TimeSeriesDataset(X_scaled, y, seq_length=seq_length)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    models = [
        ('LSTM', LSTMModel(input_dim=38, hidden_dim=64, num_layers=2)),
        ('GRU', GRUModel(input_dim=38, hidden_dim=64, num_layers=2)),
        ('AttentionLSTM', AttentionLSTM(input_dim=38, hidden_dim=64, num_layers=2)),
        ('Transformer', TransformerTimeSeries(input_dim=38, d_model=64, nhead=4, num_layers=2))
    ]
    
    for name, model in models:
        logger.info(f"Training {name}...")
        history = model.train_model(loader, epochs=30, lr=0.001)
        logger.info(f"{name} trained successfully. Final loss: {history['train_loss'][-1]:.4f}")
        
        sample_X = X_scaled[:seq_length].reshape(1, seq_length, 38)
        pred = model.predict(sample_X)
        logger.info(f"{name} prediction: {pred[0][0]:.4f}")


def test_ensemble_models():
    """测试集成模型"""
    logger.info("\nTesting Ensemble Models...")
    
    X, y = generate_sample_data(n_samples=500, n_features=38)
    
    data_processor = DataProcessor()
    X_scaled, _ = data_processor.fit_transform(X, scaler_type='standard')
    
    ensemble_types = ['voting', 'stacking', 'blending', 'weighted']
    
    for ensemble_type in ensemble_types:
        logger.info(f"Creating {ensemble_type} ensemble...")
        ensemble = ModelEnsembleFactory.create_classifier_ensemble(ensemble_type=ensemble_type)
        ensemble.fit(X_scaled, y)
        
        y_pred = ensemble.predict(X_scaled)
        accuracy = (y_pred == y).mean()
        logger.info(f"{ensemble_type} ensemble accuracy: {accuracy:.4f}")


def test_trainer():
    """测试训练框架"""
    logger.info("\nTesting Trainer Framework...")
    
    X, y = generate_sample_data(n_samples=500, n_features=38)
    
    data_processor = DataProcessor()
    X_scaled, _ = data_processor.fit_transform(X, scaler_type='standard')
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    selector = ModelSelector()
    model = selector.get_model('xgb', 'classifier', n_estimators=100)
    
    cv = CrossValidator(n_folds=5)
    cv_results = cv.cross_validate(model, X_train, y_train, metrics=['accuracy', 'f1'])
    
    logger.info("Cross Validation Results:")
    for metric, scores in cv_results.items():
        if metric != 'time':
            logger.info(f"  {metric}: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
    
    model.fit(X_train, y_train)
    
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_classifier(model, X_test, y_test, verbose=False)
    logger.info(f"Test Set Results: {results}")


def main():
    global logger, run_manager
    
    run_manager = RunManager(module_name='ai_model_demo')
    run_manager.start_run()
    run_id = run_manager.run_id
    
    logger = get_logger('ai_model_demo', run_id)
    
    logger.info("="*60)
    logger.info("MyQuant AI Model Layer Demo")
    logger.info("="*60)
    
    try:
        test_nn_models()
        test_ensemble_models()
        test_trainer()
        
        logger.info("="*60)
        logger.info("AI Model Layer Demo Complete!")
        logger.info("="*60)
        
        run_manager.end_run(status='completed')
        
    except Exception as e:
        logger.error(f"Demo failed with error: {str(e)}", exc_info=True)
        run_manager.end_run(status='failed')


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    main()