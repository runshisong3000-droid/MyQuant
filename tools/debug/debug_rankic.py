"""Debug RankIC calculation"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

print("="*60)
print("Debug RankIC Calculation")
print("="*60)

data_path = 'data/processed/research_lite_prices.parquet'
price_data = pd.read_parquet(data_path)
price_data = price_data.sort_values(['stock', 'date']).reset_index(drop=True)
price_data['future_return'] = price_data.groupby('stock')['close'].pct_change().shift(-1)

future_returns = price_data[['date', 'stock', 'future_return']].dropna()
future_returns = future_returns.set_index(['date', 'stock'])['future_return']

from src.factors.neural.sequence_dataset import SequenceDataset

LOOKBACK_WINDOW = 20
RAW_FEATURES = ['open', 'high', 'low', 'close', 'volume']
TARGET_HORIZON = 1

dataset = SequenceDataset(
    df=price_data,
    lookback_window=LOOKBACK_WINDOW,
    features=RAW_FEATURES,
    target_horizon=TARGET_HORIZON,
    normalize=True
)

X, metadata = dataset.get_samples()

splits = dataset.get_train_val_test_split(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
X_test, meta_test = splits['test']

try:
    import torch
    import torch.nn as nn
    from src.factors.neural.autoencoder import SequenceAutoEncoder
    
    INPUT_DIM = len(RAW_FEATURES)
    EMBEDDING_DIM = 8
    DEVICE = 'cpu'
    
    print("\nTraining MLP encoder...")
    
    model = SequenceAutoEncoder(
        input_dim=INPUT_DIM,
        hidden_dim=32,
        embedding_dim=EMBEDDING_DIM,
        lookback_window=LOOKBACK_WINDOW,
        encoder_type='mlp'
    ).to(DEVICE)
    
    X_train, meta_train = splits['train']
    X_train_tensor = torch.FloatTensor(X_train)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_tensor, X_train_tensor),
        batch_size=64, shuffle=True
    )
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(3):
        model.train()
        total_loss = 0
        for batch_X, _ in train_loader:
            batch_X = batch_X.to(DEVICE)
            recon, emb = model(batch_X)
            loss = criterion(recon, batch_X)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}: loss={total_loss/len(train_loader):.6f}")
    
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
        _, embeddings = model(X_test_tensor)
    
    embeddings = embeddings.cpu().numpy()
    
    from src.factors.neural.neural_factor_extractor import NeuralFactorExtractor
    
    extractor = NeuralFactorExtractor(embedding_dim=EMBEDDING_DIM)
    factors_df = extractor.embedding_to_dataframe(embeddings, meta_test.reset_index(drop=True))
    
    factors_dict = {}
    for i in range(EMBEDDING_DIM):
        factor_col = f'neural_factor_{i}'
        if factor_col in factors_df.columns:
            series = factors_df.set_index(['signal_date', 'stock'])[factor_col]
            aligned = pd.Series(
                series.values,
                index=pd.MultiIndex.from_arrays([
                    series.index.get_level_values('signal_date'),
                    series.index.get_level_values('stock')
                ], names=['date', 'stock'])
            )
            factors_dict[factor_col] = aligned
    
    print("\nTesting RankIC calculation...")
    print("  Number of factors:", len(factors_dict))
    
    from src.factors.auto.factor_evaluator import FactorEvaluator
    
    evaluator = FactorEvaluator()
    
    for factor_name, factor_data in factors_dict.items():
        print(f"\n  {factor_name}:")
        print(f"    Factor data shape: {factor_data.shape}")
        print(f"    Returns data shape: {future_returns.shape}")
        
        common_idx = factor_data.index.intersection(future_returns.index)
        print(f"    Common index count: {len(common_idx)}")
        
        dates = factor_data.index.get_level_values(0).unique()
        print(f"    Number of dates: {len(dates)}")
        
        # Check duplicates
        print(f"    Factor duplicates: {factor_data.index.duplicated().sum()}")
        print(f"    Future returns duplicates: {future_returns.index.duplicated().sum()}")
        
        # Let's look at one day's data
        if len(dates) > 0:
            sample_date = dates[0]
            print(f"\n    Sample date: {sample_date}")
            factor_slice = factor_data[factor_data.index.get_level_values(0) == sample_date]
            return_slice = future_returns[future_returns.index.get_level_values(0) == sample_date]
            print(f"      Factor slice shape: {factor_slice.shape}")
            print(f"      Return slice shape: {return_slice.shape}")
            print(f"      Factor index first 5: {factor_slice.index[:5]}")
            print(f"      Return index first 5: {return_slice.index[:5]}")
        
        print("\n    Now using FactorEvaluator:")
        result = evaluator.calculate_rank_ic(factor_data, future_returns)
        print(f"    Evaluator mean: {result['mean']:.4f}")
        print(f"    Evaluator count: {result['count']}")
        print(f"    Evaluator std: {result['std']:.4f}")
        
        # Also test full evaluation
        eval_result = evaluator.evaluate_single(factor_data, future_returns)
        print(f"\n    Full evaluation:")
        print(f"      RankIC mean: {eval_result['rank_ic']['mean']:.4f}")
        print(f"      ICIR: {eval_result['icir']:.4f}")
        print(f"      Coverage: {eval_result['coverage']:.4f}")
        break
        
except Exception as e:
    import traceback
    print(f"\nERROR: {e}")
    traceback.print_exc()

print("\nDone!")
