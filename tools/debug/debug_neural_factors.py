"""Debug neural factor evaluation"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

print("="*60)
print("Debug Neural Factor Evaluation")
print("="*60)

data_path = 'data/processed/research_lite_prices.parquet'
price_data = pd.read_parquet(data_path)
price_data = price_data.sort_values(['stock', 'date']).reset_index(drop=True)
price_data['future_return'] = price_data.groupby('stock')['close'].pct_change().shift(-1)

future_returns = price_data[['date', 'stock', 'future_return']].dropna()
future_returns = future_returns.set_index(['date', 'stock'])['future_return']

print("\nFuture Returns Info:")
print("  Shape:", future_returns.shape)
print("  Index type:", type(future_returns.index))
print("  Index levels:", future_returns.index.names)
print("  First 5 index:", future_returns.index[:5])

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

print("\nMetadata Info:")
print("  Shape:", metadata.shape)
print("  Columns:", metadata.columns.tolist())

splits = dataset.get_train_val_test_split(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
X_test, meta_test = splits['test']

print("\nTest Metadata Info:")
print("  Shape:", meta_test.shape)
print("  First 5 rows:")
print(meta_test.head())

print("\nIndex Alignment Check:")

temp_test_data = pd.DataFrame({
    'date': meta_test['signal_date'].values,
    'stock': meta_test['stock'].values
})
temp_test_data['dummy_factor'] = np.random.randn(len(temp_test_data))
temp_test_data = temp_test_data.set_index(['date', 'stock'])['dummy_factor']

common_idx = temp_test_data.index.intersection(future_returns.index)
print("  Common index count:", len(common_idx))

print("\nDone!")
