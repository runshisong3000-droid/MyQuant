#!/usr/bin/env python
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import pandas as pd
from src.data.loader import DataLoader
from pathlib import Path


def main():
    config_path = "config/data.yaml"
    
    loader = DataLoader(config_path)
    universe = loader.get_universe()
    
    print(f"Downloading data for {len(universe)} stocks...")
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    for i, symbol in enumerate(universe, 1):
        try:
            print(f"  {i}/{len(universe)}: {symbol}")
            df = loader.get_price_data(symbol, use_cache=False)
            df.to_csv(data_dir / f"{symbol}.csv")
            print(f"  Saved to data/{symbol}.csv")
        except Exception as e:
            print(f"  Error downloading {symbol}: {e}")
    
    print("\nData download complete!")


if __name__ == "__main__":
    main()
