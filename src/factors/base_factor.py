import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')


class BaseFactor(ABC):
    def __init__(self, name: str, category: str, version: str = "1.0.0"):
        self.name = name
        self.category = category
        self.version = version
        self.metadata = {
            "name": name,
            "category": category,
            "version": version,
            "description": ""
        }

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        pass

    def get_metadata(self) -> Dict:
        return self.metadata

    def set_description(self, description: str):
        self.metadata["description"] = description


class TechnicalFactor(BaseFactor):
    def __init__(self, name: str, version: str = "1.0.0"):
        super().__init__(name, "technical", version)


class FundamentalFactor(BaseFactor):
    def __init__(self, name: str, version: str = "1.0.0"):
        super().__init__(name, "fundamental", version)


class StyleFactor(BaseFactor):
    def __init__(self, name: str, version: str = "1.0.0"):
        super().__init__(name, "style", version)


class MomentumFactor(TechnicalFactor):
    def __init__(self, period: int = 20, version: str = "1.0.0"):
        super().__init__(f"momentum_{period}", version)
        self.period = period
        self.metadata["parameters"] = {"period": period}

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        return data['close'].pct_change(self.period)


class RSIFactor(TechnicalFactor):
    def __init__(self, period: int = 14, version: str = "1.0.0"):
        super().__init__(f"rsi_{period}", version)
        self.period = period
        self.metadata["parameters"] = {"period": period}

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class MACDFactor(TechnicalFactor):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9,
                 version: str = "1.0.0"):
        super().__init__(f"macd_{fast}_{slow}_{signal}", version)
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.metadata["parameters"] = {
            "fast": fast, "slow": slow, "signal": signal
        }

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        ema_fast = data['close'].ewm(span=self.fast, adjust=False).mean()
        ema_slow = data['close'].ewm(span=self.slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=self.signal, adjust=False).mean()
        return macd - signal_line


class VolatilityFactor(TechnicalFactor):
    def __init__(self, period: int = 20, version: str = "1.0.0"):
        super().__init__(f"volatility_{period}", version)
        self.period = period
        self.metadata["parameters"] = {"period": period}

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        returns = data['close'].pct_change()
        return returns.rolling(window=self.period).std() * np.sqrt(252)


class ATRFactor(TechnicalFactor):
    def __init__(self, period: int = 14, version: str = "1.0.0"):
        super().__init__(f"atr_{period}", version)
        self.period = period
        self.metadata["parameters"] = {"period": period}

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)

        atr = true_range.rolling(window=self.period).mean()
        return atr


class MAFactor(TechnicalFactor):
    def __init__(self, period: int = 20, version: str = "1.0.0"):
        super().__init__(f"ma_{period}", version)
        self.period = period
        self.metadata["parameters"] = {"period": period}

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        return data['close'].rolling(window=self.period).mean()


class MARatioFactor(TechnicalFactor):
    def __init__(self, short_period: int = 20, long_period: int = 60,
                 version: str = "1.0.0"):
        super().__init__(f"ma_ratio_{short_period}_{long_period}", version)
        self.short_period = short_period
        self.long_period = long_period
        self.metadata["parameters"] = {
            "short_period": short_period,
            "long_period": long_period
        }

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        ma_short = data['close'].rolling(window=self.short_period).mean()
        ma_long = data['close'].rolling(window=self.long_period).mean()
        return ma_short / ma_long


class VolumeRatioFactor(TechnicalFactor):
    def __init__(self, period: int = 20, version: str = "1.0.0"):
        super().__init__(f"volume_ratio_{period}", version)
        self.period = period
        self.metadata["parameters"] = {"period": period}

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        volume_ma = data['volume'].rolling(window=self.period).mean()
        return data['volume'] / volume_ma


class BollingerBandWidthFactor(TechnicalFactor):
    def __init__(self, period: int = 20, num_std: float = 2.0,
                 version: str = "1.0.0"):
        super().__init__(f"bollinger_width_{period}", version)
        self.period = period
        self.num_std = num_std
        self.metadata["parameters"] = {
            "period": period,
            "num_std": num_std
        }

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        ma = data['close'].rolling(window=self.period).mean()
        std = data['close'].rolling(window=self.period).std()
        upper = ma + self.num_std * std
        lower = ma - self.num_std * std
        return (upper - lower) / ma


class KDJFactor(TechnicalFactor):
    def __init__(self, period: int = 9, version: str = "1.0.0"):
        super().__init__(f"kdj_{period}", version)
        self.period = period
        self.metadata["parameters"] = {"period": period}

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        low_n = data['low'].rolling(window=self.period).min()
        high_n = data['high'].rolling(window=self.period).max()

        rsv = (data['close'] - low_n) / (high_n - low_n) * 100

        k = rsv.ewm(com=2, adjust=False).mean()
        d = k.ewm(com=2, adjust=False).mean()
        j = 3 * k - 2 * d

        return j


class TurnoverRateFactor(TechnicalFactor):
    def __init__(self, period: int = 20, version: str = "1.0.0"):
        super().__init__(f"turnover_rate_{period}", version)
        self.period = period
        self.metadata["parameters"] = {"period": period}

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        turnover = data['volume'] / data['volume'].rolling(window=self.period).mean()
        return turnover


class FactorRegistry:
    def __init__(self):
        self.factors: Dict[str, BaseFactor] = {}

    def register(self, factor: BaseFactor):
        key = f"{factor.category}.{factor.name}"
        self.factors[key] = factor

    def get(self, name: str, category: str) -> Optional[BaseFactor]:
        key = f"{category}.{name}"
        return self.factors.get(key)

    def list_factors(self, category: Optional[str] = None) -> List[str]:
        if category:
            return [k for k in self.factors.keys() if k.startswith(f"{category}.")]
        return list(self.factors.keys())

    def get_factor_metadata(self, name: str, category: str) -> Optional[Dict]:
        factor = self.get(name, category)
        return factor.get_metadata() if factor else None
