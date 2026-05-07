"""
技术因子库

功能:
    - 动量因子（RSI、MACD、动量指标）
    - 波动率因子（ATR、布林带、历史波动率）
    - 量价因子（OBV、VWAP、成交量比率）
    - 趋势因子（均线、ADX、SAR）
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
from abc import ABC, abstractmethod


class FactorBase(ABC):
    """因子基类"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值"""
        pass


class MomentumFactors:
    """动量因子计算"""

    @staticmethod
    def rsi(data: pd.DataFrame, window: int = 14) -> pd.Series:
        """相对强弱指数"""
        close = data['close']
        delta = close.diff()

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi.rename('rsi')

    @staticmethod
    def macd(data: pd.DataFrame, short_window: int = 12, long_window: int = 26, signal_window: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD指标"""
        close = data['close']

        ema_short = close.ewm(span=short_window, adjust=False).mean()
        ema_long = close.ewm(span=long_window, adjust=False).mean()

        macd_line = ema_short - ema_long
        signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line.rename('macd'), signal_line.rename('signal'), histogram.rename('histogram')

    @staticmethod
    def momentum(data: pd.DataFrame, window: int = 12) -> pd.Series:
        """动量指标"""
        close = data['close']
        momentum = close.diff(window) / close.shift(window)

        return momentum.rename('momentum')

    @staticmethod
    def roc(data: pd.DataFrame, window: int = 12) -> pd.Series:
        """变动率指标"""
        close = data['close']
        roc = ((close - close.shift(window)) / close.shift(window)) * 100

        return roc.rename('roc')

    @staticmethod
    def wma(data: pd.DataFrame, window: int = 20) -> pd.Series:
        """加权移动平均"""
        close = data['close']

        weights = np.arange(1, window + 1)
        wma = close.rolling(window=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )

        return wma.rename(f'wma_{window}')

    @staticmethod
    def ema(data: pd.DataFrame, window: int = 20) -> pd.Series:
        """指数移动平均"""
        close = data['close']
        ema = close.ewm(span=window, adjust=False).mean()

        return ema.rename(f'ema_{window}')

    @staticmethod
    def ma_diff(data: pd.DataFrame, short_window: int = 20, long_window: int = 60) -> pd.Series:
        """均线差"""
        close = data['close']

        ma_short = close.rolling(window=short_window).mean()
        ma_long = close.rolling(window=long_window).mean()

        return (ma_short - ma_long).rename(f'ma_diff_{short_window}_{long_window}')


class VolatilityFactors:
    """波动率因子计算"""

    @staticmethod
    def atr(data: pd.DataFrame, window: int = 14) -> pd.Series:
        """平均真实波动幅度"""
        high = data['high']
        low = data['low']
        close = data['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(window=window).mean()

        return atr.rename('atr')

    @staticmethod
    def bollinger_bands(data: pd.DataFrame, window: int = 20, num_std: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """布林带"""
        close = data['close']

        middle_band = close.rolling(window=window).mean()
        std = close.rolling(window=window).std()

        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)

        return upper_band.rename('bb_upper'), middle_band.rename('bb_middle'), lower_band.rename('bb_lower')

    @staticmethod
    def bb_width(data: pd.DataFrame, window: int = 20, num_std: int = 2) -> pd.Series:
        """布林带宽度"""
        upper, middle, lower = VolatilityFactors.bollinger_bands(data, window, num_std)
        width = ((upper - lower) / middle) * 100

        return width.rename('bb_width')

    @staticmethod
    def bb_percent(data: pd.DataFrame, window: int = 20, num_std: int = 2) -> pd.Series:
        """布林带百分比"""
        close = data['close']
        upper, _, lower = VolatilityFactors.bollinger_bands(data, window, num_std)

        bb_percent = ((close - lower) / (upper - lower)) * 100

        return bb_percent.rename('bb_percent')

    @staticmethod
    def historical_volatility(data: pd.DataFrame, window: int = 20, annualize: bool = True) -> pd.Series:
        """历史波动率"""
        close = data['close']
        returns = close.pct_change()

        volatility = returns.rolling(window=window).std()

        if annualize:
            volatility = volatility * np.sqrt(252)

        return volatility.rename(f'volatility_{window}')

    @staticmethod
    def downside_deviation(data: pd.DataFrame, window: int = 20) -> pd.Series:
        """下行偏差"""
        close = data['close']
        returns = close.pct_change()

        downside_returns = returns.where(returns < 0, 0)
        downside_deviation = np.sqrt((downside_returns ** 2).rolling(window=window).mean())

        return downside_deviation.rename('downside_deviation')


class VolumeFactors:
    """量价因子计算"""

    @staticmethod
    def obv(data: pd.DataFrame) -> pd.Series:
        """能量潮指标"""
        close = data['close']
        volume = data['volume']

        direction = np.where(close > close.shift(), 1, np.where(close < close.shift(), -1, 0))
        obv = (volume * direction).cumsum()

        return pd.Series(obv, index=data.index).rename('obv')

    @staticmethod
    def vwap(data: pd.DataFrame) -> pd.Series:
        """成交量加权平均价"""
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']

        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()

        return vwap.rename('vwap')

    @staticmethod
    def volume_ratio(data: pd.DataFrame, window: int = 20) -> pd.Series:
        """成交量比率"""
        volume = data['volume']
        avg_volume = volume.rolling(window=window).mean()

        volume_ratio = volume / avg_volume

        return volume_ratio.rename('volume_ratio')

    @staticmethod
    def volume_change(data: pd.DataFrame, window: int = 5) -> pd.Series:
        """成交量变化率"""
        volume = data['volume']
        volume_change = volume.pct_change(window)

        return volume_change.rename('volume_change')

    @staticmethod
    def vpt(data: pd.DataFrame) -> pd.Series:
        """量价趋势指标"""
        close = data['close']
        volume = data['volume']

        vpt = ((close - close.shift()) / close.shift()) * volume
        vpt = vpt.cumsum()

        return pd.Series(vpt, index=data.index).rename('vpt')


class TrendFactors:
    """趋势因子计算"""

    @staticmethod
    def adx(data: pd.DataFrame, window: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """平均趋向指数"""
        high = data['high']
        low = data['low']
        close = data['close']

        up_move = high - high.shift()
        down_move = low.shift() - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)

        atr = tr.rolling(window=window).mean()

        plus_di = 100 * (plus_dm.rolling(window=window).sum() / atr)
        minus_di = 100 * (minus_dm.rolling(window=window).sum() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=window).mean()

        return adx.rename('adx'), plus_di.rename('plus_di'), minus_di.rename('minus_di')

    @staticmethod
    def sar(data: pd.DataFrame, acceleration: float = 0.02, max_acceleration: float = 0.2) -> pd.Series:
        """抛物线转向指标"""
        high = data['high']
        low = data['low']

        sar = pd.Series(index=data.index)
        sar.iloc[0] = low.iloc[0]

        trend = 1
        ep = high.iloc[0]
        af = acceleration

        for i in range(1, len(data)):
            if trend == 1:
                sar.iloc[i] = sar.iloc[i-1] + af * (ep - sar.iloc[i-1])
                if low.iloc[i] < sar.iloc[i]:
                    trend = -1
                    sar.iloc[i] = ep
                    ep = low.iloc[i]
                    af = acceleration
                else:
                    if high.iloc[i] > ep:
                        ep = high.iloc[i]
                        af = min(af + acceleration, max_acceleration)
            else:
                sar.iloc[i] = sar.iloc[i-1] - af * (sar.iloc[i-1] - ep)
                if high.iloc[i] > sar.iloc[i]:
                    trend = 1
                    sar.iloc[i] = ep
                    ep = high.iloc[i]
                    af = acceleration
                else:
                    if low.iloc[i] < ep:
                        ep = low.iloc[i]
                        af = min(af + acceleration, max_acceleration)

        return sar.rename('sar')

    @staticmethod
    def ma_crossover_signal(data: pd.DataFrame, short_window: int = 20, long_window: int = 60) -> pd.Series:
        """均线交叉信号"""
        close = data['close']

        ma_short = close.rolling(window=short_window).mean()
        ma_long = close.rolling(window=long_window).mean()

        signal = np.where(ma_short > ma_long, 1, np.where(ma_short < ma_long, -1, 0))

        return pd.Series(signal, index=data.index).rename('ma_crossover')

    @staticmethod
    def slope(data: pd.DataFrame, window: int = 20) -> pd.Series:
        """价格斜率"""
        close = data['close']

        x = np.arange(window)
        slope = close.rolling(window=window).apply(
            lambda y: np.polyfit(x, y, 1)[0], raw=True
        )

        return slope.rename('slope')


class TechnicalFactorEngine:
    """技术因子引擎"""

    def __init__(self):
        self.factors = {
            'momentum': [
                ('rsi', MomentumFactors.rsi),
                ('momentum', MomentumFactors.momentum),
                ('roc', MomentumFactors.roc),
                ('wma_20', lambda df: MomentumFactors.wma(df, 20)),
                ('ema_20', lambda df: MomentumFactors.ema(df, 20)),
                ('ma_diff', MomentumFactors.ma_diff),
            ],
            'volatility': [
                ('atr', VolatilityFactors.atr),
                ('bb_width', VolatilityFactors.bb_width),
                ('bb_percent', VolatilityFactors.bb_percent),
                ('volatility_20', lambda df: VolatilityFactors.historical_volatility(df, 20)),
                ('downside_deviation', VolatilityFactors.downside_deviation),
            ],
            'volume': [
                ('obv', VolumeFactors.obv),
                ('vwap', VolumeFactors.vwap),
                ('volume_ratio', VolumeFactors.volume_ratio),
                ('volume_change', VolumeFactors.volume_change),
                ('vpt', VolumeFactors.vpt),
            ],
            'trend': [
                ('adx', lambda df: TrendFactors.adx(df)[0]),
                ('sar', TrendFactors.sar),
                ('ma_crossover', TrendFactors.ma_crossover_signal),
                ('slope', TrendFactors.slope),
            ]
        }

    def compute_all(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算所有技术因子"""
        results = pd.DataFrame(index=data.index)

        for category, factor_list in self.factors.items():
            for name, func in factor_list:
                try:
                    result = func(data)
                    if isinstance(result, pd.Series):
                        results[name] = result
                    else:
                        for i, ser in enumerate(result):
                            results[f"{name}_{i}"] = ser
                except Exception as e:
                    print(f"Failed to compute {name}: {e}")
                    results[name] = np.nan

        return results

    def compute_by_category(self, data: pd.DataFrame, category: str) -> pd.DataFrame:
        """按类别计算因子"""
        if category not in self.factors:
            raise ValueError(f"Category {category} not found. Available: {list(self.factors.keys())}")

        results = pd.DataFrame(index=data.index)

        for name, func in self.factors[category]:
            try:
                result = func(data)
                if isinstance(result, pd.Series):
                    results[name] = result
            except Exception as e:
                print(f"Failed to compute {name}: {e}")
                results[name] = np.nan

        return results

    def get_factor_names(self, category: Optional[str] = None) -> List[str]:
        """获取因子名称列表"""
        if category:
            if category not in self.factors:
                raise ValueError(f"Category {category} not found")
            return [name for name, _ in self.factors[category]]
        else:
            return [name for cat in self.factors.values() for name, _ in cat]

    def compute_selected(self, data: pd.DataFrame, factor_names: List[str]) -> pd.DataFrame:
        """计算指定因子"""
        results = pd.DataFrame(index=data.index)

        for cat_factors in self.factors.values():
            for name, func in cat_factors:
                if name in factor_names:
                    try:
                        result = func(data)
                        if isinstance(result, pd.Series):
                            results[name] = result
                    except Exception as e:
                        print(f"Failed to compute {name}: {e}")
                        results[name] = np.nan

        return results
