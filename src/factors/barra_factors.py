"""
BARRA风格因子库

功能:
    - BARRA CNE5风格因子
    - 因子暴露计算
    - 因子正交化处理
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
from sklearn.decomposition import PCA


class BARRAFactors:
    """BARRA CNE5风格因子"""

    @staticmethod
    def size_factor(market_cap: pd.Series) -> pd.Series:
        """规模因子 (Size)"""
        size = np.log(market_cap)
        return size.rename('size')

    @staticmethod
    def value_factor(pe: pd.Series, pb: pd.Series, ps: pd.Series) -> pd.Series:
        """价值因子 (Value)"""
        value_scores = []

        if pe is not None and len(pe.dropna()) > 0:
            value_scores.append(-pe.rank(pct=True))

        if pb is not None and len(pb.dropna()) > 0:
            value_scores.append(-pb.rank(pct=True))

        if ps is not None and len(ps.dropna()) > 0:
            value_scores.append(-ps.rank(pct=True))

        if value_scores:
            value = pd.concat(value_scores, axis=1).mean(axis=1)
        else:
            value = pd.Series(np.nan, index=pe.index if pe is not None else pb.index if pb is not None else ps.index)

        return value.rename('value')

    @staticmethod
    def momentum_factor(returns: pd.Series, window: int = 12) -> pd.Series:
        """动量因子 (Momentum)"""
        momentum = returns.rolling(window=window).sum()
        return momentum.rename('momentum')

    @staticmethod
    def quality_factor(
        roe: pd.Series,
        roa: pd.Series,
        operating_margin: pd.Series
    ) -> pd.Series:
        """质量因子 (Quality)"""
        quality_scores = []

        if roe is not None and len(roe.dropna()) > 0:
            quality_scores.append(roe.rank(pct=True))

        if roa is not None and len(roa.dropna()) > 0:
            quality_scores.append(roa.rank(pct=True))

        if operating_margin is not None and len(operating_margin.dropna()) > 0:
            quality_scores.append(operating_margin.rank(pct=True))

        if quality_scores:
            quality = pd.concat(quality_scores, axis=1).mean(axis=1)
        else:
            quality = pd.Series(np.nan, index=roe.index if roe is not None else roa.index if roa is not None else operating_margin.index)

        return quality.rename('quality')

    @staticmethod
    def volatility_factor(returns: pd.Series, window: int = 60) -> pd.Series:
        """波动率因子 (Volatility)"""
        volatility = returns.rolling(window=window).std()
        return (-volatility).rename('volatility')

    @staticmethod
    def beta_factor(
        stock_returns: pd.Series,
        market_returns: pd.Series,
        window: int = 60
    ) -> pd.Series:
        """Beta因子"""
        cov = stock_returns.rolling(window=window).cov(market_returns)
        market_var = market_returns.rolling(window=window).var()
        beta = cov / market_var
        return beta.rename('beta')

    @staticmethod
    def liquidity_factor(volume: pd.Series, price: pd.Series, window: int = 20) -> pd.Series:
        """流动性因子"""
        turnover = volume / (price * 1e8)
        avg_turnover = turnover.rolling(window=window).mean()
        return avg_turnover.rename('liquidity')

    @staticmethod
    def dividend_yield_factor(dividend: pd.Series, price: pd.Series) -> pd.Series:
        """股息收益率因子"""
        yield_ = dividend / price
        return yield_.rename('dividend_yield')

    @staticmethod
    def leverage_factor(debt: pd.Series, equity: pd.Series) -> pd.Series:
        """杠杆因子"""
        leverage = debt / equity
        return leverage.rename('leverage')


class BARRAFactorEngine:
    """BARRA风格因子引擎"""

    def __init__(self):
        self.factor_names = [
            'size', 'value', 'momentum', 'quality', 'volatility',
            'beta', 'liquidity', 'dividend_yield', 'leverage'
        ]

    def compute_all(
        self,
        data: pd.DataFrame,
        market_returns: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """计算所有BARRA风格因子"""
        results = pd.DataFrame(index=data.index)

        factors = [
            ('size', lambda: BARRAFactors.size_factor(data['market_cap']) if 'market_cap' in data.columns else pd.Series(np.nan, index=data.index)),
            ('value', lambda: BARRAFactors.value_factor(
                data['pe'] if 'pe' in data.columns else None,
                data['pb'] if 'pb' in data.columns else None,
                data['ps'] if 'ps' in data.columns else None
            )),
            ('momentum', lambda: BARRAFactors.momentum_factor(data['returns']) if 'returns' in data.columns else pd.Series(np.nan, index=data.index)),
            ('quality', lambda: BARRAFactors.quality_factor(
                data['roe'] if 'roe' in data.columns else None,
                data['roa'] if 'roa' in data.columns else None,
                data['operating_margin'] if 'operating_margin' in data.columns else None
            )),
            ('volatility', lambda: BARRAFactors.volatility_factor(data['returns']) if 'returns' in data.columns else pd.Series(np.nan, index=data.index)),
            ('beta', lambda: BARRAFactors.beta_factor(data['returns'], market_returns, 60) if 'returns' in data.columns and market_returns is not None else pd.Series(np.nan, index=data.index)),
            ('liquidity', lambda: BARRAFactors.liquidity_factor(data['volume'], data['close']) if 'volume' in data.columns and 'close' in data.columns else pd.Series(np.nan, index=data.index)),
            ('dividend_yield', lambda: BARRAFactors.dividend_yield_factor(data['dividend'], data['close']) if 'dividend' in data.columns and 'close' in data.columns else pd.Series(np.nan, index=data.index)),
            ('leverage', lambda: BARRAFactors.leverage_factor(data['debt'], data['equity']) if 'debt' in data.columns and 'equity' in data.columns else pd.Series(np.nan, index=data.index)),
        ]

        for name, func in factors:
            try:
                result = func()
                results[name] = result
            except Exception as e:
                print(f"Failed to compute {name}: {e}")
                results[name] = np.nan

        return results

    def orthogonalize(self, factors: pd.DataFrame, reference_factor: str = 'size') -> pd.DataFrame:
        """因子正交化"""
        if reference_factor not in factors.columns:
            return factors

        result = factors.copy()

        for col in factors.columns:
            if col != reference_factor:
                y = factors[col].dropna()
                X = factors[[reference_factor]].loc[y.index]

                if len(y) > 0:
                    beta = (X.values * y.values).mean() / (X.values ** 2).mean()
                    result.loc[y.index, col] = y - beta * X.values.flatten()

        return result

    def normalize(self, factors: pd.DataFrame) -> pd.DataFrame:
        """因子标准化"""
        result = factors.copy()

        for col in factors.columns:
            if len(factors[col].dropna()) > 0:
                mean = factors[col].mean()
                std = factors[col].std()
                if std > 0:
                    result[col] = (factors[col] - mean) / std

        return result

    def winsorize(self, factors: pd.DataFrame, quantile: float = 0.01) -> pd.DataFrame:
        """因子截尾处理"""
        result = factors.copy()

        for col in factors.columns:
            if len(factors[col].dropna()) > 0:
                lower = factors[col].quantile(quantile)
                upper = factors[col].quantile(1 - quantile)
                result[col] = np.clip(factors[col], lower, upper)

        return result

    def get_factor_names(self) -> List[str]:
        """获取因子名称"""
        return self.factor_names


class FactorAnalyzer:
    """因子分析工具"""

    @staticmethod
    def compute_ic(factor: pd.Series, returns: pd.Series, period: int = 1) -> float:
        """计算信息系数 (IC)"""
        factor_shifted = factor.shift(period)
        aligned = pd.concat([factor_shifted, returns], axis=1).dropna()

        if len(aligned) < 2:
            return np.nan

        return aligned.corr().iloc[0, 1]

    @staticmethod
    def compute_ic_series(factor: pd.Series, returns: pd.Series, period: int = 1) -> pd.Series:
        """计算滚动IC"""
        ic_values = []
        dates = []

        for i in range(period, len(factor)):
            ic = FactorAnalyzer.compute_ic(
                factor.iloc[:i],
                returns.iloc[:i],
                period
            )
            ic_values.append(ic)
            dates.append(factor.index[i])

        return pd.Series(ic_values, index=dates).rename('ic')

    @staticmethod
    def compute_ir(factor: pd.Series, returns: pd.Series, period: int = 1) -> float:
        """计算信息比率 (IR)"""
        ic_series = FactorAnalyzer.compute_ic_series(factor, returns, period)

        if len(ic_series.dropna()) == 0:
            return np.nan

        return ic_series.mean() / ic_series.std()

    @staticmethod
    def factor_correlation(factors: pd.DataFrame) -> pd.DataFrame:
        """计算因子相关性矩阵"""
        return factors.corr()

    @staticmethod
    def factor_exposure(factors: pd.DataFrame, returns: pd.Series) -> pd.Series:
        """计算因子暴露"""
        factors_df = factors.dropna()
        returns_aligned = returns.loc[factors_df.index]

        if len(factors_df) == 0:
            return pd.Series(index=factors.columns, dtype=float)

        exposures = []
        for col in factors.columns:
            corr = factors_df[col].corr(returns_aligned)
            exposures.append(corr)

        return pd.Series(exposures, index=factors.columns).rename('exposure')

    @staticmethod
    def pca_analysis(factors: pd.DataFrame, n_components: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
        """PCA分析"""
        factors_clean = factors.dropna()

        if len(factors_clean) < n_components:
            return pd.DataFrame(), pd.Series()

        pca = PCA(n_components=n_components)
        components = pca.fit_transform(factors_clean)
        explained_variance = pd.Series(pca.explained_variance_ratio_, index=[f'PC{i+1}' for i in range(n_components)])

        components_df = pd.DataFrame(components, index=factors_clean.index, columns=[f'PC{i+1}' for i in range(n_components)])

        return components_df, explained_variance
