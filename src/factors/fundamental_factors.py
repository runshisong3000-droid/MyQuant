"""
基本面因子库

功能:
    - 估值因子（PE、PB、PS等）
    - 盈利能力因子（ROE、ROA等）
    - 成长因子（营收增长、净利润增长等）
    - 现金流因子
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List


class FundamentalFactors:
    """基本面因子计算"""

    @staticmethod
    def pe_ratio(price: pd.Series, earnings: pd.Series) -> pd.Series:
        """市盈率 (Price-to-Earnings)"""
        pe = price / earnings
        return pe.rename('pe')

    @staticmethod
    def pb_ratio(price: pd.Series, book_value: pd.Series) -> pd.Series:
        """市净率 (Price-to-Book)"""
        pb = price / book_value
        return pb.rename('pb')

    @staticmethod
    def ps_ratio(price: pd.Series, revenue: pd.Series, shares: pd.Series) -> pd.Series:
        """市销率 (Price-to-Sales)"""
        ps = price / (revenue / shares)
        return ps.rename('ps')

    @staticmethod
    def pcf_ratio(price: pd.Series, cash_flow: pd.Series, shares: pd.Series) -> pd.Series:
        """市现率 (Price-to-Cash Flow)"""
        pcf = price / (cash_flow / shares)
        return pcf.rename('pcf')

    @staticmethod
    def ev_ebitda(ev: pd.Series, ebitda: pd.Series) -> pd.Series:
        """EV/EBITDA"""
        ev_ebitda = ev / ebitda
        return ev_ebitda.rename('ev_ebitda')

    @staticmethod
    def roe(net_income: pd.Series, equity: pd.Series) -> pd.Series:
        """净资产收益率"""
        roe = net_income / equity
        return roe.rename('roe')

    @staticmethod
    def roa(net_income: pd.Series, assets: pd.Series) -> pd.Series:
        """总资产收益率"""
        roa = net_income / assets
        return roa.rename('roa')

    @staticmethod
    def roic(ebit: pd.Series, invested_capital: pd.Series) -> pd.Series:
        """投入资本回报率"""
        roic = ebit / invested_capital
        return roic.rename('roic')

    @staticmethod
    def gross_margin(gross_profit: pd.Series, revenue: pd.Series) -> pd.Series:
        """毛利率"""
        margin = gross_profit / revenue
        return margin.rename('gross_margin')

    @staticmethod
    def operating_margin(operating_income: pd.Series, revenue: pd.Series) -> pd.Series:
        """营业利润率"""
        margin = operating_income / revenue
        return margin.rename('operating_margin')

    @staticmethod
    def net_margin(net_income: pd.Series, revenue: pd.Series) -> pd.Series:
        """净利润率"""
        margin = net_income / revenue
        return margin.rename('net_margin')

    @staticmethod
    def eps(net_income: pd.Series, shares: pd.Series) -> pd.Series:
        """每股收益"""
        eps = net_income / shares
        return eps.rename('eps')

    @staticmethod
    def revenue_growth(revenue: pd.Series, periods: int = 4) -> pd.Series:
        """营收增长率"""
        growth = revenue.pct_change(periods)
        return growth.rename('revenue_growth')

    @staticmethod
    def net_income_growth(net_income: pd.Series, periods: int = 4) -> pd.Series:
        """净利润增长率"""
        growth = net_income.pct_change(periods)
        return growth.rename('net_income_growth')

    @staticmethod
    def eps_growth(eps: pd.Series, periods: int = 4) -> pd.Series:
        """EPS增长率"""
        growth = eps.pct_change(periods)
        return growth.rename('eps_growth')

    @staticmethod
    def asset_growth(assets: pd.Series, periods: int = 4) -> pd.Series:
        """总资产增长率"""
        growth = assets.pct_change(periods)
        return growth.rename('asset_growth')

    @staticmethod
    def equity_growth(equity: pd.Series, periods: int = 4) -> pd.Series:
        """净资产增长率"""
        growth = equity.pct_change(periods)
        return growth.rename('equity_growth')

    @staticmethod
    def debt_to_equity(debt: pd.Series, equity: pd.Series) -> pd.Series:
        """资产负债率"""
        ratio = debt / equity
        return ratio.rename('debt_to_equity')

    @staticmethod
    def current_ratio(current_assets: pd.Series, current_liabilities: pd.Series) -> pd.Series:
        """流动比率"""
        ratio = current_assets / current_liabilities
        return ratio.rename('current_ratio')

    @staticmethod
    def quick_ratio(quick_assets: pd.Series, current_liabilities: pd.Series) -> pd.Series:
        """速动比率"""
        ratio = quick_assets / current_liabilities
        return ratio.rename('quick_ratio')

    @staticmethod
    def interest_coverage(ebit: pd.Series, interest_expense: pd.Series) -> pd.Series:
        """利息保障倍数"""
        coverage = ebit / interest_expense
        return coverage.rename('interest_coverage')

    @staticmethod
    def free_cash_flow(operating_cf: pd.Series, capex: pd.Series) -> pd.Series:
        """自由现金流"""
        fcf = operating_cf - capex
        return fcf.rename('free_cash_flow')

    @staticmethod
    def fcf_yield(fcf: pd.Series, market_cap: pd.Series) -> pd.Series:
        """自由现金流收益率"""
        yield_ = fcf / market_cap
        return yield_.rename('fcf_yield')

    @staticmethod
    def dividend_yield(dividend: pd.Series, price: pd.Series) -> pd.Series:
        """股息收益率"""
        yield_ = dividend / price
        return yield_.rename('dividend_yield')

    @staticmethod
    def payout_ratio(dividend: pd.Series, net_income: pd.Series) -> pd.Series:
        """股息支付率"""
        ratio = dividend / net_income
        return ratio.rename('payout_ratio')


class FundamentalFactorEngine:
    """基本面因子引擎"""

    def __init__(self):
        self.required_fields = [
            'price', 'earnings', 'book_value', 'revenue', 'shares',
            'cash_flow', 'ev', 'ebitda', 'net_income', 'equity',
            'assets', 'ebit', 'invested_capital', 'gross_profit',
            'operating_income', 'debt', 'current_assets', 'current_liabilities',
            'quick_assets', 'interest_expense', 'operating_cf', 'capex',
            'market_cap', 'dividend'
        ]

    def _validate_input(self, data: pd.DataFrame):
        """验证输入数据"""
        missing_fields = [f for f in self.required_fields if f not in data.columns]
        if missing_fields:
            print(f"Warning: Missing fields for fundamental factors: {missing_fields}")

    def compute_all(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算所有基本面因子"""
        self._validate_input(data)
        results = pd.DataFrame(index=data.index)

        factors = [
            ('pe', lambda df: FundamentalFactors.pe_ratio(df['price'], df['earnings']) if 'price' in df.columns and 'earnings' in df.columns else pd.Series(np.nan, index=df.index)),
            ('pb', lambda df: FundamentalFactors.pb_ratio(df['price'], df['book_value']) if 'price' in df.columns and 'book_value' in df.columns else pd.Series(np.nan, index=df.index)),
            ('ps', lambda df: FundamentalFactors.ps_ratio(df['price'], df['revenue'], df['shares']) if all(f in df.columns for f in ['price', 'revenue', 'shares']) else pd.Series(np.nan, index=df.index)),
            ('pcf', lambda df: FundamentalFactors.pcf_ratio(df['price'], df['cash_flow'], df['shares']) if all(f in df.columns for f in ['price', 'cash_flow', 'shares']) else pd.Series(np.nan, index=df.index)),
            ('ev_ebitda', lambda df: FundamentalFactors.ev_ebitda(df['ev'], df['ebitda']) if 'ev' in df.columns and 'ebitda' in df.columns else pd.Series(np.nan, index=df.index)),
            ('roe', lambda df: FundamentalFactors.roe(df['net_income'], df['equity']) if 'net_income' in df.columns and 'equity' in df.columns else pd.Series(np.nan, index=df.index)),
            ('roa', lambda df: FundamentalFactors.roa(df['net_income'], df['assets']) if 'net_income' in df.columns and 'assets' in df.columns else pd.Series(np.nan, index=df.index)),
            ('roic', lambda df: FundamentalFactors.roic(df['ebit'], df['invested_capital']) if 'ebit' in df.columns and 'invested_capital' in df.columns else pd.Series(np.nan, index=df.index)),
            ('gross_margin', lambda df: FundamentalFactors.gross_margin(df['gross_profit'], df['revenue']) if 'gross_profit' in df.columns and 'revenue' in df.columns else pd.Series(np.nan, index=df.index)),
            ('operating_margin', lambda df: FundamentalFactors.operating_margin(df['operating_income'], df['revenue']) if 'operating_income' in df.columns and 'revenue' in df.columns else pd.Series(np.nan, index=df.index)),
            ('net_margin', lambda df: FundamentalFactors.net_margin(df['net_income'], df['revenue']) if 'net_income' in df.columns and 'revenue' in df.columns else pd.Series(np.nan, index=df.index)),
            ('eps', lambda df: FundamentalFactors.eps(df['net_income'], df['shares']) if 'net_income' in df.columns and 'shares' in df.columns else pd.Series(np.nan, index=df.index)),
            ('revenue_growth', lambda df: FundamentalFactors.revenue_growth(df['revenue']) if 'revenue' in df.columns else pd.Series(np.nan, index=df.index)),
            ('net_income_growth', lambda df: FundamentalFactors.net_income_growth(df['net_income']) if 'net_income' in df.columns else pd.Series(np.nan, index=df.index)),
            ('debt_to_equity', lambda df: FundamentalFactors.debt_to_equity(df['debt'], df['equity']) if 'debt' in df.columns and 'equity' in df.columns else pd.Series(np.nan, index=df.index)),
            ('current_ratio', lambda df: FundamentalFactors.current_ratio(df['current_assets'], df['current_liabilities']) if 'current_assets' in df.columns and 'current_liabilities' in df.columns else pd.Series(np.nan, index=df.index)),
            ('free_cash_flow', lambda df: FundamentalFactors.free_cash_flow(df['operating_cf'], df['capex']) if 'operating_cf' in df.columns and 'capex' in df.columns else pd.Series(np.nan, index=df.index)),
            ('fcf_yield', lambda df: FundamentalFactors.fcf_yield(df['free_cash_flow'](df), df['market_cap']) if 'market_cap' in df.columns else pd.Series(np.nan, index=df.index)),
            ('dividend_yield', lambda df: FundamentalFactors.dividend_yield(df['dividend'], df['price']) if 'dividend' in df.columns and 'price' in df.columns else pd.Series(np.nan, index=df.index)),
        ]

        for name, func in factors:
            try:
                result = func(data)
                results[name] = result
            except Exception as e:
                print(f"Failed to compute {name}: {e}")
                results[name] = np.nan

        return results

    def get_factor_names(self) -> List[str]:
        """获取因子名称"""
        return [
            'pe', 'pb', 'ps', 'pcf', 'ev_ebitda',
            'roe', 'roa', 'roic',
            'gross_margin', 'operating_margin', 'net_margin',
            'eps', 'revenue_growth', 'net_income_growth',
            'debt_to_equity', 'current_ratio',
            'free_cash_flow', 'fcf_yield', 'dividend_yield'
        ]
