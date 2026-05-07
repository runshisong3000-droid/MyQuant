"""
Date Utilities - 日期工具
"""

from datetime import datetime, timedelta


def calculate_start_date(end_date: datetime, history_months: int = 12) -> datetime:
    """
    计算开始日期，正确处理跨年

    Args:
        end_date: 结束日期
        history_months: 历史月数

    Returns:
        开始日期
    """
    total_days = history_months * 30
    start_date = end_date - timedelta(days=total_days)

    return start_date


def calculate_start_date_str(end_date: datetime, history_months: int = 12) -> str:
    """
    计算开始日期，返回YYYYMMDD字符串

    Args:
        end_date: 结束日期
        history_months: 历史月数

    Returns:
        YYYYMMDD格式字符串
    """
    start_date = calculate_start_date(end_date, history_months)
    return start_date.strftime('%Y%m%d')


def subtract_months(dt: datetime, months: int) -> datetime:
    """
    安全地减去指定月数

    Args:
        dt: 原始日期
        months: 月数

    Returns:
        新日期
    """
    total_days = months * 30
    return dt - timedelta(days=total_days)
