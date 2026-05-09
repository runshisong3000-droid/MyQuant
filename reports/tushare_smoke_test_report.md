# Tushare Provider Smoke Test Report

## Test Summary

| Item | Value |
|------|-------|
| TUSHARE_TOKEN Present | True |
| Token Printed | False |
| Interfaces Tested | 7 |
| Success Count | 5 |
| Failed Count | 0 |
| Can Use For Research | True |
| Can Use For Live Trading | False |

## Interfaces Tested

- stock_basic
- trade_cal
- daily
- adj_factor
- daily_basic
- stk_limit
- suspend_d

## Standardized Fields

- adj_factor
- amount
- close
- date
- down_limit
- high
- industry
- is_trading_day
- list_date
- low
- market
- open
- pct_change
- stock
- stock_name
- up_limit
- volume

## Failed Reasons

- daily_basic: Permission denied or no data
- suspend_d: Permission denied or no data