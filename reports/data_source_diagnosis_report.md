# Data Source Diagnosis Report

Generated: 2026-05-09 10:21:03

## Summary

| Status | Count |
|--------|-------|
| OK | 1 |
| WARN | 1 |
| FAIL | 4 |

## Detailed Steps

### Environment Info
- Status: OK
- Elapsed: 0.95s
- Details: Python=3.11.9, akshare=1.18.60

### Direct Requests
- Status: FAIL
- Elapsed: 0.51s
- Error Type: ConnectionError
- Error: ('Connection aborted.', RemoteDisconnected('Remote

### AkShare Single Stock
- Status: FAIL
- Elapsed: 0.49s
- Error Type: ConnectionError
- Error: ('Connection aborted.', RemoteDisconnected('Remote

### Eastmoney Direct Provider
- Status: FAIL
- Elapsed: 2.44s
- Error Type: DataFetchError
- Error: Request error: ('Connection aborted.', RemoteDisco

### DataSourceManager (5 stocks)
- Status: FAIL
- Elapsed: 716.43s
- Error Type: AttributeError
- Error: 'NoneType' object has no attribute 'get'

### DataSourceManager (20 stocks)
- Status: WARN
- Elapsed: 0.00s
- Details: Skipped to save time
