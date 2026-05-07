"""
商业化级别注册表管理器

功能:
    - 统一管理实验、数据、因子、模型、回测的注册表
    - 支持审计追踪、策略归因、版本回溯
    - CSV格式存储，便于查询和分析
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import json


class RegistryManager:
    def __init__(self, registry_dir: Path):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_registry = self.registry_dir / "experiment_registry.csv"
        self.data_registry = self.registry_dir / "data_registry.csv"
        self.factor_registry = self.registry_dir / "factor_registry.csv"
        self.model_registry = self.registry_dir / "model_registry.csv"
        self.backtest_registry = self.registry_dir / "backtest_registry.csv"
        self.trade_registry = self.registry_dir / "trade_registry.csv"

        self._init_registries()

    def _init_registries(self):
        """初始化注册表"""
        if not self.experiment_registry.exists():
            df = pd.DataFrame(columns=[
                "run_id", "version", "module", "name", "created_at",
                "input_path", "output_path", "config_path", "metrics_path",
                "status", "notes", "git_commit", "python_version",
                "start_time", "end_time", "duration_seconds",
                "environment", "parameters"
            ])
            df.to_csv(self.experiment_registry, index=False, encoding='utf-8')

        if not self.data_registry.exists():
            df = pd.DataFrame(columns=[
                "run_id", "version", "data_name", "data_type", "source",
                "start_date", "end_date", "symbol_count", "total_rows",
                "missing_rate", "anomaly_rate", "suspended_days",
                "output_path", "created_at", "status", "notes",
                "git_commit", "source_version"
            ])
            df.to_csv(self.data_registry, index=False, encoding='utf-8')

        if not self.factor_registry.exists():
            df = pd.DataFrame(columns=[
                "run_id", "version", "factor_name", "factor_type",
                "input_data_version", "output_path",
                "ic_mean", "ic_std", "ir", "ic_t_stat", "ic_p_value",
                "missing_rate", "correlation_with_benchmark",
                "universe_count", "time_period",
                "created_at", "status", "notes", "git_commit", "parameters"
            ])
            df.to_csv(self.factor_registry, index=False, encoding='utf-8')

        if not self.model_registry.exists():
            df = pd.DataFrame(columns=[
                "run_id", "version", "model_name", "model_type",
                "input_data_version", "input_factor_version",
                "output_path", "feature_list", "label_definition",
                "train_start_date", "train_end_date",
                "val_start_date", "val_end_date",
                "train_auc", "val_auc", "train_accuracy", "val_accuracy",
                "train_f1", "val_f1", "train_logloss", "val_logloss",
                "feature_importance_path", "parameter_hash",
                "created_at", "status", "notes", "git_commit", "parameters"
            ])
            df.to_csv(self.model_registry, index=False, encoding='utf-8')

        if not self.backtest_registry.exists():
            df = pd.DataFrame(columns=[
                "run_id", "version", "strategy_name", "strategy_version",
                "model_version", "factor_version", "data_version",
                "initial_capital", "commission_rate", "slippage_rate",
                "start_date", "end_date", "time_period_days",
                "total_return", "annual_return", "sharpe_ratio",
                "sortino_ratio", "max_drawdown", "max_drawdown_days",
                "annual_volatility", "calmar_ratio", "profit_factor",
                "win_rate", "num_trades", "avg_trade_profit",
                "max_consecutive_wins", "max_consecutive_losses",
                "turnover_rate", "information_ratio",
                "output_path", "created_at", "status", "notes",
                "git_commit", "parameters"
            ])
            df.to_csv(self.backtest_registry, index=False, encoding='utf-8')

        if not self.trade_registry.exists():
            df = pd.DataFrame(columns=[
                "trade_id", "run_id", "backtest_version",
                "symbol", "action", "order_time", "fill_time",
                "order_price", "fill_price", "quantity",
                "commission", "slippage", "pnl", "position_size",
                "portfolio_value", "risk_exposure", "created_at"
            ])
            df.to_csv(self.trade_registry, index=False, encoding='utf-8')

    def _append_to_registry(self, registry_file: Path, data: Dict[str, Any]):
        """追加记录到注册表"""
        df = pd.read_csv(registry_file)
        new_row = pd.DataFrame([data])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(registry_file, index=False, encoding='utf-8')

    def _update_registry(self, registry_file: Path, run_id: str, updates: Dict[str, Any]):
        """更新注册表中的记录"""
        df = pd.read_csv(registry_file)
        mask = df['run_id'] == run_id
        if mask.any():
            for key, value in updates.items():
                if key in df.columns:
                    df.loc[mask, key] = value
            df.to_csv(registry_file, index=False, encoding='utf-8')

    def register_experiment(
        self,
        run_id: str,
        module: str,
        name: str,
        version: str,
        status: str = "running",
        notes: str = "",
        git_commit: str = "",
        python_version: str = "",
        environment: str = "development",
        parameters: Optional[Dict] = None,
        input_path: str = "",
        output_path: str = "",
        config_path: str = "",
        metrics_path: str = "",
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        duration_seconds: Optional[float] = None
    ):
        """注册实验"""
        data = {
            "run_id": run_id,
            "version": version,
            "module": module,
            "name": name,
            "created_at": datetime.now().isoformat(),
            "input_path": input_path,
            "output_path": output_path,
            "config_path": config_path,
            "metrics_path": metrics_path,
            "status": status,
            "notes": notes,
            "git_commit": git_commit,
            "python_version": python_version,
            "environment": environment,
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": duration_seconds,
            "parameters": json.dumps(parameters) if parameters else ""
        }
        self._append_to_registry(self.experiment_registry, data)

    def register_data(
        self,
        run_id: str,
        data_name: str,
        version: str,
        data_type: str,
        source: str,
        start_date: str,
        end_date: str,
        symbol_count: int,
        total_rows: int,
        missing_rate: float,
        anomaly_rate: float,
        suspended_days: int = 0,
        output_path: str = "",
        status: str = "completed",
        notes: str = "",
        git_commit: str = "",
        source_version: str = ""
    ):
        """注册数据"""
        data = {
            "run_id": run_id,
            "version": version,
            "data_name": data_name,
            "data_type": data_type,
            "source": source,
            "start_date": start_date,
            "end_date": end_date,
            "symbol_count": symbol_count,
            "total_rows": total_rows,
            "missing_rate": missing_rate,
            "anomaly_rate": anomaly_rate,
            "suspended_days": suspended_days,
            "output_path": output_path,
            "created_at": datetime.now().isoformat(),
            "status": status,
            "notes": notes,
            "git_commit": git_commit,
            "source_version": source_version
        }
        self._append_to_registry(self.data_registry, data)

    def register_factor(
        self,
        run_id: str,
        factor_name: str,
        version: str,
        factor_type: str,
        input_data_version: str,
        output_path: str,
        ic_mean: Optional[float] = None,
        ic_std: Optional[float] = None,
        ir: Optional[float] = None,
        ic_t_stat: Optional[float] = None,
        ic_p_value: Optional[float] = None,
        missing_rate: Optional[float] = None,
        correlation_with_benchmark: Optional[float] = None,
        universe_count: Optional[int] = None,
        time_period: Optional[str] = None,
        status: str = "completed",
        notes: str = "",
        git_commit: str = "",
        parameters: Optional[Dict] = None
    ):
        """注册因子"""
        data = {
            "run_id": run_id,
            "version": version,
            "factor_name": factor_name,
            "factor_type": factor_type,
            "input_data_version": input_data_version,
            "output_path": output_path,
            "ic_mean": ic_mean if ic_mean else "",
            "ic_std": ic_std if ic_std else "",
            "ir": ir if ir else "",
            "ic_t_stat": ic_t_stat if ic_t_stat else "",
            "ic_p_value": ic_p_value if ic_p_value else "",
            "missing_rate": missing_rate if missing_rate else "",
            "correlation_with_benchmark": correlation_with_benchmark if correlation_with_benchmark else "",
            "universe_count": universe_count if universe_count else "",
            "time_period": time_period if time_period else "",
            "created_at": datetime.now().isoformat(),
            "status": status,
            "notes": notes,
            "git_commit": git_commit,
            "parameters": json.dumps(parameters) if parameters else ""
        }
        self._append_to_registry(self.factor_registry, data)

    def register_model(
        self,
        run_id: str,
        model_name: str,
        version: str,
        model_type: str,
        input_data_version: str,
        input_factor_version: str,
        output_path: str,
        feature_list: List[str],
        label_definition: str,
        train_start_date: str,
        train_end_date: str,
        val_start_date: Optional[str] = None,
        val_end_date: Optional[str] = None,
        train_auc: Optional[float] = None,
        val_auc: Optional[float] = None,
        train_accuracy: Optional[float] = None,
        val_accuracy: Optional[float] = None,
        train_f1: Optional[float] = None,
        val_f1: Optional[float] = None,
        train_logloss: Optional[float] = None,
        val_logloss: Optional[float] = None,
        feature_importance_path: str = "",
        parameter_hash: str = "",
        status: str = "completed",
        notes: str = "",
        git_commit: str = "",
        parameters: Optional[Dict] = None
    ):
        """注册模型"""
        data = {
            "run_id": run_id,
            "version": version,
            "model_name": model_name,
            "model_type": model_type,
            "input_data_version": input_data_version,
            "input_factor_version": input_factor_version,
            "output_path": output_path,
            "feature_list": json.dumps(feature_list) if feature_list else "",
            "label_definition": label_definition,
            "train_start_date": train_start_date,
            "train_end_date": train_end_date,
            "val_start_date": val_start_date if val_start_date else "",
            "val_end_date": val_end_date if val_end_date else "",
            "train_auc": train_auc if train_auc else "",
            "val_auc": val_auc if val_auc else "",
            "train_accuracy": train_accuracy if train_accuracy else "",
            "val_accuracy": val_accuracy if val_accuracy else "",
            "train_f1": train_f1 if train_f1 else "",
            "val_f1": val_f1 if val_f1 else "",
            "train_logloss": train_logloss if train_logloss else "",
            "val_logloss": val_logloss if val_logloss else "",
            "feature_importance_path": feature_importance_path,
            "parameter_hash": parameter_hash,
            "created_at": datetime.now().isoformat(),
            "status": status,
            "notes": notes,
            "git_commit": git_commit,
            "parameters": json.dumps(parameters) if parameters else ""
        }
        self._append_to_registry(self.model_registry, data)

    def register_backtest(
        self,
        run_id: str,
        strategy_name: str,
        version: str,
        strategy_version: str = "",
        model_version: str = "",
        factor_version: str = "",
        data_version: str = "",
        initial_capital: float = 1000000.0,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        start_date: str = "",
        end_date: str = "",
        time_period_days: Optional[int] = None,
        total_return: Optional[float] = None,
        annual_return: Optional[float] = None,
        sharpe_ratio: Optional[float] = None,
        sortino_ratio: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        max_drawdown_days: Optional[int] = None,
        annual_volatility: Optional[float] = None,
        calmar_ratio: Optional[float] = None,
        profit_factor: Optional[float] = None,
        win_rate: Optional[float] = None,
        num_trades: Optional[int] = None,
        avg_trade_profit: Optional[float] = None,
        max_consecutive_wins: Optional[int] = None,
        max_consecutive_losses: Optional[int] = None,
        turnover_rate: Optional[float] = None,
        information_ratio: Optional[float] = None,
        output_path: str = "",
        status: str = "completed",
        notes: str = "",
        git_commit: str = "",
        parameters: Optional[Dict] = None
    ):
        """注册回测"""
        data = {
            "run_id": run_id,
            "version": version,
            "strategy_name": strategy_name,
            "strategy_version": strategy_version,
            "model_version": model_version,
            "factor_version": factor_version,
            "data_version": data_version,
            "initial_capital": initial_capital,
            "commission_rate": commission_rate,
            "slippage_rate": slippage_rate,
            "start_date": start_date,
            "end_date": end_date,
            "time_period_days": time_period_days if time_period_days else "",
            "total_return": total_return if total_return else "",
            "annual_return": annual_return if annual_return else "",
            "sharpe_ratio": sharpe_ratio if sharpe_ratio else "",
            "sortino_ratio": sortino_ratio if sortino_ratio else "",
            "max_drawdown": max_drawdown if max_drawdown else "",
            "max_drawdown_days": max_drawdown_days if max_drawdown_days else "",
            "annual_volatility": annual_volatility if annual_volatility else "",
            "calmar_ratio": calmar_ratio if calmar_ratio else "",
            "profit_factor": profit_factor if profit_factor else "",
            "win_rate": win_rate if win_rate else "",
            "num_trades": num_trades if num_trades else "",
            "avg_trade_profit": avg_trade_profit if avg_trade_profit else "",
            "max_consecutive_wins": max_consecutive_wins if max_consecutive_wins else "",
            "max_consecutive_losses": max_consecutive_losses if max_consecutive_losses else "",
            "turnover_rate": turnover_rate if turnover_rate else "",
            "information_ratio": information_ratio if information_ratio else "",
            "output_path": output_path,
            "created_at": datetime.now().isoformat(),
            "status": status,
            "notes": notes,
            "git_commit": git_commit,
            "parameters": json.dumps(parameters) if parameters else ""
        }
        self._append_to_registry(self.backtest_registry, data)

    def register_trade(
        self,
        trade_id: str,
        run_id: str,
        backtest_version: str,
        symbol: str,
        action: str,
        order_time: str,
        fill_time: str,
        order_price: float,
        fill_price: float,
        quantity: int,
        commission: float,
        slippage: float,
        pnl: float,
        position_size: float,
        portfolio_value: float,
        risk_exposure: float
    ):
        """注册交易"""
        data = {
            "trade_id": trade_id,
            "run_id": run_id,
            "backtest_version": backtest_version,
            "symbol": symbol,
            "action": action,
            "order_time": order_time,
            "fill_time": fill_time,
            "order_price": order_price,
            "fill_price": fill_price,
            "quantity": quantity,
            "commission": commission,
            "slippage": slippage,
            "pnl": pnl,
            "position_size": position_size,
            "portfolio_value": portfolio_value,
            "risk_exposure": risk_exposure,
            "created_at": datetime.now().isoformat()
        }
        self._append_to_registry(self.trade_registry, data)

    def update_status(self, run_id: str, status: str, registry_type: str = "experiment"):
        """更新运行状态"""
        registry_map = {
            "experiment": self.experiment_registry,
            "data": self.data_registry,
            "factor": self.factor_registry,
            "model": self.model_registry,
            "backtest": self.backtest_registry
        }
        if registry_type in registry_map:
            self._update_registry(registry_map[registry_type], run_id, {"status": status})

    def get_latest_runs(self, registry_type: str = "experiment", limit: int = 10) -> pd.DataFrame:
        """获取最新的运行记录"""
        registry_map = {
            "experiment": self.experiment_registry,
            "data": self.data_registry,
            "factor": self.factor_registry,
            "model": self.model_registry,
            "backtest": self.backtest_registry,
            "trade": self.trade_registry
        }
        if registry_type not in registry_map:
            return pd.DataFrame()
        df = pd.read_csv(registry_map[registry_type])
        if len(df) > 0:
            df = df.sort_values('created_at', ascending=False).head(limit)
        return df

    def get_run_by_id(self, run_id: str, registry_type: str = "experiment") -> Optional[Dict]:
        """根据 run_id 获取运行记录"""
        registry_map = {
            "experiment": self.experiment_registry,
            "data": self.data_registry,
            "factor": self.factor_registry,
            "model": self.model_registry,
            "backtest": self.backtest_registry,
            "trade": self.trade_registry
        }
        if registry_type not in registry_map:
            return None
        df = pd.read_csv(registry_map[registry_type])
        mask = df['run_id'] == run_id
        if mask.any():
            return df[mask].iloc[0].to_dict()
        return None

    def compare_runs(self, run_ids: List[str], registry_type: str = "backtest") -> pd.DataFrame:
        """对比多个运行的结果"""
        registry_map = {
            "experiment": self.experiment_registry,
            "data": self.data_registry,
            "factor": self.factor_registry,
            "model": self.model_registry,
            "backtest": self.backtest_registry
        }
        if registry_type not in registry_map:
            return pd.DataFrame()
        df = pd.read_csv(registry_map[registry_type])
        result_df = df[df['run_id'].isin(run_ids)]
        return result_df

    def list_versions(self, registry_type: str = "model") -> List[str]:
        """列出所有版本"""
        df = self.get_latest_runs(registry_type, limit=1000)
        if 'version' in df.columns:
            return df['version'].dropna().unique().tolist()
        return []

    def get_metrics_summary(self, registry_type: str = "backtest") -> Dict[str, Any]:
        """获取指标汇总"""
        df = self.get_latest_runs(registry_type, limit=100)
        summary = {
            "total_runs": len(df),
            "by_status": df['status'].value_counts().to_dict() if 'status' in df.columns else {},
            "latest_run_id": df['run_id'].iloc[0] if len(df) > 0 else None,
            "latest_created": df['created_at'].iloc[0] if len(df) > 0 else None
        }
        if registry_type == "backtest":
            for col in ['total_return', 'sharpe_ratio', 'max_drawdown', 'annual_return']:
                if col in df.columns:
                    valid_vals = df[col].dropna()
                    if len(valid_vals) > 0:
                        summary[f"avg_{col}"] = valid_vals.mean()
                        summary[f"max_{col}"] = valid_vals.max()
                        summary[f"min_{col}"] = valid_vals.min()
        return summary

    def get_data_quality_report(self, run_id: str) -> Optional[Dict]:
        """获取数据质量报告"""
        df = pd.read_csv(self.data_registry)
        mask = df['run_id'] == run_id
        if mask.any():
            row = df[mask].iloc[0]
            return {
                "data_name": row['data_name'],
                "version": row['version'],
                "source": row['source'],
                "date_range": f"{row['start_date']} to {row['end_date']}",
                "symbol_count": row['symbol_count'],
                "total_rows": row['total_rows'],
                "missing_rate": row['missing_rate'],
                "anomaly_rate": row['anomaly_rate'],
                "suspended_days": row['suspended_days'],
                "status": row['status']
            }
        return None

    def get_factor_analysis_report(self, factor_name: str) -> Optional[pd.DataFrame]:
        """获取因子分析报告"""
        df = pd.read_csv(self.factor_registry)
        mask = df['factor_name'] == factor_name
        if mask.any():
            return df[mask]
        return None


_global_registry: Optional[RegistryManager] = None


def get_registry(registry_dir: Optional[Path] = None) -> RegistryManager:
    """获取全局注册表实例"""
    global _global_registry
    if _global_registry is None:
        if registry_dir is None:
            registry_dir = Path("./registry")
        _global_registry = RegistryManager(registry_dir)
    return _global_registry
