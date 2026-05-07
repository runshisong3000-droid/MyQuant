"""
结构化日志模块

功能:
    - JSON 格式日志
    - 支持多种日志级别: info, warning, error, metric, artifact
    - 同时输出到控制台和文件
    - 集成 run_id 追踪
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime
from enum import Enum
import traceback


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    METRIC = "METRIC"
    ARTIFACT = "ARTIFACT"


class StructuredLogger:
    def __init__(
        self,
        module_name: str,
        run_dir: Optional[Path] = None,
        run_id: Optional[str] = None,
        log_file: Optional[str] = "run_log.json",
        console_output: bool = True
    ):
        self.module_name = module_name
        self.run_id = run_id
        self.run_dir = Path(run_dir) if run_dir else None
        self.log_file_name = log_file
        self.console_output = console_output

        if self.run_dir:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = self.run_dir / self.log_file_name
            self._init_log_file()

    def _init_log_file(self):
        """初始化日志文件"""
        if self.log_file and not self.log_file.exists():
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write("")

    def _format_log(
        self,
        level: LogLevel,
        message: str,
        extra: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """格式化日志条目"""
        log_entry = {
            "timestamp": datetime.now().isoformat(timespec='milliseconds'),
            "level": level.value,
            "module": self.module_name,
            "run_id": self.run_id,
            "message": message
        }

        if extra:
            log_entry["extra"] = extra

        return log_entry

    def _write_log(self, log_entry: Dict[str, Any]):
        """写入日志"""
        log_line = json.dumps(log_entry, ensure_ascii=False)

        if self.console_output:
            color_map = {
                "DEBUG": "\033[36m",
                "INFO": "\033[32m",
                "WARNING": "\033[33m",
                "ERROR": "\033[31m",
                "METRIC": "\033[35m",
                "ARTIFACT": "\033[34m"
            }
            reset = "\033[0m"
            color = color_map.get(log_entry["level"], "")

            if sys.stdout.encoding != 'utf-8':
                message = log_entry["message"]
                if isinstance(message, str):
                    message = message.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                log_entry["message"] = message

            print(f"{color}{log_line}{reset}")
        else:
            print(log_line)

        if self.log_file:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(log_line + "\n")
            except Exception as e:
                print(f"Failed to write log: {e}", file=sys.stderr)

    def info(self, message: str, **kwargs):
        """信息日志"""
        extra = kwargs if kwargs else None
        log_entry = self._format_log(LogLevel.INFO, message, extra)
        self._write_log(log_entry)

    def warning(self, message: str, **kwargs):
        """警告日志"""
        extra = kwargs if kwargs else None
        log_entry = self._format_log(LogLevel.WARNING, message, extra)
        self._write_log(log_entry)

    def error(self, message: str, exc_info: Optional[Exception] = None, **kwargs):
        """错误日志"""
        extra = kwargs if kwargs else {}

        if exc_info:
            extra["exception"] = {
                "type": type(exc_info).__name__,
                "message": str(exc_info),
                "traceback": traceback.format_exc()
            }

        log_entry = self._format_log(LogLevel.ERROR, message, extra if extra else None)
        self._write_log(log_entry)

    def metric(self, name: str, value: Any, **kwargs):
        """指标日志"""
        extra = {
            "metric_name": name,
            "metric_value": value,
            **kwargs
        }
        log_entry = self._format_log(LogLevel.METRIC, f"Metric: {name} = {value}", extra)
        self._write_log(log_entry)

    def artifact(
        self,
        name: str,
        path: str,
        artifact_type: str = "file",
        **kwargs
    ):
        """产物日志"""
        extra = {
            "artifact_name": name,
            "artifact_path": path,
            "artifact_type": artifact_type,
            **kwargs
        }
        log_entry = self._format_log(
            LogLevel.ARTIFACT,
            f"Artifact: {name} ({artifact_type})",
            extra
        )
        self._write_log(log_entry)

    def debug(self, message: str, **kwargs):
        """调试日志"""
        extra = kwargs if kwargs else None
        log_entry = self._format_log(LogLevel.DEBUG, message, extra)
        self._write_log(log_entry)

    def log_event(self, event: str, **kwargs):
        """通用事件日志"""
        extra = kwargs if kwargs else None
        log_entry = self._format_log(LogLevel.INFO, f"Event: {event}", extra)
        self._write_log(log_entry)

    def log_dict(self, data: Dict[str, Any], prefix: str = ""):
        """记录字典数据"""
        for key, value in data.items():
            message = f"{prefix}{key}" if prefix else str(key)
            if isinstance(value, (dict, list)):
                self.info(f"{message}: {json.dumps(value, ensure_ascii=False)}")
            else:
                self.info(f"{message}: {value}")

    def set_run_id(self, run_id: str):
        """设置 run_id"""
        self.run_id = run_id

    def set_run_dir(self, run_dir: Path):
        """设置运行目录"""
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.run_dir / self.log_file_name
        self._init_log_file()


class LoggerManager:
    _instance = None
    _loggers: Dict[str, StructuredLogger] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_logger(
        self,
        module_name: str,
        run_dir: Optional[Path] = None,
        run_id: Optional[str] = None
    ) -> StructuredLogger:
        """获取或创建 logger"""
        key = f"{module_name}_{run_id}" if run_id else module_name

        if key not in self._loggers:
            self._loggers[key] = StructuredLogger(
                module_name=module_name,
                run_dir=run_dir,
                run_id=run_id
            )

        return self._loggers[key]

    def set_run_id_for_all(self, run_id: str):
        """为所有 logger 设置 run_id"""
        for logger in self._loggers.values():
            logger.set_run_id(run_id)

    def clear(self):
        """清除所有 logger"""
        self._loggers.clear()


_global_logger_manager = LoggerManager()


def get_logger(
    module_name: str,
    run_dir: Optional[Path] = None,
    run_id: Optional[str] = None
) -> StructuredLogger:
    """获取全局 logger 实例"""
    return _global_logger_manager.get_logger(module_name, run_dir, run_id)


def set_run_id_for_all(run_id: str):
    """为所有 logger 设置 run_id"""
    _global_logger_manager.set_run_id_for_all(run_id)
