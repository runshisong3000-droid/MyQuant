"""
运行管理器模块

功能:
    - 生成唯一的 run_id
    - 创建运行目录结构
    - 管理运行生命周期
    - 集成日志和注册表
    - Git 信息追踪
"""

import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import subprocess
import sys


class RunManager:
    def __init__(
        self,
        module_name: str,
        base_dir: Optional[Path] = None,
        registry_dir: Optional[Path] = None,
        logs_dir: Optional[Path] = None,
        git_enabled: bool = True
    ):
        self.module_name = module_name
        self.base_dir = Path(base_dir) if base_dir else Path("./")
        self.git_enabled = git_enabled

        self.logs_dir = Path(logs_dir) if logs_dir else self.base_dir / "logs"
        self.runs_dir = self.logs_dir / "runs"
        self.registry_dir = Path(registry_dir) if registry_dir else self.base_dir / "registry"

        self.runs_dir.mkdir(parents=True, exist_ok=True)

        self.run_id: Optional[str] = None
        self.run_dir: Optional[Path] = None
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.status: str = "initialized"
        self.git_commit: str = ""
        self.logger = None
        self.config_snapshot_path: Optional[Path] = None

    def _generate_run_id(self) -> str:
        """生成 run_id"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = uuid.uuid4().hex[:6]
        return f"{timestamp}_{self.module_name}_{short_uuid}"

    def _get_git_commit(self) -> str:
        """获取当前 Git commit hash"""
        if not self.git_enabled:
            return ""

        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()[:8]
        except Exception:
            pass

        return ""

    def _get_git_branch(self) -> str:
        """获取当前 Git 分支"""
        if not self.git_enabled:
            return ""

        try:
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

        return ""

    def _create_run_dir(self) -> Path:
        """创建运行目录"""
        run_dir = self.runs_dir / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        for subdir in ["config_snapshot", "data", "factors", "models", "backtests", "reports"]:
            (run_dir / subdir).mkdir(exist_ok=True)

        return run_dir

    def _init_logger(self):
        """初始化日志器"""
        from .logger import get_logger

        self.logger = get_logger(
            module_name=self.module_name,
            run_dir=self.run_dir,
            run_id=self.run_id
        )

    def _save_run_info(self):
        """保存运行信息"""
        run_info = {
            "run_id": self.run_id,
            "module_name": self.module_name,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status,
            "git_commit": self.git_commit,
            "git_branch": self._get_git_branch(),
            "python_version": sys.version.split()[0],
            "run_dir": str(self.run_dir.absolute()) if self.run_dir else None
        }

        if self.run_dir:
            info_file = self.run_dir / "run_info.json"
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(run_info, f, indent=2, ensure_ascii=False)

    def start_run(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_dir: Optional[Path] = None,
        name: Optional[str] = None,
        notes: str = ""
    ) -> str:
        """
        开始一次运行

        Args:
            config: 配置字典
            config_dir: 配置目录路径
            name: 运行名称
            notes: 备注

        Returns:
            run_id
        """
        self.run_id = self._generate_run_id()
        self.start_time = datetime.now()
        self.status = "running"
        self.git_commit = self._get_git_commit()

        self.run_dir = self._create_run_dir()
        self._init_logger()
        self._save_run_info()

        from .config_snapshot import ConfigSnapshot
        from .registry import get_registry

        config_snapshot = ConfigSnapshot(self.run_dir)

        if config_dir and Path(config_dir).exists():
            self.config_snapshot_path = config_snapshot.save_config_dir(config_dir)
        elif config:
            self.config_snapshot_path = config_snapshot.save_config_dict(config)

        if self.logger:
            self.logger.info(f"Run started: {self.run_id}")
            self.logger.info(f"Module: {self.module_name}")
            self.logger.info(f"Git commit: {self.git_commit}")
            self.logger.info(f"Python version: {sys.version.split()[0]}")

        registry = get_registry(self.registry_dir)
        registry.register_experiment(
            run_id=self.run_id,
            module=self.module_name,
            name=name or self.module_name,
            version=self.run_id,
            status=self.status,
            notes=notes,
            git_commit=self.git_commit,
            parameters=config,
            config_path=str(self.config_snapshot_path) if self.config_snapshot_path else "",
            output_path=str(self.run_dir.absolute()) if self.run_dir else ""
        )

        return self.run_id

    def end_run(
        self,
        status: str = "completed",
        metrics: Optional[Dict[str, Any]] = None,
        artifacts: Optional[Dict[str, str]] = None,
        notes: str = ""
    ):
        """
        结束一次运行

        Args:
            status: 运行状态 (completed, failed, stopped)
            metrics: 运行指标
            artifacts: 运行产物
            notes: 备注
        """
        if not self.run_id:
            raise RuntimeError("No active run. Call start_run() first.")

        self.end_time = datetime.now()
        self.status = status

        if self.logger:
            self.logger.info(f"Run ended: {self.run_id}, status: {status}")

        if metrics and self.logger:
            for key, value in metrics.items():
                self.logger.metric(key, value)

        if artifacts and self.logger:
            for name, path in artifacts.items():
                self.logger.artifact(name, path)

        if metrics and self.run_dir:
            metrics_file = self.run_dir / "metrics.json"
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)

        if artifacts and self.run_dir:
            artifacts_file = self.run_dir / "artifacts.json"
            with open(artifacts_file, 'w', encoding='utf-8') as f:
                json.dump(artifacts, f, indent=2, ensure_ascii=False)

        self._save_run_info()

        from .registry import get_registry
        registry = get_registry(self.registry_dir)
        registry.update_status(self.run_id, status)

    def fail_run(self, error: Exception, notes: str = ""):
        """
        标记运行失败

        Args:
            error: 异常对象
            notes: 备注
        """
        if not self.run_id:
            raise RuntimeError("No active run. Call start_run() first.")

        self.end_time = datetime.now()
        self.status = "failed"

        if self.logger:
            self.logger.error(f"Run failed: {self.run_id}", exc_info=error)

        self._save_run_info()

        from .registry import get_registry
        registry = get_registry(self.registry_dir)
        registry.update_status(self.run_id, "failed")

    def save_artifact(self, name: str, data: Any, artifact_type: str = "json"):
        """
        保存运行产物

        Args:
            name: 产物名称
            data: 产物数据
            artifact_type: 产物类型
        """
        if not self.run_dir:
            raise RuntimeError("No active run. Call start_run() first.")

        artifact_dir = self.run_dir / "artifacts"
        artifact_dir.mkdir(exist_ok=True)

        if artifact_type == "json":
            file_path = artifact_dir / f"{name}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        elif artifact_type == "csv":
            import pandas as pd
            file_path = artifact_dir / f"{name}.csv"
            if isinstance(data, pd.DataFrame):
                data.to_csv(file_path, index=False)
            elif isinstance(data, str) and Path(data).suffix == '.csv':
                import shutil
                shutil.copy(data, file_path)
            else:
                pd.DataFrame([data]).to_csv(file_path, index=False)
        else:
            file_path = artifact_dir / name
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(str(data))

        if self.logger:
            self.logger.artifact(name, str(file_path.absolute()), artifact_type)

    def get_run_info(self) -> Dict[str, Any]:
        """获取运行信息"""
        if not self.run_id:
            return {}

        return {
            "run_id": self.run_id,
            "module_name": self.module_name,
            "run_dir": str(self.run_dir.absolute()) if self.run_dir else None,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status,
            "git_commit": self.git_commit,
            "duration_seconds": (
                (self.end_time - self.start_time).total_seconds()
                if self.end_time and self.start_time else None
            )
        }

    def __enter__(self):
        """上下文管理器入口"""
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        if exc_type is not None:
            self.fail_run(exc_val)
        else:
            self.end_run()
        return False


def start_run(
    module_name: str,
    config: Optional[Dict[str, Any]] = None,
    config_dir: Optional[Path] = None,
    base_dir: Optional[Path] = None,
    name: Optional[str] = None,
    notes: str = ""
) -> RunManager:
    """
    快捷函数：开始一次运行

    Args:
        module_name: 模块名称
        config: 配置字典
        config_dir: 配置目录路径
        base_dir: 基础目录
        name: 运行名称
        notes: 备注

    Returns:
        RunManager 实例
    """
    manager = RunManager(
        module_name=module_name,
        base_dir=base_dir
    )
    manager.start_run(
        config=config,
        config_dir=config_dir,
        name=name,
        notes=notes
    )
    return manager


def end_run(manager: RunManager, status: str = "completed", metrics: Optional[Dict] = None):
    """
    快捷函数：结束一次运行

    Args:
        manager: RunManager 实例
        status: 运行状态
        metrics: 运行指标
    """
    manager.end_run(status=status, metrics=metrics)
