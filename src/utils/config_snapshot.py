"""
配置快照模块

功能:
    - 复制当前 config/*.yaml 到日志目录
    - 保存完整参数，保证回测和训练可复现
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import shutil


class ConfigSnapshot:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.snapshot_dir = self.run_dir / "config_snapshot"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def save_config_dir(self, config_dir: Path, name: str = "config") -> Path:
        """
        保存整个配置目录的快照

        Args:
            config_dir: 配置目录路径
            name: 快照名称

        Returns:
            快照文件路径
        """
        config_dir = Path(config_dir)

        snapshot_file = self.snapshot_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        snapshot_data = {}

        if config_dir.exists():
            for yaml_file in config_dir.glob("*.yaml"):
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    snapshot_data[yaml_file.stem] = yaml.safe_load(f)

            with open(snapshot_file, 'w', encoding='utf-8') as f:
                yaml.dump(snapshot_data, f, default_flow_style=False, allow_unicode=True)

        metadata_file = self.snapshot_dir / f"{name}_metadata.json"
        metadata = {
            "source_dir": str(config_dir.absolute()),
            "snapshot_file": str(snapshot_file.absolute()),
            "timestamp": datetime.now().isoformat(),
            "files_included": [str(f.relative_to(config_dir)) for f in config_dir.glob("*.yaml")]
        }
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return snapshot_file

    def save_config_dict(self, config: Dict[str, Any], name: str = "config") -> Path:
        """
        保存配置字典的快照

        Args:
            config: 配置字典
            name: 快照名称

        Returns:
            快照文件路径
        """
        snapshot_file = self.snapshot_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"

        with open(snapshot_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        return snapshot_file

    def save_json_config(self, config: Dict[str, Any], name: str = "config") -> Path:
        """
        保存JSON格式的配置快照

        Args:
            config: 配置字典
            name: 快照名称

        Returns:
            快照文件路径
        """
        snapshot_file = self.snapshot_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(snapshot_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        return snapshot_file

    def load_snapshot(self, snapshot_file: Path) -> Dict[str, Any]:
        """
        加载配置快照

        Args:
            snapshot_file: 快照文件路径

        Returns:
            配置字典
        """
        snapshot_file = Path(snapshot_file)

        if snapshot_file.suffix == '.yaml':
            with open(snapshot_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        elif snapshot_file.suffix == '.json':
            with open(snapshot_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {snapshot_file.suffix}")

    @staticmethod
    def get_latest_snapshot(run_dir: Path, name: str = "config") -> Optional[Path]:
        """
        获取最新的配置快照

        Args:
            run_dir: 运行目录
            name: 快照名称

        Returns:
            最新快照文件路径
        """
        run_dir = Path(run_dir)
        snapshot_dir = run_dir / "config_snapshot"

        if not snapshot_dir.exists():
            return None

        snapshots = list(snapshot_dir.glob(f"{name}_*.yaml"))
        if not snapshots:
            snapshots = list(snapshot_dir.glob(f"{name}_*.json"))

        if not snapshots:
            return None

        return max(snapshots, key=lambda p: p.stat().st_mtime)
