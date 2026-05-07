"""
版本管理模块

功能:
    - 生成带版本号的文件名
    - 自动递增版本号
    - 避免文件覆盖
"""

import re
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime
import pandas as pd


class VersionManager:
    def __init__(self, base_dir: Path, registry_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir)
        self.registry_dir = Path(registry_dir) if registry_dir else self.base_dir / "registry"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def generate_version(
        self,
        prefix: str,
        date_format: str = "%Y%m%d",
        include_counter: bool = True
    ) -> str:
        """
        生成版本号

        Args:
            prefix: 版本前缀，例如 "data", "model", "backtest"
            date_format: 日期格式
            include_counter: 是否包含序号

        Returns:
            版本号字符串，格式: prefix_vYYYYMMDD_001
        """
        date_str = datetime.now().strftime(date_format)
        counter = self._get_next_counter(prefix, date_str)
        if include_counter:
            return f"{prefix}_v{date_str}_{counter:03d}"
        return f"{prefix}_v{date_str}"

    def _get_next_counter(self, prefix: str, date_str: str) -> int:
        """获取下一个序号"""
        pattern = f"{prefix}_v{date_str}_(\\d{{3}})"
        max_counter = 0

        if self.registry_dir.exists():
            for reg_file in self.registry_dir.glob("*.csv"):
                try:
                    df = pd.read_csv(reg_file)
                    if 'version' in df.columns:
                        for version in df['version'].dropna():
                            match = re.match(f"^{pattern}$", str(version))
                            if match:
                                counter = int(match.group(1))
                                max_counter = max(max_counter, counter)
                except Exception:
                    continue

        return max_counter + 1

    def get_next_version(self, registry_file: str, prefix: str) -> str:
        """
        获取下一个版本号（基于特定注册表文件）

        Args:
            registry_file: 注册表文件名
            prefix: 版本前缀

        Returns:
            下一个版本号
        """
        registry_path = self.registry_dir / registry_file

        if not registry_path.exists():
            return self.generate_version(prefix)

        try:
            df = pd.read_csv(registry_path)
            if 'version' not in df.columns:
                return self.generate_version(prefix)

            versions = df['version'].dropna().tolist()
            pattern = f"^{prefix}_v\\d{{8}}_(\\d{{3}})$"

            max_counter = 0
            for v in versions:
                match = re.match(pattern, str(v))
                if match:
                    counter = int(match.group(1))
                    max_counter = max(max_counter, counter)

            date_str = datetime.now().strftime("%Y%m%d")
            return f"{prefix}_v{date_str}_{max_counter + 1:03d}"

        except Exception:
            return self.generate_version(prefix)

    def generate_file_path(
        self,
        category: str,
        prefix: str,
        extension: str = ".csv",
        subdir: Optional[str] = None
    ) -> Tuple[Path, str]:
        """
        生成带版本号的文件路径

        Args:
            category: 类别，例如 "data", "factors", "models", "backtests"
            prefix: 文件前缀
            extension: 文件扩展名
            subdir: 子目录

        Returns:
            (文件路径, 版本号)
        """
        version = self.get_next_version(f"{category}_registry.csv", prefix)

        if subdir:
            output_dir = self.base_dir / category / subdir
        else:
            output_dir = self.base_dir / category

        output_dir.mkdir(parents=True, exist_ok=True)

        file_path = output_dir / f"{version}{extension}"

        counter = 1
        while file_path.exists():
            version = self.get_next_version(f"{category}_registry.csv", prefix)
            file_path = output_dir / f"{version}{extension}"
            counter += 1
            if counter > 100:
                raise RuntimeError(f"Cannot find available version for {prefix}")

        return file_path, version

    def parse_version(self, version: str) -> Tuple[str, str, int]:
        """
        解析版本号

        Args:
            version: 版本号字符串

        Returns:
            (prefix, date_str, counter)
        """
        pattern = r"^(.+)_v(\d{8})_(\d{3})$"
        match = re.match(pattern, version)

        if not match:
            raise ValueError(f"Invalid version format: {version}")

        return match.group(1), match.group(2), int(match.group(3))

    def get_version_info(self, version: str) -> dict:
        """
        获取版本详细信息

        Args:
            version: 版本号字符串

        Returns:
            版本信息字典
        """
        prefix, date_str, counter = self.parse_version(version)

        return {
            "version": version,
            "prefix": prefix,
            "date": f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}",
            "counter": counter,
            "timestamp": datetime.strptime(date_str, "%Y%m%d").isoformat()
        }

    def list_versions(self, prefix: str, limit: Optional[int] = None) -> list:
        """
        列出所有版本

        Args:
            prefix: 版本前缀
            limit: 返回数量限制

        Returns:
            版本号列表
        """
        versions = []

        if self.registry_dir.exists():
            for reg_file in self.registry_dir.glob("*_registry.csv"):
                try:
                    df = pd.read_csv(reg_file)
                    if 'version' in df.columns:
                        pattern = f"^{prefix}_v\\d{{8}}_\\d{{3}}$"
                        for v in df['version'].dropna():
                            if re.match(pattern, str(v)):
                                versions.append(v)
                except Exception:
                    continue

        versions.sort(reverse=True)

        if limit:
            return versions[:limit]

        return versions

    def compare_versions(self, version1: str, version2: str) -> dict:
        """
        对比两个版本

        Args:
            version1: 版本1
            version2: 版本2

        Returns:
            对比结果
        """
        info1 = self.get_version_info(version1)
        info2 = self.get_version_info(version2)

        return {
            "version1": info1,
            "version2": info2,
            "same_date": info1["date"] == info2["date"],
            "days_diff": (
                datetime.strptime(info1["date"], "%Y-%m-%d") -
                datetime.strptime(info2["date"], "%Y-%m-%d")
            ).days
        }
