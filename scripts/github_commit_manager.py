#!/usr/bin/env python
"""
GitHub 提交管理脚本

功能:
    - 分批次提交代码
    - 自动生成提交信息
    - 按阶段分组提交
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Dict


class GitCommitManager:
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)

    def run_git(self, command: List[str]) -> tuple:
        """运行git命令"""
        try:
            result = subprocess.run(
                command,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode, result.stdout, result.stderr
        except Exception as e:
            return 1, "", str(e)

    def get_status(self) -> Dict:
        """获取git状态"""
        code, stdout, stderr = self.run_git(["git", "status", "--short"])
        return {
            "code": code,
            "modified": stdout
        }

    def add_files(self, files: List[str]) -> bool:
        """添加文件到暂存区"""
        for file in files:
            code, _, stderr = self.run_git(["git", "add", file])
            if code != 0:
                print(f"Failed to add {file}: {stderr}")
                return False
        return True

    def commit(self, message: str) -> bool:
        """提交"""
        code, _, stderr = self.run_git(["git", "commit", "-m", message])
        if code != 0:
            print(f"Failed to commit: {stderr}")
            return False
        return True

    def push(self, branch: str = "main") -> bool:
        """推送到远程"""
        code, _, stderr = self.run_git(["git", "push", "origin", branch])
        if code != 0:
            print(f"Failed to push: {stderr}")
            return False
        return True

    def get_current_branch(self) -> str:
        """获取当前分支"""
        code, stdout, _ = self.run_git(["git", "branch", "--show-current"])
        return stdout.strip() if code == 0 else "main"


def phase1_commit():
    """Phase 1: 商业化级日志系统与实验注册表"""
    print("\n" + "=" * 70)
    print("Phase 1: Commercial-grade Logging System & Experiment Registry")
    print("=" * 70)

    manager = GitCommitManager()

    files = [
        "src/utils/__init__.py",
        "src/utils/logger.py",
        "src/utils/run_manager.py",
        "src/utils/registry.py",
        "src/utils/versioning.py",
        "src/utils/config_snapshot.py",
        "scripts/run_backtest.py",
        "scripts/ai_stock_picking_demo.py"
    ]

    print("\nFiles to commit:")
    for f in files:
        print(f"  - {f}")

    message = """feat(infra): add commercial-grade experiment registry and structured logging

Phase 1: Logging System & Experiment Registry Foundation

Infrastructure:
- RunManager: Run lifecycle management with unique run_id generation
- StructuredLogger: JSON format logging (INFO/WARNING/ERROR/METRIC/ARTIFACT)
- RegistryManager: 6 registries (experiment, data, factor, model, backtest, trade)
- VersionManager: Auto-incrementing version numbers
- ConfigSnapshot: Preserve run configuration for reproducibility

Features:
- Auto-generated run_id with timestamp_module_uuid format
- Git commit hash and Python version tracking
- Complete audit trail for all experiments
- Commercial-grade backtest registry with full metrics
- Trade registry for every trade record

Registries:
- experiment_registry.csv: All experiment runs
- data_registry.csv: Data quality tracking
- factor_registry.csv: Factor IC/IR analysis
- model_registry.csv: Model training metrics
- backtest_registry.csv: Backtest results with 20+ metrics
- trade_registry.csv: Individual trade records

Integrated into:
- scripts/run_backtest.py
- scripts/ai_stock_picking_demo.py

This establishes the foundation for audit tracking, strategy attribution,
version reconciliation, and risk liability allocation - essential for
commercial deployment."""

    print("\nCommit message preview:")
    print("-" * 70)
    print(message[:500] + "...")
    print("-" * 70)

    return files, message


def phase2_commit():
    """Phase 2: 数据层标准化与数据质量报告"""
    print("\n" + "=" * 70)
    print("Phase 2: Data Layer Standardization & Quality Reports")
    print("=" * 70)

    manager = GitCommitManager()

    files = [
        "src/data/sources/__init__.py",
        "src/data/sources/base_source.py",
        "src/data/sources/akshare_source.py",
        "src/data/quality_report.py",
        "src/data/data_manager.py",
        "src/data/__init__.py",
        "scripts/test_quality_report.py"
    ]

    print("\nFiles to commit:")
    for f in files:
        print(f"  - {f}")

    message = """feat(data): add data layer abstraction and quality reporting

Phase 2: Data Standardization & Quality Infrastructure

Data Source Architecture:
- BaseDataSource: Abstract interface for all data sources
- AkShareDataSource: AkShare adapter with caching
- DataManager: Unified data access with quality reports

Data Quality Reporter:
- Missing value detection and analysis
- Anomaly detection (price/volume/extreme changes)
- Suspended day detection
- Data quality scoring (0-100)
- Statistical summary generation
- JSON format reports

Features:
- Data source abstraction (not coupled to specific provider)
- Local caching with expiration
- Automatic quality score calculation
- Recommendations for data improvements
- Integration with data registry

Quality Metrics:
- Overall missing rate
- Price anomalies (3-sigma rule)
- Volume anomalies (IQR method)
- Extreme price changes (>20%)
- Suspension days count
- Per-column statistics

Commercial Benefits:
- Data provenance tracking
- Quality assurance before research
- Audit-ready data documentation
- Reproducible data pipelines"""

    print("\nCommit message preview:")
    print("-" * 70)
    print(message[:500] + "...")
    print("-" * 70)

    return files, message


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("GitHub Commit Manager for MyQuant")
    print("=" * 70)

    manager = GitCommitManager()

    print("\nCurrent status:")
    status = manager.get_status()
    print(f"Modified files: {len(status['modified'].splitlines()) if status['modified'] else 0}")

    print("\nAvailable commits:")
    print("  1. Phase 1: Logging System & Registry")
    print("  2. Phase 2: Data Layer & Quality Reports")
    print("  3. Both Phases (1 -> 2)")
    print("  4. Show current status")
    print("  5. Exit")

    choice = input("\nSelect option (1-5): ").strip()

    if choice == "1":
        files, message = phase1_commit()
        confirm = input("\nProceed with commit? (y/n): ").strip().lower()
        if confirm == 'y':
            manager.add_files(files)
            manager.commit(message)
            print("\nPhase 1 committed successfully!")

    elif choice == "2":
        files, message = phase2_commit()
        confirm = input("\nProceed with commit? (y/n): ").strip().lower()
        if confirm == 'y':
            manager.add_files(files)
            manager.commit(message)
            print("\nPhase 2 committed successfully!")

    elif choice == "3":
        files1, msg1 = phase1_commit()
        confirm1 = input("\nCommit Phase 1? (y/n): ").strip().lower()
        if confirm1 == 'y':
            manager.add_files(files1)
            manager.commit(msg1)
            print("\nPhase 1 committed!")

        files2, msg2 = phase2_commit()
        confirm2 = input("\nCommit Phase 2? (y/n): ").strip().lower()
        if confirm2 == 'y':
            manager.add_files(files2)
            manager.commit(msg2)
            print("\nPhase 2 committed!")

        push = input("\nPush to remote? (y/n): ").strip().lower()
        if push == 'y':
            branch = manager.get_current_branch()
            manager.push(branch)
            print(f"\nPushed to {branch}!")

    elif choice == "4":
        print("\nCurrent git status:")
        code, stdout, _ = manager.run_git(["git", "status"])
        print(stdout)

    elif choice == "5":
        print("\nExiting...")
        sys.exit(0)

    else:
        print("\nInvalid option")


if __name__ == "__main__":
    main()
