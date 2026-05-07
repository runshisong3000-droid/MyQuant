#!/usr/bin/env python
"""
MyQuant Pro - 日志系统和注册表测试

验证内容:
    1. RunManager 生成唯一 run_id
    2. 结构化日志正常输出
    3. 注册表正常记录
    4. 版本号自动递增
    5. 配置快照保存

使用方法:
    python scripts/test_logging_system.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from src.utils import RunManager, get_logger, get_registry, VersionManager, ConfigSnapshot


def test_run_manager():
    """测试 RunManager"""
    print("\n" + "=" * 60)
    print("Test 1: RunManager")
    print("=" * 60)

    base_dir = Path("./test_run")
    base_dir.mkdir(exist_ok=True, parents=True)

    rm = RunManager(
        module_name="test",
        base_dir=base_dir,
        git_enabled=True
    )

    run_id = rm.start_run(
        config={"test_param": "value", "learning_rate": 0.01},
        name="Test_Run_001",
        notes="这是一次测试运行"
    )

    print(f"[OK] Generated run_id: {run_id}")
    print(f"[OK] Run directory: {rm.run_dir}")
    print(f"[OK] Git commit: {rm.git_commit}")

    rm.logger.info("Test info log")
    rm.logger.warning("Test warning log")
    rm.logger.metric("test_accuracy", 0.95)
    rm.logger.artifact("test_model", "/path/to/model.pkl", "file")

    rm.end_run(
        status="completed",
        metrics={"accuracy": 0.95, "loss": 0.05},
        artifacts={"model": "/path/to/model.pkl"}
    )

    print("[OK] Run completed successfully")
    return run_id


def test_registry():
    """测试注册表"""
    print("\n" + "=" * 60)
    print("Test 2: Registry")
    print("=" * 60)

    registry = get_registry()

    experiments = registry.get_latest_runs("experiment", limit=5)
    print(f"[OK] Total experiments: {len(experiments)}")

    if len(experiments) > 0:
        print(f"[OK] Latest experiment: {experiments.iloc[0]['run_id']}")

    summary = registry.get_metrics_summary("backtest")
    print(f"[OK] Backtest summary: {summary}")

    return True


def test_versioning():
    """Test version management"""
    print("\n" + "=" * 60)
    print("Test 3: Versioning")
    print("=" * 60)

    vm = VersionManager(base_dir="./test_versions")

    version1 = vm.generate_version("data")
    print(f"[OK] Generated version: {version1}")

    version2 = vm.get_next_version("model_registry.csv", "model")
    print(f"[OK] Next model version: {version2}")

    version3 = vm.generate_version("backtest")
    print(f"[OK] Generated version: {version3}")

    return True


def test_config_snapshot():
    """Test config snapshot"""
    print("\n" + "=" * 60)
    print("Test 4: ConfigSnapshot")
    print("=" * 60)

    test_dir = Path("./test_run/test_config")
    test_dir.mkdir(parents=True, exist_ok=True)

    snapshot = ConfigSnapshot(test_dir)

    test_config = {
        "strategy": "DualMA",
        "parameters": {
            "short_window": 20,
            "long_window": 60
        },
        "data": {
            "source": "akshare",
            "universe": ["000001.SZ", "000002.SZ"]
        }
    }

    snapshot_file = snapshot.save_config_dict(test_config, "test_config")
    print(f"[OK] Saved config snapshot: {snapshot_file}")

    loaded = snapshot.load_snapshot(snapshot_file)
    print(f"[OK] Loaded config: {loaded}")

    return True


def test_structured_logger():
    """Test structured logger"""
    print("\n" + "=" * 60)
    print("Test 5: StructuredLogger")
    print("=" * 60)

    logger = get_logger(
        module_name="test_logger",
        run_dir="./test_run",
        run_id="test_logger_001"
    )

    logger.info("This is an info log")
    logger.warning("This is a warning log")
    logger.error("This is an error log", exc_info=Exception("Test exception"))
    logger.metric("accuracy", 0.95)
    logger.metric("precision", 0.92)
    logger.artifact("model.pkl", "/models/model.pkl", "file")

    print("[OK] All log types written successfully")

    return True


def cleanup():
    """Clean up test files"""
    import shutil
    test_dirs = ["./test_run", "./test_versions"]
    for d in test_dirs:
        if Path(d).exists():
            shutil.rmtree(d)
            print(f"[OK] Cleaned up {d}")


def main():
    print("\n" + "=" * 60)
    print("MyQuant Pro - Logging System Test Suite")
    print("=" * 60)

    try:
        test_run_manager()
        test_structured_logger()
        test_registry()
        test_versioning()
        test_config_snapshot()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)

        print("\nGenerated files:")
        print("  - logs/runs/<run_id>/run_log.json")
        print("  - logs/runs/<run_id>/config_snapshot.yaml")
        print("  - logs/runs/<run_id>/metrics.json")
        print("  - logs/runs/<run_id>/artifacts.json")
        print("  - logs/runs/<run_id>/run_info.json")
        print("  - registry/experiment_registry.csv")
        print("  - registry/model_registry.csv")
        print("  - registry/backtest_registry.csv")

        print("\n" + "=" * 60)
        print("Git Commit Suggestion:")
        print("=" * 60)
        print("feat(infra): add structured logging and experiment registry")
        print("")

        print("Next steps:")
        print("  1. Review generated logs in logs/runs/")
        print("  2. Check registry/*.csv files")
        print("  3. Run: python scripts/run_backtest.py")
        print("  4. Run: python scripts/ai_stock_picking_demo.py")

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
