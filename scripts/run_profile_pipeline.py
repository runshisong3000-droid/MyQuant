"""
Profile Pipeline Runner - 总控脚本

功能:
    - 按顺序调用各个 pipeline
    - 支持 profile 参数
    - 不复制核心逻辑，只做调度
    - 记录每个步骤的状态和耗时

用法:
    python run_profile_pipeline.py --profile research_medium_trial
    python run_profile_pipeline.py --profile research_medium
"""

import sys
import os
import argparse
import subprocess
import time
import json
from datetime import datetime


def load_config(profile_name):
    """加载指定 profile 的配置"""
    config_path = 'config/compute_profile.yaml'
    if not os.path.exists(config_path):
        print(f"[ERROR] 配置文件不存在: {config_path}")
        return None
    
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    profiles = config.get('profiles', {})
    if profile_name not in profiles:
        print(f"[ERROR] Profile '{profile_name}' 不存在于配置文件中")
        print(f"[INFO] 可用的 profiles: {list(profiles.keys())}")
        return None
    
    return config


def run_command(cmd, cwd=None, env=None):
    """运行命令并返回结果"""
    print(f"\n[RUN] {cmd}")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd or os.getcwd(),
            env=env,
            capture_output=True,
            text=True,
            timeout=None
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"[OK] 命令成功完成，耗时: {elapsed:.2f}s")
            if result.stdout:
                print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
            return True, elapsed, result.stdout, result.stderr
        else:
            print(f"[FAIL] 命令失败，耗时: {elapsed:.2f}s")
            if result.stderr:
                print(result.stderr)
            if result.stdout:
                print(result.stdout)
            return False, elapsed, result.stdout, result.stderr
    
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"[FAIL] 命令超时，耗时: {elapsed:.2f}s")
        return False, elapsed, "", "Timeout"
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[FAIL] 命令执行异常: {e}，耗时: {elapsed:.2f}s")
        return False, elapsed, "", str(e)


def main():
    parser = argparse.ArgumentParser(description='Profile Pipeline Runner')
    parser.add_argument('--profile', required=True, help='Profile name from compute_profile.yaml')
    parser.add_argument('--dry-run', action='store_true', help='Only check configuration, do not run pipelines')
    args = parser.parse_args()
    
    profile_name = args.profile
    dry_run = args.dry_run
    
    print("=" * 80)
    print(f"MyQuant Profile Pipeline Runner")
    print(f"Profile: {profile_name}")
    print("=" * 80)
    
    # 加载配置
    config = load_config(profile_name)
    if not config:
        sys.exit(1)
    
    profile_config = config['profiles'][profile_name]
    print(f"\n[CONFIG] 使用 profile: {profile_name}")
    print(f"  - stock_count: {profile_config.get('stock_count')}")
    print(f"  - history_months: {profile_config.get('history_months')}")
    print(f"  - device: {profile_config.get('device')}")
    
    # 设置环境变量传递 profile
    env = os.environ.copy()
    env['MYQUANT_PROFILE'] = profile_name
    
    # 步骤定义
    steps = [
        {
            'name': 'Research Factor Pipeline',
            'script': 'scripts/run_research_lite_pipeline.py',
            'args': f'--profile {profile_name}',
            'required': True
        },
        {
            'name': 'Neural Factor Pipeline',
            'script': 'scripts/run_neural_factor_pipeline.py',
            'args': f'--profile {profile_name}',
            'required': True
        },
        {
            'name': 'OOS Validation Pipeline',
            'script': 'scripts/run_oos_validation_pipeline.py',
            'args': f'--profile {profile_name}',
            'required': True
        },
        {
            'name': 'Trading Constraints Pipeline',
            'script': 'scripts/run_trading_constraints_pipeline.py',
            'args': f'--profile {profile_name}',
            'required': True
        }
    ]
    
    # Profile-specific 目录
    profile_dir = f'data/processed/profiles/{profile_name}'
    dashboard_dir = f'data/dashboard/profiles/{profile_name}'
    research_lite_dir = 'data/dashboard'
    
    # Dry-run 模式
    if dry_run:
        print("\n" + "=" * 80)
        print("[DRY-RUN] Configuration Check")
        print("=" * 80)
        
        dry_run_results = []
        
        # 检查 profile 是否存在
        print("\n[CHECK] Profile Configuration")
        print(f"  [OK] Profile '{profile_name}' exists in config")
        dry_run_results.append({'check': 'Profile exists', 'status': 'OK'})
        
        # 检查 DataSourceManager
        print("\n[CHECK] DataSourceManager")
        ds_manager_path = 'src/data/data_source_manager.py'
        if os.path.exists(ds_manager_path):
            print(f"  [OK] DataSourceManager exists: {ds_manager_path}")
            dry_run_results.append({'check': 'DataSourceManager exists', 'status': 'OK'})
            
            # 检查 DataSourceManager 是否可以导入
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            try:
                from src.data.data_source_manager import DataSourceManager
                ds_manager = DataSourceManager()
                print(f"  [OK] DataSourceManager can be imported")
                dry_run_results.append({'check': 'DataSourceManager importable', 'status': 'OK'})
                
                # 检查 profile config 能读取
                profile_config_result = ds_manager.load_profile_config(profile_name)
                if profile_config_result:
                    print(f"  [OK] Profile config loaded for '{profile_name}'")
                    dry_run_results.append({'check': f'Profile config loadable: {profile_name}', 'status': 'OK'})
                else:
                    print(f"  [FAIL] Failed to load profile config for '{profile_name}'")
                    dry_run_results.append({'check': f'Profile config loadable: {profile_name}', 'status': 'FAIL'})
            except Exception as e:
                print(f"  [FAIL] Failed to import DataSourceManager: {e}")
                dry_run_results.append({'check': 'DataSourceManager importable', 'status': 'FAIL'})
        else:
            print(f"  [FAIL] DataSourceManager not found: {ds_manager_path}")
            dry_run_results.append({'check': 'DataSourceManager exists', 'status': 'FAIL'})
        
        # 检查子 pipeline 是否存在
        print("\n[CHECK] Pipeline Scripts")
        for step in steps:
            script_path = step['script']
            if os.path.exists(script_path):
                print(f"  [OK] {script_path}")
                dry_run_results.append({'check': f'Script exists: {script_path}', 'status': 'OK'})
            else:
                print(f"  [FAIL] {script_path} - NOT FOUND")
                dry_run_results.append({'check': f'Script missing: {script_path}', 'status': 'FAIL'})
        
        # 检查输出目录
        print("\n[CHECK] Output Directories")
        print(f"  Profile dir: {profile_dir}")
        print(f"  Dashboard dir: {dashboard_dir}")
        
        # 检查 prices_metadata.json
        print("\n[CHECK] Price Metadata")
        metadata_path = os.path.join(profile_dir, 'prices_metadata.json')
        if os.path.exists(metadata_path):
            print(f"  [OK] Price metadata exists: {metadata_path}")
            dry_run_results.append({'check': 'Price metadata exists', 'status': 'OK'})
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            required_fields = ['profile', 'target_stock_count', 'actual_stock_count', 
                             'success_symbols', 'failed_symbols', 'data_source', 'can_use_for_live_trading']
            missing_fields = [f for f in required_fields if f not in metadata]
            if not missing_fields:
                print(f"  [OK] Metadata fields are complete")
                dry_run_results.append({'check': 'Metadata fields complete', 'status': 'OK'})
                
                # 检查 actual_stock_count 是否满足阈值
                actual_count = metadata.get('actual_stock_count', 0)
                target_count = metadata.get('target_stock_count', 0)
                
                print(f"  [INFO] Target stock count: {target_count}")
                print(f"  [INFO] Actual stock count: {actual_count}")
                
                # 检查是否满足最小股票数要求
                min_stocks = config.get('data_sources', {}).get('min_absolute_stocks', {}).get(profile_name, 50)
                if actual_count >= min_stocks:
                    print(f"  [OK] Actual stock count ({actual_count}) >= minimum ({min_stocks})")
                    dry_run_results.append({'check': f'Stock count threshold ({min_stocks})', 'status': 'OK'})
                elif actual_count >= 50:
                    print(f"  [WARN] Actual stock count ({actual_count}) < minimum ({min_stocks})")
                    dry_run_results.append({'check': f'Stock count threshold ({min_stocks})', 'status': 'WARN'})
                else:
                    print(f"  [FAIL] Actual stock count ({actual_count}) < 50")
                    dry_run_results.append({'check': 'Stock count >= 50', 'status': 'FAIL'})
            else:
                print(f"  [FAIL] Missing metadata fields: {missing_fields}")
                dry_run_results.append({'check': 'Metadata fields complete', 'status': 'FAIL'})
        else:
            print(f"  [WARN] Price metadata not found: {metadata_path}")
            dry_run_results.append({'check': 'Price metadata exists', 'status': 'WARN'})
        
        # 检查是否会覆盖 research_lite
        print("\n[CHECK] Isolation Check")
        artifacts_to_check = [
            'formula_factors.parquet',
            'neural_factors.parquet', 
            'factor_summary.parquet',
            'oos_feature_comparison.parquet'
        ]
        
        will_overwrite = False
        for artifact in artifacts_to_check:
            global_path = os.path.join(research_lite_dir, artifact)
            profile_path = os.path.join(dashboard_dir, artifact)
            if os.path.exists(global_path):
                print(f"  [WARN] Global artifact exists: {global_path}")
                print(f"    Profile path: {profile_path}")
                dry_run_results.append({'check': f'Artifact isolation for {artifact}', 'status': 'WARN'})
        
        # 检查子 pipeline 是否支持 --profile 参数
        print("\n[CHECK] Profile Parameter Support")
        for step in steps:
            script_path = step['script']
            if os.path.exists(script_path):
                with open(script_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if '--profile' in content and 'argparse' in content:
                    print(f"  [OK] {script_path} supports --profile")
                    dry_run_results.append({'check': f'Profile support: {script_path}', 'status': 'OK'})
                else:
                    print(f"  [FAIL] {script_path} does NOT support --profile")
                    dry_run_results.append({'check': f'Profile support: {script_path}', 'status': 'FAIL'})
        
        # 输出预计路径
        print("\n[CHECK] Expected Paths")
        print(f"  Input prices: {profile_dir}/prices.parquet")
        print(f"  Output artifacts: {dashboard_dir}/")
        
        # 生成 dry-run 报告
        print("\n" + "=" * 80)
        print("[DRY-RUN REPORT]")
        print("=" * 80)
        
        print("\nSummary:")
        ok_count = sum(1 for r in dry_run_results if r['status'] == 'OK')
        warn_count = sum(1 for r in dry_run_results if r['status'] == 'WARN')
        fail_count = sum(1 for r in dry_run_results if r['status'] == 'FAIL')
        
        print(f"  OK: {ok_count}")
        print(f"  WARN: {warn_count}")
        print(f"  FAIL: {fail_count}")
        
        # 保存 dry-run 报告
        dry_run_report = {
            'profile': profile_name,
            'dry_run': True,
            'run_at': datetime.now().isoformat(),
            'checks': dry_run_results,
            'profile_config': profile_config,
            'expected_input_path': f'{profile_dir}/prices.parquet',
            'expected_output_dir': dashboard_dir,
            'can_proceed': fail_count == 0,
            'can_use_for_live_trading': False
        }
        
        report_dir = 'reports'
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, f'{profile_name}_dry_run_report.json')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(dry_run_report, f, ensure_ascii=False, indent=2)
        
        print(f"\n[INFO] Dry-run report saved to: {report_path}")
        
        if fail_count > 0:
            print("\n[FAIL] Dry-run failed! Please fix the issues before running.")
            sys.exit(1)
        else:
            print("\n[OK] Dry-run passed! Ready to run the pipeline.")
            sys.exit(0)
        
        return
    
    # 运行各个步骤
    results = []
    total_start = time.time()
    
    for step in steps:
        print("\n" + "=" * 80)
        print(f"[Step] {step['name']}")
        print("=" * 80)
        
        cmd = f"py14venv\\Scripts\\python.exe -u {step['script']} {step['args']}"
        success, elapsed, stdout, stderr = run_command(cmd, env=env)
        
        results.append({
            'name': step['name'],
            'script': step['script'],
            'success': success,
            'elapsed': elapsed,
            'stdout': stdout,
            'stderr': stderr,
            'required': step['required']
        })
        
        # 如果是必需步骤且失败，停止执行
        if step['required'] and not success:
            print(f"\n[CRITICAL] 必需步骤 '{step['name']}' 失败，停止执行")
            break
    
    # 生成报告
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 80)
    print("[FINAL REPORT]")
    print("=" * 80)
    
    print(f"\nProfile: {profile_name}")
    print(f"Total Time: {total_elapsed:.2f} seconds")
    print("\nStep Results:")
    
    for result in results:
        status = "✅" if result['success'] else "❌"
        print(f"  {status} {result['name']}: {'Success' if result['success'] else 'Failed'} ({result['elapsed']:.2f}s)")
    
    # 保存报告
    report = {
        'profile': profile_name,
        'profile_config': profile_config,
        'run_at': datetime.now().isoformat(),
        'total_time': total_elapsed,
        'steps': results,
        'success': all(r['success'] for r in results),
        'can_use_for_live_trading': False
    }
    
    report_dir = 'reports'
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f'{profile_name}_report.json')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n[INFO] 报告已保存到: {report_path}")
    
    # 输出总结
    if all(r['success'] for r in results):
        print("\n🎉 Profile pipeline 运行成功！")
    else:
        print("\n⚠️ Profile pipeline 运行失败！")
        sys.exit(1)


if __name__ == '__main__':
    main()