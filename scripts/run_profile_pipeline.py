"""
Profile Pipeline Runner - 总控脚本

功能:
    - 按顺序调用各个 pipeline
    - 支持 profile 参数
    - 支持 --resume 和 --skip-completed 参数
    - 记录每个步骤的状态和耗时
    - 支持从失败点恢复

用法:
    python run_profile_pipeline.py --profile research_medium_trial
    python run_profile_pipeline.py --profile research_medium_trial --resume
    python run_profile_pipeline.py --profile research_medium_trial --resume --skip-completed
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


def load_stage_status(dashboard_dir):
    """加载阶段状态文件"""
    status_path = os.path.join(dashboard_dir, 'stage_status.json')
    if os.path.exists(status_path):
        with open(status_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_stage_status(dashboard_dir, stage, status, artifact=None, elapsed_seconds=None, error_message=None):
    """保存阶段状态"""
    status_path = os.path.join(dashboard_dir, 'stage_status.json')
    
    if os.path.exists(status_path):
        with open(status_path, 'r', encoding='utf-8') as f:
            status_dict = json.load(f)
    else:
        status_dict = {}
    
    status_dict[stage] = {
        'stage': stage,
        'status': status,
        'artifact': artifact,
        'started_at': datetime.now().isoformat(),
        'finished_at': datetime.now().isoformat(),
        'elapsed_seconds': elapsed_seconds,
        'error_message': error_message
    }
    
    with open(status_path, 'w', encoding='utf-8') as f:
        json.dump(status_dict, f, ensure_ascii=False, indent=2)


def check_artifact_exists(dashboard_dir, artifact_name, required_columns=None):
    """检查 artifact 是否存在且字段完整"""
    artifact_path = os.path.join(dashboard_dir, artifact_name)
    
    if not os.path.exists(artifact_path):
        return False, None
    
    print(f"[SKIP] Artifact exists: {artifact_name}")
    
    if required_columns and artifact_name.endswith('.parquet'):
        try:
            import pandas as pd
            df = pd.read_parquet(artifact_path)
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                print(f"[WARN] Artifact {artifact_name} missing columns: {missing_cols}")
                return False, df
            return True, df
        except Exception as e:
            print(f"[WARN] Failed to check artifact {artifact_name}: {e}")
            return True, None
    
    return True, None


def main():
    parser = argparse.ArgumentParser(description='Profile Pipeline Runner')
    parser.add_argument('--profile', required=True, help='Profile name from compute_profile.yaml')
    parser.add_argument('--dry-run', action='store_true', help='Only check configuration, do not run pipelines')
    parser.add_argument('--resume', action='store_true', help='Resume from last completed stage')
    parser.add_argument('--skip-completed', action='store_true', help='Skip stages with existing artifacts')
    args = parser.parse_args()
    
    profile_name = args.profile
    dry_run = args.dry_run
    resume = args.resume
    skip_completed = args.skip_completed
    
    print("=" * 80)
    print(f"MyQuant Profile Pipeline Runner")
    print(f"Profile: {profile_name}")
    print(f"Resume: {resume}")
    print(f"Skip Completed: {skip_completed}")
    print("=" * 80)
    
    # 加载配置
    config = load_config(profile_name)
    if not config:
        sys.exit(1)
    
    profile_config = config['profiles'][profile_name]
    
    # Profile-specific 目录
    profile_dir = f'data/processed/profiles/{profile_name}'
    dashboard_dir = f'data/dashboard/profiles/{profile_name}'
    os.makedirs(dashboard_dir, exist_ok=True)
    
    print(f"\n[CONFIG] 使用 profile: {profile_name}")
    print(f"  - stock_count: {profile_config.get('stock_count')}")
    print(f"  - history_months: {profile_config.get('history_months')}")
    print(f"  - device: {profile_config.get('device')}")
    print(f"  - Profile dir: {profile_dir}")
    print(f"  - Dashboard dir: {dashboard_dir}")
    
    # 设置环境变量传递 profile
    env = os.environ.copy()
    env['MYQUANT_PROFILE'] = profile_name
    
    # 步骤定义，包含对应的 artifact 检查
    steps = [
        {
            'name': 'Research Factor Pipeline',
            'script': 'scripts/run_research_lite_pipeline.py',
            'args': f'--profile {profile_name}',
            'required': True,
            'artifact': 'factor_summary.parquet',
            'required_columns': ['factor_name', 'rank_ic_mean', 'icir', 'coverage']
        },
        {
            'name': 'Neural Factor Pipeline',
            'script': 'scripts/run_neural_factor_pipeline.py',
            'args': f'--profile {profile_name}',
            'required': True,
            'artifact': 'neural_factors.parquet',
            'required_columns': ['date', 'stock', 'signal_date']
        },
        {
            'name': 'OOS Validation Pipeline',
            'script': 'scripts/run_oos_validation_pipeline.py',
            'args': f'--profile {profile_name}',
            'required': True,
            'artifact': 'oos_feature_comparison.parquet',
            'required_columns': ['factor_name', 'ic', 'icir']
        },
        {
            'name': 'Trading Constraints Pipeline',
            'script': 'scripts/run_trading_constraints_pipeline.py',
            'args': f'--profile {profile_name}',
            'required': True,
            'artifact': 'trading_constraint_summary.parquet',
            'required_columns': ['stock', 'date', 'is_tradable']
        }
    ]
    
    # 加载阶段状态
    stage_status = load_stage_status(dashboard_dir)
    
    # Dry-run 模式
    if dry_run:
        print("\n" + "=" * 80)
        print("[DRY-RUN] Configuration Check")
        print("=" * 80)
        
        dry_run_results = []
        
        print("\n[CHECK] Profile Configuration")
        print(f"  [OK] Profile '{profile_name}' exists in config")
        dry_run_results.append({'check': 'Profile exists', 'status': 'OK'})
        
        # 检查各个 artifact 是否存在
        print("\n[CHECK] Existing Artifacts")
        for step in steps:
            artifact_exists, _ = check_artifact_exists(dashboard_dir, step['artifact'])
            if artifact_exists:
                print(f"  [OK] {step['artifact']} exists")
                dry_run_results.append({'check': f'Artifact exists: {step["artifact"]}', 'status': 'OK'})
            else:
                print(f"  [INFO] {step['artifact']} missing")
                dry_run_results.append({'check': f'Artifact missing: {step["artifact"]}', 'status': 'INFO'})
        
        print("\n[CHECK] Stage Status")
        if os.path.exists(os.path.join(dashboard_dir, 'stage_status.json')):
            print(f"  [OK] stage_status.json exists")
            dry_run_results.append({'check': 'Stage status exists', 'status': 'OK'})
        else:
            print(f"  [INFO] stage_status.json missing")
            dry_run_results.append({'check': 'Stage status missing', 'status': 'INFO'})
        
        print("\n[CHECK] Expected Paths")
        print(f"  Input prices: {profile_dir}/prices.parquet")
        print(f"  Output artifacts: {dashboard_dir}/")
        
        print("\n" + "=" * 80)
        print("[DRY-RUN REPORT]")
        print("=" * 80)
        
        ok_count = sum(1 for r in dry_run_results if r['status'] == 'OK')
        info_count = sum(1 for r in dry_run_results if r['status'] == 'INFO')
        
        print(f"\nOK: {ok_count}")
        print(f"INFO: {info_count}")
        
        report = {
            'profile': profile_name,
            'dry_run': True,
            'run_at': datetime.now().isoformat(),
            'checks': dry_run_results,
            'can_proceed': True,
            'can_use_for_live_trading': False
        }
        
        report_path = os.path.join('reports', f'{profile_name}_dry_run_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n[INFO] Dry-run report saved to: {report_path}")
        print("\n[OK] Dry-run passed!")
        sys.exit(0)
    
    # 运行各个步骤
    results = []
    total_start = time.time()
    skip_count = 0
    
    for step in steps:
        print("\n" + "=" * 80)
        print(f"[Step] {step['name']}")
        print("=" * 80)
        
        # 检查是否需要跳过
        should_skip = False
        
        if resume or skip_completed:
            artifact_exists, _ = check_artifact_exists(
                dashboard_dir, 
                step['artifact'], 
                step.get('required_columns')
            )
            if artifact_exists:
                # 检查阶段状态
                stage_info = stage_status.get(step['name'], {})
                if stage_info.get('status') == 'completed':
                    print(f"[SKIP] Stage '{step['name']}' already completed")
                    should_skip = True
                    skip_count += 1
                elif skip_completed:
                    print(f"[SKIP] Artifact '{step['artifact']}' exists, skipping stage")
                    should_skip = True
                    skip_count += 1
        
        if should_skip:
            results.append({
                'name': step['name'],
                'script': step['script'],
                'success': True,
                'elapsed': 0,
                'stdout': '',
                'stderr': '',
                'required': step['required'],
                'skipped': True
            })
            save_stage_status(dashboard_dir, step['name'], 'completed', step['artifact'], 0)
            continue
        
        # 运行命令
        save_stage_status(dashboard_dir, step['name'], 'running')
        step_start = time.time()
        
        cmd = f"py14venv\\Scripts\\python.exe -u {step['script']} {step['args']}"
        success, elapsed, stdout, stderr = run_command(cmd, env=env)
        
        # 保存阶段状态
        if success:
            save_stage_status(dashboard_dir, step['name'], 'completed', step['artifact'], elapsed)
        else:
            save_stage_status(dashboard_dir, step['name'], 'failed', step['artifact'], elapsed, stderr)
        
        results.append({
            'name': step['name'],
            'script': step['script'],
            'success': success,
            'elapsed': elapsed,
            'stdout': stdout,
            'stderr': stderr,
            'required': step['required'],
            'skipped': False
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
    print(f"Steps Skipped: {skip_count}")
    print("\nStep Results:")
    
    for result in results:
        status = "[SKIP]" if result.get('skipped') else "[OK]" if result['success'] else "[FAIL]"
        elapsed_str = "(skipped)" if result.get('skipped') else f"({result['elapsed']:.2f}s)"
        print(f"  {status} {result['name']}: {'Skipped' if result.get('skipped') else ('Success' if result['success'] else 'Failed')} {elapsed_str}")
    
    # 保存报告
    report = {
        'profile': profile_name,
        'profile_config': profile_config,
        'run_at': datetime.now().isoformat(),
        'total_time': total_elapsed,
        'steps_skipped': skip_count,
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
        print("\n[SUCCESS] Profile pipeline 运行成功！")
    else:
        print("\n[FAILURE] Profile pipeline 运行失败！")
        sys.exit(1)


if __name__ == '__main__':
    main()
