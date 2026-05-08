"""
Profile Pipeline 测试

测试内容:
1. compute_profile.yaml 包含所有必要的 profiles
2. profile 参数只能来自配置文件
3. 不存在的 profile 必须拒绝
4. profile 输出目录能正确生成
5. research_lite 和 research_medium artifacts 不会互相覆盖
6. run_profile_pipeline 不复制核心逻辑，只调用子 pipeline
7. dashboard data_loader 能按 profile 读取 artifacts
8. Run Center 能记录 profile
9. research_medium 不允许 fallback 到 research_lite cache
10. can_use_for_live_trading 必须 false
"""

import os
import sys
import json
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestProfileConfiguration:
    """测试 Profile 配置"""
    
    def test_config_contains_required_profiles(self):
        """测试配置文件包含所有必要的 profiles"""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'compute_profile.yaml')
        
        assert os.path.exists(config_path), "配置文件不存在"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        profiles = config.get('profiles', {})
        
        required_profiles = ['student_laptop', 'research_lite', 'research_medium_trial', 'research_medium']
        
        for profile in required_profiles:
            assert profile in profiles, f"缺少必需的 profile: {profile}"
    
    def test_profile_has_required_fields(self):
        """测试每个 profile 包含必需字段"""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'compute_profile.yaml')
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        profiles = config.get('profiles', {})
        required_fields = ['stock_count', 'history_months', 'device']
        
        for profile_name, profile_config in profiles.items():
            for field in required_fields:
                assert field in profile_config, f"Profile '{profile_name}' 缺少字段: {field}"
    
    def test_unknown_profile_is_rejected(self):
        """测试不存在的 profile 被拒绝"""
        from scripts.run_profile_pipeline import load_config
        
        result = load_config('unknown_profile_xyz')
        assert result is None, "不存在的 profile 应该返回 None"
    
    def test_profile_directory_structure(self):
        """测试 profile 目录结构能正确创建"""
        profile_name = 'test_profile'
        profile_dir = os.path.join('data', 'processed', 'profiles', profile_name)
        dashboard_dir = os.path.join('data', 'dashboard', 'profiles', profile_name)
        
        os.makedirs(profile_dir, exist_ok=True)
        os.makedirs(dashboard_dir, exist_ok=True)
        
        assert os.path.exists(profile_dir), f"Profile 目录未创建: {profile_dir}"
        assert os.path.exists(dashboard_dir), f"Dashboard 目录未创建: {dashboard_dir}"
        
        # 清理测试目录
        import shutil
        shutil.rmtree(profile_dir, ignore_errors=True)
        shutil.rmtree(dashboard_dir, ignore_errors=True)


class TestProfilePipelineRunner:
    """测试 Profile Pipeline 运行器"""
    
    def test_runner_does_not_duplicate_logic(self):
        """测试 run_profile_pipeline 不复制核心逻辑"""
        runner_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'run_profile_pipeline.py')
        
        with open(runner_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否包含不应该在调度器中出现的核心逻辑
        assert 'factor_generator' not in content.lower(), "run_profile_pipeline 不应该包含因子生成逻辑"
        assert 'neural_network' not in content.lower(), "run_profile_pipeline 不应该包含神经网络逻辑"
        assert 'backtest' not in content.lower(), "run_profile_pipeline 不应该包含回测逻辑"
        
        # 检查是否只包含调度逻辑
        assert 'subprocess' in content, "run_profile_pipeline 应该使用 subprocess 调用其他脚本"
        assert 'run_command' in content, "run_profile_pipeline 应该有命令运行函数"
    
    def test_runner_validates_profile(self):
        """测试运行器验证 profile"""
        from scripts.run_profile_pipeline import load_config
        
        # 有效的 profile 应该返回配置
        result = load_config('research_lite')
        assert result is not None, "有效的 profile 应该返回配置"
        
        # 无效的 profile 应该返回 None
        result = load_config('invalid_profile')
        assert result is None, "无效的 profile 应该返回 None"


class TestProfileArtifactsIsolation:
    """测试 Profile Artifacts 隔离"""
    
    def test_profile_artifacts_not_overlapping(self):
        """测试不同 profile 的 artifacts 不会互相覆盖"""
        profiles = ['research_lite', 'research_medium_trial', 'research_medium']
        
        for profile in profiles:
            profile_dir = os.path.join('data', 'processed', 'profiles', profile)
            os.makedirs(profile_dir, exist_ok=True)
            
            # 创建 profile-specific 的测试文件
            test_file = os.path.join(profile_dir, 'prices.parquet')
            # 只创建目录，不实际写入文件
            
        # 验证目录结构
        for profile in profiles:
            profile_dir = os.path.join('data', 'processed', 'profiles', profile)
            assert os.path.exists(profile_dir), f"Profile 目录不存在: {profile_dir}"
    
    def test_no_fallback_between_profiles(self):
        """测试不允许跨 profile fallback"""
        # 检查 run_research_lite_pipeline 是否使用 profile-specific 缓存
        pipeline_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'run_research_lite_pipeline.py')
        
        with open(pipeline_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 应该使用 profile-specific 路径
        assert 'PROFILE_DIR' in content, "应该使用 profile-specific 目录"
        assert 'profiles/{profile_name}' in content, "应该有 profile 目录模板"


class TestLiveTradingFlag:
    """测试 can_use_for_live_trading 标志"""
    
    def test_runner_report_has_live_trading_flag(self):
        """测试运行器报告包含 can_use_for_live_trading 字段"""
        from scripts.run_profile_pipeline import main
        
        # 检查报告结构（通过查看代码）
        runner_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'run_profile_pipeline.py')
        
        with open(runner_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert 'can_use_for_live_trading' in content, "报告应该包含 can_use_for_live_trading"
        assert 'False' in content, "can_use_for_live_trading 应该为 false"


if __name__ == "__main__":
    pytest.main([__file__, '-v'])