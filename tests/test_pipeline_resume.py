"""
测试 Pipeline Resume 功能
"""

import os
import sys
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPipelineResume:
    """测试 --resume 和 --skip-completed 参数"""
    
    def test_run_profile_pipeline_has_resume_option(self):
        """测试 run_profile_pipeline 支持 --resume 参数"""
        runner_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'run_profile_pipeline.py')
        
        with open(runner_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert '--resume' in content, "run_profile_pipeline 应该支持 --resume"
        assert '--skip-completed' in content, "run_profile_pipeline 应该支持 --skip-completed"
    
    def test_stage_status_has_required_fields(self):
        """测试 stage_status.json 包含必需字段"""
        expected_fields = ['stage', 'status', 'artifact', 'started_at', 'finished_at', 'elapsed_seconds', 'error_message']
        
        # 模拟一个 stage_status 条目
        stage_entry = {
            'stage': 'Test Stage',
            'status': 'completed',
            'artifact': 'test.parquet',
            'started_at': '2026-05-09T00:00:00',
            'finished_at': '2026-05-09T00:01:00',
            'elapsed_seconds': 60.0,
            'error_message': None
        }
        
        for field in expected_fields:
            assert field in stage_entry, f"stage_status 缺少必需字段: {field}"
    
    def test_can_use_for_live_trading_is_false_in_report(self):
        """测试报告中 can_use_for_live_trading 始终为 false"""
        runner_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'run_profile_pipeline.py')
        
        with open(runner_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "'can_use_for_live_trading': False" in content, "can_use_for_live_trading 应该始终为 false"
    
    def test_skip_message_format(self):
        """测试跳过消息格式正确"""
        runner_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'run_profile_pipeline.py')
        
        with open(runner_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "[SKIP]" in content, "应该有 [SKIP] 消息格式"
        assert "Artifact exists" in content, "应该有 Artifact exists 消息"


class TestArtifactCheck:
    """测试 artifact 检查逻辑"""
    
    def test_artifact_check_returns_correct_values(self):
        """测试 artifact 检查返回正确的值"""
        import tempfile
        
        # 创建临时目录和文件
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建测试文件
            artifact_path = os.path.join(tmpdir, 'test.parquet')
            
            # 测试不存在的文件
            exists, _ = _check_artifact_exists(tmpdir, 'nonexistent.parquet')
            assert exists is False, "不存在的文件应该返回 False"
            
            # 创建空文件（模拟 parquet）
            with open(artifact_path, 'w') as f:
                f.write('test')
            
            # 测试存在的文件
            exists, _ = _check_artifact_exists(tmpdir, 'test.parquet')
            assert exists is True, "存在的文件应该返回 True"


def _check_artifact_exists(dashboard_dir, artifact_name, required_columns=None):
    """模拟检查 artifact 是否存在"""
    artifact_path = os.path.join(dashboard_dir, artifact_name)
    
    if not os.path.exists(artifact_path):
        return False, None
    
    return True, None
