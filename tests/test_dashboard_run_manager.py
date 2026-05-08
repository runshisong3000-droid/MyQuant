"""
测试 Dashboard Run Manager
"""

import unittest
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dashboard.run_manager import RunManager


class TestRunManagerWhitelist(unittest.TestCase):
    """测试白名单机制"""
    
    def setUp(self):
        self.run_manager = RunManager()
    
    def test_allowed_scripts_in_whitelist(self):
        """测试白名单脚本可以运行"""
        for script in RunManager.ALLOWED_SCRIPTS:
            self.assertTrue(self.run_manager.is_script_allowed(script))
    
    def test_disallowed_script_rejected(self):
        """测试非白名单脚本被拒绝"""
        disallowed = [
            'scripts/malicious.py',
            'scripts/../etc/passwd',
            '/bin/rm -rf /',
            'python -c "import os; os.system(\'rm -rf /\')"'
        ]
        for script in disallowed:
            self.assertFalse(self.run_manager.is_script_allowed(script))
    
    def test_no_shell_command_injection(self):
        """测试禁止 shell 命令注入"""
        malicious_scripts = [
            'scripts/run_student_laptop_pipeline.py; rm -rf /',
            'scripts/run_student_laptop_pipeline.py && rm -rf /',
            'scripts/run_student_laptop_pipeline.py || rm -rf /',
            'scripts/run_student_laptop_pipeline.py | cat /etc/passwd'
        ]
        for script in malicious_scripts:
            self.assertFalse(self.run_manager.is_script_allowed(script))


class TestRunManagerRunId(unittest.TestCase):
    """测试 run_id 生成"""
    
    def setUp(self):
        self.run_manager = RunManager()
    
    def test_run_id_generated(self):
        """测试 run_id 可以生成"""
        run_id1 = self.run_manager.generate_run_id()
        run_id2 = self.run_manager.generate_run_id()
        
        self.assertIsNotNone(run_id1)
        self.assertIsNotNone(run_id2)
        self.assertNotEqual(run_id1, run_id2)
    
    def test_run_id_format(self):
        """测试 run_id 格式正确"""
        run_id = self.run_manager.generate_run_id()
        parts = run_id.split('_')
        
        self.assertEqual(len(parts), 3)
        self.assertTrue(len(parts[0]) >= 8)  # 至少是 YYYYMMDD
        self.assertTrue(len(parts[1]) > 0)    # 非空


class TestRunManagerHistory(unittest.TestCase):
    """测试运行历史"""
    
    def setUp(self):
        self.run_manager = RunManager()
        self.test_history = [
            {
                'run_id': 'test_run_001',
                'pipeline_name': 'test_pipeline',
                'script_path': 'scripts/test.py',
                'start_time': '2026-01-01T00:00:00',
                'end_time': '2026-01-01T00:01:00',
                'status': 'SUCCESS',
                'return_code': 0,
                'log_path': 'logs/test.log',
                'generated_artifacts': []
            }
        ]
    
    def test_get_empty_history(self):
        """测试获取空历史"""
        history = self.run_manager.get_run_history()
        self.assertIsInstance(history, list)
    
    def test_run_history_can_be_saved(self):
        """测试运行历史可以保存"""
        test_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'dashboard', 'test_history.json')
        
        try:
            with open(test_path, 'w', encoding='utf-8') as f:
                json.dump(self.test_history, f)
            
            # Mock the path for testing
            original_path = self.run_manager.run_history_path
            self.run_manager.run_history_path = test_path
            
            history = self.run_manager.get_run_history()
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]['run_id'], 'test_run_001')
        finally:
            self.run_manager.run_history_path = original_path
            if os.path.exists(test_path):
                os.remove(test_path)


class TestRunManagerPipelineInfo(unittest.TestCase):
    """测试 pipeline 信息"""
    
    def setUp(self):
        self.run_manager = RunManager()
    
    def test_get_pipeline_info(self):
        """测试获取 pipeline 信息"""
        for script in RunManager.ALLOWED_SCRIPTS:
            info = self.run_manager.get_pipeline_info(script)
            self.assertIn('name', info)
            self.assertIn('description', info)
            self.assertIn('estimated_time', info)
            self.assertIn('input_data', info)
            self.assertIn('output_artifacts', info)


class TestRunManagerSafety(unittest.TestCase):
    """测试安全性"""
    
    def setUp(self):
        self.run_manager = RunManager()
    
    def test_can_use_for_live_trading_false(self):
        """测试 can_use_for_live_trading 仍然是 false"""
        from dashboard.data_loader import loader
        reliability_status = loader.load_reliability_status()
        status_dict = {item['item']: item['status'] for item in reliability_status}
        
        # Check manifest
        manifest, _ = loader.load_dashboard_manifest()
        if manifest is not None:
            self.assertFalse(manifest.get('can_use_for_live_trading', True))
        
        # Check reliability status
        status, _ = loader.load_json('reliability_status.json', directory='dashboard')
        if status is not None:
            self.assertFalse(status.get('can_use_for_live_trading', True))
    
    def test_no_live_trading_capability(self):
        """测试没有实盘交易能力"""
        # RunManager should not have any live trading methods
        self.assertFalse(hasattr(self.run_manager, 'execute_live_trade'))
        self.assertFalse(hasattr(self.run_manager, 'connect_to_broker'))
        self.assertFalse(hasattr(self.run_manager, 'place_order'))


class TestRunManagerRunningCheck(unittest.TestCase):
    """测试重复运行检查"""
    
    def setUp(self):
        self.run_manager = RunManager()
    
    def test_running_pipelines_detection(self):
        """测试正在运行的 pipeline 检测"""
        # Test with empty history
        running = self.run_manager.get_running_pipelines()
        self.assertIsInstance(running, list)


class TestRunManagerLogReading(unittest.TestCase):
    """测试日志读取"""
    
    def setUp(self):
        self.run_manager = RunManager()
    
    def test_log_not_found(self):
        """测试日志不存在时返回友好信息"""
        log_content = self.run_manager.get_log_content('nonexistent_run_id')
        self.assertEqual(log_content, "Log file not found")


if __name__ == '__main__':
    unittest.main()
