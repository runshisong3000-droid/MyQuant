"""
测试 Dashboard App import 安全性
"""

import unittest
import os
import sys
import ast


class TestDashboardAppImport(unittest.TestCase):
    """测试 app.py 导入安全性"""

    def test_app_syntax_valid(self):
        """测试 app.py 语法正确"""
        app_path = os.path.join(os.path.dirname(__file__), '..', 'dashboard', 'app.py')
        with open(app_path, 'r', encoding='utf-8') as f:
            content = f.read()
        try:
            ast.parse(content)
            self.assertTrue(True)
        except SyntaxError as e:
            self.fail(f"Syntax error: {e}")

    def test_app_has_main_function(self):
        """测试 app.py 有 main 函数"""
        app_path = os.path.join(os.path.dirname(__file__), '..', 'dashboard', 'app.py')
        with open(app_path, 'r', encoding='utf-8') as f:
            content = f.read()
        tree = ast.parse(content)
        
        has_main = any(
            isinstance(node, ast.FunctionDef) and node.name == 'main'
            for node in ast.walk(tree)
        )
        self.assertTrue(has_main, "main() function not found in app.py")

    def test_no_st_run_call(self):
        """测试 app.py 中不存在 st.run 调用"""
        app_path = os.path.join(os.path.dirname(__file__), '..', 'dashboard', 'app.py')
        with open(app_path, 'r', encoding='utf-8') as f:
            content = f.read()
        self.assertNotIn('st.run()', content)
        self.assertNotIn('st.run(', content)

    def test_data_loader_import(self):
        """测试 data_loader.py 可以正常 import"""
        try:
            from dashboard.data_loader import DataLoader, loader
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Import failed: {e}")

    def test_run_manager_import(self):
        """测试 run_manager.py 可以正常 import"""
        try:
            from dashboard.run_manager import RunManager, run_manager
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Import failed: {e}")

    def test_no_shell_injection_in_run_manager(self):
        """测试 run_manager 中没有 shell 注入漏洞"""
        run_manager_path = os.path.join(os.path.dirname(__file__), '..', 'dashboard', 'run_manager.py')
        with open(run_manager_path, 'r', encoding='utf-8') as f:
            content = f.read()
        self.assertNotIn('os.system', content)
        self.assertNotIn('shell=True', content)


if __name__ == '__main__':
    unittest.main()
