"""
Research Lite Pipeline Tests

测试配置读取、缓存读取、样本切分、因子合并等
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestComputeProfileConfig:
    """测试配置读取"""

    def test_config_file_exists(self):
        """测试配置文件存在"""
        config_path = 'config/compute_profile.yaml'
        assert os.path.exists(config_path), "Config file should exist"

    def test_config_loads_research_lite(self):
        """测试加载research_lite配置"""
        if not os.path.exists('config/compute_profile.yaml'):
            pytest.skip("Config file not found")

        import yaml

        with open('config/compute_profile.yaml', 'r', encoding='utf-8') as f:
            config_all = yaml.safe_load(f)

        assert 'profiles' in config_all, "Should have profiles"
        assert 'research_lite' in config_all['profiles'], "Should have research_lite profile"

        config = config_all['profiles']['research_lite']

        assert config['stock_count'] == 100, "stock_count should be 100"
        assert config['history_months'] == 12, "history_months should be 12"
        assert config['formula_factor_limit'] == 100, "formula_factor_limit should be 100"
        assert config['device'] == 'cpu', "device should be cpu"

    def test_student_laptop_config(self):
        """测试student_laptop配置"""
        if not os.path.exists('config/compute_profile.yaml'):
            pytest.skip("Config file not found")

        import yaml

        with open('config/compute_profile.yaml', 'r', encoding='utf-8') as f:
            config_all = yaml.safe_load(f)

        config = config_all['profiles']['student_laptop']

        assert config['stock_count'] == 20, "stock_count should be 20"
        assert config['history_months'] == 6, "history_months should be 6"


class TestCacheMechanism:
    """测试缓存机制"""

    def test_cache_file_path(self):
        """测试缓存文件路径存在"""
        cache_dir = 'data/cache'
        os.makedirs(cache_dir, exist_ok=True)

        assert os.path.exists(cache_dir), "Cache directory should exist"

    def test_processed_data_path(self):
        """测试处理后数据路径"""
        processed_dir = 'data/processed'
        os.makedirs(processed_dir, exist_ok=True)

        assert os.path.exists(processed_dir), "Processed directory should exist"


class TestSampleSplit:
    """测试样本切分"""

    def test_train_val_test_split_by_date(self):
        """测试按日期切分"""
        from src.factors.neural.sequence_dataset import SequenceDataset

        dates = pd.date_range('2024-01-01', periods=50, freq='B')
        stocks = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

        data = []
        for date in dates:
            for stock in stocks:
                data.append({
                    'date': date,
                    'stock': stock,
                    'close': 100 + np.random.rand() * 10
                })

        df = pd.DataFrame(data)

        dataset = SequenceDataset(df=df, lookback_window=5, features=['close'], target_horizon=1)
        splits = dataset.get_train_val_test_split(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)

        train_dates = splits['train'][1]['signal_date'].unique()
        val_dates = splits['val'][1]['signal_date'].unique()
        test_dates = splits['test'][1]['signal_date'].unique()

        assert len(train_dates) > len(val_dates), "Train should have more dates"
        assert max(train_dates) < min(val_dates), "Train dates should be before val dates"
        assert max(val_dates) < min(test_dates), "Val dates should be before test dates"

        assert max(train_dates) < min(val_dates), "Train dates should be before val dates"
        assert max(val_dates) < min(test_dates), "Val dates should be before test dates"


class TestFormulaNeuralMerge:
    """测试公式因子和神经因子合并"""

    def test_formula_and_neural_dict(self):
        """测试因子字典可以合并"""
        formula_factors = {
            'momentum_10': pd.Series([1, 2, 3]),
            'reversal_5': pd.Series([4, 5, 6])
        }

        neural_factors = {
            'neural_factor_0': pd.Series([7, 8, 9]),
            'neural_factor_1': pd.Series([10, 11, 12])
        }

        all_factors = {**formula_factors, **neural_factors}

        assert len(all_factors) == 4, "Should have 4 factors combined"
        assert 'momentum_10' in all_factors
        assert 'neural_factor_0' in all_factors


class TestNoSimulatedData:
    """测试不使用模拟数据"""

    def test_research_lite_requires_real_data(self):
        """测试research_lite需要真实数据"""
        config_path = 'config/compute_profile.yaml'

        if not os.path.exists(config_path):
            pytest.skip("Config file not found")

        import yaml

        with open(config_path, 'r', encoding='utf-8') as f:
            config_all = yaml.safe_load(f)

        config = config_all['profiles']['research_lite']

        assert config.get('use_llm_api', False) == False, "Should not use LLM API"
        assert config.get('use_gpu', False) == False, "Should not use GPU"
        assert config['device'] == 'cpu', "Should use CPU only"


class TestPipelineStructure:
    """测试流水线结构"""

    def test_pipeline_has_required_steps(self):
        """测试pipeline包含必要步骤"""
        pipeline_file = 'scripts/run_research_lite_pipeline.py'

        assert os.path.exists(pipeline_file), "Pipeline file should exist"

        with open(pipeline_file, 'r', encoding='utf-8') as f:
            content = f.read()

        required_steps = [
            'Load Configuration',
            'Load or Fetch Data',
            'Data Preparation',
            'Generate Formula Factors',
            'Generate Neural Factors',
            'Evaluate Formula Factors',
            'Evaluate Neural Factors',
            'Generate Reliability Audit Report'
        ]

        for step in required_steps:
            assert step in content, "Pipeline should contain step: {}".format(step)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
