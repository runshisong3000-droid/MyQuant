"""
研究实验模块 - Experiment

核心功能:
    - 实验对象管理
    - 实验配置
    - 实验结果存储
    - 实验比较
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import os
import hashlib


class Experiment:
    """
    实验对象
    
    记录一次完整的研究实验：
    - 实验ID、名称、描述
    - 数据来源和时间范围
    - 因子配置
    - 模型配置
    - 回测结果
    - 评估指标
    """

    def __init__(
        self,
        experiment_id: Optional[str] = None,
        name: str = "Untitled Experiment",
        description: str = "",
        author: str = "Anonymous"
    ):
        self.experiment_id = experiment_id or self._generate_id()
        self.name = name
        self.description = description
        self.author = author
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        
        self.data_config = {}
        self.factor_config = {}
        self.model_config = {}
        self.backtest_config = {}
        
        self.results = {}
        self.metrics = {}
        self.status = "created"
        
    def _generate_id(self) -> str:
        """生成唯一实验ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_str = hashlib.md5(timestamp.encode()).hexdigest()[:8]
        return f"exp_{timestamp}_{hash_str}"
    
    def set_data_config(self, config: Dict[str, Any]):
        """设置数据配置"""
        self.data_config = config
        self.updated_at = datetime.now()
    
    def set_factor_config(self, config: Dict[str, Any]):
        """设置因子配置"""
        self.factor_config = config
        self.updated_at = datetime.now()
    
    def set_model_config(self, config: Dict[str, Any]):
        """设置模型配置"""
        self.model_config = config
        self.updated_at = datetime.now()
    
    def set_backtest_config(self, config: Dict[str, Any]):
        """设置回测配置"""
        self.backtest_config = config
        self.updated_at = datetime.now()
    
    def set_results(self, results: Dict[str, Any]):
        """设置实验结果"""
        self.results = results
        self.updated_at = datetime.now()
    
    def set_metrics(self, metrics: Dict[str, float]):
        """设置评估指标"""
        self.metrics = metrics
        self.updated_at = datetime.now()
    
    def mark_completed(self):
        """标记实验完成"""
        self.status = "completed"
        self.updated_at = datetime.now()
    
    def mark_failed(self, error_message: str):
        """标记实验失败"""
        self.status = "failed"
        self.results["error"] = error_message
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "author": self.author,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "status": self.status,
            "data_config": self.data_config,
            "factor_config": self.factor_config,
            "model_config": self.model_config,
            "backtest_config": self.backtest_config,
            "results": self.results,
            "metrics": self.metrics
        }
    
    def save(self, directory: str = "experiments"):
        """保存实验"""
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, f"{self.experiment_id}.json")
        
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    @classmethod
    def load(cls, file_path: str) -> "Experiment":
        """加载实验"""
        with open(file_path, "r") as f:
            data = json.load(f)
        
        experiment = cls(
            experiment_id=data["experiment_id"],
            name=data["name"],
            description=data.get("description", ""),
            author=data.get("author", "Anonymous")
        )
        
        experiment.created_at = datetime.fromisoformat(data["created_at"])
        experiment.updated_at = datetime.fromisoformat(data["updated_at"])
        experiment.status = data["status"]
        experiment.data_config = data.get("data_config", {})
        experiment.factor_config = data.get("factor_config", {})
        experiment.model_config = data.get("model_config", {})
        experiment.backtest_config = data.get("backtest_config", {})
        experiment.results = data.get("results", {})
        experiment.metrics = data.get("metrics", {})
        
        return experiment


class ExperimentManager:
    """
    实验管理器
    
    管理多个实验的创建、存储、查询和比较
    """

    def __init__(self, storage_dir: str = "experiments"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
    
    def create_experiment(
        self,
        name: str = "Untitled Experiment",
        description: str = "",
        author: str = "Anonymous"
    ) -> Experiment:
        """创建新实验"""
        experiment = Experiment(name=name, description=description, author=author)
        experiment.save(self.storage_dir)
        return experiment
    
    def load_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """加载实验"""
        file_path = os.path.join(self.storage_dir, f"{experiment_id}.json")
        
        if os.path.exists(file_path):
            return Experiment.load(file_path)
        return None
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """列出所有实验"""
        experiments = []
        
        for filename in os.listdir(self.storage_dir):
            if filename.endswith(".json"):
                experiment_id = filename.replace(".json", "")
                experiment = self.load_experiment(experiment_id)
                if experiment:
                    experiments.append({
                        "experiment_id": experiment.experiment_id,
                        "name": experiment.name,
                        "description": experiment.description,
                        "author": experiment.author,
                        "created_at": experiment.created_at,
                        "status": experiment.status,
                        "metrics": experiment.metrics
                    })
        
        return sorted(experiments, key=lambda x: x["created_at"], reverse=True)
    
    def compare_experiments(self, experiment_ids: List[str]) -> pd.DataFrame:
        """比较多个实验"""
        comparison_data = []
        
        for exp_id in experiment_ids:
            experiment = self.load_experiment(exp_id)
            if experiment and experiment.status == "completed":
                row = {
                    "experiment_id": experiment.experiment_id,
                    "name": experiment.name,
                    "created_at": experiment.created_at,
                    **experiment.metrics
                }
                comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def delete_experiment(self, experiment_id: str) -> bool:
        """删除实验"""
        file_path = os.path.join(self.storage_dir, f"{experiment_id}.json")
        
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False