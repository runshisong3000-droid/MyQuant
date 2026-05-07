"""
候选因子定义 - Factor Candidate

核心功能:
    - 候选因子对象
    - 因子表达式
    - 因子元数据
    - 因子状态管理
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
import hashlib


class FactorCandidate:
    """
    候选因子对象
    
    表示一个待评估的因子候选者：
    - 因子ID和名称
    - 因子表达式/公式
    - 因子数据
    - 评估结果
    - 状态（pending/evaluated/approved/rejected）
    """

    def __init__(
        self,
        expression: str,
        name: Optional[str] = None,
        description: str = "",
        source: str = "auto"
    ):
        self.expression = expression
        self.name = name or self._generate_name()
        self.description = description
        self.source = source
        self.created_at = datetime.now()
        
        self.factor_id = self._generate_id()
        self.data = None
        self.evaluation_results = None
        self.gatekeeper_results = None
        self.status = "pending"
        self.metadata = {}
    
    def _generate_id(self) -> str:
        """生成唯一因子ID"""
        hash_str = hashlib.md5(self.expression.encode()).hexdigest()[:12]
        timestamp = datetime.now().strftime("%Y%m%d")
        return f"factor_{timestamp}_{hash_str}"
    
    def _generate_name(self) -> str:
        """生成因子名称"""
        return f"auto_factor_{hashlib.md5(self.expression.encode()).hexdigest()[:8]}"
    
    def set_data(self, data: pd.Series):
        """设置因子数据"""
        self.data = data
        self.metadata['n_samples'] = len(data.dropna())
        self.metadata['coverage'] = len(data.dropna()) / len(data) if len(data) > 0 else 0
    
    def set_evaluation_results(self, results: Dict[str, Any]):
        """设置评估结果"""
        self.evaluation_results = results
        self.status = "evaluated"
    
    def set_gatekeeper_results(self, results: Dict[str, Any]):
        """设置审核结果"""
        self.gatekeeper_results = results
        self.status = "approved" if results.get('approved', False) else "rejected"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'factor_id': self.factor_id,
            'name': self.name,
            'expression': self.expression,
            'description': self.description,
            'source': self.source,
            'created_at': self.created_at.isoformat(),
            'status': self.status,
            'metadata': self.metadata,
            'evaluation_results': self.evaluation_results,
            'gatekeeper_results': self.gatekeeper_results
        }
    
    def save(self, directory: str = "factors"):
        """保存因子"""
        import os
        import json
        
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, f"{self.factor_id}.json")
        
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    @classmethod
    def load(cls, file_path: str) -> "FactorCandidate":
        """加载因子"""
        import json
        
        with open(file_path, "r") as f:
            data = json.load(f)
        
        candidate = cls(
            expression=data['expression'],
            name=data['name'],
            description=data.get('description', ""),
            source=data.get('source', "auto")
        )
        
        candidate.factor_id = data['factor_id']
        candidate.created_at = datetime.fromisoformat(data['created_at'])
        candidate.status = data['status']
        candidate.metadata = data.get('metadata', {})
        candidate.evaluation_results = data.get('evaluation_results')
        candidate.gatekeeper_results = data.get('gatekeeper_results')
        
        return candidate


class FactorStore:
    """
    因子存储
    
    管理已批准的因子：
    - 因子存储和加载
    - 因子查询
    - 因子版本管理
    """

    def __init__(self, store_path: str = "data/factor_store"):
        self.store_path = store_path
        self.factors = {}
        
        import os
        os.makedirs(store_path, exist_ok=True)
        self._load_existing_factors()
    
    def _load_existing_factors(self):
        """加载已存在的因子"""
        import os
        
        for filename in os.listdir(self.store_path):
            if filename.endswith(".pkl"):
                factor_id = filename.replace(".pkl", "")
                try:
                    factor = self.load_factor(factor_id)
                    self.factors[factor_id] = factor
                except Exception as e:
                    print(f"Failed to load factor {factor_id}: {e}")
    
    def save_factor(self, candidate: FactorCandidate):
        """保存因子"""
        import joblib
        
        factor_data = {
            'factor_id': candidate.factor_id,
            'name': candidate.name,
            'expression': candidate.expression,
            'data': candidate.data,
            'metadata': candidate.metadata,
            'evaluation_results': candidate.evaluation_results
        }
        
        file_path = os.path.join(self.store_path, f"{candidate.factor_id}.pkl")
        joblib.dump(factor_data, file_path)
        
        self.factors[candidate.factor_id] = factor_data
    
    def load_factor(self, factor_id: str):
        """加载因子"""
        import joblib
        
        file_path = os.path.join(self.store_path, f"{factor_id}.pkl")
        
        if os.path.exists(file_path):
            return joblib.load(file_path)
        return None
    
    def get_factor_data(self, factor_id: str) -> Optional[pd.Series]:
        """获取因子数据"""
        factor = self.load_factor(factor_id)
        return factor.get('data') if factor else None
    
    def list_factors(self) -> List[Dict[str, Any]]:
        """列出所有因子"""
        factor_list = []
        
        for factor_id, factor_data in self.factors.items():
            factor_list.append({
                'factor_id': factor_id,
                'name': factor_data.get('name'),
                'expression': factor_data.get('expression'),
                'ic': factor_data.get('evaluation_results', {}).get('rank_ic', {}).get('mean')
            })
        
        return sorted(factor_list, key=lambda x: x.get('ic', 0), reverse=True)
    
    def delete_factor(self, factor_id: str) -> bool:
        """删除因子"""
        import os
        
        file_path = os.path.join(self.store_path, f"{factor_id}.pkl")
        
        if os.path.exists(file_path):
            os.remove(file_path)
            if factor_id in self.factors:
                del self.factors[factor_id]
            return True
        return False


class FactorRegistry:
    """
    因子注册表
    
    记录所有因子的元数据：
    - 因子基本信息
    - 因子版本
    - 因子状态
    - 因子依赖
    """

    def __init__(self, registry_path: str = "data/factor_registry.json"):
        self.registry_path = registry_path
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """加载注册表"""
        import os
        import json
        
        if os.path.exists(self.registry_path):
            with open(self.registry_path, "r") as f:
                return json.load(f)
        return {'factors': {}, 'version': '1.0'}
    
    def register_factor(self, candidate: FactorCandidate):
        """注册因子"""
        self.registry['factors'][candidate.factor_id] = {
            'factor_id': candidate.factor_id,
            'name': candidate.name,
            'expression': candidate.expression,
            'source': candidate.source,
            'created_at': candidate.created_at.isoformat(),
            'status': candidate.status,
            'metadata': candidate.metadata
        }
        
        self._save_registry()
    
    def update_factor_status(self, factor_id: str, status: str):
        """更新因子状态"""
        if factor_id in self.registry['factors']:
            self.registry['factors'][factor_id]['status'] = status
            self._save_registry()
    
    def get_factor_info(self, factor_id: str) -> Optional[Dict[str, Any]]:
        """获取因子信息"""
        return self.registry['factors'].get(factor_id)
    
    def list_all_factors(self) -> List[Dict[str, Any]]:
        """列出所有已注册因子"""
        return list(self.registry['factors'].values())
    
    def get_factor_by_source(self, source: str) -> List[Dict[str, Any]]:
        """按来源获取因子"""
        return [
            factor for factor in self.registry['factors'].values()
            if factor.get('source') == source
        ]
    
    def _save_registry(self):
        """保存注册表"""
        import json
        
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=2)