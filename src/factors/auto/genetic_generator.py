"""
遗传编程生成器 - Genetic Generator

核心功能:
    - 遗传编程算法
    - 因子公式进化
    - 交叉和变异操作
    - 适应度评估
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any, Tuple
import random
import hashlib


class GeneticGenerator:
    """
    遗传编程生成器
    
    使用遗传算法进化因子公式：
    1. 初始化种群
    2. 评估适应度
    3. 选择优秀个体
    4. 交叉繁殖
    5. 变异
    6. 重复迭代
    """

    def __init__(self):
        self.fields = ['open', 'high', 'low', 'close', 'volume', 'amount', 'turnover', 'market_cap']
        self.unary_ops = ['rank', 'zscore', 'log', 'abs']
        self.binary_ops = ['delta', 'delay', 'rolling_mean', 'rolling_std', 'rolling_sum']
        self.comparison_ops = ['+', '-', '*', '/']
        self.windows = [3, 5, 10, 20, 60]
        
        self.population_size = 50
        self.generations = 10
        self.crossover_rate = 0.7
        self.mutation_rate = 0.2
    
    def generate_population(self, size: int = None) -> List[str]:
        """
        生成初始种群
        
        Args:
            size: 种群大小
            
        Returns:
            表达式列表
        """
        size = size or self.population_size
        population = []
        
        for _ in range(size):
            expr = self._generate_random_expression()
            population.append(expr)
        
        return population
    
    def _generate_random_expression(self) -> str:
        """生成随机表达式"""
        depth = random.randint(1, 3)
        return self._generate_expression(depth)
    
    def _generate_expression(self, depth: int) -> str:
        """递归生成表达式"""
        if depth == 1 or random.random() < 0.3:
            return random.choice(self.fields)
        
        if random.random() < 0.5:
            op = random.choice(self.unary_ops)
            arg = self._generate_expression(depth - 1)
            return f"{op}({arg})"
        else:
            op = random.choice(self.binary_ops)
            arg = random.choice(self.fields)
            window = random.choice(self.windows)
            return f"{op}({arg}, {window})"
    
    def evaluate_fitness(self, expression: str, factor_evaluator, data: pd.DataFrame, returns: pd.Series) -> float:
        """
        评估适应度
        
        适应度函数: score = IC均值 / IC标准差 - 相关性惩罚 - 换手惩罚 - 复杂度惩罚
        
        Args:
            expression: 因子表达式
            factor_evaluator: 因子评估器
            data: 原始数据
            returns: 收益率数据
            
        Returns:
            适应度分数
        """
        try:
            factor_data = self._evaluate_expression(expression, data)
            
            if factor_data is None or factor_data.isna().mean() > 0.3:
                return -1.0
            
            evaluation = factor_evaluator.evaluate_single(factor_data, returns)
            
            ic_mean = evaluation.get('rank_ic', {}).get('mean', 0)
            ic_std = evaluation.get('rank_ic', {}).get('std', 1)
            
            turnover = evaluation.get('turnover', 1)
            complexity = self._calculate_complexity(expression)
            
            ic_score = abs(ic_mean) / (ic_std + 0.01)
            turnover_penalty = min(turnover, 1) * 0.3
            complexity_penalty = complexity * 0.1
            
            score = ic_score - turnover_penalty - complexity_penalty
            
            return max(score, 0)
        
        except Exception as e:
            return -1.0
    
    def _evaluate_expression(self, expression: str, data: pd.DataFrame) -> Optional[pd.Series]:
        """评估表达式"""
        try:
            local_vars = {
                'data': data,
                'rank': lambda x: x.rank(pct=True),
                'zscore': lambda x: (x - x.mean()) / x.std(),
                'delta': lambda x, n: x.diff(n),
                'delay': lambda x, n: x.shift(n),
                'rolling_mean': lambda x, n: x.rolling(n).mean(),
                'rolling_std': lambda x, n: x.rolling(n).std(),
                'rolling_sum': lambda x, n: x.rolling(n).sum(),
                'log': lambda x: np.log(x.where(x > 0)),
                'abs': lambda x: abs(x)
            }
            
            for field in self.fields:
                if field in data.columns:
                    local_vars[field] = data[field]
            
            return eval(expression, {}, local_vars)
        except:
            return None
    
    def _calculate_complexity(self, expression: str) -> int:
        """计算表达式复杂度"""
        return expression.count('(')
    
    def select(self, population: List[str], fitness_scores: List[float]) -> List[str]:
        """
        选择操作
        
        使用轮盘赌选择或锦标赛选择
        
        Args:
            population: 种群
            fitness_scores: 适应度分数
            
        Returns:
            选中的个体
        """
        selected = []
        
        for _ in range(len(population)):
            idx1, idx2, idx3 = random.sample(range(len(population)), 3)
            best_idx = max([idx1, idx2, idx3], key=lambda i: fitness_scores[i])
            selected.append(population[best_idx])
        
        return selected
    
    def crossover(self, parent1: str, parent2: str) -> Tuple[str, str]:
        """
        交叉操作
        
        交换两个父代的子表达式
        
        Args:
            parent1: 父代1
            parent2: 父代2
            
        Returns:
            两个子代
        """
        if random.random() > self.crossover_rate:
            return parent1, parent2
        
        try:
            split_point1 = self._find_split_point(parent1)
            split_point2 = self._find_split_point(parent2)
            
            if split_point1 is None or split_point2 is None:
                return parent1, parent2
            
            child1 = parent1[:split_point1] + parent2[split_point2:]
            child2 = parent2[:split_point2] + parent1[split_point1:]
            
            return child1, child2
        except:
            return parent1, parent2
    
    def _find_split_point(self, expression: str) -> Optional[int]:
        """找到有效的分割点"""
        stack = []
        
        for i, char in enumerate(expression):
            if char == '(':
                stack.append(i)
            elif char == ')':
                if stack:
                    stack.pop()
                    if not stack:
                        return i + 1
        
        if '(' in expression:
            return expression.find('(') + 1
        return None
    
    def mutate(self, expression: str) -> str:
        """
        变异操作
        
        随机修改表达式的一部分
        
        Args:
            expression: 原始表达式
            
        Returns:
            变异后的表达式
        """
        if random.random() > self.mutation_rate:
            return expression
        
        try:
            if random.random() < 0.3:
                new_field = random.choice(self.fields)
                old_field = random.choice(self.fields)
                return expression.replace(old_field, new_field, 1)
            
            elif random.random() < 0.5:
                new_window = random.choice(self.windows)
                import re
                return re.sub(r', (\d+)', f', {new_window}', expression, 1)
            
            else:
                return self._generate_random_expression()
        
        except:
            return expression
    
    def run(
        self,
        factor_evaluator,
        data: pd.DataFrame,
        returns: pd.Series,
        generations: int = None,
        population_size: int = None
    ) -> Tuple[List[str], List[float]]:
        """
        运行遗传编程
        
        Args:
            factor_evaluator: 因子评估器
            data: 原始数据
            returns: 收益率数据
            generations: 代数
            population_size: 种群大小
            
        Returns:
            (最佳表达式列表, 适应度分数列表)
        """
        generations = generations or self.generations
        population_size = population_size or self.population_size
        
        population = self.generate_population(population_size)
        best_expressions = []
        best_scores = []
        
        for gen in range(generations):
            fitness_scores = [
                self.evaluate_fitness(expr, factor_evaluator, data, returns)
                for expr in population
            ]
            
            best_idx = np.argmax(fitness_scores)
            best_expr = population[best_idx]
            best_score = fitness_scores[best_idx]
            
            best_expressions.append(best_expr)
            best_scores.append(best_score)
            
            print(f"Generation {gen+1}/{generations}, Best Score: {best_score:.4f}, Best Expr: {best_expr}")
            
            selected = self.select(population, fitness_scores)
            
            new_population = []
            for i in range(0, len(selected), 2):
                parent1 = selected[i]
                parent2 = selected[i+1] if i+1 < len(selected) else selected[i]
                
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:population_size]
        
        return best_expressions, best_scores