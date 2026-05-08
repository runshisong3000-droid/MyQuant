# 01_architecture_overview.md

## 项目架构概览

### 目录结构

```
MyQuant/
├── src/                    # 源代码目录
│   ├── data/               # 数据获取与处理
│   ├── factors/            # 因子模块
│   │   ├── auto/           # 公式因子、自动因子
│   │   └── neural/         # 神经特征学习
│   ├── core/               # 核心引擎（回测、组合、交易成本）
│   ├── metrics/            # 绩效和风险指标
│   ├── strategy/           # 策略模块
│   ├── validation/         # 可信度验证
│   └── utils/              # 工具函数
├── scripts/                # 可执行脚本
├── tests/                  # 测试文件
├── reports/                # 实验报告
├── dashboard/              # 可视化页面
└── docs/                   # 文档目录
```

### 模块关系

| 模块 | 输入 | 输出 |
|------|------|------|
| `src/data` | 原始行情数据 | 清洗后的 DataFrame |
| `src/factors/auto` | 价格数据 | 公式因子 |
| `src/factors/neural` | 序列数据 | Neural factors |
| `src/core` | 因子、权重 | 回测结果 |
| `src/metrics` | 收益序列 | 绩效指标 |
| `src/validation` | 数据、因子 | OK/WARN/FAIL |

### Pipeline 关系

1. **student_laptop_pipeline**：基础研究 pipeline，适合学生笔记本
2. **neural_factor_pipeline**：神经因子 pipeline，工程验证完成
3. **research_lite_pipeline**：综合研究 pipeline，包含可信度审计

---

**更新日期：** 2026-05-08