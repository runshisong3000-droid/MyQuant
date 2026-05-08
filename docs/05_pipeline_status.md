# 05_pipeline_status.md

## Pipeline 状态

### 当前状态汇总

| Pipeline | 状态 | 说明 |
|----------|------|------|
| student_laptop_pipeline | ✅ 已跑通 | 基础研究 pipeline |
| neural_factor_pipeline | ✅ 已跑通 | 神经因子工程验证完成 |
| research_lite_pipeline | ✅ 可信度修复完成 | 第二阶段修复完成 |

### 详细状态

#### student_laptop_pipeline
- 使用 AkShare 获取真实 A 股数据
- 包含基本因子计算
- 支持简单回测
- 适合学生笔记本运行

#### neural_factor_pipeline
- 序列数据处理
- 神经网络 encoder
- embedding 转换为 neural factors
- 工程验证已通过

#### research_lite_pipeline
- 综合研究能力
- 可信度审计模块
- 因子评价系统
- **修复内容**：
  - target_alignment: FAIL → OK
  - formula factors: 0/100 → 100/100
  - 测试: 50/51 → 59/59

### 测试状态

```
总测试数：59
通过：59
失败：0
通过率：100%
```

### 下一阶段

| 阶段 | 名称 | 状态 |
|------|------|------|
| Phase 1 | AGENTS/docs/prompts | 进行中 |
| Phase 2 | Dashboard Visual MVP | 待开发 |
| Phase 3 | Neural Feature Learning 扩展 | 待开发 |
| Phase 4 | 交易约束完善 | 待开发 |
| Phase 5 | 样本外验证 | 待开发 |
| Phase 6 | Paper Trading | 待开发 |

### 注意事项

⚠️ **当前结果仍不能直接实盘**，只能用于研究和展示。

---

**更新日期：** 2026-05-08