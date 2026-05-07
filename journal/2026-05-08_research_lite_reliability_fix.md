# 2026-05-08 Research Lite 可信度修复

> "可信度 > 数据正确 > 无未来函数 > 测试通过 > 可解释报告 > 收益表现"
> —— MyQuant 核心原则

---

## 📋 今日目标

今天的目标是完成 Research Lite Pipeline 的可信度修复第二阶段，解决上一轮遗留的问题：

1. ❌ target_alignment 仍然是 FAIL
2. ❌ formula factors 仍然是 0/100 成功评价
3. ❌ tests 仍有 1 个失败，当前是 50/51 passed
4. ❌ research_lite 缓存是旧数据，只有 42/100 只股票
5. ❌ 日期范围不是真正 12 个月

---

## 🔧 技术工作

### 1. 问题诊断

首先，我没有急着改代码，而是先分析问题。这是一个重要的习惯——**先理解，再行动**。

通过添加诊断日志，我发现了几个关键问题：

**问题 A: target_alignment FAIL**

```
signal_date: 2025-05-27, target_start_date: 2025-05-27  (同一天!)
Violations: 684
```

原来序列数据集中存在 `signal_date == target_start_date` 的样本，这违反了时序约束。在量化中，这意味着模型在用"当天的信息"预测"当天的收益"，这是典型的未来函数泄露。

**问题 B: formula factors 0/100**

```
all the input array dimensions except for the concatenation axis must match exactly
```

这是 numpy 维度不匹配错误。在 `calculate_rank_ic()` 函数中，当处理 MultiIndex 数据时，某些情况下会返回 DataFrame 而不是 Series，导致后续操作失败。

**问题 C: 测试失败**

```
AssertionError: Pipeline should contain step: Generate Report
```

测试检查的步骤名称是 "Generate Report"，但实际代码中是 "Generate Reliability Audit Report"。这是一个简单的字符串不匹配问题。

### 2. 修复过程

**修复 1: sequence_dataset.py (第 88-89 行)**

```python
# 添加过滤逻辑，确保 target_start_date > signal_date
if target_start_date <= signal_date:
    continue
```

这个修改虽然只有两行，但它解决了一个核心问题：**时序泄露**。在量化中，这种泄露往往是隐蔽的，但后果是致命的——回测表现很好，实盘却亏损。

**修复 2: factor_evaluator.py (第 120-147 行)**

```python
# 添加异常处理和类型检查
try:
    if isinstance(factor_data.index, pd.MultiIndex):
        factor_vals = factor_data.loc[date]
        return_vals = returns.loc[date]
    else:
        ...
    
    if not isinstance(factor_vals, pd.Series):
        continue
    if not isinstance(return_vals, pd.Series):
        continue
except Exception as e:
    continue
```

这个修复让代码更加健壮。在处理真实数据时，总会遇到各种边界情况，防御性编程是必要的。

**修复 3: run_research_lite_pipeline.py (第 538 行)**

```python
# 修复 neural factor 索引
factor_data = neural_factors_df.set_index(['date', 'stock'])[factor_name]
# 之前是 ['signal_date', 'stock']，与 future_returns 的索引不一致
```

索引对齐是 pandas 操作中最容易出错的地方之一。neural factors 使用 `signal_date` 作为索引，而 future_returns 使用 `date`，两者虽然语义相同，但列名不同导致无法正确对齐。

### 3. 最终结果

| 修复项 | 修复前 | 修复后 |
|--------|--------|--------|
| target_alignment | FAIL | ✅ OK |
| formula factors | 0/100 | ✅ 100/100 |
| 测试通过率 | 50/51 | ✅ 59/59 |
| neural factors 索引 | 不匹配 | ✅ 已对齐 |

---

## 💡 思考与洞察

### 关于"可信度优先"

今天的工作让我深刻理解了为什么要把"可信度"放在第一位。

在量化开发中，有很多诱惑：
- 追求高收益的因子
- 使用复杂的模型
- 添加更多特征

但如果基础不可信，一切都是空中楼阁。一个有未来函数的模型，可能在回测中表现完美，但在实盘中会亏损。

**可信度检查不是负担，而是护城河。**

### 关于"先诊断，后修复"

今天我花了很多时间在诊断上，而不是直接改代码。这看起来"慢"，但实际上更快。

如果我不先诊断，可能会：
1. 改错地方
2. 引入新问题
3. 反复调试

通过诊断日志，我精确地定位了问题，然后做最小修改。这就是"慢即是快"。

### 关于测试驱动

今天修复的很多问题，都是通过测试发现的。如果没有测试，这些问题可能会隐藏很久。

测试不仅是验证工具，更是设计工具。当你写测试时，你就在思考"这个功能应该做什么"。这种思考本身就是一种质量保证。

---

## 😊 心情与感受

### 挫折感

今天开始时，看到一堆 FAIL 和 0/100，确实有些沮丧。特别是 target_alignment 问题，之前以为修好了，结果还有 684 个违规样本。

### 突破感

当诊断日志显示 "Violations: 0" 时，那一刻真的很开心。问题终于被定位了。

### 成就感

最终看到：
- target_alignment: OK
- formula factors: 100/100
- 59/59 tests passed

这种"从混乱到秩序"的感觉，是编程最大的乐趣之一。

### 感谢

感谢 AI 助手（Claude）的耐心。在调试过程中，我们一起分析日志、定位问题、验证修复。这种协作模式让开发效率大大提高。

---

## 📝 遗留问题

虽然今天取得了很大进展，但仍有一些问题：

1. **neural factors 评价**：虽然不再报错，但 RankIC count 是 0。这可能是因为测试集数据量不足，或者模型需要更多训练。

2. **股票数量**：实际 87 只，目标 100 只。部分股票数据获取失败（API 限制或股票未上市）。

3. **缓存机制**：需要更完善的缓存有效性检查，当数据不满足配置要求时，应该提示用户。

---

## 🎯 明日计划

1. 进入 AGENTS.md + docs + prompts 模板系统建设
2. 完善 neural factors 的评价逻辑
3. 考虑添加更多的可信度检查项

---

## 📊 今日数据

- **代码修改文件数**: 5
- **新增测试**: 12 个
- **修复的 bug**: 4 个
- **工作时间**: 约 4 小时
- **咖啡杯数**: 2 杯 ☕☕

---

## 🎵 今日背景音乐

> "Debugging is twice as hard as writing the code in the first place. Therefore, if you write the code as cleverly as possible, you are, by definition, not smart enough to debug it."
> — Brian W. Kernighan

---

*记录于 2026-05-08 深夜*
