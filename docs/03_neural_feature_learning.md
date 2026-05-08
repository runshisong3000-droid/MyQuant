# 03_neural_feature_learning.md

## 神经特征学习

### 类视觉 AI 特征学习范式

```
Raw OHLCV → Sequence Tensor → Encoder → Embedding → Neural Factors → RankIC/ICIR → Gatekeeper → 可视化
```

### 流程说明

#### 1. Raw OHLCV
- 原始开盘价、最高价、最低价、收盘价、成交量
- 按股票分组的时间序列数据

#### 2. Sequence Tensor
- 将时间序列转换为固定长度的张量
- 每个样本包含 lookback_window 天的数据
- 形状：(num_samples, lookback_window, num_features)

#### 3. Encoder
- 使用神经网络对序列进行编码
- 支持 MLP、CNN、Transformer 等架构
- 提取序列的深层特征

#### 4. Embedding
- Encoder 的输出
- 低维稠密向量
- 包含原始序列的压缩信息

#### 5. Neural Factors
- 将 embedding 的每一维作为一个因子
- 每个维度代表一个学习到的特征
- 需要像普通因子一样进行评价

#### 6. RankIC/ICIR 评价
- 使用横截面 RankIC 评价每个 neural factor
- 计算 ICIR 衡量稳定性
- 只有通过评价的因子才能进入下一步

#### 7. Gatekeeper
- 审核所有 neural factors
- 过滤表现差的因子
- 确保没有未来函数

#### 8. 可视化
- 展示学习到的特征
- 解释神经网络学到了什么
- 帮助理解因子的经济意义

### 核心优势

1. **自动特征发现**：无需人工设计因子
2. **捕捉非线性关系**：神经网络擅长处理复杂模式
3. **数据驱动**：从数据中学习特征

---

**更新日期：** 2026-05-08