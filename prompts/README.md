# prompts/README.md

## 提示词模板目录

### 使用方式

以后你可以这样说：

- "按 bug_fix 模板执行"
- "按 audit 模板执行"
- "按 dashboard 模板执行"
- "按 explain_code 模板执行"
- "按 neural_feature_learning 模板执行"

### 模板列表

| 模板文件 | 用途 |
|----------|------|
| 01_bug_fix.md | 修复 bug |
| 02_reliability_audit.md | 可信度审计 |
| 03_new_pipeline.md | 创建新 pipeline |
| 04_dashboard.md | 可视化页面 |
| 05_explain_code.md | 代码解释 |
| 06_run_tests.md | 运行测试 |
| 07_neural_feature_learning.md | 神经特征学习 |

### 调用流程

1. 用户说一句话命令
2. AI 读取 AGENTS.md 了解项目上下文
3. AI 读取对应的模板文件
4. AI 按模板步骤执行

---

**更新日期：** 2026-05-08