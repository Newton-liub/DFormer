---
name: research-idea
description: Quickly records a new research idea into doc/ideas/ideas.md with content, motivation, related question and status, without turning it into a plan. Use when the user mentions a new idea, hypothesis, possible method, or something worth investigating later in this project.
---

# 研究 Idea 快速记录

## 何时使用

- 用户提到新想法、假设、可能的方法或值得以后调查的点：快速记录到 `doc/ideas/ideas.md`。
- 不因为一个 idea 自动生成方案、测试或实现任务。

## 记录格式

```markdown
### YYYY-MM-DD 一句话标题
- 状态：待研究 / 调查中 / 计划实现 / 已实现 / 否决 / 暂存
- 内容：
- 来源/动机：
- 关联问题：
- 链接：
- 处置：
```

## 规则

- 一条 idea 只写一段，保持简短，不展开成计划。
- 状态变化时就地更新该条；被否决时写一句原因；实现后链接对应实验或代码位置。
- 真正晋升为研究方向后，再转入 `doc/plans/`，并按“先定论文方向，再设计实验”的原则推进。
- idea 记录不代表授权，不改变实验与云资源边界。
