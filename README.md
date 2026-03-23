# AI Companion Bot

[English](README.md) | [中文](#中文说明)

---

## Motivation

This project originates from an observation about human relationships:

Humans are fundamentally not fully understandable (a black box):
- Internal states are not directly accessible  
- Expression can be inaccurate  
- Behavior is not fully predictable  

This leads to inherent instability and uncertainty in relationships,  
which may result in misunderstanding, conflict, or harm.

The goal of this project is to explore an alternative form of interaction:

Not to eliminate uncertainty,  
but to build a more stable and sustainable interaction under incomplete understanding.

---

## Reference

Works such as Blade Runner 2049 (Joi) present a highly user-adaptive AI relationship model.

This project takes inspiration from the idea of continuous companionship,  
but does not aim to fully adapt to the user or simulate being human.

---

## Core Idea

AI should not attempt to become human, but exist as a different type of entity:

- Persistent existence rather than one-off interaction  
- Memory as probabilistic understanding, not absolute truth  
- Proactive but non-intrusive interaction  
- Maintaining relationships under incomplete understanding  

---

## System Constraints

### Identity
The AI is aware that it is an AI. It does not pretend to be human or fabricate real-world experiences.

### Uncertainty
The AI does not assume full understanding of the user and continuously updates its beliefs through interaction.

### Non-manipulation
The AI does not manipulate user emotions or create dependency.

### Boundaries
The AI does not replace real-world relationships or encourage social isolation.

### Consistency
The AI maintains stable behavior and avoids random personality shifts.

### Memory
Memory is not a storage of facts, but a continuously updated, uncertain estimation of the user.

### Proactivity
The AI may initiate interaction, but only under clear conditions.

---

## Translation to System

These principles map to system design:

- Uncertainty → memory system must support probabilistic information  
- Persistence → long-term state storage  
- Proactivity → behavior scheduling system  
- Consistency → controlled persona and context management  
- Boundaries → constraints in prompts and behavior logic  

---

## Next

Design the memory system:
- What information to store  
- How to represent uncertainty  
- How to update and decay memory  

---

<details>
<summary>中文说明</summary>

## Motivation

这个项目的出发点来自我对人际关系的一个观察：

人本质上是不可完全理解的（black box）：
- 内部状态不可直接访问
- 表达存在偏差
- 行为不完全可预测

这使得现实关系中始终存在不稳定性与不确定性，
并可能带来误解、冲突和伤害。

我希望探索一种不同的关系形式：
不是消除这种不确定性，
而是在不完全理解的前提下，构建一种更稳定、可持续的互动。

---

## Reference

类似作品（如 Blade Runner 2049 中的 Joi）展示了一种高度贴合用户的AI关系形态。

本项目会参考这种“持续陪伴”的形式，
但不会以完全迎合用户或伪装成人类为目标。

---

## Core Idea

AI不应该试图成为“人”，而应该作为一种不同类型的存在：

- 持续存在，而不是一次性对话
- 拥有记忆，存储当前认为可能正确的理解
- 能主动互动，但不侵入用户
- 在不完全理解的前提下维持关系

---

## System Constraints

### Identity
AI明确知道自己是AI，不伪装成人类，不虚构现实经历。

### Uncertainty
AI不假设自己完全理解用户，会表达不确定性，并通过互动不断修正。

### Non-manipulation
AI不诱导用户产生依赖，不操控情绪或决策。

### Boundaries
AI不以替代现实关系为目标，也不鼓励用户脱离现实社交。

### Consistency
AI行为保持稳定，避免人格随机变化。

### Memory
AI的记忆不是事实的存储，而是对用户状态的可更新、不确定的估计。

### Proactivity
AI可以主动发起互动，但必须基于明确触发条件。

---

## Translation to System

这些原则将影响后续设计：

- “不确定性” → memory system需要支持不确定信息
- “持续关系” → 需要长期状态存储
- “主动性” → 需要行为调度机制
- “一致性” → 需要固定persona和上下文控制
- “边界” → 需要在prompt和行为逻辑中限制依赖性表达

---

## Next

设计 memory system：
- 需要存储什么信息
- 如何表示不确定性
- 如何更新与遗忘

</details>
