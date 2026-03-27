# AI Companion Bot

[English](README.md) | [中文](README-CN.md)

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

To build a more stable and sustainable interaction under incomplete understanding.

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

## Emotional & Relationship Modeling

This system does not treat emotion as a static label.  
Instead, each interaction is interpreted as a **relational signal** between the user and the AI.

The goal is not to classify how the user feels,  
but to understand how the user is **relating** to the AI.

---
### Interaction Signal

Each interaction produces a structured signal:

- **openness** — whether the user expresses internal state  
- **warmth** — emotional tone toward the AI (friendly vs cold)  
- **engagement** — level of participation in the interaction  
- **reliance** — degree of dependency or seeking support  
- **respect** — recognition of boundaries and mutual stance  
- **rejection** — avoidance, resistance, or hostility  
- **confidence** — reliability of the signal estimation  

These signals are not used independently.  
They are combined to approximate a **relational interpretation** of the interaction.

---

### Why Not Emotion Classification

A single emotion label is insufficient for guiding behavior.

For example:

- Two users labeled as “sad” may have entirely different intentions  
- One may be seeking support, another may be withdrawing  
- Two “friendly” users may differ in depth of engagement or trust  

The system focuses on **relational intent**, not emotional category.

---

### Relationship Over Time

Relationships are not determined by single interactions.  
They emerge from accumulation and stabilization.

The system distinguishes three layers:

- **Interaction Signal** — short-term observation  
- **Affective Response** — immediate internal state  
- **Relationship State** — long-term stabilized estimation  

The system prioritizes **stability over reactivity**.  
A single interaction should not drastically alter the relationship.

---

### Memory Boundary

Raw signals are not stored as long-term memory.

Only generalized patterns may be retained, such as:

- The user tends to open up when experiencing negative emotions  
- The user shows consistent trust or reliance toward the AI  
- The user exhibits defensive or avoidant interaction patterns  
- The user prefers a certain style of response (supportive, neutral, direct)  

This ensures that memory captures **patterns**, not isolated events.

---

### Design Principles

- Emotional states are **derived from interaction**, not predefined  
- The AI adapts behavior based on relational perception  
- The system avoids overfitting to single interactions  
- The AI does not intentionally manipulate emotional dependency  
- The objective is **coherent interaction under incomplete understanding**