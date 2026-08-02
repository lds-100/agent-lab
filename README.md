# agent-lab
A research sandbox for training, evaluating, and understanding tool-using language-model agents.

# Project 0 — Tiny Tool-Using Agent

A small research project exploring whether **reinforcement learning can improve an LLM's sequential tool-use decisions** without changing the underlying model.

The goal is not to make a tiny model generally smarter. Instead, we want to understand how much we can improve a model's performance on a **specific tool-use task** through environment design, rewards, evaluation, and eventually RL training.

## Research Question

> **Can reinforcement learning improve a language model's sequential tool-use policy?**

We use **Qwen 2.5 1.5B Instruct** as the underlying model and build a small environment around it.

---

## Current Architecture

The current agent has two tools:

* Calculator
* Lookup

The basic loop is:

```text
             Task
               │
               ▼
             Qwen
               │
          choose tool
               │
        ┌──────┴──────┐
        ▼             ▼
   Calculator       Lookup
        │             │
        └──────┬──────┘
               │
          Observation
               │
               ▼
             Qwen
               │
               ▼
          Final Answer
               │
               ▼
             Reward
```

This gives us our first sequential tool-use environment.

---

## Current Baseline

We created a small 10-task evaluation set covering calculator and lookup requests.

Current untrained baseline:

```text
9 / 10 correct
90% average reward
```

The reward is currently simple:

```python
correct tool → 1
incorrect/no tool → 0
```

This is intentionally simple. The goal at this stage is to establish a measurable baseline before introducing more complicated rewards or training.

---

## Trajectories

Each agent run is represented as a trajectory:

```python
{
    "task": task,
    "action": action,
    "tool": tool,
    "observation": result,
    "answer": final_answer,
}
```

For example:

```text
Task:
What is 37 * 48?

Action:
CALL_CALCULATOR(37*48)

Tool:
calculator

Observation:
1776

Answer:
The final answer is 1776.

Reward:
1
```

This representation will eventually become the basis for RL training data.

---

## Project Status

### Completed

* [x] Load Qwen 2.5 1.5B
* [x] Separate model loading from agent execution
* [x] Build calculator tool
* [x] Add lookup tool
* [x] Implement tool-selection protocol
* [x] Execute selected tools
* [x] Feed tool observations back to Qwen
* [x] Generate a final answer
* [x] Build baseline evaluation set
* [x] Implement basic reward function
* [x] Represent agent trajectories
* [x] Establish 90% baseline performance

### Not Yet Started

* [ ] Persist trajectories as a dataset
* [ ] Build genuinely multi-step tasks
* [ ] Design richer reward functions
* [ ] Build RL training loop
* [ ] Train the model
* [ ] Compare trained vs. untrained policies

---

## Roadmap

The project will progress in stages.

### 1. Baseline Environment

```text
Task
 ↓
Tool choice
 ↓
Tool
 ↓
Observation
 ↓
Final answer
 ↓
Reward
```

This stage is mostly complete.

### 2. Trajectory Dataset

Save agent interactions as structured data:

```text
state → action → observation → reward
```

This gives us persistent data that can be inspected and eventually used for training.

### 3. Multi-Step Environment

Introduce tasks where one tool call is not enough:

```text
Task
 ↓
Qwen → Tool A
 ↓
Observation
 ↓
Qwen → Tool B
 ↓
Observation
 ↓
Qwen → Final answer
 ↓
Reward
```

This is where sequential decision-making becomes substantially more interesting.

### 4. RL Training

Train the model's policy using rewards from the environment.

The eventual loop is:

```text
Task
 ↓
Qwen
 ↓
Action
 ↓
Tool
 ↓
Observation
 ↓
Qwen
 ↓
...
 ↓
Final answer
 ↓
Reward
 ↓
RL update
 ↓
Improved policy
```

### 5. Evaluation

Compare the trained model against the original untrained baseline.

Important metrics will include:

* Tool-selection accuracy
* Task success rate
* Number of tool calls
* Invalid tool calls
* Reward
* Performance on unseen tasks

The central question is whether training produces a measurable improvement in **tool-use behavior**, rather than simply making the model better at answering questions in general.

---

## Philosophy

Keep the environment deliberately small.

The purpose of Project 0 is to understand the mechanics of:

**model → environment → action → observation → reward → learning**

before moving to larger models, more complicated tools, or sophisticated RL algorithms.
