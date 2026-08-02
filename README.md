# agent-lab
A research sandbox for training, evaluating, and understanding tool-using language-model agents.

# Project 0 — Tiny Tool-Using Agent

A small research project exploring whether reinforcement learning can improve an LLM's sequential tool-use decisions without changing the underlying model.

The goal is not to make a tiny model generally smarter. Instead, we want to understand how much we can improve a model's performance on a specific tool-use task through environment design, rewards, evaluation, and eventually RL training.

## Research Question

**Can reinforcement learning improve a language model's sequential tool-use policy?**

We use **Qwen 2.5 1.5B Instruct** as the underlying model and build a small environment around it.

---

## Current Architecture

The agent currently has two tools:

* Calculator
* Lookup

The basic interaction is:

```text
             Task
               │
               ▼
             Qwen
               │
          choose action
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
```

The agent can now also make multiple tool-use decisions within a single episode.

A multi-step episode looks like:

```text
Task
 ↓
Qwen → Tool
 ↓
Observation
 ↓
Qwen → Tool
 ↓
Observation
 ↓
Qwen → Final Answer
```

The model is not trained yet. We are currently measuring the behavior of the untrained baseline.

---

## Single-Step Baseline

We created a 10-task evaluation set covering calculator and lookup requests.

Current untrained baseline:

**9 / 10 correct — 90% average reward**

The initial reward is intentionally simple:

```text
correct tool → 1
incorrect/no tool → 0
```

This measures whether the untrained model selects the appropriate tool.

The baseline has been frozen as:

```text
baseline_trajectories.jsonl
```

This file is version-controlled and should not be modified during later experiments.

---

## Trajectory Representation

Each single-step agent run is represented as:

```json
{
    "task": "...",
    "action": "...",
    "tool": "...",
    "observation": "...",
    "answer": "...",
    "reward": 1
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
1776

Reward:
1
```

This representation gives us structured records of the agent's behavior.

---

## Multi-Step Baseline

We introduced a small set of five tasks requiring the model to reason about the result of a previous calculation.

Example:

```text
What is 23 * 17, and is the result greater than 400?
```

A successful sequence could look like:

```text
Task
 ↓
CALL_CALCULATOR(23*17)
 ↓
391
 ↓
CALL_CALCULATOR(391 > 400)
 ↓
False
 ↓
Final answer
```

The untrained model does not consistently make the correct sequence of decisions. Some tasks receive one tool call, some receive multiple calls, and some calls are redundant or incorrect.

The current five-task untrained multi-step baseline achieved:

**2 / 5 rewarded — 40%**

The multi-step baseline has been frozen as:

```text
baseline_multistep_trajectories.jsonl
```

This is also version-controlled and should not be modified.

The multi-step reward currently measures final-answer correctness using a simple `True` / `False` or `Yes` / `No` check.

---

## Dataset Structure

Dataset writing is separated from evaluation.

### `dataset.py`

Responsible for saving trajectories to JSONL.

### `generate_dataset.py`

Responsible for running tasks, calculating rewards, and generating datasets.

### `evaluate.py`

Responsible only for evaluating model performance and printing metrics.

This separation prevents ordinary evaluation runs from accidentally modifying frozen baseline datasets.

---

## Project Structure

Current structure:

```text
agent-lab/
├── agent.py
├── model.py
├── env.py
├── reward.py
├── dataset.py
├── generate_dataset.py
├── evaluate.py
├── baseline_trajectories.jsonl
├── baseline_multistep_trajectories.jsonl
└── README.md
```

---

## Current Status

### Completed

* Load Qwen 2.5 1.5B Instruct
* Separate model loading from agent execution
* Build calculator tool
* Add lookup tool
* Implement tool-selection protocol
* Execute selected tools
* Feed tool observations back to Qwen
* Generate final answers
* Build single-step evaluation set
* Implement single-step reward
* Represent agent trajectories
* Establish 90% single-step baseline
* Save and freeze single-step baseline trajectories
* Introduce multi-step agent episodes
* Build five-task multi-step evaluation set
* Implement initial multi-step reward
* Save and freeze multi-step baseline trajectories
* Separate evaluation from dataset generation

### Not Yet Started

* Formal RL state/action/observation interface
* More robust multi-step task design
* More sophisticated reward design
* RL training loop
* Train the model
* Compare trained vs. untrained policies

---

## Roadmap

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

**Complete.**

---

### 2. Trajectory Dataset

Save agent interactions as structured data:

```text
state → action → observation → reward
```

**Initial dataset work is complete.**

We now have frozen single-step and multi-step untrained datasets.

---

### 3. Multi-Step Environment

Allow the agent to make repeated decisions:

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

A basic two-step environment has been implemented and tested.

The next step is to formalize the RL episode representation and state/action interface.

---

### 4. RL Training

Eventually, the environment will produce trajectories such as:

```text
State
 ↓
Action
 ↓
Observation
 ↓
Next State
 ↓
Action
 ↓
...
 ↓
Terminal State
 ↓
Reward
```

The model's policy will then be trained using those rewards.

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

---

### 5. Evaluation

Compare the trained model against the frozen untrained baselines.

Important metrics will include:

* Tool-selection accuracy
* Task success rate
* Sequential decision accuracy
* Number of tool calls
* Invalid tool calls
* Redundant tool calls
* Reward
* Performance on unseen tasks

The central question is whether training produces a measurable improvement in tool-use behavior, rather than simply making the model better at answering questions in general.

---

## Philosophy

Keep the environment deliberately small.

The purpose of Project 0 is to understand the mechanics of:

```text
model
 ↓
environment
 ↓
action
 ↓
observation
 ↓
reward
 ↓
learning
```

before moving to larger models, more complicated tools, or sophisticated RL algorithms.

The important experimental principle is to **freeze the untrained baselines before training** so that later improvements can be measured against a stable reference.
