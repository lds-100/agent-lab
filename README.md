# agent-lab

A research sandbox for training, evaluating, and understanding tool-using language-model agents.

# Project 0 — Tiny Tool-Using Agent

A small research project exploring whether reinforcement learning (RL) can improve an LLM's ability to make better tool-use decisions without changing the underlying model.

We use **Qwen 2.5 1.5B Instruct** and build a small environment around it.

## Research Question

**Can reinforcement learning improve a language model's sequential tool-use policy?**

Here, a **policy** simply means the model's decisions about what action to take next.

We are deliberately building the project in stages. The current work is the **untrained baseline**, not RL training.

```text
✅ Single-tool baseline
        ↓
✅ Two tools
        ↓
✅ Rewards + trajectories
        ↓
✅ Evaluation
        ↓
✅ Multi-step tasks
        ↓
🔜 RL environment
        ↓
🔜 RL training
        ↓
🔜 Compare trained vs. untrained
```

The goal is to first understand how well the untrained model uses tools, then measure whether training improves that behavior.

---
## What We Want to Learn

The project is not just about making the agent answer more questions correctly. It is designed to explore how reinforcement learning changes **tool-use decisions**.

As the environment becomes more complex, we want to understand:

* **Reward design:** How should correct, unnecessary, and invalid tool calls be rewarded?
* **Credit assignment:** If an agent makes several decisions before receiving a reward, which decisions deserve credit or blame?
* **Exploration:** How does the agent discover better tool-use strategies?
* **Policy learning:** Does training actually change which actions the model chooses?
* **Process vs. outcome rewards:** Is it better to reward only the final answer, or also individual steps?

The eventual goal is to train a small model in this environment and compare its learned behavior against the frozen untrained baseline.

We will start with a simple RL method such as **REINFORCE** before investigating more advanced methods such as **PPO** or **GRPO**.

---

## Current Architecture

The agent currently has two tools:

* Calculator
* Lookup

The basic interaction is:

```text
Task
 ↓
Qwen
 ↓
Choose tool
 ↓
Tool
 ↓
Observation
 ↓
Qwen
 ↓
Final answer
```

For multi-step tasks, Qwen can make another tool decision after receiving an observation:

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
Qwen → Final answer
```

This is the behavior we eventually want to improve through RL.

---

## Single-Step Baseline

We created a 10-task evaluation set covering calculator and lookup requests.

The untrained model currently achieves:

**9 / 10 correct — 90% average reward**

The initial reward is intentionally simple:

```text
correct tool → 1
incorrect/no tool → 0
```

This gives us a measurable **baseline**: the untrained model's performance that future versions can be compared against.

The baseline is frozen as:

```text
baseline_trajectories.jsonl
```

---

## Trajectories

A **trajectory** is the sequence of actions and observations produced during one task.

A single-step trajectory looks like:

```json
{
    "task": "What is 37 * 48?",
    "action": "CALL_CALCULATOR(37*48)",
    "tool": "calculator",
    "observation": 1776,
    "answer": "1776",
    "reward": 1
}
```

These records let us inspect what the agent did and will eventually provide the data structure needed for RL.

---

## Multi-Step Baseline

We created five tasks where the model must use a calculation to answer a follow-up question.

Example:

```text
What is 23 * 17, and is the result greater than 400?
```

A successful sequence could be:

```text
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

The untrained model currently achieves:

**2 / 5 rewarded — 40%**

The frozen multi-step baseline is:

```text
baseline_multistep_trajectories.jsonl
```

The current multi-step reward checks whether the final answer matches the expected `True` or `False` result.

---

## Dataset Generation

Evaluation and dataset generation are kept separate so that running an evaluation does not accidentally change our frozen baselines.

```text
evaluate.py
→ measures performance

generate_dataset.py
→ runs tasks and creates datasets

dataset.py
→ saves trajectories as JSONL
```

---

## Project Structure

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
* Build calculator and lookup tools
* Implement tool selection
* Execute tools and return observations to Qwen
* Generate final answers
* Build single-step evaluation set
* Implement basic rewards
* Represent and save trajectories
* Establish and freeze the 90% single-step baseline
* Build and test multi-step tasks
* Establish and freeze the multi-step baseline
* Separate evaluation from dataset generation

### Next

* Formalize the RL environment
* Define states, actions, observations, and rewards
* Improve multi-step task design
* Build the RL training loop
* Train the model
* Compare trained vs. untrained performance

---

## Roadmap

### 1. Build the environment

```text
Task
 ↓
Qwen
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

### 2. Introduce sequential decisions

Allow the model to make multiple tool-use decisions before producing a final answer.

**Initial version complete.**

### 3. Build the RL environment

Formalize what the model sees, what actions it can take, what observations it receives, and how rewards are assigned.

### 4. Train with RL

Eventually:

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

### 5. Compare

Compare the trained model against the frozen untrained baselines.

Key measurements:

* Correct tool selection
* Correct tool sequences
* Task completion
* Number of tool calls
* Invalid or unnecessary calls
* Reward
* Performance on unseen tasks

The central question remains:

> **Can reinforcement learning improve Qwen's sequential tool-use decisions?**
