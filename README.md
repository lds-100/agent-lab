# agent-lab

A research sandbox for training, evaluating, and understanding tool-using language-model agents.

# Project 0 — Tiny Tool-Using Agent

A small research project exploring whether reinforcement learning (RL) can improve an LLM's ability to make better tool-use decisions without changing the underlying model.

We use **Qwen 2.5 1.5B Instruct** and build a small environment around it.

## Research Question

**Can reinforcement learning improve a language model's sequential tool-use policy?**

Here, a **policy** simply means the model's decisions about what action to take next.

We are deliberately building the project in stages. The current work is still the **untrained baseline and evaluation stage**, not RL training.

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
🟡 Multi-step efficiency evaluation
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
* **Efficiency:** Can the model learn to solve tasks with fewer unnecessary tool calls?

The eventual goal is to train a small model in this environment and compare its learned behavior against the frozen untrained baseline.

We will start with a simple RL method such as **REINFORCE** before investigating more advanced methods such as PPO or GRPO.

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

# Multi-Step Evaluation

The project now includes a separate **8-task multi-step evaluation set** using a synthetic profile subject.

The tasks require combinations of:

* lookup
* lookup → lookup
* lookup → lookup → calculator
* lookup → lookup → lookup → calculator

Examples include:

```text
Where was the profile subject born, and what book did they write?

What book did the profile subject write, and when was it published?

When was the profile subject born, when did they die,
and how old were they when they died?

What book did the profile subject write, when was it published,
when did they die, and how many years after publication did they die?
```

The expected answers are:

```text
Portland, Oregon; The Glass Harbor
The Glass Harbor; 2015
1987; 2021
1987; 2021; 34
Portland, Oregon; The Glass Harbor; 2015
The Glass Harbor; 2015; 2021; 6
1987; Portland, Oregon; 2021; 34
The Glass Harbor; 2015; 2021; 6
```

These tasks are deliberately more difficult than the original two-step calculator tasks because the agent must maintain information across multiple tool calls and sometimes perform a final calculation.

---

## Final-Answer Judge

Multi-step correctness is currently evaluated using a **Qwen-based semantic judge** rather than exact string matching.

The evaluator:

1. Receives the task.
2. Receives the expected answer.
3. Receives the agent's final answer.
4. Checks whether all required facts are present.
5. Rejects obvious tool-call outputs and environment failures.
6. Returns exactly `TRUE` or `FALSE`.

The judge is intentionally strict about missing facts.

For example:

```text
EXPECTED:
1987; 2021; 34

AGENT:
The profile subject was born in 1987 and died in 2021.

→ FALSE
```

Whereas:

```text
EXPECTED:
1987; 2021; 34

AGENT:
The profile subject was born in 1987, died in 2021,
and was 34 years old when they died.

→ TRUE
```

The judge is currently an **evaluation aid**, not ground truth. Its decisions should be monitored for false positives and false negatives before being used as an RL reward signal.

---

# Multi-Step Efficiency Reward

The current efficiency reward combines final-answer correctness with penalties for inefficient tool use.

The reward is:

```text
R = R_correctness
    - 0.01 × unnecessary_calls
    - 0.05 × invalid_calls
```

where:

```text
R_correctness =
    1 if judge_answer(...) == TRUE
    0 otherwise
```

Therefore, correctness currently dominates efficiency:

```text
Correct answer, no penalties → 1.00

Correct answer + 2 unnecessary calls → 0.98

Incorrect answer + 2 unnecessary calls → -0.02

Incorrect answer + 1 unnecessary call → -0.01
```

This is intentional for the current experiment: **getting the task right is much more important than saving a small number of tool calls.**

Importantly, tool correctness does **not** override the final-answer judge.

A trajectory can therefore have:

```text
ANSWER CORRECT: True
TOOL SELECTION: False
REWARD: 0.98
```

The reward is primarily based on whether the final answer was judged correct, with tool-use penalties applied afterward.

---

## Current Multi-Step Evaluation Run

The latest evaluation generated:

```text
8 trajectories
```

and saved them as:

```text
multistep_v2_trajectories_20260803_120535_profile_subject.jsonl
```

The latest run exposed several important problems in the current agent/evaluator combination.

### Agent problems

The untrained model sometimes:

* combines multiple tool calls into a single malformed action
* repeats the same lookup unnecessarily
* calls the wrong lookup
* produces tool calls instead of a final answer
* performs incorrect calculations
* uses information from the environment incorrectly
* stops before completing all required subtasks

Examples from the latest run include:

```text
CALL_LOOKUP(profile subject birthplace)_CALL_LOOKUP(profile subject book)
```

and:

```text
CALL_LOOKUP(profile subject book)
```

repeated several times.

The model also produced incorrect answers such as:

```text
The book ... was published in 1948.
```

when the expected publication year was 2015.

### Evaluator problems

The latest run also showed that the evaluator itself still needs work.

For example, one trajectory produced:

```text
ANSWER:
INVALID_CALCULATOR_EXPRESSION: 2021 - AGE(profile subject death year)
```

but the Qwen judge marked the answer as correct.

This means the current semantic judge can still produce **false positives** despite the explicit rejection rules.

The tool-trajectory evaluator also currently treats malformed combined actions as invalid tool sequences, which means the reported tool-selection and efficiency metrics should be treated as **diagnostic rather than final benchmark numbers**.

This is an important distinction:

> The current evaluation is useful for finding failure modes, but it is not yet a clean RL-ready benchmark.

---

# Current Status

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
* Build multi-step tasks
* Implement multi-step trajectory recording
* Implement a semantic final-answer judge
* Add explicit rejection of obvious tool-call/non-answer outputs
* Add an efficiency reward
* Run the first multi-step efficiency evaluation
* Save multi-step evaluation trajectories

### Current

* 🟡 Multi-step evaluator is functional but still needs validation
* 🟡 Semantic judge can produce occasional false positives
* 🟡 Tool-trajectory parsing needs to handle malformed/combined actions more cleanly
* 🟡 Efficiency metrics are currently diagnostic
* 🟡 Reward design is implemented but has not been used for RL training
* 🔴 No model parameters have been updated yet
* 🔴 No RL training has been performed

---

# What We Have Learned So Far

The baseline is already exposing the type of behavior that makes this project interesting.

The model is not simply failing because it cannot answer questions. It can sometimes retrieve the correct information but still:

* take unnecessary actions
* repeat actions
* fail to combine observations correctly
* produce malformed tool calls
* stop too early
* perform calculations incorrectly
* output an internal tool call instead of an answer

This suggests that the eventual RL problem is genuinely about **sequential decision-making**, rather than simply teaching the model more facts.

The current results also demonstrate why the evaluator needs to be reliable before it becomes an RL reward signal.

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

This separation becomes increasingly important once RL training begins.

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
├── multistep_v2_trajectories_*.jsonl
└── README.md
```

---

# Roadmap

## 1. Build the environment

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

## 2. Introduce sequential decisions

Allow the model to make multiple tool-use decisions before producing a final answer.

**Initial version complete.**

## 3. Validate the evaluator

Before RL training, make sure that:

* final-answer correctness is reliable
* obvious tool-call outputs are rejected
* malformed actions are represented consistently
* unnecessary calls are counted correctly
* reward values match the intended equation
* evaluation results can be reproduced

**Current priority.**

## 4. Formalize the RL environment

Define:

* **State:** task + previous actions + observations
* **Action:** tool call or final answer
* **Observation:** tool result
* **Terminal condition:** final answer or maximum step limit
* **Reward:** final correctness plus efficiency penalties

## 5. Train with RL

Start with a simple method such as:

```text
REINFORCE
```

Then investigate:

```text
PPO
GRPO
```

if the environment and reward signal are stable enough.

## 6. Compare trained vs. untrained

The trained model should be compared against the frozen baseline on:

* Final-answer correctness
* Correct tool selection
* Correct tool sequences
* Task completion
* Number of tool calls
* Invalid calls
* Unnecessary calls
* Average reward
* Performance on unseen tasks

The key requirement is to preserve a **frozen untrained baseline** so that improvements are measurable.

---

# The Central Experiment

The final experiment is:

```text
Frozen Qwen
    ↓
Baseline evaluation
    ↓
Observe tool-use behavior

        versus

RL-trained Qwen
    ↓
Same evaluation
    ↓
Measure behavioral changes
```

The central question remains:

> **Can reinforcement learning improve Qwen's sequential tool-use decisions?**

At the current stage, the project has reached the point where the next major milestone is not more model experimentation. It is making the **evaluation/reward loop trustworthy enough to serve as an RL signal**.
