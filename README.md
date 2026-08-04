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

The single-step baseline trajectories are frozen as:

```text
baseline_trajectories.jsonl
```

---

## Baseline Snapshot

Before beginning RL training, the current multi-step evaluation results are being preserved as a separate baseline artifact.

The baseline metrics are stored in:

```text
baselines/multistep_baseline_v1.json
```

This file records the performance of the **untrained Qwen 2.5 1.5B Instruct model** before any RL updates.

The baseline should not be overwritten during later experiments.

The baseline is intended to provide a fixed point of comparison:

```text
Frozen baseline
      ↓
RL training
      ↓
Evaluate checkpoint
      ↓
Compare against baseline
```

The current baseline evaluation uses 9 trajectories.

---

## Reward Version

The current efficiency reward used for the first RL experiment is designated:

```text
Reward version: v1
```

The reward implementation remains:

```text
calculate_multistep_efficiency_reward(...)
```

The function name describes its purpose; `v1` identifies the specific reward definition used for this experiment.

Once RL training begins, the reward definition should remain fixed for the duration of the experiment so that changes between checkpoints can be interpreted meaningfully.

Future changes to the reward should receive a new version, such as:

```text
Reward version: v2
Reward version: v3
```

---

## Frozen Evaluation Set

The current `MULTISTEP_TASKS` set is the frozen evaluation set for the first RL experiment.

It should remain unchanged during training.

```python
# Frozen evaluation set for RL experiment v1.
# Do not modify during training.
MULTISTEP_TASKS = [...]
```

Future unseen or generalization tasks should be placed in a separate evaluation set rather than modifying the frozen baseline tasks.

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

The project now includes a separate **9-task multi-step evaluation set** using a synthetic profile subject.

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

An additional calculator task is included in the current 9-trajectory evaluation run.

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

The current efficiency reward combines final-answer correctness with tool-use quality.

**Reward version: v1**

The current implementation rewards:

* correct final answers
* correct tool sequences
* correct tool arguments

and penalizes:

* missing required steps
* unnecessary calls
* invalid calls

Conceptually:

```text
R = R_correctness
    + 0.10 × tool_selection_correct
    + 0.10 × argument_correct
    - 0.05 × missing_required_steps
    - 0.05 × unnecessary_calls
    - 0.10 × invalid_calls
```

where:

```text
R_correctness =
    1 if judge_answer(...) == TRUE
    0 otherwise
```

This means final-answer correctness remains the largest component of the reward, while tool-use behavior provides additional signal.

For example, a trajectory with:

```text
ANSWER CORRECT: True
TOOL SELECTION: True
ARGUMENT CORRECT: True
```

receives:

```text
1.20
```

before any penalties.

A correct answer with poor tool behavior can still receive a lower reward because of missing, unnecessary, or invalid calls.

Importantly, **tool correctness does not override the final-answer judge**.

---

## Current Multi-Step Evaluation Run

The latest evaluation generated:

```text
9 trajectories
```

and saved them as:

```text
multistep_v2_trajectories_20260804_080714_profile_subject.jsonl
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
CALL_LOOKUP(profile subject birthplace)
CALL_LOOKUP(profile subject book)
```

being emitted as one malformed action, as well as repeated calls such as:

```text
CALL_LOOKUP(profile subject book)
```

The model also produced incorrect answers and sometimes returned internal tool-call text instead of an answer.

### Evaluator problems

The latest runs also showed that the evaluator itself still needs validation.

The semantic judge has previously produced false positives for answers containing incomplete or invalid information. This is important because a judge error could eventually become a reward error during RL.

The current tool-trajectory evaluator also treats malformed combined actions as invalid tool sequences. Therefore, the reported tool-selection and efficiency metrics should currently be treated as **diagnostic rather than final benchmark numbers**.

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
* Define Reward v1
* Freeze the current multi-step evaluation set for RL experiment v1
* Save the multi-step baseline trajectories
* Save the baseline metrics
* Run the first multi-step efficiency evaluation

### Current

* 🟡 Multi-step evaluator is functional but still needs validation
* 🟡 Semantic judge needs continued validation against false positives/negatives
* 🟡 Tool-trajectory parsing needs to handle malformed/combined actions more cleanly
* 🟡 Efficiency metrics are currently diagnostic
* 🟡 Reward v1 is implemented but has not been used for RL training
* 🟡 Frozen baseline artifacts are now being preserved for comparison
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
├── baselines/
│   └── multistep_baseline_v1.json
├── experiments/
│   └── rl_v1_results.jsonl
├── baseline_trajectories.jsonl
├── baseline_multistep_trajectories.jsonl
├── multistep_v2_trajectories_*.jsonl
└── README.md
```

The `baselines/` directory contains frozen baseline metrics.

The `experiments/` directory will contain checkpoint-level RL results.

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

## 3. Validate and freeze the evaluator

Before RL training, make sure that:

* final-answer correctness is reliable
* obvious tool-call outputs are rejected
* malformed actions are represented consistently
* unnecessary calls are counted correctly
* reward values match the intended equation
* evaluation results can be reproduced
* baseline metrics and trajectories are preserved

**Current priority / final pre-RL step.**

## 4. Formalize the RL environment

Define:

* **State:** task + previous actions + observations
* **Action:** tool call or final answer
* **Observation:** tool result
* **Terminal condition:** final answer or maximum step limit
* **Reward:** final correctness plus tool-use quality

## 5. Run a small RL experiment

Start with a small number of training updates rather than a long training run.

Evaluate the model periodically against the **same frozen evaluation set**.

Track:

* Final-answer accuracy
* Average reward
* Invalid calls
* Unnecessary calls

Store the results in:

```text
experiments/rl_v1_results.jsonl
```

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

At the current stage, the project has reached the point where the next major milestone is not more model experimentation. It is making the **evaluation/reward loop trustworthy enough to serve as an RL signal**, then running a small controlled RL experiment against the frozen baseline.
