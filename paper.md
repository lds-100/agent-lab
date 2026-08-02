# A Small Experimental Environment for Studying Sequential Tool Use in Language Models

## Abstract

Language models can be augmented with external tools such as calculators, search systems, and databases. When tool use requires multiple decisions, the model must decide which tool to use, when to use it, and how to use the resulting information.

This project develops a small, controlled environment for studying these decisions.

We use **Qwen 2.5 1.5B Instruct** as the underlying language model and provide it with two simple tools: a calculator and a lookup tool. We establish an untrained baseline on single-step and multi-step tasks.

On a 10-task single-step evaluation set, the model selected the expected tool on 9 of 10 tasks, achieving 90% average reward. On a five-task multi-step evaluation set, the model received reward on 2 of 5 tasks, achieving 40% average reward.

These results establish a small, reproducible baseline for future experiments investigating whether reinforcement learning can improve sequential tool-use decisions.

---

## 1. Introduction

Language models are increasingly used as agents that interact with external tools. Rather than producing an answer entirely from their internal parameters, an agent can decide to call a calculator, retrieve information, or interact with another external system.

Tool use therefore introduces a decision-making problem. A model must determine which action to take given the current task and the information obtained from previous actions.

For a simple task, the interaction may look like:

```text
Task → Tool choice → Observation → Answer
```

For a multi-step task, the model must repeatedly make decisions:

```text
State → Action → Observation → Next state
```

where:

* **State** is the information currently available to the model.
* **Action** is the decision made by the model, such as selecting a tool.
* **Observation** is the result returned by the tool.

This project investigates whether reinforcement learning could eventually improve this sequential tool-use behavior.

Before applying reinforcement learning, however, we first need to establish how well an untrained model performs in a controlled environment.

The central research question is:

> **Can reinforcement learning improve a language model's sequential tool-use decisions?**

The present work focuses on building the environment and measuring the untrained baseline.

---

## 2. Experimental Environment

### 2.1 Language Model

We use **Qwen 2.5 1.5B Instruct** as the underlying language model.

The model is responsible for interpreting the task, selecting an action, processing tool observations, and producing the final answer.

No reinforcement learning training has been performed in the current experiment.

### 2.2 Tools

The environment provides two tools:

1. **Calculator** — evaluates arithmetic expressions.
2. **Lookup** — returns information for a requested topic.

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

For multi-step tasks, the model can make another tool decision after receiving an observation:

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

At this stage, the environment is deliberately small. The purpose is to isolate tool-use behavior before introducing more complicated tools or training procedures.

---

## 3. Task Design

We use two types of tasks.

### 3.1 Single-Step Tasks

Single-step tasks require the model to select the appropriate tool.

Examples include:

> What is 23 * 17?

and:

> Who was Albert Einstein?

The expected action for the first task is the calculator, while the expected action for the second is the lookup tool.

The single-step evaluation set contains 10 tasks:

* 5 calculator tasks
* 5 lookup tasks

### 3.2 Multi-Step Tasks

Multi-step tasks require the model to use information obtained during the task to produce a final answer.

For example:

> What is 23 * 17, and is the result greater than 400?

The intended calculation is:

```text
23 * 17 = 391
```

The resulting comparison is:

```text
391 > 400 → False
```

The multi-step evaluation set contains five such tasks.

The current environment allows the model to make repeated tool calls, although the untrained model does not consistently use this ability correctly.

---

## 4. Reward

The initial single-step reward is intentionally simple.

If the model selects the expected tool:

```text
Reward = 1
```

Otherwise:

```text
Reward = 0
```

For a set of N evaluation tasks, average reward is:

```text
Average reward = total reward / number of tasks
```

For example, if the model receives rewards:

```text
1, 1, 1, 0, 1
```

then:

```text
Average reward = 4 / 5 = 0.80
```

For multi-step tasks, the current reward checks whether the model's final answer matches the expected answer:

```text
Correct final answer → Reward 1
Incorrect final answer → Reward 0
```

This reward is deliberately simple. It gives us a measurable baseline without introducing additional assumptions about how the model should behave internally.

More sophisticated rewards are outside the scope of the current experiment.

---

## 5. Trajectory Representation

A **trajectory** is the sequence of actions and observations produced while solving one task.

For a single-step task, the representation is:

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

For multi-step tasks, the trajectory contains multiple steps:

```json
{
    "task": "What is 23 * 17, and is the result greater than 400?",
    "steps": [
        {
            "action": "CALL_CALCULATOR(23*17)",
            "tool": "calculator",
            "observation": 391
        }
    ],
    "answer": "False",
    "reward": 1
}
```

This representation preserves the interaction history needed to analyze the model's behavior and provides a natural data structure for future reinforcement learning experiments.

---

## 6. Baseline Results

### 6.1 Single-Step Baseline

The untrained model was evaluated on 10 tasks.

The model selected the expected tool on 9 of the 10 tasks.

Therefore:

```text
Tool-selection accuracy = 9 / 10 = 90%
Average reward = 9 / 10 = 0.90
```

The model made one incorrect tool decision on:

> Who wrote Romeo and Juliet?

The expected tool was lookup, but the model did not select the lookup tool.

The resulting dataset has been frozen as:

```text
baseline_trajectories.jsonl
```

---

### 6.2 Multi-Step Baseline

The untrained model was evaluated on five multi-step tasks.

The observed rewards were:

```text
0, 0, 1, 1, 0
```

Therefore:

```text
Average reward = (0 + 0 + 1 + 1 + 0) / 5
               = 2 / 5
               = 0.40
               = 40%
```

The model's behavior illustrates an important difference between single-step and multi-step tasks.

For example, on one task the model correctly calculated:

```text
23 * 17 = 391
```

but then produced an incorrect conclusion about whether 391 is greater than 400.

On another task, the model correctly calculated:

```text
12 * 12 = 144
```

and produced the expected answer.

The model also sometimes made redundant or inappropriate tool calls. This demonstrates that having access to tools does not guarantee that the model will use them effectively across multiple steps.

The multi-step dataset has been frozen as:

```text
baseline_multistep_trajectories.jsonl
```

---

## 7. Discussion

The current results establish two different baseline behaviors.

For single-step tool selection, the untrained model performs relatively well:

```text
90% tool-selection accuracy
```

However, performance decreases substantially when the task requires multiple decisions:

```text
40% average reward
```

This difference motivates the eventual reinforcement learning experiment.

The multi-step setting introduces a larger decision space. The model must not only select a useful tool, but may also need to decide whether another tool call is necessary, what input to provide to the tool, and how to interpret the resulting observation.

Importantly, the current results do **not** demonstrate that reinforcement learning improves tool use. No RL training has yet been performed.

Instead, they establish the reference point against which future training can be evaluated.

---

## 8. Limitations

This experiment is intentionally small.

The current environment has only two tools and a small number of evaluation tasks. The tasks are also relatively simple compared with real-world agent tasks.

The reward functions are binary and primarily measure tool selection or final-answer correctness. They do not yet distinguish between efficient and inefficient tool use.

The current experiment also does not evaluate generalization to a larger unseen task distribution.

These limitations are intentional at this stage. The goal is to create a simple environment in which changes in tool-use behavior can be measured clearly.

---

## 9. Future Work

The next stage is to formalize the environment as a reinforcement learning problem.

A future trajectory can be represented conceptually as:

```text
State
 ↓
Action
 ↓
Observation
 ↓
Next state
 ↓
Action
 ↓
Observation
 ↓
...
 ↓
Final reward
```

The eventual goal is to learn a policy: a rule that determines which action the model should take given its current state.

Future experiments will investigate:

* More robust multi-step tasks
* More informative reward functions
* Unnecessary and invalid tool-call penalties
* Reinforcement learning methods such as REINFORCE
* Credit assignment across multiple tool-use decisions
* Exploration of alternative tool-use sequences
* Comparison of trained and untrained policies

The key outcome will be whether training produces measurable improvements in sequential tool-use behavior.

---

## 10. Conclusion

We built a small tool-use environment around Qwen 2.5 1.5B Instruct with calculator and lookup tools. The environment supports both single-step and basic multi-step interactions.

The untrained model achieved:

```text
Single-step:
9 / 10 correct
90% average reward

Multi-step:
2 / 5 rewarded
40% average reward
```

These results establish frozen baselines for future reinforcement learning experiments.

The next stage is to formalize the sequential decision structure and investigate whether reinforcement learning can improve the model's tool-use behavior.
