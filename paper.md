# A Small Experimental Environment for Studying Sequential Tool Use in Language Models

## Abstract

Language models can be augmented with external tools such as calculators, search systems, and databases. When tool use requires multiple decisions, however, the model must decide not only which tool to use, but also when to use it and how to use the resulting observation. This project develops a small controlled environment for studying these decisions.

We use **Qwen 2.5 1.5B Instruct** as the underlying language model and provide it with two simple tools: a calculator and a lookup tool. We first establish an untrained baseline on single-step and multi-step tasks. On a 10-task single-step evaluation set, the model selected the expected tool on 9 of 10 tasks, achieving a reward of 0.90. On a five-task multi-step evaluation set, the model received reward on 2 of 5 tasks, for an average reward of 0.40.

These results establish a small, reproducible baseline for future experiments investigating whether reinforcement learning can improve sequential tool-use decisions.

---

## 1. Introduction

Language models are increasingly used as agents that interact with external tools. Rather than producing an answer entirely from their internal parameters, an agent can decide to call a calculator, retrieve information, execute code, or interact with another external system.

Tool use therefore introduces a decision-making problem. A model must determine which action to take given the current task and the information obtained from previous actions.

For a simple task, this may require only one decision:

$$
\text{Task} \rightarrow \text{Tool choice} \rightarrow \text{Observation} \rightarrow \text{Answer}.
$$

For a multi-step task, the model must repeatedly make decisions:

$$
s_t \rightarrow a_t \rightarrow o_t \rightarrow s_{t+1},
$$

where $s_t$ represents the information available to the model at step $t$, $a_t$ is the selected action, and $o_t$ is the resulting observation.

This project investigates whether reinforcement learning could eventually improve this sequential tool-use behavior. Before applying reinforcement learning, however, we first need to establish how well an untrained model performs in a controlled environment.

The central research question is:

> **Can reinforcement learning improve a language model's sequential tool-use decisions?**

The present work addresses the first part of that question: building the environment and measuring the untrained baseline.

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

For multi-step tasks, the model may make another tool decision after receiving an observation:

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

The intended reasoning structure is:

$$
23 \times 17 = 391
$$

followed by the comparison:

$$
391 > 400 = \text{False}.
$$

The multi-step evaluation set contains five such tasks.

The current environment allows the model to make repeated tool calls, although the untrained model does not consistently use this ability correctly.

---

## 4. Reward

The initial single-step reward is intentionally simple.

Let $a$ be the tool selected by the model and $a^*$ be the expected tool. The reward is:

$$
R(a,a^*) =
\begin{cases}
1 & \text{if } a=a^* \\
0 & \text{otherwise}
\end{cases}
$$

Thus, for a set of $N$ evaluation tasks, the average reward is:

$$
\bar{R} = \frac{1}{N}\sum_{i=1}^{N} R_i.
$$

For multi-step tasks, the current reward checks whether the model's final answer matches the expected answer:

$$
R =
\begin{cases}
1 & \text{if the final answer is correct} \
0 & \text{otherwise}.
\end{cases}
$$

This reward is deliberately simple. It gives us a measurable baseline without introducing additional assumptions about how the model should behave internally.

More sophisticated rewards are outside the scope of the current experiment.

---

## 5. Trajectory Representation

Each agent run is stored as a trajectory.

A trajectory is the sequence of actions and observations produced while solving one task.

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
    "task": "...",
    "steps": [
        {
            "action": "CALL_CALCULATOR(...)",
            "tool": "calculator",
            "observation": 391
        }
    ],
    "answer": "...",
    "reward": 0
}
```

A trajectory can be represented more generally as:

$$
\tau =
(s_0,a_0,o_0,s_1,a_1,o_1,\ldots,s_T).
$$

Here:

* $s_t$ is the state at step $t$.
* $a_t$ is the action selected by the model.
* $o_t$ is the observation returned by the environment.
* $T$ is the final step of the episode.

This representation preserves the interaction history needed to analyze the model's behavior and provides a natural data structure for future reinforcement learning experiments.

---

## 6. Baseline Results

### 6.1 Single-Step Baseline

The untrained model was evaluated on 10 tasks.

The model selected the expected tool on 9 of the 10 tasks.

Therefore:

$$
\text{Tool-selection accuracy}
=
\frac{9}{10}
=
0.90
=
90\%.
$$

The average reward is also:

$$
\bar{R}
=
\frac{9}{10}
=
0.90.
$$

The model made one incorrect tool decision on the task:

> Who wrote Romeo and Juliet?

The expected tool was lookup, but the model did not select the lookup tool.

The resulting dataset has been frozen as the single-step baseline.

```text
baseline_trajectories.jsonl
```

---

### 6.2 Multi-Step Baseline

The untrained model was evaluated on five multi-step tasks.

The rewards were:

$$
0,;0,;1,;1,;0.
$$

Therefore:

$$
\bar{R}
=======

# \frac{0+0+1+1+0}{5}

# \frac{2}{5}

0.40.
$$

Thus the current multi-step baseline reward is **40%**.

The model's behavior illustrates an important difference between single-step and multi-step tasks.

For example, on one task the model correctly calculated:

$$
23 \times 17 = 391,
$$

but then produced an incorrect conclusion about whether $391 > 400$.

On another task, the model correctly calculated:

$$
12 \times 12 = 144
$$

and produced the expected answer.

The model also sometimes made redundant or inappropriate tool calls, demonstrating that having access to tools does not guarantee that the model will use them effectively across multiple steps.

The multi-step dataset has been frozen as the untrained multi-step baseline.

```text
baseline_multistep_trajectories.jsonl
```

---

## 7. Discussion

The current results establish two different baseline behaviors.

For single-step tool selection, the untrained model performs relatively well:

$$
90% \text{ tool-selection accuracy}.
$$

However, performance decreases substantially when the task requires multiple decisions:

$$
40% \text{ average reward}.
$$

This difference motivates the eventual reinforcement learning experiment.

The multi-step setting introduces a larger decision space. A model must not only select a useful tool, but potentially decide whether another tool call is necessary, what input to provide to that tool, and how to interpret the resulting observation.

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

A future trajectory can be represented as:

$$
\tau =
(s_0,a_0,o_0,s_1,a_1,o_1,\ldots,s_T,a_T,r).
$$

The eventual objective will be to learn a policy:

$$
\pi_\theta(a_t \mid s_t),
$$

which represents the probability of selecting action $a_t$ given the current state $s_t$.

This formalization will allow us to study how training changes the model's action-selection behavior.

Future experiments will investigate:

* More robust multi-step tasks
* More informative reward functions
* Unnecessary and invalid tool-call penalties
* Reinforcement learning methods such as REINFORCE
* Credit assignment across multiple tool-use decisions
* Exploration of alternative tool-use sequences
* Comparison of trained and untrained policies

The key outcome of interest will be whether training produces measurable improvements in sequential tool-use behavior.

---

## 10. Conclusion

We built a small tool-use environment around Qwen 2.5 1.5B Instruct with calculator and lookup tools. The environment supports both single-step and basic multi-step interactions.

The untrained model achieved:

$$
\text{Single-step accuracy} = 90%
$$

and:

$$
\text{Multi-step average reward} = 40%.
$$

These results establish frozen baselines for future reinforcement learning experiments.

The next stage is to formalize the sequential decision structure and investigate whether reinforcement learning can improve the model's tool-use policy.

The broader goal is to understand whether a small language model can learn better sequential tool-use behavior through reinforcement learning, rather than simply improving its general ability to answer questions.
