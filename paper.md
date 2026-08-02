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

## 1.1 Motivation

A central motivation for this project is the **credit assignment problem** in reinforcement learning.

A tool-using agent may make many sequential decisions before receiving a final reward:

$$
s_0 \rightarrow a_0 \rightarrow s_1 \rightarrow a_1 \rightarrow \cdots \rightarrow s_T \rightarrow R
$$

where $s_t$ is the state, $a_t$ is the action, and $R$ is the final reward.

If the agent succeeds, which earlier decisions contributed to that success? If it fails, which decisions contributed to the failure? This becomes increasingly difficult as tasks become longer and rewards become more delayed.

We begin with a deliberately small version of this problem: an agent that selects tools, receives observations, and makes sequential decisions. The current work establishes the environment, trajectory representation, rewards, and untrained baselines needed to study credit assignment later.

The longer-term question is:

> **When an agent receives a reward only after a sequence of tool-use decisions, how can reinforcement learning improve those decisions?**


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

Let $a$ be the tool selected by the model and $a_{\text{expected}}$ be the expected tool. The reward is:

$$
R(a,a_{\text{expected}}) =
\begin{cases}
1 & \text{if } a = a_{\text{expected}} \
0 & \text{otherwise}
\end{cases}
$$

Thus, for a set of $N$ evaluation tasks, the average reward is:

$\bar{R} = \frac{1}{N}\sum_{i=1}^{N}R_i$.

For multi-step tasks, the current reward checks whether the model's final answer matches the expected answer:

$$
R =
\begin{cases}
1 & \text{if the final answer is correct} \
0 & \text{otherwise}
\end{cases}
$$

This reward is deliberately simple. In particular, the current multi-step reward does **not** penalize unnecessary tool calls. As a result, an agent can receive a reward of 1 even if it reaches the correct answer through an inefficient sequence of actions.

This distinction is important for the eventual reinforcement learning experiment. A richer reward could distinguish between simply completing a task and completing it efficiently.

---

## 6. Baseline Results

### 6.1 Single-Step Baseline

The untrained model was evaluated on 10 tasks.

The model selected the expected tool on 9 of the 10 tasks.

Therefore, the tool-selection accuracy is:

$9/10 = 0.90 = 90%$.

The average reward is:

$\bar{R} = 9/10 = 0.90$.

The model made one incorrect tool decision on the task:

> Who wrote Romeo and Juliet?

The expected tool was lookup, but the model did not select the lookup tool.

The resulting dataset has been frozen as:

`baseline_trajectories.jsonl`

---

### 6.2 Multi-Step Baseline

The untrained model was evaluated on five multi-step tasks.

The rewards were:

$0, 0, 1, 1, 0$.

Therefore, the average reward is:

$\bar{R} = (0 + 0 + 1 + 1 + 0)/5 = 2/5 = 0.40$.

Thus, the current multi-step baseline reward is **40%**.

The trajectories reveal several distinct failure modes.

#### Incorrect reasoning after a correct calculation

For the task:

> What is 23 * 17, and is the result greater than 400?

the model correctly calculated:

$23 \times 17 = 391$.

However, it then repeated the same calculator call and ultimately produced the incorrect conclusion that 391 was greater than 400.

The calculator observation was therefore correct, but the model failed to use that observation correctly when producing the final answer.

#### Incorrect interpretation of a comparison

For:

> What is 100 - 37, and is the result less than 70?

the model correctly calculated:

$100 - 37 = 63$.

It then produced the incorrect conclusion that 63 was not less than 70.

Again, the arithmetic tool produced the correct observation, but the final reasoning was incorrect.

#### Correct multi-step reasoning

For:

> What is 12 * 12, and is the result equal to 144?

the model calculated:

$12 \times 12 = 144$

and produced the correct answer.

This task received reward 1.

#### Unnecessary lookup after a calculator observation

A particularly notable behavior occurred on:

> What is 500 / 10, and is the result greater than 40?

The model first calculated:

$500 / 10 = 50$.

It then made a lookup call to determine whether 50 was greater than 40.

This lookup call was unnecessary because the calculator observation already contained sufficient information to answer the question:

$50 > 40 \rightarrow \text{True}$.

The final answer was correct, so the trajectory received reward 1 under the current reward function.

This exposes a limitation of the current reward design: **correctness alone does not distinguish efficient tool use from unnecessary tool use.**

#### Another unnecessary lookup attempt

For:

> What is 15 + 25, and is the result less than 50?

the model correctly calculated:

$15 + 25 = 40$.

However, instead of directly comparing 40 with 50, it produced:

`CALL LOOKUP(result < 50)`

and did not complete the task correctly.

This provides another example of the model treating the lookup tool as a possible general-purpose mechanism for numerical comparison.

Because the current task set contains only five multi-step examples, these observations should be treated as preliminary rather than evidence of a general rule about the model's behavior.

---

## 7. Discussion

The single-step and multi-step results show an important difference in the behavior of the untrained model.

For single-step tool selection, the model achieved:

$90%$ accuracy.

For the five multi-step tasks, the model achieved:

$40%$ average reward.

The trajectories suggest that the difficulty of multi-step tool use is not limited to choosing the correct tool initially. The model must also interpret observations, determine whether another action is necessary, and produce a correct final answer.

Several behaviors are visible in the current trajectories.

First, the model can correctly use the calculator and produce accurate intermediate results. For example:

$12 \times 12 = 144$

and:

$500 / 10 = 50$.

Second, correct intermediate results do not always lead to correct final answers. In the 23 * 17 task, the model obtained the correct value 391 but incorrectly concluded that it was greater than 400.

Third, the model sometimes makes additional tool calls when no additional information is required. In particular, it sometimes uses the lookup tool to answer numerical comparisons after the calculator has already produced the relevant number.

This behavior is important because it suggests that sequential tool use involves more than simple tool selection. A useful agent must also learn **when to stop using tools** and **when the current observation is sufficient to complete the task**.

The current reward function does not capture this distinction. A trajectory that reaches the correct answer after an unnecessary lookup receives the same reward as a trajectory that reaches the answer directly.

This motivates richer reward functions in future experiments. For example, a future reward could combine task correctness with tool-use efficiency:

$R = R_{\text{correct}} - \lambda C_{\text{unnecessary}}$,

where $C_{\text{unnecessary}}$ represents unnecessary tool calls and $\lambda$ controls their penalty.

Such a reward would encourage the model to maximize correctness while reducing unnecessary actions.

Importantly, no reinforcement learning training has yet been performed. The current results therefore establish only the behavior of the untrained baseline. They do not demonstrate that reinforcement learning will improve performance.

---

## 8. Limitations

This experiment is intentionally small.

The current environment has only two tools and a small number of evaluation tasks. The multi-step evaluation contains only five tasks, so individual behaviors should not be interpreted as statistically strong evidence about the model as a whole.

The current reward functions are also binary. The single-step reward measures whether the expected tool was selected, while the multi-step reward measures final-answer correctness.

Neither reward currently captures:

* unnecessary tool calls,
* invalid tool calls,
* redundant tool calls,
* the number of steps required,
* or whether an intermediate observation was used correctly.

The current lookup tool also acts as a relatively broad information-retrieval mechanism. This makes it possible for the model to attempt to use lookup for tasks, such as simple numerical comparisons, where no external information is required.

These limitations are useful for the current research stage because they expose specific behaviors that can later be incorporated into a more precise reward and environment design.

---

## 9. Future Work

The next stage is to formalize the environment as a reinforcement learning problem.

A future trajectory can be represented as:

$$
\tau =
(s_0,a_0,o_0,s_1,a_1,o_1,\ldots,s_T,a_T,r).
$$

The eventual objective will be to learn a policy:

$\pi_\theta(a_t \mid s_t)$,

which represents the probability of selecting action $a_t$ given the current state $s_t$.

Future experiments will investigate:

* More robust multi-step tasks
* Larger evaluation sets
* More informative reward functions
* Penalties for unnecessary tool calls
* Penalties for invalid tool calls
* More explicit stopping behavior
* Reinforcement learning methods such as REINFORCE
* Credit assignment across multiple tool-use decisions
* Comparison of trained and untrained policies

A particularly important question is whether a richer reward can teach the model not only to produce correct answers, but also to make better sequential decisions about **which tool to use, when to use it, and when no further tool call is necessary**.

The eventual comparison will be between the frozen untrained baseline and the trained policy.
