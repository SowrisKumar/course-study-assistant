# What the assistant looks like in use

Three real sessions with `uv run python -m study_assistant.chat -v "<question>"` (Llama 3.1 8B, local, hybrid retrieval), copied unedited from the terminal. The `·` lines are the trace that `-v` shows: one line per step of the LangGraph workflow.

## 1. A normal answer, with citations

```text
you> Why use sinusoidal positional encoding rather than just adding the position index?
  · route: mode=answer, search='Why use sinusoidal positional encoding rather than just adding the position index?'
  · retrieve (hybrid): 8 chunks -> 5 passages
  ·   [1] 0.58 Lecture 9 · Module 9 – Transformer Architecture and Advanced GAN, slide 7
  ·   [2] 0.47 Lecture 9 · Module 9 – Transformer Architecture and Advanced GAN, slide 8
  ·   [3] 0.35 Lecture 9 · Module 9 – Transformer Architecture and Advanced GAN, slide 6
  ·   [4] 0.41 Lecture 9 · Module 8 — Practical: Building a GPT from Scratch, cells 21-22 (6. Token and positional embeddings)
  ·   [5] 0.24 Lecture 9 · Module 8 — Practical: Building a GPT from Scratch, cells 35-36 (10. Looking inside the model › 10.2 What did the embeddings learn?)
  · check: grader says relevant=[1, 2, 4]
  · generate (ollama/llama3.1:8b, 14.6s)
  · cite: 2 source(s) cited

We should be careful about how we encode the positions, as simply adding the position index can result in large values that might interfere with the token embedding [1]. Using sinusoidal positional encoding addresses this problem and conserves relative distances between positions [1]. This encoding scheme keeps the dimension fixed and never runs out of encodings, no matter how long the context [2].

Sources:
[1] Lecture 9 · Module 9 – Transformer Architecture and Advanced GAN, slide 7
[2] Lecture 9 · Module 9 – Transformer Architecture and Advanced GAN, slide 8
```

Every sentence is cited; both slides do say this. Passage 4 was judged relevant but not needed, so it isn't listed.

## 2. A near-miss, declined

The extended Lecture 11 deck *names* PPO, but no course material explains its clipped objective.

```text
you> How does PPO's clipped objective keep policy updates stable?
  · route: mode=answer, search="How does PPO's clipped objective keep policy updates stable?"
  · retrieve (hybrid): 8 chunks -> 5 passages
  ·   [1] 0.30 Lecture 10 · Module 10 – Basics of Reinforcement Learning (extended deck), slide 20
  ·   [2] 0.28 Lecture 10 · Module 10 – Basics of Reinforcement Learning (extended deck), slide 18
  ·   [3] 0.29 Lecture 10 · Module 9 — Practical: Basics of Reinforcement Learning, cell 35 (9. Summary)
  ·   [4] 0.29 Lecture 11 · Module 11 – Application of RL to Generative AI (extended deck), slide 11
  ·   [5] 0.26 Lecture 11 · Module 10 — Applying Reinforcement Learning to Language, cells 20-23 (🔧 Visual check: log probabilities must require gradients › Next, let's train the model. Each epoch consists of batchsi…)
  · check: grader says relevant=[1, 2]
  · generate (ollama/llama3.1:8b, 10.1s)
  · cite: model answered 'not covered' -> abstain

This isn't covered in the course materials.
```

This trace also shows the report's finding about the relevance check. The LLM grader called two general RL slides "relevant", and it was the answer prompt that declined. The same model *without* the course explains PPO's clipping confidently, which a student would take as course content.

## 3. A practice quiz

```text
you> Quiz me on GANs
  · route: mode=quiz, search='GANs'
  · retrieve (hybrid): 8 chunks -> 5 passages
  ·   [1] 0.57 Lecture 6 · Module 6 – Modern Deep Learning Architectures Part 3, slide 4
  ·   [2] 0.47 Lecture 6 · Module 6: Practical - GAN, cells 2-6 (Introduction)
  ·   [3] 0.39 Lecture 1 · Module 1 – Introduction, slide 5
  ·   [4] 0.43 Lecture 1 · Module 1 – Introduction, slide 14
  ·   [5] 0.33 Lecture 5 · Module 5 – Modern Deep Learning Architectures Part 2, slide 5
  · check: grader says relevant=[1, 2]
  · generate (ollama/llama3.1:8b, 20.1s)
  · cite: 2 source(s) cited

Q1. What is the primary goal of the Generator (G) in a Generative Adversarial Network (GAN)?

A) To distinguish real data from fake data. B) To generate synthetic data from random noise. C) To evaluate samples drawn from a training set. D) To sample from the real data distribution.

Answer: B) To generate synthetic data from random noise. [1]

Q2. What is the problem with using the likelihood of the data distribution in the loss function of the Discriminator?

A) We do not know the data distribution. B) We can sample from the data distribution. C) The data distribution is too complex. D) The data distribution is too simple.

Answer: A) We do not know the data distribution. [2]

Q3. What is the purpose of the Discriminator (D) in a Generative Adversarial Network (GAN)?

A) To generate synthetic data from random noise. B) To evaluate samples drawn from a training set. C) To distinguish real data from fake data. D) To sample from the real data distribution.

Answer: C) To distinguish real data from fake data. [1]

Sources:
[1] Lecture 6 · Module 6 – Modern Deep Learning Architectures Part 3, slide 4
[2] Lecture 6 · Module 6: Practical - GAN, cells 2-6 (Introduction)
```

The router removed "Quiz me on" from the search, and each answer cites the slide or notebook it comes from. Q1 and Q3 overlap (generator vs discriminator). An 8B model's quizzes are serviceable rather than great.
