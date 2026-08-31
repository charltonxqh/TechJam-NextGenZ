---
name: recommender_research
description: Guide recommender-system reasoning about ranking objectives, sparse feedback, auxiliary tasks, interaction modeling, and metric alignment.
---

# Recommender-System Research Skill

## Questions to ask

- Is the task pointwise prediction or ranking?
- What is the ranking group?
- What information is available at inference time?
- Are histories long enough for sequence modeling?
- Are auxiliary behaviors dense enough for multi-task learning?
- Is cold-start important?
- Are user/item IDs dominant sparse features?
- Is temporal drift present?
- Does the evaluation metric align with the training objective?

## General principle

Choose research directions based on the interaction between:
- metric
- data structure
- current model limitations
- experimental evidence
- external research evidence