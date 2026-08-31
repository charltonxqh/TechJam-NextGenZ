---
name: eda
description: Guide autonomous dataset inspection, assumption testing, leakage checks, sparsity analysis, and metric-relevant EDA.
---

# Exploratory Data Analysis Skill

## Purpose

Use EDA to discover dataset properties that affect ML research decisions.
Do not use EDA to directly prescribe a model.

## Principles

1. Measure assumptions before relying on them.
2. Prefer analyses that can change the research direction.
3. Separate observations from interpretations.
4. Use train and validation only.
5. Never use hidden-test information.
6. Avoid spending experiment iterations on questions that can be answered directly from the data.

## Recommendation-system analyses to consider

### Target structure
- positive rate
- imbalance
- target drift across time

### User structure
- number of unique users
- interactions per user
- median and tail history length
- cold/warm user prevalence

### Item structure
- number of unique items
- item exposure distribution
- popularity skew
- cold-item prevalence

### Ranking structure
- group size per user
- fraction of singleton groups
- positive count per ranking group

### Feature structure
- cardinality
- missingness
- sparsity
- train/validation coverage

### Temporal structure
- distribution drift
- feature drift
- popularity drift
- user/item churn

### Auxiliary labels
- density
- imbalance
- coverage
- correlation with primary target

## Decision rule

Only request an analysis when its result could materially change:
- model family
- objective
- feature engineering
- sampling
- regularization
- evaluation strategy