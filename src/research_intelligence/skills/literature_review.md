---
name: literature_review
description: Guide search, source assessment, evidence extraction, and adaptation of papers or public ML solutions.
---

# Literature and Solution Research

## Purpose

Use external research when the current dataset evidence, experiment history,
or model implementation is insufficient to justify the next experiment.

Research is not limited to academic papers.

Useful sources may include:

- academic papers
- official documentation
- public GitHub repositories
- benchmark implementations
- competition solutions
- engineering blogs
- technical reports
- framework documentation

## Procedure

1. Define the research question before searching.

2. Search broadly first when the solution space is unclear.

3. Use academic search when scientific evidence or model assumptions matter.

4. Use web search when looking for:
   - implementations
   - engineering practices
   - open-source code
   - competition approaches
   - documentation

5. Read promising sources before drawing conclusions.

6. Extract:
   - problem addressed
   - method
   - assumptions
   - relevant findings
   - implementation requirements
   - applicability to the current dataset

7. Compare source assumptions with autonomously discovered dataset evidence.

8. Do not assume a method is appropriate merely because it is state of the art.

9. Store useful evidence in ResearchKnowledgeStore.

10. Prefer evidence from multiple independent sources when making a major architectural change.

## Research discipline

Do not search for the hidden-test answer.

Do not use hidden-test metrics.

Do not treat search snippets alone as strong evidence when the underlying
source can be inspected.

Do not blindly copy code without understanding its assumptions and
compatibility with the current evaluation protocol.