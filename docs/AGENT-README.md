# TechJam NextGenZ

Autonomous Machine Learning Research Agent for Recommender Systems — TikTok TechJam 2026 Track 2.

## Agent Loop

The system autonomously proposes ML hypotheses, implements and evaluates experiments, learns from the results, and iterates toward better recommendation performance.

```text
                         ┌─────────────────────┐
                         │    Session Start    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Starter Baseline  │
                         │  Run & get metrics  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      Researcher     │
                         │ What should we try? │
                         └──────────┬──────────┘
                                    │
                                    │ ExperimentSpec
                                    ▼
                         ┌─────────────────────┐
                         │ Coding Agent / TRAE │
                         │ Implement the idea  │
                         └──────────┬──────────┘
                                    │
                                    │ ImplementedExperiment
                                    ▼
                         ┌─────────────────────┐
                         │  Experiment Runner  │
                         │  Train & evaluate   │
                         └──────────┬──────────┘
                                    │
                                    │ ExperimentResult
                                    ▼
                         ┌─────────────────────┐
                         │       Memory        │
                         │   Store experiment  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      Reflector      │
                         │ What did we learn?  │
                         └──────────┬──────────┘
                                    │
                                    │ Reflection
                                    ▼
                         ┌─────────────────────┐
                         │       Policy        │
                         │ Continue or stop?   │
                         └──────────┬─────┬────┘
                                    │     │
                          Continue  │     │ Stop
                   ┌────────────────┘     └──────────► End
                   │
                   └──────────────► Researcher
```

## Team

- **Agent / Orchestration:** Charlton & David
- **Skills / Tools / MCP:** Hayden
- **Memory:** Esther