"""
Description: Coordinates all components and controls the end-to-end autonomous ML research workflow.
Owner: Charlton / David
Input: Research session configuration and system components
Output: Completed research session and best experiment result
"""

import time

from src.agents.policy import should_stop
from src.agents.researcher import Researcher
from src.agents.reflector import Reflector

from src.memory.experiment_store import ExperimentStore

from src.schemas import (
    RunState,
    IterationLog,
    RunSummary,
)

from src.tools.llm_client import GeminiClient
from src.tools.starterkit_runner import run_starterkit_baseline
from src.tools.experiment_runner import run_experiment


from src.tools.trae_coding_agent import TraeCodingAgent


class Orchestrator:

    def __init__(
        self,
        researcher: Researcher,
        reflector: Reflector,
        coding_agent: TraeCodingAgent,
        experiment_store: ExperimentStore,
    ) -> None:
        self.researcher = researcher
        self.reflector = reflector
        self.coding_agent = coding_agent
        self.store = experiment_store

    def run(self) -> RunState:

        print("Starting autonomous ML research session...")

        # =====================================================
        # 1. Establish baseline
        # =====================================================

        baseline_result = run_starterkit_baseline()

        if baseline_result.status != "success":
            raise RuntimeError(
                f"Failed to reproduce baseline: "
                f"{baseline_result.error}"
            )

        state = RunState(
            iteration=0,
            best_experiment_id=baseline_result.experiment_id,
            best_primary=baseline_result.primary,
        )

        print(
            f"Baseline established: "
            f"Primary = {baseline_result.primary:.4f}"
        )

        # =====================================================
        # 2. Autonomous research loop
        # =====================================================

        while True:

            state.iteration += 1

            print(
                f"\n===== Iteration {state.iteration} ====="
            )

            # ---------------------------------------------
            # Load previous research
            # ---------------------------------------------

            history = self.store.get_history()

            # ---------------------------------------------
            # Researcher proposes experiment
            # ---------------------------------------------

            experiment_id = (
                f"exp_{state.iteration:03d}"
            )

            spec = self.researcher.propose(
                experiment_id=experiment_id,
                baseline_result=baseline_result,
                history=history,
            )

            print(
                f"Hypothesis: {spec.hypothesis}"
            )

            # ---------------------------------------------
            # Coding agent implements experiment
            # ---------------------------------------------

            implemented_experiment = (
                self.coding_agent.implement(spec)
            )

            # ---------------------------------------------
            # Run experiment
            # ---------------------------------------------

            result = run_experiment(
                implemented_experiment
            )

            # ---------------------------------------------
            # Reflect on result
            # ---------------------------------------------

            reflection = self.reflector.reflect(
                spec=spec,
                result=result,
                previous_best_primary=state.best_primary,
                history=history,
            )

            # ---------------------------------------------
            # Update best result
            # ---------------------------------------------

            improvement = 0.0
            is_new_best = False

            if (
                result.status == "success"
                and result.primary is not None
            ):

                improvement = (
                    result.primary
                    - state.best_primary
                )

                if result.primary > state.best_primary:

                    state.best_primary = result.primary
                    state.best_experiment_id = (
                        result.experiment_id
                    )

                    is_new_best = True

            state.improvements.append(
                improvement
            )

            # ---------------------------------------------
            # Save experiment
            # ---------------------------------------------

            self.store.save_iteration(iteration_log)
            print(
                f"Primary: {result.primary}"
            )

            print(
                f"Reflection: {reflection.verdict}"
            )

            # ---------------------------------------------
            # Check stopping policy
            # ---------------------------------------------

            if should_stop(
                iteration=state.iteration,
                improvements=state.improvements,
            ):
                break

        # =====================================================
        # 3. Session complete
        # =====================================================

        print("\nResearch session finished.")

        print(
            f"Best experiment: "
            f"{state.best_experiment_id}"
        )

        print(
            f"Best Primary: "
            f"{state.best_primary:.4f}"
        )

        return state