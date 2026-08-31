"""
Description: Coordinates the autonomous ML research loop using one Researcher LLM call per iteration, deterministic execution, diagnostics, decision policy, and persistent memory.
Owner: Charlton / David
Input: Research session configuration and system components
Output: Completed research session and best experiment result
"""

import time

from dataclasses import (
    asdict,
)

from pathlib import Path

from src.agents.decision import (
    decide_experiment,
)

from src.agents.policy import (
    should_stop,
)

from src.agents.researcher import (
    Researcher,
)

from src.config import (
    STARTER_KIT_DIR,
    DATA_DIR,
)

from src.memory.memory_store import (
    MemoryStore,
)

from src.memory.records import (
    FailureType,
    IterationRecord,
    Metrics as MemoryMetrics,
    ResourceUsage,
)

from src.schemas import (
    ImplementedExperiment,
    RunState,
)

from src.tools.candidate_builder import (
    build_candidate,
)

from tools.experiment_analysis import (
    analyze_experiment,
    format_diagnostics,
)

from src.tools.experiment_runner import (
    run_experiment,
)

from src.tools.final_evaluator import (
    run_final_test,
)

from src.tools.starterkit_runner import (
    run_starterkit_baseline,
)


class Orchestrator:

    def __init__(
        self,
        researcher: Researcher,
        memory_store: MemoryStore,
    ) -> None:

        self.researcher = (
            researcher
        )

        self.memory = (
            memory_store
        )

    def run(
        self,
    ) -> RunState:

        print(
            "Starting autonomous "
            "ML research session..."
        )

        session_start_time = (
            time.time()
        )

        # =====================================================
        # 1. Establish baseline
        # =====================================================

        baseline_result = (
            run_starterkit_baseline()
        )

        if (
            baseline_result.status
            != "success"
            or baseline_result.primary
            is None
        ):

            raise RuntimeError(
                f"Failed to reproduce "
                f"baseline: "
                f"{baseline_result.error}"
            )

        baseline_primary = (
            baseline_result.primary
        )

        baseline_path = (
            STARTER_KIT_DIR
            / "baseline.py"
        )

        baseline_code = (
            baseline_path.read_text(
                encoding="utf-8"
            )
        )
        
        baseline_implemented_experiment = (
            ImplementedExperiment(
                experiment_id="baseline",
                workspace_path=str(
                    STARTER_KIT_DIR.resolve()
                ),
                command=[
                    "python",
                    "baseline.py",
                    "--data_dir",
                    str(DATA_DIR),
                    "--model",
                    "fm",
                    "--split",
                    "valid",
                ],
                test_command=[
                    "python",
                    "baseline.py",
                    "--data_dir",
                    str(DATA_DIR),
                    "--model",
                    "fm",
                    "--split",
                    "test",
                ],
                full_code=baseline_code,
                status="success",
            )
        )

        state = RunState(
            iteration=0,
            best_experiment_id=(
                baseline_result
                .experiment_id
            ),
            best_primary=(
                baseline_primary
            ),
            best_primary_history=[
                baseline_primary
            ],
        )

        # The full code from which the next experiment branches.
        current_best_code = (
            baseline_code
        )

        # Runnable candidate associated with the validation-best
        # autonomous experiment.
        best_implemented_experiment = (
            baseline_implemented_experiment
        )

        # -----------------------------------------------------
        # Store baseline as reference memory
        # -----------------------------------------------------

        baseline_exists = any(
            record.is_baseline
            for record
            in self.memory.history
        )

        if not baseline_exists:

            stored_baseline_code = (
                self.memory
                .compute_code_diff(
                    baseline_code
                )
            )

            baseline_record = (
                IterationRecord(
                    iteration=0,

                    experiment_id=(
                        baseline_result
                        .experiment_id
                    ),

                    hypothesis=(
                        "Official starter-kit "
                        "baseline used as the "
                        "reference benchmark."
                    ),

                    rationale=(
                        "This baseline is the "
                        "starting validation "
                        "benchmark that autonomous "
                        "research attempts to "
                        "improve. It is not a "
                        "previous research "
                        "hypothesis."
                    ),

                    stage="baseline",

                    code_diff=(
                        stored_baseline_code
                    ),

                    code_summary=(
                        "Official starter-kit "
                        "baseline implementation."
                    ),

                    metrics=MemoryMetrics(
                        gauc=(
                            baseline_result.gauc
                        ),
                        ndcg5=(
                            baseline_result.ndcg5
                        ),
                        primary=(
                            baseline_result.primary
                        ),
                    ),

                    failure=(
                        FailureType.NONE
                    ),

                    resource_usage=(
                        ResourceUsage(
                            wall_clock_sec=(
                                baseline_result
                                .runtime_seconds
                                or 0.0
                            ),
                        )
                    ),

                    is_baseline=True,
                )
            )

            self.memory.add(
                baseline_record
            )

        print(
            f"Baseline established: "
            f"Validation Primary = "
            f"{baseline_primary:.4f}"
        )

        # =====================================================
        # 2. Autonomous research loop
        # =====================================================

        while True:

            state.iteration += 1

            print(
                f"\n===== Iteration "
                f"{state.iteration} ====="
            )

            # ---------------------------------------------
            # Load compressed research memory
            # ---------------------------------------------

            memory_context = (
                self.memory
                .get_prompt_context()
            )

            # ---------------------------------------------
            # Researcher reasons + proposes + writes code
            # ---------------------------------------------

            experiment_id = (
                f"exp_"
                f"{state.iteration:03d}"
            )

            action = (
                self.researcher
                .propose(
                    experiment_id=(
                        experiment_id
                    ),
                    memory_context=(
                        memory_context
                    ),
                    current_best_code=(
                        current_best_code
                    ),
                    current_best_primary=(
                        state.best_primary
                    ),
                    baseline_primary=(
                        baseline_primary
                    ),
                )
            )

            spec = (
                action.spec
            )

            print(
                f"Hypothesis: "
                f"{spec.hypothesis}"
            )

            # ---------------------------------------------
            # Memory computes candidate diff vs current best
            # ---------------------------------------------

            code_diff = (
                self.memory
                .compute_code_diff(
                    current_full_code=(
                        action.full_code
                    ),
                    reference_full_code=(
                        current_best_code
                    ),
                )
            )

            # ---------------------------------------------
            # Materialize candidate deterministically
            # ---------------------------------------------

            implemented_experiment = (
                build_candidate(
                    experiment_id=(
                        experiment_id
                    ),
                    full_code=(
                        action.full_code
                    ),
                )
            )

            # ---------------------------------------------
            # Run validation experiment
            # ---------------------------------------------

            previous_best_primary = (
                state.best_primary
            )

            result = run_experiment(
                implemented_experiment
            )

            # ---------------------------------------------
            # Deterministic factual diagnostics
            # ---------------------------------------------

            diagnostics = (
                analyze_experiment(
                    result=result,
                    previous_best_primary=(
                        previous_best_primary
                    ),
                    baseline_primary=(
                        baseline_primary
                    ),
                )
            )

            diagnostic_text = (
                format_diagnostics(
                    diagnostics
                )
            )

            # ---------------------------------------------
            # Deterministic keep / reject / retry
            # ---------------------------------------------

            decision = (
                decide_experiment(
                    result=result,
                    previous_best_primary=(
                        previous_best_primary
                    ),
                )
            )

            # ---------------------------------------------
            # Update best result
            # ---------------------------------------------

            improvement = 0.0

            if (
                result.status
                == "success"
                and result.primary
                is not None
            ):

                improvement = (
                    result.primary
                    - previous_best_primary
                )

            if (
                decision
                == "keep"
            ):

                state.best_primary = (
                    result.primary
                )

                state.best_experiment_id = (
                    result.experiment_id
                )

                current_best_code = (
                    action.full_code
                )

                best_implemented_experiment = (
                    implemented_experiment
                )

            state.improvements.append(
                improvement
            )

            state.best_primary_history.append(
                state.best_primary
            )

            # ---------------------------------------------
            # Failure classification
            # ---------------------------------------------

            failure = (
                self._classify_failure(
                    result=result,
                    decision=decision,
                )
            )

            recovery_events = (
                implemented_experiment
                .recovery_events
                + result.recovery_events
            )

            # ---------------------------------------------
            # Store iteration in memory
            # ---------------------------------------------

            record = IterationRecord(
                iteration=(
                    state.iteration
                ),

                experiment_id=(
                    spec.experiment_id
                ),

                hypothesis=(
                    spec.hypothesis
                ),

                rationale=(
                    spec.rationale
                ),

                stage=(
                    spec.change_type
                ),

                code_diff=(
                    code_diff
                ),

                metrics=MemoryMetrics(
                    gauc=result.gauc,
                    ndcg5=result.ndcg5,
                    primary=result.primary,
                ),

                failure=(
                    failure
                ),

                error_message=(
                    result.error
                    or implemented_experiment
                    .error
                ),

                manual_intervention=False,

                resource_usage=(
                    ResourceUsage(
                        wall_clock_sec=(
                            result
                            .runtime_seconds
                            or 0.0
                        ),
                    )
                ),

                verdict=(
                    decision
                ),

                analysis=(
                    diagnostic_text
                ),

                diagnostics=(
                    asdict(
                        diagnostics
                    )
                ),

                recovery_events=[
                    asdict(
                        event
                    )
                    for event
                    in recovery_events
                ],
            )

            self.memory.add(
                record
            )

            # ---------------------------------------------
            # Display result
            # ---------------------------------------------

            print(
                diagnostic_text
            )

            print(
                f"Decision: "
                f"{decision}"
            )

            # ---------------------------------------------
            # Check stopping policy
            # ---------------------------------------------

            if should_stop(
                iteration=(
                    state.iteration
                ),
                best_primary_history=(
                    state
                    .best_primary_history
                ),
            ):

                break

        # =====================================================
        # 3. Research complete
        # =====================================================

        total_runtime_seconds = (
            time.time()
            - session_start_time
        )

        state.manual_interventions = (
            self.memory
            .manual_intervention_count()
        )

        print(
            "\nResearch session finished."
        )

        print(
            f"Best experiment: "
            f"{state.best_experiment_id}"
        )

        print(
            f"Best Validation Primary: "
            f"{state.best_primary:.4f}"
        )

        # =====================================================
        # 4. Final one-time test evaluation
        # =====================================================

        if (
            best_implemented_experiment
            is not None
        ):

            print(
                "\nRunning final one-time "
                "test evaluation..."
            )

            final_test_result = (
                run_final_test(
                    best_implemented_experiment
                )
            )

            if (
                final_test_result.status
                == "success"
            ):

                state.final_test_gauc = (
                    final_test_result.gauc
                )

                state.final_test_ndcg5 = (
                    final_test_result.ndcg5
                )

                state.final_test_primary = (
                    final_test_result
                    .primary
                )

                print(
                    f"Final Test GAUC: "
                    f"{state.final_test_gauc:.4f}"
                )

                print(
                    f"Final Test nDCG@5: "
                    f"{state.final_test_ndcg5:.4f}"
                )

                print(
                    f"Final Test Primary: "
                    f"{state.final_test_primary:.4f}"
                )

            else:

                print(
                    "Final test evaluation "
                    "failed: "
                    f"{final_test_result.error}"
                )

        else:

            print(
                "\nBaseline remained the "
                "best validation result. "
                "No autonomous experiment "
                "was selected for final "
                "test evaluation."
            )

        # =====================================================
        # 5. Export memory / run logs
        # =====================================================

        log_path = Path(
            self.memory.log_path
        )

        markdown_path = (
            log_path.with_suffix(
                ".md"
            )
        )

        json_path = (
            log_path.with_name(
                "run_log_full.json"
            )
        )

        self.memory.export_run_log_markdown(
            str(
                markdown_path
            )
        )

        self.memory.export_run_log_json(
            str(
                json_path
            )
        )

        # =====================================================
        # 6. Session summary
        # =====================================================

        print(
            f"Total iterations: "
            f"{state.iteration}"
        )

        print(
            f"Manual interventions: "
            f"{state.manual_interventions}"
        )

        print(
            f"Total runtime: "
            f"{total_runtime_seconds:.2f} "
            f"seconds"
        )

        return state

    def _classify_failure(
        self,
        result,
        decision: str,
    ) -> FailureType:
        """
        Map the factual experiment outcome into the memory failure taxonomy.
        """

        if (
            result.status
            == "success"
        ):

            if (
                decision
                == "keep"
            ):

                return (
                    FailureType.NONE
                )

            return (
                FailureType
                .NO_IMPROVEMENT
            )

        error_text = (
            result.error
            or ""
        ).lower()

        if (
            "timeout"
            in error_text
            or "timed out"
            in error_text
        ):

            return (
                FailureType.TIMEOUT
            )

        recovery_stages = {
            event.stage
            for event
            in result.recovery_events
        }

        if (
            "metric_parsing"
            in recovery_stages
        ):

            return (
                FailureType.BAD_OUTPUT
            )

        if (
            "experiment_execution"
            in recovery_stages
        ):

            return (
                FailureType.CODE_ERROR
            )

        return (
            FailureType.OTHER
        )