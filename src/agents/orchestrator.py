"""
Description: Coordinates autonomous research actions, coding, deterministic candidate validation and repair, ML experiments, diagnostics, decision policy, persistent memory, and debugging artifacts.
Owner: Charlton / David
Input: Research session configuration and system components
Output: Completed research session and best experiment result
"""

import time

from dataclasses import (
    asdict,
)

from pathlib import (
    Path,
)

from src.agents.coder import (
    Coder,
)

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
    DATA_DIR,
    STARTER_KIT_DIR,
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

from src.research_intelligence.context_builder import (
    build_research_context,
)

from src.research_intelligence.eda.tools import (
    run_eda_tool,
)

from src.research_intelligence.knowledge_store import (
    ResearchKnowledgeStore,
)

from src.research_intelligence.retrieval.evidence_extractor import (
    EvidenceExtractor,
)

from src.research_intelligence.retrieval.research_runner import (
    ResearchRunner,
)

from src.research_intelligence.retrieval.research_tool import (
    MLResearchTool,
)

from src.research_intelligence.skill_loader import (
    load_skills,
)

from src.schemas import (
    ExperimentResult,
    ImplementedExperiment,
    RecoveryEvent,
    RunState,
)

from src.tools.candidate_builder import (
    build_candidate,
)

from src.tools.candidate_validator import (
    validate_candidate,
)

from src.tools.experiment_analysis import (
    analyze_experiment,
    format_diagnostics,
)

from src.tools.experiment_runner import (
    run_experiment,
)

from src.tools.final_evaluator import (
    run_final_test,
)

from src.tools.llm_client import (
    GeminiClient,
)

from src.tools.starterkit_runner import (
    run_starterkit_baseline,
)


INFORMATION_ACTION_BUDGET = 4

MAX_CODE_REPAIR_ATTEMPTS = 2


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

        self.coder = (
            Coder(
                GeminiClient()
            )
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

        run_dir = (
            Path(
                self.memory.log_path
            )
            .parent
        )

        debug_dir = (
            run_dir
            / "debug"
        )

        debug_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # =====================================================
        # 0. Initialize research intelligence
        # =====================================================

        knowledge_store = (
            ResearchKnowledgeStore(
                path=str(
                    run_dir
                    / "research_knowledge.jsonl"
                )
            )
        )

        research_llm = (
            GeminiClient()
        )

        research_runner = (
            ResearchRunner(
                research_tool=(
                    MLResearchTool()
                ),
                evidence_extractor=(
                    EvidenceExtractor(
                        research_llm
                    )
                ),
                knowledge_store=(
                    knowledge_store
                ),
            )
        )

        skills_context = (
            load_skills(
                [
                    "eda",
                    "literature_review",
                    "experiment_design",
                    "recommender_research",
                ]
            )
        )

        # =====================================================
        # Load authoritative implementation context
        # =====================================================

        data_path = (
            STARTER_KIT_DIR
            / "data.py"
        )

        if data_path.exists():

            data_context = (
                data_path.read_text(
                    encoding="utf-8"
                )
            )

        else:

            data_context = (
                "data.py was not found. "
                "Do not assume any data structure "
                "that is not visible in "
                "current_best_code."
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
                experiment_id=(
                    "baseline"
                ),
                workspace_path=str(
                    STARTER_KIT_DIR
                    .resolve()
                ),
                command=[
                    "python",
                    "baseline.py",
                    "--data_dir",
                    str(
                        DATA_DIR
                    ),
                    "--model",
                    "fm",
                    "--split",
                    "valid",
                ],
                test_command=[
                    "python",
                    "baseline.py",
                    "--data_dir",
                    str(
                        DATA_DIR
                    ),
                    "--model",
                    "fm",
                    "--split",
                    "test",
                ],
                full_code=(
                    baseline_code
                ),
                status=(
                    "success"
                ),
            )
        )

        state = (
            RunState(
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

                    stage=(
                        "baseline"
                    ),

                    code_diff=(
                        stored_baseline_code
                    ),

                    code_summary=(
                        "Official starter-kit "
                        "baseline implementation."
                    ),

                    metrics=(
                        MemoryMetrics(
                            gauc=(
                                baseline_result
                                .gauc
                            ),
                            ndcg5=(
                                baseline_result
                                .ndcg5
                            ),
                            primary=(
                                baseline_result
                                .primary
                            ),
                        )
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

            next_iteration = (
                state.iteration
                + 1
            )

            experiment_id = (
                f"exp_"
                f"{next_iteration:03d}"
            )

            information_actions = 0

            while True:

                # ---------------------------------------------
                # Load current memory + learned knowledge
                # ---------------------------------------------

                memory_context = (
                    self.memory
                    .get_prompt_context()
                )

                research_context = (
                    build_research_context(
                        knowledge_store
                    )
                )

                # ---------------------------------------------
                # Researcher chooses next action
                # ---------------------------------------------

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
                        research_context=(
                            research_context
                        ),
                        skills_context=(
                            skills_context
                        ),
                        information_actions_used=(
                            information_actions
                        ),
                        information_action_budget=(
                            INFORMATION_ACTION_BUDGET
                        ),
                    )
                )

                print(
                    f"\nResearch action: "
                    f"{action.action_type}"
                )

                print(
                    f"Reason: "
                    f"{action.reason}"
                )

                # ---------------------------------------------
                # Online research action
                # ---------------------------------------------

                if (
                    action.action_type
                    == "research"
                ):

                    information_actions += 1

                    print(
                        f"Research query: "
                        f"{action.research_query}"
                    )

                    added = (
                        research_runner.run(
                            query=(
                                action.research_query
                                or ""
                            ),
                            source=(
                                action.research_source
                                or "both"
                            ),
                            task_context=(
                                research_context
                            ),
                            iteration=(
                                next_iteration
                            ),
                            research_action_index=(
                                information_actions
                            ),
                        )
                    )

                    print(
                        f"Research evidence added: "
                        f"{added}"
                    )

                    continue

                # ---------------------------------------------
                # EDA action
                # ---------------------------------------------

                if (
                    action.action_type
                    == "eda"
                ):

                    information_actions += 1

                    print(
                        f"EDA request: "
                        f"{action.eda_tool}"
                    )

                    try:

                        findings = (
                            run_eda_tool(
                                name=(
                                    action.eda_tool
                                    or ""
                                ),
                                data_dir=(
                                    DATA_DIR
                                ),
                            )
                        )

                        added = (
                            knowledge_store
                            .add_many(
                                findings
                            )
                        )

                        print(
                            f"EDA facts added: "
                            f"{added}"
                        )

                    except Exception as error:

                        print(
                            f"EDA action failed: "
                            f"{error}"
                        )

                    continue

                # ---------------------------------------------
                # Experiment action
                # ---------------------------------------------

                break

            state.iteration = (
                next_iteration
            )

            print(
                f"\n===== Iteration "
                f"{state.iteration} ====="
            )

            spec = (
                action.spec
            )

            if spec is None:

                raise RuntimeError(
                    "Experiment action did not "
                    "contain an ExperimentSpec."
                )

            print(
                f"Hypothesis: "
                f"{spec.hypothesis}"
            )

            experiment_debug_dir = (
                debug_dir
                / experiment_id
            )

            experiment_debug_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            # ---------------------------------------------
            # Coder implements researcher-specified experiment
            # ---------------------------------------------

            candidate_code = (
                self.coder
                .implement(
                    spec=(
                        spec
                    ),
                    current_best_code=(
                        current_best_code
                    ),
                    data_context=(
                        data_context
                    ),
                )
            )

            self._save_debug_artifact(
                directory=(
                    experiment_debug_dir
                ),
                name=(
                    "candidate_initial.py"
                ),
                content=(
                    candidate_code
                ),
            )

            # ---------------------------------------------
            # Validate / repair / run candidate
            # ---------------------------------------------

            repair_attempt = 0

            implemented_experiment = None

            result = None

            repair_events = []

            while True:

                validation = (
                    validate_candidate(
                        candidate_code
                    )
                )

                if not (
                    validation.valid
                ):

                    validation_error = (
                        validation
                        .format_errors()
                    )

                    print(
                        "\nCandidate validation "
                        "failed:"
                    )

                    print(
                        validation_error
                    )

                    self._save_debug_artifact(
                        directory=(
                            experiment_debug_dir
                        ),
                        name=(
                            f"validation_error_"
                            f"{repair_attempt:02d}.txt"
                        ),
                        content=(
                            validation_error
                        ),
                    )

                    if (
                        repair_attempt
                        >= MAX_CODE_REPAIR_ATTEMPTS
                    ):

                        repair_events.append(
                            RecoveryEvent(
                                stage=(
                                    "candidate_validation"
                                ),
                                error=(
                                    validation_error
                                ),
                                action=(
                                    "Candidate validation "
                                    "failed after exhausting "
                                    "the automatic repair "
                                    "budget."
                                ),
                                success=False,
                            )
                        )

                        result = (
                            ExperimentResult(
                                experiment_id=(
                                    experiment_id
                                ),
                                status=(
                                    "failed"
                                ),
                                error=(
                                    validation_error
                                ),
                            )
                        )

                        break

                    repair_attempt += 1

                    print(
                        f"Automatic code repair "
                        f"{repair_attempt}/"
                        f"{MAX_CODE_REPAIR_ATTEMPTS}"
                    )

                    candidate_code = (
                        self.coder
                        .repair(
                            spec=(
                                spec
                            ),
                            current_best_code=(
                                current_best_code
                            ),
                            data_context=(
                                data_context
                            ),
                            candidate_code=(
                                candidate_code
                            ),
                            error=(
                                validation_error
                            ),
                            repair_attempt=(
                                repair_attempt
                            ),
                        )
                    )

                    self._save_debug_artifact(
                        directory=(
                            experiment_debug_dir
                        ),
                        name=(
                            f"candidate_repair_"
                            f"{repair_attempt:02d}.py"
                        ),
                        content=(
                            candidate_code
                        ),
                    )

                    repair_events.append(
                        RecoveryEvent(
                            stage=(
                                "candidate_validation"
                            ),
                            error=(
                                validation_error
                            ),
                            action=(
                                "Coder repaired the "
                                "candidate within the "
                                "same scientific "
                                "iteration."
                            ),
                            success=True,
                        )
                    )

                    continue

                # -----------------------------------------
                # Materialize validated candidate
                # -----------------------------------------

                implemented_experiment = (
                    build_candidate(
                        experiment_id=(
                            experiment_id
                        ),
                        full_code=(
                            candidate_code
                        ),
                    )
                )

                # -----------------------------------------
                # Execute validation experiment
                # -----------------------------------------

                result = (
                    run_experiment(
                        implemented_experiment
                    )
                )

                if (
                    result.status
                    == "success"
                ):

                    break

                technical_failure = (
                    self
                    ._is_repairable_failure(
                        result
                    )
                )

                if not (
                    technical_failure
                ):

                    break

                failure_text = (
                    result.error
                    or result.stderr
                    or result.stdout
                    or (
                        "Experiment failed "
                        "without an error message."
                    )
                )

                print(
                    "\nImplementation failure:"
                )

                print(
                    failure_text
                )

                self._save_debug_artifact(
                    directory=(
                        experiment_debug_dir
                    ),
                    name=(
                        f"runtime_error_"
                        f"{repair_attempt:02d}.txt"
                    ),
                    content=(
                        failure_text
                    ),
                )

                if (
                    repair_attempt
                    >= MAX_CODE_REPAIR_ATTEMPTS
                ):

                    break

                repair_attempt += 1

                print(
                    f"Experiment implementation "
                    f"failed. Automatic code "
                    f"repair "
                    f"{repair_attempt}/"
                    f"{MAX_CODE_REPAIR_ATTEMPTS}"
                )

                repair_events.extend(
                    result.recovery_events
                )

                candidate_code = (
                    self.coder
                    .repair(
                        spec=(
                            spec
                        ),
                        current_best_code=(
                            current_best_code
                        ),
                        data_context=(
                            data_context
                        ),
                        candidate_code=(
                            candidate_code
                        ),
                        error=(
                            failure_text
                        ),
                        repair_attempt=(
                            repair_attempt
                        ),
                    )
                )

                self._save_debug_artifact(
                    directory=(
                        experiment_debug_dir
                    ),
                    name=(
                        f"candidate_repair_"
                        f"{repair_attempt:02d}.py"
                    ),
                    content=(
                        candidate_code
                    ),
                )

            if result is None:

                raise RuntimeError(
                    "Candidate repair loop "
                    "ended without producing "
                    "an experiment result."
                )

            # ---------------------------------------------
            # Memory computes candidate diff vs current best
            # ---------------------------------------------

            code_diff = (
                self.memory
                .compute_code_diff(
                    current_full_code=(
                        candidate_code
                    ),
                    reference_full_code=(
                        current_best_code
                    ),
                )
            )

            # ---------------------------------------------
            # Deterministic factual diagnostics
            # ---------------------------------------------

            previous_best_primary = (
                state.best_primary
            )

            diagnostics = (
                analyze_experiment(
                    result=(
                        result
                    ),
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
                    result=(
                        result
                    ),
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
                    candidate_code
                )

                best_implemented_experiment = (
                    implemented_experiment
                )

            state.improvements.append(
                improvement
            )

            # Technical failures do not count as
            # scientific no-improvement observations.
            if (
                result.status
                == "success"
            ):

                state.best_primary_history.append(
                    state.best_primary
                )

            # ---------------------------------------------
            # Failure classification
            # ---------------------------------------------

            failure = (
                self._classify_failure(
                    result=(
                        result
                    ),
                    decision=(
                        decision
                    ),
                    repair_events=(
                        repair_events
                    ),
                )
            )

            implementation_recovery_events = []

            if (
                implemented_experiment
                is not None
            ):

                implementation_recovery_events = (
                    implemented_experiment
                    .recovery_events
                )

            recovery_events = (
                repair_events
                + implementation_recovery_events
                + result.recovery_events
            )

            # ---------------------------------------------
            # Store iteration in memory
            # ---------------------------------------------

            record = (
                IterationRecord(
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

                    metrics=(
                        MemoryMetrics(
                            gauc=(
                                result.gauc
                            ),
                            ndcg5=(
                                result.ndcg5
                            ),
                            primary=(
                                result.primary
                            ),
                        )
                    ),

                    failure=(
                        failure
                    ),

                    error_message=(
                        result.error
                        or (
                            implemented_experiment
                            .error
                            if (
                                implemented_experiment
                                is not None
                            )
                            else None
                        )
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

            if (
                should_stop(
                    iteration=(
                        state.iteration
                    ),
                    best_primary_history=(
                        state
                        .best_primary_history
                    ),
                )
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

        # =====================================================
        # 5. Export memory / run logs
        # =====================================================

        log_path = (
            Path(
                self.memory.log_path
            )
        )

        markdown_path = (
            log_path
            .with_suffix(
                ".md"
            )
        )

        json_path = (
            log_path
            .with_name(
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

    def _save_debug_artifact(
        self,
        directory: Path,
        name: str,
        content: str,
    ) -> None:
        """
        Persist one debugging artifact immediately.
        """

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        path = (
            directory
            / name
        )

        path.write_text(
            content,
            encoding="utf-8",
        )

    def _is_repairable_failure(
        self,
        result: ExperimentResult,
    ) -> bool:
        """
        Return True when an experiment failed for a technical
        implementation reason that can be repaired without changing
        the scientific hypothesis.
        """

        if (
            result.status
            == "success"
        ):

            return False

        repairable_stages = {
            "experiment_execution",
            "metric_parsing",
        }

        recovery_stages = {
            event.stage
            for event
            in result.recovery_events
        }

        return bool(
            recovery_stages
            & repairable_stages
        )

    def _classify_failure(
        self,
        result: ExperimentResult,
        decision: str,
        repair_events: list[
            RecoveryEvent
        ],
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
            in (
                repair_events
                + result.recovery_events
            )
        }

        if (
            "candidate_validation"
            in recovery_stages
        ):

            return (
                FailureType.CODE_ERROR
            )

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