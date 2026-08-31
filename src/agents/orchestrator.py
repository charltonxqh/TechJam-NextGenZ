"""
Description: Coordinates autonomous research actions, on-demand procedural skills, evidence-driven research, coding, deterministic candidate validation and research-integrity validation, semantic implementation verification, automatic repair, ML experiments, diagnostics, decision policy, persistent memory, and debugging artifacts.
Owner: Charlton / David
Input: Research session configuration and system components
Output: Completed research session and best experiment result
"""

import json
import time

from dataclasses import (
    asdict,
)

from datetime import (
    datetime,
    timezone,
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

from src.agents.implementation_verifier import (
    ImplementationVerifier,
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
    build_skill_catalog,
    get_skill_catalog_records,
    load_skill,
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

from src.tools.research_integrity_validator import (
    validate_research_integrity,
)

from src.tools.starterkit_runner import (
    run_starterkit_baseline,
)


INFORMATION_ACTION_BUDGET = 4

RESEARCH_REFRESH_AFTER_REJECTIONS = 2

MAX_CODE_REPAIR_ATTEMPTS = 2

MAX_ROUTING_CORRECTION_ATTEMPTS = 2


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

        self.implementation_verifier = (
            ImplementationVerifier(
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

        # =====================================================
        # Initialize progressive skill disclosure
        # =====================================================

        skill_catalog = (
            build_skill_catalog()
        )

        skill_catalog_records = (
            get_skill_catalog_records()
        )

        skill_catalog_path = (
            run_dir
            / "skill_catalog.json"
        )

        skill_catalog_path.write_text(
            json.dumps(
                skill_catalog_records,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        skill_trace_path = (
            run_dir
            / "skill_trace.jsonl"
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

        # Tracks scientifically valid experiments that execute successfully
        # but fail to improve the current validation best.
        successful_non_improving_streak = 0

        # Deterministic EDA tools operate on the same static dataset.
        # Once one has successfully completed, rerunning it provides
        # no new information during this research session.
        completed_eda_tools: set[
            str
        ] = set()

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

            research_actions_this_iteration = 0

            loaded_skills: dict[
                str,
                str,
            ] = {}

            attempted_research_queries: list[
                str
            ] = []

            attempted_research_query_keys: set[
                str
            ] = set()

            routing_correction_attempts = 0

            require_external_research = (
                next_iteration
                == 1
                or successful_non_improving_streak
                >= RESEARCH_REFRESH_AFTER_REJECTIONS
            )

            if (
                next_iteration
                == 1
            ):

                research_requirement_reason = (
                    "Fresh run: obtain external "
                    "evidence before the first "
                    "scientific experiment."
                )

            elif (
                successful_non_improving_streak
                >= RESEARCH_REFRESH_AFTER_REJECTIONS
            ):

                research_requirement_reason = (
                    "The last "
                    f"{successful_non_improving_streak} "
                    "scientifically valid experiments "
                    "did not improve the current best. "
                    "Refresh external evidence before "
                    "continuing."
                )

            else:

                research_requirement_reason = ""

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

                loaded_skills_context = (
                    self._build_loaded_skills_context(
                        loaded_skills
                    )
                )

                research_required_now = (
                    require_external_research
                    and research_actions_this_iteration
                    == 0
                )

                evidence_sufficiency_checkpoint = (
                    (
                        research_actions_this_iteration
                        >= 1
                        and bool(
                            completed_eda_tools
                        )
                    )
                    or (
                        research_actions_this_iteration
                        >= 2
                    )
                )

                allowed_actions = (
                    self._get_allowed_actions(
                        information_actions=(
                            information_actions
                        ),
                        research_required_now=(
                            research_required_now
                        ),
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
                        skill_catalog=(
                            skill_catalog
                        ),
                        loaded_skills_context=(
                            loaded_skills_context
                        ),
                        loaded_skill_names=list(
                            loaded_skills.keys()
                        ),
                        information_actions_used=(
                            information_actions
                        ),
                        information_action_budget=(
                            INFORMATION_ACTION_BUDGET
                        ),
                        research_actions_this_iteration=(
                            research_actions_this_iteration
                        ),
                        require_external_research=(
                            research_required_now
                        ),
                        research_requirement_reason=(
                            research_requirement_reason
                        ),
                        allowed_actions=(
                            allowed_actions
                        ),
                        completed_eda_tools=sorted(
                            completed_eda_tools
                        ),
                        attempted_research_queries=(
                            attempted_research_queries
                        ),
                        evidence_sufficiency_checkpoint=(
                            evidence_sufficiency_checkpoint
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
                # Enforce currently allowed action types
                # ---------------------------------------------

                if (
                    action.action_type
                    not in allowed_actions
                ):

                    routing_correction_attempts += 1

                    print(
                        "Research action rejected: "
                        f"'{action.action_type}' is "
                        "not currently allowed."
                    )

                    print(
                        "Allowed actions: "
                        + ", ".join(
                            allowed_actions
                        )
                    )

                    if (
                        routing_correction_attempts
                        >= MAX_ROUTING_CORRECTION_ATTEMPTS
                    ):

                        if (
                            "experiment"
                            in allowed_actions
                        ):

                            print(
                                "Routing correction: "
                                "the Researcher should "
                                "proceed to an experiment "
                                "unless a valid available "
                                "action is selected."
                            )

                        elif (
                            "research"
                            in allowed_actions
                        ):

                            print(
                                "Routing correction: "
                                "external research is "
                                "currently required."
                            )

                        routing_correction_attempts = 0

                    continue

                routing_correction_attempts = 0

                # ---------------------------------------------
                # On-demand skill action
                # ---------------------------------------------

                if (
                    action.action_type
                    == "load_skill"
                ):

                    requested_skills = (
                        action.skills
                        or []
                    )

                    newly_loaded = []

                    already_loaded = []

                    failed_skills = []

                    for skill_name in requested_skills:

                        if (
                            skill_name
                            in loaded_skills
                        ):

                            already_loaded.append(
                                skill_name
                            )

                            continue

                        try:

                            skill_content = (
                                load_skill(
                                    skill_name
                                )
                            )

                        except Exception as error:

                            failed_skills.append(
                                {
                                    "skill": (
                                        skill_name
                                    ),
                                    "error": (
                                        str(
                                            error
                                        )
                                    ),
                                }
                            )

                            continue

                        loaded_skills[
                            skill_name
                        ] = (
                            skill_content
                        )

                        newly_loaded.append(
                            skill_name
                        )

                    print(
                        "Skills requested: "
                        + (
                            ", ".join(
                                requested_skills
                            )
                            if requested_skills
                            else "None"
                        )
                    )

                    print(
                        "Skills loaded: "
                        + (
                            ", ".join(
                                newly_loaded
                            )
                            if newly_loaded
                            else "None"
                        )
                    )

                    if already_loaded:

                        print(
                            "Skills already loaded: "
                            + ", ".join(
                                already_loaded
                            )
                        )

                    print(
                        f"Loaded skills total: "
                        f"{len(loaded_skills)}"
                    )

                    self._append_jsonl(
                        path=(
                            skill_trace_path
                        ),
                        payload={
                            "timestamp": (
                                datetime.now(
                                    timezone.utc
                                )
                                .isoformat()
                            ),
                            "iteration": (
                                next_iteration
                            ),
                            "experiment_id": (
                                experiment_id
                            ),
                            "action_type": (
                                "load_skill"
                            ),
                            "reason": (
                                action.reason
                            ),
                            "requested_skills": (
                                requested_skills
                            ),
                            "newly_loaded": (
                                newly_loaded
                            ),
                            "already_loaded": (
                                already_loaded
                            ),
                            "failed_skills": (
                                failed_skills
                            ),
                            "loaded_skill_names": (
                                list(
                                    loaded_skills
                                    .keys()
                                )
                            ),
                            "loaded_skill_chars": {
                                name: len(
                                    content
                                )
                                for (
                                    name,
                                    content,
                                )
                                in loaded_skills
                                .items()
                            },
                        },
                    )

                    continue

                # ---------------------------------------------
                # Online research action
                # ---------------------------------------------

                if (
                    action.action_type
                    == "research"
                ):

                    research_query = (
                        action.research_query
                        or ""
                    ).strip()

                    knowledge_gap = (
                        action.knowledge_gap
                        or ""
                    ).strip()

                    research_query_key = (
                        self._normalize_research_query(
                            research_query
                        )
                    )

                    if (
                        research_query_key
                        in attempted_research_query_keys
                    ):

                        print(
                            "Research action skipped: "
                            "this research query was "
                            "already attempted during "
                            "the current scientific "
                            "iteration."
                        )

                        continue

                    if (
                        evidence_sufficiency_checkpoint
                        and not knowledge_gap
                    ):

                        print(
                            "Research action skipped: "
                            "the evidence-sufficiency "
                            "checkpoint is active, but "
                            "no concrete unresolved "
                            "knowledge gap was provided."
                        )

                        continue

                    if (
                        evidence_sufficiency_checkpoint
                        and self
                        ._research_uses_budget_as_reason(
                            reason=(
                                action.reason
                            ),
                            knowledge_gap=(
                                knowledge_gap
                            ),
                        )
                    ):

                        print(
                            "Research action skipped: "
                            "remaining information budget "
                            "is not a valid reason for "
                            "additional research."
                        )

                        continue

                    attempted_research_query_keys.add(
                        research_query_key
                    )

                    attempted_research_queries.append(
                        research_query
                    )

                    print(
                        f"Research knowledge gap: "
                        f"{knowledge_gap}"
                    )

                    print(
                        f"Research query: "
                        f"{research_query}"
                    )

                    added = (
                        research_runner.run(
                            query=(
                                research_query
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
                                + 1
                            ),
                        )
                    )

                    information_actions += 1

                    research_actions_this_iteration += 1

                    print(
                        f"Research evidence added: "
                        f"{added}"
                    )

                    if (
                        require_external_research
                    ):

                        successful_non_improving_streak = 0

                    continue

                # ---------------------------------------------
                # EDA action
                # ---------------------------------------------

                if (
                    action.action_type
                    == "eda"
                ):

                    eda_tool = (
                        action.eda_tool
                        or ""
                    )

                    if (
                        eda_tool
                        in completed_eda_tools
                    ):

                        print(
                            f"EDA action skipped: "
                            f"'{eda_tool}' has already "
                            "been completed during this "
                            "research session."
                        )

                        continue

                    print(
                        f"EDA request: "
                        f"{eda_tool}"
                    )

                    try:

                        findings = (
                            run_eda_tool(
                                name=(
                                    eda_tool
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

                        completed_eda_tools.add(
                            eda_tool
                        )

                        information_actions += 1

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
            # Log skills actually injected for this experiment
            # ---------------------------------------------

            self._append_jsonl(
                path=(
                    skill_trace_path
                ),
                payload={
                    "timestamp": (
                        datetime.now(
                            timezone.utc
                        )
                        .isoformat()
                    ),
                    "iteration": (
                        state.iteration
                    ),
                    "experiment_id": (
                        experiment_id
                    ),
                    "action_type": (
                        "experiment_skill_context"
                    ),
                    "loaded_skill_names": (
                        list(
                            loaded_skills
                            .keys()
                        )
                    ),
                    "loaded_skill_chars": {
                        name: len(
                            content
                        )
                        for (
                            name,
                            content,
                        )
                        in loaded_skills
                        .items()
                    },
                },
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
            # Validate / integrity-check / verify / repair / run
            # ---------------------------------------------

            repair_attempt = 0

            implemented_experiment = None

            result = None

            repair_events = []

            while True:

                # -----------------------------------------
                # Static candidate validation
                # -----------------------------------------

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
                # Research-integrity validation
                # -----------------------------------------

                integrity_validation = (
                    validate_research_integrity(
                        candidate_code
                    )
                )

                if not (
                    integrity_validation.valid
                ):

                    integrity_error = (
                        integrity_validation
                        .format_errors()
                    )

                    print(
                        "\nResearch integrity "
                        "validation failed:"
                    )

                    print(
                        integrity_error
                    )

                    self._save_debug_artifact(
                        directory=(
                            experiment_debug_dir
                        ),
                        name=(
                            f"integrity_error_"
                            f"{repair_attempt:02d}.txt"
                        ),
                        content=(
                            integrity_error
                        ),
                    )

                    if (
                        repair_attempt
                        >= MAX_CODE_REPAIR_ATTEMPTS
                    ):

                        repair_events.append(
                            RecoveryEvent(
                                stage=(
                                    "research_integrity"
                                ),
                                error=(
                                    integrity_error
                                ),
                                action=(
                                    "Candidate was blocked "
                                    "by the research-integrity "
                                    "guard after exhausting "
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
                                    integrity_error
                                ),
                            )
                        )

                        break

                    repair_attempt += 1

                    print(
                        f"Research-integrity repair "
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
                                integrity_error
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
                                "research_integrity"
                            ),
                            error=(
                                integrity_error
                            ),
                            action=(
                                "Coder repaired a "
                                "research-integrity "
                                "violation within the "
                                "same scientific "
                                "iteration."
                            ),
                            success=True,
                        )
                    )

                    continue

                # -----------------------------------------
                # Researcher <-> Coder implementation verification
                # -----------------------------------------

                implementation_verification = (
                    self.implementation_verifier
                    .verify(
                        spec=(
                            spec
                        ),
                        current_best_code=(
                            current_best_code
                        ),
                        candidate_code=(
                            candidate_code
                        ),
                        data_context=(
                            data_context
                        ),
                    )
                )

                if not (
                    implementation_verification
                    .faithful
                ):

                    verification_error = (
                        self.implementation_verifier
                        .format_failure(
                            implementation_verification
                        )
                    )

                    print(
                        "\nImplementation "
                        "verification failed:"
                    )

                    print(
                        verification_error
                    )

                    self._save_debug_artifact(
                        directory=(
                            experiment_debug_dir
                        ),
                        name=(
                            f"implementation_verification_"
                            f"error_"
                            f"{repair_attempt:02d}.txt"
                        ),
                        content=(
                            verification_error
                        ),
                    )

                    if (
                        repair_attempt
                        >= MAX_CODE_REPAIR_ATTEMPTS
                    ):

                        repair_events.append(
                            RecoveryEvent(
                                stage=(
                                    "implementation_verification"
                                ),
                                error=(
                                    verification_error
                                ),
                                action=(
                                    "Candidate failed "
                                    "Researcher-to-Coder "
                                    "implementation fidelity "
                                    "verification after "
                                    "exhausting the automatic "
                                    "repair budget."
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
                                    verification_error
                                ),
                            )
                        )

                        break

                    repair_attempt += 1

                    print(
                        f"Implementation-fidelity "
                        f"repair "
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
                                verification_error
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
                                "implementation_verification"
                            ),
                            error=(
                                verification_error
                            ),
                            action=(
                                "Coder repaired the "
                                "candidate to better match "
                                "the Researcher's fixed "
                                "ExperimentSpec."
                            ),
                            success=True,
                        )
                    )

                    continue

                self._save_debug_artifact(
                    directory=(
                        experiment_debug_dir
                    ),
                    name=(
                        "implementation_verification_"
                        "passed.txt"
                    ),
                    content=(
                        implementation_verification
                        .summary
                    ),
                )

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

            # Technical/integrity/fidelity failures do not count as
            # scientific no-improvement observations.
            if (
                result.status
                == "success"
            ):

                state.best_primary_history.append(
                    state.best_primary
                )

                if (
                    decision
                    == "keep"
                ):

                    successful_non_improving_streak = 0

                else:

                    successful_non_improving_streak += 1

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

    def _get_allowed_actions(
        self,
        information_actions: int,
        research_required_now: bool,
    ) -> list[str]:
        """
        Determine which Researcher actions are currently legal.
        """

        if research_required_now:

            return [
                "research"
            ]

        if (
            information_actions
            >= INFORMATION_ACTION_BUDGET
        ):

            return [
                "load_skill",
                "experiment",
            ]

        return [
            "load_skill",
            "experiment",
            "research",
            "eda",
        ]

    def _normalize_research_query(
        self,
        query: str,
    ) -> str:
        """
        Normalize one research query for deterministic duplicate detection.
        """

        return " ".join(
            query.lower()
            .split()
        )

    def _research_uses_budget_as_reason(
        self,
        reason: str,
        knowledge_gap: str,
    ) -> bool:
        """
        Detect research requests whose stated justification is merely the
        existence of unused information budget rather than a technical gap.
        """

        reason_text = (
            reason.lower()
        )

        gap_text = (
            knowledge_gap.lower()
        )

        budget_phrases = [
            "remaining information-gathering budget",
            "remaining information gathering budget",
            "remaining research budget",
            "remaining budget",
            "budget remaining",
            "still have budget",
            "have remaining budget",
        ]

        generic_gap_phrases = [
            "more evidence",
            "more research",
            "more information",
            "effective architectures",
            "effective model architectures",
            "state-of-the-art methods",
            "state of the art methods",
            "better recommendation methods",
            "better recommender models",
        ]

        reason_uses_budget = any(
            phrase in reason_text
            for phrase
            in budget_phrases
        )

        gap_is_generic = any(
            gap_text.strip()
            == phrase
            for phrase
            in generic_gap_phrases
        )

        return (
            reason_uses_budget
            or gap_is_generic
        )

    def _build_loaded_skills_context(
        self,
        loaded_skills: dict[
            str,
            str,
        ],
    ) -> str:
        """
        Build the full procedural context for only skills selected during
        the current scientific iteration.
        """

        if not loaded_skills:

            return ""

        blocks = []

        for (
            skill_name,
            skill_content,
        ) in loaded_skills.items():

            blocks.append(
                (
                    f"=== SKILL: "
                    f"{skill_name} ===\n"
                    f"{skill_content}"
                )
            )

        return "\n\n".join(
            blocks
        )

    def _append_jsonl(
        self,
        path: Path,
        payload: dict,
    ) -> None:
        """
        Append one structured trace event.
        """

        with path.open(
            "a",
            encoding="utf-8",
        ) as file:

            file.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )

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
            "research_integrity"
            in recovery_stages
        ):

            return (
                FailureType.CODE_ERROR
            )

        if (
            "implementation_verification"
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