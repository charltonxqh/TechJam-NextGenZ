"""
Description: Initializes all system components and starts the autonomous ML research session.
Owner: Charlton / David
Input: System configuration
Output: Completed autonomous research session
"""

from datetime import (
    datetime,
)

from src.agents.orchestrator import (
    Orchestrator,
)

from src.agents.researcher import (
    Researcher,
)

from src.config import (
    RUNS_DIR,
)

from src.memory.memory_store import (
    MemoryStore,
)

from src.tools.llm_client import (
    GeminiClient,
)


def main() -> None:
    """
    Create all components and start one autonomous research session.
    """

    # ---------------------------------------------
    # Create unique run ID
    # ---------------------------------------------

    run_id = (
        datetime.now().strftime(
            "run_%Y%m%d_%H%M%S"
        )
    )

    print(
        f"Run ID: "
        f"{run_id}"
    )

    # ---------------------------------------------
    # Shared LLM client
    # ---------------------------------------------

    llm_client = (
        GeminiClient()
    )

    # ---------------------------------------------
    # Autonomous research agent
    # ---------------------------------------------

    researcher = (
        Researcher(
            llm_client=(
                llm_client
            ),
        )
    )

    # ---------------------------------------------
    # Persistent research memory
    # ---------------------------------------------

    run_dir = (
        RUNS_DIR
        / run_id
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    memory_store = (
        MemoryStore(
            log_path=str(
                run_dir
                / "run_log.jsonl"
            )
        )
    )

    # ---------------------------------------------
    # Main workflow controller
    # ---------------------------------------------

    orchestrator = (
        Orchestrator(
            researcher=(
                researcher
            ),
            memory_store=(
                memory_store
            ),
        )
    )

    # ---------------------------------------------
    # Start autonomous research session
    # ---------------------------------------------

    final_state = (
        orchestrator.run()
    )

    # ---------------------------------------------
    # Final result
    # ---------------------------------------------

    print(
        "\n========== Final Result =========="
    )

    print(
        f"Best experiment: "
        f"{final_state.best_experiment_id}"
    )

    print(
        f"Best Validation Primary: "
        f"{final_state.best_primary:.4f}"
    )

    if (
        final_state.final_test_primary
        is not None
    ):

        print(
            f"Final Test GAUC: "
            f"{final_state.final_test_gauc:.4f}"
        )

        print(
            f"Final Test nDCG@5: "
            f"{final_state.final_test_ndcg5:.4f}"
        )

        print(
            f"Final Test Primary: "
            f"{final_state.final_test_primary:.4f}"
        )


if __name__ == "__main__":
    main()