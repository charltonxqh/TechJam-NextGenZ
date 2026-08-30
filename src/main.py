"""
Description: Initializes all system components and starts the autonomous ML research session.
Owner: Charlton / David
Input: System configuration
Output: Completed autonomous research session
"""

from datetime import datetime

from src.agents.orchestrator import Orchestrator
from src.agents.researcher import Researcher
from src.agents.reflector import Reflector

from src.memory.experiment_store import ExperimentStore

from src.tools.llm_client import GeminiClient
from src.tools.trae_coding_agent import TraeCodingAgent


def main() -> None:
    """
    Create all components and start one autonomous research session.
    """

    # ---------------------------------------------
    # Create unique run ID
    # ---------------------------------------------

    run_id = datetime.now().strftime(
        "run_%Y%m%d_%H%M%S"
    )

    print(
        f"Run ID: {run_id}"
    )

    # ---------------------------------------------
    # Shared LLM client
    # ---------------------------------------------

    llm_client = GeminiClient()

    # ---------------------------------------------
    # Agent reasoning components
    # ---------------------------------------------

    researcher = Researcher(
        llm_client=llm_client,
    )

    reflector = Reflector(
        llm_client=llm_client,
    )

    # ---------------------------------------------
    # Coding agent
    # ---------------------------------------------

    coding_agent = TraeCodingAgent()

    # ---------------------------------------------
    # Persistent experiment memory
    # ---------------------------------------------

    experiment_store = ExperimentStore(
        run_id=run_id,
    )

    # ---------------------------------------------
    # Main workflow controller
    # ---------------------------------------------

    orchestrator = Orchestrator(
        researcher=researcher,
        reflector=reflector,
        coding_agent=coding_agent,
        experiment_store=experiment_store,
    )

    # ---------------------------------------------
    # Start autonomous research session
    # ---------------------------------------------

    final_state = orchestrator.run()

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
        f"Best Primary: "
        f"{final_state.best_primary:.4f}"
    )


if __name__ == "__main__":
    main()