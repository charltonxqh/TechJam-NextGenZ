"""
Description: Defines and loads shared configuration settings used across the autonomous research system.
Owner: Shared
Input: Environment variables and configuration values
Output: System configuration
"""

import os

from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


# =========================
# Project Paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STARTER_KIT_DIR = (
    PROJECT_ROOT
    / "kuairand-starter-kit"
)

# The directory holding the KuaiRand-Pure CSVs. Passed straight through to
# baseline.py as --data_dir, so it must be the folder containing the logs,
# not its parent. Override with DATA_DIR in .env if the kit lives elsewhere.
DATA_DIR = Path(
    os.getenv(
        "DATA_DIR",
        STARTER_KIT_DIR
        / "KuaiRand-Pure"
        / "data",
    )
)

RUNS_DIR = PROJECT_ROOT / "runs"

WORKSPACES_DIR = (
    PROJECT_ROOT
    / "workspaces"
)


# =========================
# Research Loop
# =========================

MAX_ITERATIONS = 50

CONVERGENCE_EPSILON = 0.002

CONVERGENCE_PATIENCE = 3

EXPERIMENT_TIMEOUT = 3600


# =========================
# Gemini
# =========================

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY",
    "",
)

RESEARCHER_MODEL = (
    "gemini-3.5-flash-lite"
)

CODER_MODEL = (
    "gemini-3.5-flash-lite"
)

IMPLEMENTATION_VERIFIER_MODEL = (
    "gemini-3.5-flash-lite"
)

DEFAULT_MODEL = (
    "gemini-3.5-flash-lite"
)

LIGHT_MODEL = (
    "gemini-3.1-flash-lite"
)