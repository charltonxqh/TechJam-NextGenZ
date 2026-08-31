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

DATA_DIR = PROJECT_ROOT / "data"

STARTER_KIT_DIR = (
    PROJECT_ROOT
    / "kuairand-starter-kit"
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
    "gemini-3.7-flash"
)

CODER_MODEL = (
    "gemini-3.7-flash"
)

IMPLEMENTATION_VERIFIER_MODEL = (
    "gemini-3.7-flash"
)

DEFAULT_MODEL = (
    "gemini-3.7-flash"
)

LIGHT_MODEL = (
    "gemini-3.1-flash-lite"
)