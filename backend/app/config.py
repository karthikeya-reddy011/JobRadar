import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_ROOT / ".env")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://jobradar:jobradar@localhost:5434/jobradar",
)
USER_AGENT = os.getenv("USER_AGENT", "JobRadar/0.1 (personal project)")

# backend/app/config.py -> repo root is two levels above backend/app
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPANIES_FILE = REPO_ROOT / "companies.yaml"
