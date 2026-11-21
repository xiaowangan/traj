"""Resource roots belonging exclusively to this migrated application."""
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESOURCE_ROOT = PROJECT_ROOT / "resources"
EXAMPLE_ROOT = PROJECT_ROOT / "data" / "examples"
