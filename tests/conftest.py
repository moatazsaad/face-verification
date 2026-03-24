import sys
from pathlib import Path

# Get the absolute path of the project root directory.
# __file__ is the location of this file (tests/conftest.py).
# parents[1] moves one level up from tests/ to the project root folder.
ROOT = Path(__file__).resolve().parents[1]

# Add the project root to Python's module search path.
# This allows test files to import modules using:
# from src.metrics import ...
# because Python will now look inside the project root where the src folder exists.
sys.path.insert(0, str(ROOT))