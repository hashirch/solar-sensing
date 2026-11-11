"""Test bootstrap: point the app at a throwaway database before db/server are imported."""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_TMP = tempfile.mkdtemp(prefix="solar_tests_")
os.environ["SOLAR_DB_PATH"] = os.path.join(_TMP, "test.db")
