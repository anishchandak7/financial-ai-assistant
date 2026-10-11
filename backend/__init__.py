import sys
from pathlib import Path

# Ensure backend directory is in sys.path so submodules can be resolved directly
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
