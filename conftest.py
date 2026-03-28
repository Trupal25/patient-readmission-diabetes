# Root conftest — pytest discovers src/ as a package from project root
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `from src.x import y` works
sys.path.insert(0, str(Path(__file__).resolve().parent))
