"""Compatibility entry point for the dashboard browser check and captures."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("check_browser.py")), run_name="__main__")
