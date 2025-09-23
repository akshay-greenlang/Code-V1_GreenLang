"""
Main entry point for the gl CLI when running as module
"""

import sys
import os

# Fix encoding issues on Windows
if sys.platform == "win32":
    # Set environment variable for proper encoding
    os.environ["PYTHONIOENCODING"] = "utf-8"
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        # Python < 3.7 fallback
        import codecs

        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

from . import main

if __name__ == "__main__":
    main()
