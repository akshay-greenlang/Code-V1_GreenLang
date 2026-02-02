"""
Pytest Configuration for Framework_GreenLang Tools Tests

Ensures proper path setup for imports.
"""

import sys
import os
from pathlib import Path

# Add Framework_GreenLang directory to path for imports
_framework_path = Path(__file__).parent.parent.parent
if str(_framework_path) not in sys.path:
    sys.path.insert(0, str(_framework_path))
