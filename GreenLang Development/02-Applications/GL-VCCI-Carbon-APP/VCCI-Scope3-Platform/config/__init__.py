# -*- coding: utf-8 -*-
# GL-VCCI Configuration Module
# Configuration management and settings

"""
VCCI Configuration
==================

Centralized configuration management for the VCCI Scope 3 Platform.

Configuration Files:
-------------------
- vcci_config.yaml: Main application configuration
- .env: Environment variables and secrets
- pack.yaml: GreenLang pack specification
- gl.yaml: Agent runtime configuration

Usage:
------
```python
from config import load_config, get_setting

# Load configuration
config = load_config("config/vcci_config.yaml")

# Get specific setting
db_host = get_setting("database.primary.host", default="localhost")

# Access nested settings
llm_provider = config.llm.default_provider
max_workers = config.performance.max_workers
```

Environment Variables:
---------------------
Configuration supports environment variable substitution:

```yaml
database:
  host: "${DATABASE_HOST:localhost}"
  password: "${DATABASE_PASSWORD}"  # Required, no default
```

Settings Hierarchy:
------------------
1. Environment variables (highest priority)
2. .env file
3. vcci_config.yaml
4. Default values (lowest priority)
"""

__version__ = "1.0.0"

__all__ = [
    # "load_config",
    # "get_setting",
    # "VCCIConfig",
]
