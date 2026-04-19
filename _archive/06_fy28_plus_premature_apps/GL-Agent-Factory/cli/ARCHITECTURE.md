# GreenLang Agent Factory CLI - Architecture

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER TERMINAL                               │
│                                                                 │
│  $ gl agent create specs/my-agent.yaml                         │
│  $ gl agent test agents/my-agent --coverage                    │
│  $ gl registry publish agents/my-agent                         │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  CLI ENTRY POINT (main.py)                      │
│                                                                 │
│  ┌─────────────────────────────────────────────────────┐       │
│  │  Typer Application                                  │       │
│  │  - Global options (--version, --quiet, --help)     │       │
│  │  - Context management                               │       │
│  │  - Command routing                                  │       │
│  └─────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   AGENT      │   │  TEMPLATE    │   │   REGISTRY   │
│  COMMANDS    │   │  COMMANDS    │   │   COMMANDS   │
│              │   │              │   │              │
│ commands/    │   │ commands/    │   │ commands/    │
│ agent.py     │   │ template.py  │   │ registry.py  │
└──────────────┘   └──────────────┘   └──────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   CONSOLE    │   │    CONFIG    │   │  TEMPLATES   │
│   UTILITIES  │   │ MANAGEMENT   │   │              │
│              │   │              │   │              │
│ utils/       │   │ utils/       │   │ templates/   │
│ console.py   │   │ config.py    │   │ *.yaml       │
└──────────────┘   └──────────────┘   └──────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   EXTERNAL SYSTEMS                              │
│                                                                 │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐               │
│  │ File       │  │  Agent     │  │  Registry  │               │
│  │ System     │  │  SDK       │  │  API       │               │
│  └────────────┘  └────────────┘  └────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. CLI Entry Point (main.py)

```
main.py
├── app: Typer()
│   ├── callback() - Global options handler
│   ├── init() - Project initialization
│   └── Command groups:
│       ├── agent (agent.app)
│       ├── template (template.app)
│       └── registry (registry.app)
└── cli_main() - Entry point function
```

### 2. Agent Commands (commands/agent.py)

```
agent.py
├── app: Typer() - Agent command group
├── Commands:
│   ├── create()    - Generate agent
│   ├── validate()  - Validate spec
│   ├── test()      - Run tests
│   ├── publish()   - Publish to registry
│   ├── list()      - List agents
│   └── info()      - Show details
└── Helpers:
    ├── validate_spec()
    ├── generate_core_agent()
    ├── generate_config_files()
    ├── generate_tests()
    ├── generate_documentation()
    ├── generate_deployment_configs()
    ├── run_agent_tests()
    ├── package_agent()
    └── upload_to_registry()
```

### 3. Console Utilities (utils/console.py)

```
console.py
├── console: Console() - Rich console instance
├── Output Functions:
│   ├── print_error()
│   ├── print_success()
│   ├── print_warning()
│   └── print_info()
├── Display Functions:
│   ├── create_agent_table()
│   ├── create_directory_tree()
│   ├── create_progress_bar()
│   ├── create_info_panel()
│   ├── display_code()
│   ├── display_yaml()
│   └── display_markdown()
└── Specialized Functions:
    ├── print_validation_results()
    ├── print_test_results()
    ├── print_generation_summary()
    └── confirm_action()
```

### 4. Configuration Management (utils/config.py)

```
config.py
├── DEFAULT_CONFIG - Default settings
├── Functions:
│   ├── get_config_path()
│   ├── load_config()
│   ├── save_config()
│   ├── get_config_value()
│   └── update_config_value()
└── Configuration Structure:
    ├── version
    ├── defaults
    ├── registry
    ├── generator
    ├── validation
    └── testing
```

## Data Flow

### Agent Creation Flow

```
1. User Input
   $ gl agent create specs/my-agent.yaml --output agents/my-agent
                            │
                            ▼
2. Command Parsing (main.py)
   - Parse arguments
   - Load configuration
   - Route to agent.create()
                            │
                            ▼
3. Specification Loading (agent.py)
   - Read YAML file
   - Parse with PyYAML
   - Create spec dict
                            │
                            ▼
4. Validation (validate_spec)
   - Check required fields
   - Validate structure
   - Generate warnings/errors
   - Display results (console.py)
                            │
                            ▼
5. Generation (if valid)
   ├── generate_core_agent()
   │   └── Create agent.py
   ├── generate_config_files()
   │   └── Create agent.yaml
   ├── generate_tests()
   │   └── Create test files
   ├── generate_documentation()
   │   └── Create README.md
   └── generate_deployment_configs()
       └── Create Dockerfile
                            │
                            ▼
6. Progress Display (console.py)
   - Show progress bars
   - Display file tree
   - Print summary
                            │
                            ▼
7. Output
   ✓ Agent generated successfully!

   📦 my-agent
   ├── 📄 agent.py
   ├── 📄 agent.yaml
   ├── 📄 README.md
   ├── 📄 Dockerfile
   └── 📁 tests
       └── 📄 test_agent.py
```

### Testing Flow

```
1. User Input
   $ gl agent test agents/my-agent --coverage
                            │
                            ▼
2. Test Discovery
   - Find test directory
   - Locate test files
   - Check test framework
                            │
                            ▼
3. Test Execution
   - Run pytest
   - Collect results
   - Generate coverage
                            │
                            ▼
4. Result Display (console.py)
   ┏━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
   ┃ Test Results                ┃
   ┡━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
   │ Total Tests    │ 10      │
   │ Passed         │ 9       │
   │ Failed         │ 1       │
   │ Success Rate   │ 90.0%   │
   └────────────────┴─────────┘
```

### Registry Flow

```
1. Search Registry
   $ gl registry search "compliance"
                            │
                            ▼
2. API Request
   - Load config (registry URL)
   - Build search query
   - Send HTTP request
                            │
                            ▼
3. Parse Results
   - Parse JSON response
   - Filter results
   - Format for display
                            │
                            ▼
4. Display (console.py)
   ┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
   ┃ Search Results                    ┃
   ┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
   │ nfpa86-agent    │ v1.2.0        │
   │ compliance-tool │ v2.0.1        │
   └─────────────────┴───────────────┘
```

## Class Hierarchy

```
Typer (External)
└── app (main.py)
    ├── agent.app (commands/agent.py)
    │   ├── create()
    │   ├── validate()
    │   ├── test()
    │   ├── publish()
    │   ├── list()
    │   └── info()
    ├── template.app (commands/template.py)
    │   ├── list()
    │   ├── init()
    │   └── show()
    └── registry.app (commands/registry.py)
        ├── search()
        ├── pull()
        ├── push()
        ├── login()
        └── logout()

Rich (External)
└── Console
    ├── print()
    ├── print_json()
    ├── status()
    └── input()
        └── Used by console.py utilities
```

## File Structure Map

```
cli/
├── Package Root
│   ├── __init__.py           ┐
│   ├── main.py              │ Core CLI
│   │                         ┘
│   ├── commands/             ┐
│   │   ├── __init__.py      │
│   │   ├── agent.py         │ Command
│   │   ├── template.py      │ Groups
│   │   └── registry.py      │
│   │                         ┘
│   ├── utils/                ┐
│   │   ├── __init__.py      │
│   │   ├── console.py       │ Utilities
│   │   └── config.py        │
│   │                         ┘
│   └── templates/            ┐
│       ├── basic-agent-spec.yaml     │ Templates
│       └── regulatory-agent-spec.yaml┘
│
├── Configuration Files
│   ├── pyproject.toml        ┐
│   ├── setup.py             │ Packaging
│   ├── requirements.txt     │
│   └── MANIFEST.in          ┘
│
├── Documentation
│   ├── README.md             ┐
│   ├── INSTALL.md           │
│   ├── QUICKSTART.md        │ Docs
│   ├── CHANGELOG.md         │
│   └── ARCHITECTURE.md      │ (this file)
│                             ┘
├── Development
│   ├── tests/                ┐
│   │   ├── __init__.py      │
│   │   ├── conftest.py      │ Testing
│   │   ├── test_cli_main.py │
│   │   └── test_config.py   │
│   │                         ┘
│   ├── Makefile              ┐
│   ├── .gitignore           │ Dev Tools
│   └── LICENSE              ┘
│
└── Generated at Runtime
    ├── build/
    ├── dist/
    ├── *.egg-info/
    ├── __pycache__/
    └── htmlcov/
```

## Technology Stack

### Core Technologies

```
┌──────────────────────────────────────────┐
│         Python 3.11+                     │
└──────────────────────────────────────────┘
              │
    ┌─────────┼─────────┐
    │         │         │
    ▼         ▼         ▼
┌────────┐ ┌────────┐ ┌────────┐
│ Typer  │ │  Rich  │ │ PyYAML │
│ 0.12.0+│ │ 13.7.0+│ │ 6.0.1+ │
└────────┘ └────────┘ └────────┘
    │         │         │
    │         │         │
    ▼         ▼         ▼
┌────────────────────────────────┐
│    CLI Application             │
│  - Commands                    │
│  - Rich UI                     │
│  - Configuration               │
└────────────────────────────────┘
```

### Dependencies Graph

```
CLI Package
├── typer[all] >=0.12.0
│   └── click >=8.1.7
│       └── colorama (Windows support)
├── rich >=13.7.0
│   ├── markdown-it-py
│   ├── pygments (syntax highlighting)
│   └── typing-extensions
├── pyyaml >=6.0.1
├── pydantic >=2.5.0
│   └── typing-extensions
├── requests >=2.31.0
│   └── urllib3
└── jinja2 >=3.1.2
    └── MarkupSafe

Development Dependencies
├── pytest >=7.4.3
│   └── pluggy
├── pytest-cov >=4.1.0
│   └── coverage
├── black >=23.12.0
│   ├── click
│   └── platformdirs
├── ruff >=0.1.8
└── mypy >=1.7.1
    └── typing-extensions
```

## Design Patterns

### 1. Command Pattern
```python
# Each command is a separate function
@app.command()
def create(...):
    # Command implementation
    pass
```

### 2. Factory Pattern
```python
# Console utilities create different display types
def create_agent_table(...) -> Table:
    # Create and return table

def create_progress_bar(...) -> Progress:
    # Create and return progress bar
```

### 3. Configuration Pattern
```python
# Centralized configuration management
config = load_config()
value = get_config_value("key.nested.path", default="value")
```

### 4. Template Pattern
```python
# Agent generation follows template
def generate_agent(spec, template):
    # Use template to generate structure
    pass
```

## Extension Points

### 1. Custom Commands
```python
# Add new command to agent.py
@app.command()
def new_command():
    # Implementation
    pass
```

### 2. Custom Templates
```python
# Add template to templates/
# Use in: gl template init <name>
```

### 3. Custom Generators
```python
# Extend generate_* functions
def generate_custom_files(spec, output_dir):
    # Custom generation logic
    pass
```

### 4. Custom Validators
```python
# Extend validate_spec()
def validate_spec(spec, custom_rules=None):
    # Add custom validation
    pass
```

## Security Considerations

### 1. Input Validation
- All file paths validated
- YAML parsing with safe_load
- Command injection prevention
- Path traversal protection

### 2. Configuration Security
- Secrets not logged
- Credentials encrypted
- Config file permissions
- Environment variable support

### 3. Registry Security
- HTTPS only
- Authentication required
- Token-based auth
- Version verification

## Performance Optimization

### 1. Lazy Loading
```python
# Import only when needed
def heavy_operation():
    import heavy_module
    # Use module
```

### 2. Progress Indication
```python
# Show progress for long operations
with create_progress_bar() as progress:
    task = progress.add_task("Processing...")
    # Long operation
```

### 3. Parallel Execution
```python
# Tests can run in parallel
gl agent test --parallel
```

## Error Handling Strategy

```
Error occurs
    │
    ▼
┌─────────────────────┐
│ Catch Exception     │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ Format Error        │
│ - User-friendly msg │
│ - Color coding      │
│ - Suggestions       │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ Display (console.py)│
│ print_error()       │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ Exit with code 1    │
└─────────────────────┘
```

## Future Architecture

### Planned Enhancements

1. **Plugin System**
```
cli/
└── plugins/
    ├── __init__.py
    ├── loader.py
    └── custom/
        └── my_plugin.py
```

2. **API Server Mode**
```
cli/
└── server/
    ├── __init__.py
    ├── app.py (FastAPI)
    └── routes/
```

3. **Web UI**
```
cli/
└── web/
    ├── static/
    ├── templates/
    └── app.py
```

4. **Advanced Templates**
```
templates/
├── engines/
│   ├── jinja2/
│   └── mustache/
└── library/
    ├── basic/
    ├── regulatory/
    └── custom/
```

## Conclusion

The CLI architecture is:
- **Modular**: Clear separation of concerns
- **Extensible**: Easy to add new commands/features
- **Maintainable**: Clean code organization
- **User-friendly**: Rich terminal experience
- **Production-ready**: Professional quality

---

**Last Updated**: 2024-12-09
**Version**: 0.1.0
**Status**: Production-Ready
