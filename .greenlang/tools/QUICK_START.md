# Quick Start Guide - AI-Powered Developer Tools

Get started with GreenLang's AI-powered developer tools in 5 minutes!

## Installation

```bash
# 1. Navigate to tools directory
cd .greenlang/tools/

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify installation
cd ../..
python .greenlang/cli/greenlang.py --help
```

## First Steps

### 1. Search for Infrastructure (30 seconds)

```bash
# Find caching solutions
greenlang search "cache API responses"

# Browse LLM components
greenlang search --category llm

# Find validation tools
greenlang search --tag validation
```

**What you get:** Top 5 matching components with code examples

---

### 2. Check Your Code Health (1 minute)

```bash
# Check current directory
greenlang health-check

# Check specific app
greenlang health-check --directory GL-CBAM-APP --format html --output health.html
```

**What you get:** IUM score (0-100) and actionable recommendations

---

### 3. Get Code Recommendations (1 minute)

```bash
# Analyze a file
greenlang recommend my_agent.py

# Analyze entire directory
greenlang recommend GL-CBAM-APP/ --format html --output recommendations.html
```

**What you get:** Specific suggestions to improve your code

---

### 4. Ask the AI Assistant (instant)

```bash
# Interactive chat
greenlang chat

# Quick question
greenlang chat "How do I add caching to my agent?"

# Help on a topic
greenlang chat --help-topic llm
```

**What you get:** Instant answers with code examples

---

### 5. Generate New Code (30 seconds)

```bash
# From natural language
greenlang smart-generate "Create an agent that validates CSV files"

# Interactive mode
greenlang smart-generate --interactive

# Preview first
greenlang smart-generate "Reporting agent" --preview
```

**What you get:** Complete agent with tests and documentation

---

## Common Workflows

### Workflow 1: "I need to add feature X"

```bash
# 1. Search for relevant infrastructure
greenlang search "feature X"

# 2. Get code example
greenlang chat "How to implement X?"

# 3. Generate starter code
greenlang smart-generate "Agent for X" --preview

# 4. Check health after adding
greenlang health-check
```

---

### Workflow 2: "My code needs improvement"

```bash
# 1. Check current health
greenlang health-check --directory my-app/

# 2. Get specific recommendations
greenlang recommend my-app/ --format html

# 3. Use migration wizard
greenlang migrate-wizard

# 4. Verify improvements
greenlang health-check --directory my-app/
```

---

### Workflow 3: "I'm new to the codebase"

```bash
# 1. Explore infrastructure
greenlang explore

# 2. See dependency graph
greenlang dep-graph --interactive

# 3. Read documentation
greenlang docs

# 4. Ask questions
greenlang chat
```

---

### Workflow 4: "Performance issues"

```bash
# 1. Profile your code
greenlang profile slow_agent.py

# 2. Get optimization suggestions
greenlang recommend slow_agent.py

# 3. Check cache performance
greenlang chat "How to improve cache hit rate?"

# 4. Re-profile
greenlang profile slow_agent.py
```

---

## All Commands At-a-Glance

### Discovery & Search
```bash
greenlang search <query>              # Semantic search
greenlang explore                     # Web-based explorer
greenlang chat <question>             # AI assistant
```

### Code Analysis
```bash
greenlang recommend <path>            # Get recommendations
greenlang health-check                # IUM score & health
greenlang dep-graph                   # Dependency visualization
greenlang profile <file>              # Performance profiling
```

### Code Generation
```bash
greenlang smart-generate <desc>       # From natural language
greenlang generate --type agent       # Template-based
greenlang docs                        # Auto-documentation
```

### Migration
```bash
greenlang migrate-wizard              # Interactive wizard
greenlang migrate --app <name>        # Batch migration
```

---

## Tips & Tricks

### 1. Pipe outputs for processing
```bash
greenlang search "cache" --format json | jq '.[]'
```

### 2. Combine tools
```bash
# Generate, then check health
greenlang smart-generate "My agent" --output agents/my/
greenlang health-check --directory agents/my/
```

### 3. Save reports for later
```bash
greenlang health-check --format html --output daily-health.html
greenlang dep-graph --interactive --output deps-$(date +%Y%m%d).html
```

### 4. Use chat for quick help
```bash
# Faster than searching docs
greenlang chat "What's the difference between BaseAgent and regular class?"
```

### 5. Preview before generating
```bash
# See what will be created
greenlang smart-generate "Agent X" --preview
```

---

## Troubleshooting

### "sentence-transformers not found"
```bash
pip install sentence-transformers
# Or use without: search will use keyword matching
```

### "streamlit not found"
```bash
pip install streamlit
# Required for greenlang explore
```

### "Command not found: greenlang"
```bash
# Use direct path
python .greenlang/cli/greenlang.py <command>

# Or add to PATH
export PATH="$PATH:$(pwd)/.greenlang/cli"
alias greenlang='python $(pwd)/.greenlang/cli/greenlang.py'
```

### "No module named 'shared'"
```bash
# Run from repository root
cd /path/to/Code-V1_GreenLang
greenlang <command>
```

---

## Next Steps

1. **Explore all tools:** Try each command at least once
2. **Check your apps:** Run health-check on all applications
3. **Migrate incrementally:** Use migrate-wizard for one component at a time
4. **Generate new code:** Use smart-generate for new features
5. **Ask questions:** Use chat whenever stuck

---

## Getting Help

### Built-in Help
```bash
greenlang --help                      # All commands
greenlang search --help               # Specific command
greenlang chat --help-topic llm       # Topic help
```

### AI Assistant
```bash
greenlang chat                        # Interactive Q&A
greenlang chat "Your question"        # Quick question
```

### Documentation
```bash
greenlang explore                     # Web UI
greenlang docs                        # Generate API docs
cat .greenlang/tools/README.md        # Full README
```

---

## What's Next?

After getting comfortable with the basics:

- **Integrate with CI/CD:** Add health-check to your pipeline
- **Team adoption:** Share dep-graph with team
- **Metrics tracking:** Regular health-check reports
- **Code generation:** Use smart-generate for new features
- **Migration:** Systematic migration with migrate-wizard

---

**Happy coding with GreenLang AI tools!** 🌿

For more information, see:
- Full README: `.greenlang/tools/README.md`
- Delivery Report: `.greenlang/tools/DELIVERY_REPORT.md`
- Individual tool help: `greenlang <command> --help`
