# GreenLang Training Exercises

This directory contains hands-on exercises for GreenLang training programs.

## Exercise Overview

| Exercise | Target Audience | Duration | Difficulty |
|----------|-----------------|----------|------------|
| [01_basic_calculations.py](01_basic_calculations.py) | All | 30 min | Beginner |
| [02_pipeline_creation.py](02_pipeline_creation.py) | Developers | 45 min | Intermediate |
| [03_data_validation.py](03_data_validation.py) | Operators | 30 min | Beginner |
| [04_report_generation.py](04_report_generation.py) | Operators | 30 min | Beginner |
| [05_api_integration.py](05_api_integration.py) | Developers | 60 min | Intermediate |
| [06_custom_agent.py](06_custom_agent.py) | Developers | 90 min | Advanced |
| [07_troubleshooting_scenarios.md](07_troubleshooting_scenarios.md) | All | 60 min | Intermediate |

## Prerequisites

Before starting the exercises, ensure you have:

1. Completed the relevant training module
2. GreenLang installed and configured
3. Access to the test environment
4. Python 3.10+ installed

## Setup

```bash
# Clone exercises (if not already done)
cd greenlang/docs/training/training_exercises

# Install exercise dependencies
pip install -r requirements.txt

# Verify setup
python verify_setup.py
```

## Running Exercises

Each exercise is self-contained. Run them as follows:

```bash
# Run an exercise
python 01_basic_calculations.py

# Run with solutions shown
python 01_basic_calculations.py --show-solutions

# Run in interactive mode
python 01_basic_calculations.py --interactive
```

## Exercise Structure

Each exercise file follows this structure:

```python
"""
Exercise: Title
Difficulty: Beginner/Intermediate/Advanced
Duration: XX minutes
Learning Objectives: ...
"""

# Exercise tasks with TODO comments
# Solution verification
# Hints if stuck
```

## Getting Help

If you get stuck:

1. Check the hints provided in each exercise
2. Review the relevant training documentation
3. Run with `--show-solutions` to see the answer
4. Ask in the training Slack channel

## Certification

Complete all exercises for your role to be eligible for certification:

- **Operators**: Exercises 01, 03, 04, 07
- **Developers**: Exercises 01, 02, 05, 06
- **Administrators**: Exercises 01, 07 (plus admin-specific scenarios)
