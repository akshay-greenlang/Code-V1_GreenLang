# GreenLang Documentation Writing Style Guide

## Core Principles

### 1. Clarity First
Write for understanding, not to impress. Every sentence should serve the reader's goal of learning and implementing GreenLang successfully.

### 2. Progressive Disclosure
Start simple, add complexity gradually. Don't overwhelm beginners, but provide depth for experts.

### 3. Action-Oriented
Use imperative mood for instructions. Tell readers what to do, not what could be done.

### 4. Inclusive Language
Write for a global, diverse audience. Avoid idioms, cultural references, and assumptions about reader background.

## Voice and Tone

### Voice Characteristics
- **Professional** but not formal
- **Friendly** but not casual
- **Confident** but not arrogant
- **Helpful** but not patronizing
- **Technical** but not jargon-heavy

### Tone Guidelines

#### For Beginners
- Encouraging and supportive
- Patient with explanations
- Celebratory of small wins
- Clear about next steps

**Example:**
✅ "Congratulations! You've just created your first GreenLang agent. Let's explore what you can do next."
❌ "Now that you've completed the trivial task of agent creation, we can move on to more complex topics."

#### For Advanced Users
- Respectful of expertise
- Direct and efficient
- Technical when appropriate
- Focused on optimization

**Example:**
✅ "For production deployments, configure the memory cache with Redis to handle 10,000+ concurrent requests."
❌ "You might want to think about maybe using Redis if you have lots of users."

## Writing Standards

### Sentence Structure

#### Use Short Sentences
- Average sentence length: 15-20 words
- Maximum sentence length: 30 words
- One idea per sentence

**Example:**
✅ "GreenLang uses agents to process data. Each agent specializes in a specific task. Agents can work together in chains."
❌ "GreenLang uses agents to process data, with each agent specializing in a specific task, and these agents can be combined together in chains to create complex workflows that handle sophisticated business logic."

### Paragraph Structure

#### Keep Paragraphs Focused
- 3-5 sentences per paragraph
- One topic per paragraph
- Use transition sentences

**Example:**
✅ "Data validation is the first step in any GreenLang workflow. The validation agent checks incoming data against predefined schemas. It identifies and reports any inconsistencies or errors.

Once validation is complete, the data moves to the transformation stage. Here, specialized agents convert the data into the required format for processing."

### Active Voice

Always prefer active voice over passive voice.

**Examples:**
✅ "The agent processes the data"
❌ "The data is processed by the agent"

✅ "Configure the database connection"
❌ "The database connection should be configured"

✅ "GreenLang validates your input"
❌ "Your input is validated by GreenLang"

### Present Tense

Use present tense for current functionality.

**Examples:**
✅ "GreenLang supports multiple data formats"
❌ "GreenLang will support multiple data formats"

✅ "The API returns a JSON response"
❌ "The API returned a JSON response"

## Formatting Standards

### Headings

#### Hierarchy
- **H1:** Page title only (one per page)
- **H2:** Major sections
- **H3:** Subsections
- **H4:** Detailed points (sparingly)

#### Capitalization
Use sentence case for all headings.

**Examples:**
✅ "Getting started with GreenLang"
❌ "Getting Started With GreenLang"

✅ "Configure your first agent"
❌ "Configure Your First Agent"

### Lists

#### Bulleted Lists
Use for unordered information:
- Items of equal importance
- Options to choose from
- Features or benefits

#### Numbered Lists
Use for sequential steps:
1. Install GreenLang
2. Configure the environment
3. Run your first agent

#### Checklist Format
Use for validation or requirements:
- [ ] Python 3.8 or higher installed
- [ ] API credentials configured
- [ ] Database connection established

### Code Examples

#### Every Concept Needs Code
Never introduce a concept without a code example.

#### Complete Examples
Provide runnable code whenever possible.

**Good Example:**
```python
# Complete, runnable example
from greenlang import Agent, Chain

# Create an agent for carbon calculation
carbon_agent = Agent(
    name="carbon_calculator",
    model="gpt-4",
    tools=["calculator", "database"]
)

# Create a chain with the agent
chain = Chain(agents=[carbon_agent])

# Run the chain with input data
result = chain.run({
    "activity": "shipping",
    "distance": 1000,
    "weight": 500
})

print(f"Carbon emissions: {result['emissions']} kg CO2e")
```

**Bad Example:**
```python
# Incomplete example
agent = Agent(...)  # Configure as needed
result = chain.run(data)  # Pass your data
```

#### Language Variety
Provide examples in multiple languages when relevant:
- Python (primary)
- TypeScript/JavaScript
- cURL for API calls
- CLI commands

### Tables

Use tables for structured comparison or reference data.

**Example:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | string | Yes | Agent identifier |
| `model` | string | Yes | LLM model to use |
| `tools` | array | No | Available tools |
| `memory` | object | No | Memory configuration |

### Callouts and Notices

#### Information Hierarchy

**Note:** General information
> **Note:** This feature requires GreenLang version 2.0 or higher.

**Important:** Critical information
> **Important:** Always validate user input before processing.

**Warning:** Potential issues
> **Warning:** This operation will delete all data. This action cannot be undone.

**Tip:** Helpful suggestions
> **Tip:** Use batch processing for datasets larger than 10,000 records.

**Example:** Practical demonstration
> **Example:** Here's how to implement retry logic in your workflow.

## Content Standards

### API Documentation

#### Endpoint Format
```markdown
### Create Carbon Report

Generate a carbon emissions report for the specified data.

**Endpoint:** `POST /api/v1/reports/carbon`

**Authentication:** Required (Bearer token)

**Request Body:**
```json
{
  "data_source": "string",
  "date_range": {
    "start": "2025-01-01",
    "end": "2025-12-31"
  },
  "format": "PDF"
}
```

**Response:** `200 OK`
```json
{
  "report_id": "rpt_123abc",
  "status": "processing",
  "estimated_completion": "2025-01-15T10:30:00Z"
}
```

**Error Responses:**
- `400 Bad Request` - Invalid input data
- `401 Unauthorized` - Missing or invalid token
- `429 Too Many Requests` - Rate limit exceeded
```

### Tutorial Format

```markdown
# Building Your First Carbon Calculator

In this tutorial, you'll learn how to create a carbon emissions calculator using GreenLang. By the end, you'll have a working application that can calculate emissions for various activities.

## What you'll learn

- How to create and configure agents
- How to connect to data sources
- How to generate reports
- How to deploy your application

## Prerequisites

Before starting, ensure you have:
- [ ] GreenLang installed (version 2.0+)
- [ ] Python 3.8 or higher
- [ ] Basic Python knowledge
- [ ] API credentials (get them at https://greenlang.io/api)

## Time required

Approximately 30 minutes

## Step 1: Set up your environment

First, create a new project directory and install dependencies:

```bash
mkdir carbon-calculator
cd carbon-calculator
pip install greenlang
```

[Continue with detailed steps...]
```

### How-To Format

```markdown
# How to optimize agent performance

This guide explains how to improve the performance of your GreenLang agents for production workloads.

## Problem

Your agents are processing data slowly, causing delays in report generation.

## Solution

Implement these optimization strategies:

### 1. Enable caching

Cache frequently accessed data to reduce API calls:

```python
from greenlang import Agent, Cache

agent = Agent(
    name="optimized_agent",
    cache=Cache(
        backend="redis",
        ttl=3600  # 1 hour
    )
)
```

### 2. Use batch processing

[Continue with solutions...]

## Results

After implementing these optimizations, you should see:
- 50% reduction in processing time
- 75% fewer API calls
- 90% cache hit rate
```

## Terminology and Language

### Technical Terms

#### First Use
Define technical terms on first use.

**Example:**
"GreenLang uses **agents** (autonomous processing units that execute specific tasks) to handle data transformation."

#### Glossary
Maintain a comprehensive glossary for reference.

### Acronyms

#### First Use
Spell out acronyms on first use per page.

**Example:**
"The Environmental, Social, and Governance (ESG) report includes..."
(Subsequent uses: "ESG")

### Common Terms

#### Standardized Terminology
Use consistent terminology throughout:

| Use | Don't Use |
|-----|-----------|
| agent | processor, worker, handler |
| chain | pipeline, workflow, sequence |
| memory | state, cache, storage |
| tool | function, utility, helper |
| report | document, output, result |

## Accessibility Guidelines

### Alt Text for Images
Every image must have descriptive alt text.

**Example:**
```markdown
![Screenshot of GreenLang dashboard showing carbon emissions graph with monthly trends from January to December 2025](./images/dashboard.png)
```

### Link Text
Use descriptive link text, never "click here."

**Examples:**
✅ "Learn more about [agent configuration](./agents.md)"
❌ "To learn more, [click here](./agents.md)"

### Keyboard Navigation
Ensure all interactive elements are keyboard accessible.

### Screen Reader Compatibility
Test documentation with screen readers.

## Internationalization

### Simple English
Write at an 8th-grade reading level for easier translation.

### Avoid Idioms
Don't use cultural idioms or colloquialisms.

**Examples:**
✅ "This process is very fast"
❌ "This process is lightning fast"

✅ "Handle many requests simultaneously"
❌ "Handle requests like there's no tomorrow"

### Date and Time Formats
Use ISO 8601 format: `2025-12-31T23:59:59Z`

### Numbers and Units
- Use spaces for thousands: 10 000
- Include both metric and imperial when relevant
- Specify currency: USD, EUR, etc.

## Code Style

### Python Code Standards
Follow PEP 8 with these additions:
- Use type hints for all functions
- Include docstrings for all classes and functions
- Use meaningful variable names

**Example:**
```python
from typing import Dict, List, Optional

def calculate_emissions(
    activity: str,
    quantity: float,
    unit: str = "kg"
) -> Dict[str, float]:
    """
    Calculate carbon emissions for an activity.

    Args:
        activity: Type of activity (e.g., "transport", "energy")
        quantity: Amount of activity
        unit: Unit of measurement (default: "kg")

    Returns:
        Dictionary containing emissions data

    Example:
        >>> calculate_emissions("transport", 100, "km")
        {"co2": 23.5, "unit": "kg CO2e"}
    """
    # Implementation here
    pass
```

### JavaScript/TypeScript Standards
Follow Airbnb style guide with TypeScript strict mode.

```typescript
interface EmissionsData {
  co2: number;
  unit: string;
  timestamp: Date;
}

async function calculateEmissions(
  activity: string,
  quantity: number,
  unit: string = 'kg'
): Promise<EmissionsData> {
  // Implementation
  return {
    co2: 23.5,
    unit: 'kg CO2e',
    timestamp: new Date()
  };
}
```

## Review Checklist

Before publishing any documentation, verify:

### Content Quality
- [ ] Technically accurate
- [ ] Complete information
- [ ] Working code examples
- [ ] Clear explanations
- [ ] Proper grammar and spelling

### Structure
- [ ] Logical flow
- [ ] Consistent formatting
- [ ] Proper heading hierarchy
- [ ] Effective use of lists
- [ ] Appropriate callouts

### Style
- [ ] Active voice
- [ ] Present tense
- [ ] Short sentences
- [ ] Clear paragraphs
- [ ] Consistent terminology

### Accessibility
- [ ] Alt text for images
- [ ] Descriptive links
- [ ] Keyboard navigation
- [ ] Screen reader compatible
- [ ] Color contrast compliance

### Metadata
- [ ] Page title
- [ ] Meta description
- [ ] Keywords
- [ ] Last updated date
- [ ] Version information

## Version Documentation

### Version Notices
Clearly indicate version requirements.

**Example:**
```markdown
> **Version:** This feature is available in GreenLang 2.0 and later.

> **Deprecated:** This method is deprecated as of version 2.5. Use `new_method()` instead.

> **Breaking Change:** Version 3.0 changes the response format. See [migration guide](./migration.md).
```

## Common Mistakes to Avoid

### Writing Mistakes
❌ Using future tense for existing features
❌ Passive voice in instructions
❌ Long, complex sentences
❌ Unexplained jargon
❌ Missing code examples
❌ Incomplete information
❌ Outdated screenshots
❌ Broken links
❌ Inconsistent terminology
❌ Cultural idioms

### Formatting Mistakes
❌ Inconsistent heading levels
❌ Missing alt text
❌ "Click here" links
❌ Walls of text
❌ Unformatted code
❌ Missing syntax highlighting
❌ Inconsistent punctuation
❌ Mixed date formats
❌ Unclear callout usage
❌ Poor table formatting

## Style Guide Updates

This style guide is a living document. Updates are:
- Reviewed quarterly
- Approved by documentation team lead
- Communicated to all writers
- Applied to new content immediately
- Retrofitted to existing content gradually

## Questions and Feedback

For style guide questions:
- Check the FAQ section
- Ask in #docs-style Slack channel
- Email docs-team@greenlang.io
- Submit suggestions via GitHub