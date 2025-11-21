# GreenLang QA Engineer Onboarding Guide

## Welcome to GreenLang Quality Engineering!

This guide will help you get started as a QA Engineer at GreenLang. By the end of your onboarding, you'll be ready to contribute to our mission of building reliable, compliant carbon accounting software.

## Week 1: Foundation

### Day 1: Orientation

#### Morning Session
- [ ] Company overview and mission
- [ ] Meet your team and buddy
- [ ] HR paperwork and benefits
- [ ] Equipment setup

#### Afternoon Session
- [ ] GreenLang product demo
- [ ] Overview of carbon accounting
- [ ] Introduction to environmental regulations
- [ ] Team structure and roles

### Day 2: Development Environment Setup

#### Required Software
```bash
# Install Python and pip
python --version  # Should be 3.11+
pip --version

# Install development tools
pip install pytest pytest-cov pytest-mock
pip install black isort flake8 mypy
pip install requests aiohttp
pip install pandas numpy

# Install Node.js for k6 and other tools
node --version  # Should be 18+
npm --version

# Install k6 for performance testing
# macOS
brew install k6

# Windows (using Chocolatey)
choco install k6

# Linux
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
echo "deb https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
sudo apt-get update
sudo apt-get install k6
```

#### Repository Access
```bash
# Clone main repository
git clone https://github.com/greenlang/greenlang-core.git
cd greenlang-core

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-test.txt

# Run initial tests to verify setup
pytest tests/unit/ -v
```

#### Tool Access Setup
- [ ] Jira account and permissions
- [ ] TestRail access
- [ ] GitHub/GitLab account
- [ ] Slack workspace
- [ ] DataDog/monitoring access
- [ ] AWS/Cloud console access

### Day 3: Domain Knowledge - Carbon Accounting Basics

#### Key Concepts

**Greenhouse Gas (GHG) Emissions**:
- **Scope 1**: Direct emissions from owned sources
- **Scope 2**: Indirect emissions from purchased energy
- **Scope 3**: All other indirect emissions in value chain

**Emission Factors**:
- Conversion factors for activity data to CO2 equivalents
- Example: 1 liter of diesel = 2.68 kg CO2e

**Key Regulations**:
- **EU CBAM**: Carbon Border Adjustment Mechanism
- **GHG Protocol**: Corporate accounting standards
- **ISO 14064**: GHG quantification and reporting

#### Practice Exercise
```python
# Calculate simple emissions
def calculate_emissions(fuel_type, quantity):
    """Calculate CO2 emissions from fuel consumption."""
    emission_factors = {
        "diesel": 2.68,      # kg CO2e per liter
        "gasoline": 2.35,    # kg CO2e per liter
        "natural_gas": 1.93  # kg CO2e per m3
    }
    return quantity * emission_factors.get(fuel_type, 0)

# Write your first test
def test_diesel_emissions():
    result = calculate_emissions("diesel", 100)
    assert result == 268.0  # 100L * 2.68 kg/L
```

### Day 4: Testing Framework Overview

#### Test Structure
```python
# tests/unit/test_emission_calculator.py
import pytest
from greenlang.agents import EmissionCalculatorAgent

class TestEmissionCalculator:
    """Test suite for emission calculations."""

    @pytest.fixture
    def agent(self):
        """Create agent instance."""
        return EmissionCalculatorAgent()

    def test_calculation_accuracy(self, agent):
        """Test calculation accuracy."""
        result = agent.calculate(
            fuel_type="diesel",
            quantity=1000
        )
        assert result.emissions == 2680.0

    @pytest.mark.parametrize("fuel,qty,expected", [
        ("diesel", 100, 268.0),
        ("gasoline", 100, 235.0),
    ])
    def test_multiple_fuels(self, agent, fuel, qty, expected):
        """Test different fuel types."""
        result = agent.calculate(fuel, qty)
        assert result.emissions == expected
```

#### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=greenlang --cov-report=html

# Run specific test file
pytest tests/unit/test_emission_calculator.py

# Run with markers
pytest -m "not slow"

# Run in parallel
pytest -n 4
```

### Day 5: First Contributions

#### Your First Test Case
1. Find an untested function in the codebase
2. Write unit tests achieving 85% coverage
3. Submit PR for review
4. Address feedback and merge

#### Code Review Checklist
- [ ] Tests are clear and readable
- [ ] Test names describe what they test
- [ ] Edge cases are covered
- [ ] Mocks are used appropriately
- [ ] No hardcoded values
- [ ] Tests run in isolation

## Week 2: Deep Dive

### Day 6-7: Integration Testing

#### Database Testing
```python
# tests/integration/test_database.py
import pytest
from sqlalchemy import create_engine
from greenlang.models import EmissionRecord

@pytest.fixture
def db_session():
    """Create test database session."""
    engine = create_engine("sqlite:///:memory:")
    # Setup tables
    Base.metadata.create_all(engine)
    session = Session(engine)
    yield session
    session.close()

def test_emission_record_persistence(db_session):
    """Test saving emission records."""
    record = EmissionRecord(
        fuel_type="diesel",
        quantity=100,
        emissions=268.0
    )
    db_session.add(record)
    db_session.commit()

    retrieved = db_session.query(EmissionRecord).first()
    assert retrieved.emissions == 268.0
```

#### API Testing
```python
# tests/integration/test_api.py
import pytest
import requests

@pytest.fixture
def api_client():
    """Create API test client."""
    return requests.Session()

def test_emission_calculation_api(api_client):
    """Test emission calculation endpoint."""
    response = api_client.post(
        "http://localhost:8000/api/v1/calculate",
        json={
            "fuel_type": "diesel",
            "quantity": 100
        }
    )
    assert response.status_code == 200
    assert response.json()["emissions"] == 268.0
```

### Day 8-9: Performance Testing

#### k6 Load Test Script
```javascript
// tests/performance/emission_api.js
import http from 'k6/http';
import { check } from 'k6';

export let options = {
  stages: [
    { duration: '2m', target: 50 },   // Ramp up
    { duration: '5m', target: 50 },   // Stay at 50 users
    { duration: '2m', target: 0 },    // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'], // 95% under 500ms
    http_req_failed: ['rate<0.05'],   // Error rate under 5%
  },
};

export default function () {
  const payload = JSON.stringify({
    fuel_type: 'diesel',
    quantity: Math.random() * 1000,
  });

  const params = {
    headers: { 'Content-Type': 'application/json' },
  };

  const res = http.post('http://localhost:8000/api/v1/calculate', payload, params);

  check(res, {
    'status is 200': (r) => r.status === 200,
    'response time < 500ms': (r) => r.timings.duration < 500,
  });
}
```

### Day 10: Security Testing

#### Basic Security Tests
```python
# tests/security/test_input_validation.py
import pytest

def test_sql_injection_prevention():
    """Test SQL injection protection."""
    malicious_input = "'; DROP TABLE users; --"

    with pytest.raises(ValidationError):
        process_user_input(malicious_input)

def test_xss_prevention():
    """Test XSS attack prevention."""
    xss_payload = "<script>alert('XSS')</script>"

    result = sanitize_input(xss_payload)
    assert "<script>" not in result

def test_authentication_required():
    """Test endpoints require authentication."""
    response = requests.get("http://localhost:8000/api/v1/protected")
    assert response.status_code == 401
```

## Week 3: Advanced Topics

### Chaos Engineering

#### Failure Injection Example
```python
# tests/chaos/test_resilience.py
import pytest
import random
from chaos import inject_failure

def test_database_failure_handling():
    """Test system handles database failures."""
    with inject_failure("database", probability=1.0):
        result = process_emission_data(test_data)
        assert result.status == "QUEUED"  # Should queue for retry

def test_network_latency():
    """Test system handles network delays."""
    with inject_latency(delay_ms=5000):
        result = fetch_emission_factors()
        assert result is not None  # Should use cached data
```

### Test Data Management

#### Test Data Generator
```python
# tests/fixtures/data_generator.py
from faker import Faker
import random

class TestDataGenerator:
    def __init__(self):
        self.faker = Faker()
        Faker.seed(42)  # Deterministic data

    def generate_shipment(self):
        """Generate CBAM shipment data."""
        return {
            "id": self.faker.uuid4(),
            "product": random.choice(["cement", "steel", "aluminum"]),
            "quantity": round(random.uniform(1, 100), 2),
            "origin": random.choice(["CN", "IN", "RU"]),
            "date": self.faker.date_between("-1y", "today")
        }

    def generate_batch(self, size=100):
        """Generate batch of test data."""
        return [self.generate_shipment() for _ in range(size)]
```

### Compliance Testing

#### CBAM Compliance Test
```python
# tests/compliance/test_cbam.py
def test_cbam_required_fields():
    """Test CBAM report contains all required fields."""
    report = generate_cbam_report(test_data)

    required_fields = [
        "cn_code",
        "quantity",
        "emissions",
        "country_of_origin",
        "installation_id"
    ]

    for field in required_fields:
        assert field in report
        assert report[field] is not None

def test_calculation_methodology():
    """Test calculations follow EU methodology."""
    result = calculate_embedded_emissions(
        product="cement",
        quantity=1000,
        emission_factor=0.83
    )

    expected = 1000 * 0.83  # Simple method
    assert abs(result - expected) < 0.01  # Tolerance for rounding
```

## Week 4: Integration and Practice

### Real Project Work

#### Project Assignment
You'll be assigned to a real feature team to:
1. Review existing test coverage
2. Identify testing gaps
3. Write new test cases
4. Participate in code reviews
5. Contribute to test automation

### Best Practices

#### Test Writing Guidelines

**DO**:
- Write descriptive test names
- Use fixtures for setup
- Test one thing per test
- Include positive and negative cases
- Clean up after tests

**DON'T**:
- Use production data
- Hardcode credentials
- Write brittle tests
- Ignore flaky tests
- Skip documentation

#### Test Naming Convention
```python
# Good test names
def test_calculate_emissions_returns_correct_value_for_diesel():
    pass

def test_invalid_fuel_type_raises_validation_error():
    pass

def test_large_quantity_does_not_cause_overflow():
    pass

# Bad test names
def test1():
    pass

def test_emissions():
    pass

def test_works():
    pass
```

### Continuous Learning Resources

#### Documentation
- [GreenLang Testing Wiki](internal-link)
- [Python Testing Best Practices](https://docs.pytest.org/en/latest/goodpractices.html)
- [k6 Performance Testing Guide](https://k6.io/docs/)

#### Courses
- **Internal**: GreenLang Domain Training (LMS)
- **External**: ISTQB Certification
- **Udemy**: Advanced Python Testing
- **Coursera**: Performance Testing Fundamentals

#### Tools to Master
1. **pytest**: Main testing framework
2. **k6**: Performance testing
3. **Postman**: API testing
4. **SonarQube**: Code quality
5. **Docker**: Test environments

### Mentorship Program

#### Your Buddy
- Daily check-ins for first week
- Weekly 1:1s for first month
- Code review guidance
- Domain knowledge transfer

#### Your Manager
- Weekly 1:1s
- Goal setting
- Career development
- Performance feedback

### Success Metrics (First 30 Days)

#### Week 1
- [ ] Development environment setup
- [ ] Run existing test suite
- [ ] Understand codebase structure

#### Week 2
- [ ] Write first unit tests
- [ ] Submit first PR
- [ ] Participate in code review

#### Week 3
- [ ] Write integration tests
- [ ] Create performance test
- [ ] Identify testing gaps

#### Week 4
- [ ] Own a test module
- [ ] Contribute to test strategy
- [ ] Present learning to team

## FAQ

**Q: How do I run tests locally?**
A: Use `pytest` for unit tests, `docker-compose up` for integration tests.

**Q: What's our coverage target?**
A: 85% for unit tests, 70% for integration tests.

**Q: How do I report bugs?**
A: Create Jira ticket with reproduction steps, expected vs. actual behavior.

**Q: Where do I find emission factors?**
A: Check the `data/emission_factors.json` file or EPA database.

**Q: Who reviews my code?**
A: Your buddy for first month, then rotating team members.

## Useful Commands Cheatsheet

```bash
# Testing
pytest                           # Run all tests
pytest -k test_emissions        # Run specific tests
pytest --lf                     # Run last failed
pytest --pdb                    # Debug on failure

# Coverage
pytest --cov=greenlang          # Generate coverage
pytest --cov-report=html        # HTML report

# Performance
k6 run test.js                  # Run load test
k6 run --vus 100 --duration 30s test.js  # Custom load

# Git
git checkout -b feature/test-xyz  # Create branch
git add -A                        # Stage changes
git commit -m "Add test for XYZ"  # Commit
git push origin feature/test-xyz  # Push branch

# Docker
docker-compose up -d             # Start services
docker-compose logs -f           # View logs
docker-compose down              # Stop services

# Database
psql -U postgres -d greenlang    # Connect to DB
\dt                              # List tables
\q                               # Quit
```

## Getting Help

- **Slack Channels**:
  - #qa-team - General QA discussions
  - #testing-help - Get help with testing
  - #ci-cd - CI/CD pipeline issues

- **Documentation**:
  - [Internal Wiki](https://wiki.greenlang.io)
  - [API Documentation](https://api.greenlang.io/docs)
  - [Testing Playbook](https://playbook.greenlang.io)

- **Key Contacts**:
  - QA Lead: qa-lead@greenlang.io
  - DevOps: devops@greenlang.io
  - Security: security@greenlang.io

Welcome to the team! We're excited to have you help us build quality into every aspect of GreenLang. Remember, quality is everyone's responsibility, and as a QA Engineer, you're the champion of that mission.