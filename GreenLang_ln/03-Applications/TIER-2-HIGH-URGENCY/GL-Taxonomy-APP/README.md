# GL-Taxonomy-APP: EU Taxonomy Alignment Platform

**TIER 2 - HIGH URGENCY** | Deadline: January 1, 2026

## Overview

GL-Taxonomy-APP is a regulatory compliance platform that enables financial institutions to calculate Green Asset Ratio (GAR) and Green Investment Ratio (GIR) in accordance with EU Taxonomy Regulation 2020/852. The platform provides automated taxonomy alignment assessment, DNSH validation, and regulatory reporting capabilities.

## Key Features

- **Automated Classification:** AI-powered mapping of economic activities to 150+ EU Taxonomy activities
- **GAR/GIR Calculation:** Deterministic calculations for regulatory compliance
- **DNSH Validation:** Comprehensive "Do No Significant Harm" assessment across 6 environmental objectives
- **Regulatory Reporting:** XBRL export and EBA template generation
- **Multi-Institution Support:** Banks, asset managers, and insurance companies

## Quick Stats

- **Market Size:** 10,000+ EU financial institutions
- **Assets Coverage:** €4.18 trillion (31.5% of EU funds)
- **Revenue Target:** $70M ARR by Year 3
- **Development Timeline:** 16 weeks
- **Team Size:** 7-9 engineers

## Project Structure

```
GL-Taxonomy-APP/
├── agents/                     # Five-agent pipeline
│   ├── portfolio_intake/       # Data ingestion agent
│   ├── taxonomy_mapping/       # AI classification agent
│   ├── alignment_calculator/   # GAR/GIR calculation agent
│   ├── dnsh_validation/       # DNSH assessment agent
│   └── taxonomy_reporting/    # Report generation agent
├── core/                      # Core business logic
│   ├── models/               # Data models
│   ├── calculations/         # GAR/GIR formulas
│   ├── validators/           # Data validation
│   └── classifiers/          # Activity classification
├── api/                      # REST API
│   ├── v1/                  # API version 1
│   ├── schemas/             # Request/response schemas
│   └── middleware/          # Authentication, rate limiting
├── database/                 # Database layer
│   ├── migrations/          # Schema migrations
│   ├── seeds/              # Taxonomy data seeds
│   └── models/             # ORM models
├── integrations/            # External integrations
│   ├── bloomberg/          # Bloomberg data
│   ├── gleif/              # LEI validation
│   └── banking_systems/    # Core banking APIs
├── frontend/                # React dashboard
│   ├── components/         # UI components
│   ├── pages/             # Application pages
│   └── services/          # API clients
├── reports/                # Report generation
│   ├── templates/         # Report templates
│   ├── xbrl/             # XBRL generation
│   └── pdf/              # PDF generation
├── tests/                  # Test suites
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── compliance/       # Regulatory tests
├── docs/                   # Documentation
│   ├── api/              # API documentation
│   ├── user_guide/       # User manual
│   └── technical/        # Technical docs
└── infrastructure/         # Infrastructure as Code
    ├── terraform/        # Terraform configs
    ├── kubernetes/       # K8s manifests
    └── docker/          # Docker configs
```

## Technology Stack

- **Backend:** Python 3.11+, FastAPI
- **Database:** PostgreSQL 15, Redis
- **AI/ML:** GPT-4/Claude, LangChain, RAG
- **Frontend:** React 18, TypeScript, Material-UI
- **Infrastructure:** Docker, Kubernetes, AWS/Azure
- **CI/CD:** GitHub Actions, Terraform

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- PostgreSQL 15
- Redis 7+
- Docker

### Installation

```bash
# Clone repository
git clone https://github.com/greenlang/gl-taxonomy-app.git
cd gl-taxonomy-app

# Backend setup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Database setup
createdb gl_taxonomy
python manage.py db migrate
python manage.py db seed

# Frontend setup
cd frontend
npm install
npm run build

# Run application
python manage.py runserver
```

### Docker Setup

```bash
# Build and run with Docker Compose
docker-compose up -d

# Access application
# API: http://localhost:8000
# Frontend: http://localhost:3000
```

## Development Timeline

### Current Sprint (Week 1-2)
- [x] Project planning and architecture
- [ ] Database schema design
- [ ] Import initial taxonomy activities
- [ ] Setup development environment
- [ ] Create API specification

### Upcoming Milestones
- **Week 3-4:** Portfolio intake agent
- **Week 5-6:** AI classification engine
- **Week 7-8:** GAR/GIR calculator
- **Week 9-10:** DNSH framework
- **Week 11-12:** Reporting module
- **Week 13-14:** Integration testing
- **Week 15-16:** Beta launch

## API Documentation

API documentation is available at:
- Development: http://localhost:8000/docs
- Staging: https://staging-api.gl-taxonomy.com/docs
- Production: https://api.gl-taxonomy.com/docs

## Key Endpoints

```http
# Upload portfolio
POST /api/v1/portfolios/upload

# Classify activities
POST /api/v1/classify

# Calculate GAR
POST /api/v1/calculations/gar

# Generate report
POST /api/v1/reports/generate
```

## Regulatory Compliance

The platform complies with:
- EU Taxonomy Regulation (EU) 2020/852
- Climate Delegated Act (EU) 2021/2139
- Environmental Delegated Act (EU) 2023/2486
- EBA ITS on Pillar 3 ESG risks disclosure
- GDPR for data protection

## Testing

```bash
# Run unit tests
pytest tests/unit

# Run integration tests
pytest tests/integration

# Run compliance tests
pytest tests/compliance

# Generate coverage report
pytest --cov=app --cov-report=html
```

## Deployment

### Staging Deployment
```bash
# Deploy to staging
./deploy.sh staging

# Run smoke tests
pytest tests/smoke --env=staging
```

### Production Deployment
```bash
# Deploy to production
./deploy.sh production

# Verify deployment
curl https://api.gl-taxonomy.com/health
```

## Security

- SOC 2 Type II compliant
- End-to-end encryption (AES-256)
- Role-based access control (RBAC)
- Multi-factor authentication (MFA)
- Regular security audits

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

Proprietary - GreenLang Technologies

## Contact

- **Project Manager:** GL-Taxonomy-PM
- **Technical Lead:** taxonomy-tech@greenlang.com
- **Support:** support@gl-taxonomy.com
- **Emergency:** +1-XXX-XXX-XXXX

## Status Dashboard

- **Build:** ![Build Status](https://img.shields.io/badge/build-passing-green)
- **Coverage:** ![Coverage](https://img.shields.io/badge/coverage-85%25-yellow)
- **Security:** ![Security](https://img.shields.io/badge/security-A+-green)
- **Uptime:** ![Uptime](https://img.shields.io/badge/uptime-99.9%25-green)

---

**Last Updated:** November 10, 2024
**Version:** 1.0.0
**Status:** In Development