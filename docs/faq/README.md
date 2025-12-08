# GreenLang Frequently Asked Questions

## Overview

This FAQ document answers the most common questions about GreenLang Process Heat Agents. Questions are organized into categories for easy navigation.

**Last Updated:** December 2025
**Version:** 1.0.0

---

## Table of Contents

1. [General Questions](#general-questions)
2. [Technical Questions](#technical-questions)
3. [Safety and Compliance Questions](#safety-and-compliance-questions)
4. [Pricing and Licensing Questions](#pricing-and-licensing-questions)

---

## General Questions

### G1. What is GreenLang?

GreenLang is an AI-powered platform for process heat management in industrial settings. It uses machine learning to monitor, predict, and optimize thermal processes, helping organizations improve energy efficiency, ensure safety compliance, and reduce operational costs.

### G2. Who is GreenLang designed for?

GreenLang is designed for industrial operations that involve process heat, including:
- Steel and metals manufacturing
- Glass production
- Cement and ceramite manufacturing
- Chemical processing
- Food and beverage processing
- Heat treatment operations
- Power generation

### G3. What are the main benefits of using GreenLang?

The primary benefits include:
- **Energy Efficiency:** 15-30% reduction in energy consumption
- **Safety Compliance:** Automated monitoring for ISA 18.2, NFPA 86, and OSHA PSM
- **Predictive Maintenance:** 40% reduction in unplanned downtime
- **Operational Excellence:** 20% improvement in productivity

### G4. How does GreenLang differ from traditional SCADA/DCS systems?

While GreenLang complements SCADA and DCS systems, it adds intelligent capabilities:
- Machine learning predictions and anomaly detection
- Natural language explanations for AI decisions
- Automated compliance monitoring
- Optimization recommendations based on AI analysis

GreenLang integrates with existing control systems rather than replacing them.

### G5. What industries does GreenLang serve?

GreenLang serves any industry with process heat requirements:
- Primary metals (steel, aluminum, copper)
- Glass and ceramics
- Cement and building materials
- Chemicals and petrochemicals
- Food and beverage
- Pulp and paper
- Automotive manufacturing
- Aerospace

### G6. Is GreenLang suitable for small operations?

Yes. GreenLang scales from single-furnace installations to enterprise deployments with hundreds of agents. Our Quick Start deployment can be running in minutes.

### G7. What is the typical ROI for a GreenLang deployment?

Most customers see ROI within 6-12 months. Energy savings alone often justify the investment, with additional value from reduced downtime, improved quality, and compliance benefits.

### G8. How long does it take to deploy GreenLang?

- **Evaluation/POC:** 1-2 days
- **Single-site production:** 2-4 weeks
- **Enterprise deployment:** 2-3 months

### G9. What support does GreenLang provide?

Support options include:
- **Community:** Free community forum and documentation
- **Standard:** Email support, 24-hour response time
- **Premium:** Phone and email, 4-hour response time
- **Enterprise:** Dedicated support, 1-hour response, on-site available

### G10. Can GreenLang be used in air-gapped environments?

Yes. GreenLang supports fully air-gapped deployments with no internet connectivity required. Updates are delivered via secure offline packages.

### G11. What languages does GreenLang support?

The GreenLang interface is currently available in:
- English
- German
- French
- Spanish
- Portuguese
- Chinese (Simplified)
- Japanese
- Korean

Additional languages are added based on customer demand.

### G12. Does GreenLang offer training programs?

Yes. GreenLang offers:
- Self-paced online training
- Instructor-led virtual training
- On-site training (Enterprise customers)
- Certification programs for operators, administrators, and developers

### G13. What is GreenLang's uptime guarantee?

- **Standard:** 99.5% uptime
- **Premium:** 99.9% uptime
- **Enterprise:** 99.99% uptime with SLA guarantees

For on-premises deployments, uptime depends on customer infrastructure.

### G14. Can GreenLang integrate with our ERP system?

Yes. GreenLang provides APIs and pre-built connectors for:
- SAP
- Oracle
- Microsoft Dynamics
- Infor
- Custom ERP systems via REST API

### G15. What happens to our data?

For cloud deployments, data is stored in SOC 2 Type II certified data centers. For on-premises, data never leaves your facility. Data retention policies are configurable by the customer.

### G16. Is there a mobile app?

GreenLang's web dashboard is fully responsive and works on mobile devices. Native iOS and Android apps are on the roadmap for 2026.

### G17. Can we white-label GreenLang?

Enterprise customers can customize branding, including logos, colors, and domain names. Contact sales for white-label options.

### G18. What is the minimum contract term?

- **Standard:** Month-to-month
- **Premium:** Annual
- **Enterprise:** Multi-year with volume discounts

### G19. Does GreenLang offer a free trial?

Yes. We offer a 30-day free trial with full features. No credit card required. Contact sales@greenlang.io.

### G20. How do I get started?

1. Contact sales or sign up at greenlang.io
2. Complete the 5-Minute Quick Start
3. Schedule an onboarding session
4. Deploy your first agent

---

## Technical Questions

### T1. What protocols does GreenLang support for data collection?

GreenLang supports:
- **OPC-UA:** Primary industrial protocol
- **OPC-DA:** Legacy systems (via gateway)
- **Modbus TCP/RTU:** PLCs and devices
- **MQTT:** IoT and edge devices
- **HTTP/REST:** Custom integrations
- **Kafka:** Streaming platforms
- **Database connectors:** Direct SQL queries

### T2. What are the hardware requirements?

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16+ GB |
| Storage | 50 GB SSD | 200+ GB SSD |
| Network | 100 Mbps | 1 Gbps |

For production, requirements scale with the number of agents and data volume.

### T3. What operating systems are supported?

- Ubuntu 20.04 LTS, 22.04 LTS
- Red Hat Enterprise Linux 8, 9
- CentOS Stream 8, 9
- Windows Server 2019, 2022
- macOS 12+ (development only)

### T4. Can GreenLang run in containers?

Yes. GreenLang is designed for containerized deployment:
- Docker and Docker Compose for simple deployments
- Kubernetes for orchestrated deployments
- Helm charts provided for Kubernetes
- OpenShift supported

### T5. What cloud platforms are supported?

GreenLang can be deployed on:
- AWS (EC2, EKS)
- Microsoft Azure (VMs, AKS)
- Google Cloud Platform (GCE, GKE)
- Private cloud (VMware, OpenStack)
- Hybrid configurations

### T6. What database does GreenLang use?

- **Primary database:** PostgreSQL 14+
- **Time-series:** TimescaleDB
- **Caching:** Redis
- **Search (optional):** Elasticsearch

### T7. How much data does GreenLang store?

Data volume depends on configuration:
- Default retention: 365 days
- Compressed storage: ~10 MB per agent per day
- Aggregated data can extend retention indefinitely

### T8. How does GreenLang handle high availability?

High availability features include:
- Active-passive database replication
- Multi-replica API servers behind load balancer
- Redis Sentinel for cache HA
- Kubernetes pod anti-affinity
- Cross-zone deployment support

### T9. What is the API structure?

GreenLang provides:
- **REST API:** Full CRUD operations
- **GraphQL:** Flexible querying
- **WebSocket:** Real-time streaming
- **SSE:** Server-sent events for updates

OpenAPI (Swagger) documentation is available at /api/docs.

### T10. How do I authenticate to the API?

GreenLang uses JWT-based authentication:
1. Obtain token via /api/v1/auth/token
2. Include token in Authorization header
3. Tokens expire (configurable, default 1 hour)
4. Refresh tokens for extended sessions

OAuth2 and SAML integration available for enterprise.

### T11. Can I create custom agents?

Yes. GreenLang provides:
- Python SDK for custom agent development
- Agent templates and base classes
- Custom module development
- Full API access for automation

### T12. How does the ML model training work?

- Initial models are pre-trained on industry data
- Models adapt to your specific process via transfer learning
- Auto-retraining available (weekly default)
- Manual training trigger available
- Champion-challenger deployment prevents degradation

### T13. What ML models does GreenLang use?

- **Prediction:** LSTM, GRU, Transformer models
- **Anomaly Detection:** Isolation Forest, Autoencoders
- **Classification:** Gradient Boosting, Random Forest
- **Optimization:** Bayesian Optimization, Reinforcement Learning

### T14. Can I use my own ML models?

Yes. Enterprise customers can:
- Deploy custom TensorFlow/PyTorch models
- Use custom scikit-learn pipelines
- Integrate external ML services
- Build custom training pipelines

### T15. How is data encrypted?

- **At rest:** AES-256 encryption
- **In transit:** TLS 1.2/1.3
- **Database:** Transparent data encryption
- **Backups:** Encrypted with customer-managed keys

### T16. What authentication methods are supported?

- Local username/password
- LDAP/Active Directory
- SAML 2.0
- OAuth 2.0/OIDC
- Multi-factor authentication (TOTP, WebAuthn)

### T17. How are backups performed?

- **Database:** Daily full, hourly incremental
- **Configuration:** Daily snapshots
- **Time-series:** Continuous replication
- **ML models:** Versioned storage

Backups can be stored locally, on S3, or Azure Blob.

### T18. What monitoring is built in?

GreenLang exposes:
- Prometheus metrics endpoint
- Health check endpoints
- Structured JSON logging
- Distributed tracing (OpenTelemetry)
- Custom alerting via webhooks

### T19. How do I troubleshoot issues?

1. Check health endpoints: /health, /ready
2. Review logs: docker-compose logs or kubectl logs
3. Check metrics in Prometheus/Grafana
4. Use CLI diagnostic tools: greenlang-cli diagnose
5. Contact support with diagnostic bundle

### T20. What is the latency for real-time data?

- **Data collection to dashboard:** < 1 second typical
- **ML inference:** 50-100ms typical
- **Alert generation:** < 500ms from condition
- **API response:** < 200ms P95

### T21. Can GreenLang process historical data?

Yes. Historical data import is supported:
- CSV, JSON, Parquet formats
- Database migration tools
- API batch import
- Backfill for training models

### T22. How do webhooks work?

GreenLang can send webhooks for events:
- Alarm activation/acknowledgement
- Prediction threshold exceeded
- Anomaly detected
- Job completion

Webhooks include retry logic and dead-letter queue.

### T23. Is there rate limiting on the API?

- **Authenticated:** 1000 requests/minute
- **Unauthenticated:** 10 requests/minute
- **Burst:** Up to 100 requests/second
- **Enterprise:** Custom limits available

### T24. How do I migrate from another system?

Migration tools support:
- Configuration export/import
- Historical data migration
- User account migration
- Parallel operation during transition

Professional services available for complex migrations.

### T25. What SDKs are available?

Official SDKs:
- Python (pip install greenlang-sdk)
- JavaScript/TypeScript (npm install @greenlang/sdk)
- Go (go get github.com/greenlang/sdk-go)
- Java (Maven: io.greenlang:sdk)

### T26. Can I run GreenLang at the edge?

Yes. GreenLang Edge provides:
- Lightweight agent runtime
- Local data buffering
- Store-and-forward for connectivity gaps
- ML inference at the edge

### T27. How do I scale GreenLang?

- **Vertical:** Increase resources on existing nodes
- **Horizontal:** Add API/agent replicas
- **Database:** Read replicas, sharding for extreme scale
- **Auto-scaling:** Kubernetes HPA supported

### T28. What happens during network outages?

- Edge agents buffer data locally
- Automatic reconnection and sync
- No data loss with proper buffering
- Graceful degradation of cloud features

### T29. How are schema migrations handled?

- Automatic migration on upgrade
- Rollback capability
- Migration scripts for manual control
- Zero-downtime migrations for HA deployments

### T30. What logging formats are supported?

- JSON (default)
- Plain text
- Syslog (RFC 5424)
- Custom formats via configuration

Logs can be sent to file, stdout, syslog, or external systems.

---

## Safety and Compliance Questions

### S1. What safety standards does GreenLang support?

GreenLang provides monitoring and documentation for:
- **ISA 18.2:** Alarm management
- **NFPA 86:** Furnace and oven safety
- **OSHA PSM (29 CFR 1910.119):** Process safety management
- **IEC 61511:** Functional safety
- **ISO 45001:** Occupational health and safety

### S2. Is GreenLang certified for safety applications?

GreenLang is a monitoring and advisory system. It does not replace certified safety instrumented systems (SIS). Use GreenLang alongside, not instead of, certified safety equipment.

### S3. How does alarm management work?

GreenLang implements ISA 18.2 principles:
- Alarm rationalization database
- Priority assignment and justification
- Performance metrics (alarms/hour, etc.)
- Nuisance alarm detection
- Compliance reporting

### S4. Can GreenLang generate compliance reports?

Yes. Built-in reports include:
- ISA 18.2 alarm performance reports
- NFPA 86 equipment compliance status
- OSHA PSM element tracking
- Audit preparation packages
- Custom report builder

### S5. How long is audit data retained?

- Default retention: 7 years
- Configurable up to indefinite
- Tamper-evident logging
- Archived to compliant storage

### S6. Does GreenLang support safety interlocks?

GreenLang monitors interlock status but does not implement safety interlocks directly. Safety interlocks should remain in dedicated safety systems.

### S7. How does GreenLang handle alarm flooding?

- Alarm suppression during equipment startup
- Related alarm grouping
- Alarm shelving with auto-return
- Operator load management alerts

### S8. What is the audit trail capability?

All changes are logged:
- User identification
- Timestamp (UTC)
- Old and new values
- Action description
- Tamper-evident checksums

### S9. Can we customize compliance rules?

Yes. You can:
- Modify threshold values
- Add custom compliance checks
- Create custom reports
- Integrate company-specific requirements

### S10. How does training record management work?

GreenLang can track:
- Operator certifications
- Training completion dates
- Recertification requirements
- Training matrix compliance

Integration with external LMS systems is available.

### S11. Does GreenLang support Management of Change (MOC)?

Yes. MOC features include:
- Change request workflow
- Impact assessment forms
- Approval routing
- Pre-startup safety review
- Change documentation archive

### S12. How are incidents documented?

Incident investigation features:
- Incident report forms
- Timeline reconstruction
- Root cause analysis tools
- Corrective action tracking
- Lessons learned database

### S13. What about cybersecurity compliance?

GreenLang supports:
- IEC 62443 principles
- NIST Cybersecurity Framework
- SOC 2 Type II (cloud deployment)
- Role-based access control
- Audit logging

### S14. Can GreenLang detect safety-related anomalies?

Yes. ML models can detect:
- Unusual process conditions
- Equipment degradation
- Process deviations
- Precursor patterns to incidents

Alerts are clearly flagged as ML-generated, not safety-certified.

### S15. How do we prepare for regulatory audits?

1. Use the Audit Preparation module
2. Generate compliance reports
3. Review self-audit findings
4. Compile documentation packages
5. Practice data retrieval

GreenLang support can assist with audit preparation.

---

## Pricing and Licensing Questions

### P1. How is GreenLang priced?

GreenLang uses a subscription model based on:
- Number of agents
- Deployment type (cloud vs. on-premises)
- Support level
- Additional modules

Contact sales@greenlang.io for a custom quote.

### P2. What editions are available?

| Edition | Agents | Features | Support |
|---------|--------|----------|---------|
| Starter | Up to 5 | Core features | Community |
| Professional | Up to 50 | All features | Standard |
| Enterprise | Unlimited | All features + custom | Premium |

### P3. Is there a free tier?

- 30-day full-featured trial
- Community edition for non-production use
- Open-source components available

### P4. What does the subscription include?

All subscriptions include:
- Software access
- Updates and upgrades
- Documentation
- Community support (all tiers)
- Email support (Professional+)
- Phone support (Enterprise)

### P5. Are there additional costs?

Potential additional costs:
- Professional services (implementation, training)
- Custom development
- Premium support add-ons
- Third-party infrastructure (cloud hosting)

### P6. Can I pay annually?

Yes. Annual payment provides:
- 2 months free (17% discount)
- Price lock for term
- Simplified billing

### P7. Is there a perpetual license option?

Enterprise customers can negotiate perpetual licenses with separate maintenance agreements. Contact sales for details.

### P8. How do I upgrade my plan?

1. Contact sales or use the billing portal
2. Review new plan details
3. Confirm upgrade
4. Features activate immediately
5. Prorated billing applies

### P9. What is your refund policy?

- Monthly subscriptions: Cancel anytime, no refund for partial month
- Annual subscriptions: Pro-rated refund within first 30 days
- Enterprise: Per contract terms

### P10. Are educational or non-profit discounts available?

Yes. Special pricing for:
- Educational institutions
- Non-profit organizations
- Research facilities
- Startups (under 2 years, under $5M revenue)

Contact sales with verification documentation.

---

## Need More Help?

If your question isn't answered here:

- **Documentation:** https://docs.greenlang.io
- **Community Forum:** https://community.greenlang.io
- **Email Support:** support@greenlang.io
- **Enterprise Support:** enterprise-support@greenlang.io
- **Sales Inquiries:** sales@greenlang.io

---

*FAQ Version: 1.0.0*
*Total Questions: 75*
*Last Updated: December 2025*
