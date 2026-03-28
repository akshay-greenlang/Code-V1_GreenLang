# GreenLang V2 Ownership Matrix

| Domain | Owner Group | Owner Name | Backup Owner | Responsibilities | Approval Authority | Escalation |
| --- | --- | --- | --- | --- | --- | --- |
| Runtime and CLI | Platform Runtime Team | Akshay Kulkarni | Neha Sharma | command semantics, exit code policy, contract compatibility | Architecture Board Chair | Architecture Board |
| Pack Lifecycle | Ecosystem Team | Priya Menon | Arjun Rao | tier promotion/demotion, signing policy, pack quality scorecards | Release Board Chair | Release Board |
| Agent Lifecycle | Agent Platform Team | Rohan Iyer | Sneha Das | registry metadata, version/deprecation enforcement | Architecture Board Chair | Architecture Board |
| Connector Reliability | App Reliability Team | Kavya Nair | Vikas Patel | retries/idempotency/timeouts/SLOs, incident response | SRE On-call Lead | SRE On-call Board |
| Security and Policy | Security Council | Ananya Gupta | Rahul Verma | policy bundles, security scans, exception approval | Security Council Lead | Security Council Lead |
| Determinism and Auditability | Compliance Engineering | Meera Subramanian | Tanvi Singh | deterministic replay contract, audit bundle schema | Release Board Chair | Release Board |
| Release Trains | Release Engineering | Karan Malhotra | Aditi Joshi | train cadence, RC soak, go/no-go evidence | Release Board | Release Board |
| Enterprise UX | Frontend Platform Team | Ishaan Kapoor | Nidhi Batra | shell UX standards, e2e and visual quality gates | Product Steering Chair | Product Steering |
| Docs and DX | Developer Experience Team | Devika Rao | Mohit Jain | handbook, runbooks, migration playbooks | Program PMO Lead | Program PMO |

## Coverage Rule

No V2-critical capability may be unowned. Ownership changes require release-board approval.

## Review Cadence

- weekly architecture and release review (minimum once per week)
- monthly ownership verification for active V2 capabilities
