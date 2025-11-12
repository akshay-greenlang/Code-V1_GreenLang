# GreenLang Frontend Architecture Diagrams

Visual representations of the complete GreenLang frontend ecosystem.

## Table of Contents
1. [Platform Overview](#platform-overview)
2. [Component Hierarchy](#component-hierarchy)
3. [Data Flow Architecture](#data-flow-architecture)
4. [State Management](#state-management)
5. [Authentication Flow](#authentication-flow)
6. [Deployment Architecture](#deployment-architecture)
7. [Development Workflow](#development-workflow)

---

## Platform Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          GreenLang Ecosystem                                 │
│                    "The LangChain for Climate Intelligence"                  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          User-Facing Platforms                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐   │
│  │  Developer Portal  │  │  Visual Builder    │  │  GreenLang Hub     │   │
│  │  docs.greenlang.io │  │  (No-Code Tool)    │  │  (Agent Registry)  │   │
│  ├────────────────────┤  ├────────────────────┤  ├────────────────────┤   │
│  │ • 1,000+ docs      │  │ • Drag & drop      │  │ • 5,000+ agents    │   │
│  │ • Playground       │  │ • Real-time collab │  │ • One-click install│   │
│  │ • API reference    │  │ • Export to code   │  │ • Ratings/reviews  │   │
│  │ • Tutorials        │  │ • Testing tools    │  │ • Categories       │   │
│  └────────────────────┘  └────────────────────┘  └────────────────────┘   │
│                                                                               │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐   │
│  │  GreenLang Studio  │  │ Enterprise Dash    │  │  Marketplace       │   │
│  │  (Observability)   │  │ (Management)       │  │  (Commerce)        │   │
│  ├────────────────────┤  ├────────────────────┤  ├────────────────────┤   │
│  │ • Trace viewer     │  │ • Agent monitoring │  │ • Premium agents   │   │
│  │ • Performance      │  │ • Analytics        │  │ • Subscriptions    │   │
│  │ • Debug tools      │  │ • Cost tracking    │  │ • Licenses         │   │
│  │ • A/B testing      │  │ • Team mgmt        │  │ • Revenue reports  │   │
│  └────────────────────┘  └────────────────────┘  └────────────────────┘   │
│                                                                               │
│  ┌────────────────────┐                                                      │
│  │  IDE Extensions    │                                                      │
│  │  (Developer Tools) │                                                      │
│  ├────────────────────┤                                                      │
│  │ • VSCode           │                                                      │
│  │ • IntelliJ IDEA    │                                                      │
│  │ • PyCharm          │                                                      │
│  │ • WebStorm         │                                                      │
│  └────────────────────┘                                                      │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          Shared Component Layer                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  Design System                                                        │  │
│  │  • Color Palette  • Typography  • Spacing  • Icons  • Animations    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  UI Components (50+)                                                  │  │
│  │  • CodeBlock  • DataTable  • Charts  • Forms  • Navigation          │  │
│  │  • Search     • Modals     • Alerts  • Cards  • Badges              │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  Common Utilities                                                     │  │
│  │  • API Client  • Authentication  • Analytics  • Error Handling      │  │
│  │  • State Management  • Routing  • Validation  • Formatting          │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          Backend Services Layer                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐               │
│  │  Agent API     │  │  Trace API     │  │  User API      │               │
│  │  (CRUD agents) │  │  (Tracing)     │  │  (Auth/Profile)│               │
│  └────────────────┘  └────────────────┘  └────────────────┘               │
│                                                                               │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐               │
│  │  Analytics API │  │  Payment API   │  │  Workflow API  │               │
│  │  (Metrics)     │  │  (Stripe)      │  │  (Execution)   │               │
│  └────────────────┘  └────────────────┘  └────────────────┘               │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          Data Layer                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ PostgreSQL   │  │ ClickHouse   │  │ Redis        │  │ Elasticsearch│  │
│  │ (Primary DB) │  │ (Analytics)  │  │ (Cache)      │  │ (Search)     │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Hierarchy

```
App Component Tree Structure
└── Root
    ├── Providers
    │   ├── AuthProvider
    │   ├── ThemeProvider
    │   ├── QueryClientProvider
    │   └── AnalyticsProvider
    │
    ├── Layout
    │   ├── Header
    │   │   ├── Logo
    │   │   ├── Navigation
    │   │   ├── Search
    │   │   └── UserMenu
    │   │
    │   ├── Sidebar (conditional)
    │   │   ├── NavLinks
    │   │   └── QuickActions
    │   │
    │   ├── Main Content
    │   │   └── [Page Components]
    │   │
    │   └── Footer
    │       ├── Links
    │       └── Copyright
    │
    └── Global Components
        ├── NotificationCenter
        ├── CommandPalette
        ├── ErrorBoundary
        └── LoadingOverlay

Page Component Example (Agent Detail)
└── AgentDetailPage
    ├── AgentHeader
    │   ├── AgentIcon
    │   ├── AgentInfo
    │   │   ├── Name
    │   │   ├── Description
    │   │   └── Metadata
    │   └── ActionButtons
    │       ├── InstallButton
    │       ├── ForkButton
    │       └── StarButton
    │
    ├── AgentTabs
    │   ├── OverviewTab
    │   │   ├── ReadmeContent
    │   │   ├── FeaturesList
    │   │   └── ExampleCode
    │   │
    │   ├── DocumentationTab
    │   │   ├── TableOfContents
    │   │   └── DocContent
    │   │
    │   ├── ReviewsTab
    │   │   ├── ReviewSummary
    │   │   ├── ReviewForm
    │   │   └── ReviewList
    │   │
    │   └── VersionsTab
    │       ├── VersionSelector
    │       └── VersionHistory
    │
    └── AgentSidebar
        ├── StatsCard
        ├── DependenciesCard
        └── RelatedAgentsCard
```

---

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Data Flow Patterns                               │
└─────────────────────────────────────────────────────────────────────────┘

1. API Request Flow
─────────────────────

User Action
    │
    ▼
Component Event Handler
    │
    ▼
React Query (useMutation/useQuery)
    │
    ▼
API Client
    │
    ├─► Request Interceptor
    │   ├─► Add Auth Token
    │   ├─► Add Headers
    │   └─► Track Request
    │
    ▼
HTTP Request (axios)
    │
    ▼
Backend API
    │
    ▼
Response
    │
    ├─► Response Interceptor
    │   ├─► Handle Errors
    │   ├─► Refresh Token
    │   └─► Track Response
    │
    ▼
React Query Cache Update
    │
    ▼
Component Re-render
    │
    ▼
UI Update


2. Real-time Data Flow
──────────────────────

Backend Event
    │
    ▼
WebSocket / SSE
    │
    ▼
Event Listener
    │
    ▼
State Update (Zustand/React)
    │
    ▼
Component Re-render
    │
    ▼
UI Update


3. Authentication Flow
──────────────────────

Login Action
    │
    ▼
Auth API Call
    │
    ▼
Receive Tokens
    │
    ├─► Store in Memory
    ├─► Store in Cookie (httpOnly)
    └─► Update Auth State
    │
    ▼
Redirect to Dashboard
    │
    ▼
Subsequent Requests
    │
    ├─► Add Token to Header
    └─► Refresh on Expiry
```

---

## State Management

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     State Management Architecture                        │
└─────────────────────────────────────────────────────────────────────────┘

Global State (Zustand)
├── Auth State
│   ├── user: User | null
│   ├── isAuthenticated: boolean
│   ├── permissions: string[]
│   └── actions: { login, logout, refresh }
│
├── UI State
│   ├── theme: 'light' | 'dark'
│   ├── sidebarOpen: boolean
│   ├── commandPaletteOpen: boolean
│   └── actions: { toggle... }
│
└── Preferences State
    ├── language: string
    ├── timezone: string
    └── actions: { update... }

Server State (React Query)
├── Agents
│   ├── useAgents() → list
│   ├── useAgent(id) → single
│   ├── useCreateAgent() → mutation
│   └── useUpdateAgent() → mutation
│
├── Traces
│   ├── useTraces() → list
│   ├── useTrace(id) → single
│   └── useTraceMetrics() → analytics
│
└── Users
    ├── useUser() → current user
    └── useTeamMembers() → team

Local State (React)
├── Form State
│   ├── Input values
│   ├── Validation errors
│   └── Submission status
│
├── UI State
│   ├── Modal visibility
│   ├── Tab selection
│   └── Accordion state
│
└── Component State
    ├── Loading indicators
    ├── Error messages
    └── Temporary data

Real-time State (WebSocket/SSE)
├── Live Metrics
│   ├── Current throughput
│   ├── Active users
│   └── System health
│
├── Notifications
│   ├── New alerts
│   ├── System messages
│   └── Updates
│
└── Collaboration
    ├── User cursors
    ├── Document changes
    └── Comments
```

---

## Authentication Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Authentication Architecture                        │
└─────────────────────────────────────────────────────────────────────────┘

Login Flow
──────────
┌────────┐
│ User   │
│ Enters │
│ Creds  │
└───┬────┘
    │
    ▼
┌────────────────┐
│ Login Form     │
│ Validates      │
└───┬────────────┘
    │
    ▼
┌────────────────┐       ┌──────────────┐
│ POST /auth/    │──────►│ Auth Service │
│ login          │       │ (Backend)    │
└───┬────────────┘       └───┬──────────┘
    │                        │
    │                        ▼
    │                   ┌─────────────┐
    │                   │ Validate    │
    │                   │ Credentials │
    │                   └─────┬───────┘
    │                         │
    │◄────────────────────────┘
    │ { accessToken, refreshToken }
    │
    ▼
┌────────────────┐
│ Store Tokens   │
│ • Memory       │
│ • httpOnly     │
│   Cookie       │
└───┬────────────┘
    │
    ▼
┌────────────────┐
│ Update Auth    │
│ State (Zustand)│
└───┬────────────┘
    │
    ▼
┌────────────────┐
│ Redirect to    │
│ Dashboard      │
└────────────────┘

Token Refresh Flow
──────────────────
┌────────────────┐
│ API Request    │
│ Returns 401    │
└───┬────────────┘
    │
    ▼
┌────────────────┐       ┌──────────────┐
│ POST /auth/    │──────►│ Auth Service │
│ refresh        │       │ (Backend)    │
│ (refreshToken) │       └───┬──────────┘
└───┬────────────┘           │
    │                        ▼
    │                   ┌─────────────┐
    │                   │ Validate    │
    │                   │ Refresh     │
    │                   │ Token       │
    │                   └─────┬───────┘
    │                         │
    │◄────────────────────────┘
    │ { accessToken }
    │
    ▼
┌────────────────┐
│ Update Token   │
│ in Memory      │
└───┬────────────┘
    │
    ▼
┌────────────────┐
│ Retry Original │
│ Request        │
└────────────────┘

SSO Flow (SAML/OAuth)
─────────────────────
┌────────────────┐
│ User Clicks    │
│ "SSO Login"    │
└───┬────────────┘
    │
    ▼
┌────────────────┐
│ Redirect to    │
│ Identity       │
│ Provider       │
└───┬────────────┘
    │
    ▼
┌────────────────┐
│ User           │
│ Authenticates  │
│ with IdP       │
└───┬────────────┘
    │
    ▼
┌────────────────┐
│ IdP Redirects  │
│ with Token     │
└───┬────────────┘
    │
    ▼
┌────────────────┐       ┌──────────────┐
│ POST /auth/    │──────►│ Auth Service │
│ sso/callback   │       │ Validates    │
└───┬────────────┘       │ Token        │
    │                    └───┬──────────┘
    │                        │
    │◄───────────────────────┘
    │ { accessToken, refreshToken }
    │
    ▼
┌────────────────┐
│ Store & Login  │
└────────────────┘
```

---

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Production Deployment                               │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                            CDN Layer                                      │
│                         (CloudFront)                                      │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  Static Assets: Images, Fonts, CSS, JS                            │ │
│  │  Cache: 1 year                                                     │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐         ┌───────────────┐         ┌───────────────┐
│   Vercel      │         │   Vercel      │         │   Vercel      │
│   Edge        │         │   Edge        │         │   Edge        │
│  (US-East)    │         │  (EU-West)    │         │  (Asia-Pac)   │
└───────┬───────┘         └───────┬───────┘         └───────┬───────┘
        │                         │                         │
        └─────────────────────────┼─────────────────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │   Next.js Applications   │
                    ├──────────────────────────┤
                    │ • Developer Portal       │
                    │ • Visual Builder         │
                    │ • Hub                    │
                    │ • Studio                 │
                    │ • Dashboard              │
                    │ • Marketplace            │
                    └──────────┬───────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  PostgreSQL  │      │   Redis      │      │ ClickHouse   │
│  (Primary)   │      │   (Cache)    │      │ (Analytics)  │
│              │      │              │      │              │
│  RDS Multi-AZ│      │ ElastiCache  │      │ Managed      │
└──────────────┘      └──────────────┘      └──────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                          Backend Services                                 │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                │
│  │  Agent   │  │  Trace   │  │  User    │  │ Payment  │                │
│  │  API     │  │  API     │  │  API     │  │  API     │                │
│  │          │  │          │  │          │  │          │                │
│  │ ECS/K8s  │  │ ECS/K8s  │  │ ECS/K8s  │  │ ECS/K8s  │                │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘                │
│                                                                            │
│  Load Balancer (ALB)                                                      │
│  Auto Scaling Groups                                                      │
│  Health Checks                                                            │
│                                                                            │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                     Monitoring & Observability                            │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │  Grafana     │  │  Prometheus  │  │  Loki        │                  │
│  │  (Dashboards)│  │  (Metrics)   │  │  (Logs)      │                  │
│  └──────────────┘  └──────────────┘  └──────────────┘                  │
│                                                                            │
│  ┌──────────────┐  ┌──────────────┐                                     │
│  │  Sentry      │  │  PostHog     │                                     │
│  │  (Errors)    │  │  (Analytics) │                                     │
│  └──────────────┘  └──────────────┘                                     │
│                                                                            │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Development Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Development Workflow                              │
└─────────────────────────────────────────────────────────────────────────┘

Local Development
─────────────────
┌────────────────┐
│ Developer      │
│ Machine        │
└───┬────────────┘
    │
    ▼
┌────────────────┐
│ Make Changes   │
│ to Code        │
└───┬────────────┘
    │
    ▼
┌────────────────┐
│ Hot Reload     │
│ (Next.js)      │
└───┬────────────┘
    │
    ▼
┌────────────────┐
│ Test Locally   │
│ localhost:3000 │
└───┬────────────┘
    │
    ▼
┌────────────────┐
│ Commit to      │
│ Feature Branch │
└───┬────────────┘
    │
    ▼
┌────────────────┐
│ Push to GitHub │
└────────────────┘

CI/CD Pipeline
──────────────
┌────────────────┐
│ GitHub Push    │
└───┬────────────┘
    │
    ▼
┌────────────────────────────┐
│ GitHub Actions Triggered   │
├────────────────────────────┤
│                            │
│  1. Install Dependencies   │
│     npm install            │
│                            │
│  2. Lint & Format Check    │
│     eslint, prettier       │
│                            │
│  3. Type Check             │
│     tsc --noEmit           │
│                            │
│  4. Unit Tests             │
│     jest                   │
│                            │
│  5. Build                  │
│     next build             │
│                            │
│  6. E2E Tests              │
│     playwright             │
│                            │
│  7. Bundle Analysis        │
│     Check size < 500KB     │
│                            │
└───┬────────────────────────┘
    │
    ▼
┌────────────────┐
│ All Checks     │
│ Pass?          │
└───┬────────────┘
    │
    ├─── No ──► Fail Build, Notify Team
    │
    ▼ Yes
┌────────────────┐
│ Deploy to      │
│ Preview        │
│ (Vercel)       │
└───┬────────────┘
    │
    ▼
┌────────────────┐
│ Preview URL    │
│ Generated      │
└───┬────────────┘
    │
    ▼
┌────────────────┐
│ Code Review    │
│ on GitHub      │
└───┬────────────┘
    │
    ▼
┌────────────────┐
│ Merge to Main  │
└───┬────────────┘
    │
    ▼
┌────────────────┐
│ Deploy to      │
│ Production     │
│ (Automatic)    │
└────────────────┘

Release Process
───────────────
┌────────────────┐
│ Version Bump   │
│ (Semver)       │
└───┬────────────┘
    │
    ▼
┌────────────────┐
│ Tag Release    │
│ v1.2.3         │
└───┬────────────┘
    │
    ▼
┌────────────────┐
│ Generate       │
│ Changelog      │
└───┬────────────┘
    │
    ▼
┌────────────────┐
│ Create GitHub  │
│ Release        │
└───┬────────────┘
    │
    ▼
┌────────────────┐
│ Deploy to Prod │
│ with Tag       │
└───┬────────────┘
    │
    ▼
┌────────────────┐
│ Monitor        │
│ Metrics        │
└────────────────┘
```

---

## Performance Optimization Strategy

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   Performance Optimization Flow                          │
└─────────────────────────────────────────────────────────────────────────┘

Initial Load
────────────
User Request
    │
    ▼
┌────────────────┐
│ CDN Lookup     │
│ (CloudFront)   │
└───┬────────────┘
    │
    ├─── Cache Hit ──► Serve Cached Content (Fast!)
    │
    ▼ Cache Miss
┌────────────────┐
│ Edge Function  │
│ (Vercel)       │
└───┬────────────┘
    │
    ▼
┌────────────────┐
│ SSR or SSG     │
│ (Next.js)      │
└───┬────────────┘
    │
    ▼
┌────────────────┐
│ HTML Response  │
│ (with critical │
│  CSS inline)   │
└───┬────────────┘
    │
    ▼
┌────────────────┐
│ Browser Parses │
│ & Renders      │
└───┬────────────┘
    │
    ▼
┌────────────────┐
│ Load JS Chunks │
│ (code-split)   │
└───┬────────────┘
    │
    ▼
┌────────────────┐
│ Hydration      │
└───┬────────────┘
    │
    ▼
┌────────────────┐
│ Prefetch       │
│ Next Routes    │
└────────────────┘

Runtime Optimization
────────────────────
┌────────────────┐
│ User           │
│ Interaction    │
└───┬────────────┘
    │
    ▼
┌────────────────┐
│ Check Cache    │
│ (React Query)  │
└───┬────────────┘
    │
    ├─── Hit ──► Return Cached (Instant!)
    │
    ▼ Miss
┌────────────────┐
│ API Request    │
│ (with debounce)│
└───┬────────────┘
    │
    ▼
┌────────────────┐
│ Optimistic     │
│ Update UI      │
└───┬────────────┘
    │
    ▼
┌────────────────┐
│ Response       │
│ Received       │
└───┬────────────┘
    │
    ▼
┌────────────────┐
│ Update Cache   │
│ & UI           │
└───┬────────────┘
    │
    ▼
┌────────────────┐
│ Background     │
│ Revalidation   │
└────────────────┘

Image Optimization
──────────────────
┌────────────────┐
│ <Image> Tag    │
│ (Next.js)      │
└───┬────────────┘
    │
    ▼
┌────────────────┐
│ Automatic      │
│ Formats        │
│ (WebP, AVIF)   │
└───┬────────────┘
    │
    ▼
┌────────────────┐
│ Lazy Loading   │
│ (below fold)   │
└───┬────────────┘
    │
    ▼
┌────────────────┐
│ Placeholder    │
│ (blur)         │
└───┬────────────┘
    │
    ▼
┌────────────────┐
│ CDN Delivery   │
│ (optimized)    │
└────────────────┘
```

---

**Last Updated:** November 12, 2025
**Version:** 1.0
**Status:** Architecture Complete