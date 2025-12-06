# GreenLang Process Heat Platform - Quick Start Guide

**Document Version:** 1.0.0
**Last Updated:** 2025-12-06
**Time to Complete:** 5-10 minutes

---

## Overview

This guide will help you get started with the GreenLang Process Heat Platform in minutes. By the end, you will have:

1. Logged in to the platform
2. Viewed the system dashboard
3. Run your first efficiency calculation
4. Reviewed the results and recommendations

---

## Prerequisites

Before you begin, ensure you have:

- [ ] Valid user account (contact your administrator)
- [ ] Network access to the GreenLang console
- [ ] Authenticator app installed (for MFA)
- [ ] Modern web browser (Chrome, Firefox, Edge, Safari)

---

## Step 1: Log In to the Platform

**Time: 1 minute**

### 1.1 Access the Console

Open your web browser and navigate to:
- **Cloud:** `https://processheat.greenlang.io`
- **On-premises:** `https://[your-server]/processheat`

### 1.2 Enter Your Credentials

```
+-------------------------------------------+
|         GreenLang Process Heat            |
|                                           |
|  Username: [your.email@company.com     ]  |
|  Password: [************************   ]  |
|                                           |
|         [    Sign In    ]                 |
+-------------------------------------------+
```

[Screenshot placeholder: Login screen]

### 1.3 Complete Multi-Factor Authentication

Enter the 6-digit code from your authenticator app:

```
+-------------------------------------------+
|      Two-Factor Authentication            |
|                                           |
|  Enter the code from your authenticator:  |
|                                           |
|         [  1  2  3  4  5  6  ]           |
|                                           |
|         [    Verify    ]                  |
+-------------------------------------------+
```

**Success!** You are now logged in.

---

## Step 2: Explore the Dashboard

**Time: 2 minutes**

After logging in, you will see the main dashboard:

```
+------------------------------------------------------------------+
|  [Logo] GreenLang Process Heat    [Alarms: 0] [User: You v]      |
+------------------------------------------------------------------+
|  Home | Agents | Alarms | Trends | Reports | Settings            |
+------------------------------------------------------------------+
|                                                                    |
|  +------------------+  +------------------+  +------------------+  |
|  | SYSTEM STATUS    |  | EFFICIENCY       |  | EMISSIONS        |  |
|  | [RUNNING]        |  | 87.3%            |  | COMPLIANT        |  |
|  | Agents: 18/18    |  | Target: 88.0%    |  | CO2: 1,245 lb/hr |  |
|  +------------------+  +------------------+  +------------------+  |
|                                                                    |
|  +---------------------------------------------------------------+ |
|  |                    BOILER OVERVIEW                             | |
|  | +---------------+ +---------------+ +---------------+          | |
|  | | Boiler B-001  | | Boiler B-002  | | Boiler B-003  |          | |
|  | | [ONLINE]      | | [ONLINE]      | | [STANDBY]     |          | |
|  | | Load: 85%     | | Load: 72%     | | Load: 0%      |          | |
|  | | Eff: 87.1%    | | Eff: 86.9%    | | Eff: --       |          | |
|  | +---------------+ +---------------+ +---------------+          | |
|  +---------------------------------------------------------------+ |
+------------------------------------------------------------------+
```

[Screenshot placeholder: Main dashboard]

### Key Areas of the Dashboard

| Area | Description |
|------|-------------|
| **System Status** | Shows overall platform health - green means all systems operational |
| **Efficiency** | Current plant-wide thermal efficiency with target comparison |
| **Emissions** | Compliance status and current emission rates |
| **Boiler Overview** | Quick view of each boiler's status and performance |
| **Recommendations** | AI-generated optimization suggestions |
| **Recent Alarms** | Latest alerts requiring attention |

### Navigation Menu

| Menu Item | What You'll Find |
|-----------|------------------|
| **Home** | Main dashboard (current view) |
| **Agents** | Detailed status of all optimization agents |
| **Alarms** | Active and historical alarms |
| **Trends** | Real-time and historical data charts |
| **Reports** | Generate efficiency and compliance reports |
| **Settings** | Personal preferences and configuration |

---

## Step 3: Run Your First Calculation

**Time: 2 minutes**

Let's calculate boiler efficiency for one of your boilers.

### 3.1 Navigate to Boiler Agent

1. Click on **Agents** in the navigation menu
2. Find **GL-002-B001** (Boiler Optimizer B-001)
3. Click on the agent to open details

### 3.2 View Current Efficiency

The agent detail screen shows:

```
+------------------------------------------------------------------+
|  GL-002-B001 - Boiler Optimizer B-001                            |
+------------------------------------------------------------------+
|  Status: RUNNING               Version: 1.0.0                     |
|  Safety Level: SIL-2           Last Calculation: 2 seconds ago   |
+------------------------------------------------------------------+
|                                                                    |
|  CURRENT METRICS                                                  |
|  +------------------------+  +------------------------+           |
|  | Net Efficiency         |  | Combustion Efficiency  |           |
|  | 87.1%                  |  | 89.2%                  |           |
|  | Target: 88.0%          |  | Target: 90.0%          |           |
|  +------------------------+  +------------------------+           |
|                                                                    |
|  +------------------------+  +------------------------+           |
|  | Excess Air             |  | Stack Temperature      |           |
|  | 15.2%                  |  | 385 F                  |           |
|  | Optimal: 12-15%        |  | Optimal: <400 F        |           |
|  +------------------------+  +------------------------+           |
|                                                                    |
+------------------------------------------------------------------+
```

[Screenshot placeholder: Agent detail view]

### 3.3 Request Manual Calculation

To trigger an immediate efficiency calculation:

1. Click the **Calculate Now** button
2. The agent processes current sensor data
3. Results appear within 2-3 seconds

```
+------------------------------------------+
|  CALCULATION COMPLETE                     |
+------------------------------------------+
|  Calculation ID: calc_2025120614325001   |
|  Duration: 245 ms                        |
|                                          |
|  NET EFFICIENCY: 87.1%                   |
|  Standard: ASME PTC 4.1                  |
|                                          |
|  LOSS BREAKDOWN:                         |
|  - Dry flue gas loss: 5.8%              |
|  - Moisture loss: 3.2%                   |
|  - Radiation loss: 0.8%                  |
|  - Blowdown loss: 1.1%                   |
|  - Other losses: 2.0%                    |
|  - Total losses: 12.9%                   |
|                                          |
|  Provenance Hash: a3f2c9b7...            |
|                                          |
|         [View Details]  [Close]          |
+------------------------------------------+
```

[Screenshot placeholder: Calculation results]

**Congratulations!** You have completed your first efficiency calculation with full audit trail.

---

## Step 4: Understanding Results

**Time: 2 minutes**

### 4.1 Efficiency Metrics Explained

| Metric | Description | Good Range |
|--------|-------------|------------|
| **Net Efficiency** | Overall boiler efficiency after all losses | > 82% |
| **Combustion Efficiency** | How completely fuel is burned | > 88% |
| **Excess Air** | Extra air beyond stoichiometric | 12-18% |
| **Stack Temperature** | Flue gas exit temperature | < 400 F |

### 4.2 Loss Breakdown

The platform calculates losses per ASME PTC 4.1:

| Loss Type | Typical Range | What It Means |
|-----------|---------------|---------------|
| **Dry Flue Gas** | 4-8% | Heat lost in hot exhaust gases |
| **Moisture** | 2-4% | Heat to vaporize water in fuel |
| **Radiation** | 0.5-2% | Heat lost from boiler surfaces |
| **Blowdown** | 0.5-2% | Heat lost with blowdown water |

### 4.3 Viewing Recommendations

Scroll down to see AI-generated optimization recommendations:

```
+------------------------------------------------------------------+
|  OPTIMIZATION RECOMMENDATIONS                                      |
+------------------------------------------------------------------+
|                                                                    |
|  [HIGH] Optimize Air-Fuel Ratio                                   |
|  O2 is 0.3% above optimal. Reduce to 2.5% for +0.3% efficiency.  |
|  Current: 2.8%  |  Recommended: 2.5%  |  Savings: 0.3%           |
|  Difficulty: Low  |  [Apply]  [Dismiss]                          |
|                                                                    |
|  [MEDIUM] Clean Economizer                                        |
|  Economizer effectiveness at 78% vs design 85%.                  |
|  Schedule cleaning during next outage.                           |
|  Estimated Savings: 1.0%  |  Difficulty: Medium                  |
|  [Schedule]  [Dismiss]                                            |
|                                                                    |
+------------------------------------------------------------------+
```

[Screenshot placeholder: Recommendations panel]

### 4.4 Calculation Provenance

Every calculation includes a provenance hash for audit purposes:

- **Provenance Hash**: Unique identifier proving calculation integrity
- **Calculation Method**: Standard used (ASME PTC 4.1)
- **Data Sources**: Sensor inputs used in calculation
- **Timestamp**: Exact time of calculation

Click **View Audit Trail** to see full calculation details.

---

## Step 5: Next Steps

You have completed the Quick Start! Here are recommended next steps:

### Immediate (Today)

- [ ] Explore other agents (Steam, Emissions, Maintenance)
- [ ] Review current active alarms
- [ ] Check the trend graphs for your equipment

### This Week

- [ ] Review the [Operator Manual](./operator_manual.md) for detailed procedures
- [ ] Set up personal dashboard preferences
- [ ] Familiarize yourself with alarm response procedures

### Ongoing

- [ ] Monitor efficiency trends daily
- [ ] Act on optimization recommendations
- [ ] Complete shift handover reports
- [ ] Report any issues to your supervisor

---

## Quick Reference Card

### Common Tasks

| Task | How To |
|------|--------|
| View dashboard | Click **Home** |
| Check agent status | Click **Agents** > select agent |
| Acknowledge alarm | Click **Alarms** > [Ack] button |
| Run calculation | Agent detail > **Calculate Now** |
| Generate report | Click **Reports** > select type |
| View trends | Click **Trends** > select metric |

### Key Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl + H` | Go to Home |
| `Ctrl + A` | Go to Alarms |
| `Ctrl + R` | Refresh data |
| `Esc` | Close dialog |

### Status Indicators

| Color | Meaning |
|-------|---------|
| Green | Healthy/Normal |
| Yellow | Warning/Attention |
| Red | Error/Critical |
| Blue | Information |
| Gray | Offline/Disabled |

---

## Getting Help

### In-App Help

- Click the **?** icon in any screen for context help
- Use the search bar to find specific topics

### Documentation

- [Operator Manual](./operator_manual.md) - Complete operations guide
- [Administrator Guide](./administrator_guide.md) - System administration
- [Troubleshooting Guide](./troubleshooting.md) - Problem resolution
- [Glossary](./glossary.md) - Term definitions

### Support

- **Help Desk:** helpdesk@company.com
- **GreenLang Support:** support@greenlang.io
- **Emergency Hotline:** 1-800-XXX-XXXX

---

## Frequently Asked Questions

**Q: Why can't I see certain boilers?**

A: You may not have permission to view all equipment. Contact your administrator to request additional access.

**Q: How often are calculations updated?**

A: The system runs calculations every 30 seconds by default. You can also trigger manual calculations anytime.

**Q: What does the provenance hash mean?**

A: The provenance hash is a cryptographic fingerprint that proves the calculation was performed correctly and hasn't been tampered with. It's used for regulatory compliance and audit purposes.

**Q: How do I report a problem?**

A: Click **Settings > Report Issue** or contact your supervisor. Include the provenance hash if related to a calculation.

**Q: Can I customize my dashboard?**

A: Yes! Click the gear icon on the dashboard and select "Edit Layout" to rearrange and add widgets.

---

*Welcome to GreenLang Process Heat! For questions, contact support@greenlang.io*
