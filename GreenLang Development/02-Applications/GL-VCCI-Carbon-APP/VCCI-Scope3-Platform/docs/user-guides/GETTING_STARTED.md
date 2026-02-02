# Getting Started with GL-VCCI Scope 3 Carbon Intelligence Platform - User Guide

**Audience**: New users, sustainability managers, procurement teams, and administrators
**Prerequisites**: Valid user account credentials, modern web browser (Chrome, Firefox, Edge, Safari)
**Time**: 15-30 minutes to complete initial setup and orientation

---

## Overview

Welcome to the GL-VCCI Scope 3 Carbon Intelligence Platform! This guide will help you get started with the platform quickly and efficiently. Whether you're tracking supplier emissions, generating compliance reports, or analyzing carbon hotspots, this guide covers everything you need to know to begin your journey.

### What You'll Learn
- Platform architecture and key capabilities
- First login and profile configuration
- Navigation fundamentals
- Complete a 15-minute quickstart tutorial
- Understand common workflows
- Access help and support resources

### Platform Overview

The GL-VCCI Platform is an enterprise-grade solution for managing Scope 3 carbon emissions across your value chain. Built on GreenLang's intelligence framework, it provides:

**Core Capabilities**:
- **Supplier Engagement**: Collect Product Carbon Footprint (PCF) data using PACT Pathfinder standard
- **Multi-Standard Reporting**: Generate ESRS E1, CDP, GHG Protocol, ISO 14083, and IFRS S2 reports
- **Hotspot Analysis**: AI-powered identification of emission reduction opportunities
- **Data Quality Management**: Automated validation and quality scoring
- **ERP Integration**: Connect to SAP, Oracle, Workday for procurement data
- **Policy Engine**: Enforce data quality and compliance rules

**Key Benefits**:
- 80% reduction in manual data processing time
- Automated multi-standard compliance reporting
- Real-time emission tracking and analysis
- Enterprise-grade security (SOC 2 Type II compliant)
- AI-assisted decision making for decarbonization

---

## Section 1: First Login and Account Setup

### Accessing the Platform

1. **Open your web browser** and navigate to: `https://vcci.greenlang.io`
2. **Enter your credentials**:
   - Email address (provided by your administrator)
   - Temporary password (sent via email)
3. **Click "Sign In"**

[Screenshot: Login page with email and password fields]

**💡 Tip**: Bookmark the platform URL for quick access. Enable your browser's password manager to securely store credentials.

**⚠️ Warning**: Do not share your login credentials with others. Each user should have their own account for audit trail purposes.

### First-Time Password Setup

On your first login, you'll be prompted to change your password:

1. **Enter your temporary password** in the "Current Password" field
2. **Create a new password** that meets the requirements:
   - Minimum 12 characters
   - At least one uppercase letter
   - At least one lowercase letter
   - At least one number
   - At least one special character (!@#$%^&*)
3. **Confirm your new password**
4. **Click "Update Password"**

[Screenshot: Password change dialog with strength indicator]

**💡 Tip**: Use a passphrase like "GreenSupply2025#Carbon!" for a strong, memorable password.

### Multi-Factor Authentication (MFA) Setup

For enhanced security, MFA is required:

1. **Choose your authentication method**:
   - Authenticator app (recommended): Google Authenticator, Microsoft Authenticator, Authy
   - SMS text message
   - Email verification code
2. **Scan the QR code** with your authenticator app (if using app method)
3. **Enter the 6-digit verification code**
4. **Save your backup codes** in a secure location
5. **Click "Enable MFA"**

[Screenshot: MFA setup screen with QR code and backup codes]

**⚠️ Warning**: Store backup codes in a secure location. You'll need them if you lose access to your authentication device.

### Profile Configuration

Complete your user profile for personalized experience:

1. **Click your avatar** in the top-right corner
2. **Select "Profile Settings"**
3. **Complete required fields**:
   - Full name
   - Job title
   - Department
   - Phone number
   - Time zone
   - Preferred language (English, German, French, Spanish, Japanese)
4. **Set notification preferences**:
   - Email notifications for report completion
   - Alert notifications for data quality issues
   - Weekly summary emails
5. **Upload a profile picture** (optional)
6. **Click "Save Profile"**

[Screenshot: Profile settings page with all fields]

**💡 Tip**: Set your correct time zone to ensure report timestamps and scheduled tasks run at the right time.

---

## Section 2: Platform Navigation

### Main Navigation Bar

The platform uses a persistent top navigation bar:

**Left Side**:
- **GreenLang Logo**: Click to return to home dashboard
- **Platform Selector**: Switch between GL-VCCI and other GreenLang products

**Center**:
- **Dashboards**: Overview and analytics
- **Suppliers**: Manage supplier relationships and data
- **Reports**: Generate and view compliance reports
- **Data**: Upload and manage procurement data
- **Analytics**: Deep-dive analysis and hotspot identification

**Right Side**:
- **Search**: Global platform search (Ctrl+K shortcut)
- **Notifications**: Bell icon shows alerts and updates
- **Help**: Access documentation and support
- **User Menu**: Profile, settings, and logout

[Screenshot: Main navigation bar with all elements labeled]

**💡 Tip**: Use the Ctrl+K (Windows) or Cmd+K (Mac) keyboard shortcut to quickly search across the platform.

### Side Navigation Panel

Each main section has a collapsible side panel:

**Dashboards Section**:
- Overview Dashboard
- Emissions Dashboard
- Supplier Dashboard
- Hotspot Dashboard
- Data Quality Dashboard

**Suppliers Section**:
- Active Suppliers
- Pending Invitations
- Supplier Portal Access
- Engagement Campaigns

**Reports Section**:
- Generate New Report
- Report Library
- Scheduled Reports
- Report Templates

**Data Section**:
- Upload Data
- Data Sources
- Validation Queue
- Upload History

**Analytics Section**:
- Hotspot Analysis
- Trend Analysis
- Category Breakdown
- Reduction Scenarios

[Screenshot: Side navigation panel expanded showing all options]

**💡 Tip**: Click the hamburger icon (≡) to collapse the side panel and gain more screen space.

### Dashboard Cards and Widgets

The home dashboard displays key metrics in card format:

**Top Row - Key Metrics**:
- Total Scope 3 Emissions (tCO2e)
- Number of Active Suppliers
- Data Quality Score
- Reporting Compliance Status

**Second Row - Trends**:
- Emissions Trend Chart (last 12 months)
- Top 5 Emission Categories
- Supplier Engagement Progress

**Third Row - Actions**:
- Recent Uploads
- Pending Actions
- Quick Links

[Screenshot: Home dashboard with all cards visible]

**💡 Tip**: Click any metric card to drill down into detailed views. Use the date range selector at the top to change the time period.

---

## Section 3: 15-Minute Quickstart Tutorial

Let's walk through a complete workflow to help you understand the platform capabilities.

### Tutorial Overview

In this tutorial, you'll:
1. Upload sample procurement data
2. Invite a supplier to submit PCF data
3. View the emissions dashboard
4. Generate a basic report

**Time Required**: 15 minutes

### Step 1: Upload Sample Procurement Data (5 minutes)

**Objective**: Import your organization's procurement transactions.

1. **Navigate to Data > Upload Data**
2. **Click "Download Template"** button
3. **Open the Excel template** and review the required fields:
   - Transaction ID
   - Supplier Name
   - Product/Service Description
   - Spend Amount
   - Purchase Date
   - Category (GHG Protocol Scope 3 category)
   - Unit of Measurement
   - Quantity

[Screenshot: Data upload page with template download button]

4. **For this tutorial, use the sample data**:
   - Click "Load Sample Data" button
   - Select "Q4 2024 Procurement Sample"
   - Click "Load Sample"

5. **Review the data preview**:
   - 50 sample transactions loaded
   - Covering categories 1, 4, 6, and 11
   - Date range: Oct-Dec 2024
   - Total spend: $2.5M

6. **Click "Validate Data"**:
   - System checks for required fields
   - Validates data formats
   - Flags any issues

7. **Review validation results**:
   - Green checkmark indicates success
   - Review any warnings or errors
   - Click "Import Data"

8. **Confirm import**:
   - Click "Confirm Import"
   - Wait for processing (10-15 seconds)
   - Success notification appears

[Screenshot: Data validation results with green checkmarks]

**💡 Tip**: Real uploads will take longer depending on file size. You can continue working while the upload processes in the background.

**Result**: You've successfully imported procurement data that will be used for emission calculations.

### Step 2: Invite a Supplier to Submit PCF Data (3 minutes)

**Objective**: Engage a supplier to provide Product Carbon Footprint data.

1. **Navigate to Suppliers > Active Suppliers**
2. **Find "Acme Manufacturing"** in the supplier list (from sample data)
3. **Click the three-dot menu** next to Acme Manufacturing
4. **Select "Invite to Portal"**

[Screenshot: Supplier list with invite option highlighted]

5. **Complete the invitation form**:
   - **Supplier Contact Name**: John Smith
   - **Email**: john.smith@acmemanufacturing.com (use your test email)
   - **Products to Request**: Select "Industrial Components"
   - **Message**: "Dear John, We're working to measure our Scope 3 emissions and would appreciate your participation. Please submit PCF data for the products we purchased in 2024. Thank you!"
   - **Due Date**: 30 days from today

6. **Click "Send Invitation"**

7. **Confirmation**:
   - Success message appears
   - Invitation status changes to "Pending"
   - Supplier will receive email with portal access link

[Screenshot: Supplier invitation form completed]

**💡 Tip**: You can customize email templates in Settings > Supplier Communications.

**Result**: Supplier has been invited and will receive portal access to submit their PCF data.

### Step 3: View Emissions Dashboard (4 minutes)

**Objective**: Understand your current Scope 3 emission profile.

1. **Navigate to Dashboards > Emissions Dashboard**
2. **Review the overview metrics**:
   - Total Scope 3 Emissions: 1,450 tCO2e (estimated)
   - Number of Categories: 4
   - Data Coverage: 45% (based on spend)
   - Quality Score: 3.2/5 (mostly spend-based estimates)

[Screenshot: Emissions dashboard overview with key metrics]

3. **Explore the Category Breakdown chart**:
   - Category 1 (Purchased Goods): 890 tCO2e (61%)
   - Category 4 (Upstream Transportation): 320 tCO2e (22%)
   - Category 6 (Business Travel): 180 tCO2e (12%)
   - Category 11 (Use of Sold Products): 60 tCO2e (5%)

4. **Click on "Category 1"** to drill down:
   - Shows emission by supplier
   - Top emitter: Acme Manufacturing (340 tCO2e)
   - Calculation method: Spend-based (EEIO factors)
   - Data quality: 2/5 (needs improvement)

[Screenshot: Category 1 drill-down showing supplier breakdown]

5. **Review the Trends chart**:
   - Select "Last 12 Months"
   - Observe quarterly trends
   - Note seasonality in business travel

6. **Check Data Quality indicator**:
   - 5% Primary Data (supplier-provided)
   - 40% Secondary Data (industry averages)
   - 55% Estimated (spend-based)
   - Goal: Increase primary data to 50%

**💡 Tip**: Hover over any chart element to see detailed tooltips with exact values and methodology.

**Result**: You now understand your emission profile and key improvement areas.

### Step 4: Generate a Basic Report (3 minutes)

**Objective**: Create a GHG Protocol Scope 3 report.

1. **Navigate to Reports > Generate New Report**
2. **Select report type**: "GHG Protocol Corporate Standard - Scope 3"
3. **Configure report parameters**:
   - **Reporting Period**: Q4 2024 (Oct 1 - Dec 31, 2024)
   - **Organizational Boundary**: Entire Organization
   - **Categories to Include**: Select All (1, 4, 6, 11)
   - **Base Year**: 2023
   - **Report Format**: PDF

[Screenshot: Report generation form with parameters]

4. **Click "Generate Report"**

5. **Wait for processing** (15-20 seconds):
   - Progress bar shows generation status
   - Notification will alert when complete

6. **View the generated report**:
   - Click "View Report" when ready
   - PDF opens in new tab
   - Review sections:
     - Executive Summary
     - Methodology
     - Category-by-Category Analysis
     - Data Quality Statement
     - Assurance Statement (if applicable)

[Screenshot: Generated PDF report preview]

7. **Download or share**:
   - Click "Download PDF"
   - Or click "Share" to email to stakeholders

**💡 Tip**: Save frequently-used report configurations as templates for one-click generation.

**✅ Tutorial Complete!**

Congratulations! You've completed the quickstart tutorial and learned:
- ✅ How to upload procurement data
- ✅ How to engage suppliers for PCF data
- ✅ How to navigate the emissions dashboard
- ✅ How to generate compliance reports

---

## Section 4: Common Workflows

### Workflow 1: Monthly Data Upload and Review

**Frequency**: Monthly
**Time Required**: 30-45 minutes
**User Role**: Sustainability Analyst

**Steps**:
1. Export procurement data from ERP system (SAP, Oracle, Workday)
2. Upload to GL-VCCI platform
3. Review validation results and fix errors
4. Check data quality score
5. Review emissions dashboard for changes
6. Document any significant variances
7. Update stakeholders

**💡 Tip**: Set up automated ERP connector to eliminate manual export/upload steps.

### Workflow 2: Quarterly Supplier Engagement Campaign

**Frequency**: Quarterly
**Time Required**: 2-3 hours (initial setup), 30 min/week (follow-up)
**User Role**: Procurement Manager, Sustainability Manager

**Steps**:
1. Identify top emitting suppliers (typically top 80% of emissions)
2. Prioritize based on spend and emission impact
3. Create personalized engagement campaigns
4. Send portal invitations with PCF data requests
5. Monitor submission status
6. Follow up on pending submissions
7. Review and validate submitted PCF data
8. Calculate emission improvements
9. Report results to leadership

**💡 Tip**: Use the Supplier Engagement Agent to automate follow-up emails and track response rates.

### Workflow 3: Annual Compliance Reporting

**Frequency**: Annually (typically Q1 for prior year)
**Time Required**: 1-2 weeks
**User Role**: Sustainability Director, ESG Manager

**Steps**:
1. Ensure all data for reporting year is uploaded
2. Review and finalize data quality scores
3. Document methodology and data sources
4. Generate draft reports (CDP, ESRS E1, IFRS S2)
5. Review with internal stakeholders
6. Make necessary adjustments
7. Obtain third-party assurance (if required)
8. Finalize and submit reports
9. Publish to website/ESG portal
10. Archive for audit trail

**💡 Tip**: Start report preparation in December to allow time for data quality improvements and stakeholder review.

### Workflow 4: Hotspot Analysis and Reduction Planning

**Frequency**: Bi-annually
**Time Required**: 4-8 hours
**User Role**: Sustainability Team, Procurement Team

**Steps**:
1. Navigate to Analytics > Hotspot Analysis
2. Run AI-powered hotspot identification
3. Review top 20 emission hotspots
4. Assess reduction feasibility for each:
   - Technical feasibility
   - Cost implications
   - Supplier capability
   - Timeline
5. Create reduction scenarios using the scenario planner
6. Model emission impact of each scenario
7. Prioritize initiatives based on cost-effectiveness
8. Create implementation roadmap
9. Assign owners and deadlines
10. Track progress monthly

**💡 Tip**: Focus on the "low-hanging fruit" - high-impact, low-cost reduction opportunities first.

### Workflow 5: Data Quality Improvement Initiative

**Frequency**: Ongoing
**Time Required**: 1-2 hours/week
**User Role**: Sustainability Analyst

**Steps**:
1. Review Data Quality Dashboard
2. Identify transactions using lowest quality methods (spend-based)
3. Prioritize by emission magnitude
4. For each priority item:
   - Check if supplier-specific data available
   - Request PCF from supplier if not
   - Apply industry average if supplier data unavailable
   - Document data source and quality score
5. Update calculation methodology
6. Recalculate emissions
7. Monitor quality score improvement
8. Report progress monthly

**💡 Tip**: Set a goal to improve quality score by 0.5 points per quarter through systematic supplier engagement.

---

## Section 5: Key Features Deep Dive

### AI-Powered Features

**Hotspot Analysis Agent**:
- Automatically identifies top emission sources
- Analyzes patterns across categories and suppliers
- Suggests reduction opportunities
- Estimates potential savings
- Access: Analytics > Hotspot Analysis

**Supplier Engagement Agent**:
- Drafts personalized outreach emails
- Tracks engagement status and follow-ups
- Provides response rate analytics
- Suggests optimal timing for follow-ups
- Access: Suppliers > Engagement Campaigns

**Reporting Agent**:
- Generates narrative explanations for emission trends
- Creates executive summaries
- Drafts methodology disclosures
- Suggests report improvements
- Access: Reports > Generate New Report > Use AI Assistant

**Data Quality Agent**:
- Identifies low-quality data
- Suggests improvement methods
- Prioritizes data collection efforts
- Validates submitted data
- Access: Data > Validation Queue

**💡 Tip**: AI agents learn from your organization's patterns over time, becoming more accurate with use.

### Integration Capabilities

**ERP Connectors**:
- **SAP S/4HANA**: Direct integration with MM (Materials Management) module
- **Oracle Fusion Procurement**: Real-time procurement data sync
- **Workday**: Supplier and spend data integration
- **Setup**: Settings > Integrations > Configure Connector

**PCF Data Exchange**:
- **PACT Pathfinder**: Native support for industry standard format
- **WBCSD**: Compatible with WBCSD guidance
- **API Access**: RESTful API for custom integrations
- **Setup**: Settings > API Access

**Reporting Platforms**:
- **CDP Platform**: Direct submission integration
- **ESG Reporting Software**: Export in compatible formats
- **Data Warehouses**: Scheduled data exports

**💡 Tip**: Start with read-only ERP integration to test before enabling write-back capabilities.

### Collaboration Features

**Team Workspaces**:
- Share dashboards and reports
- Collaborative data review
- Comments and annotations
- Task assignments
- Access: Team > Workspaces

**Approval Workflows**:
- Multi-level report approval
- Data validation sign-off
- Change request tracking
- Audit trail
- Access: Settings > Workflows

**External Sharing**:
- Share read-only dashboards with auditors
- Provide supplier access to their data
- Generate public-facing reports
- Access: Share button on any dashboard/report

---

## Section 6: Customization Options

### Dashboard Customization

1. **Navigate to any dashboard**
2. **Click "Customize Dashboard"** button (top-right)
3. **Drag and drop widgets**:
   - Add new widgets from library
   - Resize widgets
   - Reorder widgets
   - Remove unused widgets
4. **Configure widget settings**:
   - Click gear icon on widget
   - Set data source
   - Choose visualization type
   - Set refresh interval
5. **Save as custom view**:
   - Click "Save View"
   - Name your view
   - Set as default (optional)
   - Share with team (optional)

[Screenshot: Dashboard customization mode with widget library]

**💡 Tip**: Create role-specific dashboard views (Executive, Analyst, Procurement) for quick context switching.

### Report Templates

1. **Navigate to Reports > Report Templates**
2. **Click "Create New Template"**
3. **Select base standard**: GHG Protocol, ESRS E1, CDP, ISO 14083, IFRS S2
4. **Customize sections**:
   - Add company branding (logo, colors)
   - Modify section order
   - Add custom sections
   - Configure default parameters
5. **Set calculation methodology defaults**
6. **Save template**
7. **Use template**: Select when generating new report

**💡 Tip**: Create templates for different audiences (board, regulators, investors, customers).

### Emission Factor Libraries

Customize which emission factor databases to use:

1. **Navigate to Settings > Emission Factors**
2. **Review available databases**:
   - US EPA EEIO
   - Ecoinvent 3.9
   - GHG Protocol Default Factors
   - DEFRA UK
   - Regional databases (EU, Asia, etc.)
3. **Set priority order**:
   - Drag databases to set precedence
   - Platform will use highest priority factor available
4. **Upload custom factors**:
   - For company-specific products
   - Supplier-provided factors
   - Industry collaborations
5. **Set update schedule**: Auto-update when new versions released

**💡 Tip**: Use regional factors when available for more accurate calculations (e.g., DEFRA for UK operations).

---

## Section 7: Notifications and Alerts

### Email Notifications

Configure what events trigger email notifications:

1. **Navigate to Profile > Notification Settings**
2. **Select notification types**:
   - ✅ Data upload completed
   - ✅ Validation errors found
   - ✅ Report generation complete
   - ✅ Supplier submitted PCF data
   - ✅ Data quality score changed significantly
   - ✅ Weekly summary
   - ❌ Every individual transaction processed
3. **Set frequency**:
   - Immediate
   - Daily digest
   - Weekly summary
4. **Save settings**

### In-Platform Alerts

Set up alerts for specific conditions:

1. **Navigate to Dashboards > any dashboard**
2. **Click "Create Alert"**
3. **Configure alert condition**:
   - Metric: Total Emissions
   - Condition: Increases by more than
   - Threshold: 10%
   - Time period: Month-over-month
4. **Set notification method**:
   - In-platform notification
   - Email
   - Slack/Teams integration
5. **Save alert**

**Example Alerts**:
- Total emissions exceed target
- Data quality score drops below threshold
- Supplier doesn't respond within 14 days
- Report deadline approaching
- Integration sync failure

**💡 Tip**: Use alerts sparingly to avoid notification fatigue. Focus on actionable items.

---

## Section 8: Mobile Access

### Mobile Web App

Access key features on mobile devices:

**Supported Features**:
- ✅ View dashboards
- ✅ Review reports
- ✅ Approve workflows
- ✅ Check notifications
- ✅ Quick data entry
- ❌ Full data upload (use desktop)
- ❌ Advanced analytics (use desktop)

**Accessing on Mobile**:
1. Open mobile browser
2. Navigate to: `https://vcci.greenlang.io`
3. Login with credentials
4. Platform auto-adapts to mobile screen

**💡 Tip**: Add to home screen for app-like experience (iOS: Share > Add to Home Screen; Android: Menu > Add to Home Screen).

---

## Section 9: Security and Privacy

### Data Security

**Platform Security Features**:
- SOC 2 Type II certified
- ISO 27001 compliant
- End-to-end encryption in transit (TLS 1.3)
- Encryption at rest (AES-256)
- Multi-factor authentication required
- Role-based access control (RBAC)
- Regular security audits
- Penetration testing quarterly

**Your Responsibilities**:
- Use strong, unique passwords
- Enable MFA
- Don't share credentials
- Log out on shared devices
- Report suspicious activity
- Follow data classification policies

### Data Privacy

**Data Handling**:
- GDPR compliant (EU users)
- CCPA compliant (California users)
- Data residency options (EU, US, Asia)
- Customer data isolation
- Right to deletion
- Data portability

**What We Collect**:
- Account information (name, email, role)
- Procurement and emission data you upload
- Platform usage analytics
- Log files for security

**What We Don't Collect**:
- Financial performance data (beyond spend)
- Competitive sensitive information
- Personal data of your employees (beyond platform users)

**💡 Tip**: Review our Privacy Policy at Settings > Privacy & Security > Privacy Policy.

---

## Section 10: Performance Tips

### Optimize Large Dataset Handling

**For Uploads > 10,000 rows**:
- Split into multiple files of 10,000 rows each
- Upload during off-peak hours (evenings/weekends)
- Use CSV format instead of Excel for faster processing
- Compress files before upload
- Use API for very large datasets (>100,000 rows)

**For Dashboard Performance**:
- Limit date ranges when possible
- Use summary views instead of transaction-level
- Schedule intensive reports to run overnight
- Clear browser cache if experiencing slowness

### Browser Optimization

**Recommended Browsers** (latest versions):
1. Google Chrome (best performance)
2. Microsoft Edge
3. Mozilla Firefox
4. Safari (Mac only)

**Browser Settings**:
- Enable JavaScript
- Allow cookies from vcci.greenlang.io
- Minimum screen resolution: 1280x720
- Disable browser extensions that block requests

**💡 Tip**: Use Chrome for best performance, especially with large datasets and complex dashboards.

---

## Troubleshooting

### Common Issues and Solutions

**Issue**: Cannot login - "Invalid credentials" error
**Solution**:
1. Verify email address is correct (case-sensitive)
2. Check Caps Lock is off
3. Try "Forgot Password" to reset
4. Contact your administrator to verify account is active
5. Clear browser cache and cookies

**Issue**: MFA code not working
**Solution**:
1. Verify time is correct on your device (MFA is time-based)
2. Try next code (codes refresh every 30 seconds)
3. Use backup code if available
4. Contact support to reset MFA

**Issue**: Data upload fails with "Invalid format" error
**Solution**:
1. Download the latest template (format may have changed)
2. Verify required fields are populated
3. Check date formats (YYYY-MM-DD)
4. Remove special characters from text fields
5. Ensure numeric fields contain only numbers
6. Check file size is under 50MB

**Issue**: Report generation stuck at "Processing"
**Solution**:
1. Wait up to 5 minutes (large reports take time)
2. Check notification center for completion alert
3. Refresh the page
4. If still stuck after 10 minutes, cancel and retry
5. Try generating with smaller date range

**Issue**: Dashboard shows "No data available"
**Solution**:
1. Verify date range includes periods with data
2. Check filters aren't excluding all data
3. Confirm data upload completed successfully
4. Navigate to Data > Upload History to verify
5. Contact support if data should be present

**Issue**: Supplier portal invitation not received
**Solution**:
1. Check spam/junk folder
2. Verify email address is correct
3. Resend invitation from Suppliers > Active Suppliers
4. Try alternative email address
5. Contact supplier by phone to confirm receipt

**Issue**: Emissions seem too high/low
**Solution**:
1. Review calculation methodology used
2. Check units of measurement are correct
3. Verify emission factors are appropriate for region
4. Compare to industry benchmarks
5. Review data quality score - low quality indicates estimates
6. Check for duplicate transactions

**Issue**: Integration not syncing data
**Solution**:
1. Check integration status: Settings > Integrations
2. Verify credentials haven't expired
3. Check network connectivity
4. Review integration logs for errors
5. Ensure ERP permissions are correct
6. Contact support with error codes

---

## FAQ

**Q: How often should I upload data?**
**A**: Best practice is monthly uploads to maintain current emissions tracking. For annual compliance reporting, ensure all data is uploaded before reporting deadlines. Automated ERP integration can provide daily updates.

**Q: Can I delete data after uploading?**
**A**: Yes, but with restrictions. You can delete data in "Draft" status. Once data is included in a finalized report, it's archived for audit purposes but can be marked as obsolete. Navigate to Data > Upload History to manage uploads.

**Q: How is data quality scored?**
**A**: Data quality uses a 5-point scale:
- 5: Primary data from supplier (PCF)
- 4: Secondary data (product-specific factors)
- 3: Industry average data
- 2: Proxy data (similar products)
- 1: Spend-based estimates (EEIO)
Higher scores indicate more accurate emissions.

**Q: Can multiple users work on the same report?**
**A**: Yes, the platform supports collaborative editing. Multiple users can contribute to report sections, add comments, and review. The final approval workflow ensures proper sign-off before finalization.

**Q: What happens if a supplier doesn't respond to the portal invitation?**
**A**: The system automatically sends reminder emails at 7, 14, and 21 days. You can customize reminder frequency in Settings. If supplier doesn't respond, you can:
- Use industry average data
- Apply spend-based calculation
- Engage supplier through other channels
- Work with account manager for custom engagement

**Q: How accurate are spend-based emission estimates?**
**A**: Spend-based (EEIO) estimates provide ballpark accuracy (±50-100% variance possible) as they use economy-wide averages. They're useful for initial assessment but should be replaced with supplier-specific data for accurate reporting.

**Q: Can I export raw data for analysis in Excel/BI tools?**
**A**: Yes. Navigate to any dashboard, click "Export Data", and choose format (CSV, Excel, JSON). You can also set up scheduled exports to your data warehouse via Settings > Integrations > Data Exports.

**Q: Is my company's data visible to other platform users?**
**A**: No. Data is completely isolated between organizations. Even GreenLang support cannot access your data without explicit permission and audit trail logging.

**Q: Can I benchmark against other companies?**
**A**: Yes, anonymized benchmarking is available. Navigate to Analytics > Benchmarking to compare your emission intensity (tCO2e per $M spend) against industry peers. All data is aggregated and anonymized.

**Q: What's the difference between "Draft" and "Final" reports?**
**A**:
- **Draft**: Editable, can be regenerated, not included in audit trail, can be deleted
- **Final**: Read-only, permanently archived, included in audit trail, cannot be deleted (only superseded)
Always review drafts thoroughly before finalizing.

---

## Related Resources

### Documentation
- [Supplier Portal Guide](./SUPPLIER_PORTAL_GUIDE.md) - For engaging suppliers
- [Reporting Guide](./REPORTING_GUIDE.md) - Detailed report generation instructions
- [Data Upload Guide](./DATA_UPLOAD_GUIDE.md) - Data preparation and upload procedures
- [Dashboard Usage Guide](./DASHBOARD_USAGE_GUIDE.md) - Advanced dashboard features

### Video Tutorials
- Platform Overview (10 minutes) - `https://vcci.greenlang.io/help/videos/overview`
- Data Upload Walkthrough (15 minutes) - `https://vcci.greenlang.io/help/videos/data-upload`
- Report Generation Tutorial (20 minutes) - `https://vcci.greenlang.io/help/videos/reporting`
- Supplier Engagement Best Practices (25 minutes) - `https://vcci.greenlang.io/help/videos/supplier-engagement`

### Support
- **Help Center**: Click "?" icon in top navigation
- **Live Chat**: Available Mon-Fri 9am-5pm EST
- **Email Support**: support@greenlang.io (24-hour response time)
- **Community Forum**: `https://community.greenlang.io`
- **Training Sessions**: Contact success@greenlang.io to schedule

### Technical Documentation
- API Documentation - `https://api.greenlang.io/docs`
- Integration Guides - Settings > Integrations > Documentation
- Data Schema Reference - `https://vcci.greenlang.io/help/schemas`
- Security Whitepaper - `https://greenlang.io/security`

### External Resources
- GHG Protocol Scope 3 Standard - `https://ghgprotocol.org/scope-3-standard`
- PACT Pathfinder Framework - `https://wbcsd.github.io/data-exchange-protocol/`
- ESRS E1 Standard - `https://www.efrag.org/lab6`
- ISO 14083:2023 - Quantification and reporting of GHG emissions
- CDP Climate Change Questionnaire - `https://www.cdp.net/`

---

## Next Steps

Now that you're familiar with the platform basics:

1. **Complete your profile** setup if you haven't already
2. **Upload your first dataset** (use sample data to practice)
3. **Invite 2-3 key suppliers** to the supplier portal
4. **Generate your first report** (start with a simple GHG Protocol report)
5. **Explore dashboards** to understand your emission profile
6. **Schedule a success call** with your customer success manager
7. **Join a training webinar** to learn advanced features

**Welcome to GL-VCCI!** We're excited to support your Scope 3 emission management journey.

---

*Last Updated: 2025-11-07*
*Document Version: 1.0*
*Platform Version: 2.5.0*
