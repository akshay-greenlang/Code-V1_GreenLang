# Feedback Guide - IndustrialProcessHeatAgent_AI

## Overview

This guide explains how to collect, analyze, and act on user feedback for the Industrial Process Heat Agent. Continuous feedback helps us improve accuracy, performance, and user experience.

## Collecting Feedback

### API Endpoint
```
POST /api/v1/feedback/industrial_process_heat
```

### Request Format
```json
{
  "session_id": "session_1697123456.789",
  "rating": 4.5,
  "accuracy": "high",
  "latency_acceptable": true,
  "usefulness": "very_useful",
  "suggestions": "Would like to see more detailed economic analysis",
  "feedback_type": "feature_request",
  "user_context": {
    "industry": "Food & Beverage",
    "company_size": "medium",
    "use_case": "facility_planning"
  }
}
```

### Response Format
```json
{
  "success": true,
  "feedback_id": "fb_abc123",
  "message": "Thank you for your feedback!"
}
```

## Feedback Fields

### Required Fields
- `session_id` (string): Unique session identifier from agent response
- `rating` (float): Overall rating from 1.0 to 5.0

### Optional Fields
- `accuracy` (enum): "low", "medium", "high", "very_high"
- `latency_acceptable` (boolean): Was response time acceptable?
- `usefulness` (enum): "not_useful", "somewhat_useful", "useful", "very_useful"
- `suggestions` (string): Free-form text suggestions
- `feedback_type` (enum): "bug_report", "feature_request", "accuracy_issue", "performance_issue", "general"
- `user_context` (object): Additional context about the user

## Embedding Feedback in Applications

### JavaScript/TypeScript Example
```javascript
// After receiving agent response
const agentResponse = await fetch('/api/v1/agents/industrial/process_heat/execute', {
  method: 'POST',
  body: JSON.stringify(queryData)
});

const result = await agentResponse.json();
const sessionId = result.data._session_id;

// Collect user feedback
const feedback = {
  session_id: sessionId,
  rating: 4.5,
  accuracy: "high",
  latency_acceptable: true,
  usefulness: "very_useful"
};

await fetch('/api/v1/feedback/industrial_process_heat', {
  method: 'POST',
  body: JSON.stringify(feedback)
});
```

### Python Example
```python
import requests

# Execute agent query
response = requests.post(
    "https://api.greenlang.com/api/v1/agents/industrial/process_heat/execute",
    json=query_data
)
result = response.json()
session_id = result["data"]["_session_id"]

# Submit feedback
feedback = {
    "session_id": session_id,
    "rating": 4.5,
    "accuracy": "high",
    "latency_acceptable": True,
    "usefulness": "very_useful",
    "suggestions": "Great analysis! Would love to see cost breakdown."
}

requests.post(
    "https://api.greenlang.com/api/v1/feedback/industrial_process_heat",
    json=feedback
)
```

## A/B Testing

### Feature Flags

The agent supports A/B testing for experimental features:

#### Available Feature Flags
- `solar_fraction_v2`: Enhanced f-Chart calculation with monthly resolution
- `lifecycle_emissions_v2`: Advanced LCA model with supply chain emissions
- `economic_analysis_v1`: NPV and IRR calculations
- `realtime_irradiance`: Live solar resource data integration

### Enabling Feature Flags
```python
query = {
    "industry_type": "Food & Beverage",
    "process_type": "pasteurization",
    # ... other fields ...
    "_feature_flags": {
        "solar_fraction_v2": True,
        "economic_analysis_v1": True
    }
}
```

### A/B Test Configuration
```yaml
ab_tests:
  - name: "solar_fraction_v2"
    description: "Enhanced solar fraction calculation"
    allocation:
      control: 0.5  # 50% get current version
      treatment: 0.5  # 50% get new version
    metrics:
      - accuracy_rating
      - user_satisfaction
      - adoption_rate
    duration_days: 30

  - name: "economic_analysis_v1"
    description: "New economic analysis features"
    allocation:
      control: 0.7  # 70% control
      treatment: 0.3  # 30% treatment
    metrics:
      - feature_usage
      - user_satisfaction
      - conversion_rate
    duration_days: 45
```

## Feedback Analysis

### Weekly Reports

Automated weekly feedback summary includes:
- Average rating by dimension (accuracy, latency, usefulness)
- Common themes in text feedback
- Feature request categorization
- Bug report prioritization
- Performance trend analysis

### Monthly Reviews

Monthly comprehensive analysis includes:
- Quarter-over-quarter trends
- User segment analysis
- A/B test results
- Improvement recommendations
- Roadmap adjustments

## Continuous Improvement Process

### 1. Collect Feedback (Ongoing)
- Embedded feedback forms in applications
- Post-query satisfaction surveys
- User interviews and focus groups
- Support ticket analysis
- Usage analytics

### 2. Analyze Trends (Weekly)
- Aggregate feedback metrics
- Identify common patterns
- Categorize issues and requests
- Prioritize by impact and frequency

### 3. Prioritize Improvements (Monthly)
- Review top issues and requests
- Assess technical feasibility
- Estimate effort and impact
- Align with product roadmap
- Create improvement backlog

### 4. Deploy Enhancements (Quarterly)
- Implement prioritized improvements
- Deploy with A/B testing
- Monitor metrics closely
- Collect feedback on changes
- Iterate based on results

## Metrics to Track

### User Satisfaction
- Overall rating (target: > 4.0/5.0)
- Accuracy rating (target: > 90% "high" or "very_high")
- Latency acceptance (target: > 95%)
- Usefulness rating (target: > 85% "useful" or "very_useful")

### Performance
- Response time p95 (target: < 2500ms)
- Error rate (target: < 1%)
- Cost per query (target: < $0.08)
- Success rate (target: > 98%)

### Adoption
- Active users (monthly)
- Queries per user per month
- Retention rate
- Feature adoption rate

### Quality
- Accuracy vs. engineering calculations
- Consistency across similar queries
- Completeness of analysis
- Actionability of recommendations

## Feedback-Driven Improvements

### Past Improvements from Feedback

**Q4 2024**
- Added multi-fuel comparison based on user requests
- Improved solar fraction accuracy by 15% from user-reported discrepancies
- Reduced latency by 30% based on performance feedback
- Added storage sizing recommendations

**Q1 2025**
- Implemented economic analysis tools (NPV, IRR, payback)
- Added support for batch processes
- Enhanced temperature requirements database
- Improved error messages and validation

### Upcoming Improvements

**Q2 2025** (based on current feedback)
- Integration with live solar irradiance APIs
- Multi-site batch analysis
- Enhanced lifecycle assessment
- Real-time monitoring dashboards

## Contact

For questions about feedback collection or analysis:
- Product Team: product@greenlang.com
- Engineering Team: engineering@greenlang.com
- Data Analytics: analytics@greenlang.com

## Resources

- Agent Documentation: https://docs.greenlang.com/agents/industrial-process-heat
- API Reference: https://api.greenlang.com/docs
- User Forum: https://community.greenlang.com
- GitHub Issues: https://github.com/akshay-greenlang/Code-V1_GreenLang/issues
