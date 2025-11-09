#!/bin/bash
# GreenLang Monitoring System Deployment Script
# ==============================================

set -e

echo "==============================================="
echo "GreenLang Monitoring System Deployment"
echo "==============================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Check configuration
echo -e "${YELLOW}Checking configuration...${NC}"
if [ ! -f "config.yaml" ]; then
    echo -e "${RED}✗ config.yaml not found${NC}"
    echo "Please create config.yaml from config.yaml.example"
    exit 1
fi
echo -e "${GREEN}✓ Configuration found${NC}"

# Check environment variables
echo -e "${YELLOW}Checking environment variables...${NC}"
missing_vars=()

if [ -z "$GRAFANA_API_KEY" ]; then
    missing_vars+=("GRAFANA_API_KEY")
fi

if [ -z "$SLACK_WEBHOOK_URL" ]; then
    missing_vars+=("SLACK_WEBHOOK_URL")
fi

if [ -z "$API_KEY_PROD" ]; then
    missing_vars+=("API_KEY_PROD")
fi

if [ ${#missing_vars[@]} -gt 0 ]; then
    echo -e "${YELLOW}⚠ Missing environment variables: ${missing_vars[*]}${NC}"
    echo "Set them in your environment or .env file"
else
    echo -e "${GREEN}✓ All required environment variables set${NC}"
fi

# Create output directories
echo -e "${YELLOW}Creating output directories...${NC}"
mkdir -p reports/output
mkdir -p logs
echo -e "${GREEN}✓ Directories created${NC}"

# Generate dashboards
echo -e "${YELLOW}Generating Grafana dashboards...${NC}"
python dashboards/infrastructure_usage.py
python dashboards/cost_savings.py
python dashboards/performance.py
python dashboards/compliance.py
python dashboards/productivity.py
python dashboards/health.py
echo -e "${GREEN}✓ Dashboards generated${NC}"

# Generate alert rules
echo -e "${YELLOW}Generating alert rules...${NC}"
python alerts/alert_rules.py
echo -e "${GREEN}✓ Alert rules generated${NC}"

# Test API
echo -e "${YELLOW}Testing Analytics API...${NC}"
python -c "
from api.analytics_api import app
print('API initialized successfully')
"
echo -e "${GREEN}✓ API tested${NC}"

# Test collectors
echo -e "${YELLOW}Testing data collectors...${NC}"
python -c "
from collectors.metrics_collector import MetricsCollector
from collectors.health_checker import HealthChecker
print('Collectors initialized successfully')
"
echo -e "${GREEN}✓ Collectors tested${NC}"

# Run initial monitoring cycle
echo -e "${YELLOW}Running initial monitoring cycle...${NC}"
python orchestrator.py --run-once
echo -e "${GREEN}✓ Initial monitoring cycle completed${NC}"

# Setup cron jobs (optional)
echo -e "${YELLOW}Setting up cron jobs...${NC}"
cat > /tmp/greenlang_monitoring_cron.txt << 'EOF'
# GreenLang Monitoring Cron Jobs
# ================================

# Metrics collection (every 5 minutes)
*/5 * * * * cd /path/to/greenlang && python monitoring/collectors/metrics_collector.py >> logs/metrics_collector.log 2>&1

# Log aggregation (every 5 minutes)
*/5 * * * * cd /path/to/greenlang && python monitoring/collectors/log_aggregator.py >> logs/log_aggregator.log 2>&1

# Health checks (every minute)
* * * * * cd /path/to/greenlang && python monitoring/collectors/health_checker.py >> logs/health_checker.log 2>&1

# Violation scanning (every hour)
0 * * * * cd /path/to/greenlang && python monitoring/collectors/violation_scanner.py >> logs/violation_scanner.log 2>&1

# Weekly report (Monday 9 AM)
0 9 * * 1 cd /path/to/greenlang && python monitoring/orchestrator.py --weekly-report >> logs/reports.log 2>&1

# Monthly report (1st of month, 9 AM)
0 9 1 * * cd /path/to/greenlang && python monitoring/orchestrator.py --monthly-report >> logs/reports.log 2>&1
EOF

echo -e "${YELLOW}Cron jobs configuration saved to: /tmp/greenlang_monitoring_cron.txt${NC}"
echo "To install, run: crontab -e and paste the contents"

# Print summary
echo ""
echo "==============================================="
echo -e "${GREEN}Deployment Summary${NC}"
echo "==============================================="
echo "✓ Dependencies installed"
echo "✓ Configuration validated"
echo "✓ Dashboards generated (6 dashboards)"
echo "✓ Alert rules generated (15+ rules)"
echo "✓ API tested"
echo "✓ Collectors tested"
echo "✓ Initial monitoring cycle completed"
echo ""
echo "Next Steps:"
echo "1. Start the Analytics API: python api/analytics_api.py"
echo "2. Import dashboards to Grafana: http://localhost:3000"
echo "3. Setup cron jobs (see /tmp/greenlang_monitoring_cron.txt)"
echo "4. Configure alerting channels (Slack, Email, PagerDuty)"
echo ""
echo "Dashboard URLs (after Grafana import):"
echo "- Infrastructure Usage: http://localhost:3000/d/greenlang-ium"
echo "- Cost Savings: http://localhost:3000/d/greenlang-cost-savings"
echo "- Performance: http://localhost:3000/d/greenlang-performance"
echo "- Compliance: http://localhost:3000/d/greenlang-compliance"
echo "- Productivity: http://localhost:3000/d/greenlang-productivity"
echo "- Health: http://localhost:3000/d/greenlang-health"
echo ""
echo "==============================================="
echo -e "${GREEN}Deployment Complete!${NC}"
echo "==============================================="
