/**
 * User Home Page
 *
 * Welcome page with quick actions and recent activity.
 */

import * as React from 'react';
import { Link } from 'react-router-dom';
import {
  Fuel,
  Calculator,
  Building,
  TreePine,
  FileText,
  ArrowRight,
  Activity,
  TrendingDown,
  Clock,
  CheckCircle,
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { MetricCard, MetricGrid } from '@/components/widgets/MetricCard';
import { EmissionsTrendChart, EmissionsByCategoryChart } from '@/components/charts/EmissionsChart';
import { useAuthStore } from '@/stores/authStore';
import { useDashboardMetrics, useReports } from '@/api/hooks';
import { formatEmissions, formatNumber, formatRelativeTime } from '@/utils/format';

const quickActions = [
  {
    title: 'Fuel Emissions',
    description: 'Calculate emissions from fuel consumption',
    href: '/fuel-analyzer',
    icon: Fuel,
    color: 'bg-amber-100 text-amber-600',
  },
  {
    title: 'CBAM Calculator',
    description: 'Carbon border adjustment calculations',
    href: '/cbam-calculator',
    icon: Calculator,
    color: 'bg-blue-100 text-blue-600',
  },
  {
    title: 'Building Energy',
    description: 'Analyze building energy consumption',
    href: '/building-energy',
    icon: Building,
    color: 'bg-purple-100 text-purple-600',
  },
  {
    title: 'EUDR Compliance',
    description: 'Check deforestation regulation compliance',
    href: '/eudr-compliance',
    icon: TreePine,
    color: 'bg-greenlang-100 text-greenlang-600',
  },
];

export default function Home() {
  const { user } = useAuthStore();
  const { data: metrics, isLoading: metricsLoading } = useDashboardMetrics();
  const { data: reportsResponse } = useReports({ perPage: 5 });

  // Mock data for development
  const mockMetrics = {
    totalEmissions: 45230,
    emissionsTrend: -8.5,
    totalCalculations: 1234,
    complianceRate: 96.5,
    emissionsByCategory: [
      { category: 'Transportation', value: 18500, percentage: 41 },
      { category: 'Manufacturing', value: 12400, percentage: 27 },
      { category: 'Buildings', value: 8900, percentage: 20 },
      { category: 'Other', value: 5430, percentage: 12 },
    ],
    emissionsTrendData: Array.from({ length: 12 }, (_, i) => ({
      date: new Date(2024, i, 1).toLocaleDateString('en-US', { month: 'short' }),
      emissions: Math.floor(3500 + Math.random() * 2000),
    })),
  };

  const mockRecentActivity = [
    {
      id: '1',
      type: 'calculation',
      title: 'Fuel Emissions Calculated',
      description: 'Diesel consumption for Q2 2024',
      timestamp: new Date(Date.now() - 1000 * 60 * 30).toISOString(),
      result: '12.5 tCO2e',
    },
    {
      id: '2',
      type: 'report',
      title: 'CBAM Report Generated',
      description: 'Quarterly CBAM submission report',
      timestamp: new Date(Date.now() - 1000 * 60 * 60 * 2).toISOString(),
      status: 'completed',
    },
    {
      id: '3',
      type: 'compliance',
      title: 'EUDR Check Completed',
      description: 'Palm oil supply chain verification',
      timestamp: new Date(Date.now() - 1000 * 60 * 60 * 5).toISOString(),
      result: 'Compliant',
    },
    {
      id: '4',
      type: 'calculation',
      title: 'Building Energy Analysis',
      description: 'HQ Office Building - June 2024',
      timestamp: new Date(Date.now() - 1000 * 60 * 60 * 24).toISOString(),
      result: '85.2 kgCO2e/m2',
    },
  ];

  const displayMetrics = metrics || mockMetrics;

  return (
    <div className="space-y-8">
      {/* Welcome section */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold">
            Welcome back, {user?.firstName || 'User'}
          </h1>
          <p className="text-muted-foreground">
            Here's an overview of your carbon footprint and compliance status.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" asChild>
            <Link to="/reports">
              <FileText className="h-4 w-4 mr-2" />
              View Reports
            </Link>
          </Button>
          <Button asChild>
            <Link to="/fuel-analyzer">
              New Calculation
            </Link>
          </Button>
        </div>
      </div>

      {/* Key metrics */}
      <MetricGrid columns={4}>
        <MetricCard
          title="Total Emissions (YTD)"
          value={formatEmissions(displayMetrics.totalEmissions)}
          trend={{ value: displayMetrics.emissionsTrend, label: 'vs last year', isPositiveGood: false }}
          icon={<Activity className="h-5 w-5" />}
          loading={metricsLoading}
        />
        <MetricCard
          title="Calculations"
          value={formatNumber(displayMetrics.totalCalculations)}
          subtitle="This month"
          icon={<Calculator className="h-5 w-5" />}
          loading={metricsLoading}
        />
        <MetricCard
          title="Emission Reduction"
          value="-8.5%"
          trend={{ value: 8.5, label: 'on track' }}
          icon={<TrendingDown className="h-5 w-5" />}
          loading={metricsLoading}
        />
        <MetricCard
          title="Compliance Rate"
          value={`${displayMetrics.complianceRate}%`}
          subtitle="All regulations"
          icon={<CheckCircle className="h-5 w-5" />}
          loading={metricsLoading}
        />
      </MetricGrid>

      {/* Quick actions */}
      <div>
        <h2 className="text-lg font-semibold mb-4">Quick Actions</h2>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {quickActions.map((action) => (
            <Link key={action.href} to={action.href}>
              <Card variant="interactive" className="h-full">
                <CardContent className="p-6">
                  <div className={`rounded-lg p-3 w-fit ${action.color}`}>
                    <action.icon className="h-6 w-6" />
                  </div>
                  <h3 className="font-semibold mt-4">{action.title}</h3>
                  <p className="text-sm text-muted-foreground mt-1">
                    {action.description}
                  </p>
                  <div className="flex items-center gap-1 mt-4 text-primary text-sm font-medium">
                    Get started <ArrowRight className="h-4 w-4" />
                  </div>
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>
      </div>

      {/* Charts and activity */}
      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <EmissionsTrendChart
            data={displayMetrics.emissionsTrendData}
            title="Your Emissions Trend"
            description="Monthly emissions over the past year"
            loading={metricsLoading}
          />
        </div>
        <EmissionsByCategoryChart
          data={displayMetrics.emissionsByCategory}
          title="By Source"
          description="Emissions breakdown"
          loading={metricsLoading}
          height={280}
        />
      </div>

      {/* Recent activity */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <div>
            <CardTitle>Recent Activity</CardTitle>
            <CardDescription>Your latest calculations and reports</CardDescription>
          </div>
          <Button variant="ghost" size="sm" asChild>
            <Link to="/reports" className="flex items-center gap-1">
              View All <ArrowRight className="h-4 w-4" />
            </Link>
          </Button>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {mockRecentActivity.map((activity) => (
              <div
                key={activity.id}
                className="flex items-start gap-4 pb-4 border-b last:border-0 last:pb-0"
              >
                <div
                  className={`rounded-full p-2 ${
                    activity.type === 'calculation'
                      ? 'bg-blue-100 text-blue-600'
                      : activity.type === 'report'
                        ? 'bg-purple-100 text-purple-600'
                        : 'bg-greenlang-100 text-greenlang-600'
                  }`}
                >
                  {activity.type === 'calculation' && <Calculator className="h-4 w-4" />}
                  {activity.type === 'report' && <FileText className="h-4 w-4" />}
                  {activity.type === 'compliance' && <CheckCircle className="h-4 w-4" />}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="font-medium">{activity.title}</p>
                  <p className="text-sm text-muted-foreground truncate">
                    {activity.description}
                  </p>
                </div>
                <div className="text-right">
                  {activity.result && (
                    <Badge variant={activity.result === 'Compliant' ? 'success' : 'secondary'}>
                      {activity.result}
                    </Badge>
                  )}
                  {activity.status && (
                    <Badge variant="success">{activity.status}</Badge>
                  )}
                  <p className="text-xs text-muted-foreground mt-1">
                    {formatRelativeTime(activity.timestamp)}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Tips and resources */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        <Card className="bg-greenlang-50 border-greenlang-200">
          <CardContent className="p-6">
            <h3 className="font-semibold text-greenlang-800">Reduce Your Footprint</h3>
            <p className="text-sm text-greenlang-700 mt-2">
              Based on your data, switching to renewable energy could reduce your emissions by 25%.
            </p>
            <Button variant="outline" size="sm" className="mt-4">
              Learn More
            </Button>
          </CardContent>
        </Card>
        <Card className="bg-blue-50 border-blue-200">
          <CardContent className="p-6">
            <h3 className="font-semibold text-blue-800">CBAM Deadline Approaching</h3>
            <p className="text-sm text-blue-700 mt-2">
              Q3 2024 CBAM report due in 45 days. Make sure all your import data is up to date.
            </p>
            <Button variant="outline" size="sm" className="mt-4" asChild>
              <Link to="/cbam-calculator">Start Report</Link>
            </Button>
          </CardContent>
        </Card>
        <Card className="bg-purple-50 border-purple-200">
          <CardContent className="p-6">
            <h3 className="font-semibold text-purple-800">New Feature: EUDR</h3>
            <p className="text-sm text-purple-700 mt-2">
              EU Deforestation Regulation compliance checker now available with satellite analysis.
            </p>
            <Button variant="outline" size="sm" className="mt-4" asChild>
              <Link to="/eudr-compliance">Try Now</Link>
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
