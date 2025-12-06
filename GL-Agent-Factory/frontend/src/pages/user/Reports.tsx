/**
 * Reports Page
 *
 * Report generation and history.
 */

import * as React from 'react';
import {
  FileText,
  Download,
  Plus,
  Search,
  Filter,
  Clock,
  CheckCircle,
  XCircle,
  Loader2,
  Trash2,
  Eye,
  Calendar,
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Badge } from '@/components/ui/Badge';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/Select';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/Dialog';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  TableEmpty,
} from '@/components/ui/Table';
import { Pagination, PaginationInfo } from '@/components/ui/Pagination';
import { useReports, useGenerateReport, useDeleteReport } from '@/api/hooks';
import { formatDateTime, formatRelativeTime, formatFileSize } from '@/utils/format';
import type { Report, ReportStatus } from '@/api/types';

const reportTypes = [
  { value: 'fuel', label: 'Fuel Emissions Report' },
  { value: 'cbam', label: 'CBAM Quarterly Report' },
  { value: 'building', label: 'Building Energy Report' },
  { value: 'eudr', label: 'EUDR Compliance Report' },
  { value: 'comprehensive', label: 'Comprehensive Annual Report' },
];

const formatOptions = [
  { value: 'pdf', label: 'PDF Document' },
  { value: 'excel', label: 'Excel Spreadsheet' },
  { value: 'json', label: 'JSON Data' },
  { value: 'xml', label: 'XML (EU Standard)' },
];

const statusConfig: Record<ReportStatus, { icon: typeof CheckCircle; color: string; label: string }> = {
  pending: { icon: Clock, color: 'text-gray-500', label: 'Pending' },
  processing: { icon: Loader2, color: 'text-blue-500', label: 'Processing' },
  completed: { icon: CheckCircle, color: 'text-greenlang-500', label: 'Completed' },
  failed: { icon: XCircle, color: 'text-red-500', label: 'Failed' },
};

export default function Reports() {
  const [search, setSearch] = React.useState('');
  const [typeFilter, setTypeFilter] = React.useState('all');
  const [statusFilter, setStatusFilter] = React.useState('all');
  const [page, setPage] = React.useState(1);
  const [showCreateDialog, setShowCreateDialog] = React.useState(false);
  const perPage = 10;

  // Form state for new report
  const [newReport, setNewReport] = React.useState({
    type: 'fuel' as Report['type'],
    name: '',
    format: 'pdf' as Report['format'],
    startDate: new Date(new Date().setMonth(new Date().getMonth() - 3)).toISOString().split('T')[0],
    endDate: new Date().toISOString().split('T')[0],
  });

  // Fetch data
  const { data: response, isLoading } = useReports({
    type: typeFilter !== 'all' ? typeFilter : undefined,
    status: statusFilter !== 'all' ? statusFilter : undefined,
    page,
    perPage,
  });

  const generateReport = useGenerateReport();
  const deleteReport = useDeleteReport();

  // Mock data
  const mockReports: Report[] = [
    {
      id: '1',
      type: 'cbam',
      name: 'Q2 2024 CBAM Report',
      status: 'completed',
      format: 'pdf',
      fileSize: 2450000,
      downloadUrl: '/api/reports/1/download',
      parameters: { quarter: 'Q2', year: 2024 },
      createdAt: new Date(Date.now() - 1000 * 60 * 60 * 24 * 5).toISOString(),
      completedAt: new Date(Date.now() - 1000 * 60 * 60 * 24 * 5 + 1000 * 60 * 15).toISOString(),
    },
    {
      id: '2',
      type: 'fuel',
      name: 'June 2024 Fuel Emissions',
      status: 'completed',
      format: 'excel',
      fileSize: 1850000,
      downloadUrl: '/api/reports/2/download',
      parameters: { month: 'June', year: 2024 },
      createdAt: new Date(Date.now() - 1000 * 60 * 60 * 24 * 2).toISOString(),
      completedAt: new Date(Date.now() - 1000 * 60 * 60 * 24 * 2 + 1000 * 60 * 8).toISOString(),
    },
    {
      id: '3',
      type: 'building',
      name: 'HQ Building Analysis',
      status: 'processing',
      format: 'pdf',
      parameters: { buildingId: 'HQ-01' },
      createdAt: new Date(Date.now() - 1000 * 60 * 30).toISOString(),
    },
    {
      id: '4',
      type: 'eudr',
      name: 'Supply Chain EUDR Check',
      status: 'pending',
      format: 'json',
      parameters: { suppliers: 15 },
      createdAt: new Date(Date.now() - 1000 * 60 * 5).toISOString(),
    },
    {
      id: '5',
      type: 'comprehensive',
      name: '2023 Annual Sustainability Report',
      status: 'completed',
      format: 'pdf',
      fileSize: 15600000,
      downloadUrl: '/api/reports/5/download',
      parameters: { year: 2023 },
      createdAt: new Date(Date.now() - 1000 * 60 * 60 * 24 * 30).toISOString(),
      completedAt: new Date(Date.now() - 1000 * 60 * 60 * 24 * 30 + 1000 * 60 * 45).toISOString(),
    },
  ];

  const reports = response?.items || mockReports;
  const totalItems = response?.pagination?.totalItems || mockReports.length;
  const totalPages = response?.pagination?.totalPages || 1;

  // Filter reports
  const filteredReports = React.useMemo(() => {
    if (!search) return reports;
    const searchLower = search.toLowerCase();
    return reports.filter((report) =>
      report.name.toLowerCase().includes(searchLower)
    );
  }, [reports, search]);

  const handleCreateReport = () => {
    generateReport.mutate(
      {
        type: newReport.type,
        name: newReport.name,
        format: newReport.format,
        dateRange: {
          startDate: newReport.startDate,
          endDate: newReport.endDate,
        },
        parameters: {},
      },
      {
        onSuccess: () => {
          setShowCreateDialog(false);
          setNewReport({
            type: 'fuel',
            name: '',
            format: 'pdf',
            startDate: new Date(new Date().setMonth(new Date().getMonth() - 3)).toISOString().split('T')[0],
            endDate: new Date().toISOString().split('T')[0],
          });
        },
      }
    );
  };

  const handleDelete = (reportId: string) => {
    if (confirm('Are you sure you want to delete this report?')) {
      deleteReport.mutate(reportId);
    }
  };

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold">Reports</h1>
          <p className="text-muted-foreground">
            Generate and download compliance and emissions reports.
          </p>
        </div>
        <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
          <DialogTrigger asChild>
            <Button>
              <Plus className="h-4 w-4 mr-2" />
              Generate Report
            </Button>
          </DialogTrigger>
          <DialogContent className="sm:max-w-md">
            <DialogHeader>
              <DialogTitle>Generate New Report</DialogTitle>
              <DialogDescription>
                Configure and generate a new report from your data.
              </DialogDescription>
            </DialogHeader>

            <div className="space-y-4 py-4">
              <div>
                <label className="text-sm font-medium">Report Type</label>
                <Select
                  value={newReport.type}
                  onValueChange={(value) => setNewReport({ ...newReport, type: value as Report['type'] })}
                >
                  <SelectTrigger className="mt-1.5">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {reportTypes.map((type) => (
                      <SelectItem key={type.value} value={type.value}>
                        {type.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <Input
                label="Report Name"
                value={newReport.name}
                onChange={(e) => setNewReport({ ...newReport, name: e.target.value })}
                placeholder="e.g., Q3 2024 Emissions Report"
              />

              <div>
                <label className="text-sm font-medium">Output Format</label>
                <Select
                  value={newReport.format}
                  onValueChange={(value) => setNewReport({ ...newReport, format: value as Report['format'] })}
                >
                  <SelectTrigger className="mt-1.5">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {formatOptions.map((format) => (
                      <SelectItem key={format.value} value={format.value}>
                        {format.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="grid gap-4 sm:grid-cols-2">
                <Input
                  label="Start Date"
                  type="date"
                  value={newReport.startDate}
                  onChange={(e) => setNewReport({ ...newReport, startDate: e.target.value })}
                />
                <Input
                  label="End Date"
                  type="date"
                  value={newReport.endDate}
                  onChange={(e) => setNewReport({ ...newReport, endDate: e.target.value })}
                />
              </div>
            </div>

            <DialogFooter>
              <Button variant="outline" onClick={() => setShowCreateDialog(false)}>
                Cancel
              </Button>
              <Button onClick={handleCreateReport} loading={generateReport.isPending}>
                Generate Report
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {/* Quick stats */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Card className="p-4">
          <div className="flex items-center gap-3">
            <div className="rounded-lg bg-blue-100 p-2 text-blue-600">
              <FileText className="h-5 w-5" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Total Reports</p>
              <p className="text-xl font-bold">{totalItems}</p>
            </div>
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center gap-3">
            <div className="rounded-lg bg-greenlang-100 p-2 text-greenlang-600">
              <CheckCircle className="h-5 w-5" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Completed</p>
              <p className="text-xl font-bold">
                {reports.filter((r) => r.status === 'completed').length}
              </p>
            </div>
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center gap-3">
            <div className="rounded-lg bg-amber-100 p-2 text-amber-600">
              <Loader2 className="h-5 w-5" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">In Progress</p>
              <p className="text-xl font-bold">
                {reports.filter((r) => r.status === 'processing' || r.status === 'pending').length}
              </p>
            </div>
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center gap-3">
            <div className="rounded-lg bg-purple-100 p-2 text-purple-600">
              <Calendar className="h-5 w-5" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">This Month</p>
              <p className="text-xl font-bold">
                {reports.filter((r) => {
                  const created = new Date(r.createdAt);
                  const now = new Date();
                  return created.getMonth() === now.getMonth() && created.getFullYear() === now.getFullYear();
                }).length}
              </p>
            </div>
          </div>
        </Card>
      </div>

      {/* Filters */}
      <Card>
        <CardContent className="p-4">
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="flex-1">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  placeholder="Search reports..."
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  className="pl-9"
                />
              </div>
            </div>

            <Select value={typeFilter} onValueChange={setTypeFilter}>
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="All Types" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Types</SelectItem>
                {reportTypes.map((type) => (
                  <SelectItem key={type.value} value={type.value}>
                    {type.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            <Select value={statusFilter} onValueChange={setStatusFilter}>
              <SelectTrigger className="w-[140px]">
                <SelectValue placeholder="All Status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Status</SelectItem>
                <SelectItem value="pending">Pending</SelectItem>
                <SelectItem value="processing">Processing</SelectItem>
                <SelectItem value="completed">Completed</SelectItem>
                <SelectItem value="failed">Failed</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Reports table */}
      <Card>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Report</TableHead>
              <TableHead>Type</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Format</TableHead>
              <TableHead>Size</TableHead>
              <TableHead>Created</TableHead>
              <TableHead className="w-[120px]"></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filteredReports.length === 0 ? (
              <TableEmpty
                icon={<FileText className="h-12 w-12" />}
                title="No reports found"
                description={search ? 'Try adjusting your search' : 'Generate your first report'}
                action={
                  <Button size="sm" onClick={() => setShowCreateDialog(true)}>
                    <Plus className="h-4 w-4 mr-2" />
                    Generate Report
                  </Button>
                }
              />
            ) : (
              filteredReports.map((report) => {
                const status = statusConfig[report.status];
                const StatusIcon = status.icon;

                return (
                  <TableRow key={report.id}>
                    <TableCell>
                      <div className="flex items-center gap-3">
                        <div className="rounded-lg bg-muted p-2">
                          <FileText className="h-4 w-4 text-muted-foreground" />
                        </div>
                        <div>
                          <p className="font-medium">{report.name}</p>
                          <p className="text-xs text-muted-foreground">
                            {report.completedAt && `Completed ${formatRelativeTime(report.completedAt)}`}
                          </p>
                        </div>
                      </div>
                    </TableCell>
                    <TableCell>
                      <Badge variant="secondary">
                        {reportTypes.find((t) => t.value === report.type)?.label || report.type}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <StatusIcon
                          className={`h-4 w-4 ${status.color} ${report.status === 'processing' ? 'animate-spin' : ''}`}
                        />
                        <span>{status.label}</span>
                      </div>
                    </TableCell>
                    <TableCell>
                      <Badge variant="outline">{report.format.toUpperCase()}</Badge>
                    </TableCell>
                    <TableCell>
                      {report.fileSize ? formatFileSize(report.fileSize) : '-'}
                    </TableCell>
                    <TableCell className="text-muted-foreground">
                      {formatDateTime(report.createdAt)}
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center justify-end gap-1">
                        {report.status === 'completed' && (
                          <>
                            <Button variant="ghost" size="icon-sm" title="Preview">
                              <Eye className="h-4 w-4" />
                            </Button>
                            <Button variant="ghost" size="icon-sm" title="Download">
                              <Download className="h-4 w-4" />
                            </Button>
                          </>
                        )}
                        <Button
                          variant="ghost"
                          size="icon-sm"
                          title="Delete"
                          onClick={() => handleDelete(report.id)}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                );
              })
            )}
          </TableBody>
        </Table>
      </Card>

      {/* Pagination */}
      {filteredReports.length > 0 && (
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
          <PaginationInfo
            currentPage={page}
            pageSize={perPage}
            totalItems={totalItems}
          />
          <Pagination
            currentPage={page}
            totalPages={totalPages}
            onPageChange={setPage}
          />
        </div>
      )}

      {/* Report templates */}
      <Card>
        <CardHeader>
          <CardTitle>Report Templates</CardTitle>
          <CardDescription>Quick access to common report configurations</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {[
              { name: 'CBAM Quarterly Report', type: 'cbam', description: 'EU Carbon Border Adjustment Mechanism' },
              { name: 'Monthly Emissions Summary', type: 'fuel', description: 'Scope 1, 2, 3 emissions breakdown' },
              { name: 'EUDR Compliance Report', type: 'eudr', description: 'Deforestation regulation compliance' },
            ].map((template) => (
              <Card
                key={template.name}
                variant="interactive"
                onClick={() => {
                  setNewReport({ ...newReport, type: template.type as Report['type'], name: template.name });
                  setShowCreateDialog(true);
                }}
              >
                <CardContent className="p-4">
                  <div className="flex items-center gap-3">
                    <div className="rounded-lg bg-primary/10 p-2 text-primary">
                      <FileText className="h-5 w-5" />
                    </div>
                    <div>
                      <p className="font-medium">{template.name}</p>
                      <p className="text-sm text-muted-foreground">{template.description}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
