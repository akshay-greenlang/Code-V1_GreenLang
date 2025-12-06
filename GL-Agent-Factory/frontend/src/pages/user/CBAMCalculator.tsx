/**
 * CBAMCalculator Page
 *
 * Carbon Border Adjustment Mechanism calculations.
 */

import * as React from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import {
  Calculator,
  Globe,
  Package,
  FileText,
  Download,
  History,
  Info,
  AlertTriangle,
  CheckCircle,
  TrendingUp,
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
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/Tabs';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/Table';
import { MetricCard, MetricGrid } from '@/components/widgets/MetricCard';
import { EmissionsByCategoryChart } from '@/components/charts/EmissionsChart';
import { useCalculateCBAM, useCBAMHistory } from '@/api/hooks';
import { formatEmissions, formatNumber, formatCurrency, formatDateTime, countryCodeToFlag } from '@/utils/format';
import { cn } from '@/utils/cn';
import type { CBAMCalculationResult } from '@/api/types';

// Form validation
const cbamFormSchema = z.object({
  productCategory: z.string().min(1, 'Select a product category'),
  cnCode: z.string().min(4, 'Enter a valid CN code'),
  weight: z.number().min(0.001, 'Weight must be greater than 0'),
  originCountry: z.string().min(2, 'Select origin country'),
  supplierName: z.string().optional(),
  installationId: z.string().optional(),
  specificEmissions: z.number().optional(),
  importDate: z.string().min(1, 'Import date is required'),
});

type CBAMFormData = z.infer<typeof cbamFormSchema>;

const productCategories = [
  { value: 'iron_steel', label: 'Iron and Steel', cnPrefix: '72' },
  { value: 'aluminum', label: 'Aluminum', cnPrefix: '76' },
  { value: 'cement', label: 'Cement', cnPrefix: '2523' },
  { value: 'fertilizers', label: 'Fertilizers', cnPrefix: '31' },
  { value: 'electricity', label: 'Electricity', cnPrefix: '2716' },
  { value: 'hydrogen', label: 'Hydrogen', cnPrefix: '2804' },
];

const countries = [
  { code: 'CN', name: 'China' },
  { code: 'IN', name: 'India' },
  { code: 'TR', name: 'Turkey' },
  { code: 'RU', name: 'Russia' },
  { code: 'UA', name: 'Ukraine' },
  { code: 'US', name: 'United States' },
  { code: 'BR', name: 'Brazil' },
  { code: 'ZA', name: 'South Africa' },
  { code: 'KR', name: 'South Korea' },
  { code: 'JP', name: 'Japan' },
];

export default function CBAMCalculator() {
  const [result, setResult] = React.useState<CBAMCalculationResult | null>(null);
  const [activeTab, setActiveTab] = React.useState('calculator');
  const [useSupplierData, setUseSupplierData] = React.useState(false);

  const calculateCBAM = useCalculateCBAM();
  const { data: historyResponse } = useCBAMHistory({ perPage: 10 });

  const form = useForm<CBAMFormData>({
    resolver: zodResolver(cbamFormSchema),
    defaultValues: {
      productCategory: '',
      cnCode: '',
      weight: 0,
      originCountry: '',
      supplierName: '',
      installationId: '',
      importDate: new Date().toISOString().split('T')[0],
    },
  });

  // Mock history
  const mockHistory: CBAMCalculationResult[] = [
    {
      id: '1',
      productCategory: 'iron_steel',
      cnCode: '7208',
      weight: 500,
      originCountry: 'CN',
      embeddedEmissions: { direct: 850, indirect: 150, total: 1000 },
      carbonPrice: { euEtsPrice: 85, originCountryPrice: 12, cbamAdjustment: 73 },
      dataQualityScore: 92,
      reportingPeriod: 'Q3 2024',
      calculatedAt: new Date(Date.now() - 1000 * 60 * 60 * 2).toISOString(),
    },
    {
      id: '2',
      productCategory: 'aluminum',
      cnCode: '7601',
      weight: 200,
      originCountry: 'IN',
      embeddedEmissions: { direct: 320, indirect: 180, total: 500 },
      carbonPrice: { euEtsPrice: 85, originCountryPrice: 5, cbamAdjustment: 80 },
      dataQualityScore: 85,
      reportingPeriod: 'Q3 2024',
      calculatedAt: new Date(Date.now() - 1000 * 60 * 60 * 24).toISOString(),
    },
    {
      id: '3',
      productCategory: 'cement',
      cnCode: '2523',
      weight: 1000,
      originCountry: 'TR',
      embeddedEmissions: { direct: 650, indirect: 50, total: 700 },
      carbonPrice: { euEtsPrice: 85, originCountryPrice: 8, cbamAdjustment: 77 },
      dataQualityScore: 95,
      reportingPeriod: 'Q3 2024',
      calculatedAt: new Date(Date.now() - 1000 * 60 * 60 * 48).toISOString(),
    },
  ];

  const history = historyResponse?.items || mockHistory;

  const onSubmit = (data: CBAMFormData) => {
    calculateCBAM.mutate(
      {
        productCategory: data.productCategory,
        cnCode: data.cnCode,
        weight: data.weight,
        originCountry: data.originCountry,
        supplierData: useSupplierData
          ? {
              name: data.supplierName || '',
              installationId: data.installationId,
              specificEmissions: data.specificEmissions,
            }
          : undefined,
        importDate: data.importDate,
      },
      {
        onSuccess: (response) => setResult(response),
      }
    );
  };

  // Summary stats
  const stats = React.useMemo(() => {
    const total = history.reduce((acc, h) => acc + h.embeddedEmissions.total, 0);
    const liability = history.reduce(
      (acc, h) => acc + h.embeddedEmissions.total * h.carbonPrice.cbamAdjustment,
      0
    );
    return {
      totalEmissions: total,
      totalLiability: liability,
      avgQuality: history.reduce((acc, h) => acc + h.dataQualityScore, 0) / history.length,
    };
  }, [history]);

  const displayResult = result;

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold">CBAM Calculator</h1>
          <p className="text-muted-foreground">
            Calculate Carbon Border Adjustment Mechanism liabilities for imports.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline">
            <FileText className="h-4 w-4 mr-2" />
            Generate Report
          </Button>
        </div>
      </div>

      {/* Summary metrics */}
      <MetricGrid columns={4}>
        <MetricCard
          title="Total Embedded Emissions"
          value={formatEmissions(stats.totalEmissions)}
          subtitle="This quarter"
          icon={<Globe className="h-5 w-5" />}
        />
        <MetricCard
          title="Estimated CBAM Liability"
          value={formatCurrency(stats.totalLiability, 'EUR')}
          subtitle="Based on current ETS price"
          icon={<TrendingUp className="h-5 w-5" />}
        />
        <MetricCard
          title="EU ETS Price"
          value={formatCurrency(85, 'EUR')}
          subtitle="Per tonne CO2"
          trend={{ value: 2.3, label: 'this week' }}
          icon={<TrendingUp className="h-5 w-5" />}
        />
        <MetricCard
          title="Data Quality Score"
          value={`${stats.avgQuality.toFixed(0)}%`}
          subtitle="Average across imports"
          icon={<CheckCircle className="h-5 w-5" />}
        />
      </MetricGrid>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="calculator">
            <Calculator className="h-4 w-4 mr-2" />
            Calculator
          </TabsTrigger>
          <TabsTrigger value="history">
            <History className="h-4 w-4 mr-2" />
            History
          </TabsTrigger>
          <TabsTrigger value="reports">
            <FileText className="h-4 w-4 mr-2" />
            Reports
          </TabsTrigger>
        </TabsList>

        <TabsContent value="calculator" className="space-y-6">
          <div className="grid gap-6 lg:grid-cols-2">
            {/* Input form */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Package className="h-5 w-5" />
                  Import Details
                </CardTitle>
                <CardDescription>
                  Enter the details of your import to calculate CBAM obligations.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
                  {/* Product Category */}
                  <div>
                    <label className="text-sm font-medium">Product Category</label>
                    <Select
                      value={form.watch('productCategory')}
                      onValueChange={(value) => {
                        form.setValue('productCategory', value);
                        const category = productCategories.find((c) => c.value === value);
                        if (category) {
                          form.setValue('cnCode', category.cnPrefix);
                        }
                      }}
                    >
                      <SelectTrigger className="mt-1.5">
                        <SelectValue placeholder="Select category" />
                      </SelectTrigger>
                      <SelectContent>
                        {productCategories.map((cat) => (
                          <SelectItem key={cat.value} value={cat.value}>
                            {cat.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  {/* CN Code */}
                  <Input
                    label="CN Code"
                    placeholder="e.g., 7208"
                    {...form.register('cnCode')}
                    error={form.formState.errors.cnCode?.message}
                    helperText="Combined Nomenclature code for the product"
                  />

                  {/* Weight and Origin */}
                  <div className="grid gap-4 sm:grid-cols-2">
                    <Input
                      label="Weight (tonnes)"
                      type="number"
                      step="0.001"
                      {...form.register('weight', { valueAsNumber: true })}
                      error={form.formState.errors.weight?.message}
                    />
                    <div>
                      <label className="text-sm font-medium">Origin Country</label>
                      <Select
                        value={form.watch('originCountry')}
                        onValueChange={(value) => form.setValue('originCountry', value)}
                      >
                        <SelectTrigger className="mt-1.5">
                          <SelectValue placeholder="Select country" />
                        </SelectTrigger>
                        <SelectContent>
                          {countries.map((country) => (
                            <SelectItem key={country.code} value={country.code}>
                              {countryCodeToFlag(country.code)} {country.name}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  {/* Import Date */}
                  <Input
                    label="Import Date"
                    type="date"
                    {...form.register('importDate')}
                    error={form.formState.errors.importDate?.message}
                  />

                  {/* Supplier Data Toggle */}
                  <div className="flex items-center justify-between p-4 border rounded-lg">
                    <div>
                      <p className="font-medium">Use Supplier-Specific Data</p>
                      <p className="text-sm text-muted-foreground">
                        If you have verified emission data from your supplier
                      </p>
                    </div>
                    <input
                      type="checkbox"
                      checked={useSupplierData}
                      onChange={(e) => setUseSupplierData(e.target.checked)}
                      className="h-5 w-5"
                    />
                  </div>

                  {/* Supplier fields */}
                  {useSupplierData && (
                    <div className="space-y-4 p-4 bg-muted/50 rounded-lg">
                      <Input
                        label="Supplier Name"
                        {...form.register('supplierName')}
                      />
                      <Input
                        label="Installation ID"
                        placeholder="EU ETS Installation ID"
                        {...form.register('installationId')}
                      />
                      <Input
                        label="Specific Emissions (tCO2e/t product)"
                        type="number"
                        step="0.001"
                        {...form.register('specificEmissions', { valueAsNumber: true })}
                      />
                    </div>
                  )}

                  <Button type="submit" className="w-full" loading={calculateCBAM.isPending}>
                    <Calculator className="h-4 w-4 mr-2" />
                    Calculate CBAM
                  </Button>
                </form>
              </CardContent>
            </Card>

            {/* Results */}
            <div className="space-y-6">
              {displayResult ? (
                <>
                  <Card className="border-blue-200 bg-blue-50/50">
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2 text-blue-800">
                        <Globe className="h-5 w-5" />
                        CBAM Calculation Results
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-6">
                      {/* Embedded Emissions */}
                      <div>
                        <h4 className="text-sm font-medium text-muted-foreground mb-3">
                          Embedded Emissions
                        </h4>
                        <div className="grid grid-cols-3 gap-4">
                          <div className="text-center p-3 bg-background rounded-lg">
                            <p className="text-xs text-muted-foreground">Direct</p>
                            <p className="font-semibold">
                              {formatEmissions(displayResult.embeddedEmissions.direct)}
                            </p>
                          </div>
                          <div className="text-center p-3 bg-background rounded-lg">
                            <p className="text-xs text-muted-foreground">Indirect</p>
                            <p className="font-semibold">
                              {formatEmissions(displayResult.embeddedEmissions.indirect)}
                            </p>
                          </div>
                          <div className="text-center p-3 bg-blue-100 rounded-lg">
                            <p className="text-xs text-blue-700">Total</p>
                            <p className="font-bold text-blue-800">
                              {formatEmissions(displayResult.embeddedEmissions.total)}
                            </p>
                          </div>
                        </div>
                      </div>

                      {/* Carbon Price */}
                      <div>
                        <h4 className="text-sm font-medium text-muted-foreground mb-3">
                          Carbon Price Adjustment
                        </h4>
                        <div className="space-y-2">
                          <div className="flex justify-between">
                            <span>EU ETS Price</span>
                            <span className="font-medium">
                              {formatCurrency(displayResult.carbonPrice.euEtsPrice, 'EUR')}/t
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span>Origin Country Carbon Price</span>
                            <span className="font-medium">
                              {formatCurrency(displayResult.carbonPrice.originCountryPrice, 'EUR')}/t
                            </span>
                          </div>
                          <div className="flex justify-between border-t pt-2">
                            <span className="font-medium">CBAM Adjustment</span>
                            <span className="font-bold text-blue-700">
                              {formatCurrency(displayResult.carbonPrice.cbamAdjustment, 'EUR')}/t
                            </span>
                          </div>
                        </div>
                      </div>

                      {/* Estimated Liability */}
                      <div className="p-4 bg-blue-100 rounded-lg">
                        <p className="text-sm text-blue-700">Estimated CBAM Liability</p>
                        <p className="text-2xl font-bold text-blue-800">
                          {formatCurrency(
                            displayResult.embeddedEmissions.total *
                              displayResult.carbonPrice.cbamAdjustment,
                            'EUR'
                          )}
                        </p>
                      </div>

                      {/* Data Quality */}
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-muted-foreground">Data Quality Score</span>
                        <Badge
                          variant={
                            displayResult.dataQualityScore >= 90
                              ? 'success'
                              : displayResult.dataQualityScore >= 70
                                ? 'warning'
                                : 'destructive'
                          }
                        >
                          {displayResult.dataQualityScore}%
                        </Badge>
                      </div>
                    </CardContent>
                  </Card>

                  <div className="flex gap-2">
                    <Button variant="outline" className="flex-1">
                      <Download className="h-4 w-4 mr-2" />
                      Export
                    </Button>
                    <Button className="flex-1">Add to Quarterly Report</Button>
                  </div>
                </>
              ) : (
                <Card className="flex flex-col items-center justify-center p-12 text-center">
                  <Globe className="h-12 w-12 text-muted-foreground mb-4" />
                  <h3 className="font-semibold">Calculate Your CBAM</h3>
                  <p className="text-sm text-muted-foreground max-w-sm mt-2">
                    Enter your import details to calculate embedded emissions and CBAM obligations.
                  </p>
                </Card>
              )}

              {/* Info card */}
              <Card>
                <CardContent className="p-4">
                  <div className="flex gap-3">
                    <Info className="h-5 w-5 text-blue-500 flex-shrink-0 mt-0.5" />
                    <div className="text-sm">
                      <p className="font-medium">About CBAM</p>
                      <p className="text-muted-foreground mt-1">
                        The Carbon Border Adjustment Mechanism applies to imports of cement, iron
                        and steel, aluminum, fertilizers, electricity, and hydrogen. The
                        transitional period runs until December 2025.
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="history">
          <Card>
            <CardHeader>
              <CardTitle>Calculation History</CardTitle>
              <CardDescription>Your previous CBAM calculations</CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Product</TableHead>
                    <TableHead>Origin</TableHead>
                    <TableHead className="text-right">Weight</TableHead>
                    <TableHead className="text-right">Emissions</TableHead>
                    <TableHead className="text-right">CBAM Liability</TableHead>
                    <TableHead>Quality</TableHead>
                    <TableHead>Date</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {history.map((item) => (
                    <TableRow key={item.id}>
                      <TableCell>
                        <div>
                          <p className="font-medium capitalize">
                            {item.productCategory.replace('_', ' ')}
                          </p>
                          <p className="text-xs text-muted-foreground">CN {item.cnCode}</p>
                        </div>
                      </TableCell>
                      <TableCell>
                        {countryCodeToFlag(item.originCountry)} {item.originCountry}
                      </TableCell>
                      <TableCell className="text-right">
                        {formatNumber(item.weight)} t
                      </TableCell>
                      <TableCell className="text-right font-medium">
                        {formatEmissions(item.embeddedEmissions.total)}
                      </TableCell>
                      <TableCell className="text-right font-medium text-blue-600">
                        {formatCurrency(
                          item.embeddedEmissions.total * item.carbonPrice.cbamAdjustment,
                          'EUR'
                        )}
                      </TableCell>
                      <TableCell>
                        <Badge
                          variant={
                            item.dataQualityScore >= 90
                              ? 'success'
                              : item.dataQualityScore >= 70
                                ? 'warning'
                                : 'destructive'
                          }
                        >
                          {item.dataQualityScore}%
                        </Badge>
                      </TableCell>
                      <TableCell className="text-muted-foreground">
                        {formatDateTime(item.calculatedAt)}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="reports">
          <Card>
            <CardHeader>
              <CardTitle>CBAM Quarterly Reports</CardTitle>
              <CardDescription>
                Generate and submit quarterly CBAM reports to the European Commission.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                <Card variant="interactive">
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between mb-4">
                      <Badge variant="success">Submitted</Badge>
                      <FileText className="h-5 w-5 text-muted-foreground" />
                    </div>
                    <h3 className="font-semibold">Q2 2024</h3>
                    <p className="text-sm text-muted-foreground">Submitted July 31, 2024</p>
                    <div className="mt-4 pt-4 border-t">
                      <div className="flex justify-between text-sm">
                        <span>Total Emissions</span>
                        <span className="font-medium">2,450 tCO2e</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
                <Card variant="interactive">
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between mb-4">
                      <Badge variant="warning">In Progress</Badge>
                      <FileText className="h-5 w-5 text-muted-foreground" />
                    </div>
                    <h3 className="font-semibold">Q3 2024</h3>
                    <p className="text-sm text-muted-foreground">Due October 31, 2024</p>
                    <div className="mt-4 pt-4 border-t">
                      <div className="flex justify-between text-sm">
                        <span>Total Emissions</span>
                        <span className="font-medium">2,200 tCO2e</span>
                      </div>
                    </div>
                    <Button size="sm" className="w-full mt-4">
                      Continue Report
                    </Button>
                  </CardContent>
                </Card>
                <Card variant="interactive" className="border-dashed">
                  <CardContent className="p-6 flex flex-col items-center justify-center h-full text-center">
                    <FileText className="h-8 w-8 text-muted-foreground mb-2" />
                    <h3 className="font-semibold">Q4 2024</h3>
                    <p className="text-sm text-muted-foreground">
                      Report will be available starting October 1
                    </p>
                  </CardContent>
                </Card>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
