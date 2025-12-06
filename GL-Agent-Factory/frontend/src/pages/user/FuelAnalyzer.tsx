/**
 * FuelAnalyzer Page
 *
 * Calculate emissions from fuel consumption.
 */

import * as React from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import {
  Fuel,
  Calculator,
  Info,
  Download,
  History,
  ArrowRight,
  Leaf,
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
import { EmissionsByCategoryChart } from '@/components/charts/EmissionsChart';
import { useCalculateFuel, useFuelHistory } from '@/api/hooks';
import { formatEmissions, formatNumber, formatDateTime } from '@/utils/format';
import type { FuelAnalysisResult } from '@/api/types';

// Form validation schema
const fuelFormSchema = z.object({
  fuelType: z.string().min(1, 'Please select a fuel type'),
  quantity: z.number().min(0.01, 'Quantity must be greater than 0'),
  unit: z.enum(['liters', 'gallons', 'kg', 'tonnes']),
  scope: z.number().min(1).max(3),
  location: z.string().optional(),
  startDate: z.string().min(1, 'Start date is required'),
  endDate: z.string().min(1, 'End date is required'),
});

type FuelFormData = z.infer<typeof fuelFormSchema>;

const fuelTypes = [
  { value: 'diesel', label: 'Diesel', factor: 2.68 },
  { value: 'petrol', label: 'Petrol / Gasoline', factor: 2.31 },
  { value: 'natural_gas', label: 'Natural Gas', factor: 2.02 },
  { value: 'lpg', label: 'LPG', factor: 1.51 },
  { value: 'jet_fuel', label: 'Jet Fuel (Aviation)', factor: 2.52 },
  { value: 'marine_fuel', label: 'Marine Fuel Oil', factor: 3.11 },
  { value: 'coal', label: 'Coal', factor: 2.42 },
  { value: 'biofuel', label: 'Biofuel (B20)', factor: 0.54 },
];

const scopeDescriptions = {
  1: 'Direct emissions from owned or controlled sources',
  2: 'Indirect emissions from purchased energy',
  3: 'All other indirect emissions in the value chain',
};

export default function FuelAnalyzer() {
  const [result, setResult] = React.useState<FuelAnalysisResult | null>(null);
  const [activeTab, setActiveTab] = React.useState('calculator');

  const calculateFuel = useCalculateFuel();
  const { data: historyResponse, isLoading: historyLoading } = useFuelHistory({ perPage: 10 });

  const form = useForm<FuelFormData>({
    resolver: zodResolver(fuelFormSchema),
    defaultValues: {
      fuelType: '',
      quantity: 0,
      unit: 'liters',
      scope: 1,
      location: '',
      startDate: new Date().toISOString().split('T')[0],
      endDate: new Date().toISOString().split('T')[0],
    },
  });

  // Mock history data
  const mockHistory: FuelAnalysisResult[] = [
    {
      id: '1',
      fuelType: 'diesel',
      quantity: 5000,
      unit: 'liters',
      emissions: { co2: 12.5, ch4: 0.02, n2o: 0.01, total: 12.53 },
      emissionFactor: 2.68,
      emissionFactorSource: 'DEFRA 2024',
      scope: 1,
      calculatedAt: new Date(Date.now() - 1000 * 60 * 60 * 2).toISOString(),
    },
    {
      id: '2',
      fuelType: 'petrol',
      quantity: 2500,
      unit: 'liters',
      emissions: { co2: 5.78, ch4: 0.01, n2o: 0.005, total: 5.8 },
      emissionFactor: 2.31,
      emissionFactorSource: 'DEFRA 2024',
      scope: 1,
      calculatedAt: new Date(Date.now() - 1000 * 60 * 60 * 24).toISOString(),
    },
    {
      id: '3',
      fuelType: 'natural_gas',
      quantity: 10000,
      unit: 'kg',
      emissions: { co2: 20.2, ch4: 0.05, n2o: 0.02, total: 20.27 },
      emissionFactor: 2.02,
      emissionFactorSource: 'EPA 2024',
      scope: 1,
      calculatedAt: new Date(Date.now() - 1000 * 60 * 60 * 48).toISOString(),
    },
  ];

  const history = historyResponse?.items || mockHistory;

  const onSubmit = (data: FuelFormData) => {
    calculateFuel.mutate(
      {
        fuelType: data.fuelType,
        quantity: data.quantity,
        unit: data.unit,
        scope: data.scope as 1 | 2 | 3,
        location: data.location,
        period: {
          startDate: data.startDate,
          endDate: data.endDate,
        },
      },
      {
        onSuccess: (response) => {
          setResult(response);
        },
      }
    );
  };

  // Mock result for UI development
  const mockResult: FuelAnalysisResult = {
    id: 'new',
    fuelType: form.watch('fuelType'),
    quantity: form.watch('quantity'),
    unit: form.watch('unit'),
    emissions: {
      co2: form.watch('quantity') * 0.00268,
      ch4: form.watch('quantity') * 0.00001,
      n2o: form.watch('quantity') * 0.000005,
      total: form.watch('quantity') * 0.00268,
    },
    emissionFactor: 2.68,
    emissionFactorSource: 'DEFRA 2024',
    scope: form.watch('scope'),
    calculatedAt: new Date().toISOString(),
  };

  const displayResult = result || (calculateFuel.isPending ? mockResult : null);

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div>
        <h1 className="text-2xl font-bold">Fuel Emissions Calculator</h1>
        <p className="text-muted-foreground">
          Calculate greenhouse gas emissions from fuel consumption using verified emission factors.
        </p>
      </div>

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
        </TabsList>

        <TabsContent value="calculator" className="space-y-6">
          <div className="grid gap-6 lg:grid-cols-2">
            {/* Input form */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Fuel className="h-5 w-5" />
                  Fuel Consumption Data
                </CardTitle>
                <CardDescription>
                  Enter your fuel consumption details to calculate emissions.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
                  {/* Fuel Type */}
                  <div>
                    <label className="text-sm font-medium">Fuel Type</label>
                    <Select
                      value={form.watch('fuelType')}
                      onValueChange={(value) => form.setValue('fuelType', value)}
                    >
                      <SelectTrigger className="mt-1.5" error={!!form.formState.errors.fuelType}>
                        <SelectValue placeholder="Select fuel type" />
                      </SelectTrigger>
                      <SelectContent>
                        {fuelTypes.map((fuel) => (
                          <SelectItem key={fuel.value} value={fuel.value}>
                            {fuel.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    {form.formState.errors.fuelType && (
                      <p className="text-sm text-destructive mt-1">
                        {form.formState.errors.fuelType.message}
                      </p>
                    )}
                  </div>

                  {/* Quantity and Unit */}
                  <div className="grid gap-4 sm:grid-cols-2">
                    <Input
                      label="Quantity"
                      type="number"
                      step="0.01"
                      {...form.register('quantity', { valueAsNumber: true })}
                      error={form.formState.errors.quantity?.message}
                    />
                    <div>
                      <label className="text-sm font-medium">Unit</label>
                      <Select
                        value={form.watch('unit')}
                        onValueChange={(value: 'liters' | 'gallons' | 'kg' | 'tonnes') =>
                          form.setValue('unit', value)
                        }
                      >
                        <SelectTrigger className="mt-1.5">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="liters">Liters</SelectItem>
                          <SelectItem value="gallons">Gallons</SelectItem>
                          <SelectItem value="kg">Kilograms</SelectItem>
                          <SelectItem value="tonnes">Metric Tonnes</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  {/* Period */}
                  <div className="grid gap-4 sm:grid-cols-2">
                    <Input
                      label="Start Date"
                      type="date"
                      {...form.register('startDate')}
                      error={form.formState.errors.startDate?.message}
                    />
                    <Input
                      label="End Date"
                      type="date"
                      {...form.register('endDate')}
                      error={form.formState.errors.endDate?.message}
                    />
                  </div>

                  {/* Scope */}
                  <div>
                    <label className="text-sm font-medium">Emission Scope</label>
                    <div className="grid gap-2 mt-2">
                      {[1, 2, 3].map((scope) => (
                        <label
                          key={scope}
                          className={`flex items-start gap-3 p-3 border rounded-lg cursor-pointer transition-colors ${
                            form.watch('scope') === scope
                              ? 'border-primary bg-primary/5'
                              : 'border-border hover:bg-muted/50'
                          }`}
                        >
                          <input
                            type="radio"
                            value={scope}
                            checked={form.watch('scope') === scope}
                            onChange={() => form.setValue('scope', scope)}
                            className="mt-0.5"
                          />
                          <div>
                            <span className="font-medium">Scope {scope}</span>
                            <p className="text-sm text-muted-foreground">
                              {scopeDescriptions[scope as keyof typeof scopeDescriptions]}
                            </p>
                          </div>
                        </label>
                      ))}
                    </div>
                  </div>

                  {/* Location (optional) */}
                  <Input
                    label="Location (Optional)"
                    placeholder="e.g., Germany, California"
                    {...form.register('location')}
                    helperText="For region-specific emission factors"
                  />

                  <Button type="submit" className="w-full" loading={calculateFuel.isPending}>
                    <Calculator className="h-4 w-4 mr-2" />
                    Calculate Emissions
                  </Button>
                </form>
              </CardContent>
            </Card>

            {/* Results */}
            <div className="space-y-6">
              {displayResult ? (
                <>
                  <Card className="border-greenlang-200 bg-greenlang-50/50">
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2 text-greenlang-800">
                        <Leaf className="h-5 w-5" />
                        Calculation Results
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-center mb-6">
                        <p className="text-sm text-muted-foreground">Total Emissions</p>
                        <p className="text-4xl font-bold text-greenlang-700">
                          {formatEmissions(displayResult.emissions.total)}
                        </p>
                      </div>

                      <div className="grid grid-cols-3 gap-4 mb-6">
                        <div className="text-center p-3 bg-background rounded-lg">
                          <p className="text-xs text-muted-foreground">CO2</p>
                          <p className="font-semibold">
                            {displayResult.emissions.co2.toFixed(3)} t
                          </p>
                        </div>
                        <div className="text-center p-3 bg-background rounded-lg">
                          <p className="text-xs text-muted-foreground">CH4</p>
                          <p className="font-semibold">
                            {displayResult.emissions.ch4.toFixed(4)} t
                          </p>
                        </div>
                        <div className="text-center p-3 bg-background rounded-lg">
                          <p className="text-xs text-muted-foreground">N2O</p>
                          <p className="font-semibold">
                            {displayResult.emissions.n2o.toFixed(4)} t
                          </p>
                        </div>
                      </div>

                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Emission Factor:</span>
                          <span className="font-medium">
                            {displayResult.emissionFactor} kgCO2e/L
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Data Source:</span>
                          <span className="font-medium">{displayResult.emissionFactorSource}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Scope:</span>
                          <Badge variant="secondary">Scope {displayResult.scope}</Badge>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <div className="flex gap-2">
                    <Button variant="outline" className="flex-1">
                      <Download className="h-4 w-4 mr-2" />
                      Export PDF
                    </Button>
                    <Button variant="outline" className="flex-1">
                      Add to Report
                    </Button>
                  </div>
                </>
              ) : (
                <Card className="flex flex-col items-center justify-center p-12 text-center">
                  <Calculator className="h-12 w-12 text-muted-foreground mb-4" />
                  <h3 className="font-semibold">Enter Your Data</h3>
                  <p className="text-sm text-muted-foreground max-w-sm mt-2">
                    Fill in the fuel consumption details on the left to calculate your emissions.
                  </p>
                </Card>
              )}

              {/* Emission factors info */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-base flex items-center gap-2">
                    <Info className="h-4 w-4" />
                    About Emission Factors
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">
                    We use the latest emission factors from DEFRA, EPA, and IPCC databases.
                    Factors are updated annually and include all greenhouse gases converted to
                    CO2 equivalent (CO2e) using GWP100 values.
                  </p>
                  <Button variant="link" size="sm" className="px-0 mt-2">
                    View Methodology <ArrowRight className="h-4 w-4 ml-1" />
                  </Button>
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="history">
          <Card>
            <CardHeader>
              <CardTitle>Calculation History</CardTitle>
              <CardDescription>Your previous fuel emissions calculations</CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Fuel Type</TableHead>
                    <TableHead className="text-right">Quantity</TableHead>
                    <TableHead className="text-right">Emissions</TableHead>
                    <TableHead>Scope</TableHead>
                    <TableHead>Source</TableHead>
                    <TableHead>Date</TableHead>
                    <TableHead></TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {history.map((item) => (
                    <TableRow key={item.id}>
                      <TableCell className="font-medium capitalize">
                        {item.fuelType.replace('_', ' ')}
                      </TableCell>
                      <TableCell className="text-right">
                        {formatNumber(item.quantity)} {item.unit}
                      </TableCell>
                      <TableCell className="text-right font-medium text-greenlang-600">
                        {formatEmissions(item.emissions.total)}
                      </TableCell>
                      <TableCell>
                        <Badge variant="secondary">Scope {item.scope}</Badge>
                      </TableCell>
                      <TableCell className="text-muted-foreground">
                        {item.emissionFactorSource}
                      </TableCell>
                      <TableCell className="text-muted-foreground">
                        {formatDateTime(item.calculatedAt)}
                      </TableCell>
                      <TableCell>
                        <Button variant="ghost" size="icon-sm">
                          <Download className="h-4 w-4" />
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
