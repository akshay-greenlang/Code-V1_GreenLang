/**
 * BuildingEnergy Page
 *
 * Building energy consumption analysis and benchmarking.
 */

import * as React from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import {
  Building,
  Zap,
  Flame,
  Thermometer,
  Snowflake,
  Calculator,
  Download,
  History,
  TrendingDown,
  Award,
  Lightbulb,
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
import { useCalculateBuildingEnergy, useBuildingHistory } from '@/api/hooks';
import { formatEmissions, formatNumber } from '@/utils/format';
import { cn } from '@/utils/cn';
import type { BuildingEnergyResult } from '@/api/types';

const buildingFormSchema = z.object({
  buildingType: z.string().min(1, 'Select building type'),
  floorArea: z.number().min(1, 'Floor area must be greater than 0'),
  country: z.string().min(2, 'Select country'),
  climateZone: z.string().optional(),
  electricity: z.number().min(0),
  naturalGas: z.number().min(0),
  heating: z.number().min(0),
  cooling: z.number().min(0),
  startDate: z.string(),
  endDate: z.string(),
});

type BuildingFormData = z.infer<typeof buildingFormSchema>;

const buildingTypes = [
  { value: 'office', label: 'Office Building' },
  { value: 'retail', label: 'Retail / Shopping' },
  { value: 'warehouse', label: 'Warehouse / Industrial' },
  { value: 'hotel', label: 'Hotel / Hospitality' },
  { value: 'hospital', label: 'Healthcare / Hospital' },
  { value: 'education', label: 'Education / School' },
  { value: 'residential', label: 'Residential Multi-family' },
  { value: 'data_center', label: 'Data Center' },
];

const ratingColors = {
  A: 'bg-greenlang-500',
  B: 'bg-greenlang-400',
  C: 'bg-lime-400',
  D: 'bg-yellow-400',
  E: 'bg-orange-400',
  F: 'bg-orange-500',
  G: 'bg-red-500',
};

export default function BuildingEnergy() {
  const [result, setResult] = React.useState<BuildingEnergyResult | null>(null);
  const [activeTab, setActiveTab] = React.useState('calculator');

  const calculateBuilding = useCalculateBuildingEnergy();
  const { data: historyResponse } = useBuildingHistory({ perPage: 10 });

  const form = useForm<BuildingFormData>({
    resolver: zodResolver(buildingFormSchema),
    defaultValues: {
      buildingType: '',
      floorArea: 0,
      country: 'DE',
      electricity: 0,
      naturalGas: 0,
      heating: 0,
      cooling: 0,
      startDate: new Date(new Date().setMonth(new Date().getMonth() - 1)).toISOString().split('T')[0],
      endDate: new Date().toISOString().split('T')[0],
    },
  });

  const onSubmit = (data: BuildingFormData) => {
    calculateBuilding.mutate(
      {
        buildingType: data.buildingType,
        floorArea: data.floorArea,
        location: {
          country: data.country,
          climateZone: data.climateZone,
        },
        energyConsumption: {
          electricity: data.electricity,
          naturalGas: data.naturalGas,
          heating: data.heating,
          cooling: data.cooling,
        },
        period: {
          startDate: data.startDate,
          endDate: data.endDate,
        },
      },
      {
        onSuccess: (response) => setResult(response),
      }
    );
  };

  // Mock result for demonstration
  const mockResult: BuildingEnergyResult = {
    id: 'demo',
    buildingType: 'office',
    floorArea: 5000,
    emissions: {
      electricity: 85.5,
      naturalGas: 32.4,
      heating: 15.2,
      cooling: 12.8,
      total: 145.9,
    },
    intensity: {
      perSquareMeter: 29.18,
      perOccupant: 0.73,
    },
    benchmark: {
      average: 45.5,
      percentile: 78,
      rating: 'B',
    },
    recommendations: [
      {
        category: 'Lighting',
        description: 'Upgrade to LED lighting throughout the building',
        potentialSavings: 12500,
      },
      {
        category: 'HVAC',
        description: 'Install smart thermostats and optimize schedules',
        potentialSavings: 8200,
      },
      {
        category: 'Insulation',
        description: 'Improve window glazing and wall insulation',
        potentialSavings: 15800,
      },
    ],
    calculatedAt: new Date().toISOString(),
  };

  const displayResult = result || (calculateBuilding.data ? calculateBuilding.data : null);

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div>
        <h1 className="text-2xl font-bold">Building Energy Analysis</h1>
        <p className="text-muted-foreground">
          Analyze building energy consumption and benchmark against industry standards.
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
                  <Building className="h-5 w-5" />
                  Building Information
                </CardTitle>
                <CardDescription>
                  Enter your building details and energy consumption data.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
                  {/* Building Type */}
                  <div>
                    <label className="text-sm font-medium">Building Type</label>
                    <Select
                      value={form.watch('buildingType')}
                      onValueChange={(value) => form.setValue('buildingType', value)}
                    >
                      <SelectTrigger className="mt-1.5">
                        <SelectValue placeholder="Select building type" />
                      </SelectTrigger>
                      <SelectContent>
                        {buildingTypes.map((type) => (
                          <SelectItem key={type.value} value={type.value}>
                            {type.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  {/* Floor Area and Location */}
                  <div className="grid gap-4 sm:grid-cols-2">
                    <Input
                      label="Floor Area (m2)"
                      type="number"
                      {...form.register('floorArea', { valueAsNumber: true })}
                      error={form.formState.errors.floorArea?.message}
                    />
                    <div>
                      <label className="text-sm font-medium">Country</label>
                      <Select
                        value={form.watch('country')}
                        onValueChange={(value) => form.setValue('country', value)}
                      >
                        <SelectTrigger className="mt-1.5">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="DE">Germany</SelectItem>
                          <SelectItem value="FR">France</SelectItem>
                          <SelectItem value="UK">United Kingdom</SelectItem>
                          <SelectItem value="NL">Netherlands</SelectItem>
                          <SelectItem value="US">United States</SelectItem>
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
                    />
                    <Input
                      label="End Date"
                      type="date"
                      {...form.register('endDate')}
                    />
                  </div>

                  {/* Energy Consumption */}
                  <div className="space-y-4">
                    <h4 className="text-sm font-medium">Energy Consumption (kWh)</h4>

                    <div className="grid gap-4 sm:grid-cols-2">
                      <div className="relative">
                        <Zap className="absolute left-3 top-9 h-4 w-4 text-amber-500" />
                        <Input
                          label="Electricity"
                          type="number"
                          className="pl-10"
                          {...form.register('electricity', { valueAsNumber: true })}
                        />
                      </div>
                      <div className="relative">
                        <Flame className="absolute left-3 top-9 h-4 w-4 text-orange-500" />
                        <Input
                          label="Natural Gas"
                          type="number"
                          className="pl-10"
                          {...form.register('naturalGas', { valueAsNumber: true })}
                        />
                      </div>
                      <div className="relative">
                        <Thermometer className="absolute left-3 top-9 h-4 w-4 text-red-500" />
                        <Input
                          label="District Heating"
                          type="number"
                          className="pl-10"
                          {...form.register('heating', { valueAsNumber: true })}
                        />
                      </div>
                      <div className="relative">
                        <Snowflake className="absolute left-3 top-9 h-4 w-4 text-blue-500" />
                        <Input
                          label="District Cooling"
                          type="number"
                          className="pl-10"
                          {...form.register('cooling', { valueAsNumber: true })}
                        />
                      </div>
                    </div>
                  </div>

                  <Button type="submit" className="w-full" loading={calculateBuilding.isPending}>
                    <Calculator className="h-4 w-4 mr-2" />
                    Analyze Building
                  </Button>
                </form>
              </CardContent>
            </Card>

            {/* Results */}
            <div className="space-y-6">
              {displayResult ? (
                <>
                  {/* Energy Rating */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Award className="h-5 w-5" />
                        Energy Performance Rating
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="flex items-center gap-6">
                        <div
                          className={cn(
                            'flex h-24 w-24 items-center justify-center rounded-xl text-4xl font-bold text-white',
                            ratingColors[displayResult.benchmark.rating]
                          )}
                        >
                          {displayResult.benchmark.rating}
                        </div>
                        <div>
                          <p className="text-2xl font-bold">
                            {displayResult.intensity.perSquareMeter.toFixed(1)}{' '}
                            <span className="text-base font-normal text-muted-foreground">
                              kgCO2e/m2/year
                            </span>
                          </p>
                          <p className="text-sm text-muted-foreground mt-1">
                            Better than {displayResult.benchmark.percentile}% of similar buildings
                          </p>
                          <div className="flex items-center gap-2 mt-2">
                            <Badge variant="secondary">
                              Avg: {displayResult.benchmark.average.toFixed(1)} kgCO2e/m2
                            </Badge>
                          </div>
                        </div>
                      </div>

                      {/* Rating scale */}
                      <div className="flex gap-1 mt-6">
                        {(['A', 'B', 'C', 'D', 'E', 'F', 'G'] as const).map((rating) => (
                          <div
                            key={rating}
                            className={cn(
                              'flex-1 h-3 rounded-full',
                              ratingColors[rating],
                              rating === displayResult.benchmark.rating ? 'ring-2 ring-offset-2 ring-primary' : 'opacity-50'
                            )}
                          />
                        ))}
                      </div>
                    </CardContent>
                  </Card>

                  {/* Emissions Breakdown */}
                  <Card>
                    <CardHeader>
                      <CardTitle>Emissions Breakdown</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-center mb-6">
                        <p className="text-sm text-muted-foreground">Total Emissions</p>
                        <p className="text-3xl font-bold">
                          {formatEmissions(displayResult.emissions.total)}
                        </p>
                      </div>

                      <div className="space-y-3">
                        {[
                          { label: 'Electricity', value: displayResult.emissions.electricity, icon: Zap, color: 'bg-amber-500' },
                          { label: 'Natural Gas', value: displayResult.emissions.naturalGas, icon: Flame, color: 'bg-orange-500' },
                          { label: 'Heating', value: displayResult.emissions.heating, icon: Thermometer, color: 'bg-red-500' },
                          { label: 'Cooling', value: displayResult.emissions.cooling, icon: Snowflake, color: 'bg-blue-500' },
                        ].map((item) => (
                          <div key={item.label} className="flex items-center gap-3">
                            <item.icon className="h-4 w-4 text-muted-foreground" />
                            <span className="text-sm flex-1">{item.label}</span>
                            <div className="w-32 h-2 bg-muted rounded-full overflow-hidden">
                              <div
                                className={cn('h-full rounded-full', item.color)}
                                style={{
                                  width: `${(item.value / displayResult.emissions.total) * 100}%`,
                                }}
                              />
                            </div>
                            <span className="text-sm font-medium w-20 text-right">
                              {item.value.toFixed(1)} t
                            </span>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>

                  {/* Recommendations */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Lightbulb className="h-5 w-5" />
                        Recommendations
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        {displayResult.recommendations.map((rec, index) => (
                          <div key={index} className="p-4 border rounded-lg">
                            <div className="flex items-start justify-between">
                              <div>
                                <Badge variant="secondary">{rec.category}</Badge>
                                <p className="mt-2">{rec.description}</p>
                              </div>
                              <div className="text-right">
                                <p className="text-sm text-muted-foreground">Est. Savings</p>
                                <p className="font-semibold text-greenlang-600">
                                  {formatNumber(rec.potentialSavings)} kWh/year
                                </p>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>

                  <Button variant="outline" className="w-full">
                    <Download className="h-4 w-4 mr-2" />
                    Download Full Report
                  </Button>
                </>
              ) : (
                <Card className="flex flex-col items-center justify-center p-12 text-center">
                  <Building className="h-12 w-12 text-muted-foreground mb-4" />
                  <h3 className="font-semibold">Analyze Your Building</h3>
                  <p className="text-sm text-muted-foreground max-w-sm mt-2">
                    Enter your building details and energy consumption to get your energy
                    performance rating and recommendations.
                  </p>
                </Card>
              )}
            </div>
          </div>
        </TabsContent>

        <TabsContent value="history">
          <Card className="p-12 text-center">
            <History className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <h3 className="font-semibold">No Analysis History</h3>
            <p className="text-sm text-muted-foreground mt-2">
              Your building analysis history will appear here.
            </p>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
