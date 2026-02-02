/**
 * EUDRCompliance Page
 *
 * EU Deforestation Regulation compliance checker.
 */

import * as React from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import {
  TreePine,
  MapPin,
  Shield,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Upload,
  Globe,
  Satellite,
  FileText,
  Clock,
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Textarea } from '@/components/ui/Textarea';
import { Badge } from '@/components/ui/Badge';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/Select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/Tabs';
import { useCheckEUDRCompliance, useEUDRHistory } from '@/api/hooks';
import { formatNumber, formatDateTime, countryCodeToFlag } from '@/utils/format';
import { cn } from '@/utils/cn';
import type { EUDRComplianceResult, ComplianceStatus } from '@/api/types';

const eudrFormSchema = z.object({
  commodity: z.string().min(1, 'Select commodity'),
  originCountry: z.string().min(2, 'Select origin country'),
  productionDate: z.string().min(1, 'Production date required'),
  latitude: z.number().min(-90).max(90),
  longitude: z.number().min(-180).max(180),
  supplierName: z.string().min(1, 'Supplier name required'),
  supplierCountry: z.string().min(2),
  quantity: z.number().min(0.001),
  unit: z.string(),
});

type EUDRFormData = z.infer<typeof eudrFormSchema>;

const commodities = [
  { value: 'palm_oil', label: 'Palm Oil' },
  { value: 'soy', label: 'Soy' },
  { value: 'cocoa', label: 'Cocoa' },
  { value: 'coffee', label: 'Coffee' },
  { value: 'cattle', label: 'Cattle Products' },
  { value: 'wood', label: 'Wood Products' },
  { value: 'rubber', label: 'Rubber' },
];

const riskCountries = [
  { code: 'BR', name: 'Brazil', risk: 'high' },
  { code: 'ID', name: 'Indonesia', risk: 'high' },
  { code: 'MY', name: 'Malaysia', risk: 'medium' },
  { code: 'CI', name: 'Ivory Coast', risk: 'high' },
  { code: 'GH', name: 'Ghana', risk: 'medium' },
  { code: 'CO', name: 'Colombia', risk: 'medium' },
  { code: 'PE', name: 'Peru', risk: 'medium' },
  { code: 'TH', name: 'Thailand', risk: 'low' },
];

const statusConfig: Record<ComplianceStatus, { icon: typeof CheckCircle; color: string; label: string }> = {
  compliant: { icon: CheckCircle, color: 'text-greenlang-600', label: 'Compliant' },
  non_compliant: { icon: XCircle, color: 'text-red-600', label: 'Non-Compliant' },
  pending_review: { icon: Clock, color: 'text-amber-600', label: 'Pending Review' },
  requires_action: { icon: AlertTriangle, color: 'text-orange-600', label: 'Requires Action' },
};

export default function EUDRCompliance() {
  const [result, setResult] = React.useState<EUDRComplianceResult | null>(null);
  const [activeTab, setActiveTab] = React.useState('checker');

  const checkCompliance = useCheckEUDRCompliance();
  const { data: historyResponse } = useEUDRHistory({ perPage: 10 });

  const form = useForm<EUDRFormData>({
    resolver: zodResolver(eudrFormSchema),
    defaultValues: {
      commodity: '',
      originCountry: '',
      productionDate: '',
      latitude: 0,
      longitude: 0,
      supplierName: '',
      supplierCountry: '',
      quantity: 0,
      unit: 'tonnes',
    },
  });

  const onSubmit = (data: EUDRFormData) => {
    checkCompliance.mutate(
      {
        commodity: data.commodity,
        originCountry: data.originCountry,
        productionDate: data.productionDate,
        geoLocation: {
          latitude: data.latitude,
          longitude: data.longitude,
        },
        supplierChain: [
          {
            name: data.supplierName,
            role: 'Producer',
            country: data.supplierCountry,
          },
        ],
        quantity: data.quantity,
        unit: data.unit,
      },
      {
        onSuccess: (response) => setResult(response),
      }
    );
  };

  // Mock result for demonstration
  const mockResult: EUDRComplianceResult = {
    id: 'demo',
    commodity: 'palm_oil',
    originCountry: 'ID',
    complianceStatus: 'compliant',
    riskAssessment: {
      deforestationRisk: 'low',
      legalityRisk: 'low',
      overallRisk: 'low',
      score: 92,
    },
    satelliteAnalysis: {
      forestCoverChange: -0.5,
      deforestationDetected: false,
      analysisDate: new Date().toISOString(),
      confidence: 95,
    },
    dueDiligenceChecklist: [
      { item: 'Geolocation data verified', status: 'passed' },
      { item: 'Production date before cutoff', status: 'passed' },
      { item: 'Supplier documentation complete', status: 'passed' },
      { item: 'Satellite imagery analysis', status: 'passed' },
      { item: 'Legal compliance certificates', status: 'pending', notes: 'Awaiting supplier' },
    ],
    documentationRequired: [
      'Certificate of Origin',
      'Geolocation coordinates (polygon)',
      'Production date documentation',
      'Supplier due diligence statement',
    ],
    calculatedAt: new Date().toISOString(),
  };

  const displayResult = result || (checkCompliance.data ? checkCompliance.data : null);

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold">EUDR Compliance Checker</h1>
          <p className="text-muted-foreground">
            Verify EU Deforestation Regulation compliance for your supply chain.
          </p>
        </div>
        <Badge variant="warning" className="gap-2 w-fit">
          <AlertTriangle className="h-3 w-3" />
          EUDR deadline: December 30, 2024
        </Badge>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="checker">
            <Shield className="h-4 w-4 mr-2" />
            Compliance Check
          </TabsTrigger>
          <TabsTrigger value="history">
            <Clock className="h-4 w-4 mr-2" />
            History
          </TabsTrigger>
          <TabsTrigger value="info">
            <FileText className="h-4 w-4 mr-2" />
            EUDR Info
          </TabsTrigger>
        </TabsList>

        <TabsContent value="checker" className="space-y-6">
          <div className="grid gap-6 lg:grid-cols-2">
            {/* Input form */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TreePine className="h-5 w-5" />
                  Product Information
                </CardTitle>
                <CardDescription>
                  Enter product details to check EUDR compliance status.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
                  {/* Commodity */}
                  <div>
                    <label className="text-sm font-medium">Commodity</label>
                    <Select
                      value={form.watch('commodity')}
                      onValueChange={(value) => form.setValue('commodity', value)}
                    >
                      <SelectTrigger className="mt-1.5">
                        <SelectValue placeholder="Select commodity" />
                      </SelectTrigger>
                      <SelectContent>
                        {commodities.map((c) => (
                          <SelectItem key={c.value} value={c.value}>
                            {c.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  {/* Origin Country and Date */}
                  <div className="grid gap-4 sm:grid-cols-2">
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
                          {riskCountries.map((country) => (
                            <SelectItem key={country.code} value={country.code}>
                              <span className="flex items-center gap-2">
                                {countryCodeToFlag(country.code)} {country.name}
                                <Badge
                                  variant={
                                    country.risk === 'high'
                                      ? 'destructive'
                                      : country.risk === 'medium'
                                        ? 'warning'
                                        : 'success'
                                  }
                                  size="sm"
                                >
                                  {country.risk}
                                </Badge>
                              </span>
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <Input
                      label="Production Date"
                      type="date"
                      {...form.register('productionDate')}
                      error={form.formState.errors.productionDate?.message}
                    />
                  </div>

                  {/* Geolocation */}
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <label className="text-sm font-medium">Geolocation</label>
                      <Button variant="ghost" size="sm" type="button">
                        <MapPin className="h-4 w-4 mr-1" />
                        Pick on Map
                      </Button>
                    </div>
                    <div className="grid gap-4 sm:grid-cols-2">
                      <Input
                        label="Latitude"
                        type="number"
                        step="0.000001"
                        placeholder="-2.345678"
                        {...form.register('latitude', { valueAsNumber: true })}
                      />
                      <Input
                        label="Longitude"
                        type="number"
                        step="0.000001"
                        placeholder="104.123456"
                        {...form.register('longitude', { valueAsNumber: true })}
                      />
                    </div>
                  </div>

                  {/* Supplier */}
                  <div className="grid gap-4 sm:grid-cols-2">
                    <Input
                      label="Supplier Name"
                      {...form.register('supplierName')}
                      error={form.formState.errors.supplierName?.message}
                    />
                    <div>
                      <label className="text-sm font-medium">Supplier Country</label>
                      <Select
                        value={form.watch('supplierCountry')}
                        onValueChange={(value) => form.setValue('supplierCountry', value)}
                      >
                        <SelectTrigger className="mt-1.5">
                          <SelectValue placeholder="Country" />
                        </SelectTrigger>
                        <SelectContent>
                          {riskCountries.map((country) => (
                            <SelectItem key={country.code} value={country.code}>
                              {countryCodeToFlag(country.code)} {country.name}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  {/* Quantity */}
                  <div className="grid gap-4 sm:grid-cols-2">
                    <Input
                      label="Quantity"
                      type="number"
                      step="0.001"
                      {...form.register('quantity', { valueAsNumber: true })}
                    />
                    <div>
                      <label className="text-sm font-medium">Unit</label>
                      <Select
                        value={form.watch('unit')}
                        onValueChange={(value) => form.setValue('unit', value)}
                      >
                        <SelectTrigger className="mt-1.5">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="tonnes">Tonnes</SelectItem>
                          <SelectItem value="kg">Kilograms</SelectItem>
                          <SelectItem value="liters">Liters</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  {/* Upload Documents */}
                  <div className="border-2 border-dashed rounded-lg p-6 text-center">
                    <Upload className="h-8 w-8 text-muted-foreground mx-auto mb-2" />
                    <p className="text-sm font-medium">Upload Supporting Documents</p>
                    <p className="text-xs text-muted-foreground mt-1">
                      Certificates, invoices, or geolocation data
                    </p>
                    <Button variant="outline" size="sm" className="mt-4" type="button">
                      Choose Files
                    </Button>
                  </div>

                  <Button type="submit" className="w-full" loading={checkCompliance.isPending}>
                    <Shield className="h-4 w-4 mr-2" />
                    Check Compliance
                  </Button>
                </form>
              </CardContent>
            </Card>

            {/* Results */}
            <div className="space-y-6">
              {displayResult ? (
                <>
                  {/* Compliance Status */}
                  <Card className={cn(
                    displayResult.complianceStatus === 'compliant' && 'border-greenlang-200 bg-greenlang-50/50',
                    displayResult.complianceStatus === 'non_compliant' && 'border-red-200 bg-red-50/50',
                    displayResult.complianceStatus === 'pending_review' && 'border-amber-200 bg-amber-50/50',
                    displayResult.complianceStatus === 'requires_action' && 'border-orange-200 bg-orange-50/50'
                  )}>
                    <CardHeader>
                      <div className="flex items-center justify-between">
                        <CardTitle className="flex items-center gap-2">
                          {React.createElement(statusConfig[displayResult.complianceStatus].icon, {
                            className: cn('h-5 w-5', statusConfig[displayResult.complianceStatus].color),
                          })}
                          Compliance Status
                        </CardTitle>
                        <Badge variant={
                          displayResult.complianceStatus === 'compliant' ? 'success' :
                          displayResult.complianceStatus === 'non_compliant' ? 'destructive' : 'warning'
                        }>
                          {statusConfig[displayResult.complianceStatus].label}
                        </Badge>
                      </div>
                    </CardHeader>
                    <CardContent>
                      {/* Risk Assessment */}
                      <div className="space-y-4">
                        <h4 className="text-sm font-medium">Risk Assessment</h4>
                        <div className="grid grid-cols-3 gap-4">
                          {[
                            { label: 'Deforestation', value: displayResult.riskAssessment.deforestationRisk },
                            { label: 'Legality', value: displayResult.riskAssessment.legalityRisk },
                            { label: 'Overall', value: displayResult.riskAssessment.overallRisk },
                          ].map((item) => (
                            <div key={item.label} className="text-center p-3 bg-background rounded-lg">
                              <p className="text-xs text-muted-foreground">{item.label}</p>
                              <Badge
                                variant={
                                  item.value === 'low' ? 'success' :
                                  item.value === 'medium' ? 'warning' : 'destructive'
                                }
                                className="mt-1"
                              >
                                {item.value}
                              </Badge>
                            </div>
                          ))}
                        </div>

                        {/* Score */}
                        <div className="flex items-center justify-between p-4 bg-background rounded-lg">
                          <span>Compliance Score</span>
                          <span className={cn(
                            'text-2xl font-bold',
                            displayResult.riskAssessment.score >= 80 ? 'text-greenlang-600' :
                            displayResult.riskAssessment.score >= 60 ? 'text-amber-600' : 'text-red-600'
                          )}>
                            {displayResult.riskAssessment.score}%
                          </span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  {/* Satellite Analysis */}
                  {displayResult.satelliteAnalysis && (
                    <Card>
                      <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                          <Satellite className="h-5 w-5" />
                          Satellite Analysis
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-3">
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Deforestation Detected</span>
                            <Badge variant={displayResult.satelliteAnalysis.deforestationDetected ? 'destructive' : 'success'}>
                              {displayResult.satelliteAnalysis.deforestationDetected ? 'Yes' : 'No'}
                            </Badge>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Forest Cover Change</span>
                            <span className="font-medium">
                              {displayResult.satelliteAnalysis.forestCoverChange > 0 ? '+' : ''}
                              {displayResult.satelliteAnalysis.forestCoverChange}%
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Analysis Confidence</span>
                            <span className="font-medium">
                              {displayResult.satelliteAnalysis.confidence}%
                            </span>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  )}

                  {/* Due Diligence Checklist */}
                  <Card>
                    <CardHeader>
                      <CardTitle>Due Diligence Checklist</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-2">
                        {displayResult.dueDiligenceChecklist.map((item, index) => (
                          <div key={index} className="flex items-center gap-3 p-2 rounded-lg hover:bg-muted/50">
                            {item.status === 'passed' && <CheckCircle className="h-5 w-5 text-greenlang-600" />}
                            {item.status === 'failed' && <XCircle className="h-5 w-5 text-red-600" />}
                            {item.status === 'pending' && <Clock className="h-5 w-5 text-amber-600" />}
                            <span className={cn(
                              'flex-1',
                              item.status === 'failed' && 'text-red-600'
                            )}>
                              {item.item}
                            </span>
                            {item.notes && (
                              <span className="text-xs text-muted-foreground">{item.notes}</span>
                            )}
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>

                  {/* Required Documentation */}
                  <Card>
                    <CardHeader>
                      <CardTitle>Required Documentation</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <ul className="space-y-2">
                        {displayResult.documentationRequired.map((doc, index) => (
                          <li key={index} className="flex items-center gap-2 text-sm">
                            <FileText className="h-4 w-4 text-muted-foreground" />
                            {doc}
                          </li>
                        ))}
                      </ul>
                    </CardContent>
                  </Card>
                </>
              ) : (
                <Card className="flex flex-col items-center justify-center p-12 text-center">
                  <TreePine className="h-12 w-12 text-muted-foreground mb-4" />
                  <h3 className="font-semibold">Check Your Compliance</h3>
                  <p className="text-sm text-muted-foreground max-w-sm mt-2">
                    Enter your product details to verify compliance with the EU Deforestation Regulation.
                  </p>
                </Card>
              )}
            </div>
          </div>
        </TabsContent>

        <TabsContent value="history">
          <Card className="p-12 text-center">
            <Clock className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <h3 className="font-semibold">No Compliance History</h3>
            <p className="text-sm text-muted-foreground mt-2">
              Your compliance check history will appear here.
            </p>
          </Card>
        </TabsContent>

        <TabsContent value="info">
          <Card>
            <CardHeader>
              <CardTitle>About EUDR</CardTitle>
              <CardDescription>
                EU Deforestation Regulation (EU 2023/1115)
              </CardDescription>
            </CardHeader>
            <CardContent className="prose prose-sm max-w-none">
              <h4>Covered Commodities</h4>
              <p>
                The EUDR applies to: cattle, cocoa, coffee, oil palm, rubber, soya, and wood,
                as well as products derived from them.
              </p>

              <h4>Key Requirements</h4>
              <ul>
                <li>Products must be deforestation-free (produced on land not deforested after Dec 31, 2020)</li>
                <li>Must comply with relevant legislation of the country of production</li>
                <li>Must have a due diligence statement</li>
              </ul>

              <h4>Timeline</h4>
              <ul>
                <li><strong>December 30, 2024:</strong> Entry into application for large operators</li>
                <li><strong>June 30, 2025:</strong> Entry into application for SMEs</li>
              </ul>

              <h4>Required Information</h4>
              <ul>
                <li>Geolocation coordinates of production areas</li>
                <li>Date or time range of production</li>
                <li>Product description and quantity</li>
                <li>Country and region of production</li>
              </ul>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
