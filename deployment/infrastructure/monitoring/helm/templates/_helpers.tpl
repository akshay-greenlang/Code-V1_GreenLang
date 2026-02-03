{{/*
Expand the name of the chart.
*/}}
{{- define "greenlang-monitoring.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "greenlang-monitoring.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "greenlang-monitoring.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "greenlang-monitoring.labels" -}}
helm.sh/chart: {{ include "greenlang-monitoring.chart" . }}
{{ include "greenlang-monitoring.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: greenlang
{{- end }}

{{/*
Selector labels
*/}}
{{- define "greenlang-monitoring.selectorLabels" -}}
app.kubernetes.io/name: {{ include "greenlang-monitoring.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "greenlang-monitoring.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "greenlang-monitoring.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Prometheus URL
*/}}
{{- define "greenlang-monitoring.prometheusUrl" -}}
{{- if .Values.prometheus.enabled }}
{{- printf "http://%s-prometheus-server:9090" .Release.Name }}
{{- else }}
{{- .Values.prometheus.externalUrl }}
{{- end }}
{{- end }}

{{/*
Loki URL
*/}}
{{- define "greenlang-monitoring.lokiUrl" -}}
{{- if .Values.loki.enabled }}
{{- printf "http://%s-loki:3100" .Release.Name }}
{{- else }}
{{- .Values.loki.externalUrl }}
{{- end }}
{{- end }}

{{/*
Jaeger URL
*/}}
{{- define "greenlang-monitoring.jaegerUrl" -}}
{{- if .Values.jaeger.enabled }}
{{- printf "http://%s-jaeger-query:16686" .Release.Name }}
{{- else }}
{{- .Values.jaeger.externalUrl }}
{{- end }}
{{- end }}

{{/*
Alertmanager URL
*/}}
{{- define "greenlang-monitoring.alertmanagerUrl" -}}
{{- if .Values.prometheus.alertmanager.enabled }}
{{- printf "http://%s-prometheus-alertmanager:9093" .Release.Name }}
{{- else }}
{{- .Values.alertmanager.externalUrl }}
{{- end }}
{{- end }}

{{/*
Create annotations for monitoring resources
*/}}
{{- define "greenlang-monitoring.annotations" -}}
{{- if .Values.global.annotations }}
{{- toYaml .Values.global.annotations }}
{{- end }}
{{- end }}

{{/*
Create pod annotations for monitoring
*/}}
{{- define "greenlang-monitoring.podAnnotations" -}}
prometheus.io/scrape: "true"
prometheus.io/port: "9090"
prometheus.io/path: "/metrics"
{{- end }}

{{/*
SLO helper - Get error budget remaining
*/}}
{{- define "greenlang-monitoring.errorBudget" -}}
{{- $target := .target | default 99.9 }}
{{- $errorBudget := sub 100 $target }}
{{- printf "%.3f" (divf $errorBudget 100) }}
{{- end }}

{{/*
Create ServiceMonitor endpoint configuration
*/}}
{{- define "greenlang-monitoring.serviceMonitorEndpoint" -}}
- port: {{ .port | default "metrics" }}
  interval: {{ .interval | default "30s" }}
  scrapeTimeout: {{ .scrapeTimeout | default "10s" }}
  path: {{ .path | default "/metrics" }}
  scheme: {{ .scheme | default "http" }}
  honorLabels: true
  {{- if .relabelings }}
  relabelings:
    {{- toYaml .relabelings | nindent 4 }}
  {{- end }}
  {{- if .metricRelabelings }}
  metricRelabelings:
    {{- toYaml .metricRelabelings | nindent 4 }}
  {{- end }}
{{- end }}

{{/*
Create common relabelings for ServiceMonitor
*/}}
{{- define "greenlang-monitoring.commonRelabelings" -}}
- sourceLabels: [__meta_kubernetes_pod_name]
  targetLabel: pod
- sourceLabels: [__meta_kubernetes_namespace]
  targetLabel: namespace
- sourceLabels: [__meta_kubernetes_service_name]
  targetLabel: service
{{- end }}

{{/*
Create alert severity label
*/}}
{{- define "greenlang-monitoring.alertLabels" -}}
{{- $severity := .severity | default "warning" }}
severity: {{ $severity }}
{{- if .team }}
team: {{ .team }}
{{- end }}
{{- if eq $severity "critical" }}
pagerduty: "true"
{{- end }}
{{- end }}

{{/*
Create alert annotations
*/}}
{{- define "greenlang-monitoring.alertAnnotations" -}}
summary: {{ .summary | quote }}
{{- if .description }}
description: {{ .description | quote }}
{{- end }}
{{- if .runbook }}
runbook_url: {{ .runbook | quote }}
{{- end }}
{{- end }}

{{/*
Grafana dashboard labels
*/}}
{{- define "greenlang-monitoring.grafanaDashboardLabels" -}}
grafana_dashboard: "1"
app.kubernetes.io/part-of: greenlang-monitoring
{{- end }}

{{/*
Recording rule helper
*/}}
{{- define "greenlang-monitoring.recordingRule" -}}
- record: {{ .record }}
  expr: {{ .expr | quote }}
  {{- if .labels }}
  labels:
    {{- toYaml .labels | nindent 4 }}
  {{- end }}
{{- end }}

{{/*
Create namespace selector for ServiceMonitor
*/}}
{{- define "greenlang-monitoring.namespaceSelector" -}}
{{- if .namespaces }}
namespaceSelector:
  matchNames:
    {{- range .namespaces }}
    - {{ . }}
    {{- end }}
{{- else }}
namespaceSelector:
  any: true
{{- end }}
{{- end }}

{{/*
Thanos sidecar configuration (if using Thanos)
*/}}
{{- define "greenlang-monitoring.thanosSidecar" -}}
{{- if .Values.thanos.enabled }}
sidecarContainers:
  - name: thanos-sidecar
    image: {{ .Values.thanos.image.repository }}:{{ .Values.thanos.image.tag }}
    args:
      - sidecar
      - --prometheus.url=http://127.0.0.1:9090
      - --tsdb.path=/prometheus
      {{- if .Values.thanos.objectStorage.enabled }}
      - --objstore.config-file=/etc/thanos/objstore.yaml
      {{- end }}
    ports:
      - name: http-sidecar
        containerPort: 10902
      - name: grpc
        containerPort: 10901
{{- end }}
{{- end }}
