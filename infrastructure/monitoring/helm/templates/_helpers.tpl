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
