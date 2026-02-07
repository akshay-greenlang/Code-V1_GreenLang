{{/*
GreenLang Grafana Helm Chart Helpers
GreenLang Climate OS | OBS-002
*/}}

{{/*
Expand the name of the chart.
*/}}
{{- define "grafana.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "grafana.fullname" -}}
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
{{- define "grafana.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "grafana.labels" -}}
helm.sh/chart: {{ include "grafana.chart" . }}
{{ include "grafana.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: greenlang
{{- end }}

{{/*
Selector labels
*/}}
{{- define "grafana.selectorLabels" -}}
app.kubernetes.io/name: {{ include "grafana.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: grafana
{{- end }}

{{/*
Renderer labels
*/}}
{{- define "grafana.rendererLabels" -}}
helm.sh/chart: {{ include "grafana.chart" . }}
app.kubernetes.io/name: {{ include "grafana.name" . }}-renderer
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: renderer
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: greenlang
{{- end }}

{{/*
Renderer selector labels
*/}}
{{- define "grafana.rendererSelectorLabels" -}}
app.kubernetes.io/name: {{ include "grafana.name" . }}-renderer
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: renderer
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "grafana.serviceAccountName" -}}
{{- if .Values.grafana.serviceAccount.create }}
{{- default (include "grafana.fullname" .) .Values.grafana.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.grafana.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Namespace
*/}}
{{- define "grafana.namespace" -}}
{{- default .Release.Namespace .Values.namespaceOverride }}
{{- end }}
