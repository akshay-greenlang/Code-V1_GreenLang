{{/*
Expand the name of the chart.
*/}}
{{- define "greenlang.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
*/}}
{{- define "greenlang.fullname" -}}
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
{{- define "greenlang.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "greenlang.labels" -}}
helm.sh/chart: {{ include "greenlang.chart" . }}
{{ include "greenlang.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: greenlang-platform
{{- end }}

{{/*
Selector labels
*/}}
{{- define "greenlang.selectorLabels" -}}
app.kubernetes.io/name: {{ include "greenlang.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "greenlang.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "greenlang.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Agent Runtime labels
*/}}
{{- define "greenlang.agentRuntime.labels" -}}
{{ include "greenlang.labels" . }}
app.kubernetes.io/component: agent-runtime
{{- end }}

{{/*
Agent Runtime selector labels
*/}}
{{- define "greenlang.agentRuntime.selectorLabels" -}}
{{ include "greenlang.selectorLabels" . }}
app.kubernetes.io/component: agent-runtime
{{- end }}

{{/*
API Server labels
*/}}
{{- define "greenlang.apiServer.labels" -}}
{{ include "greenlang.labels" . }}
app.kubernetes.io/component: api-server
{{- end }}

{{/*
API Server selector labels
*/}}
{{- define "greenlang.apiServer.selectorLabels" -}}
{{ include "greenlang.selectorLabels" . }}
app.kubernetes.io/component: api-server
{{- end }}

{{/*
Registry labels
*/}}
{{- define "greenlang.registry.labels" -}}
{{ include "greenlang.labels" . }}
app.kubernetes.io/component: registry
{{- end }}

{{/*
Registry selector labels
*/}}
{{- define "greenlang.registry.selectorLabels" -}}
{{ include "greenlang.selectorLabels" . }}
app.kubernetes.io/component: registry
{{- end }}

{{/*
Get the image name for a component
*/}}
{{- define "greenlang.image" -}}
{{- $registry := .global.imageRegistry -}}
{{- $repository := .image.repository -}}
{{- $tag := .image.tag | default $.Chart.AppVersion -}}
{{- printf "%s/%s:%s" $registry $repository $tag -}}
{{- end }}

{{/*
Create the namespace name
*/}}
{{- define "greenlang.namespace" -}}
{{- if .Values.namespace.create }}
{{- .Values.namespace.name | default (printf "%s-production" (include "greenlang.name" .)) }}
{{- else }}
{{- .Release.Namespace }}
{{- end }}
{{- end }}
