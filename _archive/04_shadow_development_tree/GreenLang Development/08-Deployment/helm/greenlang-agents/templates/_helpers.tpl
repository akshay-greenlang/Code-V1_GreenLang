{{/*
GreenLang Agent Factory - Template Helpers
*/}}

{{/*
Expand the name of the chart.
*/}}
{{- define "greenlang-agents.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "greenlang-agents.fullname" -}}
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
{{- define "greenlang-agents.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "greenlang-agents.labels" -}}
helm.sh/chart: {{ include "greenlang-agents.chart" . }}
{{ include "greenlang-agents.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: greenlang
{{- with .Values.global.labels }}
{{ toYaml . }}
{{- end }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "greenlang-agents.selectorLabels" -}}
app.kubernetes.io/name: {{ include "greenlang-agents.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "greenlang-agents.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "greenlang-agents.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create the namespace name
*/}}
{{- define "greenlang-agents.namespace" -}}
{{- if .Values.namespace.create }}
{{- default .Release.Namespace .Values.namespace.name }}
{{- else }}
{{- .Release.Namespace }}
{{- end }}
{{- end }}

{{/*
Create image name with registry
*/}}
{{- define "greenlang-agents.image" -}}
{{- $registry := .Values.global.imageRegistry | default "ghcr.io" }}
{{- $repository := .Values.image.repository | default "greenlang" }}
{{- $tag := .Values.image.tag | default "latest" }}
{{- printf "%s/%s:%s" $registry $repository $tag }}
{{- end }}

{{/*
Agent image helper
*/}}
{{- define "greenlang-agents.agentImage" -}}
{{- $registry := .root.Values.global.imageRegistry | default "ghcr.io" }}
{{- $repository := .agent.image.repository }}
{{- $tag := .agent.image.tag | default "latest" }}
{{- printf "%s/%s:%s" $registry $repository $tag }}
{{- end }}

{{/*
Resource limits and requests
*/}}
{{- define "greenlang-agents.resources" -}}
{{- if .resources }}
resources:
  {{- toYaml .resources | nindent 2 }}
{{- else if .common.resources }}
resources:
  {{- toYaml .common.resources | nindent 2 }}
{{- end }}
{{- end }}

{{/*
Pod security context
*/}}
{{- define "greenlang-agents.podSecurityContext" -}}
{{- if .podSecurityContext }}
securityContext:
  {{- toYaml .podSecurityContext | nindent 2 }}
{{- else if .common.podSecurityContext }}
securityContext:
  {{- toYaml .common.podSecurityContext | nindent 2 }}
{{- end }}
{{- end }}

{{/*
Container security context
*/}}
{{- define "greenlang-agents.containerSecurityContext" -}}
{{- if .containerSecurityContext }}
securityContext:
  {{- toYaml .containerSecurityContext | nindent 2 }}
{{- else if .common.containerSecurityContext }}
securityContext:
  {{- toYaml .common.containerSecurityContext | nindent 2 }}
{{- end }}
{{- end }}

{{/*
Liveness probe
*/}}
{{- define "greenlang-agents.livenessProbe" -}}
{{- if .livenessProbe }}
livenessProbe:
  {{- toYaml .livenessProbe | nindent 2 }}
{{- else if .common.livenessProbe }}
livenessProbe:
  {{- toYaml .common.livenessProbe | nindent 2 }}
{{- end }}
{{- end }}

{{/*
Readiness probe
*/}}
{{- define "greenlang-agents.readinessProbe" -}}
{{- if .readinessProbe }}
readinessProbe:
  {{- toYaml .readinessProbe | nindent 2 }}
{{- else if .common.readinessProbe }}
readinessProbe:
  {{- toYaml .common.readinessProbe | nindent 2 }}
{{- end }}
{{- end }}

{{/*
Startup probe
*/}}
{{- define "greenlang-agents.startupProbe" -}}
{{- if .startupProbe }}
startupProbe:
  {{- toYaml .startupProbe | nindent 2 }}
{{- else if .common.startupProbe }}
startupProbe:
  {{- toYaml .common.startupProbe | nindent 2 }}
{{- end }}
{{- end }}
