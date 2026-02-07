{{/*
# =============================================================================
# GreenLang Tempo - Template Helpers
# GreenLang Climate OS | OBS-003
# =============================================================================
# Standard Helm helper templates for name generation, labels, selectors,
# and per-component label variants used across all Tempo templates.
# =============================================================================
*/}}

{{/*
Expand the name of the chart.
*/}}
{{- define "tempo.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "tempo.fullname" -}}
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
{{- define "tempo.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Namespace helper.
*/}}
{{- define "tempo.namespace" -}}
{{- .Release.Namespace }}
{{- end }}

{{/*
Common labels applied to every resource.
*/}}
{{- define "tempo.labels" -}}
helm.sh/chart: {{ include "tempo.chart" . }}
{{ include "tempo.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: greenlang
greenlang.io/obs: "003"
{{- with .Values.commonLabels }}
{{ toYaml . }}
{{- end }}
{{- with .Values.global.labels }}
{{ toYaml . }}
{{- end }}
{{- end }}

{{/*
Selector labels (minimal, used in matchLabels).
*/}}
{{- define "tempo.selectorLabels" -}}
app.kubernetes.io/name: {{ include "tempo.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Component labels - adds component to the common labels.
Usage: {{ include "tempo.componentLabels" (dict "component" "distributor" "root" .) }}
*/}}
{{- define "tempo.componentLabels" -}}
{{ include "tempo.labels" .root }}
app.kubernetes.io/component: {{ .component }}
{{- end }}

{{/*
Component selector labels - adds component to the selector labels.
Usage: {{ include "tempo.componentSelectorLabels" (dict "component" "distributor" "root" .) }}
*/}}
{{- define "tempo.componentSelectorLabels" -}}
{{ include "tempo.selectorLabels" .root }}
app.kubernetes.io/component: {{ .component }}
{{- end }}

{{/*
Create the name of the service account to use.
*/}}
{{- define "tempo.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "tempo.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Distributor labels.
*/}}
{{- define "tempo.distributorLabels" -}}
{{ include "tempo.componentLabels" (dict "component" "distributor" "root" .) }}
{{- end }}

{{/*
Distributor selector labels.
*/}}
{{- define "tempo.distributorSelectorLabels" -}}
{{ include "tempo.componentSelectorLabels" (dict "component" "distributor" "root" .) }}
{{- end }}

{{/*
Ingester labels.
*/}}
{{- define "tempo.ingesterLabels" -}}
{{ include "tempo.componentLabels" (dict "component" "ingester" "root" .) }}
{{- end }}

{{/*
Ingester selector labels.
*/}}
{{- define "tempo.ingesterSelectorLabels" -}}
{{ include "tempo.componentSelectorLabels" (dict "component" "ingester" "root" .) }}
{{- end }}

{{/*
Querier labels.
*/}}
{{- define "tempo.querierLabels" -}}
{{ include "tempo.componentLabels" (dict "component" "querier" "root" .) }}
{{- end }}

{{/*
Querier selector labels.
*/}}
{{- define "tempo.querierSelectorLabels" -}}
{{ include "tempo.componentSelectorLabels" (dict "component" "querier" "root" .) }}
{{- end }}

{{/*
Query Frontend labels.
*/}}
{{- define "tempo.queryFrontendLabels" -}}
{{ include "tempo.componentLabels" (dict "component" "query-frontend" "root" .) }}
{{- end }}

{{/*
Query Frontend selector labels.
*/}}
{{- define "tempo.queryFrontendSelectorLabels" -}}
{{ include "tempo.componentSelectorLabels" (dict "component" "query-frontend" "root" .) }}
{{- end }}

{{/*
Compactor labels.
*/}}
{{- define "tempo.compactorLabels" -}}
{{ include "tempo.componentLabels" (dict "component" "compactor" "root" .) }}
{{- end }}

{{/*
Compactor selector labels.
*/}}
{{- define "tempo.compactorSelectorLabels" -}}
{{ include "tempo.componentSelectorLabels" (dict "component" "compactor" "root" .) }}
{{- end }}

{{/*
Metrics Generator labels.
*/}}
{{- define "tempo.metricsGeneratorLabels" -}}
{{ include "tempo.componentLabels" (dict "component" "metrics-generator" "root" .) }}
{{- end }}

{{/*
Metrics Generator selector labels.
*/}}
{{- define "tempo.metricsGeneratorSelectorLabels" -}}
{{ include "tempo.componentSelectorLabels" (dict "component" "metrics-generator" "root" .) }}
{{- end }}

{{/*
Monolithic labels.
*/}}
{{- define "tempo.monolithicLabels" -}}
{{ include "tempo.componentLabels" (dict "component" "monolithic" "root" .) }}
{{- end }}

{{/*
Monolithic selector labels.
*/}}
{{- define "tempo.monolithicSelectorLabels" -}}
{{ include "tempo.componentSelectorLabels" (dict "component" "monolithic" "root" .) }}
{{- end }}

{{/*
Memberlist labels (for the headless service discovery).
*/}}
{{- define "tempo.memberlistLabels" -}}
{{ include "tempo.labels" . }}
tempo-gossip-member: "true"
{{- end }}

{{/*
Memberlist selector label.
*/}}
{{- define "tempo.memberlistSelectorLabels" -}}
tempo-gossip-member: "true"
{{- end }}

{{/*
Pod anti-affinity helper for component HA.
Usage: {{ include "tempo.podAntiAffinity" (dict "component" "distributor" "root" .) | nindent 8 }}
*/}}
{{- define "tempo.podAntiAffinity" -}}
podAntiAffinity:
  preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      podAffinityTerm:
        labelSelector:
          matchLabels:
            {{- include "tempo.componentSelectorLabels" (dict "component" .component "root" .root) | nindent 12 }}
        topologyKey: kubernetes.io/hostname
    - weight: 50
      podAffinityTerm:
        labelSelector:
          matchLabels:
            {{- include "tempo.componentSelectorLabels" (dict "component" .component "root" .root) | nindent 12 }}
        topologyKey: topology.kubernetes.io/zone
{{- end }}

{{/*
Tempo image string.
*/}}
{{- define "tempo.image" -}}
{{- $registry := .Values.image.registry | default "docker.io" }}
{{- $repository := .Values.image.repository | default "grafana/tempo" }}
{{- $tag := .Values.image.tag | default .Chart.AppVersion }}
{{- printf "%s/%s:%s" $registry $repository $tag }}
{{- end }}
