{{/*
GL-003 UNIFIEDSTEAM - Helm Template Helpers
*/}}

{{/*
Expand the name of the chart.
*/}}
{{- define "unifiedsteam.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "unifiedsteam.fullname" -}}
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
{{- define "unifiedsteam.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "unifiedsteam.labels" -}}
helm.sh/chart: {{ include "unifiedsteam.chart" . }}
{{ include "unifiedsteam.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: greenlang
agent-id: gl-003
tier: steam-systems
{{- end }}

{{/*
Selector labels
*/}}
{{- define "unifiedsteam.selectorLabels" -}}
app.kubernetes.io/name: {{ include "unifiedsteam.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "unifiedsteam.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "unifiedsteam.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Return the proper image name
*/}}
{{- define "unifiedsteam.image" -}}
{{- $registryName := .Values.image.registry -}}
{{- $repositoryName := .Values.image.repository -}}
{{- $tag := .Values.image.tag | default .Chart.AppVersion -}}
{{- if $registryName }}
{{- printf "%s/%s:%s" $registryName $repositoryName $tag -}}
{{- else }}
{{- printf "%s:%s" $repositoryName $tag -}}
{{- end }}
{{- end }}

{{/*
Return the Kafka brokers
*/}}
{{- define "unifiedsteam.kafkaBrokers" -}}
{{- if .Values.kafka.enabled }}
{{- printf "%s-kafka-headless:9092" .Release.Name }}
{{- else }}
{{- .Values.config.kafka.brokers }}
{{- end }}
{{- end }}

{{/*
Return the Redis host
*/}}
{{- define "unifiedsteam.redisHost" -}}
{{- if .Values.redis.enabled }}
{{- printf "%s-redis-master" .Release.Name }}
{{- else }}
{{- .Values.config.redis.host }}
{{- end }}
{{- end }}

{{/*
Return the PostgreSQL host
*/}}
{{- define "unifiedsteam.postgresHost" -}}
{{- if .Values.postgresql.enabled }}
{{- printf "%s-postgresql" .Release.Name }}
{{- else }}
{{- .Values.config.postgres.host | default "postgres" }}
{{- end }}
{{- end }}

{{/*
Create a checksum annotation for config changes
*/}}
{{- define "unifiedsteam.configChecksum" -}}
{{- include (print $.Template.BasePath "/configmap.yaml") . | sha256sum }}
{{- end }}
