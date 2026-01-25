{{/*
Expand the name of the chart.
*/}}
{{- define "greenlang.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
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
environment: {{ .Values.global.environment }}
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
{{- if .Values.rbac.serviceAccount.create }}
{{- default (include "greenlang.fullname" .) .Values.rbac.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.rbac.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Executor labels
*/}}
{{- define "greenlang.executor.labels" -}}
{{ include "greenlang.labels" . }}
app: greenlang
component: executor
{{- end }}

{{/*
Worker labels
*/}}
{{- define "greenlang.worker.labels" -}}
{{ include "greenlang.labels" . }}
app: greenlang
component: worker
{{- end }}

{{/*
Redis labels
*/}}
{{- define "greenlang.redis.labels" -}}
{{ include "greenlang.labels" . }}
app: redis-sentinel
component: cache
{{- end }}

{{/*
Image pull secrets
*/}}
{{- define "greenlang.imagePullSecrets" -}}
{{- if .Values.image.pullSecrets }}
imagePullSecrets:
{{- range .Values.image.pullSecrets }}
  - name: {{ . }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Return the proper image name
*/}}
{{- define "greenlang.image" -}}
{{- $registry := .Values.image.registry -}}
{{- $repository := .Values.image.repository -}}
{{- $tag := .Values.image.tag | default .Chart.AppVersion -}}
{{- printf "%s/%s:%s" $registry $repository $tag -}}
{{- end }}

{{/*
Return the proper executor image name
*/}}
{{- define "greenlang.executor.image" -}}
{{- $registry := .Values.image.registry -}}
{{- $repository := .Values.executor.image.repository -}}
{{- $tag := .Values.executor.image.tag | default .Chart.AppVersion -}}
{{- printf "%s/%s:%s" $registry $repository $tag -}}
{{- end }}

{{/*
Return the proper worker image name
*/}}
{{- define "greenlang.worker.image" -}}
{{- $registry := .Values.image.registry -}}
{{- $repository := .Values.worker.image.repository -}}
{{- $tag := .Values.worker.image.tag | default .Chart.AppVersion -}}
{{- printf "%s/%s:%s" $registry $repository $tag -}}
{{- end }}

{{/*
Return the storage class name
*/}}
{{- define "greenlang.storageClass" -}}
{{- .Values.global.storageClass | default "fast-ssd" -}}
{{- end }}

{{/*
Return the namespace
*/}}
{{- define "greenlang.namespace" -}}
{{- .Values.namespace.name | default .Release.Namespace -}}
{{- end }}
