{{/*
Expand the name of the chart.
*/}}
{{- define "patroni.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "patroni.fullname" -}}
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
{{- define "patroni.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "patroni.labels" -}}
helm.sh/chart: {{ include "patroni.chart" . }}
{{ include "patroni.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: greenlang
{{- end }}

{{/*
Selector labels
*/}}
{{- define "patroni.selectorLabels" -}}
app: patroni
app.kubernetes.io/name: {{ include "patroni.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
cluster-name: {{ .Values.clusterName }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "patroni.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "patroni.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Credentials secret name
*/}}
{{- define "patroni.credentialsSecretName" -}}
{{- if .Values.credentials.existingSecret }}
{{- .Values.credentials.existingSecret }}
{{- else }}
{{- printf "%s-credentials" (include "patroni.fullname" .) }}
{{- end }}
{{- end }}

{{/*
pgBackRest S3 secret name
*/}}
{{- define "patroni.backupSecretName" -}}
{{- if .Values.backup.existingS3Secret }}
{{- .Values.backup.existingS3Secret }}
{{- else }}
{{- printf "%s-backup-s3" (include "patroni.fullname" .) }}
{{- end }}
{{- end }}

{{/*
Generate random password
*/}}
{{- define "patroni.randomPassword" -}}
{{- randAlphaNum 32 }}
{{- end }}

{{/*
Primary service name
*/}}
{{- define "patroni.primaryServiceName" -}}
{{- printf "%s-primary" (include "patroni.fullname" .) }}
{{- end }}

{{/*
Replica service name
*/}}
{{- define "patroni.replicaServiceName" -}}
{{- printf "%s-replica" (include "patroni.fullname" .) }}
{{- end }}

{{/*
Headless service name
*/}}
{{- define "patroni.headlessServiceName" -}}
{{- include "patroni.fullname" . }}
{{- end }}

{{/*
ConfigMap name
*/}}
{{- define "patroni.configMapName" -}}
{{- printf "%s-config" (include "patroni.fullname" .) }}
{{- end }}
