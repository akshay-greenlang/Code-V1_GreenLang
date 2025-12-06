{{/*
GreenLang MLflow Helm Chart - Template Helpers
*/}}

{{/*
Expand the name of the chart.
*/}}
{{- define "mlflow.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "mlflow.fullname" -}}
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
{{- define "mlflow.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "mlflow.labels" -}}
helm.sh/chart: {{ include "mlflow.chart" . }}
{{ include "mlflow.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: greenlang-mlops
{{- end }}

{{/*
Selector labels
*/}}
{{- define "mlflow.selectorLabels" -}}
app.kubernetes.io/name: {{ include "mlflow.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "mlflow.serviceAccountName" -}}
{{- if .Values.mlflow.serviceAccount.create }}
{{- default (include "mlflow.fullname" .) .Values.mlflow.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.mlflow.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Return the proper MLflow image name
*/}}
{{- define "mlflow.image" -}}
{{- $registryName := .Values.mlflow.image.registry -}}
{{- $repositoryName := .Values.mlflow.image.repository -}}
{{- $tag := .Values.mlflow.image.tag | toString -}}
{{- if .Values.global.imageRegistry }}
    {{- $registryName = .Values.global.imageRegistry -}}
{{- end -}}
{{- if $registryName }}
{{- printf "%s/%s:%s" $registryName $repositoryName $tag -}}
{{- else }}
{{- printf "%s:%s" $repositoryName $tag -}}
{{- end -}}
{{- end }}

{{/*
Return the PostgreSQL hostname
*/}}
{{- define "mlflow.postgresql.host" -}}
{{- if .Values.postgresql.enabled }}
{{- printf "%s-postgresql" (include "mlflow.fullname" .) -}}
{{- else }}
{{- .Values.externalDatabase.host -}}
{{- end }}
{{- end }}

{{/*
Return the PostgreSQL port
*/}}
{{- define "mlflow.postgresql.port" -}}
{{- if .Values.postgresql.enabled }}
{{- print "5432" -}}
{{- else }}
{{- .Values.externalDatabase.port -}}
{{- end }}
{{- end }}

{{/*
Return the PostgreSQL database name
*/}}
{{- define "mlflow.postgresql.database" -}}
{{- if .Values.postgresql.enabled }}
{{- .Values.postgresql.auth.database -}}
{{- else }}
{{- .Values.externalDatabase.database -}}
{{- end }}
{{- end }}

{{/*
Return the PostgreSQL username
*/}}
{{- define "mlflow.postgresql.username" -}}
{{- if .Values.postgresql.enabled }}
{{- .Values.postgresql.auth.username -}}
{{- else }}
{{- .Values.externalDatabase.username -}}
{{- end }}
{{- end }}

{{/*
Return the PostgreSQL secret name
*/}}
{{- define "mlflow.postgresql.secretName" -}}
{{- if .Values.postgresql.enabled }}
{{- if .Values.postgresql.auth.existingSecret }}
{{- .Values.postgresql.auth.existingSecret -}}
{{- else }}
{{- printf "%s-postgresql" (include "mlflow.fullname" .) -}}
{{- end }}
{{- else }}
{{- .Values.externalDatabase.existingSecret -}}
{{- end }}
{{- end }}

{{/*
Return the MinIO/S3 endpoint URL
*/}}
{{- define "mlflow.s3.endpoint" -}}
{{- if .Values.minio.enabled }}
{{- printf "http://%s-minio:9000" (include "mlflow.fullname" .) -}}
{{- else if .Values.externalS3.enabled }}
{{- .Values.externalS3.endpoint -}}
{{- end }}
{{- end }}

{{/*
Return the artifact root
*/}}
{{- define "mlflow.artifactRoot" -}}
{{- if .Values.minio.enabled }}
{{- printf "s3://%s/" .Values.minio.defaultBuckets -}}
{{- else if .Values.externalS3.enabled }}
{{- printf "s3://%s/" .Values.externalS3.bucket -}}
{{- else }}
{{- .Values.mlflow.config.defaultArtifactRoot -}}
{{- end }}
{{- end }}

{{/*
Return the backend store URI
*/}}
{{- define "mlflow.backendStoreUri" -}}
postgresql://$(POSTGRES_USER):$(POSTGRES_PASSWORD)@{{ include "mlflow.postgresql.host" . }}:{{ include "mlflow.postgresql.port" . }}/{{ include "mlflow.postgresql.database" . }}
{{- end }}
