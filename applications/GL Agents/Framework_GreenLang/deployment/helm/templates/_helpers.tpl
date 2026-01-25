{{/*
GreenLang Agent Helm Chart - Template Helpers
Template Version: 1.0.0
*/}}

{{/*
Expand the name of the chart.
*/}}
{{- define "greenlang-agent.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "greenlang-agent.fullname" -}}
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
{{- define "greenlang-agent.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "greenlang-agent.labels" -}}
helm.sh/chart: {{ include "greenlang-agent.chart" . }}
{{ include "greenlang-agent.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: greenlang
greenlang.io/agent-id: {{ .Values.agent.id | quote }}
greenlang.io/agent-group: {{ .Values.agent.group | quote }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "greenlang-agent.selectorLabels" -}}
app.kubernetes.io/name: {{ include "greenlang-agent.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "greenlang-agent.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "greenlang-agent.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create the image name
*/}}
{{- define "greenlang-agent.image" -}}
{{- $registry := .Values.global.imageRegistry | default "" -}}
{{- $repository := .Values.image.repository -}}
{{- $tag := .Values.image.tag | default .Chart.AppVersion -}}
{{- $digest := .Values.image.digest | default "" -}}
{{- if $registry }}
{{- if $digest }}
{{- printf "%s/%s@%s" $registry $repository $digest }}
{{- else }}
{{- printf "%s/%s:%s" $registry $repository $tag }}
{{- end }}
{{- else }}
{{- if $digest }}
{{- printf "%s@%s" $repository $digest }}
{{- else }}
{{- printf "%s:%s" $repository $tag }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create ConfigMap name
*/}}
{{- define "greenlang-agent.configMapName" -}}
{{- printf "%s-config" (include "greenlang-agent.fullname" .) }}
{{- end }}

{{/*
Create Secret name
*/}}
{{- define "greenlang-agent.secretName" -}}
{{- printf "%s-secrets" (include "greenlang-agent.fullname" .) }}
{{- end }}

{{/*
Create PDB name
*/}}
{{- define "greenlang-agent.pdbName" -}}
{{- printf "%s-pdb" (include "greenlang-agent.fullname" .) }}
{{- end }}

{{/*
Create HPA name
*/}}
{{- define "greenlang-agent.hpaName" -}}
{{- printf "%s-hpa" (include "greenlang-agent.fullname" .) }}
{{- end }}

{{/*
Create Ingress host
*/}}
{{- define "greenlang-agent.ingressHost" -}}
{{- printf "%s.greenlang.io" .Values.agent.id }}
{{- end }}

{{/*
Return the proper image pull secrets
*/}}
{{- define "greenlang-agent.imagePullSecrets" -}}
{{- $pullSecrets := list }}
{{- if .Values.global.imagePullSecrets }}
{{- range .Values.global.imagePullSecrets }}
{{- $pullSecrets = append $pullSecrets . }}
{{- end }}
{{- end }}
{{- if .Values.imagePullSecrets }}
{{- range .Values.imagePullSecrets }}
{{- $pullSecrets = append $pullSecrets . }}
{{- end }}
{{- end }}
{{- if $pullSecrets }}
imagePullSecrets:
{{- range $pullSecrets }}
  - name: {{ .name }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Return the proper storage class
*/}}
{{- define "greenlang-agent.storageClass" -}}
{{- if .Values.global.storageClass }}
{{- .Values.global.storageClass }}
{{- else if .Values.persistence.storageClass }}
{{- .Values.persistence.storageClass }}
{{- end }}
{{- end }}

{{/*
Generate environment variables from values
*/}}
{{- define "greenlang-agent.env" -}}
{{- range $key, $value := .Values.env }}
- name: {{ $key }}
  value: {{ $value | quote }}
{{- end }}
{{- if .Values.extraEnv }}
{{- toYaml .Values.extraEnv }}
{{- end }}
{{- end }}

{{/*
Generate container ports
*/}}
{{- define "greenlang-agent.ports" -}}
{{- range .Values.ports }}
- name: {{ .name }}
  containerPort: {{ .containerPort }}
  protocol: {{ .protocol | default "TCP" }}
{{- end }}
{{- end }}

{{/*
Check if we should create RBAC resources
*/}}
{{- define "greenlang-agent.createRbac" -}}
{{- and .Values.serviceAccount.create .Values.rbac.create }}
{{- end }}

{{/*
Return checksum of configmap for deployment annotation
*/}}
{{- define "greenlang-agent.configChecksum" -}}
{{- if .Values.configMap.enabled }}
checksum/config: {{ include (print $.Template.BasePath "/configmap.yaml") . | sha256sum }}
{{- end }}
{{- end }}

{{/*
Return checksum of secret for deployment annotation
*/}}
{{- define "greenlang-agent.secretChecksum" -}}
{{- if .Values.secrets.enabled }}
checksum/secret: {{ include (print $.Template.BasePath "/secret.yaml") . | sha256sum }}
{{- end }}
{{- end }}
