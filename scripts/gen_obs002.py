import json, os

BASE = "C:/Users/aksha/Code-V1_GreenLang"
DASH_DIR = f"{BASE}/deployment/monitoring/dashboards"
K8S_DIR = f"{BASE}/deployment/kubernetes/grafana"
os.makedirs(K8S_DIR, exist_ok=True)

def ds(uid):
    return {"type": "prometheus", "uid": uid}

def std_templating(ds_uid="thanos", ds_name="Thanos"):
    return {"list": [
        {"current": {"selected": False, "text": ds_name, "value": ds_uid}, "hide": 0, "includeAll": False, "multi": False, "name": "datasource", "options": [], "query": "prometheus", "refresh": 1, "type": "datasource"},
        {"allValue": ".*", "current": {"selected": False, "text": "All", "value": "$__all"}, "datasource": ds(ds_uid), "definition": "label_values(up, namespace)", "hide": 0, "includeAll": True, "multi": True, "name": "namespace", "query": {"query": "label_values(up, namespace)", "refId": "StandardVariableQuery"}, "refresh": 2, "regex": "", "sort": 1, "type": "query"},
        {"auto": True, "auto_count": 30, "auto_min": "10s", "current": {"selected": False, "text": "auto", "value": "$__auto_interval_interval"}, "hide": 0, "name": "interval", "options": [{"selected": True, "text": "auto", "value": "$__auto_interval_interval"}, {"selected": False, "text": "1m", "value": "1m"}, {"selected": False, "text": "5m", "value": "5m"}, {"selected": False, "text": "15m", "value": "15m"}, {"selected": False, "text": "1h", "value": "1h"}], "query": "1m,5m,15m,1h", "refresh": 2, "type": "interval"}
    ]}

def base_db(uid, title, desc, tags, panels, ds_uid="thanos", ds_name="Thanos", tr=None):
    if tr is None:
        tr = {"from": "now-6h", "to": "now"}
    return {"annotations": {"list": [{"builtIn": 1, "datasource": "-- Grafana --", "enable": True, "hide": True, "iconColor": "rgba(0, 211, 255, 1)", "name": "Annotations & Alerts", "type": "dashboard"}]}, "description": desc, "editable": True, "fiscalYearStartMonth": 0, "graphTooltip": 1, "id": None, "links": [{"asDropdown": True, "icon": "external link", "includeVars": True, "keepTime": True, "tags": tags[:2], "targetBlank": True, "title": title, "type": "dashboards"}], "liveNow": False, "panels": panels, "refresh": "30s", "schemaVersion": 39, "tags": tags, "templating": std_templating(ds_uid, ds_name), "time": tr, "timepicker": {"refresh_intervals": ["10s","30s","1m","5m","15m","30m","1h"], "time_options": ["5m","15m","1h","6h","12h","24h","2d","7d","30d"]}, "timezone": "utc", "title": title, "uid": uid, "version": 1, "weekStart": "monday"}

def rw(id, title, y):
    return {"collapsed": False, "gridPos": {"h": 1, "w": 24, "x": 0, "y": y}, "id": id, "title": title, "type": "row"}

def sp(id, title, desc, expr, g, unit="short", th=None, dsu="thanos"):
    if th is None:
        th = [{"color": "green", "value": None}]
    return {"datasource": ds(dsu), "description": desc, "fieldConfig": {"defaults": {"color": {"mode": "thresholds"}, "mappings": [], "thresholds": {"mode": "absolute", "steps": th}, "unit": unit}, "overrides": []}, "gridPos": g, "id": id, "options": {"colorMode": "background", "graphMode": "area", "justifyMode": "auto", "orientation": "auto", "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}, "textMode": "auto"}, "title": title, "type": "stat", "targets": [{"datasource": ds(dsu), "expr": expr, "legendFormat": title, "refId": "A", "instant": True}]}

def ga(id, title, desc, expr, g, unit="percent", dsu="thanos", mn=0, mx=100, th=None):
    if th is None:
        th = [{"color": "green", "value": None}, {"color": "yellow", "value": 50}, {"color": "red", "value": 80}]
    return {"datasource": ds(dsu), "description": desc, "fieldConfig": {"defaults": {"color": {"mode": "thresholds"}, "mappings": [], "thresholds": {"mode": "absolute", "steps": th}, "unit": unit, "min": mn, "max": mx}, "overrides": []}, "gridPos": g, "id": id, "options": {"orientation": "auto", "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}, "showThresholdLabels": False, "showThresholdMarkers": True}, "title": title, "type": "gauge", "targets": [{"datasource": ds(dsu), "expr": expr, "legendFormat": title, "refId": "A", "instant": True}]}

def ts(id, title, desc, tl, g, unit="short", dsu="thanos"):
    tgts = [{"datasource": ds(dsu), "expr": e, "legendFormat": l, "refId": chr(65+i)} for i,(e,l) in enumerate(tl)]
    return {"datasource": ds(dsu), "description": desc, "fieldConfig": {"defaults": {"color": {"mode": "palette-classic"}, "custom": {"axisCenteredZero": False, "axisColorMode": "text", "axisPlacement": "auto", "drawStyle": "line", "fillOpacity": 10, "gradientMode": "scheme", "lineInterpolation": "smooth", "lineWidth": 2, "showPoints": "never", "spanNulls": True, "stacking": {"group": "A", "mode": "none"}, "thresholdsStyle": {"mode": "off"}}, "unit": unit}, "overrides": []}, "gridPos": g, "id": id, "options": {"legend": {"calcs": ["mean","max","lastNotNull"], "displayMode": "table", "placement": "bottom"}, "tooltip": {"mode": "multi", "sort": "desc"}}, "title": title, "type": "timeseries", "targets": tgts}

def tb(id, title, desc, expr, g, dsu="thanos"):
    return {"datasource": ds(dsu), "description": desc, "fieldConfig": {"defaults": {"color": {"mode": "thresholds"}, "custom": {"align": "auto", "displayMode": "auto", "inspect": False}, "mappings": [], "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None}]}}, "overrides": []}, "gridPos": g, "id": id, "options": {"showHeader": True, "sortBy": [{"desc": True, "displayName": "Value"}], "footer": {"show": False}}, "title": title, "type": "table", "targets": [{"datasource": ds(dsu), "expr": expr, "legendFormat": "", "refId": "A", "instant": True, "format": "table"}]}

def pie(id, title, desc, expr, g, dsu="thanos", lf="{{ severity }}"):
    return {"datasource": ds(dsu), "description": desc, "fieldConfig": {"defaults": {"color": {"mode": "palette-classic"}, "mappings": []}, "overrides": []}, "gridPos": g, "id": id, "options": {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}, "pieType": "donut", "tooltip": {"mode": "multi", "sort": "desc"}, "legend": {"displayMode": "table", "placement": "right", "values": ["value", "percent"]}}, "title": title, "type": "piechart", "targets": [{"datasource": ds(dsu), "expr": expr, "legendFormat": lf, "refId": "A", "instant": True}]}

def bgp(id, title, desc, expr, g, unit="bytes", dsu="thanos"):
    return {"datasource": ds(dsu), "description": desc, "fieldConfig": {"defaults": {"color": {"mode": "thresholds"}, "custom": {"fillOpacity": 80, "gradientMode": "scheme", "lineWidth": 0}, "thresholds": {"mode": "percentage", "steps": [{"color": "green", "value": None}, {"color": "yellow", "value": 70}, {"color": "orange", "value": 85}, {"color": "red", "value": 95}]}, "unit": unit, "min": 0}, "overrides": []}, "gridPos": g, "id": id, "options": {"displayMode": "gradient", "minVizHeight": 10, "orientation": "horizontal", "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}, "showUnfilled": True}, "title": title, "type": "bargauge", "targets": [{"datasource": ds(dsu), "expr": expr, "legendFormat": "{{ persistentvolumeclaim }}", "refId": "A", "instant": True}]}

print("Helper functions defined")

p1 = [
    rw(1,"Platform Health",0),
    sp(2,"Platform Uptime (24h)","Overall platform uptime.","avg_over_time(up{job=~\".*greenlang.*\"}[24h]) * 100",{"h":6,"w":4,"x":0,"y":1},"percent",[{"color":"red","value":None},{"color":"orange","value":95},{"color":"yellow","value":99},{"color":"green","value":99.9}]),
    sp(3,"Active Agents","Total running agents.","gl_agent_factory_agents_total{state=\"running\"}",{"h":6,"w":4,"x":4,"y":1},"short",[{"color":"red","value":None},{"color":"yellow","value":10},{"color":"green","value":30}]),
    sp(4,"Active Alerts","Firing alerts.","count(ALERTS{alertstate=\"firing\"}) or vector(0)",{"h":6,"w":4,"x":8,"y":1},"short",[{"color":"green","value":None},{"color":"yellow","value":1},{"color":"orange","value":5},{"color":"red","value":10}]),
    ga(5,"Error Rate","Error rate percentage.","sum(rate(http_server_requests_total{status_code=~\"5..\"}[5m])) / sum(rate(http_server_requests_total[5m])) * 100",{"h":6,"w":4,"x":12,"y":1},"percent",th=[{"color":"green","value":None},{"color":"yellow","value":1},{"color":"orange","value":5},{"color":"red","value":10}]),
    ts(6,"API Request Rate","API throughput.",[("sum(rate(http_server_requests_total[5m]))","Total RPS"),("sum(rate(http_server_requests_total{status_code=~\"5..\"}[5m]))","5xx Errors")],{"h":6,"w":8,"x":16,"y":1},"reqps"),
    rw(7,"Performance & Resources",7),
    ts(8,"P99 Latency","P99/P95/P50 latency.",[("histogram_quantile(0.99, sum(rate(http_server_request_duration_seconds_bucket[5m])) by (le))","P99"),("histogram_quantile(0.95, sum(rate(http_server_request_duration_seconds_bucket[5m])) by (le))","P95"),("histogram_quantile(0.50, sum(rate(http_server_request_duration_seconds_bucket[5m])) by (le))","P50")],{"h":8,"w":12,"x":0,"y":8},"s"),
    bgp(9,"Storage Usage by PVC","Storage utilization.","sum(kubelet_volume_stats_used_bytes{namespace=~\"$namespace\"}) by (persistentvolumeclaim)",{"h":8,"w":12,"x":12,"y":8}),
]
with open(f"{DASH_DIR}/platform-overview.json","w") as f:
    json.dump(base_db("platform-overview","Platform Overview","GreenLang Climate OS - Executive Platform Overview.",["executive","greenlang","overview","platform"],p1,tr={"from":"now-24h","to":"now"}),f,indent=2)
print("1/6 platform-overview.json")

p2 = [
    rw(1,"Grafana Server",0),
    sp(2,"Grafana Build Info","Grafana version.","grafana_build_info",{"h":4,"w":4,"x":0,"y":1},"short",[{"color":"green","value":None}],"prometheus"),
    sp(3,"Active Users","Active Grafana users.","grafana_stat_active_users",{"h":4,"w":4,"x":4,"y":1},"short",[{"color":"green","value":None},{"color":"yellow","value":50}],"prometheus"),
    sp(4,"Dashboard Load Time P95","P95 dashboard load.","histogram_quantile(0.95, rate(grafana_api_dashboard_get_milliseconds_bucket[5m]))",{"h":4,"w":4,"x":8,"y":1},"ms",[{"color":"green","value":None},{"color":"yellow","value":2000},{"color":"red","value":3000}],"prometheus"),
    ts(5,"API Request Rate","Grafana API rate.",[("rate(grafana_http_request_total[5m])","Requests/s")],{"h":4,"w":6,"x":12,"y":1},"reqps","prometheus"),
    ts(6,"API Errors","Grafana 5xx errors.",[("rate(grafana_http_request_total{status_code=~\"5..\"}[5m])","5xx/s")],{"h":4,"w":6,"x":18,"y":1},"short","prometheus"),
    rw(7,"Backend & Alerting",5),
    ga(8,"DB Connection Pool","DB connection pool usage.","grafana_database_conn_open / grafana_database_conn_max * 100",{"h":5,"w":4,"x":0,"y":6},"percent","prometheus",th=[{"color":"green","value":None},{"color":"yellow","value":70},{"color":"red","value":90}]),
    ts(9,"Alerting Queue","Alerting queue capacity.",[("grafana_alerting_queue_capacity","Queue")],{"h":5,"w":4,"x":4,"y":6},"short","prometheus"),
    ts(10,"Rendering Duration","Image rendering.",[("rate(grafana_rendering_request_total[5m])","Renders/s")],{"h":5,"w":4,"x":8,"y":6},"short","prometheus"),
    tb(11,"Data Source Health","Data source status.","sum(rate(grafana_datasource_request_total[5m])) by (datasource, status)",{"h":5,"w":12,"x":12,"y":6},"prometheus"),
    rw(12,"System Resources",11),
    ts(13,"Memory Usage","Grafana memory.",[("process_resident_memory_bytes{job=\"grafana\"}","Memory")],{"h":6,"w":6,"x":0,"y":12},"bytes","prometheus"),
    ts(14,"CPU Usage","Grafana CPU.",[("rate(process_cpu_seconds_total{job=\"grafana\"}[5m])","CPU")],{"h":6,"w":6,"x":6,"y":12},"percentunit","prometheus"),
    ga(15,"Cache Hit Rate","Cache hit rate.","grafana_remote_cache_hit_total / (grafana_remote_cache_hit_total + grafana_remote_cache_miss_total) * 100",{"h":6,"w":4,"x":12,"y":12},"percent","prometheus",th=[{"color":"red","value":None},{"color":"yellow","value":50},{"color":"green","value":70}]),
    sp(16,"Session Count","Active sessions.","grafana_stat_active_sessions",{"h":3,"w":4,"x":16,"y":12},"short",[{"color":"green","value":None}],"prometheus"),
    sp(17,"Dashboard Count","Total dashboards.","grafana_stat_total_dashboards",{"h":3,"w":4,"x":20,"y":12},"short",[{"color":"green","value":None},{"color":"yellow","value":150},{"color":"red","value":200}],"prometheus"),
    sp(18,"Alert Rule Count","Total alert rules.","grafana_stat_total_alert_rules",{"h":3,"w":4,"x":16,"y":15},"short",[{"color":"green","value":None}],"prometheus"),
    tb(19,"Plugin Status","Plugin info.","grafana_plugin_build_info",{"h":3,"w":4,"x":20,"y":15},"prometheus"),
]
with open(f"{DASH_DIR}/grafana-health.json","w") as f:
    json.dump(base_db("grafana-health","Grafana Health","GreenLang Climate OS - Grafana self-monitoring dashboard.",["grafana","health","observability","greenlang"],p2,"prometheus","Prometheus"),f,indent=2)
print("2/6 grafana-health.json")

p3 = [
    rw(1,"User Activity",0),
    ts(2,"Daily Active Users","Active users over time.",[("grafana_stat_active_users","Active Users")],{"h":7,"w":8,"x":0,"y":1},"short","prometheus"),
    ts(3,"Dashboard Views by Folder","Dashboard views by folder.",[("sum(rate(grafana_api_dashboard_get_milliseconds_count[5m])) by (folder)","{{ folder }}")],{"h":7,"w":8,"x":8,"y":1},"reqps","prometheus"),
    tb(4,"Top 10 Dashboards","Most accessed dashboards.","topk(10, sum(increase(grafana_api_dashboard_get_milliseconds_count[24h])) by (dashboard))",{"h":7,"w":8,"x":16,"y":1},"prometheus"),
    rw(5,"Data Sources & Performance",8),
    ts(6,"Data Source Query Rate","Query rate per datasource.",[("sum(rate(grafana_datasource_request_total[5m])) by (datasource)","{{ datasource }}")],{"h":7,"w":8,"x":0,"y":9},"reqps","prometheus"),
    tb(7,"Slow Queries","High P99 latency queries.","histogram_quantile(0.99, sum(rate(grafana_datasource_request_duration_seconds_bucket[5m])) by (le, datasource))",{"h":7,"w":8,"x":8,"y":9},"prometheus"),
    ts(8,"Login Activity","Login attempts.",[("rate(grafana_api_login_post_total[1h])","Logins/h")],{"h":7,"w":4,"x":16,"y":9},"short","prometheus"),
    ts(9,"Alert Notification Rate","Notifications sent.",[("rate(grafana_alerting_notification_sent_total[1h])","Notifications/h")],{"h":7,"w":4,"x":20,"y":9},"short","prometheus"),
]
with open(f"{DASH_DIR}/grafana-usage.json","w") as f:
    json.dump(base_db("grafana-usage","Grafana Usage Analytics","GreenLang Climate OS - Grafana usage analytics.",["grafana","usage","analytics","greenlang"],p3,"prometheus","Prometheus"),f,indent=2)
print("3/6 grafana-usage.json")

p4 = [
    rw(1,"Security Overview",0),
    ga(2,"Security Score","Composite security score.","(avg(gl_auth_success_rate) * 0.2 + avg(1 - gl_rbac_denial_rate) * 0.15 + avg(gl_encryption_success_rate) * 0.15 + avg(gl_tls_compliance_score) * 0.15 + avg(gl_audit_completeness) * 0.1 + avg(gl_soc2_compliance_score) * 0.15 + avg(1 - gl_vulnerability_risk_score) * 0.1) * 100",{"h":7,"w":4,"x":0,"y":1},"percent",th=[{"color":"red","value":None},{"color":"orange","value":60},{"color":"yellow","value":80},{"color":"green","value":90}]),
    ts(3,"Auth Success/Failure","Auth attempts.",[("rate(gl_auth_total{status=\"success\"}[5m])","Success"),("rate(gl_auth_total{status=\"failure\"}[5m])","Failure")],{"h":7,"w":5,"x":4,"y":1},"reqps"),
    ts(4,"RBAC Denials","RBAC denials.",[("rate(gl_rbac_authorization_total{decision=\"deny\"}[5m])","Denials/s")],{"h":7,"w":5,"x":9,"y":1},"reqps"),
    sp(5,"Active Vulnerabilities","Open vulnerabilities.","sum(gl_security_vulnerabilities_total) by (severity)",{"h":7,"w":5,"x":14,"y":1},"short",[{"color":"green","value":None},{"color":"yellow","value":5},{"color":"orange","value":20},{"color":"red","value":50}]),
    sp(6,"SOC 2 Compliance","SOC 2 score.","gl_soc2_compliance_score",{"h":7,"w":5,"x":19,"y":1},"percent",[{"color":"red","value":None},{"color":"orange","value":60},{"color":"yellow","value":80},{"color":"green","value":95}]),
    rw(7,"Security Operations",8),
    ts(8,"Encryption Operations","Encryption ops.",[("rate(gl_encryption_operations_total[5m])","Ops/s")],{"h":7,"w":6,"x":0,"y":9},"reqps"),
    tb(9,"TLS Certificate Expiry","TLS cert expiration.","gl_tls_certificate_expiry_seconds",{"h":7,"w":6,"x":6,"y":9}),
    ts(10,"Audit Events","Audit event rate.",[("rate(gl_audit_events_total[5m])","Events/s")],{"h":7,"w":6,"x":12,"y":9},"reqps"),
    ts(11,"PII Detections","PII detection rate.",[("rate(gl_pii_detections_total[5m])","Detections/s")],{"h":7,"w":6,"x":18,"y":9},"reqps"),
]
with open(f"{DASH_DIR}/security-posture.json","w") as f:
    json.dump(base_db("security-posture","Security Posture","GreenLang Climate OS - Unified security posture dashboard.",["security","posture","greenlang","compliance"],p4),f,indent=2)
print("4/6 security-posture.json")

p5 = [
    rw(1,"Alert Summary",0),
    sp(2,"Firing Alerts","Firing alerts.","count(ALERTS{alertstate=\"firing\"}) or vector(0)",{"h":5,"w":4,"x":0,"y":1},"short",[{"color":"green","value":None},{"color":"yellow","value":1},{"color":"orange","value":5},{"color":"red","value":10}]),
    sp(3,"Pending Alerts","Pending alerts.","count(ALERTS{alertstate=\"pending\"}) or vector(0)",{"h":5,"w":4,"x":4,"y":1},"short",[{"color":"green","value":None},{"color":"yellow","value":5},{"color":"orange","value":15}]),
    pie(4,"Alerts by Severity","Firing alerts by severity.","count(ALERTS{alertstate=\"firing\"}) by (severity)",{"h":5,"w":8,"x":8,"y":1}),
    sp(5,"Silenced Alerts","Silenced alerts.","count(alertmanager_silences{state=\"active\"}) or vector(0)",{"h":5,"w":4,"x":16,"y":1},"short",[{"color":"green","value":None}]),
    sp(50,"Total Alert Rules","Total rules.","count(ALERTS) or vector(0)",{"h":5,"w":4,"x":20,"y":1},"short",[{"color":"green","value":None}]),
    rw(6,"Alert History & Details",6),
    ts(7,"Alert Timeline","Alert state changes.",[("changes(ALERTS{alertstate=\"firing\"}[1h])","State Changes")],{"h":8,"w":12,"x":0,"y":7},"short"),
    tb(8,"Alert History","Recent firing events.","sort_desc(count_over_time(ALERTS{alertstate=\"firing\"}[24h]))",{"h":8,"w":12,"x":12,"y":7}),
    tb(9,"Top Firing Rules","Most frequent firing rules.","topk(20, count_over_time(ALERTS{alertstate=\"firing\"}[24h]))",{"h":8,"w":24,"x":0,"y":15}),
]
with open(f"{DASH_DIR}/active-alerts.json","w") as f:
    json.dump(base_db("active-alerts","Active Alerts","GreenLang Climate OS - Active alert summary dashboard.",["alerts","monitoring","greenlang","observability"],p5),f,indent=2)
print("5/6 active-alerts.json")

p6 = [
    rw(1,"Application Overview",0),
    sp(2,"Total Applications","Deployed services.","count(count by (job) (up{namespace=~\"$namespace\"}))",{"h":5,"w":4,"x":0,"y":1},"short",[{"color":"green","value":None}]),
    ga(3,"Healthy Apps","Healthy app percentage.","count(up{namespace=~\"$namespace\"} == 1) / count(up{namespace=~\"$namespace\"}) * 100",{"h":5,"w":4,"x":4,"y":1},"percent",th=[{"color":"red","value":None},{"color":"orange","value":80},{"color":"yellow","value":95},{"color":"green","value":99}]),
    ts(4,"Request Rate per App","Request rate by app.",[("sum(rate(http_server_requests_total{namespace=~\"$namespace\"}[5m])) by (job)","{{ job }}")],{"h":5,"w":8,"x":8,"y":1},"reqps"),
    ts(5,"Error Rate per App","Error rate by app.",[("sum(rate(http_server_requests_total{namespace=~\"$namespace\",status_code=~\"5..\"}[5m])) by (job)","{{ job }}")],{"h":5,"w":8,"x":16,"y":1},"reqps"),
    rw(6,"Resource Usage",6),
    ts(7,"Latency by App","P99 latency per app.",[("histogram_quantile(0.99, sum(rate(http_server_request_duration_seconds_bucket{namespace=~\"$namespace\"}[5m])) by (le, job))","{{ job }}")],{"h":7,"w":8,"x":0,"y":7},"s"),
    ts(8,"Memory by App","Memory per pod.",[("sum(container_memory_working_set_bytes{namespace=~\"$namespace\",container!=\"\"}) by (pod)","{{ pod }}")],{"h":7,"w":8,"x":8,"y":7},"bytes"),
    ts(9,"CPU by App","CPU per pod.",[("sum(rate(container_cpu_usage_seconds_total{namespace=~\"$namespace\",container!=\"\"}[5m])) by (pod)","{{ pod }}")],{"h":7,"w":4,"x":16,"y":7},"short"),
    ts(10,"Pod Restarts","Pod restart count.",[("sum(increase(kube_pod_container_status_restarts_total{namespace=~\"$namespace\"}[1h])) by (pod)","{{ pod }}")],{"h":7,"w":4,"x":20,"y":7},"short"),
]
with open(f"{DASH_DIR}/application-health.json","w") as f:
    json.dump(base_db("application-health","Application Health","GreenLang Climate OS - Application health overview.",["application","health","greenlang","overview"],p6),f,indent=2)
print("6/6 application-health.json")
print("All 6 dashboard JSON files written!")
