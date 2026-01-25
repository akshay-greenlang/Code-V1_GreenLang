"""
CBAM Pack Web Application

FastAPI-based web interface for the CBAM Compliance Pack.
Provides file upload, processing, and result visualization with:
- XSD schema validation status
- Policy PASS/WARN/FAIL status
- Row-level drilldown for default factor usage
- Gap report with actionable recommendations
- Evidence folder with immutable input copies
"""

import io
import json
import os
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import yaml

from cbam_pack import __version__
from cbam_pack.pipeline import CBAMPipeline, PipelineResult


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="GreenLang CBAM Pack",
        description="EU Carbon Border Adjustment Mechanism Compliance Tool",
        version=__version__,
    )

    # Store for processing results
    app.state.results = {}
    app.state.output_dirs = {}

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request):
        """Render the main page."""
        return get_home_html()

    @app.post("/api/process")
    async def process_files(
        config_file: UploadFile = File(...),
        imports_file: UploadFile = File(...),
        mode: str = Form(default="transitional"),
        collect_errors: bool = Form(default=True),
    ):
        """Process uploaded files and generate CBAM report."""

        # Create temporary directory for processing
        temp_dir = tempfile.mkdtemp(prefix="cbam_")
        output_dir = Path(temp_dir) / "output"
        output_dir.mkdir()

        try:
            # Save uploaded files
            config_path = Path(temp_dir) / config_file.filename
            imports_path = Path(temp_dir) / imports_file.filename

            with open(config_path, "wb") as f:
                content = await config_file.read()
                f.write(content)

            with open(imports_path, "wb") as f:
                content = await imports_file.read()
                f.write(content)

            # Run pipeline
            pipeline = CBAMPipeline(
                config_path=config_path,
                imports_path=imports_path,
                output_dir=output_dir,
                verbose=False,
                dry_run=False,
            )

            result = pipeline.run()

            # Generate session ID
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Store result and output directory
            app.state.results[session_id] = result
            app.state.output_dirs[session_id] = output_dir

            # Build response
            response_data = {
                "success": result.success,
                "session_id": session_id,
                "statistics": result.statistics,
                "artifacts": result.artifacts,
                "errors": result.errors,
            }

            # Add policy status
            if result.policy_result:
                response_data["policy"] = result.policy_result

            # Add XML validation status
            if result.xml_validation:
                response_data["xml_validation"] = result.xml_validation

            # Add gap summary
            if result.gap_summary:
                response_data["gap_summary"] = result.gap_summary

            # Add lines using defaults for drilldown
            if result.lines_using_defaults:
                response_data["lines_using_defaults"] = result.lines_using_defaults

            # Build compliance status based on BOTH schema and policy validation
            # CTO Rule: Schema FAIL = hard block, Policy FAIL = soft (draft allowed)
            xml_val = result.xml_validation or {}
            schema_status = xml_val.get("status", "PASS")
            schema_valid = schema_status == "PASS"

            policy = result.policy_result or {}
            policy_status = policy.get("status", "PASS")
            default_usage = result.statistics.get("default_usage_percent", 0)

            # Determine overall compliance status
            # Schema validation FAIL → Can Export: NO (hard block)
            # Policy validation FAIL → Can Export: YES (draft allowed)
            if not schema_valid:
                compliance_status = "schema_fail"
                compliance_message = "XML Schema Validation FAILED. Report cannot be uploaded to registry. Fix schema errors first."
                can_export = False
                export_label = "INVALID - Cannot Export"
            elif policy_status == "PASS":
                compliance_status = "compliant"
                compliance_message = "All validations passed. Report is compliant and ready for submission."
                can_export = True
                export_label = "Ready for Submission"
            elif policy_status == "WARN":
                compliance_status = "warning"
                compliance_message = f"Report generated with warnings. Default factor usage: {default_usage:.1f}%. Export as draft allowed."
                can_export = True
                export_label = "Draft - Review Warnings"
            else:  # policy FAIL
                compliance_status = "policy_fail"
                compliance_message = "Policy validation failed. Export allowed as draft but NOT COMPLIANT. Review violations before submission."
                can_export = True  # Policy fail = soft fail, draft export allowed
                export_label = "Draft - NOT COMPLIANT"

            response_data["compliance"] = {
                "status": compliance_status,
                "schema_status": schema_status,
                "schema_valid": schema_valid,
                "policy_status": policy_status,
                "default_usage_percent": default_usage,
                "message": compliance_message,
                "can_export": can_export,
                "export_label": export_label,
            }

            return response_data

        except Exception as e:
            # Clean up on error
            shutil.rmtree(temp_dir, ignore_errors=True)
            return {
                "success": False,
                "errors": [str(e)],
                "statistics": {},
                "artifacts": [],
            }

    @app.get("/api/download/{session_id}")
    async def download_all(session_id: str):
        """Download all artifacts as a ZIP file."""

        if session_id not in app.state.output_dirs:
            raise HTTPException(status_code=404, detail="Session not found")

        output_dir = app.state.output_dirs[session_id]

        # Create ZIP in memory
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for file_path in output_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(output_dir)
                    zip_file.write(file_path, arcname)

        zip_buffer.seek(0)

        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename=cbam_report_{session_id}.zip"
            }
        )

    @app.get("/api/download/{session_id}/{filename:path}")
    async def download_file(session_id: str, filename: str):
        """Download a specific artifact file."""

        if session_id not in app.state.output_dirs:
            raise HTTPException(status_code=404, detail="Session not found")

        output_dir = app.state.output_dirs[session_id]
        file_path = output_dir / filename

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        return FileResponse(
            file_path,
            filename=file_path.name,
            media_type="application/octet-stream"
        )

    @app.post("/api/preview-config")
    async def preview_config(config_file: UploadFile = File(...)):
        """Preview config file after upload to verify YAML mapping."""
        try:
            content = await config_file.read()
            config_data = yaml.safe_load(content.decode('utf-8'))

            # Extract preview data
            declarant = config_data.get('declarant', {})
            reporting_period = config_data.get('reporting_period', {})
            representative = config_data.get('representative', {})

            preview = {
                "success": True,
                "declarant": {
                    "name": declarant.get('name', 'Not specified'),
                    "eori_number": declarant.get('eori_number', 'Not specified'),
                },
                "reporting_period": {
                    "quarter": reporting_period.get('quarter', 'Not specified'),
                    "year": reporting_period.get('year', 'Not specified'),
                },
                "representative": {
                    "name": representative.get('name') if representative else None,
                    "eori_number": representative.get('eori_number') if representative else None,
                } if representative else None,
                "mode": config_data.get('mode', 'transitional'),
                "settings": config_data.get('settings', {}),
            }

            return preview

        except yaml.YAMLError as e:
            return {
                "success": False,
                "error": f"Invalid YAML: {str(e)}",
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error parsing config: {str(e)}",
            }

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "version": __version__}

    return app


def get_home_html() -> str:
    """Return the main HTML page."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GreenLang CBAM Pack</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #e0e0e0;
        }

        .container {
            max-width: 1100px;
            margin: 0 auto;
            padding: 40px 20px;
        }

        header {
            text-align: center;
            margin-bottom: 40px;
        }

        .logo {
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        h1 {
            color: #4ecca3;
            font-size: 2em;
            margin-bottom: 10px;
        }

        .subtitle {
            color: #888;
            font-size: 1.1em;
        }

        .card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .upload-zone {
            border: 2px dashed #4ecca3;
            border-radius: 12px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-bottom: 20px;
        }

        .upload-zone:hover {
            background: rgba(78, 204, 163, 0.1);
            border-color: #6ee7b7;
        }

        .upload-zone.dragover {
            background: rgba(78, 204, 163, 0.2);
            border-color: #6ee7b7;
        }

        .upload-zone.has-file {
            border-color: #4ecca3;
            background: rgba(78, 204, 163, 0.1);
        }

        .upload-icon {
            font-size: 3em;
            margin-bottom: 15px;
        }

        .upload-text {
            font-size: 1.1em;
            margin-bottom: 10px;
        }

        .upload-hint {
            color: #888;
            font-size: 0.9em;
        }

        .file-name {
            color: #4ecca3;
            font-weight: bold;
            margin-top: 10px;
        }

        input[type="file"] {
            display: none;
        }

        .btn {
            background: linear-gradient(135deg, #4ecca3 0%, #45b393 100%);
            color: #1a1a2e;
            border: none;
            padding: 15px 40px;
            font-size: 1.1em;
            font-weight: bold;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            width: 100%;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(78, 204, 163, 0.3);
        }

        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        .progress-container {
            display: none;
            margin-top: 20px;
        }

        .progress-bar {
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4ecca3, #45b393);
            width: 0%;
            transition: width 0.3s ease;
        }

        .progress-text {
            text-align: center;
            margin-top: 10px;
            color: #888;
        }

        .results {
            display: none;
        }

        .results.show {
            display: block;
        }

        .stat-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .stat-card {
            background: rgba(78, 204, 163, 0.1);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
        }

        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #4ecca3;
        }

        .stat-label {
            color: #888;
            margin-top: 5px;
        }

        .validation-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .validation-card {
            background: rgba(255, 255, 255, 0.03);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .validation-card h4 {
            margin-bottom: 15px;
            color: #fff;
        }

        .validation-status {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }

        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: bold;
        }

        .status-badge.pass {
            background: rgba(78, 204, 163, 0.2);
            color: #4ecca3;
        }

        .status-badge.warn {
            background: rgba(255, 193, 7, 0.2);
            color: #ffc107;
        }

        .status-badge.fail {
            background: rgba(220, 53, 69, 0.2);
            color: #dc3545;
        }

        .validation-detail {
            color: #888;
            font-size: 0.9em;
        }

        .compliance-badge {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            margin-bottom: 20px;
        }

        .compliance-badge.compliant {
            background: rgba(78, 204, 163, 0.2);
            color: #4ecca3;
        }

        .compliance-badge.warning {
            background: rgba(255, 193, 7, 0.2);
            color: #ffc107;
        }

        .compliance-badge.error {
            background: rgba(220, 53, 69, 0.2);
            color: #dc3545;
        }

        .section-title {
            margin: 30px 0 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .drilldown-section {
            margin-top: 20px;
        }

        .drilldown-toggle {
            background: rgba(255, 193, 7, 0.1);
            border: 1px solid rgba(255, 193, 7, 0.3);
            padding: 15px;
            border-radius: 8px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .drilldown-toggle:hover {
            background: rgba(255, 193, 7, 0.15);
        }

        .drilldown-content {
            display: none;
            margin-top: 10px;
        }

        .drilldown-content.show {
            display: block;
        }

        .default-line-item {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 10px;
        }

        .default-line-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }

        .default-line-id {
            font-weight: bold;
            color: #4ecca3;
        }

        .default-line-emissions {
            color: #ffc107;
        }

        .default-line-details {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-bottom: 10px;
            font-size: 0.9em;
            color: #888;
        }

        .missing-fields {
            margin-top: 10px;
            padding: 10px;
            background: rgba(220, 53, 69, 0.1);
            border-radius: 4px;
        }

        .missing-fields-title {
            font-weight: bold;
            color: #dc3545;
            margin-bottom: 5px;
        }

        .missing-field-tag {
            display: inline-block;
            background: rgba(220, 53, 69, 0.2);
            color: #dc3545;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            margin: 2px;
        }

        .recommended-action {
            margin-top: 10px;
            padding: 10px;
            background: rgba(78, 204, 163, 0.1);
            border-radius: 4px;
            color: #4ecca3;
            font-size: 0.9em;
        }

        .artifacts-list {
            list-style: none;
        }

        .artifacts-list li {
            padding: 12px 15px;
            background: rgba(255, 255, 255, 0.05);
            margin-bottom: 8px;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .download-link {
            color: #4ecca3;
            text-decoration: none;
            padding: 5px 15px;
            border: 1px solid #4ecca3;
            border-radius: 4px;
            transition: all 0.3s ease;
        }

        .download-link:hover {
            background: #4ecca3;
            color: #1a1a2e;
        }

        .btn-download-all {
            margin-top: 20px;
        }

        .error-list {
            background: rgba(220, 53, 69, 0.1);
            border: 1px solid rgba(220, 53, 69, 0.3);
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
        }

        .error-item {
            padding: 10px;
            margin-bottom: 10px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 4px;
        }

        .error-code {
            color: #dc3545;
            font-weight: bold;
        }

        .violation-list {
            margin-top: 10px;
        }

        .violation-item {
            background: rgba(220, 53, 69, 0.1);
            border-left: 3px solid #dc3545;
            padding: 10px 15px;
            margin-bottom: 8px;
            border-radius: 0 8px 8px 0;
        }

        .warning-item {
            background: rgba(255, 193, 7, 0.1);
            border-left: 3px solid #ffc107;
            padding: 10px 15px;
            margin-bottom: 8px;
            border-radius: 0 8px 8px 0;
        }

        footer {
            text-align: center;
            padding: 40px;
            color: #666;
        }

        footer a {
            color: #4ecca3;
            text-decoration: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo">&#127807;</div>
            <h1>GreenLang CBAM Pack</h1>
            <p class="subtitle">EU Carbon Border Adjustment Mechanism Compliance Tool</p>
        </header>

        <div class="card">
            <h2 style="margin-bottom: 20px;">&#128193; Upload Files</h2>

            <div class="upload-zone" id="configZone" onclick="document.getElementById('configFile').click()">
                <div class="upload-icon">&#128196;</div>
                <div class="upload-text">Config File (YAML)</div>
                <div class="upload-hint">Drag & drop or click to select</div>
                <div class="file-name" id="configFileName"></div>
                <input type="file" id="configFile" accept=".yaml,.yml" onchange="handleFileSelect(this, 'config')">
            </div>

            <div class="upload-zone" id="importsZone" onclick="document.getElementById('importsFile').click()">
                <div class="upload-icon">&#128202;</div>
                <div class="upload-text">Import Ledger (CSV/XLSX)</div>
                <div class="upload-hint">Drag & drop or click to select</div>
                <div class="file-name" id="importsFileName"></div>
                <input type="file" id="importsFile" accept=".csv,.xlsx,.xls" onchange="handleFileSelect(this, 'imports')">
            </div>

            <!-- Config Preview Section -->
            <div id="configPreview" style="display: none; background: rgba(78, 204, 163, 0.1); border: 1px solid rgba(78, 204, 163, 0.3); border-radius: 12px; padding: 20px; margin-bottom: 20px;">
                <h4 style="color: #4ecca3; margin-bottom: 15px;">&#10003; Config Loaded</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                    <div>
                        <div style="color: #888; font-size: 0.85em;">Declarant</div>
                        <div id="previewDeclarant" style="font-weight: bold;">-</div>
                    </div>
                    <div>
                        <div style="color: #888; font-size: 0.85em;">EORI Number</div>
                        <div id="previewEORI" style="font-weight: bold;">-</div>
                    </div>
                    <div>
                        <div style="color: #888; font-size: 0.85em;">Reporting Period</div>
                        <div id="previewPeriod" style="font-weight: bold;">-</div>
                    </div>
                    <div>
                        <div style="color: #888; font-size: 0.85em;">Mode</div>
                        <div id="previewMode" style="font-weight: bold;">-</div>
                    </div>
                </div>
                <div id="previewRepresentative" style="margin-top: 15px; display: none;">
                    <div style="color: #888; font-size: 0.85em;">Representative</div>
                    <div id="previewRepName" style="font-weight: bold;">-</div>
                </div>
                <div id="previewError" style="color: #dc3545; margin-top: 10px; display: none;"></div>
            </div>

            <button class="btn" id="processBtn" onclick="processFiles()" disabled>
                &#9654; Generate Report
            </button>

            <div class="progress-container" id="progressContainer">
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
                <div class="progress-text" id="progressText">Processing...</div>
            </div>
        </div>

        <div class="results card" id="resultsCard">
            <h2 style="margin-bottom: 20px;">&#128202; Results</h2>

            <div id="complianceStatus"></div>

            <!-- Validation Status Section -->
            <div class="validation-grid" id="validationGrid"></div>

            <!-- Statistics Section -->
            <div class="stat-grid" id="statsGrid"></div>

            <!-- Lines Using Defaults Drilldown -->
            <div class="drilldown-section" id="drilldownSection" style="display: none;">
                <div class="drilldown-toggle" onclick="toggleDrilldown()">
                    <span>&#9888; <span id="defaultLinesCount">0</span> lines using default factors - Click to review</span>
                    <span id="drilldownArrow">&#9660;</span>
                </div>
                <div class="drilldown-content" id="drilldownContent"></div>
            </div>

            <!-- Policy Violations/Warnings -->
            <div id="policySection"></div>

            <h3 class="section-title">&#128230; Generated Artifacts</h3>
            <ul class="artifacts-list" id="artifactsList"></ul>

            <button class="btn btn-download-all" id="downloadAllBtn" onclick="downloadAll()">
                &#11015; Download All (ZIP)
            </button>

            <div class="error-list" id="errorList" style="display: none;">
                <h3 style="margin-bottom: 15px; color: #dc3545;">&#9888; Errors</h3>
                <div id="errorItems"></div>
            </div>
        </div>

        <footer>
            <p>GreenLang CBAM Pack v''' + __version__ + '''</p>
            <p><a href="https://greenlang.in" target="_blank">greenlang.in</a></p>
        </footer>
    </div>

    <script>
        let configFile = null;
        let importsFile = null;
        let currentSessionId = null;

        // Drag and drop handling
        ['configZone', 'importsZone'].forEach(id => {
            const zone = document.getElementById(id);

            zone.addEventListener('dragover', (e) => {
                e.preventDefault();
                zone.classList.add('dragover');
            });

            zone.addEventListener('dragleave', () => {
                zone.classList.remove('dragover');
            });

            zone.addEventListener('drop', (e) => {
                e.preventDefault();
                zone.classList.remove('dragover');

                const file = e.dataTransfer.files[0];
                if (file) {
                    const type = id === 'configZone' ? 'config' : 'imports';
                    const input = document.getElementById(type === 'config' ? 'configFile' : 'importsFile');

                    const dt = new DataTransfer();
                    dt.items.add(file);
                    input.files = dt.files;

                    handleFileSelect(input, type);
                }
            });
        });

        async function handleFileSelect(input, type) {
            const file = input.files[0];
            if (!file) return;

            const zone = document.getElementById(type === 'config' ? 'configZone' : 'importsZone');
            const nameEl = document.getElementById(type === 'config' ? 'configFileName' : 'importsFileName');

            zone.classList.add('has-file');
            nameEl.textContent = '&#10003; ' + file.name;

            if (type === 'config') {
                configFile = file;
                // Fetch config preview
                await previewConfigFile(file);
            } else {
                importsFile = file;
            }

            updateProcessButton();
        }

        async function previewConfigFile(file) {
            const previewSection = document.getElementById('configPreview');
            const previewError = document.getElementById('previewError');

            try {
                const formData = new FormData();
                formData.append('config_file', file);

                const response = await fetch('/api/preview-config', {
                    method: 'POST',
                    body: formData
                });

                const preview = await response.json();

                if (preview.success) {
                    document.getElementById('previewDeclarant').textContent = preview.declarant.name;
                    document.getElementById('previewEORI').textContent = preview.declarant.eori_number;
                    document.getElementById('previewPeriod').textContent =
                        `${preview.reporting_period.quarter} ${preview.reporting_period.year}`;
                    document.getElementById('previewMode').textContent =
                        preview.mode.charAt(0).toUpperCase() + preview.mode.slice(1);

                    if (preview.representative && preview.representative.name) {
                        document.getElementById('previewRepName').textContent =
                            `${preview.representative.name} (${preview.representative.eori_number || 'No EORI'})`;
                        document.getElementById('previewRepresentative').style.display = 'block';
                    } else {
                        document.getElementById('previewRepresentative').style.display = 'none';
                    }

                    previewError.style.display = 'none';
                    previewSection.style.display = 'block';
                } else {
                    previewError.textContent = preview.error;
                    previewError.style.display = 'block';
                    previewSection.style.display = 'block';
                }
            } catch (error) {
                previewError.textContent = 'Failed to preview config: ' + error.message;
                previewError.style.display = 'block';
                previewSection.style.display = 'block';
            }
        }

        function updateProcessButton() {
            const btn = document.getElementById('processBtn');
            btn.disabled = !(configFile && importsFile);
        }

        async function processFiles() {
            const btn = document.getElementById('processBtn');
            const progressContainer = document.getElementById('progressContainer');
            const progressFill = document.getElementById('progressFill');
            const progressText = document.getElementById('progressText');
            const resultsCard = document.getElementById('resultsCard');

            btn.disabled = true;
            progressContainer.style.display = 'block';
            resultsCard.classList.remove('show');

            let progress = 0;
            const stages = [
                'Validating inputs...',
                'Evaluating policy...',
                'Loading emission factors...',
                'Calculating emissions...',
                'Validating XML schema...',
                'Creating audit bundle...',
                'Generating gap report...',
                'Finalizing...'
            ];

            const progressInterval = setInterval(() => {
                progress += 12;
                if (progress > 90) progress = 90;
                progressFill.style.width = progress + '%';
                progressText.textContent = stages[Math.min(Math.floor(progress / 12), stages.length - 1)];
            }, 300);

            const formData = new FormData();
            formData.append('config_file', configFile);
            formData.append('imports_file', importsFile);
            formData.append('collect_errors', 'true');

            try {
                const response = await fetch('/api/process', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                clearInterval(progressInterval);
                progressFill.style.width = '100%';
                progressText.textContent = result.success ? 'Complete!' : 'Completed with issues';

                currentSessionId = result.session_id;
                displayResults(result);

            } catch (error) {
                clearInterval(progressInterval);
                progressText.textContent = 'Error: ' + error.message;
            }

            btn.disabled = false;
        }

        function displayResults(result) {
            const resultsCard = document.getElementById('resultsCard');
            const complianceStatus = document.getElementById('complianceStatus');
            const validationGrid = document.getElementById('validationGrid');
            const statsGrid = document.getElementById('statsGrid');
            const artifactsList = document.getElementById('artifactsList');
            const errorList = document.getElementById('errorList');
            const errorItems = document.getElementById('errorItems');
            const drilldownSection = document.getElementById('drilldownSection');
            const policySection = document.getElementById('policySection');

            resultsCard.classList.add('show');

            // Compliance status with schema/policy distinction
            if (result.success) {
                const compliance = result.compliance || { status: 'compliant', message: 'Report generated', can_export: true };

                // Map status to display
                let statusIcon, statusText, badgeClass;
                switch (compliance.status) {
                    case 'compliant':
                        statusIcon = '&#10003;';
                        statusText = 'Compliant';
                        badgeClass = 'compliant';
                        break;
                    case 'warning':
                        statusIcon = '&#9888;';
                        statusText = 'Review Required';
                        badgeClass = 'warning';
                        break;
                    case 'schema_fail':
                        statusIcon = '&#10060;';
                        statusText = 'SCHEMA FAIL - Cannot Submit';
                        badgeClass = 'error';
                        break;
                    case 'policy_fail':
                        statusIcon = '&#9888;';
                        statusText = 'Policy Failed (Draft OK)';
                        badgeClass = 'warning';
                        break;
                    default:
                        statusIcon = '&#10007;';
                        statusText = 'Error';
                        badgeClass = 'error';
                }

                complianceStatus.innerHTML = `
                    <div class="compliance-badge ${badgeClass}">
                        ${statusIcon} ${statusText}
                    </div>
                    <p style="margin-bottom: 20px; color: #888;">${compliance.message}</p>
                    ${compliance.export_label ? `<div style="margin-bottom: 15px;">
                        <span style="padding: 4px 12px; border-radius: 4px; font-size: 0.85em; font-weight: bold;
                            background: ${compliance.can_export ? 'rgba(78, 204, 163, 0.2)' : 'rgba(220, 53, 69, 0.2)'};
                            color: ${compliance.can_export ? '#4ecca3' : '#dc3545'};">
                            ${compliance.export_label}
                        </span>
                    </div>` : ''}
                `;

                // Update download button based on export permission
                const downloadAllBtn = document.getElementById('downloadAllBtn');
                if (!compliance.can_export) {
                    downloadAllBtn.disabled = true;
                    downloadAllBtn.innerHTML = '&#128683; Download Blocked - Fix Schema Errors First';
                    downloadAllBtn.style.background = 'linear-gradient(135deg, #dc3545 0%, #c82333 100%)';
                    downloadAllBtn.style.cursor = 'not-allowed';
                } else if (compliance.status === 'policy_fail' || compliance.status === 'warning') {
                    downloadAllBtn.disabled = false;
                    downloadAllBtn.innerHTML = '&#11015; Download All (ZIP) - DRAFT';
                    downloadAllBtn.style.background = 'linear-gradient(135deg, #ffc107 0%, #e0a800 100%)';
                    downloadAllBtn.style.cursor = 'pointer';
                } else {
                    downloadAllBtn.disabled = false;
                    downloadAllBtn.innerHTML = '&#11015; Download All (ZIP)';
                    downloadAllBtn.style.background = 'linear-gradient(135deg, #4ecca3 0%, #45b393 100%)';
                    downloadAllBtn.style.cursor = 'pointer';
                }
            } else {
                complianceStatus.innerHTML = `
                    <div class="compliance-badge error">&#10007; Failed</div>
                    <p style="margin-bottom: 20px; color: #888;">Report generation failed. See errors below.</p>
                `;
                const downloadAllBtn = document.getElementById('downloadAllBtn');
                downloadAllBtn.disabled = true;
                downloadAllBtn.innerHTML = '&#128683; Download Not Available';
                downloadAllBtn.style.background = 'linear-gradient(135deg, #dc3545 0%, #c82333 100%)';
                downloadAllBtn.style.cursor = 'not-allowed';
            }

            // Validation cards
            let validationHtml = '';

            // XML Schema Validation
            const xmlVal = result.xml_validation || {};
            validationHtml += `
                <div class="validation-card">
                    <h4>&#128196; XML Schema Validation</h4>
                    <div class="validation-status">
                        <span class="status-badge ${xmlVal.status === 'PASS' ? 'pass' : xmlVal.status === 'FAIL' ? 'fail' : 'warn'}">
                            ${xmlVal.status || 'N/A'}
                        </span>
                    </div>
                    <div class="validation-detail">
                        Schema Version: ${xmlVal.schema_version || 'N/A'}<br>
                        Schema Date: ${xmlVal.schema_date || 'N/A'}
                    </div>
                    ${xmlVal.errors && xmlVal.errors.length > 0 ?
                        `<div style="color: #dc3545; margin-top: 10px; font-size: 0.85em;">
                            ${xmlVal.errors.map(e => `&#8226; ${e}`).join('<br>')}
                        </div>` : ''}
                </div>
            `;

            // Policy Validation
            const policy = result.policy || {};
            const compliance = result.compliance || {};
            validationHtml += `
                <div class="validation-card">
                    <h4>&#128736; Policy Validation</h4>
                    <div class="validation-status">
                        <span class="status-badge ${policy.status === 'PASS' ? 'pass' : policy.status === 'WARN' ? 'warn' : 'fail'}">
                            ${policy.status || 'N/A'}
                        </span>
                    </div>
                    <div class="validation-detail">
                        Score: ${policy.overall_score ? policy.overall_score.toFixed(0) : 'N/A'}/100<br>
                        Policy Allows Export: ${policy.can_export !== undefined ? (policy.can_export ? 'Yes (Draft)' : 'No') : 'N/A'}
                    </div>
                </div>
            `;

            // Export Eligibility Summary Card
            validationHtml += `
                <div class="validation-card" style="grid-column: 1 / -1; background: ${compliance.can_export ? 'rgba(78, 204, 163, 0.1)' : 'rgba(220, 53, 69, 0.1)'}; border-color: ${compliance.can_export ? 'rgba(78, 204, 163, 0.3)' : 'rgba(220, 53, 69, 0.3)'};">
                    <h4>&#128230; Export Eligibility</h4>
                    <div class="validation-status">
                        <span class="status-badge ${compliance.can_export ? 'pass' : 'fail'}">
                            ${compliance.can_export ? 'CAN EXPORT' : 'BLOCKED'}
                        </span>
                    </div>
                    <div class="validation-detail">
                        Schema: ${compliance.schema_valid ? 'Valid' : 'INVALID (hard fail)'}<br>
                        Policy: ${compliance.policy_status || 'N/A'} (soft - draft allowed)<br>
                        <strong>Status: ${compliance.export_label || 'Unknown'}</strong>
                    </div>
                </div>
            `;

            validationGrid.innerHTML = validationHtml;

            // Statistics
            const stats = result.statistics || {};
            statsGrid.innerHTML = `
                <div class="stat-card">
                    <div class="stat-value">${stats.total_lines || 0}</div>
                    <div class="stat-label">Lines Processed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${(stats.total_emissions_tco2e || 0).toFixed(2)}</div>
                    <div class="stat-label">Total tCO2e</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${(stats.default_usage_percent || 0).toFixed(1)}%</div>
                    <div class="stat-label">Default Factor Usage</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${stats.lines_using_defaults || 0}</div>
                    <div class="stat-label">Lines Using Defaults</div>
                </div>
            `;

            // Lines using defaults drilldown
            const defaultLines = result.lines_using_defaults || [];
            if (defaultLines.length > 0) {
                drilldownSection.style.display = 'block';
                document.getElementById('defaultLinesCount').textContent = defaultLines.length;

                let drilldownHtml = '';
                defaultLines.forEach(line => {
                    drilldownHtml += `
                        <div class="default-line-item">
                            <div class="default-line-header">
                                <span class="default-line-id">${line.line_id}</span>
                                <span class="default-line-emissions">${line.total_emissions_tco2e.toFixed(2)} tCO2e</span>
                            </div>
                            <div class="default-line-details">
                                <span>&#128230; CN Code: ${line.cn_code}</span>
                                <span>&#127758; Country: ${line.country_of_origin}</span>
                                <span>&#128666; Supplier: ${line.supplier_id || 'Unknown'}</span>
                            </div>
                            <div style="color: #888; font-size: 0.9em; margin-bottom: 10px;">
                                ${line.product_description}
                            </div>
                            ${line.missing_fields && line.missing_fields.length > 0 ? `
                                <div class="missing-fields">
                                    <div class="missing-fields-title">Missing Data Fields:</div>
                                    ${line.missing_fields.map(f => `<span class="missing-field-tag">${f}</span>`).join('')}
                                </div>
                            ` : ''}
                            <div class="recommended-action">
                                &#128161; ${line.recommended_action}
                            </div>
                        </div>
                    `;
                });
                document.getElementById('drilldownContent').innerHTML = drilldownHtml;
            } else {
                drilldownSection.style.display = 'none';
            }

            // Policy violations and warnings
            let policyHtml = '';
            if (policy.violations && policy.violations.length > 0) {
                policyHtml += '<h3 class="section-title" style="color: #dc3545;">&#10060; Policy Violations</h3>';
                policyHtml += '<div class="violation-list">';
                policy.violations.forEach(v => {
                    policyHtml += `
                        <div class="violation-item">
                            <strong>${v.rule_id}: ${v.rule_name}</strong><br>
                            <span style="color: #888;">${v.message}</span>
                            ${v.remediation ? `<br><span style="color: #4ecca3;">&#128161; ${v.remediation}</span>` : ''}
                        </div>
                    `;
                });
                policyHtml += '</div>';
            }
            if (policy.warnings && policy.warnings.length > 0) {
                policyHtml += '<h3 class="section-title" style="color: #ffc107;">&#9888; Policy Warnings</h3>';
                policyHtml += '<div class="violation-list">';
                policy.warnings.forEach(w => {
                    policyHtml += `
                        <div class="warning-item">
                            <strong>${w.rule_id}: ${w.rule_name}</strong><br>
                            <span style="color: #888;">${w.message}</span>
                            ${w.remediation ? `<br><span style="color: #4ecca3;">&#128161; ${w.remediation}</span>` : ''}
                        </div>
                    `;
                });
                policyHtml += '</div>';
            }
            policySection.innerHTML = policyHtml;

            // Artifacts
            const artifacts = result.artifacts || [];
            artifactsList.innerHTML = artifacts.map(artifact => `
                <li>
                    <span>&#128196; ${artifact}</span>
                    <a href="/api/download/${currentSessionId}/${artifact}" class="download-link">Download</a>
                </li>
            `).join('');

            // Errors
            const errors = result.errors || [];
            if (errors.length > 0) {
                errorList.style.display = 'block';
                errorItems.innerHTML = errors.map(error => `
                    <div class="error-item">
                        <span class="error-code">${error}</span>
                    </div>
                `).join('');
            } else {
                errorList.style.display = 'none';
            }
        }

        function toggleDrilldown() {
            const content = document.getElementById('drilldownContent');
            const arrow = document.getElementById('drilldownArrow');
            content.classList.toggle('show');
            arrow.innerHTML = content.classList.contains('show') ? '&#9650;' : '&#9660;';
        }

        function downloadAll() {
            if (currentSessionId) {
                window.location.href = `/api/download/${currentSessionId}`;
            }
        }
    </script>
</body>
</html>'''


# Run function for CLI
def run_web_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the web server."""
    import uvicorn
    app = create_app()
    uvicorn.run(app, host=host, port=port)
