"""
CBAM Pack Web Application

FastAPI-based web interface for the CBAM Compliance Pack.
Provides file upload, processing, and result visualization.
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

            # Add compliance status
            if result.success:
                default_usage = result.statistics.get("default_usage_percent", 0)
                response_data["compliance"] = {
                    "status": "compliant" if default_usage <= 20 else "warning",
                    "default_usage_percent": default_usage,
                    "message": f"Default factor usage: {default_usage:.1f}%"
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
            max-width: 900px;
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
            <div class="logo">🌿</div>
            <h1>GreenLang CBAM Pack</h1>
            <p class="subtitle">EU Carbon Border Adjustment Mechanism Compliance Tool</p>
        </header>

        <div class="card">
            <h2 style="margin-bottom: 20px;">📁 Upload Files</h2>

            <div class="upload-zone" id="configZone" onclick="document.getElementById('configFile').click()">
                <div class="upload-icon">📄</div>
                <div class="upload-text">Config File (YAML)</div>
                <div class="upload-hint">Drag & drop or click to select</div>
                <div class="file-name" id="configFileName"></div>
                <input type="file" id="configFile" accept=".yaml,.yml" onchange="handleFileSelect(this, 'config')">
            </div>

            <div class="upload-zone" id="importsZone" onclick="document.getElementById('importsFile').click()">
                <div class="upload-icon">📊</div>
                <div class="upload-text">Import Ledger (CSV/XLSX)</div>
                <div class="upload-hint">Drag & drop or click to select</div>
                <div class="file-name" id="importsFileName"></div>
                <input type="file" id="importsFile" accept=".csv,.xlsx,.xls" onchange="handleFileSelect(this, 'imports')">
            </div>

            <button class="btn" id="processBtn" onclick="processFiles()" disabled>
                ▶ Generate Report
            </button>

            <div class="progress-container" id="progressContainer">
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
                <div class="progress-text" id="progressText">Processing...</div>
            </div>
        </div>

        <div class="results card" id="resultsCard">
            <h2 style="margin-bottom: 20px;">📊 Results</h2>

            <div id="complianceStatus"></div>

            <div class="stat-grid" id="statsGrid"></div>

            <h3 style="margin-bottom: 15px;">📦 Generated Artifacts</h3>
            <ul class="artifacts-list" id="artifactsList"></ul>

            <button class="btn btn-download-all" id="downloadAllBtn" onclick="downloadAll()">
                ⬇ Download All (ZIP)
            </button>

            <div class="error-list" id="errorList" style="display: none;">
                <h3 style="margin-bottom: 15px; color: #dc3545;">⚠ Errors</h3>
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

                    // Create a DataTransfer to set files
                    const dt = new DataTransfer();
                    dt.items.add(file);
                    input.files = dt.files;

                    handleFileSelect(input, type);
                }
            });
        });

        function handleFileSelect(input, type) {
            const file = input.files[0];
            if (!file) return;

            const zone = document.getElementById(type === 'config' ? 'configZone' : 'importsZone');
            const nameEl = document.getElementById(type === 'config' ? 'configFileName' : 'importsFileName');

            zone.classList.add('has-file');
            nameEl.textContent = '✓ ' + file.name;

            if (type === 'config') {
                configFile = file;
            } else {
                importsFile = file;
            }

            updateProcessButton();
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

            // Simulate progress
            let progress = 0;
            const stages = [
                'Validating inputs...',
                'Loading emission factors...',
                'Calculating emissions...',
                'Generating XML...',
                'Creating audit bundle...',
                'Finalizing...'
            ];

            const progressInterval = setInterval(() => {
                progress += 15;
                if (progress > 90) progress = 90;
                progressFill.style.width = progress + '%';
                progressText.textContent = stages[Math.min(Math.floor(progress / 15), stages.length - 1)];
            }, 300);

            // Create form data
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
                progressText.textContent = result.success ? 'Complete!' : 'Completed with errors';

                currentSessionId = result.session_id;

                // Display results
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
            const statsGrid = document.getElementById('statsGrid');
            const artifactsList = document.getElementById('artifactsList');
            const errorList = document.getElementById('errorList');
            const errorItems = document.getElementById('errorItems');

            resultsCard.classList.add('show');

            // Compliance status
            if (result.success) {
                const compliance = result.compliance || { status: 'compliant', message: 'Report generated successfully' };
                complianceStatus.innerHTML = `
                    <div class="compliance-badge ${compliance.status}">
                        ${compliance.status === 'compliant' ? '✓ Compliant' : '⚠ Review Required'}
                    </div>
                    <p style="margin-bottom: 20px; color: #888;">${compliance.message}</p>
                `;
            } else {
                complianceStatus.innerHTML = `
                    <div class="compliance-badge error">✗ Failed</div>
                    <p style="margin-bottom: 20px; color: #888;">Report generation failed. See errors below.</p>
                `;
            }

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
            `;

            // Artifacts
            const artifacts = result.artifacts || [];
            artifactsList.innerHTML = artifacts.map(artifact => `
                <li>
                    <span>📄 ${artifact}</span>
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
