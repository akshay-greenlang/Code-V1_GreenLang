# GreenLang 5-Minute Quickstart

Get up and running with GreenLang in under 5 minutes. This guide walks you through the complete workflow from initialization to generating your first climate report.

## Prerequisites (30 seconds)

- **Python 3.10+**: `python --version`
- **Git**: For cloning repositories
- **Optional**: Poetry, ORAS CLI for advanced features

## 1. Installation (60 seconds)

```bash
# Clone the repository
git clone https://github.com/greenlang/greenlang
cd greenlang

# Install GreenLang
pip install -e .

# Verify installation
gl --version
gl doctor
```

Expected output:
```
✓ Python 3.11+ detected
✓ Core dependencies installed  
✓ No policy violations detected
✓ Environment ready
```

## 2. Initialize Your First Pack (90 seconds)

```bash
# Create a new climate analysis pack
gl pack create my-analysis --template basic

# Navigate to the new pack
cd packs/my-analysis

# See what was created
ls -la
```

You'll see:
```
pack.yaml          # Pack manifest
gl.yaml            # Main pipeline 
CARD.md            # Documentation
datasets/          # Data files
agents/            # Analysis code
reports/           # Output templates
tests/             # Test suite
```

## 3. Validate Your Pack (30 seconds)

```bash
# Validate the pack structure and manifest
gl pack validate .

# Check for policy compliance
gl policy check .
```

Expected output:
```
✓ Pack validation passed
✓ All required files present
✓ Manifest schema valid
✓ Policy checks passed
```

## 4. Run Your First Analysis (60 seconds)

```bash
# Run the default pipeline
gl run .

# Check the generated outputs
ls -la out/
```

You should see:
```
out/
├── run.json        # Execution metadata
├── report.html     # Interactive report
├── report.pdf      # Executive summary
└── metrics.csv     # Raw results
```

## 5. View Your Results (30 seconds)

```bash
# Open the HTML report
open out/report.html   # macOS
start out/report.html  # Windows
xdg-open out/report.html  # Linux

# View key metrics
cat out/metrics.csv | head -5
```

**🎉 Congratulations!** You've just:
- ✅ Installed GreenLang
- ✅ Created your first pack
- ✅ Validated it passes policy
- ✅ Generated a climate report
- ✅ Viewed the results

## Next Steps (Optional)

### Publish Your Pack
```bash
# Generate SBOM for supply chain security
gl pack publish . --dry-run

# Actually publish (requires ORAS setup)
gl pack publish . --registry ghcr.io/your-org
```

### Install Existing Packs
```bash
# Search available packs
gl pack search solar

# Install a pack
gl pack add greenlang/boiler-solar@1.0.0

# List installed packs
gl pack list
```

### Advanced Usage
```bash
# Run with custom inputs
gl run . --env BUILDING_SIZE=50000 --env LOCATION="New York"

# Run specific agents only
gl run . --agents thermal_analysis,emissions_calc

# Generate deterministic outputs for testing
gl run . --deterministic --artifacts ./test-outputs
```

## 🏃‍♂️ Speed Run Challenge

Can you complete the full workflow in under 3 minutes? Try this:

```bash
# One-liner setup (assumes git clone already done)
pip install -e . && gl doctor && gl pack create speed-test && cd packs/speed-test && gl pack validate . && gl run . && echo "✅ DONE! Check out/report.html"
```

## Common Issues & Solutions

### Issue: `gl: command not found`
```bash
# Solution: Add to PATH or use python -m
python -m greenlang.cli --version
export PATH="$PATH:$HOME/.local/bin"  # Add to ~/.bashrc
```

### Issue: Pack validation fails
```bash
# Solution: Check the specific errors
gl pack validate . --verbose

# Fix common issues
touch datasets/sample.csv  # Add missing datasets
echo "version: 1.0" > gl.yaml  # Fix pipeline config
```

### Issue: Python version too old
```bash
# Solution: Use pyenv or conda
pyenv install 3.11
pyenv local 3.11

# Or with conda
conda create -n greenlang python=3.11
conda activate greenlang
```

### Issue: ORAS not found during publish
```bash
# Solution: Install ORAS CLI
curl -LO "https://github.com/oras-project/oras/releases/latest/download/oras_linux_amd64.tar.gz"
mkdir -p oras-install/
tar -zxf oras_*.tar.gz -C oras-install/
sudo mv oras-install/oras /usr/local/bin/
```

## What Just Happened?

1. **Initialization**: Created a pack with the standard GreenLang structure
2. **Validation**: Verified your pack meets quality and policy requirements  
3. **Execution**: Ran your climate analysis pipeline using the GL runtime
4. **Outputs**: Generated standardized reports (HTML, PDF, CSV)

## Architecture Overview

```
Your Pack Structure:
├── pack.yaml (metadata + dependencies)
├── gl.yaml (pipeline definition)  
├── agents/ (your analysis code)
├── datasets/ (input data)
└── out/ (generated results)

GreenLang Runtime:
Policy Engine → Pack Loader → Agent Runner → Report Generator
```

## Real-World Example: Boiler Optimization

For a more comprehensive example, try the boiler-solar pack:

```bash
# Install from the hub
gl pack add greenlang/boiler-solar@1.0.0

# Run with your building parameters
gl run boiler-solar \
  --env BUILDING_SIZE_SQFT=50000 \
  --env BOILER_FUEL=natural_gas \
  --env LOCATION="Chicago, IL" \
  --env SOLAR_AREA_SQM=500

# View the economic analysis
open out/report.html
```

This will generate a comprehensive report showing:
- Solar thermal integration feasibility
- CO₂ emission reduction potential  
- Economic payback analysis
- Technical recommendations

## Learning Resources

- **Documentation**: https://docs.greenlang.io
- **Examples**: Browse `packs/` directory for real implementations
- **Community**: Join Discord at https://discord.gg/greenlang
- **Videos**: YouTube channel with tutorials and case studies

## Performance Expectations

| Task | Time | Memory |
|------|------|--------|
| Pack creation | 5 seconds | 50MB |
| Validation | 2 seconds | 100MB |
| Simple analysis | 10-30 seconds | 200MB |
| Complex analysis | 1-5 minutes | 500MB-2GB |
| Report generation | 5 seconds | 100MB |

## Support

- **Issues**: https://github.com/greenlang/greenlang/issues
- **Email**: support@greenlang.io  
- **Discord**: #help channel
- **Office Hours**: Tuesdays 2-3pm PT

---

**Total Time**: 4 minutes 30 seconds ⏱️

**Next**: Explore the `/packs` directory to see real-world examples of GreenLang in action!