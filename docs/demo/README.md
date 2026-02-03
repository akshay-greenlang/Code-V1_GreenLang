# GreenLang CBAM Demo

This directory contains all materials needed to record, edit, and publish an animated terminal demonstration of the GreenLang CBAM workflow.

## Overview

The demo showcases a complete CBAM (Carbon Border Adjustment Mechanism) compliance workflow:

1. **Validate** input shipment data
2. **Calculate** embedded emissions with full breakdown
3. **Verify** provenance and reproducibility
4. **Export** regulatory-ready declarations

**Duration:** ~4 minutes
**Target Audience:** Sustainability managers, compliance officers, developers, auditors

---

## Directory Structure

```
docs/demo/
├── README.md                      # This file
├── demo-script.sh                 # Executable demo script
├── narration.md                   # Voice-over narration script
├── asciinema-config.json          # Recording configuration
├── sample-data/
│   └── cbam_shipment.json         # Sample input data
└── expected-output/
    └── emissions.json             # Expected calculation output
```

---

## Prerequisites

### Required Tools

```bash
# Install asciinema (terminal recorder)
# macOS
brew install asciinema

# Ubuntu/Debian
sudo apt install asciinema

# pip (any platform)
pip install asciinema

# Optional: pv for typing effect
brew install pv  # macOS
sudo apt install pv  # Ubuntu
```

### Optional Tools

```bash
# For GIF conversion
npm install -g svg-term-cli
pip install termtosvg

# For video conversion
brew install ffmpeg
```

---

## Recording the Demo

### Quick Start

```bash
# Navigate to demo directory
cd docs/demo

# Make script executable
chmod +x demo-script.sh

# Record the demo
asciinema rec -c "./demo-script.sh" demo.cast
```

### Recording with Configuration

```bash
# Record with custom settings
asciinema rec \
  --cols 120 \
  --rows 35 \
  --idle-time-limit 2 \
  --title "GreenLang CBAM Workflow Demo" \
  -c "./demo-script.sh" \
  demo.cast
```

### Best Practices

1. **Clean Environment**: Start with a fresh terminal session
2. **Font Size**: Use 14-16pt monospace font for readability
3. **Colors**: Ensure high contrast (dark background, light text)
4. **Resolution**: Record at 1080p minimum for sharp output
5. **Test Run**: Do a practice recording first

---

## Playing Back the Demo

### Local Playback

```bash
# Play at normal speed
asciinema play demo.cast

# Play at 1.5x speed
asciinema play -s 1.5 demo.cast

# Play with custom idle time
asciinema play -i 1 demo.cast
```

### Web Player

```bash
# Upload to asciinema.org (requires account)
asciinema upload demo.cast

# This returns a URL like:
# https://asciinema.org/a/123456
```

---

## Converting to Other Formats

### Convert to GIF

```bash
# Using agg (recommended)
agg demo.cast demo.gif

# Using svg-term
svg-term --in demo.cast --out demo.svg
```

### Convert to MP4

```bash
# Using agg + ffmpeg
agg demo.cast demo.gif
ffmpeg -i demo.gif -movflags faststart -pix_fmt yuv420p demo.mp4
```

### Convert to SVG (for embedding)

```bash
# Using svg-term-cli
svg-term --in demo.cast --out demo.svg --window

# With custom styling
svg-term \
  --in demo.cast \
  --out demo.svg \
  --window \
  --width 80 \
  --height 24 \
  --padding 10
```

---

## Embedding the Demo

### In Documentation (HTML)

```html
<!-- asciinema player embed -->
<script id="asciicast-123456" src="https://asciinema.org/a/123456.js" async></script>

<!-- Self-hosted player -->
<asciinema-player src="demo.cast" cols="120" rows="35" idle-time-limit="2"></asciinema-player>
<script src="https://cdn.jsdelivr.net/npm/asciinema-player@3.0.1/dist/bundle/asciinema-player.min.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/asciinema-player@3.0.1/dist/bundle/asciinema-player.css" />
```

### In Markdown (GitHub)

```markdown
[![GreenLang CBAM Demo](https://asciinema.org/a/123456.svg)](https://asciinema.org/a/123456)
```

### In README

```markdown
## Demo

Watch the CBAM workflow in action:

<a href="https://asciinema.org/a/123456" target="_blank">
  <img src="https://asciinema.org/a/123456.svg" width="600" />
</a>
```

---

## Customizing the Demo

### Modify the Script

Edit `demo-script.sh` to:

- Add or remove workflow steps
- Change timing/pauses
- Update sample data references
- Customize output messages

### Adjust Timing

```bash
# In demo-script.sh, modify sleep durations:
sleep 2  # 2 seconds pause (adjust as needed)

# Or use type_command function for realistic typing:
type_command "gl cbam calculate input.json"
```

### Change Colors

Edit the color variables at the top of `demo-script.sh`:

```bash
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'
```

---

## Narration Guide

The `narration.md` file contains the complete voice-over script. Use it for:

1. **Live Presentations**: Read while demo plays
2. **Video Production**: Record as voice-over track
3. **Subtitles**: Generate closed captions
4. **Translations**: Base for localized versions

### Recording Voice-Over

```bash
# Record narration separately
# Use any audio recording software (Audacity, GarageBand, etc.)

# Combine with terminal recording using video editor
# or use OBS for simultaneous recording
```

---

## Troubleshooting

### Demo Script Won't Run

```bash
# Check permissions
chmod +x demo-script.sh

# Check for Windows line endings
file demo-script.sh
# If it shows "CRLF", convert:
sed -i 's/\r$//' demo-script.sh
```

### pv Command Not Found

The `pv` (pipe viewer) command creates the typing effect. If not available:

```bash
# Install pv
brew install pv  # macOS
sudo apt install pv  # Ubuntu

# Or modify the script to remove typing effect:
# Replace: echo "$1" | pv -qL 30
# With:    echo "$1"
```

### Colors Not Displaying

```bash
# Ensure TERM is set correctly
export TERM=xterm-256color

# Check terminal supports colors
tput colors  # Should output 256
```

### Recording Too Large

```bash
# Compress the recording
asciinema-edit quantize demo.cast -o demo-small.cast

# Or reduce idle time
asciinema-edit speed demo.cast -s 2 -o demo-fast.cast
```

---

## Quality Checklist

Before publishing, verify:

- [ ] All commands execute without errors
- [ ] Output is readable at 1080p
- [ ] Timing feels natural (not too fast/slow)
- [ ] Colors are high contrast
- [ ] No sensitive data visible
- [ ] File size is reasonable (<5MB for GIF)
- [ ] Narration syncs with visuals
- [ ] Closed captions are accurate

---

## Publishing

### Internal Distribution

1. Upload `.cast` file to internal documentation site
2. Embed using self-hosted asciinema player
3. Share direct link in Slack/Teams

### Public Distribution

1. Upload to asciinema.org
2. Convert to GIF for social media
3. Create MP4 for YouTube/Vimeo
4. Embed in product documentation

### Recommended Platforms

| Platform | Format | Notes |
|----------|--------|-------|
| Documentation | asciinema embed | Interactive, smallest size |
| GitHub README | SVG/GIF | Static, good compatibility |
| Twitter/LinkedIn | GIF/MP4 | Under 15 seconds recommended |
| YouTube | MP4 | Full demo with narration |
| Presentations | MP4/GIF | Embed in slides |

---

## Demo Outline

| Scene | Timestamp | Content |
|-------|-----------|---------|
| Introduction | 0:00-0:15 | GreenLang logo and welcome |
| Environment Setup | 0:15-0:35 | Version check and configuration |
| Input Data | 0:35-1:05 | Display sample shipment JSON |
| Validation | 1:05-1:25 | Validate against CBAM schema |
| Calculation | 1:25-1:55 | Run emissions calculation |
| Results | 1:55-2:30 | Examine detailed breakdown |
| Provenance | 2:30-3:05 | Show full audit trail |
| Verification | 3:05-3:25 | Verify reproducibility |
| Export | 3:25-3:45 | Generate regulatory XML |
| Summary | 3:45-4:00 | Recap and closing |

---

## Files Reference

| File | Purpose |
|------|---------|
| `demo-script.sh` | Main executable demo script |
| `narration.md` | Voice-over narration guide |
| `asciinema-config.json` | Recording/playback settings |
| `sample-data/cbam_shipment.json` | Input data for demo |
| `expected-output/emissions.json` | Expected output reference |

---

## Support

For questions about this demo:

- **Documentation**: https://docs.greenlang.io/demo
- **Issues**: https://github.com/greenlang/greenlang/issues
- **Email**: devrel@greenlang.io

---

## License

This demo and all associated materials are part of the GreenLang project.
See the main repository LICENSE file for details.
