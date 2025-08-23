"""Text normalization utilities for stable test snapshots."""

import re

def normalize_text(s: str) -> str:
    """Normalize text for stable snapshots (timestamps, paths, whitespace)."""
    s = s.replace("\r\n", "\n")
    s = re.sub(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?", "<TIMESTAMP>", s)
    s = re.sub(r"[ A-Za-z]{3}\s\d{1,2},\s\d{4}", "<DATE>", s)
    s = re.sub(r"/tmp/\S+", "<TMP>", s)
    s = re.sub(r"[A-Za-z]:\\\S+", "<PATH>", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()