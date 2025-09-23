"""
Hub Index System
================

Simple index.json system for pack discovery.
Stores org/slug -> versions mapping with card summaries.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import requests

logger = logging.getLogger(__name__)


@dataclass
class IndexEntry:
    """Single pack entry in the hub index"""

    name: str
    org: str
    slug: str
    latest_version: str
    versions: List[str]
    description: str
    license: str
    card_summary: str
    created_at: str
    updated_at: str
    download_count: int = 0
    tags: List[str] = None
    oci_ref: str = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if not self.oci_ref:
            self.oci_ref = f"ghcr.io/{self.org}/{self.slug}"


class HubIndex:
    """
    Hub index manager for pack discovery

    Manages index.json file that contains:
    {
      "version": "1.0",
      "updated_at": "2024-01-01T12:00:00Z",
      "packs": {
        "org/slug": {
          "name": "Full Pack Name",
          "org": "org",
          "slug": "slug",
          "latest_version": "0.2.0",
          "versions": ["0.1.0", "0.2.0"],
          "description": "Pack description",
          "license": "Apache-2.0",
          "card_summary": "Brief summary from CARD.md",
          "oci_ref": "ghcr.io/org/slug",
          "created_at": "2024-01-01T10:00:00Z",
          "updated_at": "2024-01-01T12:00:00Z",
          "download_count": 42,
          "tags": ["climate", "energy"]
        }
      }
    }
    """

    DEFAULT_INDEX_URLS = [
        "https://hub.greenlang.ai/index.json",
        "https://raw.githubusercontent.com/greenlang/hub/main/index.json",
    ]

    def __init__(
        self, index_url: Optional[str] = None, cache_dir: Optional[Path] = None
    ):
        """
        Initialize hub index

        Args:
            index_url: URL to index.json (uses defaults if None)
            cache_dir: Local cache directory for index
        """
        self.index_urls = [index_url] if index_url else self.DEFAULT_INDEX_URLS
        self.cache_dir = cache_dir or (Path.home() / ".greenlang" / "hub_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "index.json"
        self._index_data: Optional[Dict[str, Any]] = None

    def load(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Load index data (from cache or remote)

        Args:
            force_refresh: Force refresh from remote

        Returns:
            Index data dictionary
        """
        if self._index_data and not force_refresh:
            return self._index_data

        # Try cache first if not forcing refresh
        if not force_refresh and self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    cache_data = json.load(f)

                # Check if cache is recent (< 1 hour old)
                cache_time = datetime.fromisoformat(
                    cache_data.get("updated_at", "1970-01-01T00:00:00")
                )
                age = datetime.now() - cache_time

                if age.total_seconds() < 3600:  # 1 hour
                    logger.debug("Using cached index data")
                    self._index_data = cache_data
                    return self._index_data

            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning(f"Invalid cache file, will refresh: {e}")

        # Fetch from remote
        for url in self.index_urls:
            try:
                logger.info(f"Fetching index from {url}")
                response = requests.get(url, timeout=10)
                response.raise_for_status()

                index_data = response.json()

                # Validate index structure
                if not self._validate_index(index_data):
                    logger.warning(f"Invalid index structure from {url}")
                    continue

                # Cache the data
                with open(self.cache_file, "w") as f:
                    json.dump(index_data, f, indent=2)

                logger.info(
                    f"Successfully loaded index with {len(index_data.get('packs', {}))} packs"
                )
                self._index_data = index_data
                return self._index_data

            except Exception as e:
                logger.warning(f"Failed to fetch from {url}: {e}")
                continue

        # If all remote sources failed, try stale cache
        if self.cache_file.exists():
            logger.warning("Using stale cache as fallback")
            try:
                with open(self.cache_file) as f:
                    self._index_data = json.load(f)
                return self._index_data
            except Exception as e:
                logger.error(f"Failed to load stale cache: {e}")

        # Return empty index as last resort
        logger.error("Could not load index from any source")
        self._index_data = {
            "version": "1.0",
            "updated_at": datetime.now().isoformat(),
            "packs": {},
        }
        return self._index_data

    def _validate_index(self, data: Dict[str, Any]) -> bool:
        """Validate index data structure"""
        required_fields = ["version", "updated_at", "packs"]
        return all(field in data for field in required_fields)

    def search(
        self, query: str = "", tags: List[str] = None, org: str = None
    ) -> List[IndexEntry]:
        """
        Search packs in the index

        Args:
            query: Search query (matches name, description)
            tags: Filter by tags
            org: Filter by organization

        Returns:
            List of matching pack entries
        """
        index_data = self.load()
        results = []

        query_lower = query.lower()
        tags_set = set(tags or [])

        for pack_key, pack_data in index_data.get("packs", {}).items():
            # Filter by org
            if org and pack_data.get("org", "") != org:
                continue

            # Filter by tags
            pack_tags = set(pack_data.get("tags", []))
            if tags_set and not tags_set.intersection(pack_tags):
                continue

            # Filter by query
            if query:
                searchable_text = f"{pack_data.get('name', '')} {pack_data.get('description', '')} {pack_data.get('card_summary', '')}".lower()
                if query_lower not in searchable_text:
                    continue

            # Create entry
            entry = IndexEntry(
                name=pack_data.get("name", pack_key),
                org=pack_data.get("org", ""),
                slug=pack_data.get("slug", ""),
                latest_version=pack_data.get("latest_version", "0.0.0"),
                versions=pack_data.get("versions", []),
                description=pack_data.get("description", ""),
                license=pack_data.get("license", ""),
                card_summary=pack_data.get("card_summary", ""),
                created_at=pack_data.get("created_at", ""),
                updated_at=pack_data.get("updated_at", ""),
                download_count=pack_data.get("download_count", 0),
                tags=pack_data.get("tags", []),
                oci_ref=pack_data.get(
                    "oci_ref",
                    f"ghcr.io/{pack_data.get('org', '')}/{pack_data.get('slug', '')}",
                ),
            )
            results.append(entry)

        # Sort by download count descending, then alphabetically
        results.sort(key=lambda x: (-x.download_count, x.name))
        return results

    def get(self, pack_ref: str) -> Optional[IndexEntry]:
        """
        Get specific pack by reference

        Args:
            pack_ref: Pack reference (org/slug or just slug)

        Returns:
            Pack entry if found
        """
        index_data = self.load()
        packs = index_data.get("packs", {})

        # Try exact match first
        if pack_ref in packs:
            pack_data = packs[pack_ref]
            return IndexEntry(
                name=pack_data.get("name", pack_ref),
                org=pack_data.get("org", ""),
                slug=pack_data.get("slug", ""),
                latest_version=pack_data.get("latest_version", "0.0.0"),
                versions=pack_data.get("versions", []),
                description=pack_data.get("description", ""),
                license=pack_data.get("license", ""),
                card_summary=pack_data.get("card_summary", ""),
                created_at=pack_data.get("created_at", ""),
                updated_at=pack_data.get("updated_at", ""),
                download_count=pack_data.get("download_count", 0),
                tags=pack_data.get("tags", []),
                oci_ref=pack_data.get("oci_ref", ""),
            )

        # Try partial match on slug
        for key, pack_data in packs.items():
            if pack_data.get("slug", "") == pack_ref or key.endswith(f"/{pack_ref}"):
                return IndexEntry(
                    name=pack_data.get("name", key),
                    org=pack_data.get("org", ""),
                    slug=pack_data.get("slug", ""),
                    latest_version=pack_data.get("latest_version", "0.0.0"),
                    versions=pack_data.get("versions", []),
                    description=pack_data.get("description", ""),
                    license=pack_data.get("license", ""),
                    card_summary=pack_data.get("card_summary", ""),
                    created_at=pack_data.get("created_at", ""),
                    updated_at=pack_data.get("updated_at", ""),
                    download_count=pack_data.get("download_count", 0),
                    tags=pack_data.get("tags", []),
                    oci_ref=pack_data.get("oci_ref", ""),
                )

        return None

    def add_or_update(self, entry: IndexEntry) -> None:
        """
        Add or update pack entry in the index

        Args:
            entry: Pack entry to add/update
        """
        if not self._index_data:
            self._index_data = self.load()

        pack_key = f"{entry.org}/{entry.slug}"

        # Update or create entry
        self._index_data.setdefault("packs", {})[pack_key] = asdict(entry)
        self._index_data["updated_at"] = datetime.now().isoformat()

        # Save to cache
        with open(self.cache_file, "w") as f:
            json.dump(self._index_data, f, indent=2)

        logger.info(f"Updated index entry: {pack_key}")

    def create_local_index(self, output_path: Path, packs_dir: Path) -> None:
        """
        Create index.json from local packs directory

        Args:
            output_path: Where to save index.json
            packs_dir: Directory containing pack directories
        """
        index_data = {
            "version": "1.0",
            "updated_at": datetime.now().isoformat(),
            "packs": {},
        }

        # Scan for pack.yaml files
        for pack_yaml in packs_dir.rglob("pack.yaml"):
            try:
                pack_dir = pack_yaml.parent

                # Load manifest
                import yaml

                with open(pack_yaml) as f:
                    manifest = yaml.safe_load(f)

                # Read CARD.md for summary
                card_summary = ""
                card_path = pack_dir / "CARD.md"
                if card_path.exists():
                    with open(card_path) as f:
                        # Take first few lines as summary
                        lines = f.readlines()[:5]
                        card_summary = " ".join(
                            line.strip() for line in lines if line.strip()
                        )[:200]

                # Extract org from directory structure or manifest
                org = manifest.get("org", "greenlang")
                slug = manifest.get("name", pack_dir.name)

                entry = IndexEntry(
                    name=manifest.get("name", slug),
                    org=org,
                    slug=slug,
                    latest_version=manifest.get("version", "0.0.0"),
                    versions=[manifest.get("version", "0.0.0")],
                    description=manifest.get("description", ""),
                    license=manifest.get("license", "Apache-2.0"),
                    card_summary=card_summary,
                    created_at=datetime.now().isoformat(),
                    updated_at=datetime.now().isoformat(),
                    download_count=0,
                    tags=manifest.get("tags", []),
                    oci_ref=f"ghcr.io/{org}/{slug}",
                )

                pack_key = f"{org}/{slug}"
                index_data["packs"][pack_key] = asdict(entry)

                logger.info(f"Added to index: {pack_key}")

            except Exception as e:
                logger.error(f"Failed to process {pack_yaml}: {e}")
                continue

        # Save index
        with open(output_path, "w") as f:
            json.dump(index_data, f, indent=2, sort_keys=True)

        logger.info(
            f"Created index with {len(index_data['packs'])} packs at {output_path}"
        )
