"""
GreenLang Provenance - Hashing Module
Cryptographic hashing for file integrity and content-addressable storage.
"""

import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# FILE INTEGRITY (SHA256 HASHING)
# ============================================================================

def hash_file(
    file_path: Union[str, Path],
    algorithm: str = "sha256",
    chunk_size: int = 65536
) -> Dict[str, Any]:
    """
    Calculate cryptographic hash of a file for integrity verification.

    Creates a unique fingerprint of the input file that can be used
    to verify that the file hasn't been tampered with. Required for
    regulatory compliance (e.g., EU CBAM).

    Args:
        file_path: Path to file to hash
        algorithm: Hash algorithm ('sha256', 'sha512', 'md5')
        chunk_size: Chunk size for reading (default: 64KB)

    Returns:
        Dictionary with hash details:
        {
            "file_path": str,
            "file_name": str,
            "file_size_bytes": int,
            "hash_algorithm": str,
            "hash_value": str (hex),
            "hash_timestamp": str (ISO 8601),
            "verification": str (how to verify),
            "human_readable_size": str
        }

    Example:
        >>> hash_info = hash_file("shipments.csv")
        >>> print(f"SHA256: {hash_info['hash_value']}")
        >>> # Later, verify integrity:
        >>> new_hash = hash_file("shipments.csv")
        >>> assert new_hash['hash_value'] == hash_info['hash_value']

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If algorithm not supported
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Select hash algorithm
    if algorithm == "sha256":
        hasher = hashlib.sha256()
    elif algorithm == "sha512":
        hasher = hashlib.sha512()
    elif algorithm == "md5":
        hasher = hashlib.md5()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    # Read file in chunks (memory-efficient for large files)
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)

    hash_value = hasher.hexdigest()
    file_size = file_path.stat().st_size

    return {
        "file_path": str(file_path.absolute()),
        "file_name": file_path.name,
        "file_size_bytes": file_size,
        "hash_algorithm": algorithm.upper(),
        "hash_value": hash_value,
        "hash_timestamp": datetime.now(timezone.utc).isoformat(),
        "verification": f"{algorithm}sum {file_path.name}",
        "human_readable_size": _format_bytes(file_size)
    }


def hash_data(
    data: Union[str, bytes, dict],
    algorithm: str = "sha256"
) -> str:
    """
    Calculate hash of in-memory data.

    Args:
        data: Data to hash (string, bytes, or dict)
        algorithm: Hash algorithm

    Returns:
        Hex digest of hash

    Example:
        >>> hash_data({"name": "test", "value": 42})
        'a1b2c3d4e5f67890...'
    """
    # Select algorithm
    if algorithm == "sha256":
        hasher = hashlib.sha256()
    elif algorithm == "sha512":
        hasher = hashlib.sha512()
    elif algorithm == "md5":
        hasher = hashlib.md5()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    # Convert data to bytes
    if isinstance(data, dict):
        import json
        data_bytes = json.dumps(data, sort_keys=True).encode('utf-8')
    elif isinstance(data, str):
        data_bytes = data.encode('utf-8')
    elif isinstance(data, bytes):
        data_bytes = data
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

    hasher.update(data_bytes)
    return hasher.hexdigest()


def _format_bytes(bytes_size: int) -> str:
    """Format bytes into human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


# ============================================================================
# MERKLE TREE FOR HIERARCHICAL HASHING
# ============================================================================

class MerkleTree:
    """
    Merkle tree implementation for hierarchical hashing.

    Provides efficient verification of large datasets by building
    a hash tree where each non-leaf node is the hash of its children.

    Example:
        >>> tree = MerkleTree()
        >>> tree.add_leaf("data1")
        >>> tree.add_leaf("data2")
        >>> tree.add_leaf("data3")
        >>> root_hash = tree.build()
        >>> print(f"Merkle root: {root_hash}")

        >>> # Later, verify data integrity:
        >>> proof = tree.get_proof(1)  # Get proof for leaf 1
        >>> is_valid = tree.verify_proof("data2", 1, proof, root_hash)
    """

    def __init__(self, algorithm: str = "sha256"):
        """
        Initialize Merkle tree.

        Args:
            algorithm: Hash algorithm to use
        """
        self.algorithm = algorithm
        self.leaves: List[str] = []
        self.tree: List[List[str]] = []
        self.root_hash: Optional[str] = None

    def add_leaf(self, data: Union[str, bytes, dict]):
        """
        Add a leaf to the tree.

        Args:
            data: Data to hash and add as leaf
        """
        leaf_hash = hash_data(data, self.algorithm)
        self.leaves.append(leaf_hash)

    def build(self) -> str:
        """
        Build the Merkle tree and return root hash.

        Returns:
            Root hash of the tree

        Raises:
            ValueError: If no leaves have been added
        """
        if not self.leaves:
            raise ValueError("Cannot build tree with no leaves")

        # Initialize tree with leaves
        self.tree = [self.leaves.copy()]

        # Build tree level by level
        while len(self.tree[-1]) > 1:
            current_level = self.tree[-1]
            next_level = []

            # Process pairs
            for i in range(0, len(current_level), 2):
                left = current_level[i]

                # If odd number of nodes, duplicate the last one
                if i + 1 < len(current_level):
                    right = current_level[i + 1]
                else:
                    right = left

                # Hash the concatenation
                combined = left + right
                parent_hash = hash_data(combined, self.algorithm)
                next_level.append(parent_hash)

            self.tree.append(next_level)

        # Root is the only element in the top level
        self.root_hash = self.tree[-1][0]
        return self.root_hash

    def get_proof(self, leaf_index: int) -> List[Dict[str, Any]]:
        """
        Get Merkle proof for a specific leaf.

        Args:
            leaf_index: Index of the leaf (0-based)

        Returns:
            List of proof elements, each with:
            {
                "hash": str,
                "position": "left" or "right"
            }

        Example:
            >>> proof = tree.get_proof(2)
            >>> # Use proof to verify leaf without full tree
        """
        if not self.tree:
            raise ValueError("Tree not built yet. Call build() first.")

        if leaf_index < 0 or leaf_index >= len(self.leaves):
            raise ValueError(f"Invalid leaf index: {leaf_index}")

        proof = []
        index = leaf_index

        # Traverse from bottom to top
        for level in range(len(self.tree) - 1):
            current_level = self.tree[level]

            # Determine sibling
            if index % 2 == 0:  # Left child
                if index + 1 < len(current_level):
                    sibling = current_level[index + 1]
                    position = "right"
                else:
                    sibling = current_level[index]  # Duplicate
                    position = "right"
            else:  # Right child
                sibling = current_level[index - 1]
                position = "left"

            proof.append({
                "hash": sibling,
                "position": position
            })

            # Move to parent index
            index = index // 2

        return proof

    def verify_proof(
        self,
        data: Union[str, bytes, dict],
        leaf_index: int,
        proof: List[Dict[str, Any]],
        root_hash: str
    ) -> bool:
        """
        Verify that data is in the tree using Merkle proof.

        Args:
            data: Original data to verify
            leaf_index: Index of the leaf
            proof: Merkle proof from get_proof()
            root_hash: Expected root hash

        Returns:
            True if proof is valid

        Example:
            >>> is_valid = tree.verify_proof("data2", 1, proof, root_hash)
            >>> assert is_valid
        """
        # Hash the data
        current_hash = hash_data(data, self.algorithm)

        # Apply proof
        for element in proof:
            sibling_hash = element["hash"]
            position = element["position"]

            if position == "left":
                combined = sibling_hash + current_hash
            else:
                combined = current_hash + sibling_hash

            current_hash = hash_data(combined, self.algorithm)

        # Check if we reached the root
        return current_hash == root_hash

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the tree.

        Returns:
            Dictionary with tree statistics
        """
        return {
            "leaves": len(self.leaves),
            "levels": len(self.tree) if self.tree else 0,
            "root_hash": self.root_hash,
            "algorithm": self.algorithm
        }
