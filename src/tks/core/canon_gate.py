"""
TKS Canon Gate (Integrity Check)

Enforces strict versioning and hashing of the Canon.
Refuses to start the agent if any Canon file is modified without an index update.
"""

import json
import hashlib
import os
import sys
from typing import Dict, List

CANON_DIR = os.path.join("canon", "normalized")
INDEX_FILE = "canon_index.json"

class CanonIntegrityError(Exception):
    pass

def calculate_file_hash(filepath: str) -> str:
    """Calculates SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read in chunks to handle large files
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def validate_canon_integrity(canon_dir: str = CANON_DIR):
    """
    Checks if all files in canon_index.json exist and match their stored hashes.
    """
    index_path = os.path.join(canon_dir, INDEX_FILE)
    
    if not os.path.exists(index_path):
        raise CanonIntegrityError(f"CRITICAL: Canon Index not found at {index_path}. Run build_canon.py first.")
        
    with open(index_path, "r") as f:
        index_data = json.load(f)
        
    required_files = index_data.get("files", [])
    stored_hashes = index_data.get("hashes", {})
    version = index_data.get("version", "unknown")
    
    print(f"Validating Canon v{version} Integrity...")
    
    for filename in required_files:
        filepath = os.path.join(canon_dir, filename)
        
        if not os.path.exists(filepath):
            raise CanonIntegrityError(f"MISSING FILE: {filename} required by Canon Index.")
            
        # If we have stored hashes (v2 feature), verify them
        if stored_hashes and filename in stored_hashes:
            current_hash = calculate_file_hash(filepath)
            expected_hash = stored_hashes[filename]
            
            if current_hash != expected_hash:
                raise CanonIntegrityError(
                    f"HASH MISMATCH: {filename} has been modified!\n"
                    f"Expected: {expected_hash}\n"
                    f"Found:    {current_hash}\n"
                    f"Action: Revert changes or run build_canon.py to update index."
                )
    
    print("[OK] Canon Integrity Verified.")

def update_canon_hashes(canon_dir: str = CANON_DIR):
    """
    Updates canon_index.json with current file hashes.
    Should be called only by authorized build scripts (Agent Data).
    """
    index_path = os.path.join(canon_dir, INDEX_FILE)
    
    with open(index_path, "r") as f:
        index_data = json.load(f)
        
    required_files = index_data.get("files", [])
    new_hashes = {}
    
    for filename in required_files:
        filepath = os.path.join(canon_dir, filename)
        if os.path.exists(filepath):
            new_hashes[filename] = calculate_file_hash(filepath)
            
    index_data["hashes"] = new_hashes
    
    with open(index_path, "w") as f:
        json.dump(index_data, f, indent=2)
        
    print(f"Updated Canon Index with new hashes for {len(new_hashes)} files.")

if __name__ == "__main__":
    # If run directly, validates the canon
    try:
        validate_canon_integrity()
    except CanonIntegrityError as e:
        print(e)
        sys.exit(1)
