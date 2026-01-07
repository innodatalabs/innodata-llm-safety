"""
Shared utility functions for benchmark verification and cleaning scripts.

This module provides common functionality used by both verification.py and clean.py
to ensure consistent behavior across scripts.

Author: Claude Code (Opus 4.5)
"""

import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Optional


def canonicalize_model_name(model_name: str) -> str:
    """
    Canonicalize a model name by removing:
    - Wrapper prefixes (moderated-, remove-thinking-, etc.)
    - Driver prefixes (openai-{hash}-, anthropic-{hash}-, etc.)
    - HuggingFace org names (hf:Qwen/, hf:microsoft/, etc.)
    - Redlite unique ID suffixes (@e611fe, @d91a38, etc.)

    Examples:
        hf:Qwen/Qwen3-VL-30B-A3B-Instruct@e611fe -> Qwen3-VL-30B-A3B-Instruct
        openai-92b5c8-mistralai/mistral-small-3.1-24b-instruct-2503 -> mistral-small-3.1-24b-instruct-2503
        moderated-openai-56f596-gpt-5.1-2025-11-13-586a3d -> gpt-5.1-2025-11-13-586a3d
        hf:microsoft/phi-4@e611fe -> phi-4
        canned -> baseline

    Args:
        model_name: Original model name from meta.json

    Returns:
        Canonicalized model name
    """
    # List of known wrappers from redlite docs
    # See: https://github.com/innodatalabs/redlite/blob/master/docs/wrappers.md
    wrappers = [
        'moderated', 'remove-thinking', 'cached', 'remove-refusal',
        'logit-bias', 'force-json', 'repeat-last-n', 'repeat-penalty',
        'temperature', 'top-p', 'top-k', 'max-tokens', 'json', 'tool-use'
    ]

    canonical = model_name

    # 1. Remove wrapper prefixes (can be chained, so loop until no more matches)
    changed = True
    while changed:
        changed = False
        for wrapper in wrappers:
            prefix = wrapper + '-'
            if canonical.startswith(prefix):
                canonical = canonical[len(prefix):]
                changed = True
                break

    # 2. Remove driver prefix (e.g., openai-92b5c8-, anthropic-abc123-)
    # Pattern: lowercase letters + dash + 6 hex digits + dash
    canonical = re.sub(r'^[a-z]+-[a-f0-9]{6}-', '', canonical)

    # 3. Remove HuggingFace org with hf: prefix (e.g., hf:Qwen/, hf:microsoft/)
    # Pattern: hf: + org_name + /
    canonical = re.sub(r'^hf:[^/]+/', '', canonical)

    # 4. Remove organization name without hf: prefix (e.g., mistralai/, Qwen/)
    # Pattern: org_name + / (but only if preceded by removing a driver)
    # This handles cases like: openai-hash-mistralai/model -> mistralai/model -> model
    canonical = re.sub(r'^[^/]+/', '', canonical)

    # 5. Remove unique ID suffix (e.g., @e611fe, @d91a38)
    # Pattern: @ followed by 6 hex digits at end
    canonical = re.sub(r'@[a-f0-9]{6}$', '', canonical)

    # 6. Rename 'canned' to 'baseline' (better name)
    if canonical == 'canned':
        canonical = 'baseline'

    return canonical


def canonicalize_dataset_name(dataset_name: str) -> str:
    """
    Canonicalize a dataset name by removing the hf:innodatalabs/ prefix.

    Examples:
        hf:innodatalabs/rt2-mine-domain -> rt2-mine-domain
        hf:innodatalabs/rt-frank -> rt-frank
        rt-some-dataset -> rt-some-dataset (no change)

    Args:
        dataset_name: Original dataset name from meta.json

    Returns:
        Canonicalized dataset name
    """
    canonical = dataset_name

    # Remove hf:innodatalabs/ prefix
    if canonical.startswith('hf:innodatalabs/'):
        canonical = canonical[len('hf:innodatalabs/'):]

    return canonical


def find_folders_missing_meta(benchmark_dir: Path) -> List[Path]:
    """
    Find all subdirectories that are missing meta.json files.

    Args:
        benchmark_dir: Path to the benchmark directory

    Returns:
        List of directory paths missing meta.json
    """
    missing_meta = []

    for item in benchmark_dir.iterdir():
        if not item.is_dir():
            continue

        meta_path = item / "meta.json"
        if not meta_path.exists():
            missing_meta.append(item)

    return missing_meta


def read_meta_files(benchmark_dir: Path, warn: bool = True) -> List[Dict]:
    """
    Read all meta.json files from the benchmark directory.

    Args:
        benchmark_dir: Path to the benchmark directory
        warn: Whether to print warnings for missing or corrupted meta.json files

    Returns:
        List of meta dictionaries from successfully parsed meta.json files
    """
    meta_files = []
    missing_folders = find_folders_missing_meta(benchmark_dir)

    # Warn about missing meta.json files
    if warn and missing_folders:
        for folder in missing_folders:
            print(f"Warning: {folder.name} is missing meta.json", file=sys.stderr)

    # Read all valid meta.json files
    for subdir in benchmark_dir.iterdir():
        if not subdir.is_dir():
            continue

        meta_path = subdir / "meta.json"
        if not meta_path.exists():
            continue

        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                meta_files.append(meta)
        except json.JSONDecodeError as e:
            if warn:
                print(f"Error: Failed to parse {meta_path}: {e}", file=sys.stderr)
        except Exception as e:
            if warn:
                print(f"Error: Failed to read {meta_path}: {e}", file=sys.stderr)

    return meta_files


def read_instance_ids(data_file: Path, warn: bool = True) -> Optional[set]:
    """
    Read instance IDs from a data.jsonl file.

    Args:
        data_file: Path to the data.jsonl file
        warn: Whether to print warnings for errors

    Returns:
        Set of instance IDs, or None if file cannot be read
    """
    if not data_file.exists():
        return None

    instance_ids = set()
    try:
        with open(data_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    if 'id' in data:
                        instance_ids.add(data['id'])
                    else:
                        if warn:
                            print(f"Warning: Line {line_num} in {data_file} missing 'id' field", file=sys.stderr)
                except json.JSONDecodeError as e:
                    if warn:
                        print(f"Error: Failed to parse line {line_num} in {data_file}: {e}", file=sys.stderr)
    except Exception as e:
        if warn:
            print(f"Error: Failed to read {data_file}: {e}", file=sys.stderr)
        return None

    return instance_ids
