#!/usr/bin/env python3
"""
Cleaning script for redlite benchmark data.

This script performs the following operations:
1. Creates a backup of the benchmark folder
2. Canonicalizes model names (removes wrappers, drivers, org names, unique IDs, renames 'canned' to 'baseline')
3. Canonicalizes dataset names (removes hf:innodatalabs/ prefix)
4. Identifies folders with missing meta.json files
5. Prompts user for confirmation
6. Deletes the invalid folders

The benchmark folder path should be provided via the REDLITE_DATA_DIR environment variable.

Usage:
    REDLITE_DATA_DIR=data-jan2026 python clean.py

Exit codes:
    0 - Cleaning completed successfully (or user cancelled)
    1 - Error occurred


Author: Claude Code (Opus 4.5)
"""

import os
import sys
import json
import shutil
import tarfile
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

# Import shared utilities
from benchmark_utils import find_folders_missing_meta, canonicalize_model_name, canonicalize_dataset_name


def create_backup(benchmark_dir: Path) -> Path:
    """
    Create a timestamped tar.gz backup of the benchmark directory.

    Returns:
        Path to the backup file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{benchmark_dir.name}_backup_{timestamp}.tar.gz"
    backup_path = benchmark_dir.parent / backup_name

    print(f"Creating backup: {backup_path}")
    print("This may take a few minutes...")
    print()

    try:
        with tarfile.open(backup_path, "w:gz") as tar:
            tar.add(benchmark_dir, arcname=benchmark_dir.name)

        backup_size = backup_path.stat().st_size / (1024 * 1024)  # Size in MB
        print(f"✓ Backup created successfully: {backup_path}")
        print(f"  Backup size: {backup_size:.2f} MB")
        print()

        return backup_path
    except Exception as e:
        print(f"Error: Failed to create backup: {e}", file=sys.stderr)
        raise


def canonicalize_names(benchmark_dir: Path) -> Tuple[int, int, Dict[str, str], Dict[str, str]]:
    """
    Canonicalize model and dataset names in all meta.json files.

    Model name canonicalization:
    - Removes wrapper prefixes (moderated-, remove-thinking-, etc.)
    - Removes driver prefixes (openai-{hash}-, anthropic-{hash}-, etc.)
    - Removes HuggingFace org names (hf:Qwen/, hf:microsoft/, etc.)
    - Removes redlite unique ID suffixes (@e611fe, @d91a38, etc.)
    - Renames 'canned' to 'baseline'

    Dataset name canonicalization:
    - Removes hf:innodatalabs/ prefix

    Args:
        benchmark_dir: Path to the benchmark directory

    Returns:
        Tuple of (model_updates, dataset_updates, model_mappings, dataset_mappings)
    """
    model_updates = 0
    dataset_updates = 0
    model_mappings = {}
    dataset_mappings = {}

    print("Canonicalizing model and dataset names in meta.json files...")
    print()

    for subdir in sorted(benchmark_dir.iterdir()):
        if not subdir.is_dir():
            continue

        meta_path = subdir / "meta.json"
        if not meta_path.exists():
            continue

        try:
            # Read meta.json
            with open(meta_path, 'r') as f:
                meta = json.load(f)

            changed = False
            changes = []

            # Canonicalize model name
            if 'model' in meta:
                original_model = meta['model']
                canonical_model = canonicalize_model_name(original_model)

                if original_model != canonical_model:
                    meta['model'] = canonical_model
                    changes.append(f"    model: {original_model} -> {canonical_model}")
                    model_mappings[original_model] = canonical_model
                    model_updates += 1
                    changed = True

            # Canonicalize dataset name
            if 'dataset' in meta:
                original_dataset = meta['dataset']
                canonical_dataset = canonicalize_dataset_name(original_dataset)

                if original_dataset != canonical_dataset:
                    meta['dataset'] = canonical_dataset
                    changes.append(f"    dataset: {original_dataset} -> {canonical_dataset}")
                    dataset_mappings[original_dataset] = canonical_dataset
                    dataset_updates += 1
                    changed = True

            # If any changes, write back to file
            if changed:
                with open(meta_path, 'w') as f:
                    json.dump(meta, f, indent=2)

                print(f"  {subdir.name}:")
                for change in changes:
                    print(change)

        except json.JSONDecodeError as e:
            print(f"  Warning: Failed to parse {meta_path}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"  Warning: Failed to process {meta_path}: {e}", file=sys.stderr)

    return model_updates, dataset_updates, model_mappings, dataset_mappings


def prompt_confirmation(folders_to_delete: List[Path]) -> bool:
    """
    Prompt user to confirm deletion of folders.

    Returns:
        True if user confirms, False otherwise
    """
    print("=" * 80)
    print("DELETION CONFIRMATION")
    print("=" * 80)
    print()
    print(f"The following {len(folders_to_delete)} folder(s) will be PERMANENTLY DELETED:")
    print()

    for folder in sorted(folders_to_delete):
        # Show folder name and what files it contains
        files = list(folder.iterdir()) if folder.exists() else []
        file_list = ", ".join([f.name for f in files[:5]])
        if len(files) > 5:
            file_list += f", ... ({len(files) - 5} more)"

        print(f"  - {folder.name}/")
        if files:
            print(f"    Contains: {file_list}")
        else:
            print(f"    (empty or inaccessible)")

    print()
    print("=" * 80)

    while True:
        response = input("Do you want to proceed with deletion? (yes/no): ").strip().lower()

        if response in ['yes', 'y']:
            return True
        elif response in ['no', 'n']:
            return False
        else:
            print("Please enter 'yes' or 'no'")


def delete_folders(folders_to_delete: List[Path]) -> tuple[int, int]:
    """
    Delete the specified folders.

    Returns:
        Tuple of (successful_deletions, failed_deletions)
    """
    successful = 0
    failed = 0

    print()
    print("Deleting folders...")
    print()

    for folder in folders_to_delete:
        try:
            shutil.rmtree(folder)
            print(f"  ✓ Deleted: {folder.name}")
            successful += 1
        except Exception as e:
            print(f"  ✗ Failed to delete {folder.name}: {e}", file=sys.stderr)
            failed += 1

    return successful, failed


def main():
    # Get the benchmark directory from environment variable
    benchmark_dir_str = os.environ.get('REDLITE_DATA_DIR')

    if not benchmark_dir_str:
        print("Error: REDLITE_DATA_DIR environment variable not set", file=sys.stderr)
        print("\nUsage: REDLITE_DATA_DIR=data-jan2026 python clean.py", file=sys.stderr)
        sys.exit(1)

    benchmark_dir = Path(benchmark_dir_str)

    if not benchmark_dir.exists():
        print(f"Error: Benchmark directory does not exist: {benchmark_dir}", file=sys.stderr)
        sys.exit(1)

    if not benchmark_dir.is_dir():
        print(f"Error: {benchmark_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    print(f"Cleaning benchmark data in: {benchmark_dir}")
    print("=" * 80)
    print()

    # Step 1: Create backup
    try:
        backup_path = create_backup(benchmark_dir)
    except Exception as e:
        print(f"Error: Backup failed. Aborting cleaning to prevent data loss.", file=sys.stderr)
        return 1

    # Step 2: Canonicalize model and dataset names
    model_updates, dataset_updates, model_mappings, dataset_mappings = canonicalize_names(benchmark_dir)

    print()
    if model_updates > 0 or dataset_updates > 0:
        print(f"✓ Updated {model_updates + dataset_updates} meta.json file(s)")
        print(f"  - Model names: {model_updates} update(s)")
        print(f"  - Dataset names: {dataset_updates} update(s)")
        print()

        # Show unique model name transformations
        if model_mappings:
            print("Model name transformations:")
            for old_name, new_name in sorted(set(model_mappings.items())):
                print(f"  {old_name} -> {new_name}")
            print()

        # Show unique dataset name transformations
        if dataset_mappings:
            print("Dataset name transformations:")
            for old_name, new_name in sorted(set(dataset_mappings.items())):
                print(f"  {old_name} -> {new_name}")
            print()
    else:
        print("✓ All model and dataset names are already canonical")
        print()

    # Step 3: Find folders missing meta.json
    print("Scanning for folders missing meta.json...")
    folders_to_delete = find_folders_missing_meta(benchmark_dir)
    print()

    if not folders_to_delete:
        print("✓ No folders missing meta.json found")
        print("  Nothing to clean!")
        print()
        print(f"Note: Backup was created at: {backup_path}")
        return 0

    print(f"Found {len(folders_to_delete)} folder(s) missing meta.json")
    print()

    # Step 4: Prompt for confirmation
    if not prompt_confirmation(folders_to_delete):
        print()
        print("Deletion cancelled by user")
        print(f"Backup is available at: {backup_path}")
        return 0

    # Step 5: Delete folders
    successful, failed = delete_folders(folders_to_delete)

    # Step 6: Report results
    print()
    print("=" * 80)
    print("CLEANING SUMMARY")
    print("=" * 80)
    print(f"Backup created: {backup_path}")
    print(f"Model names canonicalized: {model_updates} file(s)")
    print(f"Dataset names canonicalized: {dataset_updates} file(s)")
    print(f"Folders deleted: {successful}")

    if failed > 0:
        print(f"Failed deletions: {failed}")
        print()
        print("Warning: Some folders could not be deleted. Check errors above.")
        return 1

    print()
    print("✓ Cleaning completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
