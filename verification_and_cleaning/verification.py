#!/usr/bin/env python3
"""
Verification script for redlite benchmark data.

This script performs the following verifications:
1. Lists all LLMs found in the benchmark
2. Lists all datasets found in the benchmark
3. Verifies that all LLMs have been tested on all datasets (complete coverage)
4. Verifies that all LLMs were tested on the exact same instances for each dataset

The benchmark folder path should be provided via the REDLITE_DATA_DIR environment variable.

Usage:
    REDLITE_DATA_DIR=data-jan2026 python verification.py

Exit codes:
    0 - All verifications passed
    1 - Verification failures found

Author: Claude Code (Opus 4.5)
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, List, Tuple, Optional

# Import shared utilities
from benchmark_utils import read_meta_files, read_instance_ids, find_folders_missing_meta


def extract_llms_and_datasets(meta_files: List[Dict]) -> Tuple[Set[str], Set[str]]:
    """Extract unique LLMs and datasets from meta files."""
    llms = set()
    datasets = set()

    for meta in meta_files:
        if 'model' in meta:
            llms.add(meta['model'])
        else:
            print(f"Warning: meta file missing 'model' field: {meta.get('run', 'unknown')}", file=sys.stderr)

        if 'dataset' in meta:
            datasets.add(meta['dataset'])
        else:
            print(f"Warning: meta file missing 'dataset' field: {meta.get('run', 'unknown')}", file=sys.stderr)

    return llms, datasets


def build_coverage_matrix(meta_files: List[Dict]) -> Dict[str, Dict[str, bool]]:
    """Build a matrix showing which LLM/dataset combinations exist."""
    matrix = defaultdict(lambda: defaultdict(bool))

    for meta in meta_files:
        model = meta.get('model')
        dataset = meta.get('dataset')

        if model and dataset:
            matrix[model][dataset] = True

    return matrix


def verify_coverage(llms: Set[str], datasets: Set[str], matrix: Dict[str, Dict[str, bool]]) -> bool:
    """Verify that all LLMs have been tested on all datasets."""
    missing = []

    for llm in sorted(llms):
        for dataset in sorted(datasets):
            if not matrix[llm][dataset]:
                missing.append((llm, dataset))

    return missing


def build_instance_coverage(benchmark_dir: Path, meta_files: List[Dict]) -> Dict[str, Dict[str, Tuple[Set[str], int, str]]]:
    """
    Build a mapping of dataset -> model -> (instance_ids, count, run_name).

    Returns:
        Dict mapping dataset to model to tuple of (set of instance IDs, count from meta, run name)
    """
    instance_coverage = defaultdict(lambda: defaultdict(lambda: (set(), 0, "")))

    for meta in meta_files:
        dataset = meta.get('dataset')
        model = meta.get('model')
        run = meta.get('run')
        count = meta.get('score_summary', {}).get('count', 0)

        if not dataset or not model or not run:
            continue

        # Find the data.jsonl file for this run
        run_dir = benchmark_dir / run
        data_file = run_dir / "data.jsonl"

        instance_ids = read_instance_ids(data_file)
        if instance_ids is not None:
            instance_coverage[dataset][model] = (instance_ids, count, run)

    return instance_coverage


def verify_instance_consistency(instance_coverage: Dict[str, Dict[str, Tuple[Set[str], int, str]]]) -> List[str]:
    """
    Verify that all models tested on the same dataset used the same instances.

    Returns:
        List of error messages for inconsistencies found.
    """
    errors = []

    for dataset in sorted(instance_coverage.keys()):
        models_data = instance_coverage[dataset]

        if not models_data:
            continue

        # Get reference (first model's instances)
        reference_model = sorted(models_data.keys())[0]
        reference_ids, reference_count, reference_run = models_data[reference_model]

        # Check all other models against the reference
        for model in sorted(models_data.keys()):
            model_ids, model_count, model_run = models_data[model]

            # Check count from meta.json
            if model_count != reference_count:
                errors.append(
                    f"Dataset '{dataset}': Model '{model}' ({model_run}) has count={model_count} "
                    f"but reference model '{reference_model}' ({reference_run}) has count={reference_count}"
                )

            # Check actual instance IDs
            if model_ids != reference_ids:
                missing_in_model = reference_ids - model_ids
                extra_in_model = model_ids - reference_ids

                error_parts = [f"Dataset '{dataset}': Model '{model}' ({model_run}) has different instances than reference model '{reference_model}' ({reference_run})"]
                error_parts.append(f"  Reference has {len(reference_ids)} instances, model has {len(model_ids)} instances")

                if missing_in_model:
                    error_parts.append(f"  Missing {len(missing_in_model)} instances: {sorted(list(missing_in_model))[:5]}{'...' if len(missing_in_model) > 5 else ''}")

                if extra_in_model:
                    error_parts.append(f"  Extra {len(extra_in_model)} instances: {sorted(list(extra_in_model))[:5]}{'...' if len(extra_in_model) > 5 else ''}")

                errors.append("\n".join(error_parts))

    return errors


def main():
    # Get the benchmark directory from environment variable
    benchmark_dir_str = os.environ.get('REDLITE_DATA_DIR')

    if not benchmark_dir_str:
        print("Error: REDLITE_DATA_DIR environment variable not set", file=sys.stderr)
        print("\nUsage: REDLITE_DATA_DIR=data-jan2026 python verification.py", file=sys.stderr)
        sys.exit(1)

    benchmark_dir = Path(benchmark_dir_str)

    if not benchmark_dir.exists():
        print(f"Error: Benchmark directory does not exist: {benchmark_dir}", file=sys.stderr)
        sys.exit(1)

    if not benchmark_dir.is_dir():
        print(f"Error: {benchmark_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    print(f"Verifying benchmark data in: {benchmark_dir}")
    print("=" * 80)
    print()

    # Read all meta files
    meta_files = read_meta_files(benchmark_dir)

    if not meta_files:
        print("Error: No meta.json files found in benchmark directory", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(meta_files)} benchmark runs")
    print()

    # Extract LLMs and datasets
    llms, datasets = extract_llms_and_datasets(meta_files)

    print(f"LLMs ({len(llms)}):")
    print("-" * 80)
    for i, llm in enumerate(sorted(llms), 1):
        print(f"{i:3d}. {llm}")
    print()

    print(f"Datasets ({len(datasets)}):")
    print("-" * 80)
    for i, dataset in enumerate(sorted(datasets), 1):
        print(f"{i:3d}. {dataset}")
    print()

    # Build coverage matrix
    matrix = build_coverage_matrix(meta_files)

    # Verify coverage
    missing = verify_coverage(llms, datasets, matrix)

    print("Coverage Verification:")
    print("-" * 80)

    has_errors = False

    if not missing:
        print("✓ All LLMs have been tested on all datasets")
        print(f"  Total expected: {len(llms)} LLMs × {len(datasets)} datasets = {len(llms) * len(datasets)} runs")
        print(f"  Total found: {len(meta_files)} runs")
        print()
    else:
        has_errors = True
        print(f"✗ Missing {len(missing)} LLM/dataset combinations:")
        print()

        # Group by LLM for better readability
        by_llm = defaultdict(list)
        for llm, dataset in missing:
            by_llm[llm].append(dataset)

        for llm in sorted(by_llm.keys()):
            print(f"  {llm}:")
            for dataset in sorted(by_llm[llm]):
                print(f"    - {dataset}")
            print()

        print(f"Expected total: {len(llms)} LLMs × {len(datasets)} datasets = {len(llms) * len(datasets)} runs")
        print(f"Found: {len(meta_files)} runs")
        print(f"Missing: {len(missing)} runs")
        print()

    # Verify instance consistency
    print("Instance Consistency Verification:")
    print("-" * 80)
    print("Checking that all LLMs were tested on the same instances for each dataset...")
    print()

    instance_coverage = build_instance_coverage(benchmark_dir, meta_files)
    instance_errors = verify_instance_consistency(instance_coverage)

    if not instance_errors:
        print("✓ All LLMs tested on the same instances for each dataset")
        print()

        # Show summary statistics per dataset
        print("Dataset Instance Counts:")
        print("-" * 80)
        for dataset in sorted(instance_coverage.keys()):
            models_data = instance_coverage[dataset]
            if models_data:
                # Get count from first model
                first_model = sorted(models_data.keys())[0]
                _, count, _ = models_data[first_model]
                num_models = len(models_data)
                print(f"  {dataset}: {count} instances × {num_models} models")
        print()
    else:
        has_errors = True
        print(f"✗ Found {len(instance_errors)} instance consistency errors:")
        print()
        for error in instance_errors:
            print(f"  {error}")
            print()

    # Final summary
    print("=" * 80)
    if has_errors:
        print("VERIFICATION FAILED: Errors found")
        return 1
    else:
        print("VERIFICATION PASSED: All checks successful")
        return 0


if __name__ == "__main__":
    sys.exit(main())
