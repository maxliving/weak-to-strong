#!/usr/bin/env python3
"""
Disagreement-Based Active Learning Experiment Runner

This script runs experiments to test the hypothesis that disagreement-based
label selection is more efficient than random selection for weak-to-strong
generalization.

Strategies tested:
1. Baseline (existing): 100% weak labels
2. Random 25% (existing): 1,179 random GT labels
3. Disagreement 50%: 590 GT labels on highest disagreements
4. Disagreement 100%: 1,179 GT labels on highest disagreements
5. Disagreement 200%: 2,358 GT labels on highest disagreements
"""

import os
import sys
import subprocess
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple


import wandb
WANDB_AVAILABLE = True


# ============================================================================
# CONFIGURATION
# ============================================================================

WANDB_ENTITY = "maxliving-personal"
WANDB_PROJECT = "weak-to-strong-mixing"
DISAGREEMENT_FILE = "./agreement_analysis/gpt2--3danjilx_vs_gpt2-large-mr0.0--5q0356oi/disagreement_rankings.csv"
RESULTS_FOLDER = "./results"
SWEEP_SUBFOLDER = "disagreement_experiment"

# Weak labels path from the baseline gpt2 run (Run ID: 3danjilx)
# This should be the path to the weak_labels directory from the ground truth gpt2 run
WEAK_LABELS_PATH = "/lambda/nfs/us-south3-fs/weak-to-strong/results/default/bs=32-dn=boolq-e=4-ee=200-lp=0-l=xent-l=5e-05-ls=cosi_anne-mc=1024-md=0.001-mxr=0.0-mxs=sample-ms=gpt2-nd=20000-ntd=10000-o=adam-s=0-twd=0/weak_labels"

# Labeling budgets to test
LABELING_BUDGETS = [
    590,   # 50% of random baseline (half of 1,179)
    1179,  # 100% of random baseline (match 25% of 4,714)
    2358,  # 200% of random baseline (double)
]

# Training parameters (must match baseline runs for fair comparison)
TRAIN_PARAMS = {
    'ds_name': 'boolq',
    'n_docs': 20000,
    'n_test_docs': 10000,
    'model_size': 'gpt2-large',
    'epochs': 4,
    'eval_every': 100,
    'batch_size': 32,
    'lr': 1e-05,
    'seed': 0,
    'mix_strategy': 'disagreement',
}


# ============================================================================
# VALIDATION
# ============================================================================

def validate_inputs() -> Tuple[bool, List[str]]:
    """Validate that all required inputs exist and are correct.

    Returns:
        (success, error_messages): True if valid, False otherwise with error messages
    """
    errors = []

    # Check disagreement file exists
    if not Path(DISAGREEMENT_FILE).exists():
        errors.append(f"Disagreement file not found: {DISAGREEMENT_FILE}")
    else:
        # Validate it has the expected number of entries
        try:
            df = pd.read_csv(DISAGREEMENT_FILE)
            if 'idx' not in df.columns or 'confidence_diff' not in df.columns:
                errors.append(f"Disagreement file missing required columns (idx, confidence_diff)")
            # Note: May have fewer entries if dataset is smaller
        except Exception as e:
            errors.append(f"Error reading disagreement file: {e}")

    # Check weak labels path exists
    if not Path(WEAK_LABELS_PATH).exists():
        errors.append(f"Weak labels path not found: {WEAK_LABELS_PATH}")
        errors.append(f"Please ensure the baseline gpt2 run has completed and generated weak labels")

    # Check train_simple.py exists
    if not Path("train_simple.py").exists():
        errors.append("train_simple.py not found in current directory")

    # Validate budgets (allow any positive budget, dataset size will be checked at runtime)
    for budget in LABELING_BUDGETS:
        if budget <= 0:
            errors.append(f"Budget {budget} must be positive")

    return len(errors) == 0, errors


# ============================================================================
# WANDB INTEGRATION
# ============================================================================

def check_existing_runs() -> Dict[int, Optional[str]]:
    """Check WandB for existing disagreement experiment runs.

    Returns:
        Dictionary mapping labeling_budget -> run_id (or None if not found)
    """
    if not WANDB_AVAILABLE:
        print("WandB not available - cannot check for existing runs")
        return {budget: None for budget in LABELING_BUDGETS}

    print(f"\nChecking WandB for existing runs...")
    print(f"Entity: {WANDB_ENTITY}")
    print(f"Project: {WANDB_PROJECT}")

    try:
        api = wandb.Api(timeout=60)
        runs = list(api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", per_page=500))

        # Filter for disagreement strategy runs
        existing_runs = {}
        for run in runs:
            if run.state != "finished":
                continue

            config = run.config
            if config.get('mix_strategy') != 'disagreement':
                continue

            budget = config.get('labeling_budget')
            if budget in LABELING_BUDGETS:
                # Check if this matches our experiment parameters
                matches = all([
                    config.get('ds_name') == TRAIN_PARAMS['ds_name'],
                    config.get('model_size') == TRAIN_PARAMS['model_size'],
                    config.get('epochs') == TRAIN_PARAMS['epochs'],
                    config.get('seed') == TRAIN_PARAMS['seed'],
                ])

                if matches:
                    existing_runs[budget] = run.id
                    print(f"  ✓ Found existing run for budget={budget}: {run.id} ({run.name})")

        # Fill in missing budgets
        for budget in LABELING_BUDGETS:
            if budget not in existing_runs:
                existing_runs[budget] = None
                print(f"  ✗ No existing run for budget={budget}")

        return existing_runs

    except Exception as e:
        print(f"Error checking WandB: {e}")
        return {budget: None for budget in LABELING_BUDGETS}


# ============================================================================
# EXPERIMENT EXECUTION
# ============================================================================

def build_train_command(budget: int) -> List[str]:
    """Build the training command for a given labeling budget.

    Args:
        budget: Number of GT labels to use

    Returns:
        Command as list of strings
    """
    cmd = [
        'python', '-u', 'train_simple.py',  # -u for unbuffered output
        f'--ds_name={TRAIN_PARAMS["ds_name"]}',
        f'--n_docs={TRAIN_PARAMS["n_docs"]}',
        f'--n_test_docs={TRAIN_PARAMS["n_test_docs"]}',
        f'--model_size={TRAIN_PARAMS["model_size"]}',
        f'--epochs={TRAIN_PARAMS["epochs"]}',
        f'--eval_every={TRAIN_PARAMS["eval_every"]}',
        f'--batch_size={TRAIN_PARAMS["batch_size"]}',
        f'--lr={TRAIN_PARAMS["lr"]}',
        f'--seed={TRAIN_PARAMS["seed"]}',
        f'--mix_strategy={TRAIN_PARAMS["mix_strategy"]}',
        f'--mix_ratio=0.0',  # For disagreement strategy, set to 0.0 (using labeling_budget instead)
        f'--labeling_budget={budget}',
        f'--disagreement_file={DISAGREEMENT_FILE}',
        f'--weak_labels_path={WEAK_LABELS_PATH}',  # Explicit path to weak labels from baseline run
        f'--results_folder={RESULTS_FOLDER}',
        f'--sweep_subfolder={SWEEP_SUBFOLDER}',
    ]
    return cmd


def run_experiment(budget: int, log_file: Path) -> bool:
    """Run a single disagreement experiment.

    Args:
        budget: Number of GT labels to use
        log_file: Path to log file

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"RUNNING EXPERIMENT: Disagreement-based with {budget} GT labels")
    print(f"{'='*80}")

    cmd = build_train_command(budget)
    print(f"Command: {' '.join(cmd)}")
    print(f"Logging to: {log_file}")
    print()

    try:
        with open(log_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"EXPERIMENT: budget={budget}\n")
            f.write(f"Started: {datetime.now().isoformat()}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"{'='*80}\n\n")
            f.flush()

            # Run training
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            # Stream output to both console and log file
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                f.write(line)
                f.flush()

            process.wait()

            success = process.returncode == 0

            f.write(f"\n{'='*80}\n")
            f.write(f"Completed: {datetime.now().isoformat()}\n")
            f.write(f"Exit code: {process.returncode}\n")
            f.write(f"Status: {'SUCCESS' if success else 'FAILED'}\n")
            f.write(f"{'='*80}\n\n")

        if success:
            print(f"\n✓ Experiment budget={budget} completed successfully")
        else:
            print(f"\n✗ Experiment budget={budget} failed with exit code {process.returncode}")

        return success

    except Exception as e:
        error_msg = f"\n✗ Error running experiment budget={budget}: {e}"
        print(error_msg)

        # Log error to file
        try:
            with open(log_file, 'a') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"ERROR: {error_msg}\n")
                f.write(f"{'='*80}\n\n")
        except:
            pass

        return False


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def main():
    """Main experiment orchestration."""
    print("="*80)
    print("DISAGREEMENT-BASED ACTIVE LEARNING EXPERIMENT")
    print("="*80)
    print()
    print("This experiment tests whether disagreement-based label selection")
    print("is more efficient than random selection for weak-to-strong generalization.")
    print()

    # Validate inputs
    print("Validating inputs...")
    valid, errors = validate_inputs()
    if not valid:
        print("\n✗ Validation failed:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    print("✓ All inputs valid")

    # Check for existing runs
    existing_runs = check_existing_runs()

    # Determine which runs to execute
    runs_to_execute = [
        budget for budget in LABELING_BUDGETS
        if existing_runs[budget] is None
    ]

    # Print experiment plan
    print()
    print("="*80)
    print("EXPERIMENT PLAN")
    print("="*80)
    print(f"Total experiments: {len(LABELING_BUDGETS)}")
    print(f"Already completed: {len(LABELING_BUDGETS) - len(runs_to_execute)}")
    print(f"To execute: {len(runs_to_execute)}")
    print()

    if runs_to_execute:
        print("Experiments to run:")
        for budget in runs_to_execute:
            print(f"  - Budget {budget}: Select top {budget} disagreements for GT labels")
        print()
    else:
        print("All experiments already completed!")
        print()
        print("Existing runs:")
        for budget, run_id in existing_runs.items():
            if run_id:
                print(f"  - Budget {budget}: Run ID {run_id}")
        return

    # Confirm execution
    print("Training parameters:")
    for key, value in TRAIN_PARAMS.items():
        print(f"  {key}: {value}")
    print()
    print(f"Disagreement file: {DISAGREEMENT_FILE}")
    print(f"Results folder: {RESULTS_FOLDER}/{SWEEP_SUBFOLDER}")
    print()

    response = input("Proceed with experiments? [y/N]: ").strip().lower()
    if response not in ['y', 'yes']:
        print("Aborted.")
        return

    # Create log file
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    log_file = Path(f"disagreement_experiment_{timestamp}.log")
    print(f"\nLogging to: {log_file}")

    # Execute experiments
    results = {}
    for i, budget in enumerate(runs_to_execute, 1):
        print(f"\n[{i}/{len(runs_to_execute)}] Starting experiment with budget={budget}")
        success = run_experiment(budget, log_file)
        results[budget] = success

        # Stop immediately on failure
        if not success:
            print(f"\n✗ Experiment budget={budget} FAILED - stopping execution")
            print(f"Full log: {log_file}")
            sys.exit(1)

    # Summary
    print()
    print("="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    successful = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)

    print(f"Completed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print()

    if successful > 0:
        print("Successful runs:")
        for budget, success in results.items():
            if success:
                print(f"  ✓ Budget {budget}")

    if failed > 0:
        print()
        print("Failed runs:")
        for budget, success in results.items():
            if not success:
                print(f"  ✗ Budget {budget}")

    print()
    print(f"Full log: {log_file}")
    print()
    print("="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. Check WandB for run results")
    print("2. Fetch final test accuracies")
    print("3. Calculate PGR (Performance Gap Recovered)")
    print("4. Compare to random baseline (74.2% @ 1,179 labels)")
    print("5. Generate visualizations (accuracy vs. budget)")
    print("="*80)


if __name__ == '__main__':
    main()
