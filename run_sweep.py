#!/usr/bin/env python3
"""
Sweep script for mixed supervision experiments.

This script generates and runs all combinations of weak-to-strong experiments,
with easy-to-understand configuration and progress tracking.
"""

import subprocess
import sys
import os
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple
from datetime import datetime


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    dataset: str
    strong_model: str
    weak_model: Optional[str]  # None for ground truth runs
    mix_ratio: float

    def __hash__(self):
        return hash((self.dataset, self.strong_model, self.weak_model, self.mix_ratio))

    def __eq__(self, other):
        return (self.dataset == other.dataset and
                self.strong_model == other.strong_model and
                self.weak_model == other.weak_model and
                self.mix_ratio == other.mix_ratio)

    def is_ground_truth(self) -> bool:
        """Check if this is a ground truth run (weak_model is None OR mix_ratio=1.0)."""
        return self.weak_model is None or self.mix_ratio == 1.0

    def description(self) -> str:
        """Human-readable description of the experiment."""
        if self.is_ground_truth():
            return f"[GT] {self.strong_model} on {self.dataset}"
        else:
            return f"[Mix={self.mix_ratio}] {self.strong_model} (weak={self.weak_model}) on {self.dataset}"


class SweepRunner:
    """Manages and executes experiment sweeps."""

    def __init__(self,
                 n_docs: int = 20000,
                 n_test_docs: int = 10000,
                 eval_every: int = 200,
                 results_folder: str = "/tmp/results",
                 dry_run: bool = False):
        self.n_docs = n_docs
        self.n_test_docs = n_test_docs
        self.eval_every = eval_every
        self.results_folder = results_folder
        self.dry_run = dry_run

        self.completed_runs: Set[ExperimentConfig] = set()
        self.failed_runs: List[ExperimentConfig] = []

    def run_experiment(self, config: ExperimentConfig) -> bool:
        """Run a single experiment. Returns True if successful."""
        print(f"\n{'='*60}")
        print(f"Running: {config.description()}")
        print(f"{'='*60}")

        if config in self.completed_runs:
            print(f"⊙ Already completed, skipping")
            return True

        # Build command
        cmd = [
            "python", "train_simple.py",
            f"--model_size={config.strong_model}",
            f"--ds_name={config.dataset}",
            f"--n_docs={self.n_docs}",
            f"--n_test_docs={self.n_test_docs}",
            f"--eval_every={self.eval_every}",
            f"--results_folder={self.results_folder}",
        ]

        if config.is_ground_truth():
            # Ground truth run: mix_ratio=1.0, no weak labels
            cmd.append(f"--mix_ratio=1.0")
        else:
            # Mixed supervision run: use weak_model_size and mix_ratio
            cmd.append(f"--weak_model_size={config.weak_model}")
            cmd.append(f"--mix_ratio={config.mix_ratio}")
            cmd.append(f"--mix_strategy=sample")

        print(f"Command: {' '.join(cmd)}")

        if self.dry_run:
            print("✓ [DRY RUN] Would execute")
            return True

        # Execute
        try:
            result = subprocess.run(cmd, check=True)
            print(f"✓ Completed successfully")
            self.completed_runs.add(config)
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed with exit code {e.returncode}")
            self.failed_runs.append(config)
            return False
        except KeyboardInterrupt:
            print(f"\n⚠ Interrupted by user")
            raise

    def print_summary(self, total_planned: int):
        """Print summary of sweep results."""
        print("\n" + "="*60)
        print("SWEEP SUMMARY")
        print("="*60)
        print(f"Planned runs: {total_planned}")
        print(f"Completed: {len(self.completed_runs)}")
        print(f"Failed: {len(self.failed_runs)}")

        if self.failed_runs:
            print("\nFailed experiments:")
            for config in self.failed_runs:
                print(f"  - {config.description()}")

        print("\n" + "="*60)
        print(f"Results folder: {self.results_folder}")
        print(f"W&B project: https://wandb.ai/maxliving/weak-to-strong-mixing")
        print("="*60)


def generate_experiments(
    datasets: List[str],
    weak_models: List[str],
    strong_models: List[str],
    mix_ratios: List[float],
    skip_configs: Set[Tuple[str, str, Optional[str], float]] = None
) -> List[ExperimentConfig]:
    """
    Generate all experiment configurations.

    Args:
        datasets: List of dataset names
        weak_models: List of weak model names
        strong_models: List of strong model names
        mix_ratios: List of mix ratios to sweep
        skip_configs: Set of (dataset, strong_model, weak_model, mix_ratio) tuples to skip

    Returns:
        List of ExperimentConfig objects
    """
    experiments = []
    skip_configs = skip_configs or set()

    for dataset in datasets:
        for strong_model in strong_models:
            # 1. Ground truth run for strong model (mix_ratio=1.0)
            config = ExperimentConfig(
                dataset=dataset,
                strong_model=strong_model,
                weak_model=None,
                mix_ratio=1.0
            )
            if (dataset, strong_model, None, 1.0) not in skip_configs:
                experiments.append(config)

            # 2. Mixed supervision runs with each weak model
            for weak_model in weak_models:
                # Skip if weak >= strong (based on model size)
                if should_skip_pair(weak_model, strong_model):
                    continue

                for mix_ratio in mix_ratios:
                    if mix_ratio == 1.0:
                        # mix_ratio=1.0 is handled as ground truth above
                        continue

                    config = ExperimentConfig(
                        dataset=dataset,
                        strong_model=strong_model,
                        weak_model=weak_model,
                        mix_ratio=mix_ratio
                    )

                    if (dataset, strong_model, weak_model, mix_ratio) not in skip_configs:
                        experiments.append(config)

    return experiments


def should_skip_pair(weak_model: str, strong_model: str) -> bool:
    """Check if weak-to-strong pair should be skipped based on size."""
    model_sizes = {
        'gpt2': 0,
        'gpt2-medium': 1,
        'gpt2-large': 2,
        'gpt2-xl': 3,
    }

    weak_size = model_sizes.get(weak_model, -1)
    strong_size = model_sizes.get(strong_model, -1)

    # Skip if weak >= strong
    return weak_size >= strong_size


def main():
    # ========================================================================
    # CONFIGURATION
    # ========================================================================

    # Datasets to sweep
    DATASETS = [
        "boolq",
        "sciq",
        "anthropic_hh"
    ]

    # Model configurations
    WEAK_MODELS = [
        "gpt2",
        "gpt2-medium",
        "gpt2-large"
    ]

    STRONG_MODELS_PRIORITY = [
        "gpt2-large"  # Run first
    ]

    STRONG_MODELS_LAST = [
        "gpt2-xl"  # Run last (largest model)
    ]

    # Mix ratios to sweep
    MIX_RATIOS = [
        0.0,   # Pure weak supervision
        0.25,  # 75% weak, 25% GT
        0.5,   # 50% weak, 50% GT
        0.75,  # 25% weak, 75% GT
        # 1.0 is handled separately as ground truth
    ]

    # Experiments to skip (already completed)
    # Format: (dataset, strong_model, weak_model, mix_ratio)
    # Use None for weak_model in ground truth runs
    SKIP_CONFIGS = {
        ("boolq", "gpt2-large", "gpt2", 0.0),
        ("boolq", "gpt2-large", "gpt2", 0.5),
        ("boolq", "gpt2-large", "gpt2", 1.0),
    }

    # Training parameters
    N_DOCS = 20000
    N_TEST_DOCS = 10000
    EVAL_EVERY = 200
    RESULTS_FOLDER = "/tmp/results"

    # Dry run mode (set to True to preview without executing)
    DRY_RUN = False

    # ========================================================================
    # GENERATE EXPERIMENTS
    # ========================================================================

    print("="*60)
    print("MIXED SUPERVISION SWEEP")
    print("="*60)
    print(f"Datasets: {DATASETS}")
    print(f"Weak models: {WEAK_MODELS}")
    print(f"Strong models (priority): {STRONG_MODELS_PRIORITY}")
    print(f"Strong models (last): {STRONG_MODELS_LAST}")
    print(f"Mix ratios: {MIX_RATIOS}")
    print(f"Skipping {len(SKIP_CONFIGS)} already-completed configs")
    print("="*60)

    # Generate experiments for each priority group
    priority_experiments = generate_experiments(
        datasets=DATASETS,
        weak_models=WEAK_MODELS,
        strong_models=STRONG_MODELS_PRIORITY,
        mix_ratios=MIX_RATIOS,
        skip_configs=SKIP_CONFIGS
    )

    last_experiments = generate_experiments(
        datasets=DATASETS,
        weak_models=WEAK_MODELS,
        strong_models=STRONG_MODELS_LAST,
        mix_ratios=MIX_RATIOS,
        skip_configs=SKIP_CONFIGS
    )

    # Combine: priority first, then large models
    all_experiments = priority_experiments + last_experiments

    print(f"\nGenerated {len(all_experiments)} experiments:")
    print(f"  - Priority (gpt2-large): {len(priority_experiments)}")
    print(f"  - Large (gpt2-xl): {len(last_experiments)}")

    if DRY_RUN:
        print("\n⚠ DRY RUN MODE - No experiments will be executed")
        print("\nPlanned experiments:")
        for i, exp in enumerate(all_experiments, 1):
            print(f"  {i}. {exp.description()}")
        return

    # ========================================================================
    # RUN EXPERIMENTS
    # ========================================================================

    runner = SweepRunner(
        n_docs=N_DOCS,
        n_test_docs=N_TEST_DOCS,
        eval_every=EVAL_EVERY,
        results_folder=RESULTS_FOLDER,
        dry_run=DRY_RUN
    )

    print(f"\nStarting sweep at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    try:
        for i, experiment in enumerate(all_experiments, 1):
            print(f"\n[{i}/{len(all_experiments)}] ", end="")
            runner.run_experiment(experiment)
    except KeyboardInterrupt:
        print("\n\nSweep interrupted by user")
    finally:
        runner.print_summary(len(all_experiments))


if __name__ == "__main__":
    main()
