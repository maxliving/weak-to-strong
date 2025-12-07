#!/usr/bin/env python3
"""
Sweep script for mixed supervision experiments.

This script generates and runs all combinations of weak-to-strong experiments,
with easy-to-understand configuration and progress tracking.
"""

import subprocess
import sys
import os
import argparse
import threading
import queue
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple
from datetime import datetime


class TeeLogger:
    """Write to both stdout and a log file."""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure immediate write

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, cannot check for existing runs")


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
                 n_docs: int,
                 n_test_docs: int,
                 eval_every: int,
                 epochs: int,
                 results_folder: str,
                 dry_run: bool = False,
                 wandb_entity: Optional[str] = None,
                 wandb_project: str = "weak-to-strong-mixing",
                 parallel_workers: int = 1):
        self.n_docs = n_docs
        self.n_test_docs = n_test_docs
        self.eval_every = eval_every
        self.epochs = epochs
        self.results_folder = results_folder
        self.dry_run = dry_run
        self.wandb_entity = wandb_entity
        self.wandb_project = wandb_project
        self.parallel_workers = parallel_workers

        self.completed_runs: Set[ExperimentConfig] = set()
        self.failed_runs: List[ExperimentConfig] = []
        self.lock = threading.Lock()  # For thread-safe updates

        # Fetch existing runs from W&B if available
        if wandb_entity and WANDB_AVAILABLE:
            self._load_existing_runs_from_wandb()

    def _load_existing_runs_from_wandb(self):
        """Load existing runs from W&B to avoid re-running."""
        try:
            print(f"\nFetching existing runs from W&B ({self.wandb_entity}/{self.wandb_project})...")
            # Force fresh data from API (no caching)
            api = wandb.Api(timeout=60)
            runs = list(api.runs(f"{self.wandb_entity}/{self.wandb_project}", per_page=500))

            skipped_not_finished = 0
            skipped_missing_config = 0

            for run in runs:
                # Only consider finished runs (trust W&B state)
                if run.state != "finished":
                    skipped_not_finished += 1
                    continue

                config = run.config

                # Extract experiment details from config
                dataset = config.get('ds_name')
                strong_model = config.get('model_size')
                mix_ratio = config.get('mix_ratio')

                # Get weak model (could be in weak_model_size or weak_model.model_size)
                weak_model = config.get('weak_model_size')
                if weak_model is None and 'weak_model' in config:
                    weak_model = config['weak_model'].get('model_size')

                # Skip if missing essential info
                if dataset is None or strong_model is None or mix_ratio is None:
                    skipped_missing_config += 1
                    continue

                # Run finished successfully - add to completed runs
                exp_config = ExperimentConfig(
                    dataset=dataset,
                    strong_model=strong_model,
                    weak_model=weak_model if mix_ratio < 1.0 else None,
                    mix_ratio=mix_ratio
                )
                self.completed_runs.add(exp_config)

            print(f"✓ Found {len(self.completed_runs)} completed runs in W&B")
            total_skipped = skipped_not_finished + skipped_missing_config
            if total_skipped > 0:
                print(f"  (Skipped {total_skipped} runs: {skipped_not_finished} not finished, {skipped_missing_config} missing config)")
        except Exception as e:
            print(f"⚠ Warning: Could not fetch W&B runs: {e}")
            print("  Continuing without W&B check...")

    def run_experiment(self, config: ExperimentConfig, worker_id: int = 0, gpu_devices: str = None) -> bool:
        """Run a single experiment. Returns True if successful."""
        with self.lock:
            if config in self.completed_runs:
                print(f"[Worker {worker_id}] ⊙ Already completed, skipping: {config.description()}")
                return True

            print(f"\n{'='*60}")
            print(f"[Worker {worker_id}] Running: {config.description()}")
            if gpu_devices:
                print(f"[Worker {worker_id}] GPUs: {gpu_devices}")
            print(f"{'='*60}")

        # Build command
        cmd = [
            "python", "train_simple.py",
            f"--model_size={config.strong_model}",
            f"--ds_name={config.dataset}",
            f"--n_docs={self.n_docs}",
            f"--n_test_docs={self.n_test_docs}",
            f"--eval_every={self.eval_every}",
            f"--epochs={self.epochs}",
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

        with self.lock:
            print(f"[Worker {worker_id}] Command: {' '.join(cmd)}")

        if self.dry_run:
            with self.lock:
                print(f"[Worker {worker_id}] ✓ [DRY RUN] Would execute")
            return True

        # Execute with GPU assignment
        env = os.environ.copy()
        if gpu_devices:
            env['CUDA_VISIBLE_DEVICES'] = gpu_devices

        try:
            result = subprocess.run(cmd, check=True, env=env)
            with self.lock:
                print(f"[Worker {worker_id}] ✓ Completed successfully: {config.description()}")
                self.completed_runs.add(config)
            return True
        except subprocess.CalledProcessError as e:
            with self.lock:
                print(f"[Worker {worker_id}] ✗ Failed with exit code {e.returncode}: {config.description()}")
                self.failed_runs.append(config)
            return False
        except KeyboardInterrupt:
            with self.lock:
                print(f"\n[Worker {worker_id}] ⚠ Interrupted by user")
            raise

    def worker_thread(self, worker_id: int, experiment_queue: queue.Queue, gpu_devices: str):
        """Worker thread that processes experiments from the queue."""
        while True:
            try:
                # Get next experiment from queue (non-blocking with timeout)
                try:
                    experiment = experiment_queue.get(timeout=1)
                except queue.Empty:
                    # Queue is empty, worker is done
                    break

                # Run the experiment
                self.run_experiment(experiment, worker_id=worker_id, gpu_devices=gpu_devices)

                # Mark task as done
                experiment_queue.task_done()

            except Exception as e:
                with self.lock:
                    print(f"[Worker {worker_id}] ✗ Exception: {e}")
                experiment_queue.task_done()

    def run_parallel(self, experiments: List[ExperimentConfig]):
        """Run experiments in parallel using multiple workers."""
        if self.parallel_workers <= 1:
            # Sequential execution
            for i, experiment in enumerate(experiments, 1):
                print(f"\n[{i}/{len(experiments)}] ", end="")
                self.run_experiment(experiment)
        else:
            # Parallel execution
            print(f"\nRunning with {self.parallel_workers} parallel workers")

            # Create queue and add all experiments
            experiment_queue = queue.Queue()
            for experiment in experiments:
                experiment_queue.put(experiment)

            # Calculate GPU assignment for each worker
            # Assuming 8 GPUs total, split evenly across workers
            total_gpus = 8
            gpus_per_worker = total_gpus // self.parallel_workers

            print(f"\nGPU Assignment:")
            for worker_id in range(self.parallel_workers):
                start_gpu = worker_id * gpus_per_worker
                end_gpu = start_gpu + gpus_per_worker
                gpu_list = list(range(start_gpu, end_gpu))
                print(f"  Worker {worker_id}: GPUs {gpu_list}")

            # Start worker threads
            workers = []
            for worker_id in range(self.parallel_workers):
                # Assign GPUs to this worker
                start_gpu = worker_id * gpus_per_worker
                end_gpu = start_gpu + gpus_per_worker
                gpu_devices = ",".join(str(i) for i in range(start_gpu, end_gpu))

                # Create and start worker thread
                worker = threading.Thread(
                    target=self.worker_thread,
                    args=(worker_id, experiment_queue, gpu_devices)
                )
                worker.start()
                workers.append(worker)

                with self.lock:
                    print(f"Started Worker {worker_id} with CUDA_VISIBLE_DEVICES={gpu_devices}")

            # Wait for all workers to complete
            for worker in workers:
                worker.join()

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
        # 0. Ground truth runs for weak models (to generate weak labels)
        for weak_model in weak_models:
            config = ExperimentConfig(
                dataset=dataset,
                strong_model=weak_model,
                weak_model=None,
                mix_ratio=1.0
            )
            if (dataset, weak_model, None, 1.0) not in skip_configs:
                experiments.append(config)

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
    EVAL_EVERY = 100
    EPOCHS = 4  # Change this to run more/fewer epochs
    RESULTS_FOLDER = "./results"

    # W&B configuration (for automatic skip of completed runs)
    WANDB_ENTITY = "maxliving-personal"  # Your W&B username
    WANDB_PROJECT = "weak-to-strong-mixing"

    # Dry run mode (set to True to preview without executing)
    DRY_RUN = False

    # Parallel execution (set to 2 for dual parallel runs on 8 GPUs)
    PARALLEL_WORKERS = 1

    # Log file configuration
    now = datetime.now()
    LOG_FILE = f"sweep_log_{now.strftime('%Y-%m-%d-%H%M%S')}.txt"

    # ========================================================================
    # SETUP LOGGING
    # ========================================================================

    # Set up logging to both stdout and file
    log_path = os.path.join(os.getcwd(), LOG_FILE)
    tee = TeeLogger(log_path)
    sys.stdout = tee
    sys.stderr = tee

    print(f"\n[Logging to: {log_path}]")

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

    # ========================================================================
    # INITIALIZE RUNNER (to load completed runs from W&B)
    # ========================================================================

    runner = SweepRunner(
        n_docs=N_DOCS,
        n_test_docs=N_TEST_DOCS,
        eval_every=EVAL_EVERY,
        epochs=EPOCHS,
        results_folder=RESULTS_FOLDER,
        dry_run=DRY_RUN,
        wandb_entity=WANDB_ENTITY,
        wandb_project=WANDB_PROJECT,
        parallel_workers=PARALLEL_WORKERS
    )

    if DRY_RUN:
        print("\n⚠ DRY RUN MODE - No experiments will be executed")
        print("\nExperiment status:")

        to_run = []
        already_completed = []

        for i, exp in enumerate(all_experiments, 1):
            if exp in runner.completed_runs:
                already_completed.append((i, exp))
            else:
                to_run.append((i, exp))

        print(f"\n✓ Already completed ({len(already_completed)}):")
        for i, exp in already_completed:
            print(f"  {i}. {exp.description()}")

        print(f"\n→ To run ({len(to_run)}):")
        for i, exp in to_run:
            print(f"  {i}. {exp.description()}")

        print(f"\nTotal: {len(all_experiments)} experiments")
        print(f"  - Will skip: {len(already_completed)}")
        print(f"  - Will run: {len(to_run)}")
        return

    # ========================================================================
    # RUN EXPERIMENTS
    # ========================================================================

    print(f"\nStarting sweep at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    try:
        runner.run_parallel(all_experiments)
    except KeyboardInterrupt:
        print("\n\nSweep interrupted by user")
    finally:
        runner.print_summary(len(all_experiments))
        # Close log file
        tee.close()
        sys.stdout = tee.terminal
        sys.stderr = tee.terminal


if __name__ == "__main__":
    main()
