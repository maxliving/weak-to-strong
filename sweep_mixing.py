"""
Sweep script for mixed supervision experiments.

This script automates running weak-to-strong experiments across multiple
mixing ratios to study sample efficiency and optimal supervision budgets.

Usage:
    python sweep_mixing.py --mix_ratios="0,0.25,0.5,1.0" --ds_name=sciq

    python sweep_mixing.py \
        --mix_ratios="1.0,0.75,0.5,0.25,0.1,0.0" \
        --mix_strategy=sample \
        --ds_name=sciq --n_docs=20000 \
        --weak_model_size=gpt2-medium --strong_model_size=gpt2-xl
"""

import subprocess
import sys
import fire


def sweep_mix_ratios(
    mix_ratios: str = "0.0,0.1,0.25,0.5,0.75,1.0",
    mix_strategy: str = "sample",
    base_script: str = "train_simple.py",
    **kwargs
):
    """
    Run weak-to-strong experiments across multiple mixing ratios.

    Args:
        mix_ratios: Comma-separated list of mixing ratios to try (e.g., "0,0.25,0.5,1.0")
        mix_strategy: Mixing strategy - 'sample' or 'label'
        base_script: Script to run (default: train_simple.py)
        **kwargs: Additional arguments to pass to the training script
                  For train_simple.py: model_size, ds_name, weak_labels_path (required), etc.
                  For train_weak_to_strong.py: weak_model_size, strong_model_size, etc.

    Examples:
        # Using train_simple.py (recommended)
        sweep_mix_ratios(
            mix_ratios="0,0.25,0.5,1.0",
            mix_strategy="sample",
            model_size="gpt2-medium",
            ds_name="sciq",
            weak_labels_path="/tmp/results/default/{config}/weak_labels"
        )

        # Using train_weak_to_strong.py (all-in-one)
        sweep_mix_ratios(
            mix_ratios="0,0.25,0.5,1.0",
            mix_strategy="sample",
            base_script="train_weak_to_strong.py",
            ds_name="sciq",
            n_docs=10000,
            weak_model_size="gpt2",
            strong_model_size="gpt2-medium"
        )
    """
    # Parse mix ratios
    ratios = [float(r.strip()) for r in mix_ratios.split(",")]

    print(f"\n{'='*70}")
    print(f"MIXED SUPERVISION SWEEP")
    print(f"{'='*70}")
    print(f"Strategy: {mix_strategy}")
    print(f"Mix ratios: {ratios}")
    print(f"Additional args: {kwargs}")
    print(f"{'='*70}\n")

    results = []

    for i, ratio in enumerate(ratios, 1):
        print(f"\n{'='*70}")
        print(f"EXPERIMENT {i}/{len(ratios)}: mix_ratio={ratio} ({ratio*100:.0f}% GT)")
        print(f"{'='*70}\n")

        # Build command
        cmd = [sys.executable, base_script]  # Use same Python interpreter

        # Add mix ratio and strategy
        cmd.extend([
            f"--mix_ratio={ratio}",
            f"--mix_strategy={mix_strategy}",
        ])

        # Add additional kwargs
        for key, value in kwargs.items():
            # Convert underscores in argument names if needed
            arg_name = key.replace("_", "-") if "_" in key else key
            cmd.append(f"--{arg_name}={value}")

        # Print command
        print(f"Running: {' '.join(cmd)}\n")

        # Run command
        try:
            result = subprocess.run(cmd, check=True)
            results.append({
                'ratio': ratio,
                'status': 'success',
                'returncode': result.returncode
            })
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Experiment failed with return code {e.returncode}")
            results.append({
                'ratio': ratio,
                'status': 'failed',
                'returncode': e.returncode
            })
            # Continue with remaining experiments
            continue

    # Print summary
    print(f"\n\n{'='*70}")
    print(f"SWEEP SUMMARY")
    print(f"{'='*70}")
    print(f"Total experiments: {len(ratios)}")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"Failed: {sum(1 for r in results if r['status'] == 'failed')}")
    print(f"\nResults by ratio:")
    for res in results:
        status_icon = "✓" if res['status'] == 'success' else "✗"
        print(f"  {status_icon} mix_ratio={res['ratio']:.2f}: {res['status']}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    fire.Fire(sweep_mix_ratios)
