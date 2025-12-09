"""
Plot disagreement-based vs random label selection comparison.

Shows:
- X-axis: number of ground truth labels
- Y-axis: accuracy
- Two strategies: random (sample) vs disagreement-based selection
- Comparison of label efficiency
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import os

sns.set_style('whitegrid')


def fetch_all_finished_runs(entity: str, project: str, dataset: str = None, metric: str = 'best_checkpoint'):
    """Fetch all finished runs from W&B.

    Args:
        entity: W&B entity
        project: W&B project name
        dataset: Optional dataset filter
        metric: Which metric to use ('eval', 'best_checkpoint')

    Returns:
        DataFrame with run data
    """
    api = wandb.Api(timeout=60)
    runs = list(api.runs(f"{entity}/{project}", per_page=500))

    print(f"Found {len(runs)} total runs in {project}")
    print(f"Using metric: {metric}")

    records = []
    for run in runs:
        if run.state != "finished":
            continue

        config = run.config

        # Filter by dataset if specified
        if dataset is not None and config.get('ds_name') != dataset:
            continue

        # Create record from config
        record = dict(config)

        # Handle weak_model nested config
        if 'weak_model' in config and isinstance(config['weak_model'], dict):
            for k, v in config['weak_model'].items():
                record['weak_' + k] = v
            del record['weak_model']

        # Get summary metrics
        summary = run.summary._json_dict

        # Get accuracy based on metric preference
        if metric == 'eval':
            # Prioritize eval_accuracy
            if 'eval_accuracy' in summary:
                record['accuracy'] = summary['eval_accuracy']
                record['metric_source'] = 'eval_accuracy'
            else:
                continue
        elif metric == 'best_checkpoint':
            # Prioritize best checkpoint
            if 'best_checkpoint/final_test_acc' in summary:
                record['accuracy'] = summary['best_checkpoint/final_test_acc']
                record['metric_source'] = 'best_checkpoint/final_test_acc'
            elif 'eval_accuracy' in summary:
                record['accuracy'] = summary['eval_accuracy']
                record['metric_source'] = 'eval_accuracy'
            else:
                continue
        else:
            raise ValueError(f"Unknown metric: {metric}. Must be 'eval' or 'best_checkpoint'")

        # Add run metadata
        record['run_id'] = run.id
        record['run_name'] = run.name

        records.append(record)

    df = pd.DataFrame.from_records(records)
    print(f"Loaded {len(df)} finished runs with accuracy data")

    return df


def plot_disagreement_comparison(df, dataset, strong_model, weak_model=None, train_set_size=4714,
                                 output_dir='plots', output_name='disagreement_comparison'):
    """Create comparison plot of disagreement vs random selection.

    Args:
        df: DataFrame with run data
        dataset: Dataset name
        strong_model: Strong model size to analyze
        weak_model: Weak model size to filter for (None = auto-detect from disagreement runs)
        train_set_size: Size of training set for calculating GT label counts
        output_dir: Directory to save plots
        output_name: Base name for output files
    """
    # Filter to dataset and strong model
    df_filtered = df[
        (df['ds_name'] == dataset) &
        (df['model_size'] == strong_model)
    ].copy()

    # Auto-detect weak model from disagreement runs if not specified
    if weak_model is None:
        disagreement_runs = df_filtered[df_filtered['mix_strategy'] == 'disagreement']
        if len(disagreement_runs) > 0:
            # Try both weak_model_size and weak_model_model_size (from nested config)
            weak_model = disagreement_runs.iloc[0].get('weak_model_size')
            if pd.isna(weak_model):
                weak_model = disagreement_runs.iloc[0].get('weak_model_model_size')
            if weak_model:
                print(f"Auto-detected weak model: {weak_model}")
            else:
                print("Warning: Could not auto-detect weak model from disagreement runs")

    # Filter to specific weak model for fair comparison
    if weak_model:
        df_filtered = df_filtered[
            (df_filtered['weak_model_size'] == weak_model) |
            (df_filtered['weak_model_size'].isna())  # Include baseline/GT runs
        ].copy()
        print(f"Filtering to weak model: {weak_model}")

    if len(df_filtered) == 0:
        print(f"No data found for {dataset} + {strong_model}")
        return

    # Calculate number of GT labels for each run
    df_filtered['n_gt_labels'] = 0

    # For disagreement strategy, use labeling_budget directly
    disagreement_mask = df_filtered['mix_strategy'] == 'disagreement'
    df_filtered.loc[disagreement_mask, 'n_gt_labels'] = df_filtered.loc[disagreement_mask, 'labeling_budget'].fillna(0).astype(int)

    # For sample strategy, calculate from mix_ratio
    sample_mask = (df_filtered['mix_strategy'] == 'sample') | (df_filtered['mix_strategy'].isna())
    df_filtered.loc[sample_mask, 'n_gt_labels'] = (df_filtered.loc[sample_mask, 'mix_ratio'].fillna(0) * train_set_size).astype(int)

    # Get baseline (0 GT labels) and ground truth (100% GT labels)
    baseline_runs = df_filtered[df_filtered['n_gt_labels'] == 0]
    gt_runs = df_filtered[df_filtered['n_gt_labels'] == train_set_size]

    if len(baseline_runs) == 0:
        print(f"Warning: No baseline (0 GT) run found for {dataset} + {strong_model}")
        baseline_acc = None
    else:
        baseline_acc = baseline_runs.iloc[0]['accuracy']

    if len(gt_runs) == 0:
        print(f"Warning: No ground truth (100% GT) run found for {dataset} + {strong_model}")
        gt_acc = None
    else:
        gt_acc = gt_runs.iloc[0]['accuracy']

    # Separate random (sample) and disagreement runs
    df_random = df_filtered[
        ((df_filtered['mix_strategy'] == 'sample') | (df_filtered['mix_strategy'].isna())) &
        (df_filtered['n_gt_labels'] > 0) &
        (df_filtered['n_gt_labels'] < train_set_size)
    ].copy()

    df_disagreement = df_filtered[
        (df_filtered['mix_strategy'] == 'disagreement') &
        (df_filtered['n_gt_labels'] > 0)
    ].copy()

    # Sort by number of GT labels
    df_random = df_random.sort_values('n_gt_labels')
    df_disagreement = df_disagreement.sort_values('n_gt_labels')

    # Calculate PGR for all runs (if baseline and GT are available)
    if baseline_acc is not None and gt_acc is not None and gt_acc > baseline_acc:
        df_filtered['pgr'] = (df_filtered['accuracy'] - baseline_acc) / (gt_acc - baseline_acc)
        df_random['pgr'] = (df_random['accuracy'] - baseline_acc) / (gt_acc - baseline_acc)
        df_disagreement['pgr'] = (df_disagreement['accuracy'] - baseline_acc) / (gt_acc - baseline_acc)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot baseline (0 GT labels)
    if baseline_acc is not None:
        ax.axhline(y=baseline_acc, color='gray', linestyle=':', linewidth=2,
                  label=f'Baseline (0 GT, {baseline_acc:.4f})', alpha=0.7)
        ax.plot([0], [baseline_acc], 'o', color='gray', markersize=8)

    # Plot ground truth (100% GT labels)
    if gt_acc is not None:
        ax.axhline(y=gt_acc, color='black', linestyle='--', linewidth=2,
                  label=f'Ground Truth (all GT, {gt_acc:.4f})', alpha=0.7)
        ax.plot([train_set_size], [gt_acc], 'o', color='black', markersize=8)

    # Plot random selection
    if len(df_random) > 0:
        ax.plot(df_random['n_gt_labels'], df_random['accuracy'],
               marker='o', linewidth=2.5, markersize=10,
               color='#E74C3C', label='Random Selection', zorder=5)

        # Add value labels
        for _, row in df_random.iterrows():
            ax.annotate(f"{row['accuracy']:.4f}",
                       xy=(row['n_gt_labels'], row['accuracy']),
                       xytext=(0, 10), textcoords='offset points',
                       ha='center', fontsize=9, color='#E74C3C')

    # Plot disagreement-based selection
    if len(df_disagreement) > 0:
        ax.plot(df_disagreement['n_gt_labels'], df_disagreement['accuracy'],
               marker='s', linewidth=2.5, markersize=10,
               color='#3498DB', label='Disagreement-Based', zorder=5)

        # Add value labels
        for _, row in df_disagreement.iterrows():
            ax.annotate(f"{row['accuracy']:.4f}",
                       xy=(row['n_gt_labels'], row['accuracy']),
                       xytext=(0, -15), textcoords='offset points',
                       ha='center', fontsize=9, color='#3498DB')

    # Customize plot
    ax.set_xlabel('Number of Ground Truth Labels', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    title = f'Label Selection Comparison: {dataset} ({strong_model}'
    if weak_model:
        title += f', weak={weak_model}'
    title += ')'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)

    # Set x-axis limits
    ax.set_xlim(-100, train_set_size + 100)

    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{output_name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot: {plot_path}")

    # Print analysis
    print("\n" + "="*80)
    print(f"DISAGREEMENT vs RANDOM COMPARISON - {dataset} ({strong_model})")
    print("="*80)

    if baseline_acc is not None:
        print(f"\nBaseline (0 GT labels):           {baseline_acc:.4f}")
    if gt_acc is not None:
        print(f"Ground Truth (all GT labels):     {gt_acc:.4f}")

    print("\nRandom Selection Results:")
    print(f"{'GT Labels':<12} {'Accuracy':<10} {'PGR':<8} {'Gain vs Baseline':<20} {'% of Gap Closed'}")
    print("-" * 80)
    for _, row in df_random.iterrows():
        gain = row['accuracy'] - baseline_acc if baseline_acc else 0
        pct_gap = (gain / (gt_acc - baseline_acc) * 100) if (gt_acc and baseline_acc) else 0
        pgr_val = row.get('pgr', np.nan)
        pgr_str = f"{pgr_val:.4f}" if not pd.isna(pgr_val) else "N/A"
        print(f"{row['n_gt_labels']:<12} {row['accuracy']:<10.4f} {pgr_str:<8} {gain:+.4f} ({gain*100:+.2f}pp) {pct_gap:>8.1f}%")

    print("\nDisagreement-Based Selection Results:")
    print(f"{'GT Labels':<12} {'Accuracy':<10} {'PGR':<8} {'Gain vs Baseline':<20} {'% of Gap Closed'}")
    print("-" * 80)
    for _, row in df_disagreement.iterrows():
        gain = row['accuracy'] - baseline_acc if baseline_acc else 0
        pct_gap = (gain / (gt_acc - baseline_acc) * 100) if (gt_acc and baseline_acc) else 0
        pgr_val = row.get('pgr', np.nan)
        pgr_str = f"{pgr_val:.4f}" if not pd.isna(pgr_val) else "N/A"
        print(f"{row['n_gt_labels']:<12} {row['accuracy']:<10.4f} {pgr_str:<8} {gain:+.4f} ({gain*100:+.2f}pp) {pct_gap:>8.1f}%")

    # Direct comparisons at matched budgets
    print("\nDirect Comparisons (matched budget):")
    print(f"{'GT Labels':<12} {'Random':<10} {'Disagreement':<15} {'Improvement'}")
    print("-" * 70)

    for _, dis_row in df_disagreement.iterrows():
        budget = dis_row['n_gt_labels']
        # Find closest random run
        random_match = df_random.iloc[(df_random['n_gt_labels'] - budget).abs().argsort()[:1]]

        if len(random_match) > 0:
            random_row = random_match.iloc[0]
            random_budget = random_row['n_gt_labels']
            random_acc = random_row['accuracy']
            dis_acc = dis_row['accuracy']
            improvement = dis_acc - random_acc

            if abs(random_budget - budget) < 100:  # Only if budgets are close
                print(f"{budget:<12} {random_acc:<10.4f} {dis_acc:<15.4f} {improvement:+.4f} ({improvement*100:+.2f}pp)")

    # Label efficiency analysis
    print("\n" + "="*80)
    print("LABEL EFFICIENCY ANALYSIS")
    print("="*80)

    if baseline_acc and len(df_random) > 0:
        print("\nRandom Selection - Accuracy Gain per Label:")
        for i, row in df_random.iterrows():
            gain_per_label = (row['accuracy'] - baseline_acc) / row['n_gt_labels']
            print(f"  {row['n_gt_labels']} labels: {gain_per_label*10000:.2f} accuracy points per 1000 labels")

    if baseline_acc and len(df_disagreement) > 0:
        print("\nDisagreement-Based - Accuracy Gain per Label:")
        for i, row in df_disagreement.iterrows():
            gain_per_label = (row['accuracy'] - baseline_acc) / row['n_gt_labels']
            print(f"  {row['n_gt_labels']} labels: {gain_per_label*10000:.2f} accuracy points per 1000 labels")

    # Save data table
    data_path = os.path.join(output_dir, f"{output_name}_data.csv")
    save_cols = ['mix_strategy', 'n_gt_labels', 'labeling_budget', 'mix_ratio', 'accuracy', 'pgr', 'run_name', 'run_id']
    save_cols = [c for c in save_cols if c in df_filtered.columns]

    df_to_save = df_filtered[save_cols].copy()
    df_to_save = df_to_save.sort_values(['mix_strategy', 'n_gt_labels'])
    df_to_save.to_csv(data_path, index=False)
    print(f"\nSaved data: {data_path}")

    plt.close()


def main():
    # ========================================================================
    # CONFIGURATION
    # ========================================================================

    WANDB_ENTITY = "maxliving-personal"
    WANDB_PROJECT = "weak-to-strong-mixing"
    DATASET = "boolq"
    STRONG_MODEL = "gpt2-large"
    WEAK_MODEL = "gpt2-medium"
    TRAIN_SET_SIZE = 4714  # BoolQ train2 size
    OUTPUT_DIR = "plots/disagreement_analysis"
    METRIC = 'best_checkpoint'

    # ========================================================================
    # FETCH AND PLOT
    # ========================================================================

    print("="*80)
    print("Disagreement vs Random Label Selection Analysis")
    print("="*80)
    print(f"Entity: {WANDB_ENTITY}")
    print(f"Project: {WANDB_PROJECT}")
    print(f"Dataset: {DATASET}")
    print(f"Strong Model: {STRONG_MODEL}")
    print(f"Metric: {METRIC}")
    print("="*80)

    # Fetch all runs for this dataset
    df_all = fetch_all_finished_runs(WANDB_ENTITY, WANDB_PROJECT, dataset=DATASET, metric=METRIC)

    if len(df_all) == 0:
        print("\nNo data to plot!")
        return

    # Generate comparison plot
    output_name = f"{DATASET}_{STRONG_MODEL}_{WEAK_MODEL}_disagreement_comparison"
    plot_disagreement_comparison(df_all, DATASET, STRONG_MODEL,
                                weak_model=WEAK_MODEL,  # Filter to gpt2 weak model for fair comparison
                                train_set_size=TRAIN_SET_SIZE,
                                output_dir=OUTPUT_DIR,
                                output_name=output_name)

    print(f"\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == '__main__':
    main()
