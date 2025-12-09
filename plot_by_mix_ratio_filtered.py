"""
Create plots similar to notebooks/boolq.png but filtered by mix_ratio.
Uses clean runs only (filtered version).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os

from plot_mixing_analysis import fetch_all_finished_runs

sns.set_style('whitegrid')


def filter_to_clean_runs(df):
    """
    Filter to include only clean, valid runs.

    Strategy:
    - For gpt2-xl + gpt2-large weak: ONLY include runs from sweep_subfolder='gpt2xl_gpt2large_clean'
    - For everything else: include runs from sweep_subfolder='default'
    """

    # Ensure sweep_subfolder column exists
    if 'sweep_subfolder' not in df.columns:
        print("Warning: sweep_subfolder not in data, cannot filter")
        return df

    # Create filter mask
    mask = (
        # Include all default runs EXCEPT gpt2-xl with gpt2-large weak
        (
            (df['sweep_subfolder'] == 'default') &
            ~((df['model_size'] == 'gpt2-xl') & (df['weak_model_size'] == 'gpt2-large'))
        ) |
        # Include gpt2xl_gpt2large_clean runs for gpt2-xl with gpt2-large weak
        (
            (df['sweep_subfolder'] == 'gpt2xl_gpt2large_clean') &
            (df['model_size'] == 'gpt2-xl') &
            (df['weak_model_size'] == 'gpt2-large')
        ) |
        # Include all baseline/ground truth runs from default
        (
            (df['sweep_subfolder'] == 'default') &
            (df['weak_model_size'].isna())
        )
    )

    n_before = len(df)
    df_filtered = df[mask].copy()
    n_excluded = n_before - len(df_filtered)

    print(f"Filtered: kept {len(df_filtered)}/{n_before} runs (excluded {n_excluded})")

    return df_filtered


def plot_by_mix_ratio(df, dataset, mix_ratio, output_dir='plots'):
    """Create a plot similar to notebooks/boolq.png for a specific mix_ratio.

    Args:
        df: DataFrame with run data
        dataset: Dataset name to plot
        mix_ratio: Mix ratio to filter for (0.0, 0.25, 1.0, etc.)
        output_dir: Directory to save plots
    """
    # Filter to dataset and mix_ratio
    cur_df = df[(df['ds_name'] == dataset) & (df['mix_ratio'] == mix_ratio)].copy()

    if len(cur_df) == 0:
        print(f"No data found for {dataset} with mix_ratio={mix_ratio}")
        return

    print(f"\n{'='*60}")
    print(f"Plotting {dataset} with mix_ratio={mix_ratio}")
    print(f"{'='*60}")

    # Calculate base accuracies (ground truth for each model size)
    # Get all ground truth runs (mix_ratio=1.0, no weak model)
    gt_df = df[(df['ds_name'] == dataset) &
               (df['mix_ratio'] == 1.0) &
               (df['weak_model_size'].isna())].copy()

    base_accuracies = gt_df.groupby('model_size').agg({
        'accuracy': 'mean',
        'run_id': 'count'
    }).sort_values('accuracy')

    base_accuracy_lookup = base_accuracies['accuracy'].to_dict()
    base_accuracies = base_accuracies.reset_index()

    # Add strong_model_accuracy and weak_model_accuracy columns
    cur_df['strong_model_accuracy'] = cur_df['model_size'].apply(
        lambda x: base_accuracy_lookup.get(x, np.nan)
    )
    cur_df.loc[~cur_df['weak_model_size'].isna(), 'weak_model_accuracy'] = cur_df.loc[
        ~cur_df['weak_model_size'].isna(), 'weak_model_size'
    ].apply(lambda x: base_accuracy_lookup.get(x, np.nan))

    # Calculate PGR (excluding cases where weak >= strong)
    valid_pgr_index = (
        (~cur_df['weak_model_size'].isna()) &
        (cur_df['weak_model_size'] != cur_df['model_size']) &
        (cur_df['strong_model_accuracy'] > cur_df['weak_model_accuracy'])
    )
    cur_df.loc[valid_pgr_index, 'pgr'] = (
        (cur_df.loc[valid_pgr_index, 'accuracy'] - cur_df.loc[valid_pgr_index, 'weak_model_accuracy']) /
        (cur_df.loc[valid_pgr_index, 'strong_model_accuracy'] - cur_df.loc[valid_pgr_index, 'weak_model_accuracy'])
    )

    # Label ground truth runs
    cur_df.loc[cur_df['weak_model_size'].isna(), "weak_model_size"] = "ground truth"

    # Sort for plotting
    plot_df = cur_df.copy().sort_values(['strong_model_accuracy']).sort_values(['loss'], ascending=False)

    # Calculate median PGR by loss type
    pgr_results = plot_df[~plot_df['pgr'].isna()].groupby(['loss']).aggregate({"pgr": "median"})

    print(f"\nMedian PGR by loss type:")
    print(pgr_results)

    # Create color palette
    weak_models = [m for m in plot_df['weak_model_size'].unique() if m != 'ground truth']
    palette = sns.color_palette('colorblind', n_colors=len(weak_models))
    color_dict = {'ground truth': 'black'}
    for i, model in enumerate(weak_models):
        color_dict[model] = palette[i]

    # Create plot
    plt.figure(figsize=(10, 7))

    sns.lineplot(
        data=plot_df,
        x='strong_model_accuracy',
        y='accuracy',
        hue='weak_model_size',
        style='loss',
        markers=True,
        palette=color_dict
    )

    # Add PGR table
    if len(pgr_results) > 0:
        pd.plotting.table(
            plt.gca(),
            pgr_results.round(4),
            loc='lower right',
            colWidths=[0.1, 0.1],
            cellLoc='center',
            rowLoc='center'
        )

    # Set x-axis ticks
    plt.xticks(
        ticks=base_accuracies['accuracy'],
        labels=[f"{e} ({base_accuracy_lookup[e]:.4f})" for e in base_accuracies['model_size']],
        rotation=90
    )

    # Labels and title
    mix_ratio_pct = int(mix_ratio * 100)
    plt.xlabel('strong_model_accuracy')
    plt.ylabel('accuracy')
    plt.title(f"Dataset: {dataset} (mix_ratio: {mix_ratio_pct}%)")
    plt.legend(loc='upper left')

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{dataset}_mix{mix_ratio_pct}pct_clean.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {filepath}", flush=True)

    plt.close()


def main():
    # Configuration
    WANDB_ENTITY = "maxliving-personal"
    WANDB_PROJECT = "weak-to-strong-mixing"
    DATASET = "boolq"
    MIX_RATIOS = [0.0, 0.25, 1.0]  # 0%, 25%, 100%
    OUTPUT_DIR = "plots_clean"
    METRIC = 'best_checkpoint'

    print("="*60)
    print("Plotting by Mix Ratio (Clean Results)")
    print("="*60)
    print(f"Entity: {WANDB_ENTITY}")
    print(f"Project: {WANDB_PROJECT}")
    print(f"Dataset: {DATASET}")
    print(f"Mix Ratios: {[f'{int(mr*100)}%' for mr in MIX_RATIOS]}")
    print(f"Metric: {METRIC}")
    print("Using sweep_subfolder-based filtering")
    print("="*60)

    # Fetch all runs
    print("\nFetching data from W&B...", flush=True)
    df = fetch_all_finished_runs(WANDB_ENTITY, WANDB_PROJECT, dataset=DATASET, metric=METRIC)
    print(f"Fetched {len(df)} runs", flush=True)

    if len(df) == 0:
        print("\nNo data found!")
        return

    # Filter to clean runs
    print("\nFiltering to clean runs...")
    df = filter_to_clean_runs(df)

    # Filter to sample strategy only
    if 'mix_strategy' in df.columns:
        n_before = len(df)
        df = df[df['mix_strategy'] == 'sample'].copy()
        n_excluded = n_before - len(df)
        if n_excluded > 0:
            print(f"Filtered to sample strategy only (excluded {n_excluded} runs)")

    # Create plots for each mix ratio
    for mix_ratio in MIX_RATIOS:
        plot_by_mix_ratio(df, DATASET, mix_ratio, OUTPUT_DIR)

    print(f"\n{'='*60}")
    print("All plots created!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
