"""
Plot mixing rate analysis - filtered version
Uses explicit inclusion criteria for clean results
"""

import os
from plot_mixing_analysis import fetch_all_finished_runs, plot_mixing_by_strong_model

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


def main():
    WANDB_ENTITY = "maxliving-personal"
    WANDB_PROJECT = "weak-to-strong-mixing"
    DATASETS = ["boolq"]
    OUTPUT_DIR = "plots_clean"
    METRIC = 'best_checkpoint'
    EXPECTED_EPOCHS = 4
    EXPECTED_EVAL_EVERY = 100

    print("="*60)
    print("Mixing Rate Analysis (Filtered - Clean Results Only)")
    print("="*60)
    print(f"Entity: {WANDB_ENTITY}")
    print(f"Project: {WANDB_PROJECT}")
    print(f"Datasets: {DATASETS}")
    print(f"Metric: {METRIC}")
    print("Using sweep_subfolder-based filtering")
    print("="*60)

    # Fetch all runs
    print("\nFetching data from W&B...")
    df_all = fetch_all_finished_runs(WANDB_ENTITY, WANDB_PROJECT, dataset=None, metric=METRIC)

    # Filter to clean runs
    print("\nFiltering to clean runs...")
    df_all = filter_to_clean_runs(df_all)

    if len(df_all) == 0:
        print("\nNo data to plot!")
        return

    # Process each dataset
    for dataset in DATASETS:
        print("\n" + "="*60)
        print(f"PROCESSING DATASET: {dataset}")
        print("="*60)

        df = df_all[df_all['ds_name'] == dataset].copy()

        if len(df) == 0:
            print(f"\nNo data found for {dataset}, skipping...")
            continue

        # Filter to sample strategy only
        if 'mix_strategy' in df.columns:
            n_before = len(df)
            df = df[df['mix_strategy'] == 'sample'].copy()
            n_excluded = n_before - len(df)
            if n_excluded > 0:
                print(f"Filtered to sample strategy only (excluded {n_excluded} runs)")

        if len(df) == 0:
            print(f"\nNo sample-based mixing data found for {dataset}, skipping...")
            continue

        # Print summary
        print("\n" + "="*60)
        print(f"DATA SUMMARY - {dataset}")
        print("="*60)
        print(f"Total runs: {len(df)}")
        if 'sweep_subfolder' in df.columns:
            print("\nRuns by sweep_subfolder:")
            print(df['sweep_subfolder'].value_counts())
        print("="*60)

        # Generate plot
        metric_output_dir = os.path.join(OUTPUT_DIR, METRIC)
        output_name = f"{dataset}_mixing_analysis_clean"
        plot_mixing_by_strong_model(df, dataset, metric_output_dir, output_name,
                                   expected_epochs=EXPECTED_EPOCHS,
                                   expected_eval_every=EXPECTED_EVAL_EVERY)

    print(f"\n" + "="*60)
    print("All datasets processed!")
    print("="*60)


if __name__ == '__main__':
    main()
