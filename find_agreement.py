#!/usr/bin/env python3
"""
Find agreement/disagreement between two fine-tuned models (weak vs strong).

This script:
1. Loads predictions from a fine-tuned weak model (from results.pkl)
2. Loads predictions from a fine-tuned strong model (from results.pkl)
3. Identifies where they agree or disagree
4. Exports agreement/disagreement analysis to CSV files

Supports:
- Direct checkpoint paths
- WandB run IDs
- WandB run names
"""

import os
import pickle
import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import wandb


def resolve_checkpoint_path(
    identifier: str,
    wandb_entity: str = "maxliving-personal",
    wandb_project: str = "weak-to-strong-mixing",
    results_base_dir: str = "./results"
) -> str:
    """Resolve a WandB run ID/name or path to a checkpoint directory.

    Args:
        identifier: Can be:
            - Full checkpoint path (e.g., "./results/default/bs=32-...")
            - WandB run ID (e.g., "cr50yyzj")
            - WandB run name (e.g., "default_bs=32-dn=boolq-...")
        wandb_entity: WandB entity/username
        wandb_project: WandB project name
        results_base_dir: Base directory where results are stored

    Returns:
        Absolute path to checkpoint directory

    Raises:
        ValueError: If identifier cannot be resolved to a valid checkpoint
    """
    # Check if it's already a valid path
    path = Path(identifier)
    if path.exists():
        results_pkl = path / "results.pkl"
        if results_pkl.exists():
            return str(path.absolute())
        else:
            print(f"Warning: {path} exists but doesn't contain results.pkl")

    # Try to resolve as WandB run
    try:
        api = wandb.Api(timeout=60)

        # Try as run ID (8 chars alphanumeric)
        if len(identifier) == 8 and identifier.isalnum():
            try:
                run = api.run(f"{wandb_entity}/{wandb_project}/{identifier}")
                run_name = run.name
                print(f"Resolved WandB run ID '{identifier}' to run: {run_name}")
            except Exception as e:
                print(f"Could not find run ID '{identifier}': {e}")
                run_name = None
        else:
            # Try as run name
            runs = list(api.runs(f"{wandb_entity}/{wandb_project}", per_page=500))
            matching_runs = [r for r in runs if r.name == identifier]
            if matching_runs:
                run_name = matching_runs[0].name
                print(f"Found WandB run: {run_name}")
            else:
                run_name = None

        if run_name:
            # Construct checkpoint path
            # Try with "default" subfolder first (most common)
            checkpoint_path = Path(results_base_dir) / "default" / run_name
            if checkpoint_path.exists() and (checkpoint_path / "results.pkl").exists():
                return str(checkpoint_path.absolute())

            # Try without subfolder
            checkpoint_path = Path(results_base_dir) / run_name
            if checkpoint_path.exists() and (checkpoint_path / "results.pkl").exists():
                return str(checkpoint_path.absolute())

            raise ValueError(
                f"Found WandB run '{run_name}' but couldn't find checkpoint directory.\n"
                f"Tried:\n  - {results_base_dir}/default/{run_name}\n  - {results_base_dir}/{run_name}\n"
                f"Make sure the checkpoint was saved locally."
            )

    except Exception as e:
        print(f"WandB API error: {e}")

    # If we get here, couldn't resolve
    raise ValueError(
        f"Could not resolve '{identifier}' to a valid checkpoint path.\n"
        f"Please provide either:\n"
        f"  - A full path to a checkpoint directory\n"
        f"  - A WandB run ID (8 characters)\n"
        f"  - A WandB run name\n"
        f"\nMake sure the checkpoint directory contains results.pkl"
    )


def load_model_predictions_from_pkl(
    checkpoint_path: str,
    model_name: str = "model",
    use_test_results: bool = True
) -> pd.DataFrame:
    """Load fine-tuned model predictions from results.pkl file.

    Args:
        checkpoint_path: Path to the checkpoint directory containing results.pkl
        model_name: Name to use for this model in column names (e.g., "weak", "strong")
        use_test_results: If True, load 'test_results'; if False, load 'inference_results'

    Returns:
        DataFrame with columns: idx, txt, {model_name}_soft_label, {model_name}_hard_label, ground_truth

    Raises:
        FileNotFoundError: If checkpoint_path or results.pkl doesn't exist
        KeyError: If required keys are missing from results.pkl
        ValueError: If results data is empty or malformed
    """
    # Validate checkpoint path
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

    # Construct results.pkl path
    results_pkl_path = checkpoint_path / "results.pkl"
    if not results_pkl_path.exists():
        raise FileNotFoundError(
            f"results.pkl not found at: {results_pkl_path}\n"
            f"Make sure the model was trained and evaluated with save_path set."
        )

    # Load pickle file
    print(f"Loading predictions from {results_pkl_path}...")
    with open(results_pkl_path, "rb") as f:
        results_data = pickle.load(f)

    # Select dataset (test_results or inference_results)
    dataset_key = "test_results" if use_test_results else "inference_results"
    if dataset_key not in results_data:
        raise KeyError(
            f"'{dataset_key}' not found in results.pkl. "
            f"Available keys: {list(results_data.keys())}"
        )

    predictions_ds = results_data[dataset_key]

    if predictions_ds is None or len(predictions_ds) == 0:
        raise ValueError(f"No predictions found in '{dataset_key}'")

    print(f"Loaded {len(predictions_ds)} predictions from {dataset_key}")

    # Convert to DataFrame
    records = []
    for i, example in enumerate(predictions_ds):
        soft_label = example.get('soft_label')
        # soft_label is a list [prob_class_0, prob_class_1]
        # Extract probability of class 1
        if isinstance(soft_label, list):
            soft_label = soft_label[1]

        records.append({
            'idx': i,
            'txt': example.get('txt', ''),
            f'{model_name}_soft_label': float(soft_label),
            f'{model_name}_hard_label': example.get('hard_label'),
            'ground_truth': example.get('gt_label'),
        })

    return pd.DataFrame(records)


def find_agreement_disagreement(
    weak_checkpoint_path: str,
    strong_checkpoint_path: str,
    weak_name: str = "weak",
    strong_name: str = "strong",
    use_test_results: bool = True,
    validate_alignment: bool = True
) -> Dict[str, pd.DataFrame]:
    """Find where two fine-tuned models agree/disagree on predictions.

    Args:
        weak_checkpoint_path: Path to weak model checkpoint directory
        strong_checkpoint_path: Path to strong model checkpoint directory
        weak_name: Name for weak model (used in column names)
        strong_name: Name for strong model (used in column names)
        use_test_results: If True, use test_results; if False, use inference_results
        validate_alignment: If True, validate that both models used same test examples

    Returns:
        Dictionary with keys 'all', 'agree', 'disagree'

    Raises:
        ValueError: If datasets have different lengths or mismatched examples
    """
    dataset_type = "test set" if use_test_results else "inference set (train2)"
    print(f"\n=== Comparing {weak_name} vs {strong_name} on {dataset_type} ===\n")

    # Load weak model predictions
    print(f"Loading {weak_name} model predictions...")
    weak_df = load_model_predictions_from_pkl(
        weak_checkpoint_path,
        model_name=weak_name,
        use_test_results=use_test_results
    )

    # Load strong model predictions
    print(f"Loading {strong_name} model predictions...")
    strong_df = load_model_predictions_from_pkl(
        strong_checkpoint_path,
        model_name=strong_name,
        use_test_results=use_test_results
    )

    # Validate alignment
    if validate_alignment:
        print("\nValidating dataset alignment...")
        if len(weak_df) != len(strong_df):
            raise ValueError(
                f"Dataset length mismatch: {weak_name}={len(weak_df)}, {strong_name}={len(strong_df)}\n"
                f"Both models must be evaluated on the same test set."
            )

        # Check if texts match for first 10 examples
        n_check = min(10, len(weak_df))
        for i in range(n_check):
            if weak_df.iloc[i]['txt'] != strong_df.iloc[i]['txt']:
                raise ValueError(
                    f"Text mismatch at index {i}. Models may have been evaluated on different datasets.\n"
                    f"Weak: {weak_df.iloc[i]['txt'][:50]}...\n"
                    f"Strong: {strong_df.iloc[i]['txt'][:50]}..."
                )
        print(f"  ✓ Validated alignment (checked {n_check} examples)")

    # Merge on idx
    merged_df = pd.merge(weak_df, strong_df, on='idx', how='inner', suffixes=('_weak', '_strong'))

    # Keep only one txt and ground_truth column
    if 'txt_weak' in merged_df.columns:
        merged_df['txt'] = merged_df['txt_weak']
        merged_df = merged_df.drop(columns=['txt_weak', 'txt_strong'])

    if 'ground_truth_weak' in merged_df.columns:
        # Verify ground truth matches
        if validate_alignment and not (merged_df['ground_truth_weak'] == merged_df['ground_truth_strong']).all():
            raise ValueError("Ground truth labels don't match between models!")
        merged_df['ground_truth'] = merged_df['ground_truth_weak']
        merged_df = merged_df.drop(columns=['ground_truth_weak', 'ground_truth_strong'])

    # Find agreement/disagreement
    merged_df['agree'] = (
        merged_df[f'{weak_name}_hard_label'] == merged_df[f'{strong_name}_hard_label']
    )

    # Calculate prediction confidence difference
    merged_df['confidence_diff'] = abs(
        merged_df[f'{weak_name}_soft_label'] - merged_df[f'{strong_name}_soft_label']
    )

    # Create result dictionary
    results = {
        'all': merged_df,
        'agree': merged_df[merged_df['agree']],
        'disagree': merged_df[~merged_df['agree']],
    }

    return results


def print_summary(results: Dict[str, pd.DataFrame], model1_name: str, model2_name: str) -> None:
    """Print summary statistics of agreement/disagreement.

    Args:
        results: Dictionary from find_agreement_disagreement()
        model1_name: Name of first model
        model2_name: Name of second model
    """
    total = len(results['all'])
    agree_count = len(results['agree'])
    disagree_count = len(results['disagree'])

    # Calculate average confidence difference
    avg_confidence_diff = results['all']['confidence_diff'].mean()

    # Calculate accuracy against ground truth
    df = results['all']
    model1_correct = (df[f'{model1_name}_hard_label'] == df['ground_truth']).sum()
    model2_correct = (df[f'{model2_name}_hard_label'] == df['ground_truth']).sum()
    model1_acc = 100 * model1_correct / total
    model2_acc = 100 * model2_correct / total

    print("="*80)
    print("AGREEMENT SUMMARY")
    print("="*80)
    print(f"Total examples: {total}")
    print(f"Agree: {agree_count} ({100*agree_count/total:.1f}%)")
    print(f"Disagree: {disagree_count} ({100*disagree_count/total:.1f}%)")
    print(f"Average confidence difference: {avg_confidence_diff:.4f}")
    print()
    print("ACCURACY VS GROUND TRUTH:")
    print(f"  {model1_name}: {model1_correct}/{total} ({model1_acc:.1f}%)")
    print(f"  {model2_name}: {model2_correct}/{total} ({model2_acc:.1f}%)")
    print("="*80)


def print_examples(df: pd.DataFrame, n: int, title: str, model1_name: str, model2_name: str) -> None:
    """Print example texts and labels.

    Args:
        df: DataFrame with examples
        n: Number of examples to print
        title: Title for the examples section
        model1_name: Name of first model
        model2_name: Name of second model
    """
    print(f"\n{title} ({len(df)} total):")
    print("-"*80)

    for i, row in df.head(n).iterrows():
        print(f"\nExample {i}:")
        print(f"  Text: {row['txt'][:200]}...")  # First 200 chars
        print(f"  Ground truth: {row['ground_truth']}")
        print(f"  {model1_name} prediction: {row[f'{model1_name}_hard_label']} (confidence: {row[f'{model1_name}_soft_label']:.4f})")
        print(f"  {model2_name} prediction: {row[f'{model2_name}_hard_label']} (confidence: {row[f'{model2_name}_soft_label']:.4f})")
        print(f"  Confidence diff: {row['confidence_diff']:.4f}")


def main(
    weak_checkpoint_path: Optional[str] = None,
    strong_checkpoint_path: Optional[str] = None,
    weak_name: str = "weak",
    strong_name: str = "strong",
    use_test_results: bool = True,
    output_dir: Optional[str] = None,
    show_examples: bool = True,
    n_examples: int = 5,
    results_base_dir: str = "./results"
):
    """Compare predictions from two fine-tuned models.

    Args:
        weak_checkpoint_path: Path to weak model checkpoint directory (required)
                             Can be: full path, WandB run ID, or WandB run name
        strong_checkpoint_path: Path to strong model checkpoint directory (required)
                               Can be: full path, WandB run ID, or WandB run name
        weak_name: Display name for weak model (default: "weak")
        strong_name: Display name for strong model (default: "strong")
        use_test_results: Use test_results (True) or inference_results (False)
        output_dir: Output directory for CSVs (default: ./agreement_analysis/{weak_name}_vs_{strong_name})
        show_examples: Whether to print example agreements/disagreements
        n_examples: Number of examples to show
        results_base_dir: Base directory where checkpoint results are stored (default: "./results")
    """
    # ========================================================================
    # VALIDATION
    # ========================================================================

    if weak_checkpoint_path is None or strong_checkpoint_path is None:
        print("ERROR: Both weak_checkpoint_path and strong_checkpoint_path must be provided!")
        print()
        print("Usage:")
        print("  python find_agreement.py \\")
        print("    --weak_checkpoint_path='./path/to/weak/checkpoint' \\")
        print("    --strong_checkpoint_path='./path/to/strong/checkpoint'")
        print()
        print("You can also use WandB run IDs or names:")
        print("  python find_agreement.py \\")
        print("    --weak_checkpoint_path='cr50yyzj' \\")
        print("    --strong_checkpoint_path='znx4qjgf'")
        print()
        print("Optional arguments:")
        print("  --weak_name='gpt2'              # Display name for weak model")
        print("  --strong_name='gpt2-xl'         # Display name for strong model")
        print("  --use_test_results=True         # Use test_results (True) or inference_results (False)")
        print("  --output_dir='./output'         # Custom output directory")
        print("  --show_examples=True            # Show example predictions")
        print("  --n_examples=5                  # Number of examples to show")
        print("  --results_base_dir='./results'  # Base directory for checkpoint results")
        return

    # Set default output directory
    if output_dir is None:
        output_dir = os.path.join("agreement_analysis", f"{weak_name}_vs_{strong_name}")

    # ========================================================================
    # ANALYSIS
    # ========================================================================

    dataset_type = "test set" if use_test_results else "inference set (train2)"

    print("="*80)
    print(f"Fine-tuned Model Comparison: {weak_name} vs {strong_name}")
    print("="*80)
    print(f"Weak model checkpoint:   {weak_checkpoint_path}")
    print(f"Strong model checkpoint: {strong_checkpoint_path}")
    print(f"Dataset:                 {dataset_type}")
    print("="*80)
    print()

    try:
        # Resolve checkpoint paths (handles WandB run IDs/names)
        weak_checkpoint_path = resolve_checkpoint_path(weak_checkpoint_path, results_base_dir=results_base_dir)
        strong_checkpoint_path = resolve_checkpoint_path(strong_checkpoint_path, results_base_dir=results_base_dir)

        print(f"\nResolved paths:")
        print(f"  Weak:   {weak_checkpoint_path}")
        print(f"  Strong: {strong_checkpoint_path}")

        # Find agreement/disagreement
        results = find_agreement_disagreement(
            weak_checkpoint_path=weak_checkpoint_path,
            strong_checkpoint_path=strong_checkpoint_path,
            weak_name=weak_name,
            strong_name=strong_name,
            use_test_results=use_test_results,
            validate_alignment=True
        )

        # Print summary
        print_summary(results, weak_name, strong_name)

        # Show examples
        if show_examples:
            print_examples(results['agree'], n=n_examples, title="Examples where models AGREE",
                          model1_name=weak_name, model2_name=strong_name)
            print_examples(results['disagree'], n=n_examples, title="Examples where models DISAGREE",
                          model1_name=weak_name, model2_name=strong_name)

        # Export to CSV
        print()
        print("="*80)
        print("Exporting to CSV...")
        print("="*80)

        Path(output_dir).mkdir(exist_ok=True, parents=True)

        # Save metadata
        metadata = {
            "weak_checkpoint_path": str(weak_checkpoint_path),
            "strong_checkpoint_path": str(strong_checkpoint_path),
            "weak_name": weak_name,
            "strong_name": strong_name,
            "dataset_type": dataset_type,
            "use_test_results": use_test_results,
            "total_examples": len(results['all']),
            "agree_count": len(results['agree']),
            "disagree_count": len(results['disagree']),
        }

        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"  metadata: {metadata_path}")

        # Save each category
        for category, df in results.items():
            if category == 'all':
                continue
            output_path = f"{output_dir}/{category}.csv"
            df.to_csv(output_path, index=False)
            print(f"  {category}: {output_path} ({len(df)} examples)")

        # Save full dataset with agreement labels
        full_output = f"{output_dir}/all_with_labels.csv"
        results['all'].to_csv(full_output, index=False)
        print(f"  all: {full_output} ({len(results['all'])} examples)")

        print("="*80)
        print("\nDone!")

    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"\nERROR: {e}")
        print("\nTroubleshooting:")
        print("  1. Verify checkpoint paths exist")
        print("  2. Ensure results.pkl exists in each checkpoint directory")
        print("  3. Check that both models were trained on the same dataset")
        print("  4. If using WandB run ID/name, check it exists in the project")
        return


if __name__ == '__main__':
    import fire
    fire.Fire(main)
