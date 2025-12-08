#!/usr/bin/env python3
"""
Find agreement/disagreement between two fine-tuned models (weak vs strong).

This script:
1. Loads a fine-tuned weak model from checkpoint and runs fresh inference
2. Loads a fine-tuned strong model from checkpoint and runs fresh inference
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
import re
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import wandb
import torch
from datasets import load_from_disk

from weak_to_strong.datasets import load_dataset, tokenize_dataset
from weak_to_strong.model import TransformerWithHead
from weak_to_strong.eval import eval_model_acc
from weak_to_strong.common import get_tokenizer
from train_simple import get_config_foldername


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
        # Check for transformers checkpoint (config.json is always present)
        config_file = path / "config.json"
        best_config = path / "best_checkpoint" / "config.json"

        if config_file.exists() or best_config.exists():
            return str(path.absolute())
        else:
            print(f"Warning: {path} exists but doesn't contain a valid transformers checkpoint (no config.json)")

    # Try to resolve as WandB run
    try:
        api = wandb.Api(timeout=60)
        run = None

        # Try as run ID (8 chars alphanumeric)
        if len(identifier) == 8 and identifier.isalnum():
            try:
                run = api.run(f"{wandb_entity}/{wandb_project}/{identifier}")
                print(f"Resolved WandB run ID '{identifier}' to run: {run.name}")
            except Exception as e:
                print(f"Could not find run ID '{identifier}': {e}")
        else:
            # Try as run name
            runs = list(api.runs(f"{wandb_entity}/{wandb_project}", per_page=500))
            matching_runs = [r for r in runs if r.name == identifier]
            if matching_runs:
                run = matching_runs[0]
                print(f"Found WandB run: {run.name}")

        if run:
            # Get config from wandb and reconstruct config_name
            config = run.config

            # Extract sweep_subfolder (default: "default")
            sweep_subfolder = config.get('sweep_subfolder', 'default')

            # Build config dict for get_config_foldername (same as train_simple.py)
            # Remove any keys that shouldn't be in the folder name (see train_simple.py lines 226-234)
            config_for_folder = {k: v for k, v in config.items() if k not in [
                'force_retrain', 'minibatch_size_per_device', 'results_folder', 'sweep_subfolder'
            ]}

            # Handle weak_model nested config if present
            if 'weak_model' in config_for_folder and isinstance(config_for_folder['weak_model'], dict):
                # Remove weak_model dict from config_for_folder
                del config_for_folder['weak_model']

            # Reconstruct config_name using get_config_foldername
            config_name = get_config_foldername(config_for_folder)
            print(f"Reconstructed config_name: {config_name}")

            # Try with sweep subfolder first (most common)
            checkpoint_path = Path(results_base_dir) / sweep_subfolder / config_name
            if checkpoint_path.exists() and (checkpoint_path / "config.json").exists():
                return str(checkpoint_path.absolute())

            # Try with "default" subfolder as fallback
            checkpoint_path = Path(results_base_dir) / "default" / config_name
            if checkpoint_path.exists() and (checkpoint_path / "config.json").exists():
                return str(checkpoint_path.absolute())

            # Try without subfolder
            checkpoint_path = Path(results_base_dir) / config_name
            if checkpoint_path.exists() and (checkpoint_path / "config.json").exists():
                return str(checkpoint_path.absolute())

            raise ValueError(
                f"Found WandB run '{run.name}' but couldn't find checkpoint directory with model.\n"
                f"Tried:\n  - {results_base_dir}/{sweep_subfolder}/{config_name}\n"
                f"  - {results_base_dir}/default/{config_name}\n  - {results_base_dir}/{config_name}\n"
                f"Make sure the checkpoint was saved locally with config.json"
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
        f"\nMake sure the checkpoint directory contains a valid transformers model (config.json)"
    )


def generate_model_predictions(
    checkpoint_path: str,
    model_name: str = "model",
    dataset_name: str = "boolq",
    n_test_docs: int = 10000,
    n_docs: int = 20000,
    batch_size: int = 32,
    max_ctx: int = 1024,
    use_best_checkpoint: bool = True,
    use_test_results: bool = True,
    seed: int = 0
) -> pd.DataFrame:
    """Generate fresh predictions from a fine-tuned model checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint directory containing pytorch_model.bin
        model_name: Name to use for this model in column names (e.g., "weak", "strong")
        dataset_name: Name of the dataset (e.g., "boolq")
        n_test_docs: Number of test documents to evaluate
        n_docs: Number of training documents (used for train2 split)
        batch_size: Batch size for inference
        max_ctx: Maximum context length
        use_best_checkpoint: If True, load from best_checkpoint/; if False, load from root
        use_test_results: If True, use test set; if False, use train2_ds (held-out train split)
        seed: Random seed for train/train2 split

    Returns:
        DataFrame with columns: idx, txt, {model_name}_soft_label, {model_name}_hard_label, ground_truth

    Raises:
        FileNotFoundError: If checkpoint directory or model file doesn't exist
        ValueError: If model loading fails
    """
    # Validate checkpoint path
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

    # Load config
    config_path = checkpoint_path / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
            dataset_name = config.get("ds_name", dataset_name)
            n_test_docs = config.get("n_test_docs", n_test_docs)
            n_docs = config.get("n_docs", n_docs)
            batch_size = config.get("batch_size", batch_size)
            max_ctx = config.get("max_ctx", max_ctx)
            seed = config.get("seed", seed)
            if "model_size" in config:
                model_size = config["model_size"]
            else:
                # Check parent directory for config (common when using checkpoint subdirs)
                parent_config_path = checkpoint_path.parent / "config.json"
                if parent_config_path.exists():
                    with open(parent_config_path, "r") as pf:
                        parent_config = json.load(pf)
                        model_size = parent_config.get("model_size", "gpt2")
                else:
                    model_size = "gpt2"

            dataset_type = "test" if use_test_results else "train2"
            print(f"Loaded config: dataset={dataset_name}, model={model_size}, split={dataset_type}, n_docs={n_docs}, n_test_docs={n_test_docs}")

    # Determine model checkpoint directory
    if use_best_checkpoint:
        model_checkpoint_dir = checkpoint_path / "best_checkpoint"
        if not model_checkpoint_dir.exists():
            print(f"Warning: best_checkpoint/ not found, using root checkpoint")
            model_checkpoint_dir = checkpoint_path
    else:
        model_checkpoint_dir = checkpoint_path

    print(f"Loading model from {model_checkpoint_dir}...")

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model architecture (using base model name)
    model = TransformerWithHead(model_size, num_labels=2)

    # Load fine-tuned weights from checkpoint
    # This handles both single-file and sharded checkpoints
    checkpoint_file = model_checkpoint_dir / "pytorch_model.bin"
    if checkpoint_file.exists():
        # Single file checkpoint
        print("Loading single-file checkpoint...")
        state_dict = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(state_dict)
    else:
        # Sharded checkpoint
        print("Loading sharded checkpoint...")
        from transformers.modeling_utils import load_sharded_checkpoint
        load_sharded_checkpoint(model, str(model_checkpoint_dir))

    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully")

    # Load dataset (either test or train2)
    if use_test_results:
        print(f"Loading test dataset: {dataset_name}...")
        eval_ds = load_dataset(
            dataset_name,
            split_sizes=dict(train=0, test=n_test_docs),
            seed=seed
        )["test"]
    else:
        print(f"Loading train2 dataset (held-out training split): {dataset_name}...")
        print(f"  Splitting {n_docs} training docs, using second half as train2...")
        full_train = load_dataset(
            dataset_name,
            split_sizes=dict(train=n_docs, test=0),
            seed=seed
        )["train"]

        # Split train in half to get train2_ds (same as train_simple.py)
        split_data = full_train.train_test_split(test_size=0.5, seed=seed)
        eval_ds = split_data["test"]  # train2_ds is the "test" part of the split
        print(f"  train2_ds: {len(eval_ds)} examples")

    # Get tokenizer and tokenize dataset
    print(f"Getting tokenizer for {model_size}...")
    tokenizer = get_tokenizer(model_size)

    print(f"Tokenizing {len(eval_ds)} examples...")
    eval_ds = tokenize_dataset(
        eval_ds,
        tokenizer=tokenizer,
        max_ctx=max_ctx
    )

    print(f"Running inference on {len(eval_ds)} examples...")

    # Run evaluation
    with torch.no_grad():
        predictions_ds = eval_model_acc(
            model=model,
            ds=eval_ds,
            eval_batch_size=batch_size,
            dataset_name=dataset_name
        )

    print(f"Generated {len(predictions_ds)} predictions")

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
    use_best_checkpoint: bool = True,
    validate_alignment: bool = True
) -> Dict[str, pd.DataFrame]:
    """Find where two fine-tuned models agree/disagree on predictions.

    Args:
        weak_checkpoint_path: Path to weak model checkpoint directory
        strong_checkpoint_path: Path to strong model checkpoint directory
        weak_name: Name for weak model (used in column names)
        strong_name: Name for strong model (used in column names)
        use_test_results: If True, use test_results; if False, use inference_results
        use_best_checkpoint: If True, load from best_checkpoint/; if False, load from root
        validate_alignment: If True, validate that both models used same test examples

    Returns:
        Dictionary with keys 'all', 'agree', 'disagree'

    Raises:
        ValueError: If datasets have different lengths or mismatched examples
    """
    dataset_type = "test set" if use_test_results else "inference set (train2)"
    print(f"\n=== Comparing {weak_name} vs {strong_name} on {dataset_type} ===\n")

    # Generate weak model predictions
    print(f"\n=== Generating {weak_name} model predictions ===")
    weak_df = generate_model_predictions(
        weak_checkpoint_path,
        model_name=weak_name,
        use_best_checkpoint=use_best_checkpoint,
        use_test_results=use_test_results
    )

    # Generate strong model predictions
    print(f"\n=== Generating {strong_name} model predictions ===")
    strong_df = generate_model_predictions(
        strong_checkpoint_path,
        model_name=strong_name,
        use_best_checkpoint=use_best_checkpoint,
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


def export_disagreement_rankings(
    results: Dict[str, pd.DataFrame],
    output_file: str
) -> None:
    """Export disagreement rankings in training-ready format.

    This function exports a CSV file with indices sorted by disagreement
    (confidence_diff descending), suitable for use with the disagreement-based
    mixing strategy in training.

    Args:
        results: Dictionary containing 'all' DataFrame with predictions and confidence_diff
        output_file: Path to output CSV file

    The output CSV contains:
        - idx: Example index
        - confidence_diff: Absolute difference in soft label probabilities
    """
    print("\n" + "="*80)
    print("EXPORTING DISAGREEMENT RANKINGS")
    print("="*80)

    # Get all predictions
    all_df = results['all']

    # Sort by confidence_diff (descending - highest disagreement first)
    sorted_df = all_df.sort_values('confidence_diff', ascending=False)

    # Extract only idx and confidence_diff
    rankings_df = sorted_df[['idx', 'confidence_diff']].copy()

    # Save to CSV
    rankings_df.to_csv(output_file, index=False)

    # Print summary statistics
    print(f"\nSaved disagreement rankings to: {output_file}")
    print(f"Total examples: {len(rankings_df)}")
    print(f"\nConfidence difference statistics:")
    print(f"  Min:    {rankings_df['confidence_diff'].min():.6f}")
    print(f"  Max:    {rankings_df['confidence_diff'].max():.6f}")
    print(f"  Mean:   {rankings_df['confidence_diff'].mean():.6f}")
    print(f"  Median: {rankings_df['confidence_diff'].median():.6f}")
    print(f"  Std:    {rankings_df['confidence_diff'].std():.6f}")

    # Show top 10 indices by disagreement
    print(f"\nTop 10 examples by disagreement:")
    print(rankings_df.head(10).to_string(index=False))

    print("="*80)


def main(
    weak_checkpoint_path: Optional[str] = None,
    strong_checkpoint_path: Optional[str] = None,
    weak_name: str = "weak",
    strong_name: str = "strong",
    use_test_results: bool = True,
    use_best_checkpoint: bool = True,
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
        use_best_checkpoint: Load from best_checkpoint/ (True) or root checkpoint (False)
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
            use_best_checkpoint=use_best_checkpoint,
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

        # Export disagreement rankings for use in disagreement-based training
        rankings_file = os.path.join(output_dir, "disagreement_rankings.csv")
        export_disagreement_rankings(results, rankings_file)

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
