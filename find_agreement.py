#!/usr/bin/env python3
"""
Find agreement/disagreement between a fine-tuned weak model and base strong model.

This script:
1. Loads predictions from a fine-tuned weak model (from weak_labels directory)
2. Runs inference on a base strong model to get its predictions
3. Identifies where they agree or disagree
"""

import os
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from datasets import load_from_disk

from weak_to_strong.common import get_tokenizer
from weak_to_strong.datasets import tokenize_dataset
from weak_to_strong.eval import eval_model_acc
from weak_to_strong.model import TransformerWithHead
from weak_to_strong.train import ModelConfig


def load_weak_model_predictions(labels_path: str, model_name: str = "weak") -> pd.DataFrame:
    """Load fine-tuned weak model predictions from weak_labels directory.

    Args:
        labels_path: Path to the weak_labels directory
        model_name: Name to use for this model in column names

    Returns:
        DataFrame with columns: idx, txt, {model_name}_soft_label, {model_name}_hard_label, ground_truth
    """
    ds = load_from_disk(labels_path)

    # Convert to DataFrame
    records = []
    for i, example in enumerate(ds):
        records.append({
            'idx': i,
            'txt': example.get('txt', ''),
            f'{model_name}_soft_label': example.get('soft_label'),
            f'{model_name}_hard_label': example.get('hard_label'),
            'ground_truth': example.get('gt_label'),  # Ground truth label
        })

    return pd.DataFrame(records)


def generate_base_model_predictions(
    weak_labels_ds,
    model_size: str,
    model_name: str = "strong",
    batch_size: int = 32,
    max_ctx: int = 1024
) -> pd.DataFrame:
    """Generate predictions from a base (untrained) model on the same examples as weak_labels.

    Args:
        weak_labels_ds: The weak_labels dataset (loaded from disk)
        model_size: Model size (e.g., "gpt2-xl", "gpt2-large")
        model_name: Name to use for this model in column names
        batch_size: Batch size for inference
        max_ctx: Maximum context length

    Returns:
        DataFrame with columns: idx, txt, {model_name}_soft_label, {model_name}_hard_label
    """
    print(f"Loading base model: {model_size}")

    # Get tokenizer
    tokenizer = get_tokenizer(model_size)

    # Tokenize the dataset if not already tokenized
    if 'input_ids' not in weak_labels_ds.column_names:
        print("Tokenizing dataset...")
        weak_labels_ds = tokenize_dataset(weak_labels_ds, tokenizer, max_ctx)

    # Determine number of classes from the weak_labels dataset
    n_labels = len(weak_labels_ds[0]['soft_label']) if 'soft_label' in weak_labels_ds[0] else 2

    # Create model config
    model_config = ModelConfig(
        name=model_size,
        default_lr=1e-5,
        eval_batch_size=batch_size,
    )

    # Load base model (no fine-tuning)
    print(f"Initializing {model_size} model...")
    model = TransformerWithHead(
        model_config.name,
        n_labels=n_labels,
        **model_config.custom_kwargs if model_config.custom_kwargs else {}
    )

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"Running inference on {len(weak_labels_ds)} examples...")

    # Get predictions using eval_model_acc
    results_ds = eval_model_acc(model, weak_labels_ds, batch_size, dataset_name=f"{model_name} (base)")

    # Convert to DataFrame
    records = []
    for i, result in enumerate(results_ds):
        records.append({
            'idx': i,
            'txt': result['txt'],
            f'{model_name}_soft_label': result['soft_label'][1] if n_labels == 2 else max(result['soft_label']),
            f'{model_name}_hard_label': result['hard_label'],
        })

    return pd.DataFrame(records)


def find_agreement_disagreement(
    weak_labels_path: str,
    strong_model_size: str,
    weak_name: str = "weak",
    strong_name: str = "strong",
    batch_size: int = 32,
    max_ctx: int = 1024
) -> Dict[str, pd.DataFrame]:
    """Find where fine-tuned weak model and base strong model agree/disagree.

    Args:
        weak_labels_path: Path to weak model's weak_labels directory
        strong_model_size: Strong model size (e.g., "gpt2-xl")
        weak_name: Name for weak model
        strong_name: Name for strong model
        batch_size: Batch size for strong model inference
        max_ctx: Max context length

    Returns:
        Dictionary with keys 'all', 'agree', 'disagree'
    """
    # Load weak_labels dataset from disk
    print("Loading weak_labels dataset...")
    weak_labels_ds = load_from_disk(weak_labels_path)
    print(f"Loaded {len(weak_labels_ds)} examples")

    # Convert weak labels to DataFrame
    print("Extracting weak model predictions...")
    weak_df = load_weak_model_predictions(weak_labels_path, weak_name)

    # Generate base strong model predictions on the SAME examples
    print(f"\nGenerating base {strong_model_size} predictions on the same examples...")
    strong_df = generate_base_model_predictions(
        weak_labels_ds=weak_labels_ds,
        model_size=strong_model_size,
        model_name=strong_name,
        batch_size=batch_size,
        max_ctx=max_ctx
    )

    # Merge on idx (no need to merge on txt since they're guaranteed to be the same)
    merged_df = pd.merge(weak_df, strong_df, on='idx', how='inner', suffixes=('_weak', '_strong'))

    # Keep only one txt column
    if 'txt_weak' in merged_df.columns:
        merged_df['txt'] = merged_df['txt_weak']
        merged_df = merged_df.drop(columns=['txt_weak', 'txt_strong'])

    if len(merged_df) == 0:
        raise ValueError("No matching examples found. This should not happen!")

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


def main():
    # ========================================================================
    # CONFIGURATION
    # ========================================================================

    # Path to fine-tuned weak model's predictions
    # Example format: ./results/default/{config}/weak_labels
    # where {config} is like: bs=32-dn=boolq-...-ms=gpt2-...-mxr=1.0-...
    WEAK_LABELS_PATH = None  # SET THIS to weak model's weak_labels path

    # Model names (for display purposes)
    WEAK_NAME = "gpt2"
    STRONG_NAME = "gpt2-xl"

    # Inference settings
    BATCH_SIZE = 32
    MAX_CTX = 1024

    # Output options
    OUTPUT_DIR = os.path.join("agreement_analysis", f"{WEAK_NAME}_vs_{STRONG_NAME}")
    SHOW_EXAMPLES = True  # Print example texts
    N_EXAMPLES = 5  # Number of examples to show

    # ========================================================================
    # VALIDATION
    # ========================================================================

    if WEAK_LABELS_PATH is None:
        print("ERROR: WEAK_LABELS_PATH is not set!")
        print()
        print("Please set WEAK_LABELS_PATH to point to a weak_labels directory.")
        print("Example:")
        print("  WEAK_LABELS_PATH = './results/default/bs=32-dn=boolq-...-ms=gpt2-.../weak_labels'")
        print()
        print("You can find weak_labels directories by running:")
        print("  find ./results -name 'weak_labels' -type d")
        return

    # ========================================================================
    # ANALYSIS
    # ========================================================================

    print("="*80)
    print("Fine-tuned Weak Model vs Base Strong Model Agreement Analysis")
    print("="*80)
    print(f"Fine-tuned weak model labels: {WEAK_LABELS_PATH}")
    print(f"Base strong model: {STRONG_NAME}")
    print("="*80)
    print()

    # Find agreement/disagreement
    results = find_agreement_disagreement(
        weak_labels_path=WEAK_LABELS_PATH,
        strong_model_size=STRONG_NAME,
        weak_name=WEAK_NAME,
        strong_name=STRONG_NAME,
        batch_size=BATCH_SIZE,
        max_ctx=MAX_CTX
    )

    # Print summary
    print_summary(results, WEAK_NAME, STRONG_NAME)

    # Show examples
    if SHOW_EXAMPLES:
        print_examples(results['agree'], n=N_EXAMPLES, title="Examples where models AGREE",
                      model1_name=WEAK_NAME, model2_name=STRONG_NAME)
        print_examples(results['disagree'], n=N_EXAMPLES, title="Examples where models DISAGREE",
                      model1_name=WEAK_NAME, model2_name=STRONG_NAME)

    # Export to CSV
    print()
    print("="*80)
    print("Exporting to CSV...")
    print("="*80)

    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    # Save each category
    for category, df in results.items():
        if category == 'all':
            continue
        output_path = f"{OUTPUT_DIR}/{category}.csv"
        df.to_csv(output_path, index=False)
        print(f"  {category}: {output_path} ({len(df)} examples)")

    # Save full dataset with agreement labels
    full_output = f"{OUTPUT_DIR}/all_with_labels.csv"
    results['all'].to_csv(full_output, index=False)
    print(f"  all: {full_output} ({len(results['all'])} examples)")

    print("="*80)

    print("\nDone!")


if __name__ == '__main__':
    main()
