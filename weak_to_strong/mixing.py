"""
Mixed supervision for weak-to-strong generalization.

This module provides functions to mix weak model predictions with ground truth labels,
enabling study of sample efficiency and optimal supervision budget allocation.
"""

import numpy as np
from datasets import Dataset
from typing import Optional
import random

from weak_to_strong.datasets import load_dataset


def mix_datasets_sample_level(
    weak_labeled_ds: Dataset,
    ground_truth_ds: Dataset,
    mix_ratio: float,
    seed: int = 0
) -> Dataset:
    """
    Sample-level mixing: Randomly select a fraction of examples to use ground truth labels.

    This strategy randomly selects mix_ratio fraction of examples to use GT labels,
    while the remaining examples use weak model predictions. Each example gets either
    100% weak or 100% GT labels (no interpolation).

    Args:
        weak_labeled_ds: Dataset with weak model predictions in 'soft_label' field
        ground_truth_ds: Dataset with ground truth in 'soft_label' field
        mix_ratio: Fraction of examples to use GT labels for (0.0 to 1.0)
                   e.g., 0.25 means 25% ground truth, 75% weak
        seed: Random seed for reproducibility

    Returns:
        Mixed dataset where mix_ratio% of examples have GT soft_labels,
        remainder have weak soft_labels. Adds 'label_source' field indicating
        'ground_truth' or 'weak' for each example.

    Example:
        >>> weak_ds = Dataset.from_dict({'soft_label': [[0.7, 0.3], [0.6, 0.4]]})
        >>> gt_ds = Dataset.from_dict({'soft_label': [[1.0, 0.0], [0.0, 1.0]]})
        >>> mixed = mix_datasets_sample_level(weak_ds, gt_ds, mix_ratio=0.5, seed=42)
        >>> # About 50% will have GT labels, 50% will have weak labels
    """
    if not (0.0 <= mix_ratio <= 1.0):
        raise ValueError(f"mix_ratio must be between 0.0 and 1.0, got {mix_ratio}")

    if len(weak_labeled_ds) != len(ground_truth_ds):
        raise ValueError(
            f"Datasets must have same length. "
            f"weak_labeled_ds: {len(weak_labeled_ds)}, ground_truth_ds: {len(ground_truth_ds)}"
        )

    n_examples = len(weak_labeled_ds)
    n_gt = int(n_examples * mix_ratio)

    # Set random seed for reproducibility
    rng = random.Random(seed)

    # Randomly select indices for GT labels
    indices = list(range(n_examples))
    rng.shuffle(indices)
    gt_indices = set(indices[:n_gt])

    # Create mixed dataset
    mixed_examples = []
    for i in range(n_examples):
        if i in gt_indices:
            # Use ground truth label
            example = dict(ground_truth_ds[i])
            example['soft_label'] = ground_truth_ds[i]['soft_label']
            example['label_source'] = 'ground_truth'
        else:
            # Use weak label
            example = dict(weak_labeled_ds[i])
            example['soft_label'] = weak_labeled_ds[i]['soft_label']
            example['label_source'] = 'weak'

        mixed_examples.append(example)

    return Dataset.from_list(mixed_examples)


def mix_datasets_label_level(
    weak_labeled_ds: Dataset,
    ground_truth_ds: Dataset,
    mix_ratio: float
) -> Dataset:
    """
    Label-level mixing: Interpolate between weak and ground truth labels for ALL examples.

    This strategy creates a smooth blend of weak and GT labels for every example:
    soft_label[i] = (1 - mix_ratio) * weak_label[i] + mix_ratio * gt_label[i]

    Args:
        weak_labeled_ds: Dataset with weak model predictions in 'soft_label' field
        ground_truth_ds: Dataset with ground truth in 'soft_label' field
        mix_ratio: Weight for GT labels in interpolation (0.0 to 1.0)
                   e.g., 0.25 means 75% weak + 25% GT for each example

    Returns:
        Mixed dataset where each soft_label is an interpolation of weak and GT.
        Adds 'mixing_method' field set to 'label_level'.

    Example:
        >>> weak_ds = Dataset.from_dict({'soft_label': [[0.7, 0.3]]})
        >>> gt_ds = Dataset.from_dict({'soft_label': [[1.0, 0.0]]})
        >>> mixed = mix_datasets_label_level(weak_ds, gt_ds, mix_ratio=0.5)
        >>> mixed[0]['soft_label']  # [0.85, 0.15] = 0.5*[0.7,0.3] + 0.5*[1.0,0.0]
    """
    if not (0.0 <= mix_ratio <= 1.0):
        raise ValueError(f"mix_ratio must be between 0.0 and 1.0, got {mix_ratio}")

    if len(weak_labeled_ds) != len(ground_truth_ds):
        raise ValueError(
            f"Datasets must have same length. "
            f"weak_labeled_ds: {len(weak_labeled_ds)}, ground_truth_ds: {len(ground_truth_ds)}"
        )

    # Create mixed dataset by interpolating labels
    mixed_examples = []
    for i in range(len(weak_labeled_ds)):
        example = dict(weak_labeled_ds[i])

        weak_label = np.array(weak_labeled_ds[i]['soft_label'])
        gt_label = np.array(ground_truth_ds[i]['soft_label'])

        # Interpolate: (1-α)*weak + α*gt
        mixed_label = (1 - mix_ratio) * weak_label + mix_ratio * gt_label

        # Ensure it's still a valid probability distribution (should be automatic)
        mixed_label = mixed_label / mixed_label.sum()  # Normalize just in case

        example['soft_label'] = mixed_label.tolist()
        example['mixing_method'] = 'label_level'

        mixed_examples.append(example)

    return Dataset.from_list(mixed_examples)


def mix_datasets_disagreement(
    weak_labeled_ds: Dataset,
    ground_truth_ds: Dataset,
    disagreement_indices: list,
    labeling_budget: int,
    seed: int = 0
) -> Dataset:
    """
    Disagreement-based mixing: Use ground truth for top-K examples by disagreement.

    This strategy selects examples based on pre-computed disagreement rankings
    between weak and strong models. The top labeling_budget examples (highest
    disagreement) receive ground truth labels, while the rest use weak labels.

    Args:
        weak_labeled_ds: Dataset with weak model predictions in 'soft_label' field
        ground_truth_ds: Dataset with ground truth in 'soft_label' field
        disagreement_indices: List of indices sorted by disagreement (descending)
                             e.g., [42, 17, 89, ...] where 42 has highest disagreement
        labeling_budget: Absolute number of examples to use GT labels for
        seed: Random seed for reproducibility (unused but kept for API consistency)

    Returns:
        Mixed dataset where labeling_budget examples (highest disagreement) have
        GT soft_labels, remainder have weak soft_labels. Adds 'label_source' field
        indicating 'ground_truth' or 'weak' for each example.

    Example:
        >>> weak_ds = Dataset.from_dict({'soft_label': [[0.7, 0.3], [0.6, 0.4]]})
        >>> gt_ds = Dataset.from_dict({'soft_label': [[1.0, 0.0], [0.0, 1.0]]})
        >>> rankings = [1, 0]  # Index 1 has highest disagreement, then index 0
        >>> mixed = mix_datasets_disagreement(weak_ds, gt_ds, rankings, labeling_budget=1)
        >>> # Index 1 will have GT label, index 0 will have weak label
    """
    if labeling_budget <= 0:
        raise ValueError(f"labeling_budget must be positive, got {labeling_budget}")

    if len(weak_labeled_ds) != len(ground_truth_ds):
        raise ValueError(
            f"Datasets must have same length. "
            f"weak_labeled_ds: {len(weak_labeled_ds)}, ground_truth_ds: {len(ground_truth_ds)}"
        )

    if labeling_budget > len(weak_labeled_ds):
        raise ValueError(
            f"labeling_budget ({labeling_budget}) exceeds dataset size ({len(weak_labeled_ds)}). "
            f"Maximum budget: {len(weak_labeled_ds)}"
        )

    n_examples = len(weak_labeled_ds)

    # Validate disagreement indices
    invalid_indices = [idx for idx in disagreement_indices if idx < 0 or idx >= n_examples]
    if invalid_indices:
        raise ValueError(
            f"Disagreement indices contain invalid values: {invalid_indices[:10]}...\n"
            f"Dataset has {n_examples} examples (valid indices: 0-{n_examples-1})"
        )

    # Select top labeling_budget indices for ground truth
    effective_budget = min(labeling_budget, len(disagreement_indices))
    if effective_budget < labeling_budget:
        print(
            f"Warning: labeling_budget ({labeling_budget}) exceeds available "
            f"disagreements ({len(disagreement_indices)}). "
            f"Using all {len(disagreement_indices)} disagreement examples for GT labels."
        )

    gt_indices = set(disagreement_indices[:effective_budget])

    # Create mixed dataset
    mixed_examples = []
    for i in range(n_examples):
        if i in gt_indices:
            # Use ground truth label
            example = dict(ground_truth_ds[i])
            example['soft_label'] = ground_truth_ds[i]['soft_label']
            example['label_source'] = 'ground_truth'
            # Store rank in disagreement (0 = highest disagreement)
            example['disagreement_rank'] = disagreement_indices.index(i)
        else:
            # Use weak label
            example = dict(weak_labeled_ds[i])
            example['soft_label'] = weak_labeled_ds[i]['soft_label']
            example['label_source'] = 'weak'
            example['disagreement_rank'] = -1  # Not in top-K

        mixed_examples.append(example)

    return Dataset.from_list(mixed_examples)


def load_disagreement_rankings(disagreement_file: str) -> list:
    """
    Load pre-computed disagreement rankings from CSV file.

    The CSV file should contain columns 'idx' and 'confidence_diff', where
    'idx' is the example index and 'confidence_diff' is the disagreement metric.
    Higher confidence_diff values indicate higher disagreement.

    Args:
        disagreement_file: Path to CSV file with disagreement data

    Returns:
        List of indices sorted by disagreement (descending order)
        e.g., [42, 17, 89, ...] where idx 42 has highest disagreement

    Raises:
        FileNotFoundError: If disagreement_file doesn't exist
        ValueError: If file is missing required columns or is malformed
    """
    import pandas as pd

    try:
        df = pd.read_csv(disagreement_file)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Disagreement file not found: {disagreement_file}\n"
            f"Run find_agreement.py first to generate disagreement rankings."
        )
    except Exception as e:
        raise ValueError(f"Error reading disagreement file: {e}")

    # Validate required columns
    required_cols = ['idx', 'confidence_diff']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Disagreement file missing required columns: {missing_cols}\n"
            f"Expected columns: {required_cols}\n"
            f"Found columns: {list(df.columns)}"
        )

    # Check for empty file
    if len(df) == 0:
        raise ValueError(f"Disagreement file is empty: {disagreement_file}")

    # Sort by confidence_diff (descending) and extract indices
    df_sorted = df.sort_values('confidence_diff', ascending=False)
    indices = df_sorted['idx'].tolist()

    return indices


def create_mixed_supervision_dataset(
    weak_labeled_ds: Dataset,
    ground_truth_ds: Dataset,
    mix_ratio: Optional[float] = None,
    mix_strategy: str = 'sample',
    seed: int = 0,
    # New parameters for disagreement strategy
    labeling_budget: Optional[int] = None,
    disagreement_indices: Optional[list] = None
) -> Dataset:
    """
    Main entry point for creating mixed supervision datasets.

    Combines weak model predictions with ground truth labels using the specified
    mixing strategy. Supports three strategies:

    - 'sample': Randomly select mix_ratio fraction of examples to use GT (sample-level)
    - 'label': Interpolate all labels as (1-α)*weak + α*GT (label-level)
    - 'disagreement': Use GT for top-K examples by disagreement (disagreement-based)

    Args:
        weak_labeled_ds: Dataset with weak model predictions (from eval_model_acc)
        ground_truth_ds: Original dataset with ground truth labels
        mix_ratio: Mixing ratio (0.0 = pure weak, 1.0 = pure GT)
                   Required for 'sample' and 'label' strategies, optional for 'disagreement'
        mix_strategy: 'sample', 'label', or 'disagreement'
        seed: Random seed for reproducibility (used in sample-level mixing)
        labeling_budget: Number of examples to use GT labels for (disagreement strategy only)
        disagreement_indices: List of indices sorted by disagreement (disagreement strategy only)

    Returns:
        Mixed dataset ready for training, with metadata fields added

    Raises:
        ValueError: If parameters invalid or datasets misaligned

    Example:
        >>> # Sample-level mixing
        >>> mixed_ds = create_mixed_supervision_dataset(
        ...     weak_labeled_ds=weak_ds,
        ...     ground_truth_ds=train2_ds,
        ...     mix_ratio=0.25,
        ...     mix_strategy='sample',
        ...     seed=42
        ... )
        >>> # Disagreement-based mixing
        >>> mixed_ds = create_mixed_supervision_dataset(
        ...     weak_labeled_ds=weak_ds,
        ...     ground_truth_ds=train2_ds,
        ...     mix_strategy='disagreement',
        ...     labeling_budget=500,
        ...     disagreement_indices=[42, 17, 89, ...]
        ... )
    """
    # Validate inputs based on strategy
    if mix_strategy == 'disagreement':
        # Disagreement strategy requires labeling_budget and disagreement_indices
        if labeling_budget is None:
            raise ValueError("labeling_budget required for disagreement strategy")
        if disagreement_indices is None:
            raise ValueError("disagreement_indices required for disagreement strategy")
        if mix_ratio is not None and mix_ratio != 1.0:
            # Warn that mix_ratio is ignored
            print(f"Note: mix_ratio={mix_ratio} ignored for disagreement strategy (using labeling_budget={labeling_budget})")
    else:
        # Sample and label strategies require mix_ratio
        if mix_ratio is None:
            raise ValueError(f"mix_ratio required for {mix_strategy} strategy")
        if not (0.0 <= mix_ratio <= 1.0):
            raise ValueError(f"mix_ratio must be between 0.0 and 1.0, got {mix_ratio}")

    if mix_strategy not in ['sample', 'label', 'disagreement']:
        raise ValueError(
            f"mix_strategy must be 'sample', 'label', or 'disagreement', got '{mix_strategy}'"
        )

    if len(weak_labeled_ds) != len(ground_truth_ds):
        raise ValueError(
            f"Datasets must have same length. "
            f"weak: {len(weak_labeled_ds)}, gt: {len(ground_truth_ds)}"
        )

    # Validate that datasets are aligned (check a few examples)
    n_check = min(10, len(weak_labeled_ds))
    for i in range(n_check):
        if 'txt' in weak_labeled_ds.column_names and 'txt' in ground_truth_ds.column_names:
            if weak_labeled_ds[i]['txt'] != ground_truth_ds[i]['txt']:
                raise ValueError(
                    f"Datasets appear misaligned at index {i}. "
                    f"Ensure weak_labeled_ds and ground_truth_ds correspond to the same examples."
                )

    # Dispatch to appropriate mixing function
    if mix_strategy == 'disagreement':
        mixed_ds = mix_datasets_disagreement(
            weak_labeled_ds, ground_truth_ds, disagreement_indices, labeling_budget, seed
        )
    elif mix_strategy == 'sample':
        mixed_ds = mix_datasets_sample_level(
            weak_labeled_ds, ground_truth_ds, mix_ratio, seed
        )
    else:  # mix_strategy == 'label'
        mixed_ds = mix_datasets_label_level(
            weak_labeled_ds, ground_truth_ds, mix_ratio
        )

    # Add metadata (store as dataset info/description)
    # Note: HuggingFace datasets don't have a standard metadata field,
    # but we can add to the info object
    if mix_strategy == 'disagreement':
        info_text = (
            f"Mixed supervision dataset (disagreement-based): "
            f"{labeling_budget} examples with ground truth labels "
            f"(selected by disagreement), {len(weak_labeled_ds) - labeling_budget} with weak labels"
        )
    else:
        info_text = (
            f"Mixed supervision dataset: "
            f"{mix_ratio*100:.1f}% ground truth, {(1-mix_ratio)*100:.1f}% weak labels "
            f"(strategy: {mix_strategy})"
        )

    # We can't easily modify dataset.info after creation, but we've added
    # 'label_source' or 'mixing_method' fields which serve as metadata

    return mixed_ds


def validate_mixing(
    mixed_ds: Dataset,
    expected_mix_ratio: float,
    tolerance: float = 0.01
) -> dict:
    """
    Validate that mixing was performed correctly.

    Args:
        mixed_ds: Mixed dataset to validate
        expected_mix_ratio: Expected fraction of ground truth examples
        tolerance: Allowed deviation from expected ratio (for sample-level)

    Returns:
        Dictionary with validation statistics:
        - 'valid': True if validation passed
        - 'actual_gt_fraction': Measured fraction (for sample-level mixing)
        - 'avg_label_entropy': Average entropy of soft labels
        - 'checks': Dict of individual check results
    """
    results = {
        'valid': True,
        'checks': {},
        'avg_label_entropy': None,
        'actual_gt_fraction': None
    }

    # Check for sample-level mixing
    if 'label_source' in mixed_ds.column_names:
        gt_count = sum(1 for x in mixed_ds if x['label_source'] == 'ground_truth')
        actual_ratio = gt_count / len(mixed_ds)
        results['actual_gt_fraction'] = actual_ratio

        # Check if within tolerance
        ratio_check = abs(actual_ratio - expected_mix_ratio) <= tolerance
        results['checks']['mix_ratio_correct'] = ratio_check
        results['valid'] &= ratio_check

    # Check for label-level mixing
    if 'mixing_method' in mixed_ds.column_names:
        results['checks']['has_mixing_method'] = True

    # Calculate average label entropy (measures uncertainty in labels)
    entropies = []
    for example in mixed_ds:
        soft_label = np.array(example['soft_label'])
        # Shannon entropy: -sum(p * log(p))
        entropy = -np.sum(soft_label * np.log(soft_label + 1e-10))
        entropies.append(entropy)

    results['avg_label_entropy'] = float(np.mean(entropies))

    return results


def apply_mixed_supervision(
    weak_labeled_ds: Dataset,
    ds_name: str,
    weak_model_config: dict,
    n_test_docs: int,
    mix_ratio: Optional[float] = None,
    mix_strategy: str = 'sample',
    seed: int = 0,
    labeling_budget: Optional[int] = None,
    disagreement_file: Optional[str] = None,
):
    """
    Apply mixed supervision by combining weak labels with ground truth labels.

    This function implements three mixing strategies:
    1. Sample-level mixing: Randomly select mix_ratio fraction of examples to use
       ground truth labels, rest use weak labels
    2. Label-level mixing: Interpolate between weak and ground truth labels for
       every example: soft_label = (1-α)*weak + α*gt
    3. Disagreement-based mixing: Use ground truth for top-K examples by disagreement
       between weak and strong models

    Args:
        weak_labeled_ds: Dataset with weak labels (soft_label field)
        ds_name: Name of the dataset to load ground truth from
        weak_model_config: Config dict used to train the weak model (contains seed, n_docs)
        n_test_docs: Number of test documents
        mix_ratio: Fraction of ground truth labels to mix in (0.0 to 1.0)
                   Required for 'sample' and 'label' strategies, optional for 'disagreement'
        mix_strategy: 'sample', 'label', or 'disagreement'
        seed: Random seed for reproducibility
        labeling_budget: Number of examples to use GT labels for (disagreement strategy only)
        disagreement_file: Path to CSV file with disagreement rankings (disagreement strategy only)

    Returns:
        Tuple of (mixed_dataset, mixing_stats_dict)
        - mixed_dataset: Dataset with mixed supervision
        - mixing_stats_dict: Statistics about the mixing (for logging)
    """
    # Early return if no mixing requested (only for non-disagreement strategies)
    if mix_strategy != 'disagreement' and mix_ratio is not None and mix_ratio <= 0.0:
        return weak_labeled_ds, {}

    print(f"\n{'='*60}")
    print(f"MIXED SUPERVISION")
    print(f"{'='*60}")
    print(f"Strategy: {mix_strategy}")
    if mix_strategy == 'disagreement':
        print(f"Labeling budget: {labeling_budget} examples")
        print(f"Disagreement file: {disagreement_file}")
    else:
        print(f"Ground truth: {mix_ratio*100:.1f}%")
        print(f"Weak labels: {(1-mix_ratio)*100:.1f}%")
    print(f"{'='*60}\n")

    # Load disagreement rankings if using disagreement strategy
    disagreement_indices = None
    if mix_strategy == 'disagreement':
        print("Loading disagreement rankings...")
        disagreement_indices = load_disagreement_rankings(disagreement_file)
        print(f"Loaded {len(disagreement_indices)} disagreement rankings")
        print(f"Will use ground truth for top {labeling_budget} examples\n")

    # Need to reload the original ground truth dataset and split it the same way
    # to get train2_ds with ground truth labels
    print("Loading original dataset for ground truth labels...")

    original_dataset = load_dataset(
        ds_name,
        seed=weak_model_config.get('seed', seed),
        split_sizes=dict(
            train=weak_model_config.get('n_docs'),
            test=n_test_docs
        )
    )
    # Split the same way the weak labels were generated
    original_split = original_dataset['train'].train_test_split(
        test_size=0.5,
        seed=weak_model_config.get('seed', seed)
    )
    train2_ds_gt = original_split['test']  # Ground truth version of train2

    # Apply mixing using the mixing module
    mixed_ds = create_mixed_supervision_dataset(
        weak_labeled_ds=weak_labeled_ds,
        ground_truth_ds=train2_ds_gt,
        mix_ratio=mix_ratio,
        mix_strategy=mix_strategy,
        seed=seed,
        labeling_budget=labeling_budget,
        disagreement_indices=disagreement_indices
    )

    # Compute and print mixing statistics
    mixing_stats = {}
    if mix_strategy == 'sample' and 'label_source' in mixed_ds.column_names:
        gt_count = sum(1 for x in mixed_ds if x['label_source'] == 'ground_truth')
        actual_gt_fraction = gt_count / len(mixed_ds)
        print(f"Sample-level mixing: {gt_count}/{len(mixed_ds)} examples use ground truth "
              f"({actual_gt_fraction*100:.1f}%)\n")

        mixing_stats = {
            'mixing/gt_examples': gt_count,
            'mixing/weak_examples': len(mixed_ds) - gt_count,
            'mixing/actual_gt_fraction': actual_gt_fraction,
            'mixing/requested_gt_fraction': mix_ratio,
        }
    elif mix_strategy == 'label':
        # For label-level mixing, compute average label entropy
        entropies = []
        for example in mixed_ds:
            probs = np.array(example['soft_label'])
            # Avoid log(0) by adding small epsilon
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            entropies.append(entropy)
        avg_entropy = np.mean(entropies)
        print(f"Label-level mixing: Average label entropy = {avg_entropy:.3f}\n")

        mixing_stats = {
            'mixing/avg_label_entropy': avg_entropy,
            'mixing/min_label_entropy': np.min(entropies),
            'mixing/max_label_entropy': np.max(entropies),
        }
    elif mix_strategy == 'disagreement' and 'label_source' in mixed_ds.column_names:
        # For disagreement-based mixing, compute statistics
        gt_count = sum(1 for x in mixed_ds if x['label_source'] == 'ground_truth')
        weak_count = len(mixed_ds) - gt_count

        # Compute average disagreement rank of selected examples
        if 'disagreement_rank' in mixed_ds.column_names:
            gt_ranks = [x['disagreement_rank'] for x in mixed_ds if x['label_source'] == 'ground_truth']
            avg_rank = np.mean(gt_ranks) if gt_ranks else 0
            min_rank = np.min(gt_ranks) if gt_ranks else 0
            max_rank = np.max(gt_ranks) if gt_ranks else 0
        else:
            avg_rank = min_rank = max_rank = 0

        print(f"Disagreement-based mixing:")
        print(f"  Ground truth examples: {gt_count}/{len(mixed_ds)} ({gt_count/len(mixed_ds)*100:.1f}%)")
        print(f"  Weak label examples: {weak_count}/{len(mixed_ds)} ({weak_count/len(mixed_ds)*100:.1f}%)")
        print(f"  Average disagreement rank of GT examples: {avg_rank:.1f}")
        print(f"  Disagreement rank range: [{min_rank}, {max_rank}]\n")

        mixing_stats = {
            'mixing/gt_examples': gt_count,
            'mixing/weak_examples': weak_count,
            'mixing/labeling_budget': labeling_budget,
            'mixing/avg_disagreement_rank': avg_rank,
            'mixing/min_disagreement_rank': min_rank,
            'mixing/max_disagreement_rank': max_rank,
            'mixing/actual_gt_fraction': gt_count / len(mixed_ds),
        }

    return mixed_ds, mixing_stats
