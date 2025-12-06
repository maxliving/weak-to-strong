"""
Mixed supervision for weak-to-strong generalization.

This module provides functions to mix weak model predictions with ground truth labels,
enabling study of sample efficiency and optimal supervision budget allocation.
"""

import numpy as np
from datasets import Dataset
from typing import Optional
import random


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


def create_mixed_supervision_dataset(
    weak_labeled_ds: Dataset,
    ground_truth_ds: Dataset,
    mix_ratio: float,
    mix_strategy: str = 'sample',
    seed: int = 0
) -> Dataset:
    """
    Main entry point for creating mixed supervision datasets.

    Combines weak model predictions with ground truth labels using the specified
    mixing strategy. Supports two strategies:

    - 'sample': Randomly select mix_ratio fraction of examples to use GT (sample-level)
    - 'label': Interpolate all labels as (1-α)*weak + α*GT (label-level)

    Args:
        weak_labeled_ds: Dataset with weak model predictions (from eval_model_acc)
        ground_truth_ds: Original dataset with ground truth labels
        mix_ratio: Mixing ratio (0.0 = pure weak, 1.0 = pure GT)
        mix_strategy: 'sample' for sample-level or 'label' for label-level mixing
        seed: Random seed for reproducibility (used in sample-level mixing)

    Returns:
        Mixed dataset ready for training, with metadata fields added

    Raises:
        ValueError: If mix_ratio not in [0, 1], strategy invalid, or datasets misaligned

    Example:
        >>> # After training weak model and generating predictions
        >>> mixed_ds = create_mixed_supervision_dataset(
        ...     weak_labeled_ds=weak_ds,
        ...     ground_truth_ds=train2_ds,
        ...     mix_ratio=0.25,
        ...     mix_strategy='sample',
        ...     seed=42
        ... )
        >>> # Train strong model on mixed_ds
    """
    # Validate inputs
    if not (0.0 <= mix_ratio <= 1.0):
        raise ValueError(f"mix_ratio must be between 0.0 and 1.0, got {mix_ratio}")

    if mix_strategy not in ['sample', 'label']:
        raise ValueError(
            f"mix_strategy must be 'sample' or 'label', got '{mix_strategy}'"
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
    if mix_strategy == 'sample':
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
