"""
Unit tests for mixed supervision functionality.
"""

import pytest
import numpy as np
from datasets import Dataset
from unittest.mock import Mock, patch, MagicMock

from weak_to_strong.mixing import (
    mix_datasets_sample_level,
    mix_datasets_label_level,
    create_mixed_supervision_dataset,
    validate_mixing,
    apply_mixed_supervision
)


def create_dummy_datasets(n=100, seed=42):
    """Create dummy weak and ground truth datasets for testing."""
    rng = np.random.RandomState(seed)

    # Weak dataset with soft probabilities
    weak_examples = []
    for i in range(n):
        prob = rng.rand()  # Random probability between 0 and 1
        weak_examples.append({
            'txt': f'example_{i}',
            'soft_label': [1 - prob, prob],
            'hard_label': int(prob > 0.5)
        })

    # Ground truth dataset with one-hot labels
    gt_examples = []
    for i in range(n):
        label = rng.randint(0, 2)
        gt_examples.append({
            'txt': f'example_{i}',
            'soft_label': [1.0, 0.0] if label == 0 else [0.0, 1.0],
            'hard_label': label
        })

    weak_ds = Dataset.from_list(weak_examples)
    gt_ds = Dataset.from_list(gt_examples)

    return weak_ds, gt_ds


class TestSampleLevelMixing:
    """Tests for sample-level mixing strategy."""

    def test_mix_ratio_zero(self):
        """Test that mix_ratio=0 returns all weak labels."""
        weak_ds, gt_ds = create_dummy_datasets(n=100)

        mixed_ds = mix_datasets_sample_level(weak_ds, gt_ds, mix_ratio=0.0, seed=42)

        # All examples should have weak labels
        weak_count = sum(1 for x in mixed_ds if x['label_source'] == 'weak')
        assert weak_count == 100, "With mix_ratio=0, all labels should be weak"

    def test_mix_ratio_one(self):
        """Test that mix_ratio=1 returns all ground truth labels."""
        weak_ds, gt_ds = create_dummy_datasets(n=100)

        mixed_ds = mix_datasets_sample_level(weak_ds, gt_ds, mix_ratio=1.0, seed=42)

        # All examples should have GT labels
        gt_count = sum(1 for x in mixed_ds if x['label_source'] == 'ground_truth')
        assert gt_count == 100, "With mix_ratio=1, all labels should be ground truth"

    def test_mix_ratio_proportions(self):
        """Test that mixing proportions are approximately correct."""
        weak_ds, gt_ds = create_dummy_datasets(n=1000)

        for mix_ratio in [0.25, 0.5, 0.75]:
            mixed_ds = mix_datasets_sample_level(weak_ds, gt_ds, mix_ratio=mix_ratio, seed=42)

            gt_count = sum(1 for x in mixed_ds if x['label_source'] == 'ground_truth')
            actual_ratio = gt_count / len(mixed_ds)

            # Allow 5% tolerance
            assert abs(actual_ratio - mix_ratio) < 0.05, \
                f"Expected {mix_ratio:.2f}, got {actual_ratio:.2f}"

    def test_dataset_length_preserved(self):
        """Test that mixed dataset has same length as input datasets."""
        weak_ds, gt_ds = create_dummy_datasets(n=100)

        mixed_ds = mix_datasets_sample_level(weak_ds, gt_ds, mix_ratio=0.5, seed=42)

        assert len(mixed_ds) == 100, "Mixed dataset should have same length as inputs"

    def test_mixed_labels_match_source(self):
        """
        Test that when label_source is 'ground_truth', the label matches the GT dataset,
        and when it is 'weak', it matches the weak dataset.
        """
        weak_ds, gt_ds = create_dummy_datasets(n=100)
        
        # Create mixed dataset
        mixed_ds = mix_datasets_sample_level(weak_ds, gt_ds, mix_ratio=0.5, seed=42)
        
        for i, example in enumerate(mixed_ds):
            source = example['label_source']
            
            if source == 'ground_truth':
                # Should match GT label exactly
                np.testing.assert_array_almost_equal(
                    example['soft_label'], 
                    gt_ds[i]['soft_label'],
                    err_msg=f"Example {i} marked as Ground Truth but label mismatch"
                )
                # Should NOT match weak label (unless by random chance they are identical)
                # In our dummy dataset, probabilities are random floats so exact match is unlikely
                if not np.allclose(weak_ds[i]['soft_label'], gt_ds[i]['soft_label']):
                     assert not np.allclose(example['soft_label'], weak_ds[i]['soft_label'])
                         
            elif source == 'weak':
                # Should match weak label exactly
                np.testing.assert_array_almost_equal(
                    example['soft_label'], 
                    weak_ds[i]['soft_label'],
                    err_msg=f"Example {i} marked as Weak but label mismatch"
                )

    def test_label_source_field_added(self):
        """Test that label_source field is added to mixed dataset."""
        weak_ds, gt_ds = create_dummy_datasets(n=100)

        mixed_ds = mix_datasets_sample_level(weak_ds, gt_ds, mix_ratio=0.5, seed=42)

        assert 'label_source' in mixed_ds.column_names, \
            "Mixed dataset should have label_source field"

        # Check all examples have either 'weak' or 'ground_truth'
        for example in mixed_ds:
            assert example['label_source'] in ['weak', 'ground_truth']


class TestLabelLevelMixing:
    """Tests for label-level mixing strategy."""

    def test_interpolation_correctness(self):
        """Test that label interpolation is computed correctly."""
        # Create simple dataset with known values
        weak_ds = Dataset.from_dict({
            'txt': ['example_0'],
            'soft_label': [[0.7, 0.3]]
        })
        gt_ds = Dataset.from_dict({
            'txt': ['example_0'],
            'soft_label': [[1.0, 0.0]]
        })

        mixed_ds = mix_datasets_label_level(weak_ds, gt_ds, mix_ratio=0.5)

        # Expected: 0.5 * [0.7, 0.3] + 0.5 * [1.0, 0.0] = [0.85, 0.15]
        expected = [0.85, 0.15]
        actual = mixed_ds[0]['soft_label']

        np.testing.assert_array_almost_equal(actual, expected, decimal=5)

    def test_mix_ratio_zero_label_level(self):
        """Test that mix_ratio=0 returns weak labels (label-level)."""
        weak_ds, gt_ds = create_dummy_datasets(n=10)

        mixed_ds = mix_datasets_label_level(weak_ds, gt_ds, mix_ratio=0.0)

        # Should match weak labels exactly
        for i in range(len(mixed_ds)):
            np.testing.assert_array_almost_equal(
                mixed_ds[i]['soft_label'],
                weak_ds[i]['soft_label']
            )

    def test_mix_ratio_one_label_level(self):
        """Test that mix_ratio=1 returns ground truth labels (label-level)."""
        weak_ds, gt_ds = create_dummy_datasets(n=10)

        mixed_ds = mix_datasets_label_level(weak_ds, gt_ds, mix_ratio=1.0)

        # Should match GT labels exactly
        for i in range(len(mixed_ds)):
            np.testing.assert_array_almost_equal(
                mixed_ds[i]['soft_label'],
                gt_ds[i]['soft_label']
            )

    def test_probability_distribution_valid(self):
        """Test that mixed labels remain valid probability distributions."""
        weak_ds, gt_ds = create_dummy_datasets(n=100)

        mixed_ds = mix_datasets_label_level(weak_ds, gt_ds, mix_ratio=0.5)

        for example in mixed_ds:
            soft_label = example['soft_label']
            # Should sum to 1
            assert abs(sum(soft_label) - 1.0) < 1e-6, "Soft labels should sum to 1"
            # Should be non-negative
            assert all(x >= 0 for x in soft_label), "Soft labels should be non-negative"

    def test_mixing_method_field_added(self):
        """Test that mixing_method field is added."""
        weak_ds, gt_ds = create_dummy_datasets(n=10)

        mixed_ds = mix_datasets_label_level(weak_ds, gt_ds, mix_ratio=0.5)

        assert 'mixing_method' in mixed_ds.column_names
        assert all(x['mixing_method'] == 'label_level' for x in mixed_ds)


class TestCreateMixedSupervisionDataset:
    """Tests for the main entry point function."""

    def test_invalid_strategy_raises_error(self):
        """Test that invalid strategy raises ValueError."""
        weak_ds, gt_ds = create_dummy_datasets(n=10)

        with pytest.raises(ValueError, match="mix_strategy must be"):
            create_mixed_supervision_dataset(
                weak_ds, gt_ds, mix_ratio=0.5, mix_strategy='invalid'
            )

    def test_invalid_mix_ratio_raises_error(self):
        """Test that invalid mix_ratio raises ValueError."""
        weak_ds, gt_ds = create_dummy_datasets(n=10)

        with pytest.raises(ValueError, match="mix_ratio must be"):
            create_mixed_supervision_dataset(
                weak_ds, gt_ds, mix_ratio=1.5, mix_strategy='sample'
            )

    def test_mismatched_dataset_lengths_raises_error(self):
        """Test that mismatched dataset lengths raise ValueError."""
        weak_ds, _ = create_dummy_datasets(n=100)
        _, gt_ds = create_dummy_datasets(n=50)

        with pytest.raises(ValueError, match="same length"):
            create_mixed_supervision_dataset(
                weak_ds, gt_ds, mix_ratio=0.5, mix_strategy='sample'
            )

    def test_strategy_dispatch_sample(self):
        """Test that 'sample' strategy calls sample-level mixing."""
        weak_ds, gt_ds = create_dummy_datasets(n=100)

        mixed_ds = create_mixed_supervision_dataset(
            weak_ds, gt_ds, mix_ratio=0.5, mix_strategy='sample', seed=42
        )

        assert 'label_source' in mixed_ds.column_names, \
            "Sample strategy should add label_source field"

    def test_strategy_dispatch_label(self):
        """Test that 'label' strategy calls label-level mixing."""
        weak_ds, gt_ds = create_dummy_datasets(n=100)

        mixed_ds = create_mixed_supervision_dataset(
            weak_ds, gt_ds, mix_ratio=0.5, mix_strategy='label'
        )

        assert 'mixing_method' in mixed_ds.column_names, \
            "Label strategy should add mixing_method field"


class TestValidateMixing:
    """Tests for the validation function."""

    def test_validate_sample_level_mixing(self):
        """Test validation of sample-level mixing."""
        weak_ds, gt_ds = create_dummy_datasets(n=1000)

        mixed_ds = mix_datasets_sample_level(weak_ds, gt_ds, mix_ratio=0.25, seed=42)

        results = validate_mixing(mixed_ds, expected_mix_ratio=0.25, tolerance=0.05)

        assert results['valid'], "Validation should pass for correct mixing"
        assert 'actual_gt_fraction' in results
        assert abs(results['actual_gt_fraction'] - 0.25) < 0.05

    def test_validate_label_level_mixing(self):
        """Test validation of label-level mixing."""
        weak_ds, gt_ds = create_dummy_datasets(n=100)

        mixed_ds = mix_datasets_label_level(weak_ds, gt_ds, mix_ratio=0.5)

        results = validate_mixing(mixed_ds, expected_mix_ratio=0.5)

        assert 'avg_label_entropy' in results
        assert results['avg_label_entropy'] is not None


class TestApplyMixedSupervision:
    """Tests for the apply_mixed_supervision function from train_simple.py."""

    def test_mix_ratio_zero_returns_unchanged(self):
        """Test that mix_ratio=0 returns weak labels unchanged with empty stats."""
        weak_ds, _ = create_dummy_datasets(n=100)
        weak_model_config = {'seed': 42, 'n_docs': 200}

        result_ds, stats = apply_mixed_supervision(
            weak_labeled_ds=weak_ds,
            ds_name='sciq',
            weak_model_config=weak_model_config,
            n_test_docs=100,
            mix_ratio=0.0,
            mix_strategy='sample',
            seed=42
        )

        # Should return the same dataset
        assert len(result_ds) == len(weak_ds)
        # Stats should be empty
        assert stats == {}

    @patch('weak_to_strong.mixing.load_dataset')
    def test_sample_level_mixing_stats(self, mock_load_dataset):
        """Test that sample-level mixing computes correct statistics."""
        weak_ds, gt_ds = create_dummy_datasets(n=100)

        # Mock the dataset loading to return our test data
        mock_original_dataset = {'train': Dataset.from_dict({
            'txt': [f'example_{i}' for i in range(200)],
            'soft_label': [[0.5, 0.5] for _ in range(200)],
            'hard_label': [0] * 200
        })}
        mock_load_dataset.return_value = mock_original_dataset

        weak_model_config = {'seed': 42, 'n_docs': 200}

        with patch('weak_to_strong.mixing.create_mixed_supervision_dataset') as mock_create:
            # Create a mock mixed dataset with label_source field
            mock_mixed = weak_ds.add_column('label_source',
                ['ground_truth'] * 25 + ['weak'] * 75)
            mock_create.return_value = mock_mixed

            result_ds, stats = apply_mixed_supervision(
                weak_labeled_ds=weak_ds,
                ds_name='sciq',
                weak_model_config=weak_model_config,
                n_test_docs=100,
                mix_ratio=0.25,
                mix_strategy='sample',
                seed=42
            )

            # Check that stats were computed
            assert 'mixing/gt_examples' in stats
            assert 'mixing/weak_examples' in stats
            assert 'mixing/actual_gt_fraction' in stats
            assert 'mixing/requested_gt_fraction' in stats

            assert stats['mixing/gt_examples'] == 25
            assert stats['mixing/weak_examples'] == 75
            assert stats['mixing/actual_gt_fraction'] == 0.25
            assert stats['mixing/requested_gt_fraction'] == 0.25

    @patch('weak_to_strong.mixing.load_dataset')
    def test_label_level_mixing_stats(self, mock_load_dataset):
        """Test that label-level mixing computes entropy statistics."""
        weak_ds, gt_ds = create_dummy_datasets(n=100)

        # Mock the dataset loading
        mock_original_dataset = {'train': Dataset.from_dict({
            'txt': [f'example_{i}' for i in range(200)],
            'soft_label': [[0.5, 0.5] for _ in range(200)],
            'hard_label': [0] * 200
        })}
        mock_load_dataset.return_value = mock_original_dataset

        weak_model_config = {'seed': 42, 'n_docs': 200}

        with patch('weak_to_strong.mixing.create_mixed_supervision_dataset') as mock_create:
            # Create a mock mixed dataset with interpolated labels
            mock_mixed = Dataset.from_dict({
                'txt': [f'example_{i}' for i in range(100)],
                'soft_label': [[0.6, 0.4] for _ in range(100)],  # Mixed probabilities
                'hard_label': [0] * 100
            })
            mock_create.return_value = mock_mixed

            result_ds, stats = apply_mixed_supervision(
                weak_labeled_ds=weak_ds,
                ds_name='sciq',
                weak_model_config=weak_model_config,
                n_test_docs=100,
                mix_ratio=0.5,
                mix_strategy='label',
                seed=42
            )

            # Check that entropy stats were computed
            assert 'mixing/avg_label_entropy' in stats
            assert 'mixing/min_label_entropy' in stats
            assert 'mixing/max_label_entropy' in stats

            # Entropy should be positive for non-deterministic labels
            assert stats['mixing/avg_label_entropy'] > 0

    @patch('weak_to_strong.mixing.load_dataset')
    @patch('weak_to_strong.mixing.create_mixed_supervision_dataset')
    def test_loads_ground_truth_correctly(self, mock_create, mock_load_dataset):
        """Test that ground truth dataset is loaded with correct parameters."""
        weak_ds, gt_ds = create_dummy_datasets(n=100)

        # Mock the dataset loading
        mock_train_dataset = Dataset.from_dict({
            'txt': [f'example_{i}' for i in range(200)],
            'soft_label': [[0.5, 0.5] for _ in range(200)],
            'hard_label': [0] * 200
        })
        mock_original_dataset = {'train': mock_train_dataset}
        mock_load_dataset.return_value = mock_original_dataset
        mock_create.return_value = weak_ds

        weak_model_config = {
            'seed': 123,
            'n_docs': 1000
        }

        result_ds, stats = apply_mixed_supervision(
            weak_labeled_ds=weak_ds,
            ds_name='sciq',
            weak_model_config=weak_model_config,
            n_test_docs=500,
            mix_ratio=0.3,
            mix_strategy='sample',
            seed=42
        )

        # Verify load_dataset was called with correct parameters
        mock_load_dataset.assert_called_once_with(
            'sciq',
            seed=123,  # Should use weak model's seed
            split_sizes=dict(
                train=1000,  # Should use weak model's n_docs
                test=500
            )
        )

    @patch('weak_to_strong.mixing.load_dataset')
    @patch('weak_to_strong.mixing.create_mixed_supervision_dataset')
    def test_dataset_split_consistency(self, mock_create, mock_load_dataset):
        """Test that dataset is split the same way as weak labels were generated."""
        weak_ds, _ = create_dummy_datasets(n=100)

        # Mock the dataset loading with train_test_split
        mock_train_dataset = MagicMock()
        mock_split_result = {
            'train': Dataset.from_dict({'txt': ['train_ex']}),
            'test': Dataset.from_dict({'txt': ['test_ex']})
        }
        mock_train_dataset.train_test_split.return_value = mock_split_result
        mock_original_dataset = {'train': mock_train_dataset}
        mock_load_dataset.return_value = mock_original_dataset
        mock_create.return_value = weak_ds

        weak_model_config = {'seed': 999, 'n_docs': 200}

        result_ds, stats = apply_mixed_supervision(
            weak_labeled_ds=weak_ds,
            ds_name='sciq',
            weak_model_config=weak_model_config,
            n_test_docs=100,
            mix_ratio=0.5,
            mix_strategy='sample',
            seed=42
        )

        # Verify train_test_split was called with weak model's seed
        mock_train_dataset.train_test_split.assert_called_once_with(
            test_size=0.5,
            seed=999  # Should use weak model's seed for consistency
        )

        # Verify create_mixed_supervision_dataset received the 'test' split (train2_ds_gt)
        args, kwargs = mock_create.call_args
        assert kwargs['ground_truth_ds'] == mock_split_result['test']

    def test_integration_with_real_mixing_functions(self):
        """Integration test using real mixing functions (not mocked)."""
        weak_ds, _ = create_dummy_datasets(n=100)

        # Create a properly formatted mock for the original dataset
        with patch('weak_to_strong.mixing.load_dataset') as mock_load_dataset:
            _, gt_ds = create_dummy_datasets(n=200, seed=123)

            # Mock the entire flow
            mock_train_dataset = gt_ds
            mock_original_dataset = {'train': mock_train_dataset}
            mock_load_dataset.return_value = mock_original_dataset

            weak_model_config = {'seed': 123, 'n_docs': 200}

            # This should use real create_mixed_supervision_dataset
            result_ds, stats = apply_mixed_supervision(
                weak_labeled_ds=weak_ds,
                ds_name='sciq',
                weak_model_config=weak_model_config,
                n_test_docs=100,
                mix_ratio=0.25,
                mix_strategy='sample',
                seed=42
            )

            # Verify result
            assert len(result_ds) == len(weak_ds)
            assert 'label_source' in result_ds.column_names
            assert 'mixing/gt_examples' in stats
            assert 'mixing/actual_gt_fraction' in stats


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
