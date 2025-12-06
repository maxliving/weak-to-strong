import pytest
from weak_to_strong.datasets import load_dataset

def test_dataset_reconstruction_consistency():
    """
    Test that loading a dataset with the same seed and n_docs produces
    identical splits, ensuring no data leakage or shift between runs.
    """
    ds_name = "sciq"
    seed = 42
    n_docs = 100
    n_test_docs = 50
    
    # 1. First Load (Simulating Weak Model Training)
    dataset1 = load_dataset(
        ds_name, 
        seed=seed, 
        split_sizes=dict(train=n_docs, test=n_test_docs)
    )
    train_dataset1 = dataset1["train"]
    
    # Perform the split exactly as train_simple.py does
    split_data1 = train_dataset1.train_test_split(test_size=0.5, seed=seed)
    train1_ds_run1 = split_data1["train"]
    train2_ds_run1 = split_data1["test"]

    # 2. Second Load (Simulating Strong Model / Transfer Training)
    # This mimics the logic where we reload based on saved config
    dataset2 = load_dataset(
        ds_name, 
        seed=seed,  # Critical: Same seed
        split_sizes=dict(train=n_docs, test=n_test_docs) # Critical: Same size
    )
    train_dataset2 = dataset2["train"]
    
    split_data2 = train_dataset2.train_test_split(test_size=0.5, seed=seed)
    train1_ds_run2 = split_data2["train"]
    train2_ds_run2 = split_data2["test"]

    # --- Assertions ---

    # 3. Verify Determinism: The datasets should be identical
    assert len(train1_ds_run1) == len(train1_ds_run2)
    for i in range(len(train1_ds_run1)):
        # Check input text
        assert train1_ds_run1[i]['txt'] == train1_ds_run2[i]['txt']
        # Check hard label
        assert train1_ds_run1[i]['hard_label'] == train1_ds_run2[i]['hard_label']
        
    # 4. Verify No Leakage: No overlap between Train1 (Weak Train) and Train2 (Weak Inference)
    # We use the text content as a unique identifier for this check
    train1_texts = set(ex['txt'] for ex in train1_ds_run1)
    train2_texts = set(ex['txt'] for ex in train2_ds_run1)
    
    intersection = train1_texts.intersection(train2_texts)
    assert len(intersection) == 0, f"Found {len(intersection)} overlapping examples between splits!"
    
    print("\nDataset consistency test passed: Splits are reproducible and non-overlapping.")

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
