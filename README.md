**STATUS**: This codebase is not well tested and does not use the exact same settings we used in the paper, but in our experience gives qualitatively similar results when using large model size gaps and multiple seeds.  Expected results can be found for two datasets below.

## Recent Updates

- ✅ **Best Checkpoint Tracking**: Automatically saves and restores the best model based on validation accuracy (not early stopping - runs full epochs)
- ✅ **Refactored Mixing Logic**: Moved `apply_mixed_supervision()` to `mixing.py` module for proper testability
- ✅ **Comprehensive Unit Tests**: Added 6 new tests for `apply_mixed_supervision()` covering edge cases, both mixing strategies, and integration
- ✅ **W&B Auto-Logging**: Defaults to `weak-to-strong-mixing` project - just run `wandb login` once and experiments auto-log (no env vars needed!)
- ✅ **Code Documentation**: Clear separation of two execution paths (ground truth split vs. weak supervision)

# Weak-to-strong generalization

![Our setup and how it relates to superhuman AI alignment](./weak-to-strong-setup.png)

This project contains code for implementing our [paper on weak-to-strong generalization](https://cdn.openai.com/papers/weak-to-strong-generalization.pdf).

The primary codebase contains a re-implementation of our weak-to-strong learning setup for binary classification tasks.  The codebase contains code for fine-tuning pretrained language models, and also training against the labels from another language model.  We support various losses described in the paper as well, such as the confidence auxiliary loss.

The `vision` directory contains stand-alone code for weak-to-strong in the vision models setting (AlexNet -> DINO on ImageNet).

### Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

#### Installation

You need to have Python installed on your machine. The project uses `pyproject.toml` to manage dependencies. To install the dependencies, you can use a package manager like `pip`:

```
pip install .
```

For development (includes testing tools):
```
pip install ".[dev]"
```

#### Running the Script

The main script of the project is `sweep.py`. It can be run from the command line using the following command:
```
python sweep.py --model_sizes=gpt2,gpt2-medium
```

In addition to `--model_sizes`, `sweep.py` takes in almost all of the arguments that `train_simple.py` takes (e.g.
`--batch_size`, `--n_docs`, `--n_test_docs` etc., see `train_simple.py` for a full list). These arguments are simply
forwarded to `train_simple.py`.

`sweep.py` calls `train_simple.py` in the following way:
1. First, it calls `train_simple.py` for each model size to train the ground truth models
2. Then, for each pair of weak and strong models in `model_sizes` (where a model can be the strong model in the pair
   only if its index in the `model_sizes` list is >= the index of the weak model), it calls `train_simple.py` with a
   `--weak_model_size` argument so that the strong model is trained with the labels of the weak model.

E.g. the example above will run gpt2 (ground truth), gpt2-medium (ground truth), gpt2 -> gpt2, gpt2 -> gpt2-medium, and
gpt2-medium -> gpt2-medium.

If needed, you can also run `train_simple.py` directly.

Note that `sweep.py` will not accept the arguments `--weak_model_size`, `--weak_labels_path` or `--model_size` (as opposed
to `--model_sizes`, with an "s") as choosing their values automatically is precisely the point of `sweep.py`.

An example of Jupyter notebook for plotting results is found in `notebooks/Plotting.ipynb`.

#### Mixed Supervision

This codebase supports **mixed supervision**: training strong models with a combination of weak model predictions and ground truth labels. The `--mix_ratio` parameter controls the training mode.

**Three training modes:**

**1. Pure Ground Truth** (`--mix_ratio=1.0`, default)
```bash
# Standard weak-to-strong setup
python train_simple.py --model_size=gpt2 --ds_name=sciq --n_docs=10000

# Splits data 50/50, trains on first half (ground truth), generates weak labels on second half
```

**2. Pure Weak Supervision** (`--mix_ratio=0.0`)
```bash
# First generate weak labels
python train_simple.py --model_size=gpt2 --ds_name=sciq --n_docs=10000

# Then train on 100% weak labels
python train_simple.py \
    --model_size=gpt2-medium \
    --ds_name=sciq \
    --weak_labels_path=/tmp/results/default/{config}/weak_labels \
    --mix_ratio=0.0
```

**3. Mixed Supervision** (`0.0 < mix_ratio < 1.0`)
```bash
# First generate weak labels (same as above)
python train_simple.py --model_size=gpt2 --ds_name=sciq --n_docs=10000

# Then train with 75% weak + 25% ground truth
python train_simple.py \
    --model_size=gpt2-medium \
    --ds_name=sciq \
    --weak_labels_path=/tmp/results/default/{config}/weak_labels \
    --mix_ratio=0.25 \
    --mix_strategy=sample
```

**Two mixing strategies**:

- **Sample-level mixing** (`--mix_strategy=sample`): Randomly select `mix_ratio` fraction of examples to use ground truth labels, rest use weak labels
- **Label-level mixing** (`--mix_strategy=label`): Interpolate between weak and ground truth labels for every example: `soft_label = (1-α)*weak + α*gt`

**Automated sweeps**: Use `sweep_mixing.py` to run experiments across multiple mixing ratios:
```bash
# First generate weak labels
python train_simple.py --model_size=gpt2 --ds_name=sciq --n_docs=10000

# Then sweep over mixing ratios
python sweep_mixing.py \
    --mix_ratios="0,0.25,0.5,0.75,1.0" \
    --mix_strategy=sample \
    --model_size=gpt2-medium \
    --ds_name=sciq \
    --weak_labels_path=/tmp/results/default/{config}/weak_labels
```

The `mix_ratio` parameter controls the fraction of ground truth labels (0.0 = pure weak supervision, 1.0 = pure ground truth).

**Weights & Biases Logging**: Experiments are automatically logged to W&B (project: `weak-to-strong-mixing`):

```bash
# First time setup: authenticate with W&B
wandb login

# Run experiment - automatically logs to W&B
python train_simple.py \
    --model_size=gpt2-medium \
    --ds_name=sciq \
    --weak_labels_path=/tmp/results/default/{config}/weak_labels \
    --mix_ratio=0.25 \
    --mix_strategy=sample

# Optional: Use a different project name
export WANDB_PROJECT=my-custom-project

# Optional: Disable W&B logging
export WANDB_MODE=disabled
```

**Logged metrics**:
- Config parameters: `mix_ratio`, `mix_strategy`, `model_size`, `loss`, `lr`, etc.
- Mixing statistics: `mixing/gt_examples`, `mixing/actual_gt_fraction`, `mixing/avg_label_entropy`
- Training metrics: `train/loss`, `eval_accuracy` (from `train.py`)
- Final results: `final/weak_acc`, `final/strong_acc`, `final/transfer_acc`, `final/pgr` (Performance Gap Recovered)

The Performance Gap Recovered (PGR) metric shows what percentage of the gap between weak and strong models is recovered by the transfer model: `PGR = (transfer_acc - weak_acc) / (strong_acc - weak_acc)`

**Complete example with W&B sweep**:
```bash
# One-time setup: authenticate with W&B
wandb login

# Step 1: Generate weak labels
python train_simple.py --model_size=gpt2 --ds_name=sciq --n_docs=10000

# Step 2: Run mixing sweep (automatically logged to W&B)
python sweep_mixing.py \
    --mix_ratios="0,0.1,0.25,0.5,0.75,1.0" \
    --mix_strategy=sample \
    --model_size=gpt2-medium \
    --ds_name=sciq \
    --weak_labels_path=/tmp/results/default/{config}/weak_labels

# View results at: https://wandb.ai/your_username/weak-to-strong-mixing
# Analyze:
# - How PGR varies with mix_ratio
# - Optimal supervision budget
# - Sample vs label-level mixing strategies
```

#### Best Checkpoint Tracking

The training automatically tracks and saves the best model checkpoint based on validation accuracy:

```bash
python train_simple.py \
    --model_size=gpt2-medium \
    --ds_name=sciq \
    --n_docs=10000 \
    --eval_every=500 \
    --min_delta=0.001
```

**Parameters**:
- `--min_delta=X`: Minimum improvement threshold to count as progress (default: 0.0)
- `--restore_best_weights=True/False`: Load best checkpoint after training (default: True)
- `--eval_every=N`: How often to evaluate (required for checkpoint tracking)

**How it works**:
1. Model is evaluated every `eval_every` steps
2. When validation accuracy improves by at least `min_delta`, the checkpoint is saved to `{save_path}/best_checkpoint/`
3. After training completes (all epochs), the best weights are automatically restored if `restore_best_weights=True`
4. Final test evaluation uses the best checkpoint

**Example with mixed supervision**:
```bash
python train_simple.py \
    --model_size=gpt2-large \
    --ds_name=sciq \
    --weak_labels_path=/tmp/results/default/{config}/weak_labels \
    --mix_ratio=0.25 \
    --eval_every=200
```

**Note:** Training always runs for the full number of epochs. This feature just ensures you get the best checkpoint, not early stopping.

#### Code Architecture

The codebase is organized into modular components:

**Core Modules** (`weak_to_strong/`):
- `train.py` - Core training loop with best checkpoint tracking
- `mixing.py` - Mixed supervision functions for combining weak and ground truth labels
  - `mix_datasets_sample_level()` - Sample-level mixing strategy
  - `mix_datasets_label_level()` - Label-level mixing strategy
  - `create_mixed_supervision_dataset()` - Main entry point for mixing
  - `apply_mixed_supervision()` - High-level function that loads ground truth and applies mixing
  - `validate_mixing()` - Validates mixing was performed correctly
- `datasets.py` - Dataset loading and preprocessing
- `loss.py` - Loss functions (cross-entropy, logconf, product)
- `eval.py` - Model evaluation utilities
- `model.py` - Model architecture (TransformerWithHead)

**Scripts**:
- `train_simple.py` - Main CLI for training (supports two execution paths)
  - **Path 1**: Ground truth split - trains from scratch, splits data for weak label generation
  - **Path 2**: Weak supervision - loads pre-computed weak labels with optional mixing
- `sweep.py` - Sweeps across model sizes for weak-to-strong experiments
- `sweep_mixing.py` - Sweeps across mixing ratios for sample efficiency studies

**Tests** (`tests/`):
- `test_mixing.py` - Unit tests for all mixing functions including `apply_mixed_supervision`
- `test_dataset_consistency.py` - Dataset reconstruction validation

**Key Design Decisions**:
- Mixed supervision logic is in `mixing.py` module (not in train_simple.py script) for proper testing
- Best checkpoint tracking runs for full epochs (not early stopping) to maintain reproducibility
- Two mixing strategies enable different research questions about supervision budgets

#### Testing

To run the unit tests for the mixed supervision functionality:

```bash
# Install dev dependencies first
pip install ".[dev]"

# Run tests
pytest tests/test_mixing.py -v

# Or run all tests
pytest tests/ -v
```

**Test Coverage**:
- Sample-level and label-level mixing strategies
- Dataset alignment validation
- Statistics computation (GT fraction, label entropy)
- `apply_mixed_supervision()` function (dataset loading, splitting, mixing)
- Integration tests with real mixing functions

#### Expected results

<img src="notebooks/amazon_polarity.png" width="350">
<br>
<img src="notebooks/anthropic_hh.png" width="350">
<br>
<img src="notebooks/boolq.png" width="350">
<br>
<img src="notebooks/cosmos_qa.png" width="350">
<br>
<img src="notebooks/sciq.png" width="350">

### Authors

- Adrien Ecoffet
- Manas Joglekar
- Jeffrey Wu
- Jan Hendrik Kirchner
- Pavel Izmailov (vision)

### License

This project is licensed under the MIT License - see the LICENSE.md file for details.

### Acknowledgments

- Hugging Face for their open-source transformer models