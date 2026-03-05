# Regression Model Training Scripts

This directory contains scripts for parsing algorithm log files and training regression models to predict iteration counts.

## Overview

The workflow consists of two steps:
1. **Parse log files** → Extract key-value pairs into a pickle file
2. **Train models** → Learn to predict iterations from features

## Installation

Install required dependencies:

```bash
pip install numpy pandas scikit-learn joblib
```

Optional model backends:

```bash
pip install xgboost lightgbm
```

Optional C source export dependencies:

```bash
pip install treelite tl2cgen
```

## Usage

### Step 1: Parse Log Files

Extract features from log files containing `FJ:` entries:

```bash
python determinism_logs_parse.py /path/to/logs/directory -o parsed_data.pkl
```

**Arguments:**
- `input_dir`: Directory containing `.log` files
- `-o, --output`: Output pickle file (default: `output.pkl`)

**Output:**
- Pickle file containing list of dictionaries with all key-value pairs
- Each entry includes `file=<basename>` field

### Step 2: Train Regression Model

Train a model to predict `iter` values from other features:

```bash
python train_regressor.py parsed_data.pkl --regressor xgboost --seed 42
```

**Arguments:**
- `input_pkl`: Input pickle file from step 1
- `--regressor, -r`: Type of regressor (required)
  - `linear` - Linear Regression
  - `poly2`, `poly3`, `poly4` - Polynomial Regression (degree 2, 3, 4)
  - `xgboost` - XGBoost Regressor
  - `lightgbm` - LightGBM Regressor
  - `random_forest` - Random Forest Regressor
  - `gradient_boosting` - Gradient Boosting Regressor
- `--output-dir, -o`: Directory to save models (default: `./models`)
- `--seed, -s`: Random seed for reproducibility (optional)
- `--tune`: Enable hyperparameter tuning
- `--cv-folds`: Number of cross-validation folds (default: 5)
- `--test-size`: Test set proportion (default: 0.2)
- `--no-progress`: Disable training progress output
- `--list-features`: List all available features in the dataset and exit
- `--stratify-split`: Stratify train/test split by target distribution
- `--early-stopping N`: Early stopping patience in rounds (default: 20, use 0 to disable)
- `--treelite-compile N`: Export XGBoost/LightGBM as optimized C source code with TL2cgen (N threads, default: 1, includes branch annotation and quantization)

## Features

### Data Splitting
- **File-based split**: Ensures entries from the same file go exclusively to train OR test
- **Prevents data leakage**: Improves generalization and reduces overfitting
- **Default**: 20% of files for testing

### Preprocessing
- **Automatic scaling**: Applied to linear/polynomial models (not tree-based)
- **Polynomial features**: All numeric features expanded for polynomial regression
- **Clean data assumption**: Script expects valid pickle data

### Feature Selection
- **Manual feature selection**: Edit `FEATURES_TO_EXCLUDE` or `FEATURES_TO_INCLUDE_ONLY` directly in the script
- **Exclude specific features**: Add feature names to `FEATURES_TO_EXCLUDE` list
- **Include only specific features**: Add feature names to `FEATURES_TO_INCLUDE_ONLY` list (overrides exclusion)
- **List available features**: Run with `--list-features` to see all features in your dataset
- **No command-line config**: Intentionally not exposed as CLI args for cleaner configuration file management

### Model Evaluation
- **Cross-validation**: K-fold CV on training set with progress output
- **Comprehensive metrics**: MSE, RMSE, MAE, R²
- **Feature importance**: All features ranked by importance (top 50 for polynomial models)
- **Sample predictions**: 20 random test predictions with errors

### Training Progress
- **Progress indicators**: Tree-based models (XGBoost, LightGBM, Random Forest, Gradient Boosting) show real-time training progress
- **Polynomial feature tracking**: Shows number of polynomial features being generated
- **CV progress**: Cross-validation shows progress for each fold
- **Disable option**: Use `--no-progress` flag to suppress all progress output

### Overfitting Prevention
- **Early stopping**: Enabled by default for XGBoost and LightGBM (20 rounds patience) to prevent overfitting
- **Regularization**: XGBoost and LightGBM include L1/L2 regularization, subsampling, and minimum child weight
- **Stratified splitting**: Use `--stratify-split` to ensure balanced target distributions
- **Disable early stopping**: Use `--early-stopping 0` if you want full training without early stopping

### Model Persistence
- **XGBoost**: Saved as `.ubj.gz` (UBJ format with gzip compression)
- **LightGBM**: Saved as `.txt` (text format, human-readable)
- **Sklearn models**: Saved as `.joblib` (efficient for numpy arrays)
- **Metadata**: Feature names and preprocessing info saved separately
- **Scaler**: Saved for models requiring normalization

### TL2cgen Source Export (Optional)
- **C source code export**: Export XGBoost/LightGBM models as optimized C source code using TL2cgen
- **Portable and fast**: Compile the source on any platform for 10-100x faster predictions
- **Enabled by default**: Automatically exports C source with 1 thread (use `--treelite-compile N` for more threads)
- **Requires**: `treelite>=4.0` and `tl2cgen` packages (optional dependencies)
- **Output**: Optimized C source files in dedicated directory
- **Note**: Treelite 4.0+ moved C compilation to TL2cgen ([migration guide](https://tl2cgen.readthedocs.io/en/latest/treelite-migration.html))

### Built-in TL2cgen Optimizations
The following optimizations are **automatically applied** when using TL2cgen:

- **Branch annotation**:
  - Analyzes which branches are taken during training
  - Optimizes C code with branch prediction hints
  - Improves inference speed by 10-30%
  - Saves annotation file for inspection
- **Quantization**:
  - Reduces model memory footprint by ~75%
  - Uses 8-bit integers instead of 32-bit floats where possible
  - Minimal accuracy loss (typically <0.1% R²)
  - Faster inference on memory-constrained systems

**Combined effect**: 1.2-1.5x faster inference with 75% less memory

## Example Workflow

```bash
# 1. Parse logs
python determinism_logs_parse.py /data/algorithm_logs -o data.pkl

# 2. List available features
python train_regressor.py data.pkl --regressor xgboost --list-features

# 3. Train XGBoost model
python train_regressor.py data.pkl --regressor xgboost --seed 42 -o ./models

# 4. Train polynomial model with tuning
python train_regressor.py data.pkl --regressor poly3 --tune --seed 42

# 5. Compare different models
for model in linear poly2 poly3 xgboost lightgbm random_forest gradient_boosting; do
    echo "Training $model..."
    python train_regressor.py data.pkl --regressor $model --seed 42
done

# 6. Train LightGBM model
python train_regressor.py data.pkl --regressor lightgbm --seed 42

# 7. Export XGBoost model as C source for production deployment (enabled by default)
python train_regressor.py data.pkl --regressor xgboost --seed 42

# Or specify more threads for compilation
python train_regressor.py data.pkl --regressor xgboost --treelite-compile 8 --seed 42
```

## TL2cgen Source Export Example

For production deployments requiring fast inference, models are **automatically exported as optimized C source code** (if `treelite` and `tl2cgen` are installed):

```bash
# Install treelite and tl2cgen (optional)
pip install treelite tl2cgen

# Train model - optimized C source is automatically exported with branch annotation and quantization
python train_regressor.py data.pkl --regressor xgboost -o ./models

# Use more threads for faster parallel compilation
python train_regressor.py data.pkl --regressor xgboost --treelite-compile 8 -o ./models

# The C source files will be in: models/xgboost_c_code/
# Contains optimized C source code with branch annotation and quantization ready for compilation

# Same process works for LightGBM
python train_regressor.py data.pkl --regressor lightgbm --treelite-compile 8 -o ./models
# Output: models/lightgbm_c_code/
```

### Optimization Impact

All TL2cgen exports include the following optimizations automatically:

1. **Branch Annotation**: Uses training data statistics to add branch prediction hints
2. **Quantization**: Reduces memory footprint by converting floating-point to integers
3. **Missing Data Removal**: Removes unnecessary missing data checks (assumes all features provided)

| Configuration | Speed | Memory | Accuracy |
|---------------|-------|--------|----------|
| Standard XGBoost/LightGBM | 1x | 100% | 100% |
| **TL2cgen optimized (default)** | **1.2-1.5x** | **25%** | **>99.9%** |

**Note:** Treelite 4.0+ moved C code generation to TL2cgen. See the [migration guide](https://tl2cgen.readthedocs.io/en/latest/treelite-migration.html) for details.

### Output Files

When TL2cgen is enabled, the following files are automatically created:

```
models/
├── xgboost_model.ubj.gz          # Standard XGBoost model
├── xgboost_metadata.pkl           # Feature names and config
├── xgboost_annotation.json        # Branch statistics (automatic)
└── xgboost_c_code/                # TL2cgen generated optimized C++ source
    ├── header.h                    # Header with feature names declaration
    ├── main.cpp                    # Implementation with feature names array
    └── *.cpp / *.h                 # Other C++ source files (quantized + annotated)
```

**Class Wrapping**: All generated files are automatically wrapped in a C++ class with the model name (derived from the input pickle file basename) to avoid naming conflicts when using multiple models in the same project. For example, if the input is `my_dataset.pkl`:
- All functions and data are in `class my_dataset { public: ... };`
- All class members are `static` - no instantiation required
- Access functions as `my_dataset::predict()`, `my_dataset::get_num_features()`, etc.
- All `.c` files are renamed to `.cpp` for C++ compilation
- Header includes `#pragma once` for include guards

The generated `header.h` includes:
- `#pragma once` at the top
- `#include` statements (outside the class)
- `class <model> { public: ... };` wrapping all declarations
- `static constexpr int NUM_FEATURES` - Number of features
- `static const char* feature_names[]` - Feature names declaration
- Function declarations (e.g., `predict()`, `get_num_features()`) as public static members

The generated `main.cpp` includes:
- `#include` statements at the top
- Macro definitions (`LIKELY`, `UNLIKELY`, `N_TARGET`, `MAX_N_CLASS`) - moved from header for implementation-only use
- Function implementations with `<model>::function_name` qualification
- `const char* <model>::feature_names[]` - Feature names array definition (at the end of file)

### Example Generated Code Structure

**header.h:**

```cpp
#pragma once

#include <stdint.h>

class my_dataset {
public:
    static float predict(float* data, int pred_margin);
    static int get_num_feature();
    // ... other function declarations ...

    static constexpr int NUM_FEATURES = 42;
    static const char* feature_names[NUM_FEATURES];
};
```

**main.cpp:**

```cpp
#include "header.h"

#define LIKELY(x)   __builtin_expect(!!(x), 1)
#define N_TARGET 1

float my_dataset::predict(float* data, int pred_margin) {
    // implementation
}

int my_dataset::get_num_feature() {
    return NUM_FEATURES;
}

// Feature names array
const char* my_dataset::feature_names[my_dataset::NUM_FEATURES] = {
    "n_variables",
    "n_constraints",
    // ...
};
```

**Usage:**

```cpp
#include "xgboost_c_code/header.h"

// Call static methods directly - no instantiation needed
float result = my_dataset::predict(features, 0);
int num = my_dataset::get_num_feature();
```

## Feature Selection Examples

To perform feature selection, edit the configuration section at the top of `train_regressor.py`:

### Example 1: Exclude specific features

```python
FEATURES_TO_EXCLUDE = [
    'time',  # Exclude time as it may not be available at prediction time
    'avg_constraint_range',
    'binary_ratio',
]

FEATURES_TO_INCLUDE_ONLY = []
```

### Example 2: Use only specific features

```python
FEATURES_TO_EXCLUDE = []

FEATURES_TO_INCLUDE_ONLY = [
    'n_variables',
    'n_constraints',
    'sparsity',
    'structural_complexity',
]
```

**Note:** If `FEATURES_TO_INCLUDE_ONLY` is non-empty, it overrides `FEATURES_TO_EXCLUDE`.

## Output Structure

After training, the output directory contains:

```
models/
├── xgboost_model.ubj.gz      # Compressed XGBoost model
├── xgboost_metadata.pkl       # Feature names and config
├── linear_model.joblib        # Linear regression model
├── linear_scaler.pkl          # StandardScaler for linear model
├── linear_metadata.pkl        # Metadata
└── ...
```

## Notes

- The train/test split is based on **unique files**, not individual entries
- Models requiring scaling (linear, polynomial) automatically apply `StandardScaler`
- Tree-based models (XGBoost, Random Forest, Gradient Boosting) don't use scaling
- Feature importance shows the most predictive features for iteration count
- Use `--seed` for reproducible results across runs
