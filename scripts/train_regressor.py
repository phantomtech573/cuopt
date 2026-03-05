#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Train regression models to predict algorithm iterations from log features.

Usage:
    python train_regressor.py <input.feather> --regressor <type> [options]
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import joblib
import os
from typing import List, Dict, Any, Tuple
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


AVAILABLE_REGRESSORS = [
    "linear",
    "poly2",
    "poly3",
    "poly4",
    "xgboost",
    "lightgbm",
    "random_forest",
    "gradient_boosting",
]

# ============================================================================
# FEATURE SELECTION CONFIGURATION
# Edit this list to exclude specific features from training
# Leave empty to use all features (except 'file' and 'iter')
# ============================================================================
FEATURES_TO_EXCLUDE = [
    # Example usage (uncomment to exclude):
    # 'time',
    # 'avg_constraint_range',
    # 'binary_ratio',
    "avg_obj_coeff_magnitude",
    "n_of_minimums_for_exit",
    "feasibility_run",
    "fixed_var_ratio",
    "unbounded_var_ratio",
    "obj_var_ratio",
    "avg_related_vars_per_var",
    "avg_constraint_range",
    "nnz_variance",
    "avg_variable_range",
    "min_nnz_per_row",
    "constraint_var_ratio",
    "avg_var_degree",
    "equality_ratio",
    "integer_ratio",
    "binary_ratio",
    "max_related_vars",
    "problem_size_score",
    "structural_complexity",
    "tight_constraint_ratio",
    "tolerance",
    "time_limit",
    "tolerance",
    "primal_objective",
    "dual_objective",
    "gap",
    "l2_primal_residual",
    "l2_dual_residual",
    "detect_infeasibility",
    "iteration_limit",
    "termination",
    "check_infeasibility",
    "iter_since_best",
    "tid",
    "curr_obj",
    # "obj_weight",
    "L3_miss",
    "L2_miss",
    "L1_miss",
    "stores_per_iter",
    "loads_per_iter",
    # "is_feas",
    # "feas_found",
    # "viol_ratio",
    # "eval_intensity",
    # "nnz_per_move",
    "iter",
    # "max_weight",
    "avg_cstr_deg",
    "avg_var_deg",
    "lp_time",
    "node_id",
    "var_sel_time",
    "bound_str_time",
    "sb_time",
    "lp_status",
    "node_status",
    "depth",
    "cutoff_gap",
    "mem_bandwidth_gb_s",
    "max_cstr_deg",
    "viol_ratio",
    "nnz_per_move",
    "h_cstr_right_weights_loads",
    "h_cstr_left_weights_loads",
    "h_cstr_right_weights_stores",
    "h_cstr_left_weights_stores",
    "total_viol",
    "feas_found",
    "obj_weight",
    "h_tabu_nodec_until_stores",
    "h_tabu_nodec_until_loads",
    "h_tabu_noinc_until_stores",
    "h_tabu_noinc_until_loads",
    "h_tabu_lastdec_stores",
    "h_tabu_lastdec_loads",
    "h_tabu_lastinc_stores",
    "h_tabu_lastinc_loads",
    "max_weight",
    "fixed",
    "phase",
    "iters",
    "nnz/s",
    "nnz/iter",
    "runtime",
]

# Alternatively, specify ONLY the features you want to use
# If non-empty, only these features will be used (overrides FEATURES_TO_EXCLUDE)
FEATURES_TO_INCLUDE_ONLY = [
    # Example usage (uncomment to use only specific features):
    # 'n_variables',
    # 'n_constraints',
    #'sparsity',
    # "n_vars",
    # "n_cstrs",
    # #"total_nnz",
    # "mem_total_mb",
    # #"cache_hit_rate",
    # #"cstr_deg_cv"
    #  "mem_stores_mb",
    #  "mem_loads_mb",
]
# ============================================================================


def load_data(data_path: str, target_col: str = "iter") -> pd.DataFrame:
    """Load data file (supports .feather and legacy .pkl formats)."""
    ext = os.path.splitext(data_path)[1].lower()

    if ext == ".feather":
        # Fast Apache Arrow format
        df = pd.read_feather(data_path)
    elif ext == ".pkl":
        # Legacy pickle support
        with open(data_path, "rb") as f:
            data = pickle.load(f)

        if not isinstance(data, list):
            raise ValueError(
                f"Expected list of dictionaries, got {type(data)}"
            )

        if len(data) == 0:
            raise ValueError("Empty dataset")

        df = pd.DataFrame(data)
    else:
        raise ValueError(
            f"Unsupported file format: {ext}. Use .feather (recommended) or .pkl"
        )

    # Validate required columns
    if "file" not in df.columns:
        raise ValueError("Missing required 'file' column in data")
    if target_col not in df.columns:
        raise ValueError(
            f"Missing target column '{target_col}' in data. Available columns: {list(df.columns)}"
        )

    return df


def split_by_files(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = None,
    stratify_by: str = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/test sets based on unique files.
    Ensures all entries from a file go to either train or test, not both.

    Args:
    ----
        stratify_by: Optional column name to stratify split (e.g., 'iter' for balanced target distribution)
    """
    unique_files = df["file"].unique()

    # Optionally stratify by target distribution
    if stratify_by:
        # Create stratification labels based on quantiles of the specified column
        file_stats = df.groupby("file")[stratify_by].median()
        stratify_labels = pd.qcut(
            file_stats,
            q=min(5, len(unique_files)),
            labels=False,
            duplicates="drop",
        )
        train_files, test_files = train_test_split(
            unique_files,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_labels,
        )
    else:
        train_files, test_files = train_test_split(
            unique_files, test_size=test_size, random_state=random_state
        )

    train_df = df[df["file"].isin(train_files)].copy()
    test_df = df[df["file"].isin(test_files)].copy()

    # Validate no data leakage: ensure no overlap between train and test files
    train_files_set = set(train_files)
    test_files_set = set(test_files)
    overlap = train_files_set.intersection(test_files_set)

    if overlap:
        raise ValueError(
            f"Data leakage detected! {len(overlap)} file(s) appear in both train and test sets:\n"
            f"  {list(overlap)[:10]}{'...' if len(overlap) > 10 else ''}"
        )

    # Verify the actual dataframes have no file overlap
    train_files_in_df = set(train_df["file"].unique())
    test_files_in_df = set(test_df["file"].unique())
    actual_overlap = train_files_in_df.intersection(test_files_in_df)

    if actual_overlap:
        raise ValueError(
            f"Data leakage detected in dataframes! {len(actual_overlap)} file(s) appear in both:\n"
            f"  {list(actual_overlap)[:10]}{'...' if len(actual_overlap) > 10 else ''}"
        )

    print("\nData Split:")
    print(f"  Total entries: {len(df)}")
    print(f"  Train entries: {len(train_df)} ({len(train_files)} files)")
    print(f"  Test entries: {len(test_df)} ({len(test_files)} files)")
    print("  ✓ Verified: Zero file overlap between train and test sets")

    # Check distribution similarity (use stratify_by column if provided, otherwise first numeric column)
    target_col = (
        stratify_by
        if stratify_by
        else df.select_dtypes(include=[np.number]).columns[0]
    )
    if target_col in train_df.columns and target_col in test_df.columns:
        train_target_mean = train_df[target_col].mean()
        test_target_mean = test_df[target_col].mean()
        train_target_std = train_df[target_col].std()
        test_target_std = test_df[target_col].std()

        print(f"\nTarget ('{target_col}') Distribution:")
        print(
            f"  Train: mean={train_target_mean:.2f}, std={train_target_std:.2f}"
        )
        print(
            f"  Test:  mean={test_target_mean:.2f}, std={test_target_std:.2f}"
        )

        mean_diff_pct = (
            abs(train_target_mean - test_target_mean) / train_target_mean * 100
            if train_target_mean != 0
            else 0
        )
        if mean_diff_pct > 10:
            print(
                f"  ⚠️  Warning: Train/test target means differ by {mean_diff_pct:.1f}%"
            )
            print(
                "      Consider using stratified split or different random seed"
            )

    return train_df, test_df


def list_available_features(
    df: pd.DataFrame, target_col: str = "iter"
) -> List[str]:
    """
    List all available numeric features in the dataset.
    Helper function to see what features can be selected/excluded.
    """
    # Drop target and metadata columns
    cols_to_drop = [
        target_col,
        "file",
        "iter",
        "iterations",
    ]  # Drop common target column names
    X = df.drop(
        columns=[c for c in cols_to_drop if c in df.columns], errors="ignore"
    )
    X = X.select_dtypes(include=[np.number])
    return sorted(X.columns.tolist())


def validate_data_quality(
    df: pd.DataFrame, target_col: str = "iter", verbose: bool = True
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate data quality and report issues.

    Returns
    -------
        (is_valid, report_dict)
    """
    report = {
        "target_issues": [],
        "feature_issues": [],
        "rows_with_issues": [],
        "is_valid": True,
    }

    # Check target variable
    if target_col not in df.columns:
        report["is_valid"] = False
        report["target_issues"].append(f"Missing '{target_col}' column")
        return False, report

    y = df[target_col]

    # Check for NaN in target
    nan_count = y.isna().sum()
    if nan_count > 0:
        report["is_valid"] = False
        report["target_issues"].append(
            f"Target has {nan_count} NaN values ({nan_count / len(y) * 100:.1f}%)"
        )
        report["rows_with_issues"].extend(df[y.isna()].index.tolist())

    # Check for inf in target
    inf_count = np.isinf(y).sum()
    if inf_count > 0:
        report["is_valid"] = False
        report["target_issues"].append(
            f"Target has {inf_count} infinite values ({inf_count / len(y) * 100:.1f}%)"
        )
        report["rows_with_issues"].extend(df[np.isinf(y)].index.tolist())

    # Check for extreme values in target
    if not y.isna().all() and not np.isinf(y).all():
        y_clean = y[~(y.isna() | np.isinf(y))]
        if len(y_clean) > 0:
            y_max = y_clean.max()
            y_min = y_clean.min()
            if y_max > 1e10:
                report["target_issues"].append(
                    f"Target has very large values (max={y_max:.2e})"
                )
            if y_min < -1e10:
                report["target_issues"].append(
                    f"Target has very large negative values (min={y_min:.2e})"
                )

    # Check features
    X = df.drop(columns=[target_col, "file"], errors="ignore")
    X_numeric = X.select_dtypes(include=[np.number])

    for col in X_numeric.columns:
        col_data = X_numeric[col]

        # Check for NaN
        nan_count = col_data.isna().sum()
        if nan_count > 0:
            pct = nan_count / len(col_data) * 100
            if pct > 10:  # Only report if > 10% are NaN
                report["feature_issues"].append(
                    f"{col}: {nan_count} NaN ({pct:.1f}%)"
                )

        # Check for inf
        inf_count = np.isinf(col_data).sum()
        if inf_count > 0:
            report["is_valid"] = False
            pct = inf_count / len(col_data) * 100
            report["feature_issues"].append(
                f"{col}: {inf_count} infinite ({pct:.1f}%)"
            )
            report["rows_with_issues"].extend(
                df[np.isinf(col_data)].index.tolist()
            )

    # Deduplicate row indices
    report["rows_with_issues"] = sorted(list(set(report["rows_with_issues"])))

    # Print report if verbose
    if verbose and (report["target_issues"] or report["feature_issues"]):
        print("\n" + "=" * 70)
        print("DATA QUALITY ISSUES DETECTED")
        print("=" * 70)

        if report["target_issues"]:
            print("\nTarget Variable Issues:")
            for issue in report["target_issues"]:
                print(f"  ❌ {issue}")

        if report["feature_issues"]:
            print(
                f"\nFeature Issues ({len(report['feature_issues'])} features affected):"
            )
            for issue in report["feature_issues"][:10]:  # Show first 10
                print(f"  ⚠️  {issue}")
            if len(report["feature_issues"]) > 10:
                print(
                    f"  ... and {len(report['feature_issues']) - 10} more features"
                )

        if report["rows_with_issues"]:
            print(f"\nAffected Rows: {len(report['rows_with_issues'])} total")
            if len(report["rows_with_issues"]) <= 10:
                print(f"  Row indices: {report['rows_with_issues']}")
            else:
                print(f"  First 10: {report['rows_with_issues'][:10]}")

            # Show sample problematic rows with filenames
            if "file" in df.columns:
                print("\n  Sample problematic entries:")
                sample_indices = report["rows_with_issues"][:5]
                for idx in sample_indices:
                    if idx < len(df):
                        filename = df.iloc[idx].get("file", "unknown")
                        iter_val = df.iloc[idx].get("iter", "N/A")
                        print(
                            f"    Row {idx}: file={filename}, iter={iter_val}"
                        )

        print("\nSuggested Actions:")
        if not report["is_valid"]:
            print("  1. Remove rows with invalid data: --drop-invalid-rows")
            print("  2. Check your log files for data collection issues")
            print("  3. Verify algorithm didn't produce invalid results")
        else:
            print(
                "  Data is valid but has some NaN values in features (will be handled)"
            )

        print("=" * 70 + "\n")

    return report["is_valid"], report


def prepare_features(
    df: pd.DataFrame, target_col: str = "iter"
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepare features and target from DataFrame.
    Excludes 'file' and target column from features.
    Applies feature selection based on FEATURES_TO_EXCLUDE and FEATURES_TO_INCLUDE_ONLY.
    """
    # Separate target
    y = df[target_col].copy()

    # Drop non-feature columns
    X = df.drop(columns=[target_col, "file"])

    # Ensure all features are numeric
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"Warning: Dropping non-numeric columns: {non_numeric}")
        X = X.select_dtypes(include=[np.number])

    # Apply feature selection
    original_feature_count = len(X.columns)

    if FEATURES_TO_INCLUDE_ONLY:
        # Use only specified features
        available_features = [
            f for f in FEATURES_TO_INCLUDE_ONLY if f in X.columns
        ]
        missing_features = [
            f for f in FEATURES_TO_INCLUDE_ONLY if f not in X.columns
        ]

        if missing_features:
            print(
                f"Warning: Requested features not found in data: {missing_features}"
            )

        X = X[available_features]
        print(
            f"Feature selection: Using only {len(available_features)} specified features"
        )

    elif FEATURES_TO_EXCLUDE:
        # Exclude specified features
        features_to_drop = [f for f in FEATURES_TO_EXCLUDE if f in X.columns]
        if features_to_drop:
            X = X.drop(columns=features_to_drop)
            print(
                f"Feature selection: Excluded {len(features_to_drop)} features: {features_to_drop}"
            )

    feature_names = X.columns.tolist()

    if len(feature_names) == 0:
        raise ValueError(
            "No features remaining after feature selection! "
            "Check FEATURES_TO_EXCLUDE and FEATURES_TO_INCLUDE_ONLY settings."
        )

    if len(feature_names) != original_feature_count:
        print(
            f"  Using {len(feature_names)} of {original_feature_count} available features"
        )

    return X, y, feature_names


def create_regressor(
    regressor_type: str,
    random_state: int = None,
    tune_hyperparams: bool = False,
    verbose: bool = True,
):
    """
    Create a regression model with optional preprocessing pipeline.

    Returns: (model, needs_scaling)
    """
    if regressor_type == "linear":
        model = LinearRegression()
        needs_scaling = True

    elif regressor_type.startswith("poly"):
        degree = int(regressor_type[-1])
        model = Pipeline(
            [
                (
                    "poly",
                    PolynomialFeatures(degree=degree, include_bias=False),
                ),
                (
                    "regressor",
                    Ridge(alpha=1.0),
                ),  # Ridge to handle multicollinearity
            ]
        )
        needs_scaling = True

    elif regressor_type == "xgboost":
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError(
                "XGBoost not installed. Install with: pip install xgboost"
            )

        params = {
            "objective": "reg:squarederror",
            "random_state": random_state,
            "n_estimators": 100,
            "max_depth": 6,
            "tree_method": "hist",
            "learning_rate": 0.1,
            "verbosity": 1 if verbose else 0,
            # Regularization to prevent overfitting
            "min_child_weight": 3,  # Minimum sum of weights in a leaf
            "gamma": 0.1,  # Minimum loss reduction for split
            "subsample": 0.8,  # Fraction of samples per tree
            "colsample_bytree": 0.8,  # Fraction of features per tree
            "reg_alpha": 0.1,  # L1 regularization
            "reg_lambda": 1.0,  # L2 regularization
        }

        if tune_hyperparams:
            # Stronger regularization for tuned version
            params.update(
                {
                    "n_estimators": 200,
                    "max_depth": 5,  # Shallower trees
                    "learning_rate": 0.05,  # Lower learning rate
                    "min_child_weight": 5,  # Higher minimum weight
                    "gamma": 0.2,  # More conservative splits
                    "subsample": 0.7,  # More aggressive subsampling
                    "colsample_bytree": 0.7,
                    "reg_alpha": 0.5,  # Stronger L1
                    "reg_lambda": 2.0,  # Stronger L2
                }
            )

        model = xgb.XGBRegressor(**params)
        needs_scaling = True

    elif regressor_type == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError(
                "LightGBM not installed. Install with: pip install lightgbm"
            )

        params = {
            "objective": "regression",
            "random_state": random_state,
            "n_estimators": 150,
            "max_depth": 6,
            "learning_rate": 0.1,
            "verbosity": 1 if verbose else -1,
            # Regularization to prevent overfitting
            "min_child_weight": 3,
            "min_split_gain": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        }

        if tune_hyperparams:
            # Stronger regularization for tuned version
            params.update(
                {
                    "n_estimators": 200,
                    "max_depth": 5,
                    "learning_rate": 0.05,
                    "min_child_weight": 5,
                    "min_split_gain": 0.2,
                    "subsample": 0.7,
                    "colsample_bytree": 0.7,
                    "reg_alpha": 0.5,
                    "reg_lambda": 2.0,
                }
            )

        model = lgb.LGBMRegressor(**params)
        needs_scaling = True

    elif regressor_type == "random_forest":
        params = {
            "random_state": random_state,
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "verbose": 1 if verbose else 0,
        }

        if tune_hyperparams:
            params.update(
                {"n_estimators": 200, "max_depth": 20, "min_samples_split": 5}
            )

        model = RandomForestRegressor(**params)
        needs_scaling = False

    elif regressor_type == "gradient_boosting":
        params = {
            "random_state": random_state,
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "verbose": 1 if verbose else 0,
        }

        if tune_hyperparams:
            params.update(
                {"n_estimators": 200, "max_depth": 7, "learning_rate": 0.05}
            )

        model = GradientBoostingRegressor(**params)
        needs_scaling = False

    else:
        raise ValueError(f"Unknown regressor type: {regressor_type}")

    return model, needs_scaling


def get_feature_importance(
    model, feature_names: List[str], regressor_type: str
) -> None:
    """Extract and print feature importance if available."""
    print("\nFeature Importance:")

    try:
        if regressor_type in [
            "xgboost",
            "lightgbm",
            "random_forest",
            "gradient_boosting",
        ]:
            # Tree-based models have feature_importances_
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]

            for i, idx in enumerate(indices, 1):
                print(
                    f"  {i:3d}. {feature_names[idx]:40s}: {importances[idx]:.6f}"
                )

        elif regressor_type == "linear":
            # Linear regression coefficients and intercept
            print(f"\nIntercept: {model.intercept_:.6f}\n")
            print("Coefficients:")

            # Print each feature with its coefficient
            for i, (feature_name, coef) in enumerate(
                zip(feature_names, model.coef_), 1
            ):
                print(f"  {i:3d}. {feature_name:40s}: {coef:.6f}")

            # Also show sorted by absolute value for importance ranking
            print("\nRanked by absolute magnitude:")
            coefs_abs = np.abs(model.coef_)
            indices = np.argsort(coefs_abs)[::-1]
            for i, idx in enumerate(indices, 1):
                print(
                    f"  {i:3d}. {feature_names[idx]:40s}: {model.coef_[idx]:.6f} (|{coefs_abs[idx]:.6f}|)"
                )

        elif regressor_type.startswith("poly"):
            # For polynomial, get feature names and coefficients from the Ridge step
            poly_features = model.named_steps["poly"].get_feature_names_out(
                feature_names
            )
            coefs = np.abs(model.named_steps["regressor"].coef_)
            indices = np.argsort(coefs)[::-1]

            print(f"  (Showing top 50 of {len(indices)} polynomial features)")
            for i, idx in enumerate(indices[:50], 1):
                feat_name = poly_features[idx]
                # Truncate very long polynomial feature names
                if len(feat_name) > 60:
                    feat_name = feat_name[:57] + "..."
                print(f"  {i:3d}. {feat_name:60s}: {coefs[idx]:.6f}")
        else:
            print("  Feature importance not available for this model type")

    except Exception as e:
        print(f"  Could not extract feature importance: {e}")


def evaluate_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    feature_names: List[str],
    regressor_type: str,
    cv_folds: int = 5,
    verbose: int = 0,
    skip_cv: bool = False,
    X_test_original: pd.DataFrame = None,
    test_df: pd.DataFrame = None,
    log_transform: bool = False,
    y_train_original: pd.Series = None,
    y_test_original: pd.Series = None,
    log_transform_offset: float = 0.0,
) -> Tuple[float, float]:
    """Evaluate model and print metrics. Returns (train_r2, test_r2).

    Args:
    ----
        X_test_original: Unscaled X_test for displaying feature values
        test_df: Original test dataframe with 'file' column
        log_transform: If True, predictions are in log-space and need inverse transform
        y_train_original: Original (non-log) training targets (if log_transform=True)
        y_test_original: Original (non-log) test targets (if log_transform=True)
        log_transform_offset: Constant added before log transform (if log_transform=True)
    """
    # Cross-validation on training set (skip if using early stopping)
    if not skip_cv:
        print(f"\nCross-Validation on Training Set ({cv_folds}-fold):")
        try:
            # Compute RMSE and R² in log-space
            cv_scores_mse = cross_val_score(
                model,
                X_train,
                y_train,
                cv=cv_folds,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
                verbose=verbose,
            )
            cv_scores_r2 = cross_val_score(
                model,
                X_train,
                y_train,
                cv=cv_folds,
                scoring="r2",
                n_jobs=-1,
                verbose=verbose,
            )
            cv_rmse = np.sqrt(-cv_scores_mse)

            if log_transform:
                print("  Log-space metrics:")
                print(
                    f"    CV RMSE: {cv_rmse.mean():.4f} (+/- {cv_rmse.std():.4f})"
                )
                print(
                    f"    CV R²:   {cv_scores_r2.mean():.4f} (+/- {cv_scores_r2.std():.4f})"
                )

                # Also compute metrics in original space
                kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

                cv_mape_scores = []
                cv_r2_original_scores = []

                for train_idx, val_idx in kfold.split(X_train):
                    X_train_fold = (
                        X_train.iloc[train_idx]
                        if hasattr(X_train, "iloc")
                        else X_train[train_idx]
                    )
                    y_train_fold = (
                        y_train.iloc[train_idx]
                        if hasattr(y_train, "iloc")
                        else y_train[train_idx]
                    )
                    X_val_fold = (
                        X_train.iloc[val_idx]
                        if hasattr(X_train, "iloc")
                        else X_train[val_idx]
                    )
                    y_val_original_fold = (
                        y_train_original.iloc[val_idx]
                        if hasattr(y_train_original, "iloc")
                        else y_train_original[val_idx]
                    )

                    # Train on fold
                    model_fold = type(model)(**model.get_params())
                    model_fold.fit(X_train_fold, y_train_fold)

                    # Predict and transform back
                    y_pred_log = model_fold.predict(X_val_fold)
                    y_pred_original = np.exp(y_pred_log) - log_transform_offset

                    # Compute metrics in original space
                    mape = (
                        np.mean(
                            np.abs(
                                (y_val_original_fold - y_pred_original)
                                / y_val_original_fold
                            )
                        )
                        * 100
                    )
                    r2_original = r2_score(
                        y_val_original_fold, y_pred_original
                    )

                    cv_mape_scores.append(mape)
                    cv_r2_original_scores.append(r2_original)

                cv_mape = np.array(cv_mape_scores)
                cv_r2_original = np.array(cv_r2_original_scores)

                print("  Original-space metrics:")
                print(
                    f"    CV MAPE: {cv_mape.mean():.2f}% (+/- {cv_mape.std():.2f}%)"
                )
                print(
                    f"    CV R²:   {cv_r2_original.mean():.4f} (+/- {cv_r2_original.std():.4f})"
                )
            else:
                print(
                    f"  CV RMSE: {cv_rmse.mean():.4f} (+/- {cv_rmse.std():.4f})"
                )
                print(
                    f"  CV R²:   {cv_scores_r2.mean():.4f} (+/- {cv_scores_r2.std():.4f})"
                )

        except Exception as e:
            print(
                f"  CV failed (likely due to early stopping): {str(e)[:100]}"
            )
            print("  Skipping cross-validation...")
    else:
        print("\nSkipping cross-validation (incompatible with early stopping)")

    # Training set metrics
    y_train_pred = model.predict(X_train)

    # If log-transformed, also compute metrics in original space
    if log_transform:
        # Inverse transform predictions
        y_train_pred_original = np.exp(y_train_pred) - log_transform_offset
        y_test_pred_log = model.predict(X_test)
        y_test_pred_original = np.exp(y_test_pred_log) - log_transform_offset

        # Metrics in log-space
        train_mse_log = mean_squared_error(y_train, y_train_pred)
        train_rmse_log = np.sqrt(train_mse_log)
        train_r2_log = r2_score(y_train, y_train_pred)

        # Metrics in original space
        train_mse = mean_squared_error(y_train_original, y_train_pred_original)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(
            y_train_original, y_train_pred_original
        )
        train_r2 = r2_score(y_train_original, y_train_pred_original)
        train_mape = (
            np.mean(
                np.abs(
                    (y_train_original - y_train_pred_original)
                    / y_train_original
                )
            )
            * 100
        )

        print("\nTraining Set Metrics (Original Space):")
        print(f"  MSE:   {train_mse:.4f}")
        print(f"  RMSE:  {train_rmse:.4f}")
        print(f"  MAE:   {train_mae:.4f}")
        print(f"  MAPE:  {train_mape:.2f}%  (optimized metric)")
        print(f"  R²:    {train_r2:.4f}")
        print("\nTraining Set Metrics (Log Space):")
        print(f"  RMSE:  {train_rmse_log:.4f}")
        print(f"  R²:    {train_r2_log:.4f}")
    else:
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)

        print("\nTraining Set Metrics:")
        print(f"  MSE:  {train_mse:.4f}")
        print(f"  RMSE: {train_rmse:.4f}")
        print(f"  MAE:  {train_mae:.4f}")
        print(f"  R²:   {train_r2:.4f}")

    # Test set metrics
    if log_transform:
        # Already computed above
        test_mse_log = mean_squared_error(y_test, y_test_pred_log)
        test_rmse_log = np.sqrt(test_mse_log)
        test_r2_log = r2_score(y_test, y_test_pred_log)

        # Metrics in original space
        test_mse = mean_squared_error(y_test_original, y_test_pred_original)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test_original, y_test_pred_original)
        test_r2 = r2_score(y_test_original, y_test_pred_original)
        test_mape = (
            np.mean(
                np.abs(
                    (y_test_original - y_test_pred_original) / y_test_original
                )
            )
            * 100
        )

        print("\nTest Set Metrics (Original Space):")
        print(f"  MSE:   {test_mse:.4f}")
        print(f"  RMSE:  {test_rmse:.4f}")
        print(f"  MAE:   {test_mae:.4f}")
        print(f"  MAPE:  {test_mape:.2f}%  (optimized metric)")
        print(f"  R²:    {test_r2:.4f}")
        print("\nTest Set Metrics (Log Space):")
        print(f"  RMSE:  {test_rmse_log:.4f}")
        print(f"  R²:    {test_r2_log:.4f}")

        # Use original space predictions for sample display
        y_test_pred = y_test_pred_original
        y_test = y_test_original
    else:
        y_test_pred = model.predict(X_test)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        print("\nTest Set Metrics:")
        print(f"  MSE:  {test_mse:.4f}")
        print(f"  RMSE: {test_rmse:.4f}")
        print(f"  MAE:  {test_mae:.4f}")
        print(f"  R²:   {test_r2:.4f}")

    # If R² is negative, show baseline comparison for debugging
    if test_r2 < 0:
        y_test_mean = np.mean(y_test)
        baseline_pred = np.full_like(y_test_pred, y_test_mean)
        baseline_mse = mean_squared_error(y_test, baseline_pred)
        baseline_rmse = np.sqrt(baseline_mse)
        print("\n  WARNING: Negative R² detected!")
        print(
            f"  This means the model is worse than predicting the mean: {y_test_mean:.4f}"
        )
        print(f"  Baseline (mean) RMSE: {baseline_rmse:.4f}")
        print(f"  Model RMSE:           {test_rmse:.4f}")
        print(
            f"  Model is {test_rmse / baseline_rmse:.2f}x worse than baseline"
        )

    # Feature importance
    get_feature_importance(model, feature_names, regressor_type)

    # Sample predictions
    print("\n20 Sample Predictions from Test Set:")
    print(
        f"  {'Actual':>10s}  {'Predicted':>10s}  {'Error':>10s}  {'Error %':>10s}"
    )
    print(f"  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 10}")

    sample_indices = np.random.choice(
        len(y_test), min(20, len(y_test)), replace=False
    )
    for idx in sample_indices:
        actual = y_test.iloc[idx]
        predicted = y_test_pred[idx]
        error = actual - predicted
        error_pct = (error / actual * 100) if actual != 0 else 0
        print(
            f"  {actual:10.2f}  {predicted:10.2f}  {error:10.2f}  {error_pct:9.2f}%"
        )

    # Show worst predictions with feature values
    print("\n5 Worst Predictions (Largest Absolute Error):")
    abs_errors = np.abs(y_test_pred - y_test.values)
    worst_indices = np.argsort(abs_errors)[-5:][::-1]

    # Use original (unscaled) features if available, otherwise use X_test
    X_display = X_test_original if X_test_original is not None else X_test

    for rank, idx in enumerate(worst_indices, 1):
        actual = y_test.iloc[idx]
        predicted = y_test_pred[idx]
        error = actual - predicted
        error_pct = (error / actual * 100) if actual != 0 else 0

        # Get filename if available
        filename = ""
        if test_df is not None and "file" in test_df.columns:
            filename = f" (file: {test_df.iloc[idx]['file']})"

        print(
            f"\n  #{rank} - Actual: {actual:.2f}, Predicted: {predicted:.2f}, "
            f"Error: {error:.2f} ({error_pct:.1f}%){filename}"
        )

        # Get feature values (handle both DataFrame and array)
        if isinstance(X_display, pd.DataFrame):
            feature_values = X_display.iloc[idx].values
        elif isinstance(X_display, np.ndarray):
            feature_values = X_display[idx]
        else:
            feature_values = X_display[idx]

        # Display features compactly (5 per line)
        print("      Features:", end="")
        for i, (feat_name, feat_val) in enumerate(
            zip(feature_names, feature_values)
        ):
            if i % 5 == 0:
                print("\n        ", end="")
            # Format feature value
            if isinstance(feat_val, (int, np.integer)):
                print(f"{feat_name}={feat_val}", end="  ")
            else:
                print(f"{feat_name}={feat_val:.3g}", end="  ")
        print()  # Final newline

    return train_r2, test_r2


def compile_model_treelite(
    model,
    regressor_type: str,
    output_dir: str,
    num_threads: int,
    X_train=None,
    annotate: bool = False,
    quantize: bool = False,
    feature_names: List[str] = None,
    model_name: str = None,
    log_transform: bool = False,
    log_transform_offset: float = 0.0,
) -> None:
    """Compile XGBoost/LightGBM model to C source files using TL2cgen.

    Args:
    ----
        model: Trained model
        regressor_type: Type of regressor
        output_dir: Output directory
        num_threads: Number of parallel compilation threads
        X_train: Training data for branch annotation (optional)
        annotate: Whether to annotate branches for optimization
        quantize: Whether to use quantization in code generation
        feature_names: List of feature names in expected order (optional)
        model_name: Name prefix for functions (optional, derived from training file)
        log_transform: Whether model predicts in log-space (will add exp() wrapper)
        log_transform_offset: Constant added before log transform (if log_transform=True)
    """
    if regressor_type not in ["xgboost", "lightgbm"]:
        print(
            f"Warning: TL2cgen compilation only supported for XGBoost and LightGBM, skipping for {regressor_type}"
        )
        return

    try:
        import treelite
        import tl2cgen
    except ImportError:
        missing = []
        try:
            import treelite
        except ImportError:
            missing.append("treelite")
        try:
            import tl2cgen
        except ImportError:
            missing.append("tl2cgen")

        print(
            f"Warning: {', '.join(missing)} not installed. Install with: pip install {' '.join(missing)}"
        )
        print("Skipping C code generation.")
        return

    optimization_info = []
    if annotate:
        optimization_info.append("branch annotation")
    if quantize:
        optimization_info.append("quantization")

    opt_str = (
        f" with {', '.join(optimization_info)}" if optimization_info else ""
    )
    print(
        f"\nGenerating C source code with TL2cgen (threads={num_threads}){opt_str}..."
    )

    # Convert model to treelite format using frontend API
    try:
        if regressor_type == "xgboost":
            tl_model = treelite.frontend.from_xgboost(model.get_booster())
        elif regressor_type == "lightgbm":
            tl_model = treelite.frontend.from_lightgbm(model.booster_)
    except Exception as e:
        print(
            f"Warning: Failed to convert {regressor_type} model to treelite: {e}"
        )
        return

    # Annotate branches if requested and training data is available
    annotation_path = None
    if annotate and X_train is not None:
        try:
            print("  Annotating branches with training data...")
            # Convert to numpy array if it's a DataFrame
            if hasattr(X_train, "values"):
                X_train_array = X_train.values.astype(np.float32)
            else:
                X_train_array = np.asarray(X_train, dtype=np.float32)

            dmat = tl2cgen.DMatrix(X_train_array, dtype="float32")
            annotation_path = os.path.join(
                output_dir, f"{regressor_type}_annotation.json"
            )
            tl2cgen.annotate_branch(
                tl_model, dmat=dmat, path=annotation_path, verbose=False
            )
            print(f"  Branch annotations saved to: {annotation_path}")
        except Exception as e:
            print(f"  Warning: Branch annotation failed: {e}")
            print("  Continuing without branch annotation")
            annotation_path = None
    elif annotate and X_train is None:
        print(
            "  Warning: Branch annotation requested but no training data available"
        )
        print("  Skipping branch annotation")

    # Generate C source files using TL2cgen
    source_dir = os.path.join(output_dir, f"{regressor_type}_c_code")

    try:
        # params = {'parallel_comp': num_threads}
        params = {}

        # Add quantization parameter if requested
        if quantize:
            params["quantize"] = 1  # Enable quantization in code generation

        # Add annotation file if available
        if annotation_path:
            params["annotate_in"] = annotation_path

        tl2cgen.generate_c_code(
            tl_model, dirpath=source_dir, params=params, verbose=False
        )

        # Post-process generated files
        header_path = os.path.join(source_dir, "header.h")
        main_path = os.path.join(source_dir, "main.c")
        quantize_path = os.path.join(source_dir, "quantize.c")
        recipe_path = os.path.join(source_dir, "recipe.json")

        # Rename all .c files to .cpp and wrap in class
        if model_name:
            try:
                import glob

                c_files = glob.glob(os.path.join(source_dir, "*.c"))

                for c_file in c_files:
                    cpp_file = c_file[:-2] + ".cpp"

                    # Read content
                    with open(c_file, "r") as f:
                        content = f.read()

                    # Split content into includes and rest
                    lines = content.split("\n")
                    include_lines = []
                    code_lines = []
                    in_includes = True

                    for line in lines:
                        if in_includes and (
                            line.strip().startswith("#include")
                            or line.strip().startswith("#")
                            or line.strip() == ""
                        ):
                            include_lines.append(line)
                        else:
                            in_includes = False
                            code_lines.append(line)

                    # Prefix function definitions with ClassName:: (for .cpp files, not class wrapping)
                    import re

                    processed_lines = []
                    for line in code_lines:
                        # Detect function definitions (return_type function_name(...))
                        if (
                            line
                            and not line.strip().startswith("//")
                            and not line.strip().startswith("/*")
                        ):
                            # Check if it's a function definition
                            # Pattern: type name(...) or type* name(...) etc.
                            func_pattern = r"^(\s*)((?:const\s+)?(?:unsigned\s+)?(?:struct\s+)?[\w_]+(?:\s*\*)*\s+)([\w_]+)(\s*\()"
                            match = re.match(func_pattern, line)
                            if (
                                match and "::" not in line
                            ):  # Don't add if already qualified
                                indent = match.group(1)
                                return_type = match.group(2)
                                func_name = match.group(3)
                                rest = line[match.end(3) :]
                                # Prefix function name with class name
                                line = f"{indent}{return_type}{model_name}::{func_name}{rest}"
                        processed_lines.append(line)
                    code_lines = processed_lines

                    # Don't wrap in class for .cpp files - just output the definitions
                    includes_str = "\n".join(include_lines)
                    code_str = "\n".join(code_lines)

                    # For .cpp files, no class wrapper needed
                    cpp_content = f"{includes_str}\n\n{code_str}\n"

                    # Write to .cpp file
                    with open(cpp_file, "w") as f:
                        f.write(cpp_content)

                    # Remove original .c file
                    os.remove(c_file)

                # Update paths for further processing
                main_path = main_path[:-2] + ".cpp"
                quantize_path = quantize_path[:-2] + ".cpp"

                print(f"  Renamed {len(c_files)} .c files to .cpp")
            except Exception as e:
                print(f"  Warning: Failed to rename .c files: {e}")

        # Optimize main.cpp by removing unnecessary missing data checks
        # Since all features are always provided, replace !(data[X].missing != -1) with false
        if os.path.exists(main_path):
            try:
                with open(main_path, "r") as f:
                    content = f.read()

                # Replace pattern !(data[N].missing != -1) with false
                import re

                original_content = content
                content = re.sub(
                    r"!\(data\[\d+\]\.missing != -1\)", "false", content
                )

                # If log_transform is used, modify postprocess to apply exp/offset
                if log_transform:
                    offset_literal = f"{log_transform_offset:.12g}"
                    if log_transform_offset != 0:
                        offset_expr = f" - ({offset_literal})"
                    else:
                        offset_expr = ""
                    # Add <cmath> include if not present
                    if (
                        "#include <cmath>" not in content
                        and "#include<cmath>" not in content
                    ):
                        # Find the last #include and add after it
                        include_match = None
                        for match in re.finditer(
                            r'#include\s*[<"].*?[>"]', content
                        ):
                            include_match = match
                        if include_match:
                            insert_pos = include_match.end()
                            content = (
                                content[:insert_pos]
                                + "\n#include <cmath>"
                                + content[insert_pos:]
                            )

                    postprocess_pattern = re.compile(
                        r"void\s+(\w+)::postprocess\s*\(\s*double\*\s*result\s*\)\s*\{\n.*?\n\}",
                        re.DOTALL,
                    )

                    def replace_postprocess(match: re.Match) -> str:
                        class_name = match.group(1)
                        return (
                            f"void {class_name}::postprocess(double* result)\n"
                            "{\n"
                            "  for (int i = 0; i < N_TARGET; ++i) {\n"
                            f"    result[i] = std::exp(result[i]){offset_expr};\n"
                            "  }\n"
                            "}"
                        )

                    updated_content = postprocess_pattern.sub(
                        replace_postprocess, content, count=1
                    )

                    if updated_content != content:
                        content = updated_content
                        print(
                            "  Updated postprocess to convert log-space predictions to original space"
                        )
                    else:
                        print(
                            "  Warning: Failed to update postprocess for log-transform"
                        )

                if content != original_content:
                    with open(main_path, "w") as f:
                        f.write(content)
                    if not log_transform:
                        print(
                            "  Optimized main.cpp by removing unnecessary missing data checks"
                        )
            except Exception as e:
                print(f"  Warning: Failed to optimize main.cpp: {e}")

        # Wrap header.h content in class with #pragma once
        defines_to_move = []
        if model_name and os.path.exists(header_path):
            try:
                with open(header_path, "r") as f:
                    content = f.read()

                # Split content into includes, defines to move, and rest
                lines = content.split("\n")
                include_lines = []
                code_lines = []
                in_includes = True
                i = 0

                while i < len(lines):
                    line = lines[i]

                    if in_includes and (
                        line.strip().startswith("#include")
                        or line.strip() == ""
                    ):
                        include_lines.append(line)
                        i += 1
                    # Detect macros to move to main.cpp
                    elif (
                        line.strip().startswith("#if defined(__clang__)")
                        or line.strip().startswith("#define N_TARGET")
                        or line.strip().startswith("#define MAX_N_CLASS")
                    ):
                        in_includes = False
                        # Capture the entire #if block or single #define
                        if line.strip().startswith("#if defined(__clang__)"):
                            # Capture the entire #if...#endif block
                            macro_block = []
                            macro_block.append(line)
                            i += 1
                            while i < len(lines) and not lines[
                                i
                            ].strip().startswith("#endif"):
                                macro_block.append(lines[i])
                                i += 1
                            if i < len(lines):
                                macro_block.append(lines[i])  # Include #endif
                                i += 1
                            defines_to_move.append("\n".join(macro_block))
                        else:
                            # Single #define line
                            defines_to_move.append(line)
                            i += 1
                    else:
                        in_includes = False
                        code_lines.append(line)
                        i += 1

                # Add static keyword to function declarations
                import re

                processed_lines = []
                for line in code_lines:
                    # Detect function declarations/definitions (return_type function_name(...))
                    # Match lines that look like function declarations but don't already have static
                    if (
                        line
                        and not line.strip().startswith("//")
                        and not line.strip().startswith("/*")
                    ):
                        # Check if it's a function declaration/definition
                        # Pattern: type name(...) or type* name(...) or type name[...](...) etc.
                        func_pattern = r"^(\s*)((?:const\s+)?(?:unsigned\s+)?(?:struct\s+)?[\w_]+(?:\s*\*)*\s+)([\w_]+)\s*\("
                        match = re.match(func_pattern, line)
                        if match and "static" not in line:
                            indent = match.group(1)
                            return_type = match.group(2)
                            # Add static keyword
                            line = f"{indent}static {return_type}{line[len(indent) + len(return_type) :]}"
                    processed_lines.append(line)
                code_lines = processed_lines

                # Wrap code in class declaration
                includes_str = "\n".join(include_lines)
                code_str = "\n".join(code_lines)

                wrapped_content = f"#pragma once\n\n{includes_str}\n\nclass {model_name} {{\npublic:\n{code_str}\n}};  // class {model_name}\n"

                with open(header_path, "w") as f:
                    f.write(wrapped_content)

                print(
                    f"  Wrapped header.h in class '{model_name}' with #pragma once"
                )
            except Exception as e:
                print(f"  Warning: Failed to wrap header.h: {e}")

        # Add defines to main.cpp (moved from header.h)
        if defines_to_move and os.path.exists(main_path):
            try:
                with open(main_path, "r") as f:
                    content = f.read()

                # Insert defines after includes (look for where code starts - typically after blank line after includes)
                defines_str = "\n".join(defines_to_move)

                # Find the first non-include, non-blank line to insert before
                lines = content.split("\n")
                insert_pos = 0
                for i, line in enumerate(lines):
                    if line.strip() and not line.strip().startswith(
                        "#include"
                    ):
                        insert_pos = i
                        break

                # Insert defines at the position
                lines.insert(insert_pos, defines_str)
                lines.insert(
                    insert_pos + 1, ""
                )  # Add blank line after defines

                content = "\n".join(lines)
                with open(main_path, "w") as f:
                    f.write(content)
                print(
                    f"  Moved {len(defines_to_move)} macro definition(s) from header.h to main.cpp"
                )
            except Exception as e:
                print(f"  Warning: Failed to add defines to main.cpp: {e}")

        # Add feature names to header and implementation
        if (
            feature_names
            and os.path.exists(header_path)
            and os.path.exists(main_path)
        ):
            try:
                # Append to header.h (inside class)
                with open(header_path, "r") as f:
                    content = f.read()

                # Insert before closing class
                insertion = f"\n    // Feature names\n    static constexpr int NUM_FEATURES = {len(feature_names)};\n    static const char* feature_names[NUM_FEATURES];\n"
                content = content.replace(
                    f"}};  // class {model_name}\n",
                    f"{insertion}}};  // class {model_name}\n",
                )

                with open(header_path, "w") as f:
                    f.write(content)

                # Append to main.cpp (at the end of the file, outside any class)
                with open(main_path, "r") as f:
                    content = f.read()

                # Append feature array definition at the end of the file
                feature_array = f"\n// Feature names array\nconst char* {model_name}::feature_names[{model_name}::NUM_FEATURES] = {{\n"
                for i, name in enumerate(feature_names):
                    comma = "," if i < len(feature_names) - 1 else ""
                    feature_array += f'    "{name}"{comma}\n'
                feature_array += "};\n"

                # Append to end of file
                content = content.rstrip() + "\n" + feature_array

                with open(main_path, "w") as f:
                    f.write(content)

                print(
                    f"  Added {len(feature_names)} feature names to header.h and main.cpp"
                )
            except Exception as e:
                print(f"  Warning: Failed to add feature names: {e}")

        # Remove recipe.json if it exists
        if os.path.exists(recipe_path):
            try:
                os.remove(recipe_path)
                print("  Removed recipe.json")
            except Exception as e:
                print(f"  Warning: Failed to remove recipe.json: {e}")

        opt_msg = []
        if annotation_path:
            opt_msg.append("branch-annotated")
        if quantize:
            opt_msg.append("quantized")
        opt_suffix = f" ({', '.join(opt_msg)})" if opt_msg else ""

        print(f"C source code generated to: {source_dir}/")
        print(
            f"  Contains optimized model source code{opt_suffix} ready for compilation"
        )
    except Exception as e:
        print(f"Warning: TL2cgen code generation failed: {e}")
        print("  Model saved in standard format only.")


def save_model(
    model,
    scaler,
    regressor_type: str,
    output_dir: str,
    feature_names: List[str],
    log_transform: bool = False,
    log_transform_offset: float = 0.0,
) -> None:
    """Save trained model and preprocessing components to disk."""
    os.makedirs(output_dir, exist_ok=True)

    # Save metadata
    metadata = {
        "regressor_type": regressor_type,
        "feature_names": feature_names,
        "has_scaler": scaler is not None,
        "log_transform": log_transform,
        "log_transform_offset": log_transform_offset,
    }

    metadata_path = os.path.join(output_dir, f"{regressor_type}_metadata.pkl")
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"\nSaved metadata to: {metadata_path}")

    # Save scaler if exists
    if scaler is not None:
        scaler_path = os.path.join(output_dir, f"{regressor_type}_scaler.pkl")
        joblib.dump(scaler, scaler_path)
        print(f"Saved scaler to: {scaler_path}")

    # Save model
    if regressor_type == "xgboost":
        # Save as UBJ with gzip compression
        model_path = os.path.join(output_dir, f"{regressor_type}_model.ubj")
        model.save_model(model_path)

        # Gzip the file
        import gzip
        import shutil

        with open(model_path, "rb") as f_in:
            with gzip.open(model_path + ".gz", "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(model_path)  # Remove uncompressed version
        print(f"Saved model to: {model_path}.gz")
    elif regressor_type == "lightgbm":
        # Save LightGBM model as text file
        model_path = os.path.join(output_dir, f"{regressor_type}_model.txt")
        model.booster_.save_model(model_path)
        print(f"Saved model to: {model_path}")
    else:
        # Save sklearn models as joblib (more efficient than pickle for large arrays)
        model_path = os.path.join(output_dir, f"{regressor_type}_model.joblib")
        joblib.dump(model, model_path)
        print(f"Saved model to: {model_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train regression models to predict algorithm iterations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available regressors:
  linear            - Linear Regression
  poly2, poly3, poly4 - Polynomial Regression (degree 2, 3, 4)
  xgboost           - XGBoost Regressor
  lightgbm          - LightGBM Regressor
  random_forest     - Random Forest Regressor
  gradient_boosting - Gradient Boosting Regressor

Examples:
  # Train a model (default target: iter)
  python train_regressor.py data.feather --regressor xgboost --seed 42
  python train_regressor.py data.feather --regressor lightgbm --seed 42

  # Train to predict time instead of iterations
  python train_regressor.py data.feather --regressor xgboost --target time_ms --seed 42

  # Check data quality before training
  python train_regressor.py data.feather --regressor xgboost --check-data

  # Train with automatic removal of invalid rows
  python train_regressor.py data.feather --regressor xgboost --drop-invalid-rows --seed 42

  # Stratify train/test split by target column (ensures balanced distribution)
  python train_regressor.py data.feather --regressor xgboost --stratify-split --seed 42

  # Stratify split by a specific column (e.g., time_ms)
  python train_regressor.py data.feather --regressor xgboost --stratify-split time_ms --seed 42

  # Optimize for relative error (recommended for targets spanning multiple orders of magnitude)
  python train_regressor.py data.feather --regressor xgboost --log-transform --seed 42

  # Legacy pickle format
  python train_regressor.py data.pkl --regressor xgboost --seed 42
        """,
    )

    parser.add_argument(
        "input_pkl", help="Input data file (.feather or .pkl) with log data"
    )
    parser.add_argument(
        "--regressor",
        "-r",
        required=True,
        choices=AVAILABLE_REGRESSORS,
        help="Type of regressor to train",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="./models",
        help="Output directory for saved models (default: ./models)",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        help="Random seed for reproducibility (optional)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable hyperparameter tuning (uses predefined tuned parameters)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of files to use for testing (default: 0.2)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable training progress output",
    )
    parser.add_argument(
        "--list-features",
        action="store_true",
        help="List all available features in the dataset and exit",
    )
    parser.add_argument(
        "--stratify-split",
        type=str,
        nargs="?",
        const="__target__",
        default=None,
        metavar="COLUMN",
        help="Stratify train/test split by specified column distribution. If no column specified, uses target column. Example: --stratify-split time_ms or just --stratify-split for target column",
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=0,
        metavar="N",
        help="Enable early stopping for tree models (default: 20 rounds, use 0 to disable)",
    )
    parser.add_argument(
        "--treelite-compile",
        type=int,
        default=1,
        metavar="THREADS",
        help="Export XGBoost/LightGBM model as optimized C source code with TL2cgen (includes branch annotation and quantization)",
    )
    parser.add_argument(
        "--drop-invalid-rows",
        action="store_true",
        help="Drop rows with NaN or infinite values in target variable (instead of failing)",
    )
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Run data quality checks and exit (no training)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="iter",
        help="Target column to predict (default: iter). Examples: iter, time_ms, iterations",
    )
    parser.add_argument(
        "--log-transform",
        action="store_true",
        help="Use log-transform on target variable to optimize for relative error instead of absolute error. Recommended when target values span multiple orders of magnitude.",
    )
    parser.add_argument(
        "--log-transform-offset",
        type=float,
        default=0.0,
        metavar="C",
        help="Add constant C to target values before log transform (useful to avoid non-positive values). Only used with --log-transform.",
    )

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")

    # Load data
    print(f"\nLoading data from: {args.input_pkl}")
    print(f"Target column: '{args.target}'")
    df = load_data(args.input_pkl, target_col=args.target)
    print(f"Loaded {len(df)} entries with {len(df.columns)} columns")

    # Report file format
    ext = os.path.splitext(args.input_pkl)[1].lower()
    if ext == ".feather":
        print("  Format: Apache Arrow/Feather (fast I/O)")
    elif ext == ".pkl":
        print("  Format: Pickle (legacy)")

    # Extract model name from input file (for prefixing generated C functions)
    model_name = os.path.splitext(os.path.basename(args.input_pkl))[0]

    # Validate data quality
    print("\nValidating data quality...")
    is_valid, report = validate_data_quality(
        df, target_col=args.target, verbose=True
    )

    # If just checking data, exit here
    if args.check_data:
        if is_valid:
            print("\n✅ Data quality check passed! Ready for training.")
            return 0
        else:
            print(
                "\n❌ Data quality check failed! Fix issues before training."
            )
            return 1

    # Handle invalid data
    if not is_valid:
        if args.drop_invalid_rows:
            print(
                f"\nDropping {len(report['rows_with_issues'])} rows with invalid data..."
            )
            df_clean = df.drop(index=report["rows_with_issues"])
            df = df_clean.reset_index(drop=True)
            print(f"  Remaining: {len(df)} entries")

            # Re-validate
            is_valid_after, _ = validate_data_quality(df, verbose=False)
            if not is_valid_after:
                print(
                    "❌ Error: Data still invalid after dropping rows. Check your data."
                )
                return 1
            print("  ✅ Data is now valid")
        else:
            print("\n❌ Training aborted due to invalid data.")
            print(
                "   Use --drop-invalid-rows to automatically remove invalid rows, or"
            )
            print(
                "   Use --check-data to just run validation without training"
            )
            return 1

    # If listing features, do that and exit
    if args.list_features:
        features = list_available_features(df, target_col=args.target)

        # Also list potential target columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        potential_targets = [
            col for col in numeric_cols if col not in features
        ]

        print(f"\n{'=' * 70}")
        print(f"Available features in dataset ({len(features)} total):")
        print(f"{'=' * 70}")
        for i, feat in enumerate(features, 1):
            print(f"  {i:3d}. {feat}")

        if potential_targets:
            print(f"\n{'=' * 70}")
            print(
                f"Potential target columns ({len(potential_targets)} total):"
            )
            print(f"{'=' * 70}")
            for i, col in enumerate(potential_targets, 1):
                marker = " (current)" if col == args.target else ""
                print(f"  {i:3d}. {col}{marker}")

        print("\nTo exclude features, edit FEATURES_TO_EXCLUDE in the script:")
        print(f"  {__file__}")
        print("\nTo use only specific features, edit FEATURES_TO_INCLUDE_ONLY")
        print("\nTo change target column, use: --target <column_name>")
        return

    # Split data by files
    # Handle stratify-split argument
    if args.stratify_split is None:
        stratify_by = None
    elif args.stratify_split == "__target__":
        stratify_by = args.target
        print(f"Stratifying split by target column: '{args.target}'")
    else:
        stratify_by = args.stratify_split
        if stratify_by not in df.columns:
            print(
                f"\n❌ Error: Stratify column '{stratify_by}' not found in dataset"
            )
            print(f"Available columns: {list(df.columns)}")
            return 1
        print(f"Stratifying split by column: '{stratify_by}'")

    train_df, test_df = split_by_files(
        df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify_by=stratify_by,
    )

    # Prepare features
    X_train, y_train, feature_names = prepare_features(
        train_df, target_col=args.target
    )
    X_test, y_test, _ = prepare_features(test_df, target_col=args.target)

    print(f"\nFeatures: {len(feature_names)}")
    print(f"Target: {args.target} (prediction target)")

    # Apply log transform if requested (for relative error optimization)
    if args.log_transform:
        log_transform_offset = args.log_transform_offset
        y_train_original = y_train.copy()
        y_test_original = y_test.copy()

        if log_transform_offset != 0.0:
            print(
                f"  Applying log-transform offset: {log_transform_offset:+.6g}"
            )
            y_train = y_train + log_transform_offset
            y_test = y_test + log_transform_offset

        # Check for non-positive values before log transform
        if np.any(y_train <= 0) or np.any(y_test <= 0):
            n_nonpositive_train = np.sum(y_train <= 0)
            n_nonpositive_test = np.sum(y_test <= 0)
            print(
                "\n❌ Error: Cannot apply log-transform with non-positive target values!"
            )
            print(f"   Train set: {n_nonpositive_train} non-positive values")
            print(f"   Test set:  {n_nonpositive_test} non-positive values")
            print(
                f"   Target range: [{np.min(y_train):.2f}, {np.max(y_train):.2f}]"
            )
            print(
                "\nSuggestion: Use --log-transform-offset to add a small constant before log"
            )
            return 1

        print(
            "\nApplying log-transform to target variable (optimizes for relative error)"
        )
        y_train = np.log(y_train)
        y_test = np.log(y_test)

        print(
            f"  Original target range: [{np.min(y_train_original):.2f}, {np.max(y_train_original):.2f}]"
        )
        print(
            f"  Log-space target range: [{np.min(y_train):.4f}, {np.max(y_train):.4f}]"
        )
    else:
        y_train_original = None
        y_test_original = None
        log_transform_offset = 0.0

    # Enhanced diagnostics for XGBoost compatibility (only show if problems found)
    X_train_array = X_train.values if hasattr(X_train, "values") else X_train

    # XGBoost internal limits (approximate)
    xgb_max_safe = 1e38  # XGBoost uses float32 internally

    problematic_features = []
    problem_details = []

    for i, col_name in enumerate(feature_names):
        col_data = X_train_array[:, i]

        n_nan = np.sum(np.isnan(col_data))
        n_inf = np.sum(np.isinf(col_data))
        n_posinf = np.sum(np.isposinf(col_data))
        n_neginf = np.sum(np.isneginf(col_data))

        # Get statistics on finite values
        finite_mask = np.isfinite(col_data)
        if np.any(finite_mask):
            finite_data = col_data[finite_mask]
            col_min = np.min(finite_data)
            col_max = np.max(finite_data)
            col_mean = np.mean(finite_data)
            col_std = np.std(finite_data)
            abs_max = max(abs(col_min), abs(col_max))
        else:
            col_min = col_max = col_mean = col_std = abs_max = np.nan

        # Check if values are too large for XGBoost
        is_problematic = (
            n_nan > 0
            or n_inf > 0
            or (not np.isnan(abs_max) and abs_max > xgb_max_safe)
        )

        if is_problematic:
            problematic_features.append(col_name)
            detail = f"\n⚠️  '{col_name}':"
            if n_nan > 0:
                detail += f"\n    NaN:       {n_nan:8d} ({100 * n_nan / len(col_data):6.2f}%)"
            if n_posinf > 0:
                detail += f"\n    +Inf:      {n_posinf:8d} ({100 * n_posinf / len(col_data):6.2f}%)"
            if n_neginf > 0:
                detail += f"\n    -Inf:      {n_neginf:8d} ({100 * n_neginf / len(col_data):6.2f}%)"
            if not np.isnan(abs_max):
                detail += f"\n    Range:     [{col_min:.6e}, {col_max:.6e}]"
                detail += f"\n    Max abs:   {abs_max:.6e}"
                if abs_max > xgb_max_safe:
                    detail += f"\n    ❌ TOO LARGE! Exceeds XGBoost safe limit (~{xgb_max_safe:.2e})"
                detail += f"\n    Mean:      {col_mean:.6e}"
                detail += f"\n    Std:       {col_std:.6e}"
            problem_details.append(detail)

    # Check target variable
    n_nan_target = np.sum(np.isnan(y_train))
    n_inf_target = np.sum(np.isinf(y_train))
    if n_nan_target > 0 or n_inf_target > 0:
        problematic_features.append(f"TARGET[{args.target}]")
        detail = f"\n⚠️  Target '{args.target}':"
        if n_nan_target > 0:
            detail += f"\n    NaN:  {n_nan_target} ({100 * n_nan_target / len(y_train):.2f}%)"
        if n_inf_target > 0:
            detail += f"\n    Inf:  {n_inf_target} ({100 * n_inf_target / len(y_train):.2f}%)"
        problem_details.append(detail)

    # Only print if problems found
    if len(problematic_features) > 0:
        print("\n" + "=" * 70)
        print("⚠️  FEATURE VALUE PROBLEMS DETECTED")
        print("=" * 70)
        for detail in problem_details:
            print(detail)
        print("\n" + "=" * 70)
        print(f"❌ Found {len(problematic_features)} problematic feature(s):")
        for feat in problematic_features:
            print(f"   - {feat}")
        print("\nTo fix:")
        print(
            "  1. Add these features to FEATURES_TO_EXCLUDE at top of script"
        )
        print(
            "  2. Or investigate why these features have extreme/invalid values"
        )
        print("=" * 70 + "\n")

    # Create model
    print(f"\nTraining {args.regressor} regressor...")
    model, needs_scaling = create_regressor(
        args.regressor,
        random_state=args.seed,
        tune_hyperparams=args.tune,
        verbose=not args.no_progress,
    )

    # Apply scaling if needed
    scaler = None
    X_test_original = X_test.copy()  # Keep unscaled version for display
    if needs_scaling:
        print("  Applying StandardScaler to features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Sanity check: verify scaling worked correctly
        print(
            f"    Scaled features - mean: {np.mean(X_train_scaled):.6f}, std: {np.std(X_train_scaled):.6f}"
        )
        if np.any(np.isnan(X_train_scaled)) or np.any(
            np.isinf(X_train_scaled)
        ):
            print("    WARNING: NaN or Inf detected in scaled training data!")
        if np.any(np.isnan(X_test_scaled)) or np.any(np.isinf(X_test_scaled)):
            print("    WARNING: NaN or Inf detected in scaled test data!")
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    # Train model
    if args.regressor.startswith("poly"):
        degree = int(args.regressor[-1])
        n_features = X_train_scaled.shape[1]
        from math import comb

        n_poly_features = sum(
            comb(n_features + d - 1, d) for d in range(1, degree + 1)
        )
        print(
            f"  Generating {n_poly_features} polynomial features (degree {degree})..."
        )

    # Use early stopping for tree-based models if requested
    if (
        args.early_stopping
        and args.early_stopping > 0
        and args.regressor in ["xgboost", "lightgbm", "gradient_boosting"]
    ):
        if args.regressor == "xgboost":
            print(
                f"  Using early stopping (patience={args.early_stopping} rounds)..."
            )
            # Set early stopping parameter
            model.set_params(early_stopping_rounds=args.early_stopping)
            model.fit(
                X_train_scaled,
                y_train,
                eval_set=[(X_test_scaled, y_test)],
                verbose=False,
            )
            # Report best iteration
            best_iteration = (
                model.best_iteration
                if hasattr(model, "best_iteration")
                else model.n_estimators
            )
            print(
                f"  Best iteration: {best_iteration} (out of {model.n_estimators} max)"
            )
        elif args.regressor == "lightgbm":
            print(
                f"  Using early stopping (patience={args.early_stopping} rounds)..."
            )
            model.fit(
                X_train_scaled,
                y_train,
                eval_set=[(X_test_scaled, y_test)],
                callbacks=[
                    __import__("lightgbm").early_stopping(
                        stopping_rounds=args.early_stopping, verbose=False
                    )
                ],
            )
            # Report best iteration
            best_iteration = (
                model.best_iteration_
                if hasattr(model, "best_iteration_")
                else model.n_estimators
            )
            print(
                f"  Best iteration: {best_iteration} (out of {model.n_estimators} max)"
            )
        else:  # gradient_boosting
            # Gradient Boosting uses n_iter_no_change parameter
            print(
                "  Note: Use --tune with gradient_boosting for early stopping"
            )
            model.fit(X_train_scaled, y_train)
    else:
        model.fit(X_train_scaled, y_train)

    print("  Training complete!")

    # Evaluate model
    skip_cv = args.early_stopping is not None and args.early_stopping > 0
    train_r2, test_r2 = evaluate_model(
        model,
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test,
        feature_names,
        args.regressor,
        cv_folds=args.cv_folds,
        verbose=2 if not args.no_progress else 0,
        skip_cv=skip_cv,
        X_test_original=X_test_original,
        test_df=test_df,
        log_transform=args.log_transform,
        y_train_original=y_train_original,
        y_test_original=y_test_original,
        log_transform_offset=log_transform_offset,
    )

    # Save model
    save_model(
        model,
        scaler,
        args.regressor,
        args.output_dir,
        feature_names,
        log_transform=args.log_transform,
        log_transform_offset=log_transform_offset,
    )

    # Compile with TL2cgen if requested (with optimizations enabled by default)
    if args.treelite_compile is not None:
        # Use unscaled training data for branch annotation when scaling is not applied
        # Note: All models now use scaling for consistency
        X_train_for_annotation = X_train if not needs_scaling else None

        compile_model_treelite(
            model,
            args.regressor,
            args.output_dir,
            args.treelite_compile,
            X_train=X_train_for_annotation,
            annotate=True,  # Always enable branch annotation
            quantize=True,  # Always enable quantization
            feature_names=feature_names,
            model_name=model_name,
            log_transform=args.log_transform,
            log_transform_offset=log_transform_offset,
        )

    print("\n" + "=" * 70)
    print("Training completed successfully!")
    print("=" * 70)
    print("\nFinal R² Scores:")
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²:  {test_r2:.4f}")


if __name__ == "__main__":
    main()
