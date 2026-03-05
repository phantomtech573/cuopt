/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/presolve.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <utilities/memory_instrumentation.hpp>

#include <raft/core/nvtx.hpp>

#include <cstdio>

namespace cuopt::linear_programming::dual_simplex {

/**
 * @brief Feature collection structure for dual simplex runtime prediction.
 *
 * This structure collects features that can be used to train regression models
 * for predicting the runtime of dual_phase2_with_advanced_basis.
 */
template <typename i_t, typename f_t>
struct dual_simplex_features_t {
  // Model/Problem Features (static)
  i_t num_rows{0};           // m - number of constraints
  i_t num_cols{0};           // n - number of variables
  i_t num_nonzeros{0};       // nnz - total nonzeros in constraint matrix
  f_t matrix_density{0.0};   // nnz / (m * n)
  f_t avg_nnz_per_col{0.0};  // nnz / n
  f_t avg_nnz_per_row{0.0};  // nnz / m
  i_t num_bounded_vars{0};   // variables with finite lower AND upper bounds
  i_t num_free_vars{0};      // variables with infinite bounds on both sides
  i_t num_fixed_vars{0};     // variables where lower == upper

  // Iteration-based features (dynamic)
  i_t iteration{0};             // current iteration count
  i_t start_iteration{0};       // iteration at start of this call
  i_t num_refactors{0};         // number of basis refactorizations
  i_t num_basis_updates{0};     // basis updates since last refactor
  i_t sparse_delta_z_count{0};  // iterations using sparse delta_z
  i_t dense_delta_z_count{0};   // iterations using dense delta_z
  i_t total_bound_flips{0};     // cumulative bound flips

  // Sparsity during solve
  i_t num_infeasibilities{0};      // size of infeasibility_indices
  f_t delta_y_nz_percentage{0.0};  // sparsity of BTran result

  // Phase-specific features
  i_t phase{0};                  // 1 or 2
  bool slack_basis{false};       // whether starting from slack basis
  bool initialize_basis{false};  // whether basis factorization performed initially

  // Settings that impact runtime
  i_t refactor_frequency{0};  // from settings

  // Memory access statistics (aggregated from instrumentation)
  size_t byte_loads{0};   // total bytes loaded
  size_t byte_stores{0};  // total bytes stored

  // Runtime for the interval (in seconds)
  f_t interval_runtime{0.0};

  /**
   * @brief Initialize static features from problem data.
   */
  void init_from_problem(const lp_problem_t<i_t, f_t>& lp,
                         const simplex_solver_settings_t<i_t, f_t>& settings,
                         i_t phase_,
                         bool slack_basis_,
                         bool initialize_basis_)
  {
    raft::common::nvtx::range scope("DualSimplex::init_from_problem");

    num_rows     = lp.num_rows;
    num_cols     = lp.num_cols;
    num_nonzeros = lp.A.col_start[lp.num_cols];

    const f_t total_elements = static_cast<f_t>(num_rows) * static_cast<f_t>(num_cols);
    matrix_density           = (total_elements > 0) ? num_nonzeros / total_elements : 0.0;
    avg_nnz_per_col          = (num_cols > 0) ? static_cast<f_t>(num_nonzeros) / num_cols : 0.0;
    avg_nnz_per_row          = (num_rows > 0) ? static_cast<f_t>(num_nonzeros) / num_rows : 0.0;

    // Count bound types
    num_bounded_vars      = 0;
    num_free_vars         = 0;
    num_fixed_vars        = 0;
    constexpr f_t inf_val = std::numeric_limits<f_t>::infinity();
    for (i_t j = 0; j < num_cols; ++j) {
      const bool has_lower = lp.lower[j] > -inf_val;
      const bool has_upper = lp.upper[j] < inf_val;
      if (has_lower && has_upper) {
        if (lp.lower[j] == lp.upper[j]) {
          num_fixed_vars++;
        } else {
          num_bounded_vars++;
        }
      } else if (!has_lower && !has_upper) {
        num_free_vars++;
      }
    }

    phase              = phase_;
    slack_basis        = slack_basis_;
    initialize_basis   = initialize_basis_;
    refactor_frequency = settings.refactor_frequency;
  }

  /**
   * @brief Print all features on a single line in key=value format.
   *
   * Format: DS_FEATURES: iter=N m=M n=N nnz=K ...
   */
  void log_features(const simplex_solver_settings_t<i_t, f_t>& settings) const
  {
    printf(
      "DS_FEATURES: iter=%d m=%d n=%d nnz=%d density=%.6e avg_nnz_col=%.2f avg_nnz_row=%.2f "
      "bounded=%d free=%d fixed=%d phase=%d refact_freq=%d num_refacts=%d num_updates=%d "
      "sparse_dz=%d dense_dz=%d bound_flips=%d num_infeas=%d dy_nz_pct=%.2f "
      "byte_loads=%zu byte_stores=%zu runtime=%.6f\n",
      iteration,
      num_rows,
      num_cols,
      num_nonzeros,
      matrix_density,
      avg_nnz_per_col,
      avg_nnz_per_row,
      num_bounded_vars,
      num_free_vars,
      num_fixed_vars,
      phase,
      refactor_frequency,
      num_refactors,
      num_basis_updates,
      sparse_delta_z_count,
      dense_delta_z_count,
      total_bound_flips,
      num_infeasibilities,
      delta_y_nz_percentage,
      byte_loads,
      byte_stores,
      interval_runtime);
  }

  /**
   * @brief Reset per-interval counters (called after each logging interval).
   */
  void reset_interval_counters()
  {
    byte_loads       = 0;
    byte_stores      = 0;
    interval_runtime = 0.0;
  }
};

// Feature logging interval (every N iterations)
constexpr int FEATURE_LOG_INTERVAL = 100;

// Node bounds strengthening features (for B&B)
template <typename i_t, typename f_t>
struct bounds_strengthening_features_t {
  i_t m{0};                   // number of constraints
  i_t n{0};                   // number of variables
  i_t nnz{0};                 // number of nonzeros in constraint matrix
  i_t num_iterations{0};      // propagation iterations until fixpoint
  i_t num_bounds_changed{0};  // total bounds tightened
  size_t byte_loads{0};
  size_t byte_stores{0};
  f_t runtime{0.0};

  // Interval aggregates (for when bounds strengthening is called multiple times)
  i_t call_count{0};
  i_t total_iterations{0};
  i_t total_bounds_changed{0};
  size_t total_byte_loads{0};
  size_t total_byte_stores{0};
  f_t total_runtime{0.0};

  void accumulate()
  {
    call_count++;
    total_iterations += num_iterations;
    total_bounds_changed += num_bounds_changed;
    total_byte_loads += byte_loads;
    total_byte_stores += byte_stores;
    total_runtime += runtime;
  }

  void log_single(i_t m_val, i_t n_val, i_t nnz_val) const
  {
    // printf(
    //   "BOUNDS_STRENGTH_FEATURES: m=%d n=%d nnz=%d "
    //   "iterations=%d bounds_changed=%d "
    //   "byte_loads=%zu byte_stores=%zu runtime=%.6f\n",
    //   m_val,
    //   n_val,
    //   nnz_val,
    //   num_iterations,
    //   num_bounds_changed,
    //   byte_loads,
    //   byte_stores,
    //   runtime);
  }

  void reset()
  {
    call_count           = 0;
    total_iterations     = 0;
    total_bounds_changed = 0;
    total_byte_loads     = 0;
    total_byte_stores    = 0;
    total_runtime        = 0.0;
  }
};

}  // namespace cuopt::linear_programming::dual_simplex
