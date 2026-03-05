/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cstdint>
#include <cstdio>
#include <limits>

namespace cuopt::linear_programming::detail {

// Feature logging interval (every N iterations)
constexpr int PDLP_FEATURE_LOG_INTERVAL = 50;

/**
 * @brief Feature collection structure for PDLP Stable3 runtime prediction.
 *
 * This structure collects features that can be used to train regression models
 * for predicting the runtime of PDLP iterations in Stable3 mode.
 *
 * Stable3 key characteristics:
 * - No adaptive step size retries (deterministic SpMV count)
 * - Uses reflected primal/dual (Halpern update)
 * - Uses fixed point error computation on major iterations
 * - Uses conditional major iterations (frequency increases with iteration count)
 * - Never restarts to average (only to current solution)
 */
template <typename i_t, typename f_t>
struct pdlp_features_t {
  // =========================================================================
  // Problem Features (static, set once at initialization)
  // =========================================================================
  i_t n_vars{0};            // Number of variables
  i_t n_cstrs{0};           // Number of constraints
  int64_t nnz{0};           // Number of nonzeros in constraint matrix
  f_t sparsity{0.0};        // nnz / (n_vars * n_cstrs)
  f_t nnz_stddev{0.0};      // Standard deviation of row nnz counts
  f_t unbalancedness{0.0};  // nnz_stddev / mean_nnz_per_row

  // =========================================================================
  // Interval Counters (reset after each log)
  // =========================================================================
  i_t interval_iters{0};     // Iterations in this logging interval
  i_t interval_major{0};     // Major iterations in this interval
  i_t interval_restarts{0};  // Restarts in this interval

  // =========================================================================
  // SpMV Metrics
  // =========================================================================
  // In Stable3: regular iter = 2 SpMV, major iter = 3 SpMV (fixed point error)
  int64_t interval_spmv_ops{0};  // SpMV operations in this interval

  // =========================================================================
  // Cumulative Counters (never reset)
  // =========================================================================
  i_t total_iters{0};          // Total iterations since solver start
  i_t total_restarts{0};       // Total restarts since solver start
  i_t iters_since_restart{0};  // Iterations since last restart

  // =========================================================================
  // Convergence Metrics (snapshot at log time, from last major iteration)
  // =========================================================================
  f_t primal_res{0.0};  // L2 primal residual
  f_t dual_res{0.0};    // L2 dual residual
  f_t gap{0.0};         // Duality gap
  f_t kkt_score{0.0};   // KKT score (used for restart decisions)

  // =========================================================================
  // Step Parameters (can change on restarts)
  // =========================================================================
  f_t step_size{0.0};      // Current step size
  f_t primal_weight{0.0};  // Current primal weight

  // =========================================================================
  // Timing
  // =========================================================================
  f_t interval_time_ms{0.0};  // Time elapsed in this interval (milliseconds)

  // =========================================================================
  // Warm Start Info
  // =========================================================================
  bool has_warm_start{false};  // Whether warm start was provided

  /**
   * @brief Initialize static problem features.
   *
   * Called once at solver initialization. Computes sparsity metrics from
   * the problem structure.
   */
  void init_from_problem(i_t n_variables,
                         i_t n_constraints,
                         int64_t num_nonzeros,
                         f_t computed_sparsity,
                         f_t computed_nnz_stddev,
                         f_t computed_unbalancedness,
                         bool warm_start)
  {
    n_vars         = n_variables;
    n_cstrs        = n_constraints;
    nnz            = num_nonzeros;
    sparsity       = computed_sparsity;
    nnz_stddev     = computed_nnz_stddev;
    unbalancedness = computed_unbalancedness;
    has_warm_start = warm_start;
  }

  /**
   * @brief Record a regular iteration.
   *
   * In Stable3, each regular iteration does 2 SpMV operations:
   * - compute_At_y(): A^T @ y
   * - compute_A_x(): A @ x
   */
  void record_regular_iteration()
  {
    ++interval_iters;
    ++total_iters;
    ++iters_since_restart;
    interval_spmv_ops += 2;
  }

  /**
   * @brief Record a major iteration.
   *
   * Major iterations do an additional SpMV for fixed point error computation.
   * They also involve termination checks and potential restarts.
   */
  void record_major_iteration()
  {
    ++interval_major;
    // Fixed point error computation adds 1 SpMV
    interval_spmv_ops += 1;
  }

  /**
   * @brief Record a restart event.
   */
  void record_restart()
  {
    ++interval_restarts;
    ++total_restarts;
    iters_since_restart = 0;
  }

  /**
   * @brief Update convergence metrics from termination strategy.
   *
   * Called during major iterations when convergence info is computed.
   */
  void update_convergence(f_t l2_primal_residual,
                          f_t l2_dual_residual,
                          f_t duality_gap,
                          f_t computed_kkt_score)
  {
    primal_res = l2_primal_residual;
    dual_res   = l2_dual_residual;
    gap        = duality_gap;
    kkt_score  = computed_kkt_score;
  }

  /**
   * @brief Update step parameters.
   *
   * Called when step size or primal weight changes (typically after restarts).
   */
  void update_step_params(f_t current_step_size, f_t current_primal_weight)
  {
    step_size     = current_step_size;
    primal_weight = current_primal_weight;
  }

  /**
   * @brief Log all features in key=value format.
   *
   * Format: PDLP_RESULT: n_vars=N n_cstrs=M nnz=K ...
   *
   * This format is parsed by determinism_logs_parse.py with --algorithm PDLP
   */
  void log_features() const
  {
    // Compute derived metrics
    const int64_t total_nnz_processed = interval_spmv_ops * nnz;
    const double nnz_per_sec = (interval_time_ms > 0) ? static_cast<double>(total_nnz_processed) /
                                                          (interval_time_ms / 1000.0)
                                                      : 0.0;

    // printf(
    //   "PDLP_RESULT: n_vars=%d n_cstrs=%d nnz=%ld sparsity=%.6e nnz_stddev=%.4f "
    //   "unbalancedness=%.4f interval_iters=%d interval_major=%d interval_restarts=%d "
    //   "spmv_ops=%ld total_iters=%d total_restarts=%d iters_since_restart=%d "
    //   "primal_res=%.6e dual_res=%.6e gap=%.6e kkt=%.6e "
    //   "step_size=%.6e primal_weight=%.6e time_ms=%.2f nnz_per_s=%.2e warm_start=%d",
    //   n_vars,
    //   n_cstrs,
    //   nnz,
    //   sparsity,
    //   nnz_stddev,
    //   unbalancedness,
    //   interval_iters,
    //   interval_major,
    //   interval_restarts,
    //   interval_spmv_ops,
    //   total_iters,
    //   total_restarts,
    //   iters_since_restart,
    //   primal_res,
    //   dual_res,
    //   gap,
    //   kkt_score,
    //   step_size,
    //   primal_weight,
    //   interval_time_ms,
    //   nnz_per_sec,
    //   static_cast<int>(has_warm_start));
  }

  /**
   * @brief Reset per-interval counters.
   *
   * Called after each log to start fresh for the next interval.
   */
  void reset_interval_counters()
  {
    interval_iters    = 0;
    interval_major    = 0;
    interval_restarts = 0;
    interval_spmv_ops = 0;
    interval_time_ms  = 0.0;
  }

  /**
   * @brief Check if it's time to log features.
   *
   * Returns true every PDLP_FEATURE_LOG_INTERVAL iterations.
   */
  bool should_log() const { return (interval_iters >= PDLP_FEATURE_LOG_INTERVAL); }
};

}  // namespace cuopt::linear_programming::detail
