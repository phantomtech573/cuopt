/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <mip_heuristics/mip_constants.hpp>
#include <mip_heuristics/mip_scaling_strategy.cuh>
#include <utilities/logger.hpp>

#include <raft/common/nvtx.hpp>
#include <raft/util/cudart_utils.hpp>

#include <thrust/functional.h>
#include <thrust/reduce.h>

#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace cuopt::linear_programming::detail {
namespace {

constexpr int scaling_threads                    = 256;
constexpr int row_scaling_num_iterations         = 3;
constexpr int row_scaling_k_min                  = -20;
constexpr int row_scaling_k_max                  = 20;
constexpr double nearest_pow2_mantissa_threshold = 0.7071067811865476;

constexpr double big_m_abs_threshold   = 1.0e4;
constexpr double big_m_ratio_threshold = 1.0e4;

inline int get_num_blocks(size_t n)
{
  if (n == 0) { return 1; }
  size_t blocks = (n + scaling_threads - 1) / scaling_threads;
  return static_cast<int>(std::min<size_t>(65535, std::max<size_t>(1, blocks)));
}

template <typename i_t, typename f_t>
__global__ void analyze_rows_for_scaling_kernel(
  const typename problem_t<i_t, f_t>::view_t op_problem, f_t* row_inf_norm, i_t* row_skip_scaling)
{
  for (i_t row = blockIdx.x * blockDim.x + threadIdx.x; row < op_problem.n_constraints;
       row += blockDim.x * gridDim.x) {
    const auto [row_start, row_end] = op_problem.range_for_constraint(row);
    f_t row_norm                    = f_t(0);
    f_t row_min_nonzero             = std::numeric_limits<f_t>::infinity();
    i_t row_nonzero_count           = 0;
    for (i_t idx = row_start; idx < row_end; ++idx) {
      f_t abs_value = raft::abs(op_problem.coefficients[idx]);
      if (abs_value > row_norm) { row_norm = abs_value; }
      if (abs_value > f_t(0)) {
        row_min_nonzero = abs_value < row_min_nonzero ? abs_value : row_min_nonzero;
        row_nonzero_count++;
      }
    }
    row_inf_norm[row] = row_norm;

    bool skip_big_m_scaling = false;
    if (row_nonzero_count >= 2 && row_min_nonzero < std::numeric_limits<f_t>::infinity()) {
      const f_t row_ratio = row_norm / row_min_nonzero;
      skip_big_m_scaling  = row_norm >= static_cast<f_t>(big_m_abs_threshold) &&
                           row_ratio >= static_cast<f_t>(big_m_ratio_threshold);
    }
    row_skip_scaling[row] = skip_big_m_scaling ? i_t(1) : i_t(0);
  }
}

template <typename i_t, typename f_t>
__global__ void compute_row_inf_norm_kernel(const typename problem_t<i_t, f_t>::view_t op_problem,
                                            f_t* row_inf_norm)
{
  for (i_t row = blockIdx.x * blockDim.x + threadIdx.x; row < op_problem.n_constraints;
       row += blockDim.x * gridDim.x) {
    const auto [row_start, row_end] = op_problem.range_for_constraint(row);
    f_t row_norm                    = f_t(0);
    for (i_t idx = row_start; idx < row_end; ++idx) {
      const f_t abs_value = raft::abs(op_problem.coefficients[idx]);
      if (abs_value > row_norm) { row_norm = abs_value; }
    }
    row_inf_norm[row] = row_norm;
  }
}

template <typename i_t, typename f_t>
__global__ void compute_row_log2_and_active_kernel(const f_t* row_inf_norm,
                                                   const i_t* row_skip_scaling,
                                                   i_t n_rows,
                                                   double* row_log2_values,
                                                   i_t* row_active)
{
  for (i_t row = blockIdx.x * blockDim.x + threadIdx.x; row < n_rows;
       row += blockDim.x * gridDim.x) {
    if (row_skip_scaling[row] || row_inf_norm[row] <= f_t(0)) {
      row_log2_values[row] = 0.0;
      row_active[row]      = i_t(0);
      continue;
    }
    row_log2_values[row] = ::log2(static_cast<double>(row_inf_norm[row]));
    row_active[row]      = i_t(1);
  }
}

template <typename i_t, typename f_t>
__global__ void compute_iteration_scaling_kernel(const f_t* row_inf_norm,
                                                 const i_t* row_skip_scaling,
                                                 i_t n_rows,
                                                 f_t target_norm,
                                                 f_t* iteration_scaling)
{
  for (i_t row = blockIdx.x * blockDim.x + threadIdx.x; row < n_rows;
       row += blockDim.x * gridDim.x) {
    if (row_skip_scaling[row] || row_inf_norm[row] <= f_t(0) || target_norm <= f_t(0)) {
      iteration_scaling[row] = f_t(1);
      continue;
    }
    const f_t desired_scaling = target_norm / row_inf_norm[row];
    if (!::isfinite(desired_scaling) || desired_scaling <= f_t(0)) {
      iteration_scaling[row] = f_t(1);
      continue;
    }

    int exponent = 0;
    f_t mantissa = ::frexp(desired_scaling, &exponent);  // desired_scaling = mantissa * 2^exponent
    int k = mantissa >= static_cast<f_t>(nearest_pow2_mantissa_threshold) ? exponent : exponent - 1;
    if (k < row_scaling_k_min) { k = row_scaling_k_min; }
    if (k > row_scaling_k_max) { k = row_scaling_k_max; }
    iteration_scaling[row] = ::ldexp(f_t(1), k);
  }
}

template <typename i_t, typename f_t>
__global__ void apply_row_scaling_kernel(const typename problem_t<i_t, f_t>::view_t op_problem,
                                         const f_t* constraint_scaling)
{
  for (i_t row = blockIdx.x * blockDim.x + threadIdx.x; row < op_problem.n_constraints;
       row += blockDim.x * gridDim.x) {
    const f_t scaling               = constraint_scaling[row];
    const auto [row_start, row_end] = op_problem.range_for_constraint(row);
    for (i_t idx = row_start; idx < row_end; ++idx) {
      op_problem.coefficients[idx] *= scaling;
    }
    op_problem.constraint_lower_bounds[row] *= scaling;
    op_problem.constraint_upper_bounds[row] *= scaling;
  }
}

template <typename i_t, typename f_t>
__global__ void apply_row_scaling_transpose_kernel(
  const typename problem_t<i_t, f_t>::view_t op_problem, const f_t* constraint_scaling)
{
  for (i_t var = blockIdx.x * blockDim.x + threadIdx.x; var < op_problem.n_variables;
       var += blockDim.x * gridDim.x) {
    const auto [start, end] = op_problem.reverse_range_for_var(var);
    for (i_t idx = start; idx < end; ++idx) {
      i_t row = op_problem.reverse_constraints[idx];
      op_problem.reverse_coefficients[idx] *= constraint_scaling[row];
    }
  }
}

}  // namespace

template <typename i_t, typename f_t>
mip_scaling_strategy_t<i_t, f_t>::mip_scaling_strategy_t(problem_t<i_t, f_t>& op_problem_scaled)
  : handle_ptr_(op_problem_scaled.handle_ptr),
    stream_view_(handle_ptr_->get_stream()),
    op_problem_scaled_(op_problem_scaled)
{
}

template <typename i_t, typename f_t>
void mip_scaling_strategy_t<i_t, f_t>::scale_problem()
{
  raft::common::nvtx::range fun_scope("mip_scale_problem");

  const i_t n_rows = op_problem_scaled_.n_constraints;
  const i_t n_cols = op_problem_scaled_.n_variables;

  if (n_rows == 0) {
    op_problem_scaled_.is_scaled_ = true;
    return;
  }

  auto problem_view        = op_problem_scaled_.view();
  int row_scaling_blocks   = get_num_blocks(static_cast<size_t>(n_rows));
  int transpose_num_blocks = get_num_blocks(static_cast<size_t>(n_cols));

  rmm::device_uvector<f_t> row_inf_norm(static_cast<size_t>(n_rows), stream_view_);
  rmm::device_uvector<i_t> row_skip_scaling(static_cast<size_t>(n_rows), stream_view_);
  rmm::device_uvector<f_t> iteration_scaling(static_cast<size_t>(n_rows), stream_view_);
  rmm::device_uvector<double> row_log2_values(static_cast<size_t>(n_rows), stream_view_);
  rmm::device_uvector<i_t> row_active(static_cast<size_t>(n_rows), stream_view_);

  analyze_rows_for_scaling_kernel<i_t, f_t>
    <<<row_scaling_blocks, scaling_threads, 0, stream_view_>>>(
      problem_view, row_inf_norm.data(), row_skip_scaling.data());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  i_t skipped_big_m_rows = thrust::reduce(handle_ptr_->get_thrust_policy(),
                                          row_skip_scaling.begin(),
                                          row_skip_scaling.end(),
                                          i_t(0),
                                          thrust::plus<i_t>());

  CUOPT_LOG_INFO("MIP row scaling start: rows=%d cols=%d iterations=%d skip_big_m_rows=%d",
                 n_rows,
                 n_cols,
                 row_scaling_num_iterations,
                 skipped_big_m_rows);

  for (int iteration = 0; iteration < row_scaling_num_iterations; ++iteration) {
    compute_row_inf_norm_kernel<i_t, f_t>
      <<<row_scaling_blocks, scaling_threads, 0, stream_view_>>>(problem_view, row_inf_norm.data());
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    compute_row_log2_and_active_kernel<i_t, f_t>
      <<<row_scaling_blocks, scaling_threads, 0, stream_view_>>>(row_inf_norm.data(),
                                                                 row_skip_scaling.data(),
                                                                 n_rows,
                                                                 row_log2_values.data(),
                                                                 row_active.data());
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    double log2_sum      = thrust::reduce(handle_ptr_->get_thrust_policy(),
                                     row_log2_values.begin(),
                                     row_log2_values.end(),
                                     0.0,
                                     thrust::plus<double>());
    i_t active_row_count = thrust::reduce(handle_ptr_->get_thrust_policy(),
                                          row_active.begin(),
                                          row_active.end(),
                                          i_t(0),
                                          thrust::plus<i_t>());
    if (active_row_count == 0) { break; }

    f_t target_norm = static_cast<f_t>(::exp2(log2_sum / static_cast<double>(active_row_count)));
    if (!::isfinite(target_norm) || target_norm <= f_t(0)) { break; }

    compute_iteration_scaling_kernel<i_t, f_t>
      <<<row_scaling_blocks, scaling_threads, 0, stream_view_>>>(row_inf_norm.data(),
                                                                 row_skip_scaling.data(),
                                                                 n_rows,
                                                                 target_norm,
                                                                 iteration_scaling.data());
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    apply_row_scaling_kernel<i_t, f_t><<<row_scaling_blocks, scaling_threads, 0, stream_view_>>>(
      problem_view, iteration_scaling.data());
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    if (n_cols > 0) {
      apply_row_scaling_transpose_kernel<i_t, f_t>
        <<<transpose_num_blocks, scaling_threads, 0, stream_view_>>>(problem_view,
                                                                     iteration_scaling.data());
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }
  }

  op_problem_scaled_.is_scaled_ = true;
  CUOPT_LOG_INFO("MIP row scaling completed");
}

#define INSTANTIATE(F_TYPE) template class mip_scaling_strategy_t<int, F_TYPE>;

#if MIP_INSTANTIATE_FLOAT
INSTANTIATE(float)
#endif

#if MIP_INSTANTIATE_DOUBLE
INSTANTIATE(double)
#endif

}  // namespace cuopt::linear_programming::detail
