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

#include <cub/cub.cuh>

#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace cuopt::linear_programming::detail {

constexpr int row_scaling_num_iterations         = 3;
constexpr int row_scaling_k_min                  = -20;
constexpr int row_scaling_k_max                  = 20;
constexpr double nearest_pow2_mantissa_threshold = 0.7071067811865476;

constexpr double big_m_abs_threshold   = 1.0e4;
constexpr double big_m_ratio_threshold = 1.0e4;

template <typename f_t>
struct abs_value_transform_t {
  __device__ f_t operator()(f_t value) const { return raft::abs(value); }
};

template <typename f_t>
struct nonzero_abs_or_inf_transform_t {
  __device__ f_t operator()(f_t value) const
  {
    const f_t abs_value = raft::abs(value);
    return abs_value > f_t(0) ? abs_value : std::numeric_limits<f_t>::infinity();
  }
};

template <typename i_t, typename f_t>
struct nonzero_count_transform_t {
  __device__ i_t operator()(f_t value) const { return raft::abs(value) > f_t(0) ? i_t(1) : i_t(0); }
};

template <typename t_t>
struct max_op_t {
  __host__ __device__ t_t operator()(const t_t& lhs, const t_t& rhs) const
  {
    return lhs > rhs ? lhs : rhs;
  }
};

template <typename t_t>
struct min_op_t {
  __host__ __device__ t_t operator()(const t_t& lhs, const t_t& rhs) const
  {
    return lhs < rhs ? lhs : rhs;
  }
};

template <typename i_t, typename f_t>
void compute_row_inf_norm(const problem_t<i_t, f_t>& op_problem,
                          rmm::device_uvector<std::uint8_t>& temp_storage,
                          size_t temp_storage_bytes,
                          rmm::device_uvector<f_t>& row_inf_norm,
                          rmm::cuda_stream_view stream_view)
{
  auto coeff_abs_iter =
    thrust::make_transform_iterator(op_problem.coefficients.data(), abs_value_transform_t<f_t>{});
  size_t current_bytes = temp_storage_bytes;
  RAFT_CUDA_TRY(cub::DeviceSegmentedReduce::Reduce(temp_storage.data(),
                                                   current_bytes,
                                                   coeff_abs_iter,
                                                   row_inf_norm.data(),
                                                   op_problem.n_constraints,
                                                   op_problem.offsets.data(),
                                                   op_problem.offsets.data() + 1,
                                                   max_op_t<f_t>{},
                                                   f_t(0),
                                                   stream_view));
}

template <typename i_t, typename f_t>
void compute_big_m_skip_rows(const problem_t<i_t, f_t>& op_problem,
                             rmm::device_uvector<std::uint8_t>& temp_storage,
                             size_t temp_storage_bytes,
                             rmm::device_uvector<f_t>& row_inf_norm,
                             rmm::device_uvector<f_t>& row_min_nonzero,
                             rmm::device_uvector<i_t>& row_nonzero_count,
                             rmm::device_uvector<i_t>& row_skip_scaling)
{
  auto coeff_abs_iter =
    thrust::make_transform_iterator(op_problem.coefficients.data(), abs_value_transform_t<f_t>{});
  auto coeff_nonzero_min_iter = thrust::make_transform_iterator(
    op_problem.coefficients.data(), nonzero_abs_or_inf_transform_t<f_t>{});
  auto coeff_nonzero_count_iter = thrust::make_transform_iterator(
    op_problem.coefficients.data(), nonzero_count_transform_t<i_t, f_t>{});

  size_t max_bytes = temp_storage_bytes;
  RAFT_CUDA_TRY(cub::DeviceSegmentedReduce::Reduce(temp_storage.data(),
                                                   max_bytes,
                                                   coeff_abs_iter,
                                                   row_inf_norm.data(),
                                                   op_problem.n_constraints,
                                                   op_problem.offsets.data(),
                                                   op_problem.offsets.data() + 1,
                                                   max_op_t<f_t>{},
                                                   f_t(0),
                                                   op_problem.handle_ptr->get_stream()));
  size_t min_bytes = temp_storage_bytes;
  RAFT_CUDA_TRY(cub::DeviceSegmentedReduce::Reduce(temp_storage.data(),
                                                   min_bytes,
                                                   coeff_nonzero_min_iter,
                                                   row_min_nonzero.data(),
                                                   op_problem.n_constraints,
                                                   op_problem.offsets.data(),
                                                   op_problem.offsets.data() + 1,
                                                   min_op_t<f_t>{},
                                                   std::numeric_limits<f_t>::infinity(),
                                                   op_problem.handle_ptr->get_stream()));
  size_t count_bytes = temp_storage_bytes;
  RAFT_CUDA_TRY(cub::DeviceSegmentedReduce::Reduce(temp_storage.data(),
                                                   count_bytes,
                                                   coeff_nonzero_count_iter,
                                                   row_nonzero_count.data(),
                                                   op_problem.n_constraints,
                                                   op_problem.offsets.data(),
                                                   op_problem.offsets.data() + 1,
                                                   thrust::plus<i_t>{},
                                                   i_t(0),
                                                   op_problem.handle_ptr->get_stream()));

  auto row_begin = thrust::make_zip_iterator(
    thrust::make_tuple(row_inf_norm.begin(), row_min_nonzero.begin(), row_nonzero_count.begin()));
  auto row_end = thrust::make_zip_iterator(
    thrust::make_tuple(row_inf_norm.end(), row_min_nonzero.end(), row_nonzero_count.end()));
  thrust::transform(
    op_problem.handle_ptr->get_thrust_policy(),
    row_begin,
    row_end,
    row_skip_scaling.begin(),
    [] __device__(auto row_info) -> i_t {
      const f_t row_norm          = thrust::get<0>(row_info);
      const f_t row_min_non_zero  = thrust::get<1>(row_info);
      const i_t row_non_zero_size = thrust::get<2>(row_info);
      if (row_non_zero_size < i_t(2) || row_min_non_zero >= std::numeric_limits<f_t>::infinity()) {
        return i_t(0);
      }

      const f_t row_ratio = row_norm / row_min_non_zero;
      return row_norm >= static_cast<f_t>(big_m_abs_threshold) &&
                 row_ratio >= static_cast<f_t>(big_m_ratio_threshold)
               ? i_t(1)
               : i_t(0);
    });
}

template <typename i_t, typename f_t>
mip_scaling_strategy_t<i_t, f_t>::mip_scaling_strategy_t(problem_t<i_t, f_t>& op_problem_scaled)
  : handle_ptr_(op_problem_scaled.handle_ptr),
    stream_view_(handle_ptr_->get_stream()),
    op_problem_scaled_(op_problem_scaled)
{
}

template <typename i_t, typename f_t>
size_t dry_run_cub(const problem_t<i_t, f_t>& op_problem,
                   i_t n_rows,
                   rmm::device_uvector<f_t>& row_inf_norm,
                   rmm::device_uvector<f_t>& row_min_nonzero,
                   rmm::device_uvector<i_t>& row_nonzero_count,
                   rmm::cuda_stream_view stream_view)
{
  size_t temp_storage_bytes     = 0;
  size_t current_required_bytes = 0;

  auto coeff_abs_iter =
    thrust::make_transform_iterator(op_problem.coefficients.data(), abs_value_transform_t<f_t>{});
  RAFT_CUDA_TRY(cub::DeviceSegmentedReduce::Reduce(nullptr,
                                                   current_required_bytes,
                                                   coeff_abs_iter,
                                                   row_inf_norm.data(),
                                                   n_rows,
                                                   op_problem.offsets.data(),
                                                   op_problem.offsets.data() + 1,
                                                   max_op_t<f_t>{},
                                                   f_t(0),
                                                   stream_view));
  temp_storage_bytes = std::max(temp_storage_bytes, current_required_bytes);

  auto coeff_nonzero_min_iter = thrust::make_transform_iterator(
    op_problem.coefficients.data(), nonzero_abs_or_inf_transform_t<f_t>{});
  RAFT_CUDA_TRY(cub::DeviceSegmentedReduce::Reduce(nullptr,
                                                   current_required_bytes,
                                                   coeff_nonzero_min_iter,
                                                   row_min_nonzero.data(),
                                                   n_rows,
                                                   op_problem.offsets.data(),
                                                   op_problem.offsets.data() + 1,
                                                   min_op_t<f_t>{},
                                                   std::numeric_limits<f_t>::infinity(),
                                                   stream_view));
  temp_storage_bytes = std::max(temp_storage_bytes, current_required_bytes);

  auto coeff_nonzero_count_iter = thrust::make_transform_iterator(
    op_problem.coefficients.data(), nonzero_count_transform_t<i_t, f_t>{});
  RAFT_CUDA_TRY(cub::DeviceSegmentedReduce::Reduce(nullptr,
                                                   current_required_bytes,
                                                   coeff_nonzero_count_iter,
                                                   row_nonzero_count.data(),
                                                   n_rows,
                                                   op_problem.offsets.data(),
                                                   op_problem.offsets.data() + 1,
                                                   thrust::plus<i_t>{},
                                                   i_t(0),
                                                   stream_view));
  temp_storage_bytes = std::max(temp_storage_bytes, current_required_bytes);

  return temp_storage_bytes;
}

template <typename i_t, typename f_t>
void mip_scaling_strategy_t<i_t, f_t>::scale_problem()
{
  raft::common::nvtx::range fun_scope("mip_scale_problem");

  const i_t n_rows = op_problem_scaled_.n_constraints;
  const i_t n_cols = op_problem_scaled_.n_variables;
  const i_t nnz    = op_problem_scaled_.nnz;

  if (n_rows == 0 || nnz <= 0) {
    op_problem_scaled_.is_scaled_ = true;
    return;
  }

  rmm::device_uvector<f_t> row_inf_norm(static_cast<size_t>(n_rows), stream_view_);
  rmm::device_uvector<f_t> row_min_nonzero(static_cast<size_t>(n_rows), stream_view_);
  rmm::device_uvector<i_t> row_nonzero_count(static_cast<size_t>(n_rows), stream_view_);
  rmm::device_uvector<i_t> row_skip_scaling(static_cast<size_t>(n_rows), stream_view_);
  rmm::device_uvector<f_t> iteration_scaling(static_cast<size_t>(n_rows), stream_view_);
  rmm::device_uvector<i_t> coefficient_row_index(static_cast<size_t>(nnz), stream_view_);

  thrust::upper_bound(handle_ptr_->get_thrust_policy(),
                      op_problem_scaled_.offsets.begin(),
                      op_problem_scaled_.offsets.end(),
                      thrust::make_counting_iterator<i_t>(0),
                      thrust::make_counting_iterator<i_t>(nnz),
                      coefficient_row_index.begin());
  thrust::transform(
    handle_ptr_->get_thrust_policy(),
    coefficient_row_index.begin(),
    coefficient_row_index.end(),
    coefficient_row_index.begin(),
    [] __device__(i_t row_upper_bound_idx) -> i_t { return row_upper_bound_idx - 1; });

  size_t temp_storage_bytes = dry_run_cub(
    op_problem_scaled_, n_rows, row_inf_norm, row_min_nonzero, row_nonzero_count, stream_view_);

  rmm::device_uvector<std::uint8_t> temp_storage(temp_storage_bytes, stream_view_);
  compute_big_m_skip_rows(op_problem_scaled_,
                          temp_storage,
                          temp_storage_bytes,
                          row_inf_norm,
                          row_min_nonzero,
                          row_nonzero_count,
                          row_skip_scaling);

  i_t skipped_big_m_rows = thrust::count(
    handle_ptr_->get_thrust_policy(), row_skip_scaling.begin(), row_skip_scaling.end(), i_t(1));

  CUOPT_LOG_INFO("MIP row scaling start: rows=%d cols=%d iterations=%d skip_big_m_rows=%d",
                 n_rows,
                 n_cols,
                 row_scaling_num_iterations,
                 skipped_big_m_rows);

  for (int iteration = 0; iteration < row_scaling_num_iterations; ++iteration) {
    compute_row_inf_norm(
      op_problem_scaled_, temp_storage, temp_storage_bytes, row_inf_norm, stream_view_);

    using sum_count_t       = thrust::tuple<double, double>;
    auto log2_sum_and_count = thrust::transform_reduce(
      handle_ptr_->get_thrust_policy(),
      thrust::make_zip_iterator(thrust::make_tuple(row_inf_norm.begin(), row_skip_scaling.begin())),
      thrust::make_zip_iterator(thrust::make_tuple(row_inf_norm.end(), row_skip_scaling.end())),
      [] __device__(auto row_info) -> sum_count_t {
        const f_t row_norm = thrust::get<0>(row_info);
        const i_t skip_row = thrust::get<1>(row_info);
        if (skip_row || row_norm == f_t(0)) { return {0.0, 0.0}; }
        return {log2(static_cast<double>(row_norm)), 1.0};
      },
      sum_count_t{0.0, 0.0},
      [] __device__(sum_count_t a, sum_count_t b) -> sum_count_t {
        return {thrust::get<0>(a) + thrust::get<0>(b), thrust::get<1>(a) + thrust::get<1>(b)};
      });
    const double log2_sum      = thrust::get<0>(log2_sum_and_count);
    const i_t active_row_count = static_cast<i_t>(thrust::get<1>(log2_sum_and_count));
    if (active_row_count == 0) { break; }

    f_t target_norm = static_cast<f_t>(exp2(log2_sum / static_cast<double>(active_row_count)));
    cuopt_assert(isfinite(target_norm), "target_norm must be finite");
    cuopt_assert(target_norm > f_t(0), "target_norm must be positive");

    thrust::transform(
      handle_ptr_->get_thrust_policy(),
      thrust::make_zip_iterator(thrust::make_tuple(row_inf_norm.begin(), row_skip_scaling.begin())),
      thrust::make_zip_iterator(thrust::make_tuple(row_inf_norm.end(), row_skip_scaling.end())),
      iteration_scaling.begin(),
      [target_norm] __device__(auto row_info) -> f_t {
        const f_t row_norm = thrust::get<0>(row_info);
        const i_t skip_row = thrust::get<1>(row_info);
        if (skip_row || row_norm == f_t(0)) { return f_t(1); }

        const f_t desired_scaling = target_norm / row_norm;
        if (!isfinite(desired_scaling) || desired_scaling <= f_t(0)) { return f_t(1); }

        int exponent = 0;
        const f_t mantissa =
          frexp(desired_scaling, &exponent);  // desired_scaling = mantissa * 2^exponent
        int k =
          mantissa >= static_cast<f_t>(nearest_pow2_mantissa_threshold) ? exponent : exponent - 1;
        // clamp it so we don't overscale. range [1e-6,1e6]
        if (k < row_scaling_k_min) { k = row_scaling_k_min; }
        if (k > row_scaling_k_max) { k = row_scaling_k_max; }
        return ldexp(f_t(1), k);
      });

    thrust::transform(
      handle_ptr_->get_thrust_policy(),
      op_problem_scaled_.coefficients.begin(),
      op_problem_scaled_.coefficients.end(),
      thrust::make_permutation_iterator(iteration_scaling.begin(), coefficient_row_index.begin()),
      op_problem_scaled_.coefficients.begin(),
      thrust::multiplies<f_t>{});

    thrust::transform(handle_ptr_->get_thrust_policy(),
                      op_problem_scaled_.constraint_lower_bounds.begin(),
                      op_problem_scaled_.constraint_lower_bounds.end(),
                      iteration_scaling.begin(),
                      op_problem_scaled_.constraint_lower_bounds.begin(),
                      thrust::multiplies<f_t>{});
    thrust::transform(handle_ptr_->get_thrust_policy(),
                      op_problem_scaled_.constraint_upper_bounds.begin(),
                      op_problem_scaled_.constraint_upper_bounds.end(),
                      iteration_scaling.begin(),
                      op_problem_scaled_.constraint_upper_bounds.begin(),
                      thrust::multiplies<f_t>{});
  }

  op_problem_scaled_.compute_transpose_of_problem();
  combine_constraint_bounds(op_problem_scaled_, op_problem_scaled_.combined_bounds);
  op_problem_scaled_.check_problem_representation(true, true);
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
