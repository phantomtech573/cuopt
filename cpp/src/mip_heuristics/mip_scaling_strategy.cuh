/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <mip_heuristics/problem/problem.cuh>

#include <raft/core/handle.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
class mip_scaling_strategy_t {
 public:
  explicit mip_scaling_strategy_t(problem_t<i_t, f_t>& op_problem_scaled);

  void scale_problem();

 private:
  raft::handle_t const* handle_ptr_{nullptr};
  rmm::cuda_stream_view stream_view_;
  problem_t<i_t, f_t>& op_problem_scaled_;
};

}  // namespace cuopt::linear_programming::detail
