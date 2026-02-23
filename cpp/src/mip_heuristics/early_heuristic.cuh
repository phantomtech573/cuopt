/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <mip_heuristics/problem/problem.cuh>
#include <mip_heuristics/solution/solution.cuh>

#include <cuopt/linear_programming/mip/solver_settings.hpp>

#include <utilities/logger.hpp>

#include <thrust/fill.h>

#include <chrono>
#include <functional>
#include <limits>
#include <vector>

namespace cuopt::linear_programming::detail {

template <typename f_t>
using early_incumbent_callback_t =
  std::function<void(f_t objective, const std::vector<f_t>& assignment)>;

// CRTP base for early heuristics that run on the original (or papilo-presolved) problem
// during presolve to find incumbents as early as possible.
// Derived classes implement start() and stop().
template <typename i_t, typename f_t, typename Derived>
class early_heuristic_t {
 public:
  early_heuristic_t(const optimization_problem_t<i_t, f_t>& op_problem,
                    const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances,
                    early_incumbent_callback_t<f_t> incumbent_callback)
    : incumbent_callback_(std::move(incumbent_callback))
  {
    problem_ptr_ = std::make_unique<problem_t<i_t, f_t>>(op_problem, tolerances, false);
    problem_ptr_->preprocess_problem();

    solution_ptr_ = std::make_unique<solution_t<i_t, f_t>>(*problem_ptr_);
    thrust::fill(problem_ptr_->handle_ptr->get_thrust_policy(),
                 solution_ptr_->assignment.begin(),
                 solution_ptr_->assignment.end(),
                 f_t{0});
    solution_ptr_->clamp_within_bounds();
  }

  bool solution_found() const { return solution_found_; }
  f_t get_best_objective() const { return best_objective_; }
  void set_best_objective(f_t obj) { best_objective_ = obj; }
  const std::vector<f_t>& get_best_assignment() const { return best_assignment_; }

 protected:
  ~early_heuristic_t() = default;

  // NOT thread-safe
  void try_update_best(f_t user_obj, const std::vector<f_t>& assignment)
  {
    if (user_obj >= best_objective_) { return; }
    best_objective_ = user_obj;

    auto* handle_ptr = problem_ptr_->handle_ptr;
    RAFT_CUDA_TRY(cudaSetDevice(handle_ptr->get_device()));
    rmm::device_uvector<f_t> d_assignment(assignment.size(), handle_ptr->get_stream());
    raft::copy(d_assignment.data(), assignment.data(), assignment.size(), handle_ptr->get_stream());
    problem_ptr_->post_process_assignment(d_assignment);
    auto user_assignment = cuopt::host_copy(d_assignment, handle_ptr->get_stream());

    best_assignment_ = user_assignment;
    solution_found_  = true;
    double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time_).count();
    CUOPT_LOG_INFO("Early heuristics (%s) lowered the primal bound. Objective %g. Time %.2f",
                   Derived::name(),
                   user_obj,
                   elapsed);
    if (incumbent_callback_) { incumbent_callback_(user_obj, user_assignment); }
  }

  std::unique_ptr<problem_t<i_t, f_t>> problem_ptr_;
  std::unique_ptr<solution_t<i_t, f_t>> solution_ptr_;

  bool solution_found_{false};
  f_t best_objective_{std::numeric_limits<f_t>::infinity()};
  std::vector<f_t> best_assignment_;

  early_incumbent_callback_t<f_t> incumbent_callback_;
  std::chrono::steady_clock::time_point start_time_;
};

}  // namespace cuopt::linear_programming::detail
