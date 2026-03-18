/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/linear_programming/mip/solver_stats.hpp>

#include <mip_heuristics/mip_constants.hpp>
#include <mip_heuristics/problem/problem.cuh>
#include <mip_heuristics/relaxed_lp/lp_state.cuh>
#include <mip_heuristics/solution/solution.cuh>
#include <pdlp/initial_scaling_strategy/initial_scaling.cuh>
#include <utilities/work_limit_context.hpp>
#include <utilities/work_unit_scheduler.hpp>

#include <limits>
#include <mutex>
#include <vector>

#include <utilities/models/fj_predictor/header.h>
#include <utilities/work_limit_timer.hpp>
#include <utilities/work_unit_predictor.hpp>

#pragma once

// Forward declare
namespace cuopt::linear_programming::dual_simplex {
template <typename i_t, typename f_t>
class branch_and_bound_t;
}

namespace cuopt::linear_programming::detail {

struct mip_solver_work_unit_predictors_t {
  work_unit_predictor_t<fj_predictor, gpu_work_unit_scaler_t> fj_predictor{};
};
template <typename i_t, typename f_t>
class diversity_manager_t;

template <typename i_t, typename f_t>
struct solution_callback_payload_t {
  std::vector<f_t> assignment{};
  f_t user_objective{};
  f_t solver_objective{};
  internals::mip_solution_callback_info_t callback_info{};
};

template <typename i_t, typename f_t>
class solution_publication_t {
 public:
  solution_publication_t(const mip_solver_settings_t<i_t, f_t>& settings_,
                         solver_stats_t<i_t, f_t>& stats_)
    : settings(settings_), stats(stats_)
  {
  }

  void reset_published_best(f_t objective = std::numeric_limits<f_t>::max())
  {
    best_callback_feasible_objective_ = objective;
  }

  solution_callback_payload_t<i_t, f_t> build_payload(
    problem_t<i_t, f_t>* problem_ptr,
    pdlp_initial_scaling_strategy_t<i_t, f_t>& scaling,
    solution_t<i_t, f_t>& sol,
    internals::mip_solution_origin_t origin,
    double work_timestamp)
  {
    cuopt_assert(problem_ptr != nullptr, "Callback payload problem pointer must not be null");
    cuopt_assert(work_timestamp >= 0.0, "work_timestamp must not be negative");
    solution_callback_payload_t<i_t, f_t> payload{};
    payload.user_objective               = sol.get_user_objective();
    payload.solver_objective             = sol.get_objective();
    payload.callback_info.origin         = origin;
    payload.callback_info.work_timestamp = work_timestamp;
    solution_t<i_t, f_t> temp_sol(sol);
    problem_ptr->post_process_assignment(temp_sol.assignment);
    if (settings.mip_scaling) {
      rmm::device_uvector<f_t> dummy(0, temp_sol.handle_ptr->get_stream());
      scaling.unscale_solutions(temp_sol.assignment, dummy);
    }
    if (problem_ptr->has_papilo_presolve_data()) {
      problem_ptr->papilo_uncrush_assignment(temp_sol.assignment);
    }
    payload.assignment = temp_sol.get_host_assignment();
    return payload;
  }

  bool publish_new_best_feasible(const solution_callback_payload_t<i_t, f_t>& payload,
                                 double elapsed_time = -1.0)
  {
    std::lock_guard<std::mutex> lock(solution_callback_mutex_);
    cuopt_assert(std::isfinite(payload.solver_objective),
                 "Feasible incumbent objective must be finite");
    if (!(payload.solver_objective < best_callback_feasible_objective_)) { return false; }

    if (settings.benchmark_info_ptr != nullptr && elapsed_time >= 0.0) {
      settings.benchmark_info_ptr->last_improvement_of_best_feasible = elapsed_time;
    }
    invoke_get_solution_callbacks(payload);
    best_callback_feasible_objective_ = payload.solver_objective;
    return true;
  }

  void publish_terminal_solution(const solution_callback_payload_t<i_t, f_t>& payload)
  {
    std::lock_guard<std::mutex> lock(solution_callback_mutex_);
    invoke_get_solution_callbacks(payload);
    best_callback_feasible_objective_ = payload.solver_objective;
  }

 private:
  void invoke_get_solution_callbacks(const solution_callback_payload_t<i_t, f_t>& payload)
  {
    auto user_callbacks = settings.get_mip_callbacks();
    CUOPT_LOG_DEBUG("Publishing incumbent: obj=%g wut=%.6f origin=%s callbacks=%zu",
                    payload.user_objective,
                    payload.callback_info.work_timestamp,
                    internals::mip_solution_origin_to_string(payload.callback_info.origin),
                    user_callbacks.size());

    std::vector<f_t> user_objective_vec(1);
    std::vector<f_t> user_bound_vec(1);
    user_objective_vec[0] = payload.user_objective;
    user_bound_vec[0]     = stats.get_solution_bound();

    for (auto callback : user_callbacks) {
      if (callback->get_type() == internals::base_solution_callback_type::GET_SOLUTION_EXT) {
        auto get_sol_callback_ext = static_cast<internals::get_solution_callback_ext_t*>(callback);
        get_sol_callback_ext->get_solution(const_cast<f_t*>(payload.assignment.data()),
                                           user_objective_vec.data(),
                                           user_bound_vec.data(),
                                           &payload.callback_info,
                                           get_sol_callback_ext->get_user_data());
      } else if (callback->get_type() == internals::base_solution_callback_type::GET_SOLUTION) {
        auto get_sol_callback = static_cast<internals::get_solution_callback_t*>(callback);
        get_sol_callback->get_solution(const_cast<f_t*>(payload.assignment.data()),
                                       user_objective_vec.data(),
                                       user_bound_vec.data(),
                                       get_sol_callback->get_user_data());
      }
    }
  }

  const mip_solver_settings_t<i_t, f_t>& settings;
  solver_stats_t<i_t, f_t>& stats;
  std::mutex solution_callback_mutex_;
  f_t best_callback_feasible_objective_{std::numeric_limits<f_t>::max()};
};

// Processes SET_SOLUTION user callbacks: invokes the callback, validates/scales/preprocesses
// the returned assignment, and returns it for the caller to reinject.
template <typename i_t, typename f_t>
class solution_injection_t {
 public:
  solution_injection_t(const mip_solver_settings_t<i_t, f_t>& settings_,
                       solver_stats_t<i_t, f_t>& stats_)
    : settings(settings_), stats(stats_)
  {
  }

  // Invokes SET_SOLUTION callbacks with current incumbent info.
  // For each callback that returns a valid solution, calls `on_injected` with the processed
  // host assignment and solver-space objective.
  template <typename OnInjectedFn>
  void invoke_set_solution_callbacks(problem_t<i_t, f_t>* problem_ptr,
                                     pdlp_initial_scaling_strategy_t<i_t, f_t>& scaling,
                                     solution_t<i_t, f_t>& current_incumbent,
                                     OnInjectedFn&& on_injected)
  {
    auto user_callbacks = settings.get_mip_callbacks();
    for (auto callback : user_callbacks) {
      if (callback->get_type() != internals::base_solution_callback_type::SET_SOLUTION) {
        continue;
      }
      auto set_sol_callback       = static_cast<internals::set_solution_callback_t*>(callback);
      f_t user_bound              = stats.get_solution_bound();
      auto callback_num_variables = problem_ptr->original_problem_ptr->get_n_variables();
      rmm::device_uvector<f_t> incumbent_assignment(callback_num_variables,
                                                    current_incumbent.handle_ptr->get_stream());
      auto inf = std::numeric_limits<f_t>::infinity();
      current_incumbent.handle_ptr->sync_stream();
      std::vector<f_t> h_incumbent_assignment(incumbent_assignment.size());
      std::vector<f_t> h_outside_sol_objective(1, inf);
      std::vector<f_t> h_user_bound(1, user_bound);
      set_sol_callback->set_solution(h_incumbent_assignment.data(),
                                     h_outside_sol_objective.data(),
                                     h_user_bound.data(),
                                     set_sol_callback->get_user_data());
      f_t outside_sol_objective = h_outside_sol_objective[0];
      if (outside_sol_objective == inf) { continue; }

      raft::copy(incumbent_assignment.data(),
                 h_incumbent_assignment.data(),
                 incumbent_assignment.size(),
                 current_incumbent.handle_ptr->get_stream());
      if (settings.mip_scaling) { scaling.scale_solutions(incumbent_assignment); }
      bool is_valid = problem_ptr->pre_process_assignment(incumbent_assignment);
      if (!is_valid) { continue; }

      solution_t<i_t, f_t> outside_sol(current_incumbent);
      cuopt_assert(outside_sol.assignment.size() == incumbent_assignment.size(),
                   "Incumbent assignment size mismatch");
      raft::copy(outside_sol.assignment.data(),
                 incumbent_assignment.data(),
                 incumbent_assignment.size(),
                 current_incumbent.handle_ptr->get_stream());
      outside_sol.compute_feasibility();

      CUOPT_LOG_DEBUG("Injected solution feasibility = %d objective = %g excess = %g",
                      outside_sol.get_feasible(),
                      outside_sol.get_user_objective(),
                      outside_sol.get_total_excess());
      cuopt_assert(std::abs(outside_sol.get_user_objective() - outside_sol_objective) <= 1e-6,
                   "External solution objective mismatch");
      on_injected(outside_sol.get_host_assignment(),
                  outside_sol.get_objective(),
                  internals::mip_solution_origin_t::USER_INITIAL);
    }
  }

 private:
  const mip_solver_settings_t<i_t, f_t>& settings;
  solver_stats_t<i_t, f_t>& stats;
};

// Aggregate structure containing the global context of the solving process for convenience:
// The current problem, user settings, raft handle and statistics objects
template <typename i_t, typename f_t>
struct mip_solver_context_t {
  explicit mip_solver_context_t(raft::handle_t const* handle_ptr_,
                                problem_t<i_t, f_t>* problem_ptr_,
                                mip_solver_settings_t<i_t, f_t> settings_,
                                pdlp_initial_scaling_strategy_t<i_t, f_t>& scaling)
    : handle_ptr(handle_ptr_),
      problem_ptr(problem_ptr_),
      settings(settings_),
      scaling(scaling),
      solution_publication(settings, stats),
      solution_injection(settings, stats)
  {
    cuopt_assert(problem_ptr != nullptr, "problem_ptr is nullptr");
    stats.set_solution_bound(problem_ptr->maximize ? std::numeric_limits<f_t>::infinity()
                                                   : -std::numeric_limits<f_t>::infinity());
    gpu_heur_loop.deterministic = (settings.determinism_mode & CUOPT_DETERMINISM_GPU_HEURISTICS);
    cuopt_assert(settings.cpufj_work_unit_scale > 0.0, "CPUFJ work-unit scale must be positive");
    cuopt_assert(settings.gpu_heur_work_unit_scale > 0.0,
                 "GPU heuristic work-unit scale must be positive");
    gpu_heur_loop.work_unit_scale = settings.gpu_heur_work_unit_scale;
  }

  mip_solver_context_t(const mip_solver_context_t&)            = delete;
  mip_solver_context_t& operator=(const mip_solver_context_t&) = delete;

  raft::handle_t const* const handle_ptr;
  problem_t<i_t, f_t>* problem_ptr;
  dual_simplex::branch_and_bound_t<i_t, f_t>* branch_and_bound_ptr{nullptr};
  diversity_manager_t<i_t, f_t>* diversity_manager_ptr{nullptr};
  std::atomic<bool> preempt_heuristic_solver_ = false;
  const mip_solver_settings_t<i_t, f_t> settings;
  pdlp_initial_scaling_strategy_t<i_t, f_t>& scaling;
  solver_stats_t<i_t, f_t> stats;
  // TODO: ensure thread local (or use locks...?)
  mip_solver_work_unit_predictors_t work_unit_predictors;
  // Work limit context for tracking work units in deterministic mode (shared across all timers in
  // GPU heuristic loop)
  work_limit_context_t gpu_heur_loop{"GPUHeur"};
  solution_publication_t<i_t, f_t> solution_publication;
  solution_injection_t<i_t, f_t> solution_injection;

  // Root termination checker — set by mip_solver_t after construction.
  // All sub-timers should use this as parent for wall-clock safety.
  cuopt::termination_checker_t* termination{nullptr};

  // synchronization every 5 seconds for deterministic mode
  work_unit_scheduler_t work_unit_scheduler_{5.0};
};

}  // namespace cuopt::linear_programming::detail
