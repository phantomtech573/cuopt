/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <mip_heuristics/mip_constants.hpp>
#include "diversity/diversity_manager.cuh"
#include "local_search/local_search.cuh"
#include "local_search/rounding/simple_rounding.cuh"
#include "solver.cuh"

#include <pdlp/pdlp.cuh>
#include <pdlp/solve.cuh>

#include <branch_and_bound/branch_and_bound.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/solve.hpp>
#include <utilities/determinism_log.hpp>

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/cusparse_macros.hpp>

#include <cmath>
#include <future>
#include <memory>
#include <thread>

namespace cuopt::linear_programming::detail {

// This serves as both a warm up but also a mandatory initial call to setup cuSparse and cuBLAS
static void init_handler(const raft::handle_t* handle_ptr)
{
  // Init cuBlas / cuSparse context here to avoid having it during solving time
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublassetpointermode(
    handle_ptr->get_cublas_handle(), CUBLAS_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsesetpointermode(
    handle_ptr->get_cusparse_handle(), CUSPARSE_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
}

template <typename i_t, typename f_t>
mip_solver_t<i_t, f_t>::mip_solver_t(const problem_t<i_t, f_t>& op_problem,
                                     const mip_solver_settings_t<i_t, f_t>& solver_settings,
                                     pdlp_initial_scaling_strategy_t<i_t, f_t>& scaling,
                                     timer_t timer)
  : op_problem_(op_problem),
    solver_settings_(solver_settings),
    context(op_problem.handle_ptr,
            const_cast<problem_t<i_t, f_t>*>(&op_problem),
            solver_settings,
            scaling),
    timer_(timer)
{
  init_handler(op_problem.handle_ptr);
}

template <typename i_t, typename f_t>
struct branch_and_bound_solution_helper_t {
  branch_and_bound_solution_helper_t(diversity_manager_t<i_t, f_t>* dm,
                                     dual_simplex::simplex_solver_settings_t<i_t, f_t>& settings)
    : dm(dm), settings_(settings) {};

  void solution_callback(std::vector<f_t>& solution, f_t objective)
  {
    if (!settings_.deterministic) {
      dm->population.add_external_solution(
        solution, objective, internals::mip_solution_origin_t::BRANCH_AND_BOUND_NODE);
      dm->rins.new_best_incumbent_callback(solution);
    }
  }

  void solution_callback_ext(std::vector<f_t>& solution,
                             f_t objective,
                             const internals::mip_solution_callback_info_t& callback_info,
                             double work_timestamp)
  {
    if (!settings_.deterministic) {
      dm->population.add_external_solution(
        solution, objective, internals::mip_solution_origin_t::BRANCH_AND_BOUND_NODE);
      dm->rins.new_best_incumbent_callback(solution);
      return;
    }

    cuopt_assert(dm != nullptr, "Diversity manager pointer must be valid");
    cuopt_assert(dm->context.problem_ptr != nullptr, "Problem pointer must be valid");
    cuopt_assert(solution.size() == (size_t)dm->context.problem_ptr->n_variables,
                 "Deterministic B&B callback solution size mismatch");
    cuopt_assert(std::isfinite(objective), "Deterministic B&B callback objective must be finite");
    const auto payload =
      make_solution_callback_payload_from_host_solution<i_t, f_t>(dm->context.problem_ptr,
                                                                  dm->context.settings,
                                                                  dm->context.gpu_heur_loop,
                                                                  solution,
                                                                  objective,
                                                                  callback_info.origin,
                                                                  work_timestamp);
    dm->context.solution_publication.publish_new_best_feasible(payload, dm->timer.elapsed_time());
  }

  void set_simplex_solution(std::vector<f_t>& solution,
                            std::vector<f_t>& dual_solution,
                            f_t objective)
  {
    dm->set_simplex_solution(solution, dual_solution, objective);
  }

  void node_processed_callback(const std::vector<f_t>& solution, f_t objective)
  {
    dm->rins.node_callback(solution, objective);
  }

  void preempt_heuristic_solver() { dm->population.preempt_heuristic_solver(); }
  diversity_manager_t<i_t, f_t>* dm;
  dual_simplex::simplex_solver_settings_t<i_t, f_t>& settings_;
};

template <typename i_t, typename f_t>
solution_t<i_t, f_t> mip_solver_t<i_t, f_t>::run_solver()
{
  //  we need to keep original problem const
  cuopt_assert(context.problem_ptr != nullptr, "invalid problem pointer");
  context.problem_ptr->tolerances = context.settings.get_tolerances();
  cuopt_expects(context.problem_ptr->preprocess_called,
                error_type_t::RuntimeError,
                "preprocess_problem should be called before running the solver");

  diversity_manager_t<i_t, f_t> dm(context);
  if (context.problem_ptr->empty) {
    CUOPT_LOG_INFO("Problem fully reduced in presolve");
    solution_t<i_t, f_t> sol(*context.problem_ptr);
    sol.set_problem_fully_reduced();
    const auto payload = make_solution_callback_payload_from_solution<i_t, f_t>(
      context.problem_ptr,
      context.settings,
      context.scaling,
      context.gpu_heur_loop,
      sol,
      internals::mip_solution_origin_t::UNKNOWN,
      -1.0);
    context.solution_publication.invoke_get_solution_callbacks(payload);
    context.problem_ptr->post_process_solution(sol);
    return sol;
  }
  const bool deterministic_run = is_deterministic_mode(context.settings.determinism_mode);
  const f_t gpu_heur_work_limit =
    deterministic_run ? context.settings.work_limit : timer_.get_time_limit();
  if (deterministic_run)
    cuopt_assert(gpu_heur_work_limit >= 0.0,
                 "Deterministic GPU heuristic work limit must be non-negative");
  dm.timer                = work_limit_timer_t(context.gpu_heur_loop, gpu_heur_work_limit);
  const bool run_presolve = context.settings.presolver != presolver_t::None;
  f_t time_limit =
    deterministic_run ? std::numeric_limits<f_t>::infinity() : timer_.remaining_time();
  double presolve_time_limit = std::min(0.1 * time_limit, 60.0);
  presolve_time_limit =
    deterministic_run ? std::numeric_limits<f_t>::infinity() : presolve_time_limit;
  bool presolve_success = run_presolve ? dm.run_presolve(presolve_time_limit, timer_) : true;
  if (!presolve_success) {
    CUOPT_LOG_INFO("Problem proven infeasible in presolve");
    solution_t<i_t, f_t> sol(*context.problem_ptr);
    sol.set_problem_fully_reduced();
    context.problem_ptr->post_process_solution(sol);
    return sol;
  }
  if (run_presolve && context.problem_ptr->empty) {
    CUOPT_LOG_INFO("Problem full reduced in presolve");
    solution_t<i_t, f_t> sol(*context.problem_ptr);
    sol.set_problem_fully_reduced();
    const auto payload = make_solution_callback_payload_from_solution<i_t, f_t>(
      context.problem_ptr,
      context.settings,
      context.scaling,
      context.gpu_heur_loop,
      sol,
      internals::mip_solution_origin_t::UNKNOWN,
      -1.0);
    context.solution_publication.invoke_get_solution_callbacks(payload);
    context.problem_ptr->post_process_solution(sol);
    return sol;
  }

  if (timer_.check_time_limit()) {
    CUOPT_LOG_INFO("Time limit reached after presolve");
    solution_t<i_t, f_t> sol(*context.problem_ptr);
    context.stats.total_solve_time = timer_.elapsed_time();
    context.problem_ptr->post_process_solution(sol);
    return sol;
  }

  // if the problem was reduced to a LP: run concurrent LP
  if (run_presolve && context.problem_ptr->n_integer_vars == 0) {
    CUOPT_LOG_INFO("Problem reduced to a LP, running concurrent LP");
    pdlp_solver_settings_t<i_t, f_t> settings{};
    settings.time_limit = timer_.remaining_time();
    auto lp_timer       = timer_t(settings.time_limit);
    settings.method     = method_t::Concurrent;
    settings.presolver  = presolver_t::None;

    auto opt_sol = solve_lp_with_method<i_t, f_t>(*context.problem_ptr, settings, lp_timer);

    solution_t<i_t, f_t> sol(*context.problem_ptr);
    sol.copy_new_assignment(
      host_copy(opt_sol.get_primal_solution(), context.problem_ptr->handle_ptr->get_stream()));
    if (opt_sol.get_termination_status() == pdlp_termination_status_t::Optimal ||
        opt_sol.get_termination_status() == pdlp_termination_status_t::PrimalInfeasible ||
        opt_sol.get_termination_status() == pdlp_termination_status_t::DualInfeasible) {
      sol.set_problem_fully_reduced();
    }
    if (opt_sol.get_termination_status() == pdlp_termination_status_t::Optimal) {
      const auto payload = make_solution_callback_payload_from_solution<i_t, f_t>(
        context.problem_ptr,
        context.settings,
        context.scaling,
        context.gpu_heur_loop,
        sol,
        internals::mip_solution_origin_t::UNKNOWN,
        -1.0);
      context.solution_publication.invoke_get_solution_callbacks(payload);
    }
    context.problem_ptr->post_process_solution(sol);
    return sol;
  }
  context.work_unit_scheduler_.register_context(context.gpu_heur_loop);

  namespace dual_simplex = cuopt::linear_programming::dual_simplex;
  std::future<dual_simplex::mip_status_t> branch_and_bound_status_future;
  dual_simplex::user_problem_t<i_t, f_t> branch_and_bound_problem(context.problem_ptr->handle_ptr);
  context.problem_ptr->recompute_objective_integrality();
  if (context.problem_ptr->is_objective_integral()) {
    CUOPT_LOG_INFO("Objective function is integral, scale %g",
                   context.problem_ptr->presolve_data.objective_scaling_factor);
  }
  branch_and_bound_problem.objective_is_integral = context.problem_ptr->is_objective_integral();
  dual_simplex::simplex_solver_settings_t<i_t, f_t> branch_and_bound_settings;
  std::unique_ptr<dual_simplex::branch_and_bound_t<i_t, f_t>> branch_and_bound;
  branch_and_bound_solution_helper_t solution_helper(&dm, branch_and_bound_settings);
  dual_simplex::mip_solution_t<i_t, f_t> branch_and_bound_solution(1);

  bool run_bb = !context.settings.heuristics_only;
  if (run_bb) {
    // Convert the presolved problem to dual_simplex::user_problem_t
    op_problem_.get_host_user_problem(branch_and_bound_problem);
    // Resize the solution now that we know the number of columns/variables
    branch_and_bound_solution.resize(branch_and_bound_problem.num_cols);

    // Fill in the settings for branch and bound
    branch_and_bound_settings.time_limit           = timer_.get_time_limit();
    branch_and_bound_settings.node_limit           = context.settings.node_limit;
    branch_and_bound_settings.print_presolve_stats = false;
    branch_and_bound_settings.absolute_mip_gap_tol = context.settings.tolerances.absolute_mip_gap;
    branch_and_bound_settings.relative_mip_gap_tol = context.settings.tolerances.relative_mip_gap;
    branch_and_bound_settings.integer_tol = context.settings.tolerances.integrality_tolerance;
    branch_and_bound_settings.reliability_branching = solver_settings_.reliability_branching;
    branch_and_bound_settings.max_cut_passes        = context.settings.max_cut_passes;
    branch_and_bound_settings.mir_cuts              = context.settings.mir_cuts;
    branch_and_bound_settings.deterministic =
      is_deterministic_mode(context.settings.determinism_mode);

    if (is_deterministic_mode(context.settings.determinism_mode)) {
      branch_and_bound_settings.work_limit = context.settings.work_limit;
    } else {
      branch_and_bound_settings.work_limit = std::numeric_limits<f_t>::infinity();
    }
    branch_and_bound_settings.mixed_integer_gomory_cuts =
      context.settings.mixed_integer_gomory_cuts;
    branch_and_bound_settings.knapsack_cuts = context.settings.knapsack_cuts;
    branch_and_bound_settings.strong_chvatal_gomory_cuts =
      context.settings.strong_chvatal_gomory_cuts;
    branch_and_bound_settings.reduced_cost_strengthening =
      context.settings.reduced_cost_strengthening;
    branch_and_bound_settings.cut_change_threshold  = context.settings.cut_change_threshold;
    branch_and_bound_settings.cut_min_orthogonality = context.settings.cut_min_orthogonality;
    branch_and_bound_settings.mip_batch_pdlp_strong_branching =
      context.settings.mip_batch_pdlp_strong_branching;

    if (context.settings.num_cpu_threads < 0) {
      branch_and_bound_settings.num_threads = std::max(1, omp_get_max_threads() - 1);
    } else {
      branch_and_bound_settings.num_threads = std::max(1, context.settings.num_cpu_threads);
    }
    CUOPT_LOG_INFO("Using %d CPU threads for B&B", branch_and_bound_settings.num_threads);

    // Set the branch and bound -> primal heuristics callback
    branch_and_bound_settings.solution_callback =
      std::bind(&branch_and_bound_solution_helper_t<i_t, f_t>::solution_callback,
                &solution_helper,
                std::placeholders::_1,
                std::placeholders::_2);
    branch_and_bound_settings.solution_callback_ext =
      std::bind(&branch_and_bound_solution_helper_t<i_t, f_t>::solution_callback_ext,
                &solution_helper,
                std::placeholders::_1,
                std::placeholders::_2,
                std::placeholders::_3,
                std::placeholders::_4);
    // heuristic_preemption_callback is needed in both modes to properly stop the heuristic thread
    branch_and_bound_settings.heuristic_preemption_callback = std::bind(
      &branch_and_bound_solution_helper_t<i_t, f_t>::preempt_heuristic_solver, &solution_helper);
    if (context.settings.determinism_mode == CUOPT_MODE_OPPORTUNISTIC) {
      branch_and_bound_settings.set_simplex_solution_callback =
        std::bind(&branch_and_bound_solution_helper_t<i_t, f_t>::set_simplex_solution,
                  &solution_helper,
                  std::placeholders::_1,
                  std::placeholders::_2,
                  std::placeholders::_3);

      branch_and_bound_settings.node_processed_callback =
        std::bind(&branch_and_bound_solution_helper_t<i_t, f_t>::node_processed_callback,
                  &solution_helper,
                  std::placeholders::_1,
                  std::placeholders::_2);
    }

    // Create the branch and bound object
    branch_and_bound = std::make_unique<dual_simplex::branch_and_bound_t<i_t, f_t>>(
      branch_and_bound_problem, branch_and_bound_settings, timer_.get_tic_start());
    context.branch_and_bound_ptr = branch_and_bound.get();
    auto* stats_ptr              = &context.stats;
    branch_and_bound->set_user_bound_callback(
      [stats_ptr](f_t user_bound) { stats_ptr->set_solution_bound(user_bound); });

    // Set the primal heuristics -> branch and bound callback
    if (context.settings.determinism_mode == CUOPT_MODE_OPPORTUNISTIC) {
      branch_and_bound->set_concurrent_lp_root_solve(true);

      context.problem_ptr->branch_and_bound_callback =
        std::bind(&dual_simplex::branch_and_bound_t<i_t, f_t>::set_new_solution,
                  branch_and_bound.get(),
                  std::placeholders::_1);
    } else if (is_deterministic_mode(context.settings.determinism_mode)) {
      branch_and_bound->set_concurrent_lp_root_solve(false);
      // TODO once deterministic GPU heuristics are integrated
      // context.problem_ptr->branch_and_bound_callback =
      //   [bb = branch_and_bound.get()](const std::vector<f_t>& solution) {
      //     bb->queue_external_solution_deterministic(solution, 0.0);
      //   };
    }

    context.work_unit_scheduler_.register_context(branch_and_bound->get_work_unit_context());

    if (is_deterministic_mode(context.settings.determinism_mode)) {
      context.problem_ptr->set_root_relaxation_solution_callback = nullptr;
    } else {
      context.problem_ptr->set_root_relaxation_solution_callback =
        std::bind(&dual_simplex::branch_and_bound_t<i_t, f_t>::set_root_relaxation_solution,
                  branch_and_bound.get(),
                  std::placeholders::_1,
                  std::placeholders::_2,
                  std::placeholders::_3,
                  std::placeholders::_4,
                  std::placeholders::_5,
                  std::placeholders::_6);
    }

    if (timer_.check_time_limit()) {
      CUOPT_LOG_INFO("Time limit reached during B&B setup");
      solution_t<i_t, f_t> sol(*context.problem_ptr);
      context.stats.total_solve_time = timer_.elapsed_time();
      context.problem_ptr->post_process_solution(sol);
      return sol;
    }

    // Fork a thread for branch and bound
    // std::async and std::future allow us to get the return value of bb::solve()
    // without having to manually manage the thread
    // std::future.get() performs a join() operation to wait until the return status is available
    int bb_device_id = context.handle_ptr->get_device();
    branch_and_bound_status_future =
      std::async(std::launch::async, [&branch_and_bound, &branch_and_bound_solution, bb_device_id] {
        RAFT_CUDA_TRY(cudaSetDevice(bb_device_id));
        return branch_and_bound->solve(branch_and_bound_solution);
      });
  }

  // Start the primal heuristics
  context.diversity_manager_ptr = &dm;
  auto sol                      = dm.run_solver();
  if (run_bb) {
    // Wait for the branch and bound to finish
    auto bb_status = branch_and_bound_status_future.get();
    if (branch_and_bound_solution.lower_bound > -std::numeric_limits<f_t>::infinity()) {
      context.stats.set_solution_bound(
        context.problem_ptr->get_user_obj_from_solver_obj(branch_and_bound_solution.lower_bound));
    }
    if (bb_status == dual_simplex::mip_status_t::INFEASIBLE) { sol.set_problem_fully_reduced(); }
    if (is_deterministic_mode(context.settings.determinism_mode) &&
        std::isfinite(branch_and_bound_solution.objective)) {
      CUOPT_DETERMINISM_LOG_INFO(
        "Deterministic solver B&B overwrite: bb_status=%d bb_obj=%.16e bb_lower=%.16e "
        "bb_hash=0x%x dm_hash=0x%x nodes=%d simplex_iterations=%d",
        (int)bb_status,
        branch_and_bound_solution.objective,
        branch_and_bound_solution.lower_bound,
        detail::compute_hash(branch_and_bound_solution.x),
        sol.get_hash(),
        branch_and_bound_solution.nodes_explored,
        branch_and_bound_solution.simplex_iterations);
      solution_t<i_t, f_t> bb_sol(*context.problem_ptr);
      bb_sol.copy_new_assignment(branch_and_bound_solution.x);
      bb_sol.compute_feasibility();
      sol = std::move(bb_sol);
    }
    context.stats.num_nodes              = branch_and_bound_solution.nodes_explored;
    context.stats.num_simplex_iterations = branch_and_bound_solution.simplex_iterations;
  }
  sol.compute_feasibility();
  rmm::device_scalar<i_t> is_feasible(sol.handle_ptr->get_stream());
  sol.test_variable_bounds(true, is_feasible.data());
  // test_variable_bounds clears is_feasible if the test is failed
  if (!is_feasible.value(sol.handle_ptr->get_stream())) {
    CUOPT_LOG_ERROR(
      "Solution is not feasible due to variable bounds, returning infeasible solution!");
    context.stats.total_solve_time = timer_.elapsed_time();
    context.problem_ptr->post_process_solution(sol);
    return sol;
  }
  context.stats.total_solve_time = timer_.elapsed_time();
  context.problem_ptr->post_process_solution(sol);
  dm.rins.stop_rins();
  return sol;
}

// Original feasibility jump has only double
#if MIP_INSTANTIATE_FLOAT
template class mip_solver_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class mip_solver_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
