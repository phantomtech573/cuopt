/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../linear_programming/utilities/pdlp_test_utilities.cuh"
#include "determinism_utils.cuh"
#include "mip_utils.cuh"

#include <cuopt/error.hpp>
#include <cuopt/linear_programming/pdlp/pdlp_hyper_params.cuh>
#include <cuopt/linear_programming/solve.hpp>
#include <cuopt/linear_programming/utilities/internals.hpp>
#include <mip_heuristics/diversity/diversity_manager.cuh>
#include <mip_heuristics/feasibility_jump/feasibility_jump.cuh>
#include <mip_heuristics/local_search/local_search.cuh>
#include <mip_heuristics/relaxed_lp/relaxed_lp.cuh>
#include <mip_heuristics/solution/solution.cuh>
#include <mip_heuristics/solver_context.cuh>
#include <mps_parser/parser.hpp>
#include <pdlp/initial_scaling_strategy/initial_scaling.cuh>
#include <pdlp/pdlp.cuh>
#include <pdlp/restart_strategy/pdlp_restart_strategy.cuh>
#include <pdlp/step_size_strategy/adaptive_step_size_strategy.hpp>
#include <pdlp/utilities/problem_checking.cuh>
#include <utilities/common_utils.hpp>
#include <utilities/seed_generator.cuh>

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <thrust/count.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>

#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace cuopt::linear_programming::test {

void init_handler(const raft::handle_t* handle_ptr)
{
  // Init cuBlas / cuSparse context here to avoid having it during solving time
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublassetpointermode(
    handle_ptr->get_cublas_handle(), CUBLAS_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsesetpointermode(
    handle_ptr->get_cusparse_handle(), CUSPARSE_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
}

static void setup_device_symbols(rmm::cuda_stream_view stream_view) { (void)stream_view; }

enum local_search_mode_t {
  FP = 0,
  STAGED_FP,
  FJ_LINE_SEGMENT,
  FJ_ON_ZERO,
  FJ_ANNEALING,
};

// Helper function to setup MIP solver and run FJ with given settings and initial solution
static uint32_t run_fp(std::string test_instance, local_search_mode_t mode)
{
  const raft::handle_t handle_{};
  std::cout << "Running: " << test_instance << std::endl;

  auto path = cuopt::test::get_rapids_dataset_root_dir() + ("/mip/" + test_instance);
  cuopt::mps_parser::mps_data_model_t<int, double> mps_problem =
    cuopt::mps_parser::parse_mps<int, double>(path, false);
  handle_.sync_stream();
  auto op_problem = mps_data_model_to_optimization_problem(&handle_, mps_problem);
  problem_checking_t<int, double>::check_problem_representation(op_problem);

  init_handler(op_problem.get_handle_ptr());
  // run the problem constructor of MIP, so that we do bounds standardization
  detail::problem_t<int, double> problem(op_problem);
  problem.preprocess_problem();

  setup_device_symbols(op_problem.get_handle_ptr()->get_stream());

  pdlp_hyper_params::pdlp_hyper_params_t hyper_params{};
  detail::pdlp_initial_scaling_strategy_t<int, double> scaling(&handle_,
                                                               problem,
                                                               10,
                                                               1.0,
                                                               problem.reverse_coefficients,
                                                               problem.reverse_offsets,
                                                               problem.reverse_constraints,
                                                               nullptr,
                                                               hyper_params,
                                                               true);

  auto settings             = mip_solver_settings_t<int, double>{};
  settings.time_limit       = 30.;
  settings.determinism_mode = CUOPT_MODE_DETERMINISTIC;
  auto timer                = cuopt::timer_t(30);
  detail::mip_solver_t<int, double> solver(problem, settings, scaling, timer);
  problem.tolerances = settings.get_tolerances();

  rmm::device_uvector<double> lp_optimal_solution(problem.n_variables,
                                                  problem.handle_ptr->get_stream());
  thrust::fill(problem.handle_ptr->get_thrust_policy(),
               lp_optimal_solution.begin(),
               lp_optimal_solution.end(),
               0.0);
  detail::lp_state_t<int, double>& lp_state = problem.lp_state;
  // resize because some constructor might be called before the presolve
  lp_state.resize(problem, problem.handle_ptr->get_stream());
  detail::relaxed_lp_settings_t lp_settings{};
  lp_settings.time_limit            = std::numeric_limits<double>::max();
  lp_settings.tolerance             = 1e-6;
  lp_settings.return_first_feasible = false;
  lp_settings.save_state            = false;
  // lp_settings.iteration_limit       = 5;
  auto lp_result =
    detail::get_relaxed_lp_solution(problem, lp_optimal_solution, lp_state, lp_settings);
  EXPECT_EQ(lp_result.get_termination_status(), pdlp_termination_status_t::Optimal);
  clamp_within_var_bounds(lp_optimal_solution, &problem, problem.handle_ptr);

  // return detail::compute_hash(lp_optimal_solution);

  detail::local_search_t<int, double> local_search(solver.context, lp_optimal_solution);

  detail::solution_t<int, double> solution(problem);
  solution.assign_random_within_bounds();
  solution.compute_feasibility();

  printf("Model fingerprint: 0x%x\n", problem.get_fingerprint());
  printf("LP optimal hash: 0x%x\n",
         detail::compute_hash(make_span(lp_optimal_solution), problem.handle_ptr->get_stream()));
  printf("running mode: %d\n", mode);

  work_limit_context_t work_limit_context("LocalSearch");
  local_search.fp.timer = work_limit_timer_t(work_limit_context, 6000);

  detail::ls_config_t<int, double> ls_config{};

  if (mode == local_search_mode_t::FP) {
    bool is_feasible = false;
    int iterations   = 0;
    while (true) {
      is_feasible = local_search.fp.run_single_fp_descent(solution);
      printf("fp_loop it %d, is_feasible %d\n", iterations, is_feasible);
      // if feasible return true
      if (is_feasible) {
        break;
      }
      // if not feasible, it means it is a cycle
      else {
        is_feasible = local_search.fp.restart_fp(solution);
        if (is_feasible) { break; }
      }
      iterations++;
    }
  } else if (mode == local_search_mode_t::FJ_LINE_SEGMENT) {
    local_search.run_fj_line_segment(
      solution, work_limit_timer_t(work_limit_context, 6000), ls_config);
  } else if (mode == local_search_mode_t::FJ_ON_ZERO) {
    local_search.run_fj_on_zero(solution, work_limit_timer_t(work_limit_context, 6000));
  } else if (mode == local_search_mode_t::FJ_ANNEALING) {
    local_search.run_fj_annealing(
      solution, work_limit_timer_t(work_limit_context, 6000), ls_config);
  }

  std::vector<uint32_t> hashes;
  hashes.push_back(detail::compute_hash(solution.get_host_assignment()));
  printf("hashes: 0x%x, hash of the hash: 0x%x\n", hashes[0], detail::compute_hash(hashes));

  return detail::compute_hash(hashes);
  // return {host_copy(solution_vector, problem.handle_ptr->get_stream()), iterations};
}

static uint32_t run_fp_check_determinism(std::string test_instance,
                                         local_search_mode_t mode,
                                         unsigned long seed)
{
  cuopt::seed_generator::set_seed(seed);

  return run_fp(test_instance, mode);

  //    auto state     = run_fp(test_instance, fj_settings);
  //    auto& solution = state.solution;

  //    CUOPT_LOG_DEBUG("%s: Solution generated with FJ: is_feasible %d, objective %g (raw %g)",
  //                    test_instance.c_str(),
  //                    solution.get_feasible(),
  //                    solution.get_user_objective(),
  //                    solution.get_objective());

  //    static auto first_val = solution.get_user_objective();
}

class LocalSearchTestParams : public testing::TestWithParam<std::tuple<local_search_mode_t>> {};

TEST_P(LocalSearchTestParams, local_search_operator_determinism)
{
  cuopt::default_logger().set_pattern("[%n] [%-6l] %v");

  spin_stream_raii_t spin_stream_1;
  spin_stream_raii_t spin_stream_2;

  auto mode = std::get<0>(GetParam());

  for (const auto& instance : {
         //"thor50dday.mps",
         "gen-ip054.mps",
         "50v-10.mps",
         "seymour1.mps",
         "rmatr200-p5.mps",
         "tr12-30.mps",
         //"sct2.mps",
         //"uccase9.mps"
       }) {
    // for (int i = 0; i < 10; i++)
    //  while (true) {
    //    run_fp_check_determinism(instance, 1000);
    //  }

    unsigned long seed = std::getenv("CUOPT_SEED")
                           ? (unsigned long)std::stoi(std::getenv("CUOPT_SEED"))
                           : (unsigned long)std::random_device{}();
    std::cerr << "Tested with seed " << seed << "\n";
    uint32_t gold_hash = 0;
    for (int i = 0; i < 5; ++i) {
      uint32_t hash = run_fp_check_determinism(instance, mode, seed);
      if (i == 0) {
        gold_hash = hash;
        printf("Gold hash: 0x%x\n", gold_hash);
      } else {
        ASSERT_EQ(hash, gold_hash);
        printf("Hash: 0x%x\n", hash);
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(LocalSearchTests,
                         LocalSearchTestParams,
                         testing::Values(std::make_tuple(local_search_mode_t::FP),
                                         std::make_tuple(local_search_mode_t::FJ_LINE_SEGMENT),
                                         std::make_tuple(local_search_mode_t::FJ_ON_ZERO),
                                         std::make_tuple(local_search_mode_t::FJ_ANNEALING)));

}  // namespace cuopt::linear_programming::test
