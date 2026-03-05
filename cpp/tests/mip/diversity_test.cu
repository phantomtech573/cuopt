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

static uint32_t test_full_run_determinism(std::string path,
                                          unsigned long seed = std::random_device{}())
{
  const raft::handle_t handle_{};

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
  settings.time_limit       = 3000.;
  settings.work_limit       = 10;  // about 10 seconds of runtime
  settings.determinism_mode = CUOPT_MODE_DETERMINISTIC;
  settings.heuristics_only  = true;
  auto timer                = cuopt::timer_t(3000);
  detail::mip_solver_t<int, double> solver(problem, settings, scaling, timer);
  problem.tolerances = settings.get_tolerances();

  detail::diversity_manager_t<int, double> diversity_manager(solver.context);
  work_limit_context_t work_limit_context("DiversityManager");
  diversity_manager.timer = work_limit_timer_t(work_limit_context, 60000);
  diversity_manager.run_solver();

  std::vector<uint32_t> hashes;
  auto pop = diversity_manager.get_population_pointer();
  for (const auto& sol : pop->population_to_vector()) {
    hashes.push_back(sol.get_hash());
  }

  uint32_t final_hash = detail::compute_hash(hashes);
  printf("%s: final hash: 0x%x, pop size %d\n",
         path.c_str(),
         final_hash,
         (int)pop->population_to_vector().size());
  return final_hash;
}

static uint32_t test_initial_solution_determinism(std::string path,
                                                  unsigned long seed = std::random_device{}())
{
  const raft::handle_t handle_{};

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
  settings.time_limit       = 3000.;
  settings.determinism_mode = CUOPT_MODE_DETERMINISTIC;
  settings.heuristics_only  = true;
  auto timer                = cuopt::timer_t(3000);
  detail::mip_solver_t<int, double> solver(problem, settings, scaling, timer);
  problem.tolerances = settings.get_tolerances();

  detail::diversity_manager_t<int, double> diversity_manager(solver.context);
  work_limit_context_t work_limit_context("DiversityManager");
  diversity_manager.timer = work_limit_timer_t(work_limit_context, 60000);
  diversity_manager.diversity_config.initial_solution_only = true;
  diversity_manager.run_solver();

  std::vector<uint32_t> hashes;
  auto pop = diversity_manager.get_population_pointer();
  for (const auto& sol : pop->population_to_vector()) {
    hashes.push_back(sol.get_hash());
  }

  uint32_t final_hash = detail::compute_hash(hashes);
  printf("%s: final hash: 0x%x, pop size %d\n",
         path.c_str(),
         final_hash,
         (int)pop->population_to_vector().size());
  return final_hash;
}

static uint32_t test_recombiners_determinism(std::string path,
                                             unsigned long seed = std::random_device{}())
{
  const raft::handle_t handle_{};

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
  settings.time_limit       = 3000.;
  settings.determinism_mode = CUOPT_MODE_DETERMINISTIC;
  settings.heuristics_only  = true;
  auto timer                = cuopt::timer_t(3000);
  detail::mip_solver_t<int, double> solver(problem, settings, scaling, timer);
  problem.tolerances = settings.get_tolerances();

  detail::diversity_manager_t<int, double> diversity_manager(solver.context);
  work_limit_context_t work_limit_context("DiversityManager");
  diversity_manager.timer                    = work_limit_timer_t(work_limit_context, 60000);
  diversity_manager.diversity_config.dry_run = true;
  diversity_manager.run_solver();

  // Generate a population by running FJ on random starting points
  // recombine a few solutions, observe the output
  for (int i = diversity_manager.population.current_size(); i < 3; ++i) {
    detail::solution_t<int, double> random_initial_solution(problem);
    random_initial_solution.assign_random_within_bounds();
    detail::fj_settings_t fj_settings;
    fj_settings.feasibility_run = false;
    fj_settings.iteration_limit = 1000 + i * 100;
    fj_settings.seed            = seed + i;
    auto solution =
      run_fj(problem, fj_settings, fj_tweaks_t{}, random_initial_solution.get_host_assignment())
        .solution;
    printf("population %d hash: 0x%x\n", i, solution.get_hash());
    diversity_manager.population.add_solution(std::move(solution));
  }

  auto pop_vector = diversity_manager.get_population_pointer()->population_to_vector();

  std::vector<uint32_t> hashes;

  static std::map<std::tuple<std::string, int, int, detail::recombiner_enum_t>, uint32_t> hash_map;

  for (auto recombiner : {detail::recombiner_enum_t::LINE_SEGMENT,
                          detail::recombiner_enum_t::BOUND_PROP,
                          detail::recombiner_enum_t::FP}) {
    for (int i = 1; i < (int)pop_vector.size(); i++) {
      for (int j = i + 1; j < (int)pop_vector.size(); j++) {
        printf("recombining %d and %d w/ recombiner %s\n",
               i,
               j,
               detail::all_recombine_stats::recombiner_labels[(int)recombiner]);
        auto [offspring, success] =
          diversity_manager.recombine(pop_vector[i], pop_vector[j], recombiner);
        auto offspring_hash = offspring.get_hash();
        printf("for %d,%d: offspring hash: 0x%x, parent 1 hash: 0x%x, parent 2 hash: 0x%x\n",
               i,
               j,
               offspring_hash,
               pop_vector[i].get_hash(),
               pop_vector[j].get_hash());
        if (hash_map.find(std::make_tuple(path, i, j, recombiner)) == hash_map.end()) {
          hash_map[std::make_tuple(path, i, j, recombiner)] = offspring_hash;
        } else {
          if (hash_map[std::make_tuple(path, i, j, recombiner)] != offspring_hash) {
            printf("%s: hash mismatch for %d,%d: %d != %d\n",
                   path.c_str(),
                   i,
                   j,
                   hash_map[std::make_tuple(path, i, j, recombiner)],
                   offspring_hash);
            exit(1);
          }
        }
        hashes.push_back(offspring_hash);
      }
    }
  }
  return detail::compute_hash(hashes);

  auto pop = diversity_manager.get_population_pointer();
  for (const auto& sol : pop->population_to_vector()) {
    hashes.push_back(sol.get_hash());
  }

  uint32_t final_hash = detail::compute_hash(hashes);
  printf("%s: final hash: 0x%x, pop size %d\n",
         path.c_str(),
         final_hash,
         (int)pop->population_to_vector().size());
  return final_hash;
}

class DiversityTestParams : public testing::TestWithParam<std::tuple<std::string>> {};

// TEST_P(DiversityTestParams, recombiners_deterministic)
// {
//   cuopt::default_logger().set_pattern("[%n] [%-6l] %v");

//   spin_stream_raii_t spin_stream_1;
//   spin_stream_raii_t spin_stream_2;

//   auto test_instance = std::get<0>(GetParam());
//   std::cout << "Running: " << test_instance << std::endl;
//   int seed =
//     std::getenv("CUOPT_SEED") ? std::stoi(std::getenv("CUOPT_SEED")) : std::random_device{}();
//   std::cerr << "Tested with seed " << seed << "\n";
//   auto path     = make_path_absolute(test_instance);
//   test_instance = std::getenv("CUOPT_INSTANCE") ? std::getenv("CUOPT_INSTANCE") : test_instance;
//   uint32_t gold_hash = 0;
//   for (int i = 0; i < 2; ++i) {
//     cuopt::seed_generator::set_seed(seed);
//     std::cout << "Running " << test_instance << " " << i << std::endl;
//     std::cout << "-------------------------------------------------------------\n";
//     auto hash = test_recombiners_determinism(path, seed);
//     if (i == 0) {
//       gold_hash = hash;
//       std::cout << "Gold hash: " << gold_hash << std::endl;
//     } else {
//       ASSERT_EQ(hash, gold_hash);
//     }
//   }
// }

// TEST_P(DiversityTestParams, initial_solution_deterministic)
// {
//   cuopt::default_logger().set_pattern("[%n] [%-6l] %v");

//   spin_stream_raii_t spin_stream_1;
//   spin_stream_raii_t spin_stream_2;

//   auto test_instance = std::get<0>(GetParam());
//   std::cout << "Running: " << test_instance << std::endl;
//   int seed =
//     std::getenv("CUOPT_SEED") ? std::stoi(std::getenv("CUOPT_SEED")) : std::random_device{}();
//   std::cerr << "Tested with seed " << seed << "\n";
//   auto path     = make_path_absolute(test_instance);
//   test_instance = std::getenv("CUOPT_INSTANCE") ? std::getenv("CUOPT_INSTANCE") : test_instance;
//   uint32_t gold_hash = 0;
//   for (int i = 0; i < 2; ++i) {
//     cuopt::seed_generator::set_seed(seed);
//     std::cout << "Running " << test_instance << " " << i << std::endl;
//     std::cout << "-------------------------------------------------------------\n";
//     auto hash = test_initial_solution_determinism(path, seed);
//     if (i == 0) {
//       gold_hash = hash;
//       std::cout << "Gold hash: " << gold_hash << std::endl;
//     } else {
//       ASSERT_EQ(hash, gold_hash);
//     }
//   }
// }

TEST_P(DiversityTestParams, full_run_deterministic)
{
  cuopt::default_logger().set_pattern("[%n] [%-6l] %v");

  spin_stream_raii_t spin_stream_1;
  spin_stream_raii_t spin_stream_2;

  auto test_instance = std::get<0>(GetParam());
  std::cout << "Running: " << test_instance << std::endl;
  int seed =
    std::getenv("CUOPT_SEED") ? std::stoi(std::getenv("CUOPT_SEED")) : std::random_device{}();
  std::cerr << "Tested with seed " << seed << "\n";
  auto path = make_path_absolute(test_instance);
  if (std::getenv("CUOPT_INSTANCE")) {
    test_instance = std::getenv("CUOPT_INSTANCE");
    path          = make_path_absolute(test_instance);
  }
  uint32_t gold_hash = 0;
  for (int i = 0; i < 2; ++i) {
    cuopt::seed_generator::set_seed(seed);
    std::cout << "Running " << test_instance << " " << i << std::endl;
    std::cout << "-------------------------------------------------------------\n";
    auto hash = test_full_run_determinism(path, seed);
    if (i == 0) {
      gold_hash = hash;
      std::cout << "Gold hash: " << gold_hash << std::endl;
    } else {
      ASSERT_EQ(hash, gold_hash);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(DiversityTest,
                         DiversityTestParams,
                         testing::Values(  // std::make_tuple("gen-ip054.mps"),
                                           // std::make_tuple("pk1.mps")
                           std::make_tuple("uccase9.mps"),
                           // std::make_tuple("mip/sct2.mps")
                           // std::make_tuple("mip/thor50dday.mps")
                           // std::make_tuple("uccase9.mps"),
                           // std::make_tuple("mip/neos5.mps")
                           std::make_tuple("50v-10.mps")
                           // std::make_tuple("rmatr200-p5.mps")
                           ));

}  // namespace cuopt::linear_programming::test
