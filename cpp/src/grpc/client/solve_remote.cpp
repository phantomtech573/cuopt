/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/linear_programming/cpu_optimization_problem.hpp>
#include <cuopt/linear_programming/cpu_optimization_problem_solution.hpp>
#include <cuopt/linear_programming/cpu_pdlp_warm_start_data.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <utilities/logger.hpp>
#include "grpc_client.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace cuopt::linear_programming {

// ============================================================================
// Helper function to get gRPC server address from environment variables
// ============================================================================

static std::string get_grpc_server_address()
{
  const char* host = std::getenv("CUOPT_REMOTE_HOST");
  const char* port = std::getenv("CUOPT_REMOTE_PORT");

  if (host == nullptr || port == nullptr) {
    throw std::runtime_error(
      "Remote execution enabled but CUOPT_REMOTE_HOST and/or CUOPT_REMOTE_PORT not set");
  }

  return std::string(host) + ":" + std::string(port);
}

static int64_t parse_env_int64(const char* name, int64_t default_value)
{
  const char* val = std::getenv(name);
  if (val == nullptr) return default_value;
  try {
    return std::stoll(val);
  } catch (...) {
    return default_value;
  }
}

// Apply env-var overrides for chunked transfer configuration.
//   CUOPT_CHUNK_SIZE         – bytes per chunk for chunked array uploads and downloads
//   CUOPT_MAX_MESSAGE_BYTES  – client-side max gRPC message size
//   CUOPT_GRPC_DEBUG         – enable debug/throughput logging to stderr (0|1)
static void apply_env_overrides(grpc_client_config_t& config)
{
  config.chunk_size_bytes  = parse_env_int64("CUOPT_CHUNK_SIZE", config.chunk_size_bytes);
  config.max_message_bytes = parse_env_int64("CUOPT_MAX_MESSAGE_BYTES", config.max_message_bytes);
  config.enable_debug_log  = (parse_env_int64("CUOPT_GRPC_DEBUG", 0) != 0);

  CUOPT_LOG_INFO("gRPC client config: chunk_size=%lld max_message=%lld",
                 static_cast<long long>(config.chunk_size_bytes),
                 static_cast<long long>(config.max_message_bytes));
}

// ============================================================================
// Remote execution via gRPC
// ============================================================================

template <typename i_t, typename f_t>
std::unique_ptr<lp_solution_interface_t<i_t, f_t>> solve_lp_remote(
  cpu_optimization_problem_t<i_t, f_t> const& cpu_problem,
  pdlp_solver_settings_t<i_t, f_t> const& settings,
  bool problem_checking,
  bool use_pdlp_solver_mode)
{
  init_logger_t log(settings.log_file, settings.log_to_console);

  CUOPT_LOG_INFO("solve_lp_remote (CPU problem) - connecting to gRPC server");

  // Build gRPC client configuration
  grpc_client_config_t config;
  config.server_address = get_grpc_server_address();
  apply_env_overrides(config);

  // Configure log streaming based on settings
  if (settings.log_to_console) {
    config.stream_logs  = true;
    config.log_callback = [](const std::string& line) { std::cout << line << std::endl; };
  }

  // Create client and connect
  grpc_client_t client(config);
  if (!client.connect()) {
    throw std::runtime_error("Failed to connect to gRPC server: " + client.get_last_error());
  }

  CUOPT_LOG_INFO("solve_lp_remote - connected to %s, submitting problem",
                 config.server_address.c_str());

  // Call the remote solver
  auto result = client.solve_lp(cpu_problem, settings);

  if (!result.success) {
    throw std::runtime_error("Remote LP solve failed: " + result.error_message);
  }

  CUOPT_LOG_INFO("solve_lp_remote - solve completed successfully");

  return std::move(result.solution);
}

template <typename i_t, typename f_t>
std::unique_ptr<mip_solution_interface_t<i_t, f_t>> solve_mip_remote(
  cpu_optimization_problem_t<i_t, f_t> const& cpu_problem,
  mip_solver_settings_t<i_t, f_t> const& settings)
{
  init_logger_t log(settings.log_file, settings.log_to_console);

  CUOPT_LOG_INFO("solve_mip_remote (CPU problem) - connecting to gRPC server");

  // Build gRPC client configuration
  grpc_client_config_t config;
  config.server_address = get_grpc_server_address();
  apply_env_overrides(config);

  // Configure log streaming based on settings
  if (settings.log_to_console) {
    config.stream_logs  = true;
    config.log_callback = [](const std::string& line) { std::cout << line << std::endl; };
  }

  // Check if user has set incumbent callbacks
  auto mip_callbacks   = settings.get_mip_callbacks();
  bool has_incumbents  = !mip_callbacks.empty();
  bool enable_tracking = has_incumbents;

  // Initialize callbacks with problem size (needed for Python callbacks to work correctly)
  // The local MIP solver does this in solve.cu, but for remote solves we need to do it here
  if (has_incumbents) {
    size_t n_vars = cpu_problem.get_n_variables();
    for (auto* callback : mip_callbacks) {
      if (callback != nullptr) { callback->template setup<f_t>(n_vars); }
    }
  }

  // Set up incumbent callback forwarding
  if (has_incumbents) {
    CUOPT_LOG_INFO("solve_mip_remote - setting up inline incumbent callback forwarding");
    config.incumbent_callback = [&mip_callbacks](int64_t index,
                                                 double objective,
                                                 const std::vector<double>& solution) -> bool {
      // Forward incumbent to all user callbacks (invoked from main thread with GIL)
      for (auto* callback : mip_callbacks) {
        if (callback != nullptr &&
            callback->get_type() == internals::base_solution_callback_type::GET_SOLUTION) {
          auto* get_callback = static_cast<internals::get_solution_callback_t*>(callback);
          // Copy solution to non-const buffer for callback interface
          std::vector<double> solution_copy = solution;
          double obj_copy                   = objective;
          double bound_copy                 = objective;  // Use objective as bound for incumbent
          get_callback->get_solution(
            solution_copy.data(), &obj_copy, &bound_copy, callback->get_user_data());
        }
      }
      return true;  // Continue solving
    };
  }

  // Create client and connect
  grpc_client_t client(config);
  if (!client.connect()) {
    throw std::runtime_error("Failed to connect to gRPC server: " + client.get_last_error());
  }

  CUOPT_LOG_INFO("solve_mip_remote - connected to %s, submitting problem (incumbents=%s)",
                 config.server_address.c_str(),
                 enable_tracking ? "enabled" : "disabled");

  // Call the remote solver
  auto result = client.solve_mip(cpu_problem, settings, enable_tracking);

  if (!result.success) {
    throw std::runtime_error("Remote MIP solve failed: " + result.error_message);
  }

  CUOPT_LOG_INFO("solve_mip_remote - solve completed successfully");

  return std::move(result.solution);
}

// Explicit template instantiations for remote execution stubs
template std::unique_ptr<lp_solution_interface_t<int, double>> solve_lp_remote(
  cpu_optimization_problem_t<int, double> const&,
  pdlp_solver_settings_t<int, double> const&,
  bool,
  bool);

template std::unique_ptr<mip_solution_interface_t<int, double>> solve_mip_remote(
  cpu_optimization_problem_t<int, double> const&, mip_solver_settings_t<int, double> const&);

}  // namespace cuopt::linear_programming
