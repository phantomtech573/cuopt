/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#ifdef CUOPT_ENABLE_GRPC

#include "grpc_incumbent_proto.hpp"
#include "grpc_pipe_serialization.hpp"
#include "grpc_server_types.hpp"

class IncumbentPipeCallback : public cuopt::internals::get_solution_callback_t {
 public:
  IncumbentPipeCallback(std::string job_id, int fd, size_t num_vars, bool is_float)
    : job_id_(std::move(job_id)), fd_(fd)
  {
    n_variables = num_vars;
    isFloat     = is_float;
  }

  void get_solution(void* data,
                    void* objective_value,
                    void* /*solution_bound*/,
                    void* /*user_data*/) override
  {
    if (fd_ < 0 || n_variables == 0) { return; }

    double objective = 0.0;
    std::vector<double> assignment;
    assignment.resize(n_variables);

    if (isFloat) {
      const float* float_data = static_cast<const float*>(data);
      for (size_t i = 0; i < n_variables; ++i) {
        assignment[i] = static_cast<double>(float_data[i]);
      }
      objective = static_cast<double>(*static_cast<const float*>(objective_value));
    } else {
      const double* double_data = static_cast<const double*>(data);
      std::copy(double_data, double_data + n_variables, assignment.begin());
      objective = *static_cast<const double*>(objective_value);
    }

    auto buffer = build_incumbent_proto(job_id_, objective, assignment);
    std::cout << "[Worker] Incumbent callback job_id=" << job_id_ << " obj=" << objective
              << " vars=" << assignment.size() << "\n";
    std::cout.flush();
    send_incumbent_pipe(fd_, buffer);
  }

 private:
  std::string job_id_;
  int fd_;
};

static void store_simple_result(const std::string& job_id,
                                int worker_id,
                                int status,
                                const char* error_message)
{
  for (size_t i = 0; i < MAX_RESULTS; ++i) {
    if (!result_queue[i].ready) {
      copy_cstr(result_queue[i].job_id, job_id);
      result_queue[i].status       = status;
      result_queue[i].data_size    = 0;
      result_queue[i].worker_index = worker_id;
      copy_cstr(result_queue[i].error_message, error_message);
      result_queue[i].error_message[sizeof(result_queue[i].error_message) - 1] = '\0';
      result_queue[i].retrieved                                                = false;
      result_queue[i].ready                                                    = true;
      break;
    }
  }
}

void worker_process(int worker_id)
{
  std::cout << "[Worker " << worker_id << "] Started (PID: " << getpid() << ")\n";

  shm_ctrl->active_workers++;

  while (!shm_ctrl->shutdown_requested) {
    int job_slot = -1;
    for (size_t i = 0; i < MAX_JOBS; ++i) {
      if (job_queue[i].ready && !job_queue[i].claimed) {
        bool expected = false;
        if (job_queue[i].claimed.compare_exchange_strong(expected, true)) {
          job_queue[i].worker_pid   = getpid();
          job_queue[i].worker_index = worker_id;
          job_slot                  = i;
          break;
        }
      }
    }

    if (job_slot < 0) {
      usleep(10000);
      continue;
    }

    JobQueueEntry& job = job_queue[job_slot];
    std::string job_id(job.job_id);
    bool is_mip = (job.problem_type == 1);

    if (job.cancelled) {
      std::cout << "[Worker " << worker_id << "] Job cancelled before processing: " << job_id
                << "\n";
      std::cout.flush();

      store_simple_result(job_id, worker_id, 2, "Job was cancelled");

      job.worker_pid   = 0;
      job.worker_index = -1;
      job.data_sent    = false;
      job.ready        = false;
      job.claimed      = false;
      job.cancelled    = false;
      continue;
    }

    std::cout << "[Worker " << worker_id << "] Processing job: " << job_id
              << " (type: " << (is_mip ? "MIP" : "LP") << ")\n";
    std::cout.flush();

    std::string log_file = get_log_file_path(job_id);

    std::cout << "[Worker] Creating raft::handle_t...\n" << std::flush;

    raft::handle_t handle;

    std::cout << "[Worker] Handle created, starting solve...\n" << std::flush;

    std::vector<uint8_t> request_data;
    int read_fd       = worker_pipes[worker_id].worker_read_fd;
    auto pipe_recv_t0 = std::chrono::steady_clock::now();
    bool read_success = recv_job_data_pipe(read_fd, job.data_size, request_data);
    if (read_success && config.verbose) {
      auto pipe_us = std::chrono::duration_cast<std::chrono::microseconds>(
                       std::chrono::steady_clock::now() - pipe_recv_t0)
                       .count();
      double pipe_sec = pipe_us / 1e6;
      double pipe_mb  = static_cast<double>(request_data.size()) / (1024.0 * 1024.0);
      double pipe_mbs = (pipe_sec > 0.0) ? (pipe_mb / pipe_sec) : 0.0;
      std::cout << "[THROUGHPUT] phase=pipe_job_recv bytes=" << request_data.size()
                << " elapsed_ms=" << std::fixed << std::setprecision(1) << (pipe_us / 1000.0)
                << " throughput_mb_s=" << std::setprecision(1) << pipe_mbs << "\n";
      std::cout.flush();
    }
    if (!read_success) {
      std::cerr << "[Worker " << worker_id << "] Failed to read job data from pipe\n";
    }

    if (!read_success) {
      store_simple_result(job_id, worker_id, 1, "Failed to read job data");
      job.worker_pid   = 0;
      job.worker_index = -1;
      job.data_sent    = false;
      job.ready        = false;
      job.claimed      = false;
      continue;
    }

    std::vector<uint8_t> result_data;
    std::string error_message;
    bool success = false;

    try {
      cpu_optimization_problem_t<int, double> cpu_problem;
      pdlp_solver_settings_t<int, double> lp_settings;
      mip_solver_settings_t<int, double> mip_settings;
      bool enable_incumbents_flag = true;

      cuopt::remote::SubmitJobRequest submit_request;
      if (submit_request.ParseFromArray(request_data.data(),
                                        static_cast<int>(request_data.size())) &&
          (submit_request.has_lp_request() || submit_request.has_mip_request())) {
        if (submit_request.has_lp_request()) {
          const auto& req = submit_request.lp_request();
          std::cout << "[Worker] IPC path: UNARY LP (" << request_data.size() << " bytes)\n"
                    << std::flush;
          map_proto_to_problem(req.problem(), cpu_problem);
          map_proto_to_pdlp_settings(req.settings(), lp_settings);
        } else {
          const auto& req = submit_request.mip_request();
          std::cout << "[Worker] IPC path: UNARY MIP (" << request_data.size() << " bytes)\n"
                    << std::flush;
          map_proto_to_problem(req.problem(), cpu_problem);
          map_proto_to_mip_settings(req.settings(), mip_settings);
          enable_incumbents_flag = req.has_enable_incumbents() ? req.enable_incumbents() : true;
        }
      } else {
        cuopt::remote::ChunkedProblemHeader chunked_header;
        std::map<int32_t, std::vector<uint8_t>> arrays;
        if (!deserialize_chunked_request_pipe_blob(
              request_data.data(), request_data.size(), chunked_header, arrays)) {
          throw std::runtime_error("Failed to deserialize chunked request from worker pipe");
        }
        std::cout << "[Worker] IPC path: CHUNKED (" << arrays.size() << " arrays, "
                  << request_data.size() << " bytes)\n"
                  << std::flush;

        if (chunked_header.has_lp_settings()) {
          map_proto_to_pdlp_settings(chunked_header.lp_settings(), lp_settings);
        }
        if (chunked_header.has_mip_settings()) {
          map_proto_to_mip_settings(chunked_header.mip_settings(), mip_settings);
        }
        enable_incumbents_flag = chunked_header.enable_incumbents();
        map_chunked_arrays_to_problem(chunked_header, arrays, cpu_problem);
      }

      std::cout << "[Worker] Problem reconstructed: " << cpu_problem.get_n_constraints()
                << " constraints, " << cpu_problem.get_n_variables() << " variables, "
                << cpu_problem.get_nnz() << " nonzeros\n"
                << std::flush;

      request_data.clear();
      request_data.shrink_to_fit();

      if (is_mip) {
        mip_settings.log_file       = log_file;
        mip_settings.log_to_console = config.log_to_console;

        std::unique_ptr<IncumbentPipeCallback> incumbent_cb;
        if (enable_incumbents_flag) {
          incumbent_cb = std::make_unique<IncumbentPipeCallback>(
            job_id,
            worker_pipes[worker_id].worker_incumbent_write_fd,
            cpu_problem.get_n_variables(),
            false);
          mip_settings.set_mip_callback(incumbent_cb.get());
          std::cout << "[Worker] Registered incumbent callback for job_id=" << job_id
                    << " n_vars=" << cpu_problem.get_n_variables() << "\n";
          std::cout.flush();
        }

        std::cout << "[Worker] Converting CPU problem to GPU problem...\n" << std::flush;
        auto gpu_problem = cpu_problem.to_optimization_problem(&handle);

        std::cout << "[Worker] Calling solve_mip...\n" << std::flush;
        auto gpu_solution = solve_mip(*gpu_problem, mip_settings);
        std::cout << "[Worker] solve_mip done\n" << std::flush;

        std::cout << "[Worker] Converting solution to CPU format...\n" << std::flush;

        const auto& device_solution = gpu_solution.get_solution();
        std::vector<double> host_solution(device_solution.size());
        cudaMemcpy(host_solution.data(),
                   device_solution.data(),
                   device_solution.size() * sizeof(double),
                   cudaMemcpyDeviceToHost);

        cpu_mip_solution_t<int, double> cpu_solution(
          std::move(host_solution),
          gpu_solution.get_termination_status(),
          gpu_solution.get_objective_value(),
          gpu_solution.get_mip_gap(),
          gpu_solution.get_solution_bound(),
          gpu_solution.get_total_solve_time(),
          gpu_solution.get_presolve_time(),
          gpu_solution.get_max_constraint_violation(),
          gpu_solution.get_max_int_violation(),
          gpu_solution.get_max_variable_bound_violation(),
          gpu_solution.get_num_nodes(),
          gpu_solution.get_num_simplex_iterations());

        cuopt::remote::ChunkedResultHeader result_header;
        populate_chunked_result_header_mip(cpu_solution, &result_header);
        auto result_arrays = collect_mip_solution_arrays(cpu_solution);
        result_data        = serialize_result_pipe_blob(result_header, result_arrays);
        std::cout << "[Worker] Result path: MIP solution -> " << result_arrays.size()
                  << " array(s), " << result_data.size() << " bytes\n"
                  << std::flush;
        success = true;
      } else {
        lp_settings.log_file       = log_file;
        lp_settings.log_to_console = config.log_to_console;

        std::cout << "[Worker] Converting CPU problem to GPU problem...\n" << std::flush;
        auto gpu_problem = cpu_problem.to_optimization_problem(&handle);

        std::cout << "[Worker] Calling solve_lp...\n" << std::flush;
        auto gpu_solution = solve_lp(*gpu_problem, lp_settings);
        std::cout << "[Worker] solve_lp done\n" << std::flush;

        std::cout << "[Worker] Converting solution to CPU format...\n" << std::flush;

        const auto& device_primal = gpu_solution.get_primal_solution();
        const auto& device_dual   = gpu_solution.get_dual_solution();
        auto& device_reduced_cost = gpu_solution.get_reduced_cost();

        std::vector<double> host_primal(device_primal.size());
        std::vector<double> host_dual(device_dual.size());
        std::vector<double> host_reduced_cost(device_reduced_cost.size());

        cudaMemcpy(host_primal.data(),
                   device_primal.data(),
                   device_primal.size() * sizeof(double),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(host_dual.data(),
                   device_dual.data(),
                   device_dual.size() * sizeof(double),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(host_reduced_cost.data(),
                   device_reduced_cost.data(),
                   device_reduced_cost.size() * sizeof(double),
                   cudaMemcpyDeviceToHost);

        auto term_info = gpu_solution.get_additional_termination_information();

        auto cpu_ws =
          convert_to_cpu_warmstart(gpu_solution.get_pdlp_warm_start_data(), handle.get_stream());

        cpu_lp_solution_t<int, double> cpu_solution(std::move(host_primal),
                                                    std::move(host_dual),
                                                    std::move(host_reduced_cost),
                                                    gpu_solution.get_termination_status(),
                                                    gpu_solution.get_objective_value(),
                                                    gpu_solution.get_dual_objective_value(),
                                                    term_info.solve_time,
                                                    term_info.l2_primal_residual,
                                                    term_info.l2_dual_residual,
                                                    term_info.gap,
                                                    term_info.number_of_steps_taken,
                                                    term_info.solved_by_pdlp,
                                                    std::move(cpu_ws));

        cuopt::remote::ChunkedResultHeader result_header;
        populate_chunked_result_header_lp(cpu_solution, &result_header);
        auto result_arrays = collect_lp_solution_arrays(cpu_solution);
        result_data        = serialize_result_pipe_blob(result_header, result_arrays);
        std::cout << "[Worker] Result path: LP solution -> " << result_arrays.size()
                  << " array(s), " << result_data.size() << " bytes\n"
                  << std::flush;
        success = true;
      }
    } catch (const std::exception& e) {
      error_message = std::string("Exception: ") + e.what();
    }

    {
      int result_slot = -1;
      for (size_t i = 0; i < MAX_RESULTS; ++i) {
        if (!result_queue[i].ready) {
          result_slot              = i;
          ResultQueueEntry& result = result_queue[i];
          copy_cstr(result.job_id, job_id);
          result.status       = success ? 0 : 1;
          result.data_size    = success ? result_data.size() : 0;
          result.worker_index = worker_id;
          if (!success) { copy_cstr(result.error_message, error_message); }
          result.retrieved = false;
          result.ready     = true;
          if (config.verbose) {
            std::cout << "[Worker " << worker_id << "] Enqueued result metadata for job " << job_id
                      << " in result_slot=" << result_slot << " status=" << result.status
                      << " data_size=" << result.data_size << "\n";
            std::cout.flush();
          }
          break;
        }
      }

      if (success && !result_data.empty() && result_slot >= 0) {
        int write_fd = worker_pipes[worker_id].worker_write_fd;
        if (config.verbose) {
          std::cout << "[Worker " << worker_id << "] Writing " << result_data.size()
                    << " bytes of result payload to pipe for job " << job_id << "\n";
          std::cout.flush();
        }
        auto pipe_result_t0 = std::chrono::steady_clock::now();
        bool write_success  = send_result_pipe(write_fd, result_data);
        if (write_success && config.verbose) {
          auto pipe_us = std::chrono::duration_cast<std::chrono::microseconds>(
                           std::chrono::steady_clock::now() - pipe_result_t0)
                           .count();
          double pipe_sec = pipe_us / 1e6;
          double pipe_mb  = static_cast<double>(result_data.size()) / (1024.0 * 1024.0);
          double pipe_mbs = (pipe_sec > 0.0) ? (pipe_mb / pipe_sec) : 0.0;
          std::cout << "[THROUGHPUT] phase=pipe_result_send bytes=" << result_data.size()
                    << " elapsed_ms=" << std::fixed << std::setprecision(1) << (pipe_us / 1000.0)
                    << " throughput_mb_s=" << std::setprecision(1) << pipe_mbs << "\n";
          std::cout.flush();
        }
        if (!write_success) {
          std::cerr << "[Worker " << worker_id << "] Failed to write result to pipe\n";
          std::cerr.flush();
          result_queue[result_slot].status = 1;
          copy_cstr(result_queue[result_slot].error_message, "Failed to write result to pipe");
        } else if (config.verbose) {
          std::cout << "[Worker " << worker_id << "] Finished writing result payload for job "
                    << job_id << "\n";
          std::cout.flush();
        }
      } else if (config.verbose) {
        std::cout << "[Worker " << worker_id << "] No result payload write needed for job "
                  << job_id << " (success=" << success << ", result_slot=" << result_slot
                  << ", payload_bytes=" << result_data.size() << ")\n";
        std::cout.flush();
      }
    }

    job.worker_pid   = 0;
    job.worker_index = -1;
    job.data_sent    = false;
    job.ready        = false;
    job.claimed      = false;
    job.cancelled    = false;

    std::cout << "[Worker " << worker_id << "] Completed job: " << job_id
              << " (success: " << success << ")\n";
  }

  shm_ctrl->active_workers--;
  std::cout << "[Worker " << worker_id << "] Stopped\n";
  _exit(0);
}

#endif  // CUOPT_ENABLE_GRPC
