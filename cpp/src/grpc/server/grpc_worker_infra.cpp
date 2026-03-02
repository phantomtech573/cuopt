/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#ifdef CUOPT_ENABLE_GRPC

#include "grpc_server_types.hpp"

void cleanup_shared_memory()
{
  if (job_queue) {
    munmap(job_queue, sizeof(JobQueueEntry) * MAX_JOBS);
    shm_unlink(SHM_JOB_QUEUE);
  }
  if (result_queue) {
    munmap(result_queue, sizeof(ResultQueueEntry) * MAX_RESULTS);
    shm_unlink(SHM_RESULT_QUEUE);
  }
  if (shm_ctrl) {
    munmap(shm_ctrl, sizeof(SharedMemoryControl));
    shm_unlink(SHM_CONTROL);
  }
}

bool create_worker_pipes(int worker_id)
{
  while (static_cast<int>(worker_pipes.size()) <= worker_id) {
    worker_pipes.push_back({-1, -1, -1, -1, -1, -1});
  }

  WorkerPipes& wp = worker_pipes[worker_id];

  int input_pipe[2];
  if (pipe(input_pipe) < 0) {
    std::cerr << "[Server] Failed to create input pipe for worker " << worker_id << "\n";
    return false;
  }
  wp.worker_read_fd = input_pipe[0];
  wp.to_worker_fd   = input_pipe[1];

  int output_pipe[2];
  if (pipe(output_pipe) < 0) {
    std::cerr << "[Server] Failed to create output pipe for worker " << worker_id << "\n";
    close(input_pipe[0]);
    close(input_pipe[1]);
    return false;
  }
  wp.from_worker_fd  = output_pipe[0];
  wp.worker_write_fd = output_pipe[1];

  int incumbent_pipe[2];
  if (pipe(incumbent_pipe) < 0) {
    std::cerr << "[Server] Failed to create incumbent pipe for worker " << worker_id << "\n";
    if (wp.worker_read_fd >= 0) close(wp.worker_read_fd);
    if (wp.to_worker_fd >= 0) close(wp.to_worker_fd);
    if (wp.from_worker_fd >= 0) close(wp.from_worker_fd);
    if (wp.worker_write_fd >= 0) close(wp.worker_write_fd);
    wp.worker_read_fd  = -1;
    wp.to_worker_fd    = -1;
    wp.from_worker_fd  = -1;
    wp.worker_write_fd = -1;
    return false;
  }
  wp.incumbent_from_worker_fd  = incumbent_pipe[0];
  wp.worker_incumbent_write_fd = incumbent_pipe[1];

  return true;
}

void close_worker_pipes_server(int worker_id)
{
  if (worker_id < 0 || worker_id >= static_cast<int>(worker_pipes.size())) return;

  WorkerPipes& wp = worker_pipes[worker_id];
  if (wp.to_worker_fd >= 0) {
    close(wp.to_worker_fd);
    wp.to_worker_fd = -1;
  }
  if (wp.from_worker_fd >= 0) {
    close(wp.from_worker_fd);
    wp.from_worker_fd = -1;
  }
  if (wp.incumbent_from_worker_fd >= 0) {
    close(wp.incumbent_from_worker_fd);
    wp.incumbent_from_worker_fd = -1;
  }
}

void close_worker_pipes_child_ends(int worker_id)
{
  if (worker_id < 0 || worker_id >= static_cast<int>(worker_pipes.size())) return;

  WorkerPipes& wp = worker_pipes[worker_id];
  if (wp.worker_read_fd >= 0) {
    close(wp.worker_read_fd);
    wp.worker_read_fd = -1;
  }
  if (wp.worker_write_fd >= 0) {
    close(wp.worker_write_fd);
    wp.worker_write_fd = -1;
  }
  if (wp.worker_incumbent_write_fd >= 0) {
    close(wp.worker_incumbent_write_fd);
    wp.worker_incumbent_write_fd = -1;
  }
}

pid_t spawn_worker(int worker_id, bool is_replacement)
{
  if (is_replacement) { close_worker_pipes_server(worker_id); }

  if (!create_worker_pipes(worker_id)) {
    std::cerr << "[Server] Failed to create pipes for "
              << (is_replacement ? "replacement worker " : "worker ") << worker_id << "\n";
    return -1;
  }

  pid_t pid = fork();
  if (pid < 0) {
    std::cerr << "[Server] Failed to fork " << (is_replacement ? "replacement worker " : "worker ")
              << worker_id << "\n";
    close_worker_pipes_server(worker_id);
    return -1;
  } else if (pid == 0) {
    for (int j = 0; j < static_cast<int>(worker_pipes.size()); ++j) {
      if (j != worker_id) {
        if (worker_pipes[j].worker_read_fd >= 0) close(worker_pipes[j].worker_read_fd);
        if (worker_pipes[j].worker_write_fd >= 0) close(worker_pipes[j].worker_write_fd);
        if (worker_pipes[j].to_worker_fd >= 0) close(worker_pipes[j].to_worker_fd);
        if (worker_pipes[j].from_worker_fd >= 0) close(worker_pipes[j].from_worker_fd);
        if (worker_pipes[j].incumbent_from_worker_fd >= 0) {
          close(worker_pipes[j].incumbent_from_worker_fd);
        }
        if (worker_pipes[j].worker_incumbent_write_fd >= 0) {
          close(worker_pipes[j].worker_incumbent_write_fd);
        }
      }
    }
    close(worker_pipes[worker_id].to_worker_fd);
    close(worker_pipes[worker_id].from_worker_fd);
    if (worker_pipes[worker_id].incumbent_from_worker_fd >= 0) {
      close(worker_pipes[worker_id].incumbent_from_worker_fd);
      worker_pipes[worker_id].incumbent_from_worker_fd = -1;
    }
    worker_process(worker_id);
    _exit(0);
  }

  close_worker_pipes_child_ends(worker_id);
  return pid;
}

void spawn_workers()
{
  for (int i = 0; i < config.num_workers; ++i) {
    pid_t pid = spawn_worker(i, false);
    if (pid < 0) { continue; }
    worker_pids.push_back(pid);
  }
}

void wait_for_workers()
{
  for (pid_t pid : worker_pids) {
    int status;
    waitpid(pid, &status, 0);
  }
  worker_pids.clear();
}

pid_t spawn_single_worker(int worker_id) { return spawn_worker(worker_id, true); }

void mark_worker_jobs_failed(pid_t dead_worker_pid)
{
  for (size_t i = 0; i < MAX_JOBS; ++i) {
    if (job_queue[i].ready && job_queue[i].claimed && job_queue[i].worker_pid == dead_worker_pid) {
      std::string job_id(job_queue[i].job_id);
      bool was_cancelled = job_queue[i].cancelled;

      if (was_cancelled) {
        std::cerr << "[Server] Worker " << dead_worker_pid
                  << " killed for cancelled job: " << job_id << "\n";
      } else {
        std::cerr << "[Server] Worker " << dead_worker_pid
                  << " died while processing job: " << job_id << "\n";
      }

      {
        std::lock_guard<std::mutex> lock(pending_data_mutex);
        pending_job_data.erase(job_id);
      }

      for (size_t j = 0; j < MAX_RESULTS; ++j) {
        if (!result_queue[j].ready) {
          copy_cstr(result_queue[j].job_id, job_id);
          result_queue[j].status       = was_cancelled ? 2 : 1;
          result_queue[j].data_size    = 0;
          result_queue[j].worker_index = -1;
          copy_cstr(result_queue[j].error_message,
                    was_cancelled ? "Job was cancelled" : "Worker process died unexpectedly");
          result_queue[j].retrieved = false;
          result_queue[j].ready     = true;
          break;
        }
      }

      job_queue[i].worker_pid   = 0;
      job_queue[i].worker_index = -1;
      job_queue[i].data_sent    = false;
      job_queue[i].ready        = false;
      job_queue[i].claimed      = false;
      job_queue[i].cancelled    = false;

      {
        std::lock_guard<std::mutex> lock(tracker_mutex);
        auto it = job_tracker.find(job_id);
        if (it != job_tracker.end()) {
          if (was_cancelled) {
            it->second.status        = JobStatus::CANCELLED;
            it->second.error_message = "Job was cancelled";
          } else {
            it->second.status        = JobStatus::FAILED;
            it->second.error_message = "Worker process died unexpectedly";
          }
        }
      }
    }
  }
}

#endif  // CUOPT_ENABLE_GRPC
