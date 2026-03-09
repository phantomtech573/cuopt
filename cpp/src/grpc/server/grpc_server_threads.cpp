/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#ifdef CUOPT_ENABLE_GRPC

#include "grpc_incumbent_proto.hpp"
#include "grpc_pipe_serialization.hpp"
#include "grpc_server_types.hpp"

void worker_monitor_thread()
{
  std::cout << "[Server] Worker monitor thread started\n";
  std::cout.flush();

  while (keep_running) {
    for (size_t i = 0; i < worker_pids.size(); ++i) {
      pid_t pid = worker_pids[i];
      if (pid <= 0) continue;

      int status;
      pid_t result = waitpid(pid, &status, WNOHANG);

      if (result == pid) {
        int exit_code  = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
        bool signaled  = WIFSIGNALED(status);
        int signal_num = signaled ? WTERMSIG(status) : 0;

        if (signaled) {
          std::cerr << "[Server] Worker " << pid << " killed by signal " << signal_num << "\n";
          std::cerr.flush();
        } else if (exit_code != 0) {
          std::cerr << "[Server] Worker " << pid << " exited with code " << exit_code << "\n";
          std::cerr.flush();
        } else {
          if (shm_ctrl && shm_ctrl->shutdown_requested) {
            worker_pids[i] = 0;
            continue;
          }
          std::cerr << "[Server] Worker " << pid << " exited unexpectedly\n";
          std::cerr.flush();
        }

        mark_worker_jobs_failed(pid);

        if (keep_running && shm_ctrl && !shm_ctrl->shutdown_requested) {
          pid_t new_pid = spawn_single_worker(static_cast<int>(i));
          if (new_pid > 0) {
            worker_pids[i] = new_pid;
            std::cout << "[Server] Restarted worker " << i << " with PID " << new_pid << "\n";
            std::cout.flush();
          } else {
            worker_pids[i] = 0;
          }
        } else {
          worker_pids[i] = 0;
        }
      }
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  std::cout << "[Server] Worker monitor thread stopped\n";
  std::cout.flush();
}

void result_retrieval_thread()
{
  std::cout << "[Server] Result retrieval thread started\n";
  std::cout.flush();

  while (keep_running) {
    bool found = false;

    for (size_t i = 0; i < MAX_JOBS; ++i) {
      if (job_queue[i].ready && job_queue[i].claimed && !job_queue[i].data_sent &&
          !job_queue[i].cancelled) {
        std::string job_id(job_queue[i].job_id);
        int worker_idx = job_queue[i].worker_index;

        if (worker_idx >= 0) {
          bool is_chunked = job_queue[i].is_chunked.load();
          bool send_ok    = false;
          bool has_data   = false;

          if (is_chunked) {
            PendingChunkedUpload chunked;
            {
              std::lock_guard<std::mutex> lock(pending_data_mutex);
              auto it = pending_chunked_data.find(job_id);
              if (it != pending_chunked_data.end()) {
                chunked  = std::move(it->second);
                has_data = true;
                pending_chunked_data.erase(it);
              }
            }
            if (has_data) {
              int to_fd    = worker_pipes[worker_idx].to_worker_fd;
              auto pipe_t0 = std::chrono::steady_clock::now();
              send_ok      = write_chunked_request_to_pipe(to_fd, chunked.header, chunked.chunks);
              if (send_ok && config.verbose) {
                auto pipe_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                 std::chrono::steady_clock::now() - pipe_t0)
                                 .count();
                std::cout << "[THROUGHPUT] phase=pipe_chunked_send chunks=" << chunked.chunks.size()
                          << " elapsed_ms=" << std::fixed << std::setprecision(1)
                          << (pipe_us / 1000.0) << "\n";
                std::cout << "[Server] Streamed " << chunked.chunks.size() << " chunks to worker "
                          << worker_idx << " for job " << job_id << "\n";
              }
            }
          } else {
            std::vector<uint8_t> job_data;
            {
              std::lock_guard<std::mutex> lock(pending_data_mutex);
              auto it = pending_job_data.find(job_id);
              if (it != pending_job_data.end()) {
                job_data = std::move(it->second);
                has_data = true;
                pending_job_data.erase(it);
              }
            }
            if (has_data) {
              auto pipe_t0 = std::chrono::steady_clock::now();
              send_ok      = send_job_data_pipe(worker_idx, job_data);
              if (send_ok && config.verbose) {
                auto pipe_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                 std::chrono::steady_clock::now() - pipe_t0)
                                 .count();
                double pipe_sec = pipe_us / 1e6;
                double pipe_mb  = static_cast<double>(job_data.size()) / (1024.0 * 1024.0);
                double pipe_mbs = (pipe_sec > 0.0) ? (pipe_mb / pipe_sec) : 0.0;
                std::cout << "[THROUGHPUT] phase=pipe_job_send bytes=" << job_data.size()
                          << " elapsed_ms=" << std::fixed << std::setprecision(1)
                          << (pipe_us / 1000.0) << " throughput_mb_s=" << std::setprecision(1)
                          << pipe_mbs << "\n";
                std::cout << "[Server] Sent " << job_data.size() << " bytes to worker "
                          << worker_idx << " for job " << job_id << "\n";
              }
            }
          }

          if (has_data) {
            if (send_ok) {
              job_queue[i].data_sent = true;
            } else {
              std::cerr << "[Server] Failed to send job data to worker " << worker_idx << "\n";
              job_queue[i].cancelled = true;
            }
            found = true;
          }
        }
      }
    }

    for (size_t i = 0; i < MAX_RESULTS; ++i) {
      if (result_queue[i].ready && !result_queue[i].retrieved) {
        std::string job_id(result_queue[i].job_id);
        ResultStatus result_status = result_queue[i].status;
        bool success               = (result_status == RESULT_SUCCESS);
        bool cancelled             = (result_status == RESULT_CANCELLED);
        int worker_idx             = result_queue[i].worker_index;
        if (config.verbose) {
          std::cout << "[Server] Detected ready result_slot=" << i << " for job " << job_id
                    << " status=" << result_status << " data_size=" << result_queue[i].data_size
                    << " worker_idx=" << worker_idx << "\n";
          std::cout.flush();
        }

        std::string error_message;

        cuopt::remote::ChunkedResultHeader hdr;
        std::map<int32_t, std::vector<uint8_t>> arrays;

        if (success && result_queue[i].data_size > 0) {
          if (config.verbose) {
            std::cout << "[Server] Reading streamed result from worker pipe for job " << job_id
                      << "\n";
            std::cout.flush();
          }
          int from_fd       = worker_pipes[worker_idx].from_worker_fd;
          auto pipe_recv_t0 = std::chrono::steady_clock::now();
          bool read_ok      = read_result_from_pipe(from_fd, hdr, arrays);
          if (!read_ok) {
            error_message = "Failed to read result data from pipe";
            success       = false;
          }
          if (success && config.verbose) {
            auto pipe_us = std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - pipe_recv_t0)
                             .count();
            int64_t total_bytes = 0;
            for (const auto& [fid, data] : arrays) {
              total_bytes += data.size();
            }
            double pipe_sec = pipe_us / 1e6;
            double pipe_mb  = static_cast<double>(total_bytes) / (1024.0 * 1024.0);
            double pipe_mbs = (pipe_sec > 0.0) ? (pipe_mb / pipe_sec) : 0.0;
            std::cout << "[THROUGHPUT] phase=pipe_result_recv bytes=" << total_bytes
                      << " elapsed_ms=" << std::fixed << std::setprecision(1) << (pipe_us / 1000.0)
                      << " throughput_mb_s=" << std::setprecision(1) << pipe_mbs << "\n";
            std::cout.flush();
          }
        } else if (!success) {
          error_message = result_queue[i].error_message;
        }

        {
          std::lock_guard<std::mutex> lock(tracker_mutex);
          auto it = job_tracker.find(job_id);
          if (it != job_tracker.end()) {
            if (success) {
              it->second.status   = JobStatus::COMPLETED;
              int64_t total_bytes = 0;
              for (const auto& [fid, data] : arrays) {
                total_bytes += data.size();
              }
              it->second.result_header     = std::move(hdr);
              it->second.result_arrays     = std::move(arrays);
              it->second.result_size_bytes = total_bytes;

              if (config.verbose) {
                std::cout << "[Server] Marked job COMPLETED in job_tracker: " << job_id
                          << " result_arrays=" << it->second.result_arrays.size()
                          << " result_size_bytes=" << it->second.result_size_bytes << "\n";
                std::cout.flush();
              }
            } else if (cancelled) {
              it->second.status        = JobStatus::CANCELLED;
              it->second.error_message = error_message;
              if (config.verbose) {
                std::cout << "[Server] Marked job CANCELLED in job_tracker: " << job_id
                          << " msg=" << error_message << "\n";
                std::cout.flush();
              }
            } else {
              it->second.status        = JobStatus::FAILED;
              it->second.error_message = error_message;
              if (config.verbose) {
                std::cout << "[Server] Marked job FAILED in job_tracker: " << job_id
                          << " msg=" << error_message << "\n";
                std::cout.flush();
              }
            }
          }
        }

        {
          std::lock_guard<std::mutex> lock(waiters_mutex);
          auto wit = waiting_threads.find(job_id);
          if (wit != waiting_threads.end()) {
            auto waiter = wit->second;
            {
              std::lock_guard<std::mutex> waiter_lock(waiter->mutex);
              waiter->error_message = error_message;
              waiter->success       = success;
              waiter->ready         = true;
            }
            waiter->cv.notify_all();
            waiting_threads.erase(wit);
          }
        }

        result_queue[i].retrieved    = true;
        result_queue[i].worker_index = -1;
        result_queue[i].ready        = false;
        found                        = true;
      }
    }

    if (!found) { usleep(10000); }

    result_cv.notify_all();
  }

  std::cout << "[Server] Result retrieval thread stopped\n";
  std::cout.flush();
}

void incumbent_retrieval_thread()
{
  std::cout << "[Server] Incumbent retrieval thread started\n";
  std::cout.flush();

  while (keep_running) {
    std::vector<pollfd> pfds;
    pfds.reserve(worker_pipes.size());
    for (const auto& wp : worker_pipes) {
      if (wp.incumbent_from_worker_fd >= 0) {
        pollfd pfd;
        pfd.fd      = wp.incumbent_from_worker_fd;
        pfd.events  = POLLIN;
        pfd.revents = 0;
        pfds.push_back(pfd);
      }
    }

    if (pfds.empty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      continue;
    }

    int poll_result = poll(pfds.data(), pfds.size(), 100);
    if (poll_result < 0) {
      if (errno == EINTR) continue;
      std::cerr << "[Server] poll() failed in incumbent thread: " << strerror(errno) << "\n";
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      continue;
    }
    if (poll_result == 0) { continue; }

    for (const auto& pfd : pfds) {
      if (!(pfd.revents & POLLIN)) { continue; }
      std::vector<uint8_t> data;
      if (!recv_incumbent_pipe(pfd.fd, data)) { continue; }
      if (data.empty()) { continue; }

      std::string job_id;
      double objective = 0.0;
      std::vector<double> assignment;
      if (!parse_incumbent_proto(data.data(), data.size(), job_id, objective, assignment)) {
        std::cerr << "[Server] Failed to parse incumbent payload\n";
        continue;
      }

      if (job_id.empty()) { continue; }

      IncumbentEntry entry;
      entry.objective  = objective;
      size_t num_vars  = assignment.size();
      entry.assignment = std::move(assignment);

      {
        std::lock_guard<std::mutex> lock(tracker_mutex);
        auto it = job_tracker.find(job_id);
        if (it != job_tracker.end()) {
          it->second.incumbents.push_back(std::move(entry));
          std::cout << "[Server] Stored incumbent job_id=" << job_id
                    << " idx=" << (it->second.incumbents.size() - 1) << " obj=" << objective
                    << " vars=" << num_vars << "\n";
          std::cout.flush();
        }
      }
    }
  }

  std::cout << "[Server] Incumbent retrieval thread stopped\n";
  std::cout.flush();
}

void session_reaper_thread()
{
  if (config.verbose) {
    std::cout << "[Server] Session reaper thread started (timeout=" << kSessionTimeoutSeconds
              << "s)\n";
    std::cout.flush();
  }

  const auto timeout = std::chrono::seconds(kSessionTimeoutSeconds);

  while (keep_running) {
    for (int i = 0; i < 60 && keep_running; ++i) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    if (!keep_running) break;

    auto now = std::chrono::steady_clock::now();

    {
      std::lock_guard<std::mutex> lock(chunked_uploads_mutex);
      for (auto it = chunked_uploads.begin(); it != chunked_uploads.end();) {
        if (now - it->second.last_activity > timeout) {
          if (config.verbose) {
            std::cout << "[Server] Reaping stale upload session: " << it->first << "\n";
            std::cout.flush();
          }
          it = chunked_uploads.erase(it);
        } else {
          ++it;
        }
      }
    }

    {
      std::lock_guard<std::mutex> lock(chunked_downloads_mutex);
      for (auto it = chunked_downloads.begin(); it != chunked_downloads.end();) {
        if (now - it->second.created > timeout) {
          if (config.verbose) {
            std::cout << "[Server] Reaping stale download session: " << it->first << "\n";
            std::cout.flush();
          }
          it = chunked_downloads.erase(it);
        } else {
          ++it;
        }
      }
    }
  }

  if (config.verbose) {
    std::cout << "[Server] Session reaper thread stopped\n";
    std::cout.flush();
  }
}

#endif  // CUOPT_ENABLE_GRPC
