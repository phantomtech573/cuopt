/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#ifdef CUOPT_ENABLE_GRPC

#include "grpc_server_types.hpp"

// =============================================================================
// Low-level pipe I/O
// =============================================================================

bool write_to_pipe(int fd, const void* data, size_t size)
{
  const uint8_t* ptr = static_cast<const uint8_t*>(data);
  size_t remaining   = size;
  while (remaining > 0) {
    ssize_t written = ::write(fd, ptr, remaining);
    if (written <= 0) {
      if (errno == EINTR) continue;
      return false;
    }
    ptr += written;
    remaining -= written;
  }
  return true;
}

bool read_from_pipe(int fd, void* data, size_t size, int timeout_ms)
{
  uint8_t* ptr     = static_cast<uint8_t*>(data);
  size_t remaining = size;
  while (remaining > 0) {
    struct pollfd pfd;
    pfd.fd     = fd;
    pfd.events = POLLIN;

    int poll_result = poll(&pfd, 1, timeout_ms);
    if (poll_result < 0) {
      if (errno == EINTR) continue;
      std::cerr << "[Server] poll() failed on pipe: " << strerror(errno) << "\n";
      return false;
    }
    if (poll_result == 0) {
      std::cerr << "[Server] Timeout waiting for pipe data (waited " << timeout_ms << "ms)\n";
      return false;
    }
    if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) {
      std::cerr << "[Server] Pipe error/hangup detected\n";
      return false;
    }

    ssize_t nread = ::read(fd, ptr, remaining);
    if (nread <= 0) {
      if (errno == EINTR) continue;
      if (nread == 0) { std::cerr << "[Server] Pipe EOF (writer closed)\n"; }
      return false;
    }
    ptr += nread;
    remaining -= nread;
  }
  return true;
}

bool send_job_data_pipe(int worker_idx, const std::vector<uint8_t>& data)
{
  if (worker_idx < 0 || worker_idx >= static_cast<int>(worker_pipes.size())) { return false; }
  int fd = worker_pipes[worker_idx].to_worker_fd;
  if (fd < 0) return false;

  uint64_t size = data.size();
  if (!write_to_pipe(fd, &size, sizeof(size))) return false;
  if (size > 0 && !write_to_pipe(fd, data.data(), data.size())) return false;
  return true;
}

bool recv_job_data_pipe(int fd, uint64_t expected_size, std::vector<uint8_t>& data)
{
  uint64_t size;
  if (!read_from_pipe(fd, &size, sizeof(size))) return false;
  if (size != expected_size) {
    std::cerr << "[Worker] Size mismatch: expected " << expected_size << ", got " << size << "\n";
    return false;
  }
  data.resize(size);
  if (size > 0 && !read_from_pipe(fd, data.data(), size)) return false;
  return true;
}

bool send_result_pipe(int fd, const std::vector<uint8_t>& data)
{
  uint64_t size = data.size();
  if (!write_to_pipe(fd, &size, sizeof(size))) return false;
  if (size > 0 && !write_to_pipe(fd, data.data(), data.size())) return false;
  return true;
}

bool send_incumbent_pipe(int fd, const std::vector<uint8_t>& data)
{
  uint64_t size = data.size();
  if (!write_to_pipe(fd, &size, sizeof(size))) return false;
  if (size > 0 && !write_to_pipe(fd, data.data(), data.size())) return false;
  return true;
}

bool recv_incumbent_pipe(int fd, std::vector<uint8_t>& data)
{
  uint64_t size;
  if (!read_from_pipe(fd, &size, sizeof(size))) return false;
  data.resize(size);
  if (size > 0 && !read_from_pipe(fd, data.data(), size)) return false;
  return true;
}

bool recv_result_pipe(int worker_idx, uint64_t expected_size, std::vector<uint8_t>& data)
{
  if (worker_idx < 0 || worker_idx >= static_cast<int>(worker_pipes.size())) { return false; }
  int fd = worker_pipes[worker_idx].from_worker_fd;
  if (fd < 0) return false;

  uint64_t size;
  if (!read_from_pipe(fd, &size, sizeof(size))) return false;
  if (size != expected_size) {
    std::cerr << "[Server] Result size mismatch: expected " << expected_size << ", got " << size
              << "\n";
    return false;
  }
  data.resize(size);
  if (size > 0 && !read_from_pipe(fd, data.data(), size)) return false;
  return true;
}

// =============================================================================
// Job management
// =============================================================================

std::pair<bool, std::string> submit_job_async(const std::vector<uint8_t>& request_data, bool is_mip)
{
  std::string job_id = generate_job_id();

  {
    std::lock_guard<std::mutex> lock(pending_data_mutex);
    pending_job_data[job_id] = request_data;
  }

  for (size_t i = 0; i < MAX_JOBS; ++i) {
    if (!job_queue[i].ready && !job_queue[i].claimed) {
      copy_cstr(job_queue[i].job_id, job_id);
      job_queue[i].problem_type = is_mip ? 1 : 0;
      job_queue[i].data_size    = request_data.size();
      job_queue[i].worker_pid   = 0;
      job_queue[i].worker_index = -1;
      job_queue[i].data_sent    = false;
      job_queue[i].claimed      = false;
      job_queue[i].cancelled    = false;
      job_queue[i].ready        = true;

      {
        std::lock_guard<std::mutex> lock(tracker_mutex);
        JobInfo info;
        info.job_id         = job_id;
        info.status         = JobStatus::QUEUED;
        info.submit_time    = std::chrono::steady_clock::now();
        info.is_mip         = is_mip;
        info.is_blocking    = false;
        job_tracker[job_id] = info;
      }

      if (config.verbose) { std::cout << "[Server] Job submitted (async): " << job_id << "\n"; }

      return {true, job_id};
    }
  }

  {
    std::lock_guard<std::mutex> lock(pending_data_mutex);
    pending_job_data.erase(job_id);
  }
  return {false, "Job queue full"};
}

JobStatus check_job_status(const std::string& job_id, std::string& message)
{
  std::lock_guard<std::mutex> lock(tracker_mutex);
  auto it = job_tracker.find(job_id);

  if (it == job_tracker.end()) {
    message = "Job ID not found";
    return JobStatus::NOT_FOUND;
  }

  if (it->second.status == JobStatus::QUEUED) {
    for (size_t i = 0; i < MAX_JOBS; ++i) {
      if (job_queue[i].ready && job_queue[i].claimed &&
          std::string(job_queue[i].job_id) == job_id) {
        it->second.status = JobStatus::PROCESSING;
        break;
      }
    }
  }

  switch (it->second.status) {
    case JobStatus::QUEUED: message = "Job is queued"; break;
    case JobStatus::PROCESSING: message = "Job is being processed"; break;
    case JobStatus::COMPLETED: message = "Job completed"; break;
    case JobStatus::FAILED: message = "Job failed: " + it->second.error_message; break;
    case JobStatus::CANCELLED: message = "Job was cancelled"; break;
    default: message = "Unknown status";
  }

  return it->second.status;
}

bool get_job_is_mip(const std::string& job_id)
{
  std::lock_guard<std::mutex> lock(tracker_mutex);
  auto it = job_tracker.find(job_id);
  if (it == job_tracker.end()) { return false; }
  return it->second.is_mip;
}

void ensure_log_dir_exists()
{
  struct stat st;
  if (stat(LOG_DIR.c_str(), &st) != 0) { mkdir(LOG_DIR.c_str(), 0755); }
}

void delete_log_file(const std::string& job_id)
{
  std::string log_file = get_log_file_path(job_id);
  unlink(log_file.c_str());
}

int cancel_job(const std::string& job_id, JobStatus& job_status_out, std::string& message)
{
  std::lock_guard<std::mutex> lock(tracker_mutex);
  auto it = job_tracker.find(job_id);

  if (it == job_tracker.end()) {
    message        = "Job ID not found";
    job_status_out = JobStatus::NOT_FOUND;
    return 1;
  }

  JobStatus current_status = it->second.status;

  if (current_status == JobStatus::COMPLETED) {
    message        = "Cannot cancel completed job";
    job_status_out = JobStatus::COMPLETED;
    return 2;
  }

  if (current_status == JobStatus::CANCELLED) {
    message        = "Job already cancelled";
    job_status_out = JobStatus::CANCELLED;
    return 3;
  }

  if (current_status == JobStatus::FAILED) {
    message        = "Cannot cancel failed job";
    job_status_out = JobStatus::FAILED;
    return 2;
  }

  for (size_t i = 0; i < MAX_JOBS; ++i) {
    if (job_queue[i].ready && strcmp(job_queue[i].job_id, job_id.c_str()) == 0) {
      pid_t worker_pid = job_queue[i].worker_pid;

      if (worker_pid > 0 && job_queue[i].claimed) {
        if (config.verbose) {
          std::cout << "[Server] Cancelling running job " << job_id << " (killing worker "
                    << worker_pid << ")\n";
        }
        job_queue[i].cancelled = true;
        kill(worker_pid, SIGKILL);
      } else {
        if (config.verbose) { std::cout << "[Server] Cancelling queued job " << job_id << "\n"; }
        job_queue[i].cancelled = true;
      }

      it->second.status        = JobStatus::CANCELLED;
      it->second.error_message = "Job cancelled by user";
      job_status_out           = JobStatus::CANCELLED;
      message                  = "Job cancelled successfully";

      delete_log_file(job_id);

      {
        std::lock_guard<std::mutex> wlock(waiters_mutex);
        auto wit = waiting_threads.find(job_id);
        if (wit != waiting_threads.end()) {
          auto waiter = wit->second;
          {
            std::lock_guard<std::mutex> waiter_lock(waiter->mutex);
            waiter->error_message = "Job cancelled by user";
            waiter->success       = false;
            waiter->ready         = true;
          }
          waiter->cv.notify_all();
          waiting_threads.erase(wit);
        }
      }

      return 0;
    }
  }

  if (it->second.status == JobStatus::COMPLETED) {
    message        = "Cannot cancel completed job";
    job_status_out = JobStatus::COMPLETED;
    return 2;
  }

  it->second.status        = JobStatus::CANCELLED;
  it->second.error_message = "Job cancelled by user";
  job_status_out           = JobStatus::CANCELLED;
  message                  = "Job cancelled";

  {
    std::lock_guard<std::mutex> wlock(waiters_mutex);
    auto wit = waiting_threads.find(job_id);
    if (wit != waiting_threads.end()) {
      auto waiter = wit->second;
      {
        std::lock_guard<std::mutex> waiter_lock(waiter->mutex);
        waiter->error_message = "Job cancelled by user";
        waiter->success       = false;
        waiter->ready         = true;
      }
      waiter->cv.notify_all();
      waiting_threads.erase(wit);
    }
  }

  return 0;
}

std::string generate_job_id()
{
  uuid_t uuid;
  uuid_generate_random(uuid);
  char buf[37];
  uuid_unparse_lower(uuid, buf);
  return std::string(buf);
}

#endif  // CUOPT_ENABLE_GRPC
