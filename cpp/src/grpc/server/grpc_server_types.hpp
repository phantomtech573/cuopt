/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifdef CUOPT_ENABLE_GRPC

#include <grpcpp/grpcpp.h>
#include "cuopt_remote.pb.h"
#include "cuopt_remote_service.grpc.pb.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <google/protobuf/util/delimited_message_util.h>
#include <cuopt/linear_programming/cpu_optimization_problem_solution.hpp>
#include <cuopt/linear_programming/optimization_problem_interface.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <cuopt/linear_programming/utilities/internals.hpp>
#include "grpc_problem_mapper.hpp"
#include "grpc_settings_mapper.hpp"
#include "grpc_solution_mapper.hpp"

#include <cuda_runtime.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cstdio>

#include <uuid/uuid.h>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerReaderWriter;
using grpc::ServerWriter;
using grpc::Status;
using grpc::StatusCode;

using namespace cuopt::linear_programming;
// Note: NOT using "using namespace cuopt::remote" to avoid JobStatus enum conflict

// =============================================================================
// Utility functions
// =============================================================================

inline uint64_t compute_data_hash(const uint8_t* data, size_t size)
{
  constexpr uint64_t FNV_OFFSET_BASIS = 14695981039346656037ULL;
  constexpr uint64_t FNV_PRIME        = 1099511628211ULL;
  uint64_t hash                       = FNV_OFFSET_BASIS;
  for (size_t i = 0; i < size; ++i) {
    hash ^= static_cast<uint64_t>(data[i]);
    hash *= FNV_PRIME;
  }
  return hash;
}

inline std::string hash_to_hex(uint64_t hash)
{
  std::ostringstream oss;
  oss << std::hex << std::setfill('0') << std::setw(16) << hash;
  return oss.str();
}

// =============================================================================
// Shared Memory Structures (must match between main process and workers)
// =============================================================================

constexpr size_t MAX_JOBS    = 100;
constexpr size_t MAX_RESULTS = 100;

template <size_t N>
void copy_cstr(char (&dst)[N], const std::string& src)
{
  std::snprintf(dst, N, "%s", src.c_str());
}

template <size_t N>
void copy_cstr(char (&dst)[N], const char* src)
{
  std::snprintf(dst, N, "%s", src ? src : "");
}

struct JobQueueEntry {
  char job_id[64];
  uint32_t problem_type;          // 0 = LP, 1 = MIP
  uint64_t data_size;             // Size of problem data (uint64 for large problems)
  std::atomic<bool> ready;        // Job is ready to be processed
  std::atomic<bool> claimed;      // Worker has claimed this job
  std::atomic<pid_t> worker_pid;  // PID of worker that claimed this job (0 if none)
  std::atomic<bool> cancelled;    // Job has been cancelled (worker should skip)
  std::atomic<int> worker_index;  // Index of worker that claimed this job (-1 if none)
  std::atomic<bool> data_sent;    // Server has sent data to worker's pipe
};

struct ResultQueueEntry {
  char job_id[64];
  uint32_t status;     // 0 = success, 1 = error, 2 = cancelled
  uint64_t data_size;  // Size of result data (uint64 for large results)
  char error_message[1024];
  std::atomic<bool> ready;        // Result is ready
  std::atomic<bool> retrieved;    // Result has been retrieved
  std::atomic<int> worker_index;  // Index of worker that produced this result
};

struct SharedMemoryControl {
  std::atomic<bool> shutdown_requested;
  std::atomic<int> active_workers;
};

// =============================================================================
// Job status tracking (main process only)
// =============================================================================

enum class JobStatus { QUEUED, PROCESSING, COMPLETED, FAILED, NOT_FOUND, CANCELLED };

struct IncumbentEntry {
  double objective = 0.0;
  std::vector<double> assignment;
};

struct JobInfo {
  std::string job_id;
  JobStatus status;
  std::chrono::steady_clock::time_point submit_time;
  std::vector<IncumbentEntry> incumbents;
  bool is_mip;
  std::string error_message;
  bool is_blocking;
  cuopt::remote::ChunkedResultHeader result_header;
  std::map<int32_t, std::vector<uint8_t>> result_arrays;
  int64_t result_size_bytes = 0;
};

struct JobWaiter {
  std::mutex mutex;
  std::condition_variable cv;
  std::vector<uint8_t> result_data;
  std::string error_message;
  bool success;
  bool ready;
  std::atomic<int> waiters{0};
  JobWaiter() : success(false), ready(false) {}
};

// =============================================================================
// Server configuration
// =============================================================================

struct ServerConfig {
  int port            = 8765;
  int num_workers     = 1;
  bool verbose        = true;
  bool log_to_console = false;
  // gRPC max message size in MiB. 0 => unlimited (gRPC uses -1 internally).
  // --max-message-bytes overrides --max-message-mb when set (minimum 4096).
  int max_message_mb        = 256;
  int64_t max_message_b     = -1;  // -1 means use max_message_mb instead
  bool enable_transfer_hash = false;
  bool enable_tls           = false;
  bool require_client       = false;
  std::string tls_cert_path;
  std::string tls_key_path;
  std::string tls_root_path;
};

struct WorkerPipes {
  int to_worker_fd;
  int from_worker_fd;
  int worker_read_fd;
  int worker_write_fd;
  int incumbent_from_worker_fd;
  int worker_incumbent_write_fd;
};

// Chunked download session state (raw arrays from worker)
struct ChunkedDownloadState {
  bool is_mip = false;
  std::chrono::steady_clock::time_point created;
  cuopt::remote::ChunkedResultHeader result_header;
  std::map<int32_t, std::vector<uint8_t>> raw_arrays;  // ResultFieldId -> raw bytes
};

// Per-array allocation cap for chunked uploads (4 GiB).
static constexpr int64_t kMaxChunkedArrayBytes = 4LL * 1024 * 1024 * 1024;

// Maximum concurrent chunked upload + download sessions (global across all clients).
static constexpr size_t kMaxChunkedSessions = 16;

// Stale session timeout: sessions with no activity for this long are reaped.
static constexpr int kSessionTimeoutSeconds = 300;

struct ChunkedUploadState {
  bool is_mip = false;
  cuopt::remote::ChunkedProblemHeader header;
  struct FieldMeta {
    int64_t total_elements = 0;
    int64_t element_size   = 0;
    int64_t received_bytes = 0;
  };
  std::map<int32_t, FieldMeta> field_meta;
  std::vector<cuopt::remote::ArrayChunk> chunks;
  int64_t total_chunks = 0;
  int64_t total_bytes  = 0;
  std::chrono::steady_clock::time_point last_activity;
};

// =============================================================================
// Global state
// =============================================================================

inline std::atomic<bool> keep_running{true};
inline std::map<std::string, JobInfo> job_tracker;
inline std::mutex tracker_mutex;
inline std::condition_variable result_cv;

inline std::map<std::string, std::shared_ptr<JobWaiter>> waiting_threads;
inline std::mutex waiters_mutex;

inline JobQueueEntry* job_queue       = nullptr;
inline ResultQueueEntry* result_queue = nullptr;
inline SharedMemoryControl* shm_ctrl  = nullptr;

inline std::vector<pid_t> worker_pids;

inline ServerConfig config;

inline std::vector<WorkerPipes> worker_pipes;

inline std::mutex pending_data_mutex;
inline std::map<std::string, std::vector<uint8_t>> pending_job_data;

inline std::mutex chunked_uploads_mutex;
inline std::map<std::string, ChunkedUploadState> chunked_uploads;

inline std::mutex chunked_downloads_mutex;
inline std::map<std::string, ChunkedDownloadState> chunked_downloads;

inline const char* SHM_JOB_QUEUE    = "/cuopt_job_queue";
inline const char* SHM_RESULT_QUEUE = "/cuopt_result_queue";
inline const char* SHM_CONTROL      = "/cuopt_control";

inline const std::string LOG_DIR = "/tmp/cuopt_logs";

constexpr int64_t kMiB = 1024LL * 1024;
constexpr int64_t kGiB = 1024LL * 1024 * 1024;

// =============================================================================
// Inline utility functions
// =============================================================================

inline std::string get_log_file_path(const std::string& job_id)
{
  return LOG_DIR + "/job_" + job_id + ".log";
}

inline int64_t server_max_message_bytes()
{
  if (config.max_message_b >= 0) { return config.max_message_b; }
  return (config.max_message_mb <= 0) ? -1 : (static_cast<int64_t>(config.max_message_mb) * kMiB);
}

inline std::string read_file_to_string(const std::string& path)
{
  std::ifstream in(path, std::ios::in | std::ios::binary);
  if (!in.is_open()) { return ""; }
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// =============================================================================
// Signal handling
// =============================================================================

inline void signal_handler(int signal)
{
  if (signal == SIGINT || signal == SIGTERM) {
    keep_running = false;
    if (shm_ctrl) { shm_ctrl->shutdown_requested = true; }
  }
}

// =============================================================================
// Forward declarations
// =============================================================================

std::string generate_job_id();
void ensure_log_dir_exists();
void delete_log_file(const std::string& job_id);
void cleanup_shared_memory();
void spawn_workers();
void wait_for_workers();
void worker_monitor_thread();
void result_retrieval_thread();
void incumbent_retrieval_thread();
void session_reaper_thread();

bool write_to_pipe(int fd, const void* data, size_t size);
bool read_from_pipe(int fd, void* data, size_t size, int timeout_ms = 120000);
bool send_job_data_pipe(int worker_idx, const std::vector<uint8_t>& data);
bool recv_job_data_pipe(int fd, uint64_t expected_size, std::vector<uint8_t>& data);
bool send_result_pipe(int fd, const std::vector<uint8_t>& data);
bool send_incumbent_pipe(int fd, const std::vector<uint8_t>& data);
bool recv_incumbent_pipe(int fd, std::vector<uint8_t>& data);
bool recv_result_pipe(int worker_idx, uint64_t expected_size, std::vector<uint8_t>& data);

void worker_process(int worker_id);
pid_t spawn_single_worker(int worker_id);
void mark_worker_jobs_failed(pid_t dead_worker_pid);

std::pair<bool, std::string> submit_job_async(const std::vector<uint8_t>& request_data,
                                              bool is_mip);
JobStatus check_job_status(const std::string& job_id, std::string& message);
bool get_job_is_mip(const std::string& job_id);
int cancel_job(const std::string& job_id, JobStatus& job_status_out, std::string& message);

#endif  // CUOPT_ENABLE_GRPC
