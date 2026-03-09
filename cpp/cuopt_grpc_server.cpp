/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file cuopt_grpc_server.cpp
 * @brief gRPC-based remote solve server with full worker process infrastructure
 *
 * This server uses gRPC for client communication but preserves the
 * existing worker process infrastructure:
 * - Worker processes with shared memory job queues
 * - Pipe-based IPC for problem/result data
 * - Result tracking and retrieval threads
 * - Log streaming
 *
 * Only the client-facing network layer is different (gRPC vs TCP).
 */

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
#include <cuopt/linear_programming/utilities/grpc_problem_mapper.hpp>
#include <cuopt/linear_programming/utilities/grpc_settings_mapper.hpp>
#include <cuopt/linear_programming/utilities/grpc_solution_mapper.hpp>
#include <cuopt/linear_programming/utilities/internals.hpp>

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

// ============================================================================
// Pipe IPC result serialization.
// Uses standard protobuf varint-delimited format (SerializeDelimitedToCodedStream).
// The 8-byte length prefix on the pipe is transport framing only.
// ============================================================================

// Max bytes per ArrayChunk data payload on the internal pipe (64 MiB).
// Keeps each varint-delimited protobuf message well under the 2 GiB hard limit.
static constexpr size_t kPipeChunkBytes = 64ULL * 1024 * 1024;

// Serialize a result header + raw arrays into a pipe blob.
// Format: varint-delimited ChunkedResultHeader, then N varint-delimited ArrayChunk messages.
// Large arrays are split into multiple ArrayChunk messages of at most kPipeChunkBytes each.
inline std::vector<uint8_t> serialize_result_pipe_blob(
  const cuopt::remote::ChunkedResultHeader& header,
  const std::map<int32_t, std::vector<uint8_t>>& arrays)
{
  auto emit_chunks = [&](auto callback) {
    for (const auto& [fid, data] : arrays) {
      size_t remaining = data.size();
      size_t offset    = 0;
      do {
        size_t slice = std::min(remaining, kPipeChunkBytes);
        cuopt::remote::ArrayChunk ac;
        ac.set_field_id(static_cast<cuopt::remote::ArrayFieldId>(fid));
        ac.set_element_offset(static_cast<int64_t>(offset));
        ac.set_total_elements(static_cast<int64_t>(data.size()));
        ac.set_data(data.data() + offset, slice);
        callback(ac);
        offset += slice;
        remaining -= slice;
      } while (remaining > 0);
    }
  };

  size_t total = 0;
  {
    uint32_t hdr_size = static_cast<uint32_t>(header.ByteSizeLong());
    total += google::protobuf::io::CodedOutputStream::VarintSize32(hdr_size) + hdr_size;
  }
  emit_chunks([&](const cuopt::remote::ArrayChunk& ac) {
    uint32_t msg_size = static_cast<uint32_t>(ac.ByteSizeLong());
    total += google::protobuf::io::CodedOutputStream::VarintSize32(msg_size) + msg_size;
  });

  std::vector<uint8_t> blob(total);
  google::protobuf::io::ArrayOutputStream raw(blob.data(), static_cast<int>(blob.size()));
  google::protobuf::io::CodedOutputStream coded(&raw);

  google::protobuf::util::SerializeDelimitedToCodedStream(header, &coded);
  emit_chunks([&](const cuopt::remote::ArrayChunk& ac) {
    google::protobuf::util::SerializeDelimitedToCodedStream(ac, &coded);
  });
  return blob;
}

// Deserialize a result pipe blob back into header + reassembled arrays.
inline bool deserialize_result_pipe_blob(const uint8_t* data,
                                         size_t size,
                                         cuopt::remote::ChunkedResultHeader& header_out,
                                         std::map<int32_t, std::vector<uint8_t>>& arrays_out)
{
  google::protobuf::io::ArrayInputStream raw(data, static_cast<int>(size));
  google::protobuf::io::CodedInputStream coded(&raw);

  bool clean_eof = false;
  if (!google::protobuf::util::ParseDelimitedFromCodedStream(&header_out, &coded, &clean_eof)) {
    return false;
  }

  while (!clean_eof) {
    cuopt::remote::ArrayChunk ac;
    if (!google::protobuf::util::ParseDelimitedFromCodedStream(&ac, &coded, &clean_eof)) break;
    int32_t fid = static_cast<int32_t>(ac.field_id());
    auto& dest  = arrays_out[fid];
    if (dest.empty() && ac.total_elements() > 0) {
      dest.resize(static_cast<size_t>(ac.total_elements()), 0);
    }
    int64_t offset         = ac.element_offset();
    const auto& chunk_data = ac.data();
    if (offset + static_cast<int64_t>(chunk_data.size()) <= static_cast<int64_t>(dest.size())) {
      std::memcpy(dest.data() + offset, chunk_data.data(), chunk_data.size());
    }
  }
  return true;
}

// Serialize a chunked request (ChunkedProblemHeader + ArrayChunk messages) into a pipe blob.
// Format: varint-delimited ChunkedProblemHeader, then N varint-delimited ArrayChunk messages.
// ArrayChunk messages are forwarded as-is from the client upload.
inline std::vector<uint8_t> serialize_chunked_request_pipe_blob(
  const cuopt::remote::ChunkedProblemHeader& header,
  const std::vector<cuopt::remote::ArrayChunk>& chunks)
{
  size_t total = 0;
  {
    uint32_t hdr_size = static_cast<uint32_t>(header.ByteSizeLong());
    total += google::protobuf::io::CodedOutputStream::VarintSize32(hdr_size) + hdr_size;
  }
  for (const auto& ac : chunks) {
    uint32_t msg_size = static_cast<uint32_t>(ac.ByteSizeLong());
    total += google::protobuf::io::CodedOutputStream::VarintSize32(msg_size) + msg_size;
  }

  std::vector<uint8_t> blob(total);
  google::protobuf::io::ArrayOutputStream raw(blob.data(), static_cast<int>(blob.size()));
  google::protobuf::io::CodedOutputStream coded(&raw);

  google::protobuf::util::SerializeDelimitedToCodedStream(header, &coded);
  for (const auto& ac : chunks) {
    google::protobuf::util::SerializeDelimitedToCodedStream(ac, &coded);
  }
  return blob;
}

// Deserialize a chunked request pipe blob: header + ArrayChunk messages reassembled into arrays.
// The worker calls this to reconstruct complete arrays from potentially many small chunks.
inline bool deserialize_chunked_request_pipe_blob(
  const uint8_t* data,
  size_t size,
  cuopt::remote::ChunkedProblemHeader& header_out,
  std::map<int32_t, std::vector<uint8_t>>& arrays_out)
{
  google::protobuf::io::ArrayInputStream raw(data, static_cast<int>(size));
  google::protobuf::io::CodedInputStream coded(&raw);

  bool clean_eof = false;
  if (!google::protobuf::util::ParseDelimitedFromCodedStream(&header_out, &coded, &clean_eof)) {
    return false;
  }

  // Reassemble ArrayChunk messages into complete per-field arrays.
  // Each chunk specifies field_id, element_offset, total_elements, and data bytes.
  // We pre-allocate on first chunk and memcpy each slice into position.
  while (!clean_eof) {
    cuopt::remote::ArrayChunk ac;
    if (!google::protobuf::util::ParseDelimitedFromCodedStream(&ac, &coded, &clean_eof)) break;
    int32_t fid = static_cast<int32_t>(ac.field_id());
    auto& dest  = arrays_out[fid];
    if (dest.empty() && ac.total_elements() > 0) {
      int64_t elem_size = 8;
      switch (ac.field_id()) {
        case cuopt::remote::FIELD_A_INDICES:
        case cuopt::remote::FIELD_A_OFFSETS:
        case cuopt::remote::FIELD_Q_INDICES:
        case cuopt::remote::FIELD_Q_OFFSETS: elem_size = 4; break;
        case cuopt::remote::FIELD_ROW_TYPES:
        case cuopt::remote::FIELD_VARIABLE_TYPES:
        case cuopt::remote::FIELD_VARIABLE_NAMES:
        case cuopt::remote::FIELD_ROW_NAMES: elem_size = 1; break;
        default: elem_size = 8; break;
      }
      dest.resize(static_cast<size_t>(ac.total_elements() * elem_size), 0);
    }
    int64_t elem_size      = (dest.size() > 0 && ac.total_elements() > 0)
                               ? static_cast<int64_t>(dest.size()) / ac.total_elements()
                               : 1;
    int64_t byte_offset    = ac.element_offset() * elem_size;
    const auto& chunk_data = ac.data();
    if (byte_offset >= 0 && byte_offset + static_cast<int64_t>(chunk_data.size()) <=
                              static_cast<int64_t>(dest.size())) {
      std::memcpy(dest.data() + byte_offset, chunk_data.data(), chunk_data.size());
    }
  }
  return true;
}

// ============================================================================
// Server Stream Watchdog - Timeout Detection for Streaming Operations
// ============================================================================

// Chunked download session state (raw arrays from worker)
struct ChunkedDownloadState {
  bool is_mip = false;
  std::chrono::steady_clock::time_point created;
  cuopt::remote::ChunkedResultHeader result_header;
  std::map<int32_t, std::vector<uint8_t>> raw_arrays;  // ResultFieldId -> raw bytes
};

// ============================================================================
// Shared Memory Structures (must match between main process and workers)
// ============================================================================

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

// Job queue entry - small fixed size, data sent to worker via pipe
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

// Result queue entry - small fixed size, data received from worker via pipe
struct ResultQueueEntry {
  char job_id[64];
  uint32_t status;     // 0 = success, 1 = error, 2 = cancelled
  uint64_t data_size;  // Size of result data (uint64 for large results)
  char error_message[1024];
  std::atomic<bool> ready;        // Result is ready
  std::atomic<bool> retrieved;    // Result has been retrieved
  std::atomic<int> worker_index;  // Index of worker that produced this result
};

// Shared memory control block
struct SharedMemoryControl {
  std::atomic<bool> shutdown_requested;
  std::atomic<int> active_workers;
};

// ============================================================================
// Job status tracking (main process only)
// ============================================================================

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

// ============================================================================
// Global state
// ============================================================================

std::atomic<bool> keep_running{true};
std::map<std::string, JobInfo> job_tracker;
std::mutex tracker_mutex;
std::condition_variable result_cv;

std::map<std::string, std::shared_ptr<JobWaiter>> waiting_threads;
std::mutex waiters_mutex;

JobQueueEntry* job_queue       = nullptr;
ResultQueueEntry* result_queue = nullptr;
SharedMemoryControl* shm_ctrl  = nullptr;

std::vector<pid_t> worker_pids;

struct ServerConfig {
  int port            = 8765;
  int num_workers     = 1;
  bool verbose        = true;
  bool log_to_console = false;
  // gRPC max message size in MiB. 0 => unlimited (gRPC uses -1 internally).
  // --max-message-bytes overrides --max-message-mb when set (minimum 4096).
  int max_message_mb    = 256;
  int64_t max_message_b = -1;  // -1 means use max_message_mb instead
  // Enable data integrity hash logging for transfers (for testing/debugging)
  bool enable_transfer_hash = false;
  bool enable_tls           = false;
  bool require_client       = false;
  std::string tls_cert_path;
  std::string tls_key_path;
  std::string tls_root_path;
};

ServerConfig config;

struct WorkerPipes {
  int to_worker_fd;
  int from_worker_fd;
  int worker_read_fd;
  int worker_write_fd;
  int incumbent_from_worker_fd;
  int worker_incumbent_write_fd;
};

std::vector<WorkerPipes> worker_pipes;

std::mutex pending_data_mutex;
std::map<std::string, std::vector<uint8_t>> pending_job_data;

// Per-array allocation cap for chunked uploads (4 GiB).
// Covers ~500M doubles or ~1B int32s per array, well beyond realistic problem sizes.
// With ~15 array fields per upload, worst-case session footprint is ~60 GiB.
static constexpr int64_t kMaxChunkedArrayBytes = 4LL * 1024 * 1024 * 1024;

// Maximum concurrent chunked upload + download sessions (global across all clients).
static constexpr size_t kMaxChunkedSessions = 16;

// Stale session timeout: sessions with no activity for this long are reaped.
static constexpr int kSessionTimeoutSeconds = 300;

// In-progress chunked array uploads.
// Each StartChunkedUpload creates an entry; SendArrayChunk validates and stores raw chunks;
// FinishChunkedUpload forwards header + chunks to the worker, which does reassembly.
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
std::mutex chunked_uploads_mutex;
std::map<std::string, ChunkedUploadState> chunked_uploads;

std::mutex chunked_downloads_mutex;
std::map<std::string, ChunkedDownloadState> chunked_downloads;

const char* SHM_JOB_QUEUE    = "/cuopt_job_queue";
const char* SHM_RESULT_QUEUE = "/cuopt_result_queue";
const char* SHM_CONTROL      = "/cuopt_control";

const std::string LOG_DIR = "/tmp/cuopt_logs";
inline std::string get_log_file_path(const std::string& job_id)
{
  return LOG_DIR + "/job_" + job_id + ".log";
}

// ============================================================================
// Signal handling
// ============================================================================

void signal_handler(int signal)
{
  if (signal == SIGINT || signal == SIGTERM) {
    std::cout << "\n[gRPC Server] Received shutdown signal\n";
    keep_running = false;
    if (shm_ctrl) { shm_ctrl->shutdown_requested = true; }
    result_cv.notify_all();
  }
}

// ============================================================================
// Forward declarations
// ============================================================================

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

// Pipe and shared memory functions
static bool write_to_pipe(int fd, const void* data, size_t size);
static bool read_from_pipe(int fd, void* data, size_t size, int timeout_ms = 120000);
static bool send_job_data_pipe(int worker_idx, const std::vector<uint8_t>& data);
static bool recv_job_data_pipe(int fd, uint64_t expected_size, std::vector<uint8_t>& data);
static bool send_result_pipe(int fd, const std::vector<uint8_t>& data);
static bool send_incumbent_pipe(int fd, const std::vector<uint8_t>& data);
static bool recv_incumbent_pipe(int fd, std::vector<uint8_t>& data);
static bool recv_result_pipe(int worker_idx, uint64_t expected_size, std::vector<uint8_t>& data);
constexpr int64_t kMiB = 1024LL * 1024;
constexpr int64_t kGiB = 1024LL * 1024 * 1024;

// Compute the server's effective max message size in bytes (-1 = unlimited)
inline int64_t server_max_message_bytes()
{
  if (config.max_message_b >= 0) { return config.max_message_b; }
  return (config.max_message_mb <= 0) ? -1 : (static_cast<int64_t>(config.max_message_mb) * kMiB);
}

// Serialize a SubmitJobRequest directly to a pipe blob using standard protobuf.
// The worker parses this directly as a standard protobuf message.
inline std::vector<uint8_t> serialize_submit_request_to_pipe(
  const cuopt::remote::SubmitJobRequest& request)
{
  size_t byte_size = request.ByteSizeLong();
  std::vector<uint8_t> blob(byte_size);
  request.SerializeToArray(blob.data(), static_cast<int>(byte_size));
  return blob;
}

class IncumbentPipeCallback : public cuopt::internals::get_solution_callback_t {
 public:
  IncumbentPipeCallback(std::string job_id, int fd, size_t num_vars, bool is_float)
    : job_id_(std::move(job_id)), fd_(fd)
  {
    // Use inherited members from base class
    n_variables = num_vars;
    isFloat     = is_float;
  }

  void get_solution(void* data,
                    void* objective_value,
                    void* /*solution_bound*/,
                    void* /*user_data*/) override
  {
    if (fd_ < 0 || n_variables == 0) { return; }

    // Note: The MIP solver passes HOST pointers (data is already copied to host)
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

    cuopt::remote::Incumbent msg;
    msg.set_job_id(job_id_);
    msg.set_objective(objective);
    for (double v : assignment) {
      msg.add_assignment(v);
    }

    std::vector<uint8_t> buffer(msg.ByteSizeLong());
    if (!msg.SerializeToArray(buffer.data(), buffer.size())) { return; }
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

// ============================================================================
// Worker Infrastructure (shared with the remote solve server implementation)
// ============================================================================
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

void worker_process(int worker_id)
{
  std::cout << "[Worker " << worker_id << "] Started (PID: " << getpid() << ")\n";

  // Increment active worker count
  shm_ctrl->active_workers++;

  // NOTE: We create raft::handle_t AFTER stdout redirect (per-job) so that
  // CUDA logging uses the redirected output streams.

  while (!shm_ctrl->shutdown_requested) {
    // Find a job to process
    int job_slot = -1;
    for (size_t i = 0; i < MAX_JOBS; ++i) {
      if (job_queue[i].ready && !job_queue[i].claimed) {
        // Try to claim this job atomically
        bool expected = false;
        if (job_queue[i].claimed.compare_exchange_strong(expected, true)) {
          job_queue[i].worker_pid   = getpid();   // Record our PID
          job_queue[i].worker_index = worker_id;  // Record worker index for pipe mode
          job_slot                  = i;
          break;
        }
      }
    }

    if (job_slot < 0) {
      // No job available, sleep briefly
      usleep(10000);  // 10ms
      continue;
    }

    // Process the job
    JobQueueEntry& job = job_queue[job_slot];
    std::string job_id(job.job_id);
    bool is_mip = (job.problem_type == 1);

    // Check if job was cancelled before we start processing
    if (job.cancelled) {
      std::cout << "[Worker " << worker_id << "] Job cancelled before processing: " << job_id
                << "\n";
      std::cout.flush();

      // Store cancelled result in result queue
      store_simple_result(job_id, worker_id, 2, "Job was cancelled");

      // Clear job slot (don't exit/restart worker)
      job.worker_pid   = 0;
      job.worker_index = -1;
      job.data_sent    = false;
      job.ready        = false;
      job.claimed      = false;
      job.cancelled    = false;
      continue;  // Go back to waiting for next job
    }

    std::cout << "[Worker " << worker_id << "] Processing job: " << job_id
              << " (type: " << (is_mip ? "MIP" : "LP") << ")\n";
    std::cout.flush();

    std::string log_file = get_log_file_path(job_id);

    // Create RAFT handle before calling solver
    std::cout << "[Worker] Creating raft::handle_t...\n" << std::flush;

    raft::handle_t handle;

    std::cout << "[Worker] Handle created, starting solve...\n" << std::flush;

    // Read problem data from pipe (blocks until server writes data)
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
      // Store error result
      store_simple_result(job_id, worker_id, 1, "Failed to read job data");
      // Clear job slot
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

      // The main process forwards data in one of two formats:
      //   1) Unary: SubmitJobRequest serialized directly (standard protobuf)
      //   2) Chunked: varint-delimited ChunkedProblemHeader + ArrayChunk messages
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
        // Chunked path: varint-delimited ChunkedProblemHeader + ArrayChunk messages (reassembled)
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

      // Free raw request data to reduce memory pressure
      request_data.clear();
      request_data.shrink_to_fit();

      if (is_mip) {
        mip_settings.log_file       = log_file;
        mip_settings.log_to_console = config.log_to_console;

        // Set up incumbent callback if enabled
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

    // Store result: write to pipe
    {
      // PIPE MODE: Set result_queue metadata FIRST, THEN write to pipe.
      // This avoids deadlock: the main thread's result_retrieval_thread
      // needs to see ready=true before it will read from the pipe,
      // but if we write to pipe first with a large result, we'll block
      // waiting for the reader that will never come.

      // Find a free result slot and populate metadata
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
          // Set ready=true BEFORE writing to pipe so reader thread starts reading
          // This prevents deadlock with large results that exceed pipe buffer size
          result.ready = true;
          if (config.verbose) {
            std::cout << "[Worker " << worker_id << "] Enqueued result metadata for job " << job_id
                      << " in result_slot=" << result_slot << " status=" << result.status
                      << " data_size=" << result.data_size << "\n";
            std::cout.flush();
          }
          break;
        }
      }

      // Now write result data to pipe (reader thread should be ready to receive)
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
          // Mark as failed in result queue
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

    // Clear job slot
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

// Create pipes for a worker (incumbent always, data/result in pipe mode)
bool create_worker_pipes(int worker_id)
{
  // Ensure worker_pipes has enough slots
  while (static_cast<int>(worker_pipes.size()) <= worker_id) {
    worker_pipes.push_back({-1, -1, -1, -1, -1, -1});
  }

  WorkerPipes& wp = worker_pipes[worker_id];

  // Create pipe for server -> worker data
  int input_pipe[2];
  if (pipe(input_pipe) < 0) {
    std::cerr << "[Server] Failed to create input pipe for worker " << worker_id << "\n";
    return false;
  }
  wp.worker_read_fd = input_pipe[0];  // Worker reads from this
  wp.to_worker_fd   = input_pipe[1];  // Server writes to this

  // Create pipe for worker -> server results
  int output_pipe[2];
  if (pipe(output_pipe) < 0) {
    std::cerr << "[Server] Failed to create output pipe for worker " << worker_id << "\n";
    close(input_pipe[0]);
    close(input_pipe[1]);
    return false;
  }
  wp.from_worker_fd  = output_pipe[0];  // Server reads from this
  wp.worker_write_fd = output_pipe[1];  // Worker writes to this

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
  wp.incumbent_from_worker_fd  = incumbent_pipe[0];  // Server reads from this
  wp.worker_incumbent_write_fd = incumbent_pipe[1];  // Worker writes to this

  return true;
}

// Close server-side pipe ends for a worker (called when restarting)
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

// Close worker-side pipe ends in parent after fork
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
    // Child process: close all other workers' pipe fds
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
    // Close server ends of our pipes
    close(worker_pipes[worker_id].to_worker_fd);
    close(worker_pipes[worker_id].from_worker_fd);
    if (worker_pipes[worker_id].incumbent_from_worker_fd >= 0) {
      close(worker_pipes[worker_id].incumbent_from_worker_fd);
      worker_pipes[worker_id].incumbent_from_worker_fd = -1;
    }
    worker_process(worker_id);
    _exit(0);  // Should not reach here
  }

  // Parent: close worker ends of new pipes
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

// Spawn a single replacement worker and return its PID
pid_t spawn_single_worker(int worker_id) { return spawn_worker(worker_id, true); }

// Mark jobs being processed by a dead worker as failed (or cancelled if it was cancelled)
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

      // Cleanup job data: remove from pending data if not yet sent
      {
        std::lock_guard<std::mutex> lock(pending_data_mutex);
        pending_job_data.erase(job_id);
      }

      // Store result in result queue (cancelled or failed)
      for (size_t j = 0; j < MAX_RESULTS; ++j) {
        if (!result_queue[j].ready) {
          copy_cstr(result_queue[j].job_id, job_id);
          result_queue[j].status       = was_cancelled ? 2 : 1;  // 2=cancelled, 1=error
          result_queue[j].data_size    = 0;
          result_queue[j].worker_index = -1;
          copy_cstr(result_queue[j].error_message,
                    was_cancelled ? "Job was cancelled" : "Worker process died unexpectedly");
          result_queue[j].retrieved = false;
          result_queue[j].ready     = true;
          break;
        }
      }

      // Clear the job slot
      job_queue[i].worker_pid   = 0;
      job_queue[i].worker_index = -1;
      job_queue[i].data_sent    = false;
      job_queue[i].ready        = false;
      job_queue[i].claimed      = false;
      job_queue[i].cancelled    = false;

      // Update job tracker
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

// Worker monitor thread - detects dead workers and restarts them
void worker_monitor_thread()
{
  std::cout << "[Server] Worker monitor thread started\n";
  std::cout.flush();

  while (keep_running) {
    // Check all worker PIDs for dead workers
    for (size_t i = 0; i < worker_pids.size(); ++i) {
      pid_t pid = worker_pids[i];
      if (pid <= 0) continue;

      int status;
      pid_t result = waitpid(pid, &status, WNOHANG);

      if (result == pid) {
        // Worker has exited
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
          // Clean exit during shutdown - don't restart
          if (shm_ctrl && shm_ctrl->shutdown_requested) {
            worker_pids[i] = 0;
            continue;
          }
          std::cerr << "[Server] Worker " << pid << " exited unexpectedly\n";
          std::cerr.flush();
        }

        // Mark any jobs this worker was processing as failed
        mark_worker_jobs_failed(pid);

        // Spawn replacement worker (unless shutting down)
        if (keep_running && shm_ctrl && !shm_ctrl->shutdown_requested) {
          pid_t new_pid = spawn_single_worker(static_cast<int>(i));
          if (new_pid > 0) {
            worker_pids[i] = new_pid;
            std::cout << "[Server] Restarted worker " << i << " with PID " << new_pid << "\n";
            std::cout.flush();
          } else {
            worker_pids[i] = 0;  // Failed to restart
          }
        } else {
          worker_pids[i] = 0;
        }
      }
    }

    // Check every 100ms
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

    // Check for jobs that need data sent to workers
    for (size_t i = 0; i < MAX_JOBS; ++i) {
      if (job_queue[i].ready && job_queue[i].claimed && !job_queue[i].data_sent &&
          !job_queue[i].cancelled) {
        std::string job_id(job_queue[i].job_id);
        int worker_idx = job_queue[i].worker_index;

        if (worker_idx >= 0) {
          // Get pending job data (in-memory)
          std::vector<uint8_t> job_data;
          {
            std::lock_guard<std::mutex> lock(pending_data_mutex);
            auto it = pending_job_data.find(job_id);
            if (it != pending_job_data.end()) {
              job_data = std::move(it->second);
              pending_job_data.erase(it);
            }
          }

          if (!job_data.empty()) {
            // Send data to worker's pipe
            auto pipe_t0 = std::chrono::steady_clock::now();
            if (send_job_data_pipe(worker_idx, job_data)) {
              job_queue[i].data_sent = true;
              if (config.verbose) {
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
            } else {
              std::cerr << "[Server] Failed to send job data to worker " << worker_idx << "\n";
              // Mark job as failed
              job_queue[i].cancelled = true;
            }
            found = true;
          }
        }
      }
    }

    // Check for completed results
    for (size_t i = 0; i < MAX_RESULTS; ++i) {
      if (result_queue[i].ready && !result_queue[i].retrieved) {
        std::string job_id(result_queue[i].job_id);
        uint32_t result_status = result_queue[i].status;
        bool success           = (result_status == 0);
        bool cancelled         = (result_status == 2);
        int worker_idx         = result_queue[i].worker_index;
        if (config.verbose) {
          std::cout << "[Server] Detected ready result_slot=" << i << " for job " << job_id
                    << " status=" << result_status << " data_size=" << result_queue[i].data_size
                    << " worker_idx=" << worker_idx << "\n";
          std::cout.flush();
        }

        std::vector<uint8_t> result_data;
        std::string error_message;

        if (success && result_queue[i].data_size > 0) {
          // Read result from worker's output pipe
          if (config.verbose) {
            std::cout << "[Server] Reading " << result_queue[i].data_size
                      << " bytes from worker pipe for job " << job_id << "\n";
            std::cout.flush();
          }
          auto pipe_recv_t0 = std::chrono::steady_clock::now();
          if (!recv_result_pipe(worker_idx, result_queue[i].data_size, result_data)) {
            error_message = "Failed to read result data from pipe";
            success       = false;
          }
          if (success && config.verbose) {
            auto pipe_us = std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - pipe_recv_t0)
                             .count();
            double pipe_sec = pipe_us / 1e6;
            double pipe_mb  = static_cast<double>(result_data.size()) / (1024.0 * 1024.0);
            double pipe_mbs = (pipe_sec > 0.0) ? (pipe_mb / pipe_sec) : 0.0;
            std::cout << "[THROUGHPUT] phase=pipe_result_recv bytes=" << result_data.size()
                      << " elapsed_ms=" << std::fixed << std::setprecision(1) << (pipe_us / 1000.0)
                      << " throughput_mb_s=" << std::setprecision(1) << pipe_mbs << "\n";
            std::cout.flush();
          }
        } else if (!success) {
          error_message = result_queue[i].error_message;
        }

        // Update job tracker first (before moving data to waiter)
        {
          std::lock_guard<std::mutex> lock(tracker_mutex);
          auto it = job_tracker.find(job_id);
          if (it != job_tracker.end()) {
            if (success) {
              it->second.status = JobStatus::COMPLETED;
              cuopt::remote::ChunkedResultHeader hdr;
              std::map<int32_t, std::vector<uint8_t>> arrays;
              if (!deserialize_result_pipe_blob(
                    result_data.data(), result_data.size(), hdr, arrays)) {
                it->second.status        = JobStatus::FAILED;
                it->second.error_message = "Failed to deserialize result pipe blob";
              } else {
                it->second.result_header     = std::move(hdr);
                it->second.result_arrays     = std::move(arrays);
                it->second.result_size_bytes = static_cast<int64_t>(result_data.size());
              }

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

        // Check if there's a blocking waiter
        {
          std::lock_guard<std::mutex> lock(waiters_mutex);
          auto wit = waiting_threads.find(job_id);
          if (wit != waiting_threads.end()) {
            // Wake up all waiting threads sharing this waiter
            auto waiter = wit->second;
            {
              std::lock_guard<std::mutex> waiter_lock(waiter->mutex);
              waiter->result_data   = std::move(result_data);
              waiter->error_message = error_message;
              waiter->success       = success;
              waiter->ready         = true;
            }
            waiter->cv.notify_all();
            waiting_threads.erase(wit);
          } else if (config.verbose) {
            std::cout << "[Server] WARNING: result for unknown job_id (not in job_tracker): "
                      << job_id << "\n";
            std::cout.flush();
          }
        }

        result_queue[i].retrieved    = true;
        result_queue[i].worker_index = -1;
        result_queue[i].ready        = false;  // Free slot
        found                        = true;
      }
    }

    if (!found) {
      usleep(10000);  // 10ms
    }

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

      cuopt::remote::Incumbent incumbent_msg;
      if (!incumbent_msg.ParseFromArray(data.data(), data.size())) {
        std::cerr << "[Server] Failed to parse incumbent payload\n";
        continue;
      }

      const std::string job_id = incumbent_msg.job_id();
      if (job_id.empty()) { continue; }

      IncumbentEntry entry;
      entry.objective = incumbent_msg.objective();
      entry.assignment.reserve(incumbent_msg.assignment_size());
      for (int i = 0; i < incumbent_msg.assignment_size(); ++i) {
        entry.assignment.push_back(incumbent_msg.assignment(i));
      }

      {
        std::lock_guard<std::mutex> lock(tracker_mutex);
        auto it = job_tracker.find(job_id);
        if (it != job_tracker.end()) {
          it->second.incumbents.push_back(std::move(entry));
          std::cout << "[Server] Stored incumbent job_id=" << job_id
                    << " idx=" << (it->second.incumbents.size() - 1)
                    << " obj=" << incumbent_msg.objective()
                    << " vars=" << incumbent_msg.assignment_size() << "\n";
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

static std::string read_file_to_string(const std::string& path)
{
  std::ifstream in(path, std::ios::in | std::ios::binary);
  if (!in.is_open()) { return ""; }
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

static bool write_to_pipe(int fd, const void* data, size_t size)
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

// Read all data from a pipe (handles partial reads) with timeout
// timeout_ms: milliseconds to wait for data (default 120000 = 2 minutes)
static bool read_from_pipe(int fd, void* data, size_t size, int timeout_ms)
{
  uint8_t* ptr     = static_cast<uint8_t*>(data);
  size_t remaining = size;
  while (remaining > 0) {
    // Use poll() to wait for data with timeout
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

// Send job data to worker via pipe (length-prefixed)
static bool send_job_data_pipe(int worker_idx, const std::vector<uint8_t>& data)
{
  if (worker_idx < 0 || worker_idx >= static_cast<int>(worker_pipes.size())) { return false; }
  int fd = worker_pipes[worker_idx].to_worker_fd;
  if (fd < 0) return false;

  // Send size first
  uint64_t size = data.size();
  if (!write_to_pipe(fd, &size, sizeof(size))) return false;
  // Send data
  if (size > 0 && !write_to_pipe(fd, data.data(), data.size())) return false;
  return true;
}

// Stream job data from a file to the worker via pipe (length-prefixed).
// This avoids holding the entire job payload in server memory.
// Receive job data from pipe (length-prefixed) - called by worker
static bool recv_job_data_pipe(int fd, uint64_t expected_size, std::vector<uint8_t>& data)
{
  // Read size
  uint64_t size;
  if (!read_from_pipe(fd, &size, sizeof(size))) return false;
  if (size != expected_size) {
    std::cerr << "[Worker] Size mismatch: expected " << expected_size << ", got " << size << "\n";
    return false;
  }
  // Read data
  data.resize(size);
  if (size > 0 && !read_from_pipe(fd, data.data(), size)) return false;
  return true;
}

// Send result data to server via pipe (length-prefixed) - called by worker
static bool send_result_pipe(int fd, const std::vector<uint8_t>& data)
{
  // Send size first
  uint64_t size = data.size();
  if (!write_to_pipe(fd, &size, sizeof(size))) return false;
  // Send data
  if (size > 0 && !write_to_pipe(fd, data.data(), data.size())) return false;
  return true;
}

// Send incumbent data to server via pipe (length-prefixed) - called by worker
static bool send_incumbent_pipe(int fd, const std::vector<uint8_t>& data)
{
  uint64_t size = data.size();
  if (!write_to_pipe(fd, &size, sizeof(size))) return false;
  if (size > 0 && !write_to_pipe(fd, data.data(), data.size())) return false;
  return true;
}

// Receive incumbent data from worker via pipe (length-prefixed)
static bool recv_incumbent_pipe(int fd, std::vector<uint8_t>& data)
{
  uint64_t size;
  if (!read_from_pipe(fd, &size, sizeof(size))) return false;
  data.resize(size);
  if (size > 0 && !read_from_pipe(fd, data.data(), size)) return false;
  return true;
}

// Receive result data from worker via pipe (length-prefixed)
static bool recv_result_pipe(int worker_idx, uint64_t expected_size, std::vector<uint8_t>& data)
{
  if (worker_idx < 0 || worker_idx >= static_cast<int>(worker_pipes.size())) { return false; }
  int fd = worker_pipes[worker_idx].from_worker_fd;
  if (fd < 0) return false;

  // Read size
  uint64_t size;
  if (!read_from_pipe(fd, &size, sizeof(size))) return false;
  if (size != expected_size) {
    std::cerr << "[Server] Result size mismatch: expected " << expected_size << ", got " << size
              << "\n";
    return false;
  }
  // Read data
  data.resize(size);
  if (size > 0 && !read_from_pipe(fd, data.data(), size)) return false;
  return true;
}

// Submit a job asynchronously (returns job_id)
std::pair<bool, std::string> submit_job_async(const std::vector<uint8_t>& request_data, bool is_mip)
{
  std::string job_id = generate_job_id();

  // Store data in pending map (will be sent to worker pipe when it claims the job)
  {
    std::lock_guard<std::mutex> lock(pending_data_mutex);
    pending_job_data[job_id] = request_data;
  }

  // Find free job slot
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
      job_queue[i].ready        = true;  // Mark as ready last

      // Track job
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

  // No free slot - cleanup
  {
    std::lock_guard<std::mutex> lock(pending_data_mutex);
    pending_job_data.erase(job_id);
  }
  return {false, "Job queue full"};
}

// Check job status
JobStatus check_job_status(const std::string& job_id, std::string& message)
{
  std::lock_guard<std::mutex> lock(tracker_mutex);
  auto it = job_tracker.find(job_id);

  if (it == job_tracker.end()) {
    message = "Job ID not found";
    return JobStatus::NOT_FOUND;
  }

  // If status is QUEUED, check if the job has been claimed by a worker
  // (which means it's now PROCESSING)
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

// Check if a job is MIP (vs LP)
bool get_job_is_mip(const std::string& job_id)
{
  std::lock_guard<std::mutex> lock(tracker_mutex);
  auto it = job_tracker.find(job_id);
  if (it == job_tracker.end()) {
    return false;  // Default to LP if not found
  }
  return it->second.is_mip;
}

// Get job result
void ensure_log_dir_exists()
{
  struct stat st;
  if (stat(LOG_DIR.c_str(), &st) != 0) { mkdir(LOG_DIR.c_str(), 0755); }
}

// Delete log file for a job
void delete_log_file(const std::string& job_id)
{
  std::string log_file = get_log_file_path(job_id);
  unlink(log_file.c_str());  // Ignore errors if file doesn't exist
}

// Cancel job - returns: 0=success, 1=job_not_found, 2=already_completed, 3=already_cancelled
// Also returns the job's status after cancel attempt via job_status_out
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

  // Can't cancel completed jobs
  if (current_status == JobStatus::COMPLETED) {
    message        = "Cannot cancel completed job";
    job_status_out = JobStatus::COMPLETED;
    return 2;
  }

  // Already cancelled
  if (current_status == JobStatus::CANCELLED) {
    message        = "Job already cancelled";
    job_status_out = JobStatus::CANCELLED;
    return 3;
  }

  // Can't cancel failed jobs
  if (current_status == JobStatus::FAILED) {
    message        = "Cannot cancel failed job";
    job_status_out = JobStatus::FAILED;
    return 2;
  }

  // Find the job in the shared memory queue
  for (size_t i = 0; i < MAX_JOBS; ++i) {
    if (job_queue[i].ready && strcmp(job_queue[i].job_id, job_id.c_str()) == 0) {
      // Check if job is being processed by a worker
      pid_t worker_pid = job_queue[i].worker_pid;

      if (worker_pid > 0 && job_queue[i].claimed) {
        // Job is being processed - kill the worker
        if (config.verbose) {
          std::cout << "[Server] Cancelling running job " << job_id << " (killing worker "
                    << worker_pid << ")\n";
        }
        // Mark cancelled BEFORE killing so the worker monitor path (mark_worker_jobs_failed)
        // reliably observes was_cancelled=true and reports CANCELLED rather than FAILED.
        job_queue[i].cancelled = true;
        kill(worker_pid, SIGKILL);
      } else {
        // Job is queued but not yet claimed - mark as cancelled
        if (config.verbose) { std::cout << "[Server] Cancelling queued job " << job_id << "\n"; }
        job_queue[i].cancelled = true;
      }

      // Update job tracker
      it->second.status        = JobStatus::CANCELLED;
      it->second.error_message = "Job cancelled by user";
      job_status_out           = JobStatus::CANCELLED;
      message                  = "Job cancelled successfully";

      // Delete the log file for this job
      delete_log_file(job_id);

      // Wake up any threads waiting for this job
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

  // Job not found in queue (might have already finished processing)
  // Re-check status since we hold the lock
  if (it->second.status == JobStatus::COMPLETED) {
    message        = "Cannot cancel completed job";
    job_status_out = JobStatus::COMPLETED;
    return 2;
  }

  // Job must be in flight or in an edge case - mark as cancelled anyway
  it->second.status        = JobStatus::CANCELLED;
  it->second.error_message = "Job cancelled by user";
  job_status_out           = JobStatus::CANCELLED;
  message                  = "Job cancelled";

  // Wake up any threads waiting for this job
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

// ============================================================================
// Helper Functions
// ============================================================================

std::string generate_job_id()
{
  uuid_t uuid;
  uuid_generate_random(uuid);
  char buf[37];
  uuid_unparse_lower(uuid, buf);
  return std::string(buf);
}

// ============================================================================
// gRPC Service Implementation
// ============================================================================

class CuOptRemoteServiceImpl final : public cuopt::remote::CuOptRemoteService::Service {
 public:
  // SubmitJob - Submit LP/MIP job for async processing
  Status SubmitJob(ServerContext* context,
                   const cuopt::remote::SubmitJobRequest* request,
                   cuopt::remote::SubmitJobResponse* response) override
  {
    std::string job_id = generate_job_id();

    bool is_lp = request->has_lp_request();
    if (!is_lp && !request->has_mip_request()) {
      return Status(StatusCode::INVALID_ARGUMENT, "No problem data provided");
    }

    if (config.verbose && is_lp) {
      const auto& lp_req = request->lp_request();
      std::cerr << "[gRPC] SubmitJob LP fields: bytes=" << lp_req.ByteSizeLong()
                << " objective_scaling_factor=" << lp_req.problem().objective_scaling_factor()
                << " objective_offset=" << lp_req.problem().objective_offset()
                << " iteration_limit=" << lp_req.settings().iteration_limit()
                << " method=" << lp_req.settings().method() << std::endl;
    }

    auto job_data = serialize_submit_request_to_pipe(*request);
    if (config.verbose) {
      std::cout << "[gRPC] SubmitJob: UNARY " << (is_lp ? "LP" : "MIP")
                << ", pipe payload=" << job_data.size() << " bytes\n";
      std::cout.flush();
    }

    // Find and reserve a free slot in the shared job queue.
    //
    // NOTE: Unlike the legacy socket server (single-threaded accept loop),
    // gRPC can dispatch multiple SubmitJob RPCs concurrently. We must reserve
    // a slot so two SubmitJob calls don't pick the same entry.
    //
    // We use `claimed=true` as a temporary reservation while `ready=false`.
    int job_idx = -1;
    for (size_t i = 0; i < MAX_JOBS; ++i) {
      if (job_queue[i].ready.load()) { continue; }
      bool expected_claimed = false;
      if (job_queue[i].claimed.compare_exchange_strong(expected_claimed, true)) {
        job_idx = static_cast<int>(i);
        break;
      }
    }

    if (job_idx < 0) { return Status(StatusCode::RESOURCE_EXHAUSTED, "Job queue full"); }

    // Initialize job queue entry
    copy_cstr(job_queue[job_idx].job_id, job_id);
    job_queue[job_idx].problem_type = is_lp ? 0 : 1;
    job_queue[job_idx].data_size    = job_data.size();
    // `claimed` currently true as a reservation; keep it until the entry is fully initialized.
    job_queue[job_idx].cancelled.store(false);
    job_queue[job_idx].worker_index.store(-1);
    job_queue[job_idx].data_sent.store(false);

    {
      std::lock_guard<std::mutex> lock(pending_data_mutex);
      pending_job_data[job_id] = std::move(job_data);
    }

    // Add to job tracker
    {
      std::lock_guard<std::mutex> lock(tracker_mutex);
      job_tracker[job_id] =
        JobInfo{job_id, JobStatus::QUEUED, std::chrono::steady_clock::now(), {}, !is_lp, "", false};
    }

    // Publish job to workers: release reservation and set ready last.
    job_queue[job_idx].claimed.store(false);
    job_queue[job_idx].ready.store(true);

    response->set_job_id(job_id);
    response->set_message("Job submitted successfully");

    if (config.verbose) {
      std::cout << "[gRPC] Job submitted: " << job_id << " (type=" << (is_lp ? "LP" : "MIP") << ")"
                << std::endl;
    }

    return Status::OK;
  }

  // =========================================================================
  // Chunked Array Upload (3 unary RPCs replacing the old streaming UploadAndSubmit)
  // =========================================================================

  Status StartChunkedUpload(ServerContext* context,
                            const cuopt::remote::StartChunkedUploadRequest* request,
                            cuopt::remote::StartChunkedUploadResponse* response) override
  {
    (void)context;

    std::string upload_id = generate_job_id();
    const auto& header    = request->problem_header();
    bool is_mip           = (header.header().problem_type() == cuopt::remote::MIP);

    if (config.verbose) {
      std::cout << "[gRPC] StartChunkedUpload upload_id=" << upload_id << " is_mip=" << is_mip
                << "\n";
      std::cout.flush();
    }

    // Store upload state
    {
      std::lock_guard<std::mutex> lock(chunked_uploads_mutex);
      if (chunked_uploads.size() >= kMaxChunkedSessions) {
        return Status(StatusCode::RESOURCE_EXHAUSTED,
                      "Too many concurrent chunked upload sessions (limit " +
                        std::to_string(kMaxChunkedSessions) + ")");
      }
      auto& state         = chunked_uploads[upload_id];
      state.is_mip        = is_mip;
      state.header        = header;
      state.total_chunks  = 0;
      state.last_activity = std::chrono::steady_clock::now();
    }

    response->set_upload_id(upload_id);
    response->set_max_message_bytes(server_max_message_bytes());

    return Status::OK;
  }

  Status SendArrayChunk(ServerContext* context,
                        const cuopt::remote::SendArrayChunkRequest* request,
                        cuopt::remote::SendArrayChunkResponse* response) override
  {
    (void)context;

    const std::string& upload_id = request->upload_id();
    const auto& ac               = request->chunk();

    std::lock_guard<std::mutex> lock(chunked_uploads_mutex);
    auto it = chunked_uploads.find(upload_id);
    if (it == chunked_uploads.end()) {
      return Status(StatusCode::NOT_FOUND, "Unknown upload_id: " + upload_id);
    }

    auto& state         = it->second;
    state.last_activity = std::chrono::steady_clock::now();

    int32_t field_id    = static_cast<int32_t>(ac.field_id());
    int64_t elem_offset = ac.element_offset();
    int64_t total_elems = ac.total_elements();
    const auto& raw     = ac.data();

    if (!cuopt::remote::ArrayFieldId_IsValid(field_id)) {
      return Status(StatusCode::INVALID_ARGUMENT,
                    "Unknown array field_id: " + std::to_string(field_id));
    }
    if (elem_offset < 0) {
      return Status(StatusCode::INVALID_ARGUMENT, "element_offset must be non-negative");
    }
    if (total_elems < 0) {
      return Status(StatusCode::INVALID_ARGUMENT, "total_elements must be non-negative");
    }

    auto& meta = state.field_meta[field_id];
    if (meta.total_elements == 0 && total_elems > 0) {
      int64_t elem_size = 8;  // default: double
      switch (ac.field_id()) {
        case cuopt::remote::FIELD_A_INDICES:
        case cuopt::remote::FIELD_A_OFFSETS:
        case cuopt::remote::FIELD_Q_INDICES:
        case cuopt::remote::FIELD_Q_OFFSETS: elem_size = 4; break;
        case cuopt::remote::FIELD_ROW_TYPES:
        case cuopt::remote::FIELD_VARIABLE_TYPES:
        case cuopt::remote::FIELD_VARIABLE_NAMES:
        case cuopt::remote::FIELD_ROW_NAMES: elem_size = 1; break;
        default: elem_size = 8; break;
      }
      if (total_elems > kMaxChunkedArrayBytes / elem_size) {
        return Status(StatusCode::RESOURCE_EXHAUSTED,
                      "Array too large (" + std::to_string(total_elems) + " x " +
                        std::to_string(elem_size) + " bytes exceeds " +
                        std::to_string(kMaxChunkedArrayBytes) + " byte limit)");
      }
      meta.total_elements = total_elems;
      meta.element_size   = elem_size;
    }

    int64_t elem_size = meta.element_size > 0 ? meta.element_size : 1;

    if (elem_size > 1 && (raw.size() % static_cast<size_t>(elem_size)) != 0) {
      return Status(StatusCode::INVALID_ARGUMENT,
                    "Chunk data size (" + std::to_string(raw.size()) +
                      ") not aligned to element size (" + std::to_string(elem_size) + ")");
    }

    int64_t array_bytes = meta.total_elements * elem_size;
    if (elem_offset > meta.total_elements) {
      return Status(StatusCode::INVALID_ARGUMENT, "ArrayChunk offset exceeds array size");
    }
    int64_t byte_offset = elem_offset * elem_size;
    if (byte_offset + static_cast<int64_t>(raw.size()) > array_bytes) {
      return Status(StatusCode::INVALID_ARGUMENT, "ArrayChunk out of bounds");
    }

    meta.received_bytes += static_cast<int64_t>(raw.size());
    state.total_bytes += static_cast<int64_t>(raw.size());
    state.chunks.push_back(ac);
    ++state.total_chunks;

    response->set_upload_id(upload_id);
    response->set_chunks_received(state.total_chunks);
    return Status::OK;
  }

  Status FinishChunkedUpload(ServerContext* context,
                             const cuopt::remote::FinishChunkedUploadRequest* request,
                             cuopt::remote::SubmitJobResponse* response) override
  {
    (void)context;

    const std::string& upload_id = request->upload_id();

    // Extract the upload state (move out of map)
    ChunkedUploadState state;
    {
      std::lock_guard<std::mutex> lock(chunked_uploads_mutex);
      auto it = chunked_uploads.find(upload_id);
      if (it == chunked_uploads.end()) {
        return Status(StatusCode::NOT_FOUND, "Unknown upload_id: " + upload_id);
      }
      state = std::move(it->second);
      chunked_uploads.erase(it);
    }

    if (config.verbose) {
      std::cout << "[gRPC] FinishChunkedUpload upload_id=" << upload_id
                << " chunks=" << state.total_chunks << " fields=" << state.field_meta.size()
                << "\n";
      std::cout.flush();
    }

    auto worker_data = serialize_chunked_request_pipe_blob(state.header, state.chunks);
    state.chunks.clear();
    state.field_meta.clear();

    if (config.verbose) {
      std::cout << "[gRPC] FinishChunkedUpload: CHUNKED path, " << state.total_chunks
                << " chunks forwarded, pipe payload=" << worker_data.size()
                << " bytes, upload_id=" << upload_id << "\n";
      std::cout.flush();
    }

    // Enqueue the job
    std::string job_id = generate_job_id();
    int job_idx        = -1;
    for (size_t i = 0; i < MAX_JOBS; ++i) {
      if (job_queue[i].ready.load()) { continue; }
      bool expected_claimed = false;
      if (job_queue[i].claimed.compare_exchange_strong(expected_claimed, true)) {
        job_idx = static_cast<int>(i);
        break;
      }
    }
    if (job_idx < 0) { return Status(StatusCode::RESOURCE_EXHAUSTED, "Job queue full"); }

    copy_cstr(job_queue[job_idx].job_id, job_id);
    job_queue[job_idx].problem_type = state.is_mip ? 1 : 0;
    job_queue[job_idx].data_size    = static_cast<uint64_t>(worker_data.size());
    job_queue[job_idx].cancelled.store(false);
    job_queue[job_idx].worker_index.store(-1);
    job_queue[job_idx].data_sent.store(false);
    {
      std::lock_guard<std::mutex> lock(pending_data_mutex);
      pending_job_data[job_id] = std::move(worker_data);
    }

    {
      std::lock_guard<std::mutex> lock(tracker_mutex);
      job_tracker[job_id] = JobInfo{
        job_id, JobStatus::QUEUED, std::chrono::steady_clock::now(), {}, state.is_mip, "", false};
    }

    job_queue[job_idx].claimed.store(false);
    job_queue[job_idx].ready.store(true);

    response->set_job_id(job_id);
    response->set_message("Job submitted via chunked arrays");

    if (config.verbose) {
      std::cout << "[gRPC] FinishChunkedUpload enqueued job: " << job_id
                << " (type=" << (state.is_mip ? "MIP" : "LP") << ")\n";
      std::cout.flush();
    }

    return Status::OK;
  }

  // CheckStatus - Check job status
  Status CheckStatus(ServerContext* context,
                     const cuopt::remote::StatusRequest* request,
                     cuopt::remote::StatusResponse* response) override
  {
    (void)context;
    std::string job_id = request->job_id();

    // Use shared-memory job queue state to expose PROCESSING when a worker claims the job.
    // This enables reliable mid-solve cancellation tests.
    std::string message;
    JobStatus status = check_job_status(job_id, message);

    switch (status) {
      case JobStatus::QUEUED: response->set_job_status(cuopt::remote::QUEUED); break;
      case JobStatus::PROCESSING: response->set_job_status(cuopt::remote::PROCESSING); break;
      case JobStatus::COMPLETED: response->set_job_status(cuopt::remote::COMPLETED); break;
      case JobStatus::FAILED: response->set_job_status(cuopt::remote::FAILED); break;
      case JobStatus::CANCELLED: response->set_job_status(cuopt::remote::CANCELLED); break;
      default: response->set_job_status(cuopt::remote::NOT_FOUND); break;
    }
    response->set_message(message);

    response->set_max_message_bytes(server_max_message_bytes());

    int64_t result_size_bytes = 0;
    if (status == JobStatus::COMPLETED) {
      std::lock_guard<std::mutex> lock(tracker_mutex);
      auto it = job_tracker.find(job_id);
      if (it != job_tracker.end()) { result_size_bytes = it->second.result_size_bytes; }
    }
    response->set_result_size_bytes(result_size_bytes);

    return Status::OK;
  }

  // GetResult - Retrieve completed job result
  // For large results, clients should use StartChunkedDownload/GetResultChunk instead.
  Status GetResult(ServerContext* context,
                   const cuopt::remote::GetResultRequest* request,
                   cuopt::remote::ResultResponse* response) override
  {
    (void)context;
    std::string job_id = request->job_id();

    std::lock_guard<std::mutex> lock(tracker_mutex);
    auto it = job_tracker.find(job_id);

    if (it == job_tracker.end()) { return Status(StatusCode::NOT_FOUND, "Job not found"); }

    if (it->second.status != JobStatus::COMPLETED && it->second.status != JobStatus::FAILED) {
      return Status(StatusCode::UNAVAILABLE, "Result not ready");
    }

    if (it->second.status == JobStatus::FAILED) {
      response->set_status(cuopt::remote::ERROR_SOLVE_FAILED);
      response->set_error_message(it->second.error_message);
      return Status::OK;
    }

    int64_t total_result_bytes = it->second.result_size_bytes;
    const int64_t max_bytes    = server_max_message_bytes();
    if (max_bytes > 0 && total_result_bytes > max_bytes) {
      std::string msg = "Result size (~" + std::to_string(total_result_bytes) +
                        " bytes) exceeds max message size (" + std::to_string(max_bytes) +
                        " bytes). Use StartChunkedDownload/GetResultChunk RPCs instead.";
      if (config.verbose) {
        std::cout << "[gRPC] GetResult rejected for job " << job_id << ": " << msg << "\n";
        std::cout.flush();
      }
      return Status(StatusCode::RESOURCE_EXHAUSTED, msg);
    }

    // Build full protobuf from stored header + arrays
    if (it->second.is_mip) {
      cuopt::remote::MIPSolution mip_solution;
      build_mip_solution_proto<int, double>(
        it->second.result_header, it->second.result_arrays, &mip_solution);
      response->mutable_mip_solution()->Swap(&mip_solution);
    } else {
      cuopt::remote::LPSolution lp_solution;
      build_lp_solution_proto<int, double>(
        it->second.result_header, it->second.result_arrays, &lp_solution);
      response->mutable_lp_solution()->Swap(&lp_solution);
    }

    response->set_status(cuopt::remote::SUCCESS);
    if (config.verbose) {
      std::cout << "[gRPC] GetResult: UNARY response for job " << job_id << " ("
                << total_result_bytes << " bytes, " << it->second.result_arrays.size()
                << " arrays)\n";
      std::cout.flush();
    }

    return Status::OK;
  }

  // =========================================================================
  // Chunked Result Download RPCs
  // =========================================================================

  Status StartChunkedDownload(ServerContext* context,
                              const cuopt::remote::StartChunkedDownloadRequest* request,
                              cuopt::remote::StartChunkedDownloadResponse* response) override
  {
    std::string job_id = request->job_id();

    bool is_mip = false;
    ChunkedDownloadState state;
    {
      std::lock_guard<std::mutex> lock(tracker_mutex);
      auto it = job_tracker.find(job_id);
      if (it == job_tracker.end()) {
        return Status(StatusCode::NOT_FOUND, "Job not found: " + job_id);
      }
      if (it->second.status != JobStatus::COMPLETED) {
        return Status(StatusCode::FAILED_PRECONDITION, "Result not ready for job: " + job_id);
      }
      is_mip              = it->second.is_mip;
      state.is_mip        = is_mip;
      state.created       = std::chrono::steady_clock::now();
      state.result_header = it->second.result_header;
      state.raw_arrays    = it->second.result_arrays;
    }

    response->mutable_header()->CopyFrom(state.result_header);

    std::string download_id = generate_job_id();
    response->set_download_id(download_id);
    response->set_max_message_bytes(server_max_message_bytes());

    {
      std::lock_guard<std::mutex> lock(chunked_downloads_mutex);
      if (chunked_downloads.size() >= kMaxChunkedSessions) {
        return Status(StatusCode::RESOURCE_EXHAUSTED,
                      "Too many concurrent chunked download sessions (limit " +
                        std::to_string(kMaxChunkedSessions) + ")");
      }
      chunked_downloads[download_id] = std::move(state);
    }

    if (config.verbose) {
      std::cout << "[gRPC] StartChunkedDownload: CHUNKED response for job " << job_id
                << ", download_id=" << download_id
                << ", arrays=" << response->header().arrays_size() << ", is_mip=" << is_mip << "\n";
      std::cout.flush();
    }

    return Status::OK;
  }

  Status GetResultChunk(ServerContext* context,
                        const cuopt::remote::GetResultChunkRequest* request,
                        cuopt::remote::GetResultChunkResponse* response) override
  {
    std::string download_id = request->download_id();
    auto field_id           = request->field_id();
    int64_t elem_offset     = request->element_offset();
    int64_t max_elements    = request->max_elements();

    std::lock_guard<std::mutex> lock(chunked_downloads_mutex);
    auto it = chunked_downloads.find(download_id);
    if (it == chunked_downloads.end()) {
      return Status(StatusCode::NOT_FOUND, "Unknown download_id: " + download_id);
    }

    const auto& state = it->second;

    // Look up raw array by field_id
    const uint8_t* raw_bytes = nullptr;
    int64_t total_bytes      = 0;
    auto ait                 = state.raw_arrays.find(static_cast<int32_t>(field_id));
    if (ait != state.raw_arrays.end() && !ait->second.empty()) {
      raw_bytes   = ait->second.data();
      total_bytes = static_cast<int64_t>(ait->second.size());
    }

    if (raw_bytes == nullptr || total_bytes == 0) {
      return Status(StatusCode::INVALID_ARGUMENT,
                    "Unknown or empty result field: " + std::to_string(field_id));
    }

    // All result arrays are doubles; convert element offset/count to byte offset/count
    const int64_t elem_size  = sizeof(double);
    const int64_t array_size = total_bytes / elem_size;

    if (elem_offset < 0 || elem_offset >= array_size) {
      return Status(StatusCode::OUT_OF_RANGE,
                    "element_offset " + std::to_string(elem_offset) + " out of range [0, " +
                      std::to_string(array_size) + ")");
    }

    int64_t elems_available = array_size - elem_offset;
    int64_t elems_to_send =
      (max_elements > 0) ? std::min(max_elements, elems_available) : elems_available;

    response->set_download_id(download_id);
    response->set_field_id(field_id);
    response->set_element_offset(elem_offset);
    response->set_elements_in_chunk(elems_to_send);
    response->set_data(reinterpret_cast<const char*>(raw_bytes + elem_offset * elem_size),
                       static_cast<size_t>(elems_to_send) * elem_size);

    return Status::OK;
  }

  Status FinishChunkedDownload(ServerContext* context,
                               const cuopt::remote::FinishChunkedDownloadRequest* request,
                               cuopt::remote::FinishChunkedDownloadResponse* response) override
  {
    std::string download_id = request->download_id();
    response->set_download_id(download_id);

    std::lock_guard<std::mutex> lock(chunked_downloads_mutex);
    auto it = chunked_downloads.find(download_id);
    if (it == chunked_downloads.end()) {
      return Status(StatusCode::NOT_FOUND, "Unknown download_id: " + download_id);
    }

    if (config.verbose) {
      auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - it->second.created)
                          .count();
      std::cout << "[gRPC] FinishChunkedDownload: download_id=" << download_id
                << " elapsed_ms=" << elapsed_ms << "\n";
      std::cout.flush();
    }

    chunked_downloads.erase(it);
    return Status::OK;
  }

  // Other RPCs - stubs for now
  Status DeleteResult(ServerContext* context,
                      const cuopt::remote::DeleteRequest* request,
                      cuopt::remote::DeleteResponse* response) override
  {
    std::string job_id = request->job_id();

    size_t erased = 0;
    {
      std::lock_guard<std::mutex> lock(tracker_mutex);
      erased = job_tracker.erase(job_id);
    }

    if (erased == 0) {
      // Job not found in tracker
      response->set_status(cuopt::remote::ERROR_NOT_FOUND);
      response->set_message("Job not found: " + job_id);
      if (config.verbose) {
        std::cout << "[gRPC] DeleteResult job not found: " << job_id << std::endl;
      }
      return Status::OK;
    }

    delete_log_file(job_id);

    response->set_status(cuopt::remote::SUCCESS);
    response->set_message("Result deleted");

    if (config.verbose) { std::cout << "[gRPC] Result deleted for job: " << job_id << std::endl; }

    return Status::OK;
  }

  Status CancelJob(ServerContext* context,
                   const cuopt::remote::CancelRequest* request,
                   cuopt::remote::CancelResponse* response) override
  {
    (void)context;
    std::string job_id = request->job_id();

    JobStatus internal_status = JobStatus::NOT_FOUND;
    std::string message;
    int rc = cancel_job(job_id, internal_status, message);

    // Map internal status -> protobuf JobStatus
    cuopt::remote::JobStatus pb_status = cuopt::remote::NOT_FOUND;
    switch (internal_status) {
      case JobStatus::QUEUED: pb_status = cuopt::remote::QUEUED; break;
      case JobStatus::PROCESSING: pb_status = cuopt::remote::PROCESSING; break;
      case JobStatus::COMPLETED: pb_status = cuopt::remote::COMPLETED; break;
      case JobStatus::FAILED: pb_status = cuopt::remote::FAILED; break;
      case JobStatus::CANCELLED: pb_status = cuopt::remote::CANCELLED; break;
      case JobStatus::NOT_FOUND: pb_status = cuopt::remote::NOT_FOUND; break;
    }

    response->set_job_status(pb_status);
    response->set_message(message);

    // Map rc -> ResponseStatus for backward-compatible response payload.
    // (We still return gRPC Status::OK; clients should check fields.)
    if (rc == 0 || rc == 3) {
      response->set_status(cuopt::remote::SUCCESS);
    } else if (rc == 1) {
      response->set_status(cuopt::remote::ERROR_NOT_FOUND);
    } else {
      response->set_status(cuopt::remote::ERROR_INVALID_REQUEST);
    }

    if (config.verbose) {
      std::cout << "[gRPC] CancelJob job_id=" << job_id << " rc=" << rc
                << " status=" << static_cast<int>(pb_status) << " msg=" << message << "\n";
      std::cout.flush();
    }

    return Status::OK;
  }

  Status WaitForCompletion(ServerContext* context,
                           const cuopt::remote::WaitRequest* request,
                           cuopt::remote::WaitResponse* response) override
  {
    const std::string job_id = request->job_id();

    // Check if the job already reached a terminal state before blocking.
    {
      std::lock_guard<std::mutex> lock(tracker_mutex);
      auto it = job_tracker.find(job_id);
      if (it == job_tracker.end()) {
        response->set_job_status(cuopt::remote::NOT_FOUND);
        response->set_message("Job not found");
        response->set_result_size_bytes(0);
        return Status::OK;
      }
      if (it->second.status == JobStatus::COMPLETED) {
        response->set_job_status(cuopt::remote::COMPLETED);
        response->set_message("");
        response->set_result_size_bytes(it->second.result_size_bytes);
        return Status::OK;
      }
      if (it->second.status == JobStatus::FAILED) {
        response->set_job_status(cuopt::remote::FAILED);
        response->set_message(it->second.error_message);
        response->set_result_size_bytes(0);
        return Status::OK;
      }
      if (it->second.status == JobStatus::CANCELLED) {
        response->set_job_status(cuopt::remote::CANCELLED);
        response->set_message("Job was cancelled");
        response->set_result_size_bytes(0);
        return Status::OK;
      }
    }

    // Acquire (or share) a waiter and poll with cancellation checks.
    std::shared_ptr<JobWaiter> waiter;
    {
      std::lock_guard<std::mutex> lock(waiters_mutex);
      auto it = waiting_threads.find(job_id);
      if (it != waiting_threads.end()) {
        waiter = it->second;
      } else {
        waiter                  = std::make_shared<JobWaiter>();
        waiting_threads[job_id] = waiter;
      }
    }
    waiter->waiters.fetch_add(1, std::memory_order_relaxed);

    // Wait with periodic cancellation and status checks (every 200ms)
    {
      std::unique_lock<std::mutex> lock(waiter->mutex);
      while (!waiter->ready) {
        if (context->IsCancelled()) {
          waiter->waiters.fetch_sub(1, std::memory_order_relaxed);
          return Status(StatusCode::CANCELLED, "Client cancelled WaitForCompletion");
        }
        // Re-check job status in case the notification was missed
        std::string msg;
        JobStatus current = check_job_status(job_id, msg);
        if (current == JobStatus::COMPLETED || current == JobStatus::FAILED ||
            current == JobStatus::CANCELLED) {
          break;
        }
        waiter->cv.wait_for(lock, std::chrono::milliseconds(200));
      }
    }

    if (waiter->success) {
      response->set_job_status(cuopt::remote::COMPLETED);
      response->set_message("");
      {
        std::lock_guard<std::mutex> tlock(tracker_mutex);
        auto tit = job_tracker.find(job_id);
        response->set_result_size_bytes((tit != job_tracker.end()) ? tit->second.result_size_bytes
                                                                   : 0);
      }
    } else {
      // Waiter wasn't signaled or job failed/cancelled -- query authoritative status
      std::string msg;
      JobStatus status = check_job_status(job_id, msg);
      switch (status) {
        case JobStatus::COMPLETED: {
          response->set_job_status(cuopt::remote::COMPLETED);
          response->set_message("");
          std::lock_guard<std::mutex> tlock2(tracker_mutex);
          auto tit2 = job_tracker.find(job_id);
          response->set_result_size_bytes(
            (tit2 != job_tracker.end()) ? tit2->second.result_size_bytes : 0);
          break;
        }
        case JobStatus::FAILED: response->set_job_status(cuopt::remote::FAILED); break;
        case JobStatus::CANCELLED: response->set_job_status(cuopt::remote::CANCELLED); break;
        case JobStatus::NOT_FOUND: response->set_job_status(cuopt::remote::NOT_FOUND); break;
        default: response->set_job_status(cuopt::remote::FAILED); break;
      }
      if (status != JobStatus::COMPLETED) {
        response->set_message(msg);
        response->set_result_size_bytes(0);
      }
    }
    waiter->waiters.fetch_sub(1, std::memory_order_relaxed);

    if (config.verbose) {
      std::cout << "[gRPC] WaitForCompletion finished job_id=" << job_id << "\n";
      std::cout.flush();
    }

    return Status::OK;
  }

  Status StreamLogs(ServerContext* context,
                    const cuopt::remote::StreamLogsRequest* request,
                    ServerWriter<cuopt::remote::LogMessage>* writer) override
  {
    const std::string job_id   = request->job_id();
    int64_t from_byte          = request->from_byte();
    const std::string log_path = get_log_file_path(job_id);

    // Wait for the log file to appear (job might not have started yet).
    int waited_ms = 0;
    while (!context->IsCancelled()) {
      struct stat st;
      if (stat(log_path.c_str(), &st) == 0) { break; }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      waited_ms += 50;
      // Give up quickly if job doesn't exist.
      if (waited_ms >= 2000) {
        std::string msg;
        JobStatus s = check_job_status(job_id, msg);
        if (s == JobStatus::NOT_FOUND) {
          if (config.verbose) {
            std::cout << "[gRPC] StreamLogs job not found: " << job_id << std::endl;
          }
          return Status(grpc::StatusCode::NOT_FOUND, "Job not found: " + job_id);
        }
        // else job exists but log not yet created; keep waiting
        waited_ms = 0;
      }
    }

    std::ifstream in(log_path, std::ios::in | std::ios::binary);
    if (!in.is_open()) {
      cuopt::remote::LogMessage m;
      m.set_line("Failed to open log file");
      m.set_byte_offset(from_byte);
      m.set_job_complete(true);
      writer->Write(m);
      return Status::OK;
    }

    if (from_byte > 0) { in.seekg(from_byte, std::ios::beg); }

    int64_t current_offset = from_byte;
    std::string line;

    while (!context->IsCancelled()) {
      std::streampos before = in.tellg();
      if (before >= 0) { current_offset = static_cast<int64_t>(before); }

      if (std::getline(in, line)) {
        // Account for the newline consumed by getline (1 byte) if present in file.
        std::streampos after = in.tellg();
        int64_t next_offset  = current_offset;
        if (after >= 0) {
          next_offset = static_cast<int64_t>(after);
        } else {
          // tellg can be -1 at EOF; approximate
          next_offset = current_offset + static_cast<int64_t>(line.size());
        }

        cuopt::remote::LogMessage m;
        m.set_line(line);
        m.set_byte_offset(next_offset);
        m.set_job_complete(false);
        if (!writer->Write(m)) { break; }
        continue;
      }

      // No new line available: clear EOF and sleep briefly
      if (in.eof()) {
        in.clear();
      } else if (in.fail()) {
        in.clear();
      }

      // If job is in terminal state and we've drained file, finish the stream.
      std::string msg;
      JobStatus s = check_job_status(job_id, msg);
      if (s == JobStatus::COMPLETED || s == JobStatus::FAILED || s == JobStatus::CANCELLED) {
        // One last attempt to read any remaining partial line
        std::streampos before2 = in.tellg();
        if (before2 >= 0) { current_offset = static_cast<int64_t>(before2); }
        if (std::getline(in, line)) {
          std::streampos after2 = in.tellg();
          int64_t next_offset2  = current_offset + static_cast<int64_t>(line.size());
          if (after2 >= 0) { next_offset2 = static_cast<int64_t>(after2); }
          cuopt::remote::LogMessage m;
          m.set_line(line);
          m.set_byte_offset(next_offset2);
          m.set_job_complete(false);
          writer->Write(m);
        }

        cuopt::remote::LogMessage done;
        done.set_line("");
        done.set_byte_offset(current_offset);
        done.set_job_complete(true);
        writer->Write(done);
        return Status::OK;
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return Status::OK;
  }

  Status GetIncumbents(ServerContext* context,
                       const cuopt::remote::IncumbentRequest* request,
                       cuopt::remote::IncumbentResponse* response) override
  {
    (void)context;
    const std::string job_id = request->job_id();
    int64_t from_index       = request->from_index();
    int32_t max_count        = request->max_count();

    if (from_index < 0) { from_index = 0; }

    std::lock_guard<std::mutex> lock(tracker_mutex);
    auto it = job_tracker.find(job_id);
    if (it == job_tracker.end()) { return Status(StatusCode::NOT_FOUND, "Job not found"); }

    const auto& incumbents = it->second.incumbents;
    int64_t available      = static_cast<int64_t>(incumbents.size());
    if (from_index > available) { from_index = available; }

    int64_t count = available - from_index;
    if (max_count > 0 && count > max_count) { count = max_count; }

    for (int64_t i = 0; i < count; ++i) {
      const auto& inc = incumbents[static_cast<size_t>(from_index + i)];
      auto* out       = response->add_incumbents();
      out->set_index(from_index + i);
      out->set_objective(inc.objective);
      for (double v : inc.assignment) {
        out->add_assignment(v);
      }
      out->set_job_id(job_id);
    }

    response->set_next_index(available);
    bool done =
      (it->second.status == JobStatus::COMPLETED || it->second.status == JobStatus::FAILED ||
       it->second.status == JobStatus::CANCELLED);
    response->set_job_complete(done);
    if (config.verbose) {
      std::cout << "[gRPC] GetIncumbents job_id=" << job_id << " from=" << from_index
                << " returned=" << response->incumbents_size() << " next=" << available
                << " done=" << (done ? 1 : 0) << "\n";
      std::cout.flush();
    }
    return Status::OK;
  }
};

// ============================================================================
// Main
// ============================================================================

void print_usage(const char* prog)
{
  std::cout
    << "Usage: " << prog << " [options]\n"
    << "Options:\n"
    << "  -p, --port PORT         Listen port (default: 8765)\n"
    << "  -w, --workers NUM       Number of worker processes (default: 1)\n"
    << "      --max-message-mb N  gRPC max send/recv message size in MiB (default: 256, "
       "0=unlimited)\n"
    << "      --max-message-bytes N  Override max message size in bytes (min 4096, for testing)\n"
    << "      --chunk-timeout N   Per-chunk timeout in seconds for streaming (default: 60, "
       "0=disabled)\n"
    << "      --enable-transfer-hash  Log data hashes for streaming transfers (for testing)\n"
    << "      --tls               Enable TLS (requires --tls-cert and --tls-key)\n"
    << "      --tls-cert PATH     Path to PEM-encoded server certificate\n"
    << "      --tls-key PATH      Path to PEM-encoded server private key\n"
    << "      --tls-root PATH     Path to PEM root certs for client verification\n"
    << "      --require-client-cert  Require and verify client certs (mTLS)\n"
    << "      --log-to-console    Enable solver log output to console (default: off)\n"
    << "  -q, --quiet             Reduce verbosity\n"
    << "  -h, --help              Show this help\n";
}

int main(int argc, char** argv)
{
  // Parse arguments
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "-p" || arg == "--port") {
      if (i + 1 < argc) { config.port = std::stoi(argv[++i]); }
    } else if (arg == "-w" || arg == "--workers") {
      if (i + 1 < argc) { config.num_workers = std::stoi(argv[++i]); }
    } else if (arg == "--max-message-mb") {
      if (i + 1 < argc) { config.max_message_mb = std::stoi(argv[++i]); }
    } else if (arg == "--max-message-bytes") {
      if (i + 1 < argc) { config.max_message_b = std::max(4096LL, std::stoll(argv[++i])); }
    } else if (arg == "--enable-transfer-hash") {
      config.enable_transfer_hash = true;
    } else if (arg == "--tls") {
      config.enable_tls = true;
    } else if (arg == "--tls-cert") {
      if (i + 1 < argc) { config.tls_cert_path = argv[++i]; }
    } else if (arg == "--tls-key") {
      if (i + 1 < argc) { config.tls_key_path = argv[++i]; }
    } else if (arg == "--tls-root") {
      if (i + 1 < argc) { config.tls_root_path = argv[++i]; }
    } else if (arg == "--require-client-cert") {
      config.require_client = true;
    } else if (arg == "--log-to-console") {
      config.log_to_console = true;
    } else if (arg == "-q" || arg == "--quiet") {
      config.verbose = false;
    } else if (arg == "-h" || arg == "--help") {
      print_usage(argv[0]);
      return 0;
    }
  }

  std::cout << "cuOpt gRPC Remote Solve Server\n"
            << "==============================\n"
            << "Port: " << config.port << "\n"
            << "Workers: " << config.num_workers << "\n"
            << std::endl;
  std::cout.flush();

  // Setup signal handling
  signal(SIGINT, signal_handler);
  signal(SIGTERM, signal_handler);

  // Create log directory
  ensure_log_dir_exists();

  // Avoid stale shared-memory state from prior crashed/force-killed runs.
  // Old worker processes may still have the old segments mapped; unlinking
  // here ensures this server creates fresh segments with clean state.
  shm_unlink(SHM_JOB_QUEUE);
  shm_unlink(SHM_RESULT_QUEUE);
  shm_unlink(SHM_CONTROL);

  // Initialize shared memory
  int shm_fd = shm_open(SHM_JOB_QUEUE, O_CREAT | O_RDWR, 0600);
  if (shm_fd < 0) {
    std::cerr << "[Server] Failed to create shared memory for job queue: " << strerror(errno)
              << "\n";
    return 1;
  }
  if (ftruncate(shm_fd, sizeof(JobQueueEntry) * MAX_JOBS) < 0) {
    std::cerr << "[Server] Failed to ftruncate job queue: " << strerror(errno) << "\n";
    close(shm_fd);
    return 1;
  }
  job_queue = static_cast<JobQueueEntry*>(
    mmap(nullptr, sizeof(JobQueueEntry) * MAX_JOBS, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0));
  close(shm_fd);

  if (job_queue == MAP_FAILED) {
    std::cerr << "[Server] Failed to mmap job queue: " << strerror(errno) << "\n";
    return 1;
  }
  int result_shm_fd = shm_open(SHM_RESULT_QUEUE, O_CREAT | O_RDWR, 0600);
  if (result_shm_fd < 0) {
    std::cerr << "[Server] Failed to create result queue shm: " << strerror(errno) << "\n";
    return 1;
  }
  if (ftruncate(result_shm_fd, sizeof(ResultQueueEntry) * MAX_RESULTS) < 0) {
    std::cerr << "[Server] Failed to ftruncate result queue: " << strerror(errno) << "\n";
    close(result_shm_fd);
    return 1;
  }
  result_queue = static_cast<ResultQueueEntry*>(mmap(nullptr,
                                                     sizeof(ResultQueueEntry) * MAX_RESULTS,
                                                     PROT_READ | PROT_WRITE,
                                                     MAP_SHARED,
                                                     result_shm_fd,
                                                     0));
  close(result_shm_fd);
  if (result_queue == MAP_FAILED) {
    std::cerr << "[Server] Failed to mmap result queue: " << strerror(errno) << "\n";
    return 1;
  }
  int ctrl_shm_fd = shm_open(SHM_CONTROL, O_CREAT | O_RDWR, 0600);
  if (ctrl_shm_fd < 0) {
    std::cerr << "[Server] Failed to create control shm: " << strerror(errno) << "\n";
    return 1;
  }
  if (ftruncate(ctrl_shm_fd, sizeof(SharedMemoryControl)) < 0) {
    std::cerr << "[Server] Failed to ftruncate control: " << strerror(errno) << "\n";
    close(ctrl_shm_fd);
    return 1;
  }
  shm_ctrl = static_cast<SharedMemoryControl*>(
    mmap(nullptr, sizeof(SharedMemoryControl), PROT_READ | PROT_WRITE, MAP_SHARED, ctrl_shm_fd, 0));
  close(ctrl_shm_fd);
  if (shm_ctrl == MAP_FAILED) {
    std::cerr << "[Server] Failed to mmap control: " << strerror(errno) << "\n";
    return 1;
  }
  // Initialize shared memory
  for (size_t i = 0; i < MAX_JOBS; ++i) {
    memset(&job_queue[i], 0, sizeof(JobQueueEntry));
    job_queue[i].ready.store(false);
    job_queue[i].claimed.store(false);
    job_queue[i].cancelled.store(false);
    job_queue[i].worker_index.store(-1);
  }

  for (size_t i = 0; i < MAX_RESULTS; ++i) {
    memset(&result_queue[i], 0, sizeof(ResultQueueEntry));
    result_queue[i].ready.store(false);
    result_queue[i].retrieved.store(false);
  }

  shm_ctrl->shutdown_requested.store(false);
  shm_ctrl->active_workers.store(0);

  // Spawn worker processes
  spawn_workers();

  // Start result retrieval thread
  std::thread result_thread(result_retrieval_thread);

  // Start incumbent retrieval thread
  std::thread incumbent_thread(incumbent_retrieval_thread);

  // Start worker monitor thread
  std::thread monitor_thread(worker_monitor_thread);

  // Start session reaper thread (evicts stale chunked upload/download sessions)
  std::thread reaper_thread(session_reaper_thread);

  // Start gRPC server
  std::string server_address = "0.0.0.0:" + std::to_string(config.port);
  CuOptRemoteServiceImpl service;

  ServerBuilder builder;
  std::shared_ptr<grpc::ServerCredentials> creds;
  if (config.enable_tls) {
    if (config.tls_cert_path.empty() || config.tls_key_path.empty()) {
      std::cerr << "[Server] TLS enabled but --tls-cert/--tls-key not provided\n";
      return 1;
    }
    grpc::SslServerCredentialsOptions ssl_opts;
    grpc::SslServerCredentialsOptions::PemKeyCertPair key_cert;
    key_cert.cert_chain  = read_file_to_string(config.tls_cert_path);
    key_cert.private_key = read_file_to_string(config.tls_key_path);
    if (key_cert.cert_chain.empty() || key_cert.private_key.empty()) {
      std::cerr << "[Server] Failed to read TLS cert/key files\n";
      return 1;
    }
    ssl_opts.pem_key_cert_pairs.push_back(key_cert);

    if (!config.tls_root_path.empty()) {
      ssl_opts.pem_root_certs = read_file_to_string(config.tls_root_path);
      if (ssl_opts.pem_root_certs.empty()) {
        std::cerr << "[Server] Failed to read TLS root cert file\n";
        return 1;
      }
    }

    if (config.require_client) {
      if (ssl_opts.pem_root_certs.empty()) {
        std::cerr << "[Server] --require-client-cert requires --tls-root\n";
        return 1;
      }
      ssl_opts.client_certificate_request =
        GRPC_SSL_REQUEST_AND_REQUIRE_CLIENT_CERTIFICATE_AND_VERIFY;
    } else if (!ssl_opts.pem_root_certs.empty()) {
      ssl_opts.client_certificate_request = GRPC_SSL_REQUEST_CLIENT_CERTIFICATE_AND_VERIFY;
    }

    creds = grpc::SslServerCredentials(ssl_opts);
  } else {
    creds = grpc::InsecureServerCredentials();
  }

  builder.AddListeningPort(server_address, creds);
  builder.RegisterService(&service);
  // Allow large LP/MIP payloads (e.g. large MPS problems).
  // Note: gRPC uses -1 to mean unlimited.
  const int64_t max_bytes = server_max_message_bytes();
  const int channel_limit =
    (max_bytes <= 0)
      ? -1
      : static_cast<int>(std::min<int64_t>(max_bytes, std::numeric_limits<int>::max()));
  builder.SetMaxReceiveMessageSize(channel_limit);
  builder.SetMaxSendMessageSize(channel_limit);

  std::unique_ptr<Server> server(builder.BuildAndStart());
  std::cout << "[gRPC Server] Listening on " << server_address << std::endl;
  std::cout << "[gRPC Server] Workers: " << config.num_workers << std::endl;
  std::cout << "[gRPC Server] Max message size: " << server_max_message_bytes() << " bytes"
            << (config.max_message_b >= 0 ? " (--max-message-bytes override)"
                                          : " (" + std::to_string(config.max_message_mb) + " MiB)")
            << std::endl;
  std::cout << "[gRPC Server] Press Ctrl+C to shutdown" << std::endl;

  // Wait for shutdown signal. We can't rely on signal handler to break server->Wait(),
  // so use a helper thread to call Shutdown() when keep_running flips false.
  std::thread shutdown_thread([&server]() {
    while (keep_running.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (server) { server->Shutdown(); }
  });

  server->Wait();
  if (shutdown_thread.joinable()) shutdown_thread.join();

  // Cleanup
  std::cout << "\n[Server] Shutting down..." << std::endl;
  keep_running                 = false;
  shm_ctrl->shutdown_requested = true;
  result_cv.notify_all();

  if (result_thread.joinable()) result_thread.join();
  if (incumbent_thread.joinable()) incumbent_thread.join();
  if (monitor_thread.joinable()) monitor_thread.join();
  if (reaper_thread.joinable()) reaper_thread.join();

  wait_for_workers();
  cleanup_shared_memory();

  std::cout << "[Server] Shutdown complete" << std::endl;
  return 0;
}

#else  // !CUOPT_ENABLE_GRPC

#include <iostream>

int main()
{
  std::cerr << "Error: cuopt_grpc_server requires gRPC support.\n"
            << "Rebuild with gRPC enabled (CUOPT_ENABLE_GRPC=ON)" << std::endl;
  return 1;
}

#endif  // CUOPT_ENABLE_GRPC
