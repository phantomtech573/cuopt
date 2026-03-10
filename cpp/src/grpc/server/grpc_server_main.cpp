/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file grpc_server_main.cpp
 * @brief gRPC-based remote solve server entry point
 *
 * This server uses gRPC for client communication with fork-based worker
 * process infrastructure:
 * - Worker processes with shared memory job queues
 * - Pipe-based IPC for problem/result data
 * - Result tracking and retrieval threads
 * - Log streaming
 */

#ifdef CUOPT_ENABLE_GRPC

#include "grpc_server_types.hpp"

// Defined in grpc_service_impl.cpp
std::unique_ptr<grpc::Service> create_cuopt_grpc_service();

void print_usage(const char* prog)
{
  std::cout
    << "Usage: " << prog << " [options]\n"
    << "Options:\n"
    << "  -p, --port PORT         Listen port (default: 8765)\n"
    << "  -w, --workers NUM       Number of worker processes (default: 1)\n"
    << "      --max-message-mb N  gRPC max send/recv message size in MiB (default: 256)\n"
    << "      --max-message-bytes N  Set max message size in exact bytes (min 4096, for testing)\n"
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
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "-p" || arg == "--port") {
      if (i + 1 < argc) { config.port = std::stoi(argv[++i]); }
    } else if (arg == "-w" || arg == "--workers") {
      if (i + 1 < argc) { config.num_workers = std::stoi(argv[++i]); }
    } else if (arg == "--max-message-mb") {
      if (i + 1 < argc) {
        config.max_message_bytes = static_cast<int64_t>(std::stoi(argv[++i])) * kMiB;
      }
    } else if (arg == "--max-message-bytes") {
      if (i + 1 < argc) { config.max_message_bytes = std::max(4096LL, std::stoll(argv[++i])); }
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

  config.max_message_bytes =
    std::clamp(config.max_message_bytes, kServerMinMessageBytes, kServerMaxMessageBytes);

  std::cout << "cuOpt gRPC Remote Solve Server\n"
            << "==============================\n"
            << "Port: " << config.port << "\n"
            << "Workers: " << config.num_workers << "\n"
            << std::endl;
  std::cout.flush();

  signal(SIGINT, signal_handler);
  signal(SIGTERM, signal_handler);

  ensure_log_dir_exists();

  shm_unlink(SHM_JOB_QUEUE);
  shm_unlink(SHM_RESULT_QUEUE);
  shm_unlink(SHM_CONTROL);

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

  for (size_t i = 0; i < MAX_JOBS; ++i) {
    new (&job_queue[i]) JobQueueEntry{};
    job_queue[i].ready.store(false);
    job_queue[i].claimed.store(false);
    job_queue[i].cancelled.store(false);
    job_queue[i].worker_index.store(-1);
  }

  for (size_t i = 0; i < MAX_RESULTS; ++i) {
    new (&result_queue[i]) ResultQueueEntry{};
    result_queue[i].claimed.store(false);
    result_queue[i].ready.store(false);
    result_queue[i].retrieved.store(false);
  }

  shm_ctrl->shutdown_requested.store(false);
  shm_ctrl->active_workers.store(0);

  // Build credentials before spawning workers so TLS validation failures
  // don't leak worker processes or background threads.
  std::string server_address = "0.0.0.0:" + std::to_string(config.port);
  std::shared_ptr<grpc::ServerCredentials> creds;
  if (config.enable_tls) {
    if (config.tls_cert_path.empty() || config.tls_key_path.empty()) {
      std::cerr << "[Server] TLS enabled but --tls-cert/--tls-key not provided\n";
      cleanup_shared_memory();
      return 1;
    }
    grpc::SslServerCredentialsOptions ssl_opts;
    grpc::SslServerCredentialsOptions::PemKeyCertPair key_cert;
    key_cert.cert_chain  = read_file_to_string(config.tls_cert_path);
    key_cert.private_key = read_file_to_string(config.tls_key_path);
    if (key_cert.cert_chain.empty() || key_cert.private_key.empty()) {
      std::cerr << "[Server] Failed to read TLS cert/key files\n";
      cleanup_shared_memory();
      return 1;
    }
    ssl_opts.pem_key_cert_pairs.push_back(key_cert);

    if (!config.tls_root_path.empty()) {
      ssl_opts.pem_root_certs = read_file_to_string(config.tls_root_path);
      if (ssl_opts.pem_root_certs.empty()) {
        std::cerr << "[Server] Failed to read TLS root cert file\n";
        cleanup_shared_memory();
        return 1;
      }
    }

    if (config.require_client) {
      if (ssl_opts.pem_root_certs.empty()) {
        std::cerr << "[Server] --require-client-cert requires --tls-root\n";
        cleanup_shared_memory();
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

  spawn_workers();

  std::thread result_thread(result_retrieval_thread);
  std::thread incumbent_thread(incumbent_retrieval_thread);
  std::thread monitor_thread(worker_monitor_thread);
  std::thread reaper_thread(session_reaper_thread);

  auto shutdown_all = [&]() {
    keep_running                 = false;
    shm_ctrl->shutdown_requested = true;
    result_cv.notify_all();

    if (result_thread.joinable()) result_thread.join();
    if (incumbent_thread.joinable()) incumbent_thread.join();
    if (monitor_thread.joinable()) monitor_thread.join();
    if (reaper_thread.joinable()) reaper_thread.join();

    wait_for_workers();
    cleanup_shared_memory();
  };

  auto service = create_cuopt_grpc_service();

  ServerBuilder builder;
  builder.AddListeningPort(server_address, creds);
  builder.RegisterService(service.get());
  const int64_t max_bytes = server_max_message_bytes();
  const int channel_limit =
    static_cast<int>(std::min<int64_t>(max_bytes, std::numeric_limits<int>::max()));
  builder.SetMaxReceiveMessageSize(channel_limit);
  builder.SetMaxSendMessageSize(channel_limit);

  std::unique_ptr<Server> server(builder.BuildAndStart());
  if (!server) {
    std::cerr << "[Server] BuildAndStart() failed — could not bind to " << server_address << "\n";
    shutdown_all();
    return 1;
  }

  std::cout << "[gRPC Server] Listening on " << server_address << std::endl;
  std::cout << "[gRPC Server] Workers: " << config.num_workers << std::endl;
  std::cout << "[gRPC Server] Max message size: " << server_max_message_bytes() << " bytes ("
            << (server_max_message_bytes() / kMiB) << " MiB)" << std::endl;
  std::cout << "[gRPC Server] Press Ctrl+C to shutdown" << std::endl;

  std::thread shutdown_thread([&server]() {
    while (keep_running.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (server) { server->Shutdown(); }
  });

  server->Wait();
  if (shutdown_thread.joinable()) shutdown_thread.join();

  std::cout << "\n[Server] Shutting down..." << std::endl;
  shutdown_all();

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
