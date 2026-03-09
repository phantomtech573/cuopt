/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#ifdef CUOPT_ENABLE_GRPC

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <iostream>

#include <poll.h>
#include <unistd.h>

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

  // Poll once to enforce timeout before the first read. After data starts
  // flowing, blocking read() is sufficient — if the writer dies the pipe
  // closes and read() returns 0 (EOF). Avoids ~10k extra poll() syscalls
  // per bulk transfer.
  struct pollfd pfd = {fd, POLLIN, 0};
  int pr;
  do {
    pr = poll(&pfd, 1, timeout_ms);
  } while (pr < 0 && errno == EINTR);
  if (pr < 0) {
    std::cerr << "[Server] poll() failed on pipe: " << strerror(errno) << "\n";
    return false;
  }
  if (pr == 0) {
    std::cerr << "[Server] Timeout waiting for pipe data (waited " << timeout_ms << "ms)\n";
    return false;
  }
  if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) {
    std::cerr << "[Server] Pipe error/hangup detected\n";
    return false;
  }

  while (remaining > 0) {
    ssize_t nread = ::read(fd, ptr, remaining);
    if (nread > 0) {
      ptr += nread;
      remaining -= nread;
      continue;
    }
    if (nread == 0) {
      std::cerr << "[Server] Pipe EOF (writer closed)\n";
      return false;
    }
    if (errno == EINTR) continue;
    std::cerr << "[Server] Pipe read error: " << strerror(errno) << "\n";
    return false;
  }
  return true;
}

#endif  // CUOPT_ENABLE_GRPC
