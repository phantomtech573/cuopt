/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifdef CUOPT_ENABLE_GRPC

#include "cuopt_remote.pb.h"
#include "cuopt_remote_service.pb.h"
#include "grpc_field_element_size.hpp"

#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <vector>

// Requested pipe buffer size (1 MiB). The kernel default is 64 KiB, which
// forces excessive context-switching on large transfers. fcntl(F_SETPIPE_SZ)
// may silently cap this to /proc/sys/fs/pipe-max-size.
static constexpr int kPipeBufferSize = 1024 * 1024;

// Pipe I/O primitives defined in grpc_job_management.cpp.
bool write_to_pipe(int fd, const void* data, size_t size);
bool read_from_pipe(int fd, void* data, size_t size, int timeout_ms = 120000);

// =============================================================================
// Low-level: write/read a single protobuf message with a uint32 length prefix.
// Uses standard protobuf SerializeToArray / ParseFromArray for the payload.
// =============================================================================

inline bool write_protobuf_to_pipe(int fd, const google::protobuf::MessageLite& msg)
{
  uint32_t size = static_cast<uint32_t>(msg.ByteSizeLong());
  if (!write_to_pipe(fd, &size, sizeof(size))) return false;
  std::vector<uint8_t> buf(size);
  if (!msg.SerializeToArray(buf.data(), static_cast<int>(size))) return false;
  return write_to_pipe(fd, buf.data(), size);
}

inline bool read_protobuf_from_pipe(int fd, google::protobuf::MessageLite& msg)
{
  uint32_t size;
  if (!read_from_pipe(fd, &size, sizeof(size))) return false;
  std::vector<uint8_t> buf(size);
  if (!read_from_pipe(fd, buf.data(), size)) return false;
  return msg.ParseFromArray(buf.data(), static_cast<int>(size));
}

// =============================================================================
// Chunked request: server → worker pipe (ChunkedProblemHeader + raw arrays)
//
// Wire format (protobuf header + raw byte arrays):
//   [uint32 hdr_size][protobuf header bytes]
//   [uint32 num_arrays]
//   per array: [int32 field_id][uint64 total_bytes][raw bytes...]
//
// The protobuf ChunkedProblemHeader carries all metadata (settings, field
// types, element counts). Array data bypasses protobuf serialization and
// flows directly through the pipe as raw bytes.
// =============================================================================

inline bool write_chunked_request_to_pipe(int fd,
                                          const cuopt::remote::ChunkedProblemHeader& header,
                                          const std::vector<cuopt::remote::ArrayChunk>& chunks)
{
  // Step 1: write the protobuf header (settings, scalars, string arrays).
  if (!write_protobuf_to_pipe(fd, header)) return false;

  // Step 2: group incoming gRPC chunks by field_id. A single field may arrive
  // as multiple chunks (the client splits large arrays at chunk_size_bytes).
  struct FieldInfo {
    std::vector<const cuopt::remote::ArrayChunk*> chunks;
    int64_t total_bytes = 0;
  };
  std::map<int32_t, FieldInfo> fields;
  for (const auto& ac : chunks) {
    int32_t fid = static_cast<int32_t>(ac.field_id());
    auto& fi    = fields[fid];
    fi.chunks.push_back(&ac);
    if (fi.total_bytes == 0 && ac.total_elements() > 0) {
      auto elem_size = array_field_element_size(ac.field_id());
      if (elem_size > 0 && ac.total_elements() <= std::numeric_limits<int64_t>::max() / elem_size) {
        fi.total_bytes = ac.total_elements() * elem_size;
      }
    }
  }

  // Step 3: write per-field raw byte arrays.
  uint32_t num_arrays = static_cast<uint32_t>(fields.size());
  if (!write_to_pipe(fd, &num_arrays, sizeof(num_arrays))) return false;

  for (const auto& [fid, fi] : fields) {
    int32_t field_id     = fid;
    uint64_t total_bytes = static_cast<uint64_t>(fi.total_bytes);
    if (!write_to_pipe(fd, &field_id, sizeof(field_id))) return false;
    if (!write_to_pipe(fd, &total_bytes, sizeof(total_bytes))) return false;
    if (total_bytes == 0) continue;

    // Fast path: field arrived in a single chunk that covers the whole array.
    // Write directly from the protobuf bytes string, avoiding an assembly copy.
    if (fi.chunks.size() == 1 && fi.chunks[0]->element_offset() == 0 &&
        static_cast<int64_t>(fi.chunks[0]->data().size()) == fi.total_bytes) {
      if (!write_to_pipe(fd, fi.chunks[0]->data().data(), fi.chunks[0]->data().size()))
        return false;
    } else {
      // Slow path: stitch multiple chunks into a contiguous buffer, placing
      // each chunk at its element_offset * elem_size byte position.
      std::vector<uint8_t> assembled(static_cast<size_t>(fi.total_bytes), 0);
      int64_t elem_size = (fi.total_bytes > 0 && fi.chunks[0]->total_elements() > 0)
                            ? fi.total_bytes / fi.chunks[0]->total_elements()
                            : 1;
      for (const auto* ac : fi.chunks) {
        int64_t byte_offset    = ac->element_offset() * elem_size;
        const auto& chunk_data = ac->data();
        if (byte_offset >= 0 &&
            byte_offset + static_cast<int64_t>(chunk_data.size()) <= fi.total_bytes) {
          std::memcpy(assembled.data() + byte_offset, chunk_data.data(), chunk_data.size());
        }
      }
      if (!write_to_pipe(fd, assembled.data(), assembled.size())) return false;
    }
  }

  return true;
}

inline bool read_chunked_request_from_pipe(int fd,
                                           cuopt::remote::ChunkedProblemHeader& header_out,
                                           std::map<int32_t, std::vector<uint8_t>>& arrays_out)
{
  if (!read_protobuf_from_pipe(fd, header_out)) return false;

  uint32_t num_arrays;
  if (!read_from_pipe(fd, &num_arrays, sizeof(num_arrays))) return false;

  // Read each field's raw bytes directly into the output map, keyed by field_id.
  for (uint32_t i = 0; i < num_arrays; ++i) {
    int32_t field_id;
    uint64_t total_bytes;
    if (!read_from_pipe(fd, &field_id, sizeof(field_id))) return false;
    if (!read_from_pipe(fd, &total_bytes, sizeof(total_bytes))) return false;
    auto& dest = arrays_out[field_id];
    dest.resize(static_cast<size_t>(total_bytes));
    if (total_bytes > 0 && !read_from_pipe(fd, dest.data(), static_cast<size_t>(total_bytes)))
      return false;
  }

  return true;
}

// =============================================================================
// Result: worker → server pipe (ChunkedResultHeader + raw arrays)
//
// Same wire format as the chunked request above. Unlike the request path,
// result arrays are already assembled into contiguous vectors by the worker,
// so no chunk grouping or assembly is needed.
// =============================================================================

inline bool write_result_to_pipe(int fd,
                                 const cuopt::remote::ChunkedResultHeader& header,
                                 const std::map<int32_t, std::vector<uint8_t>>& arrays)
{
  if (!write_protobuf_to_pipe(fd, header)) return false;

  uint32_t num_arrays = static_cast<uint32_t>(arrays.size());
  if (!write_to_pipe(fd, &num_arrays, sizeof(num_arrays))) return false;

  // Each array is already contiguous — write field_id, size, and raw bytes.
  for (const auto& [fid, data] : arrays) {
    int32_t field_id     = fid;
    uint64_t total_bytes = data.size();
    if (!write_to_pipe(fd, &field_id, sizeof(field_id))) return false;
    if (!write_to_pipe(fd, &total_bytes, sizeof(total_bytes))) return false;
    if (total_bytes > 0 && !write_to_pipe(fd, data.data(), data.size())) return false;
  }

  return true;
}

inline bool read_result_from_pipe(int fd,
                                  cuopt::remote::ChunkedResultHeader& header_out,
                                  std::map<int32_t, std::vector<uint8_t>>& arrays_out)
{
  if (!read_protobuf_from_pipe(fd, header_out)) return false;

  uint32_t num_arrays;
  if (!read_from_pipe(fd, &num_arrays, sizeof(num_arrays))) return false;

  for (uint32_t i = 0; i < num_arrays; ++i) {
    int32_t field_id;
    uint64_t total_bytes;
    if (!read_from_pipe(fd, &field_id, sizeof(field_id))) return false;
    if (!read_from_pipe(fd, &total_bytes, sizeof(total_bytes))) return false;
    auto& dest = arrays_out[field_id];
    dest.resize(static_cast<size_t>(total_bytes));
    if (total_bytes > 0 && !read_from_pipe(fd, dest.data(), static_cast<size_t>(total_bytes)))
      return false;
  }

  return true;
}

// Serialize a SubmitJobRequest directly to a pipe blob using standard protobuf.
// Used for unary submits only (always well under 2 GiB).
inline std::vector<uint8_t> serialize_submit_request_to_pipe(
  const cuopt::remote::SubmitJobRequest& request)
{
  size_t byte_size = request.ByteSizeLong();
  std::vector<uint8_t> blob(byte_size);
  request.SerializeToArray(blob.data(), static_cast<int>(byte_size));
  return blob;
}

#endif  // CUOPT_ENABLE_GRPC
