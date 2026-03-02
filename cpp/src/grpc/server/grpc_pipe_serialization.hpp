/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifdef CUOPT_ENABLE_GRPC

#include "cuopt_remote.pb.h"
#include "cuopt_remote_service.pb.h"
#include "grpc_field_element_size.hpp"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <google/protobuf/util/delimited_message_util.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <vector>

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
    if (!google::protobuf::util::ParseDelimitedFromCodedStream(&ac, &coded, &clean_eof)) {
      if (!clean_eof) return false;
      break;
    }
    int32_t fid = static_cast<int32_t>(ac.field_id());
    auto& dest  = arrays_out[fid];
    if (dest.empty() && ac.total_elements() > 0) {
      dest.resize(static_cast<size_t>(ac.total_elements()), 0);
    }
    int64_t offset         = ac.element_offset();
    const auto& chunk_data = ac.data();
    if (offset < 0 || static_cast<size_t>(offset) > dest.size()) continue;
    if (offset + static_cast<int64_t>(chunk_data.size()) <= static_cast<int64_t>(dest.size())) {
      std::memcpy(dest.data() + offset, chunk_data.data(), chunk_data.size());
    }
  }
  return true;
}

// Serialize a chunked request (ChunkedProblemHeader + ArrayChunk messages) into a pipe blob.
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

  while (!clean_eof) {
    cuopt::remote::ArrayChunk ac;
    if (!google::protobuf::util::ParseDelimitedFromCodedStream(&ac, &coded, &clean_eof)) {
      if (!clean_eof) return false;
      break;
    }
    int32_t fid = static_cast<int32_t>(ac.field_id());
    auto& dest  = arrays_out[fid];
    if (dest.empty() && ac.total_elements() > 0) {
      int64_t elem_size = array_field_element_size(ac.field_id());
      dest.resize(static_cast<size_t>(ac.total_elements() * elem_size), 0);
    }
    int64_t elem_size      = (dest.size() > 0 && ac.total_elements() > 0)
                               ? static_cast<int64_t>(dest.size()) / ac.total_elements()
                               : 1;
    int64_t byte_offset    = ac.element_offset() * elem_size;
    const auto& chunk_data = ac.data();
    if (byte_offset < 0 || static_cast<size_t>(byte_offset) > dest.size()) continue;
    if (byte_offset + static_cast<int64_t>(chunk_data.size()) <=
        static_cast<int64_t>(dest.size())) {
      std::memcpy(dest.data() + byte_offset, chunk_data.data(), chunk_data.size());
    }
  }
  return true;
}

// Serialize a SubmitJobRequest directly to a pipe blob using standard protobuf.
inline std::vector<uint8_t> serialize_submit_request_to_pipe(
  const cuopt::remote::SubmitJobRequest& request)
{
  size_t byte_size = request.ByteSizeLong();
  std::vector<uint8_t> blob(byte_size);
  request.SerializeToArray(blob.data(), static_cast<int>(byte_size));
  return blob;
}

#endif  // CUOPT_ENABLE_GRPC
