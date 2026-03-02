/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#include "grpc_service_mapper.hpp"

#include <cuopt/linear_programming/constants.h>
#include <cuopt_remote.pb.h>
#include <cuopt_remote_service.pb.h>
#include <cuopt/linear_programming/cpu_optimization_problem.hpp>
#include <cuopt/linear_programming/optimization_problem_interface.hpp>
#include "grpc_problem_mapper.hpp"
#include "grpc_settings_mapper.hpp"

#include <algorithm>
#include <cstring>

namespace cuopt::linear_programming {

namespace {

// Append chunk requests for a single typed array.
template <typename T>
void chunk_typed_array(std::vector<cuopt::remote::SendArrayChunkRequest>& out,
                       cuopt::remote::ArrayFieldId field_id,
                       const std::vector<T>& data,
                       const std::string& upload_id,
                       int64_t chunk_data_budget)
{
  if (data.empty()) return;

  const int64_t elem_size      = static_cast<int64_t>(sizeof(T));
  const int64_t total_elements = static_cast<int64_t>(data.size());

  int64_t elems_per_chunk = chunk_data_budget / elem_size;
  if (elems_per_chunk <= 0) elems_per_chunk = 1;

  const auto* raw = reinterpret_cast<const uint8_t*>(data.data());

  for (int64_t offset = 0; offset < total_elements; offset += elems_per_chunk) {
    int64_t count       = std::min(elems_per_chunk, total_elements - offset);
    int64_t byte_offset = offset * elem_size;
    int64_t byte_count  = count * elem_size;

    cuopt::remote::SendArrayChunkRequest req;
    req.set_upload_id(upload_id);
    auto* ac = req.mutable_chunk();
    ac->set_field_id(field_id);
    ac->set_element_offset(offset);
    ac->set_total_elements(total_elements);
    ac->set_data(raw + byte_offset, byte_count);
    out.push_back(std::move(req));
  }
}

// Overload for raw uint8_t byte blobs (names, row_types, variable_types).
void chunk_byte_blob(std::vector<cuopt::remote::SendArrayChunkRequest>& out,
                     cuopt::remote::ArrayFieldId field_id,
                     const std::vector<uint8_t>& data,
                     const std::string& upload_id,
                     int64_t chunk_data_budget)
{
  chunk_typed_array(out, field_id, data, upload_id, chunk_data_budget);
}

}  // namespace

template <typename i_t, typename f_t>
cuopt::remote::SubmitJobRequest build_lp_submit_request(
  const cpu_optimization_problem_t<i_t, f_t>& cpu_problem,
  const pdlp_solver_settings_t<i_t, f_t>& settings)
{
  cuopt::remote::SubmitJobRequest submit_request;

  // Get the lp_request from the oneof
  auto* lp_request = submit_request.mutable_lp_request();

  // Set header
  auto* header = lp_request->mutable_header();
  header->set_version(1);
  header->set_problem_type(cuopt::remote::LP);

  // Map problem data to protobuf
  map_problem_to_proto(cpu_problem, lp_request->mutable_problem());

  // Map settings to protobuf
  map_pdlp_settings_to_proto(settings, lp_request->mutable_settings());

  return submit_request;
}

template <typename i_t, typename f_t>
cuopt::remote::SubmitJobRequest build_mip_submit_request(
  const cpu_optimization_problem_t<i_t, f_t>& cpu_problem,
  const mip_solver_settings_t<i_t, f_t>& settings,
  bool enable_incumbents)
{
  cuopt::remote::SubmitJobRequest submit_request;

  // Get the mip_request from the oneof
  auto* mip_request = submit_request.mutable_mip_request();

  // Set header
  auto* header = mip_request->mutable_header();
  header->set_version(1);
  header->set_problem_type(cuopt::remote::MIP);

  // Map problem data to protobuf
  map_problem_to_proto(cpu_problem, mip_request->mutable_problem());

  // Map settings to protobuf
  map_mip_settings_to_proto(settings, mip_request->mutable_settings());

  // Set enable_incumbents flag
  mip_request->set_enable_incumbents(enable_incumbents);

  return submit_request;
}

template <typename i_t, typename f_t>
std::vector<cuopt::remote::SendArrayChunkRequest> build_array_chunk_requests(
  const cpu_optimization_problem_t<i_t, f_t>& problem,
  const std::string& upload_id,
  int64_t chunk_size_bytes)
{
  std::vector<cuopt::remote::SendArrayChunkRequest> requests;

  auto values  = problem.get_constraint_matrix_values_host();
  auto indices = problem.get_constraint_matrix_indices_host();
  auto offsets = problem.get_constraint_matrix_offsets_host();
  auto obj     = problem.get_objective_coefficients_host();
  auto var_lb  = problem.get_variable_lower_bounds_host();
  auto var_ub  = problem.get_variable_upper_bounds_host();
  auto con_lb  = problem.get_constraint_lower_bounds_host();
  auto con_ub  = problem.get_constraint_upper_bounds_host();
  auto b       = problem.get_constraint_bounds_host();

  chunk_typed_array(requests, cuopt::remote::FIELD_A_VALUES, values, upload_id, chunk_size_bytes);
  chunk_typed_array(requests, cuopt::remote::FIELD_A_INDICES, indices, upload_id, chunk_size_bytes);
  chunk_typed_array(requests, cuopt::remote::FIELD_A_OFFSETS, offsets, upload_id, chunk_size_bytes);
  chunk_typed_array(requests, cuopt::remote::FIELD_C, obj, upload_id, chunk_size_bytes);
  chunk_typed_array(
    requests, cuopt::remote::FIELD_VARIABLE_LOWER_BOUNDS, var_lb, upload_id, chunk_size_bytes);
  chunk_typed_array(
    requests, cuopt::remote::FIELD_VARIABLE_UPPER_BOUNDS, var_ub, upload_id, chunk_size_bytes);
  chunk_typed_array(
    requests, cuopt::remote::FIELD_CONSTRAINT_LOWER_BOUNDS, con_lb, upload_id, chunk_size_bytes);
  chunk_typed_array(
    requests, cuopt::remote::FIELD_CONSTRAINT_UPPER_BOUNDS, con_ub, upload_id, chunk_size_bytes);
  chunk_typed_array(requests, cuopt::remote::FIELD_B, b, upload_id, chunk_size_bytes);

  auto row_types = problem.get_row_types_host();
  if (!row_types.empty()) {
    std::vector<uint8_t> rt_bytes(row_types.begin(), row_types.end());
    chunk_byte_blob(
      requests, cuopt::remote::FIELD_ROW_TYPES, rt_bytes, upload_id, chunk_size_bytes);
  }

  auto var_types = problem.get_variable_types_host();
  if (!var_types.empty()) {
    std::vector<uint8_t> vt_bytes;
    vt_bytes.reserve(var_types.size());
    for (const auto& vt : var_types) {
      switch (vt) {
        case var_t::CONTINUOUS: vt_bytes.push_back('C'); break;
        case var_t::INTEGER: vt_bytes.push_back('I'); break;
        default: vt_bytes.push_back('C'); break;
      }
    }
    chunk_byte_blob(
      requests, cuopt::remote::FIELD_VARIABLE_TYPES, vt_bytes, upload_id, chunk_size_bytes);
  }

  if (problem.has_quadratic_objective()) {
    const auto& q_values  = problem.get_quadratic_objective_values();
    const auto& q_indices = problem.get_quadratic_objective_indices();
    const auto& q_offsets = problem.get_quadratic_objective_offsets();
    chunk_typed_array(
      requests, cuopt::remote::FIELD_Q_VALUES, q_values, upload_id, chunk_size_bytes);
    chunk_typed_array(
      requests, cuopt::remote::FIELD_Q_INDICES, q_indices, upload_id, chunk_size_bytes);
    chunk_typed_array(
      requests, cuopt::remote::FIELD_Q_OFFSETS, q_offsets, upload_id, chunk_size_bytes);
  }

  auto names_to_blob = [](const std::vector<std::string>& names) -> std::vector<uint8_t> {
    if (names.empty()) return {};
    size_t total = 0;
    for (const auto& n : names)
      total += n.size() + 1;
    std::vector<uint8_t> blob(total);
    size_t pos = 0;
    for (const auto& n : names) {
      std::memcpy(blob.data() + pos, n.data(), n.size());
      pos += n.size();
      blob[pos++] = '\0';
    }
    return blob;
  };

  auto var_names_blob = names_to_blob(problem.get_variable_names());
  auto row_names_blob = names_to_blob(problem.get_row_names());
  chunk_byte_blob(
    requests, cuopt::remote::FIELD_VARIABLE_NAMES, var_names_blob, upload_id, chunk_size_bytes);
  chunk_byte_blob(
    requests, cuopt::remote::FIELD_ROW_NAMES, row_names_blob, upload_id, chunk_size_bytes);

  return requests;
}

// Explicit template instantiations
#if CUOPT_INSTANTIATE_FLOAT
template cuopt::remote::SubmitJobRequest build_lp_submit_request(
  const cpu_optimization_problem_t<int32_t, float>& cpu_problem,
  const pdlp_solver_settings_t<int32_t, float>& settings);
template cuopt::remote::SubmitJobRequest build_mip_submit_request(
  const cpu_optimization_problem_t<int32_t, float>& cpu_problem,
  const mip_solver_settings_t<int32_t, float>& settings,
  bool enable_incumbents);
template std::vector<cuopt::remote::SendArrayChunkRequest> build_array_chunk_requests(
  const cpu_optimization_problem_t<int32_t, float>& problem,
  const std::string& upload_id,
  int64_t chunk_size_bytes);
#endif

#if CUOPT_INSTANTIATE_DOUBLE
template cuopt::remote::SubmitJobRequest build_lp_submit_request(
  const cpu_optimization_problem_t<int32_t, double>& cpu_problem,
  const pdlp_solver_settings_t<int32_t, double>& settings);
template cuopt::remote::SubmitJobRequest build_mip_submit_request(
  const cpu_optimization_problem_t<int32_t, double>& cpu_problem,
  const mip_solver_settings_t<int32_t, double>& settings,
  bool enable_incumbents);
template std::vector<cuopt::remote::SendArrayChunkRequest> build_array_chunk_requests(
  const cpu_optimization_problem_t<int32_t, double>& problem,
  const std::string& upload_id,
  int64_t chunk_size_bytes);
#endif

}  // namespace cuopt::linear_programming
