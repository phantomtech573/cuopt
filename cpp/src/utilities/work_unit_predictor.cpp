/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "work_unit_predictor.hpp"

#include <cassert>
#include <chrono>
#include <cstring>
#include <limits>
#include <mip_heuristics/logger.hpp>
#include <raft/core/nvtx.hpp>

#include "models/fj_predictor/header.h"

namespace cuopt {

template <typename i_t>
static inline uint32_t compute_hash(std::vector<i_t> h_contents)
{
  // FNV-1a hash

  uint32_t hash = 2166136261u;  // FNV-1a 32-bit offset basis
  std::vector<uint8_t> byte_contents(h_contents.size() * sizeof(i_t));
  std::memcpy(byte_contents.data(), h_contents.data(), h_contents.size() * sizeof(i_t));
  for (size_t i = 0; i < byte_contents.size(); ++i) {
    hash ^= byte_contents[i];
    hash *= 16777619u;
  }
  return hash;
}

template <typename model_t, typename scaler_t>
float work_unit_predictor_t<model_t, scaler_t>::predict_scalar(
  const std::map<std::string, float>& features) const
{
  raft::common::nvtx::range range("work_unit_predictor_t::predict_scalar");

  typename model_t::Entry data[model_t::NUM_FEATURES];
  for (int i = 0; i < model_t::NUM_FEATURES; ++i) {
    if (features.find(std::string(model_t::feature_names[i])) == features.end()) {
      data[i].missing = -1;
      CUOPT_LOG_WARN("Feature %s: missing\n", model_t::feature_names[i]);
    } else {
      data[i].fvalue = features.at(std::string(model_t::feature_names[i]));
    }
  }

  std::vector<float> cache_vec;
  cache_vec.reserve(model_t::NUM_FEATURES);
  for (int i = 0; i < model_t::NUM_FEATURES; ++i) {
    cache_vec.push_back(data[i].missing != -1 ? data[i].fvalue
                                              : std::numeric_limits<float>::quiet_NaN());
  }
  uint32_t key = compute_hash(cache_vec);

  auto cached_it = prediction_cache.find(key);
  if (cached_it != prediction_cache.end()) { return cached_it->second; }

  double result = 0.0;
  auto start    = std::chrono::high_resolution_clock::now();
  model_t::predict(data, 0, &result);
  auto end                                          = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  if (debug) CUOPT_LOG_DEBUG("Prediction time: %f ms", elapsed.count());

  float scaled_result   = scaler_.scale_work_units(result);
  prediction_cache[key] = scaled_result;
  if (debug) CUOPT_LOG_DEBUG("Result: %f (scaled: %f)", result, scaled_result);

  return scaled_result;
}

template class work_unit_predictor_t<fj_predictor, gpu_work_unit_scaler_t>;

}  // namespace cuopt
