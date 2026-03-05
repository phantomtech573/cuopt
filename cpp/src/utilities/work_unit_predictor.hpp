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
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuopt/error.hpp>
#include <utilities/version_info.hpp>

namespace cuopt {

// Temporary scaling classes until I figure out better ways to do this
// to account for performance differences between the regression learning machine and the user
// machine. (e.g. integrate memory latency/bandwidth, cache topology, user-provided tuning...)
struct cpu_work_unit_scaler_t {
  cpu_work_unit_scaler_t()
  {
    constexpr double baseline_max_clock = 3800.0;
    double max_clock                    = get_cpu_max_clock_mhz();
    scaling_factor_                     = baseline_max_clock / max_clock;
  }

  double scale_work_units(double work_units) const { return work_units * scaling_factor_; }

 private:
  double scaling_factor_;
};

struct gpu_work_unit_scaler_t {
  double scale_work_units(double work_units) const { return work_units; }
};

template <typename model_t, typename scaler_t>
class work_unit_predictor_t {
 public:
  float predict_scalar(const std::map<std::string, float>& features) const;

 public:
  bool debug{false};

 private:
  mutable std::unordered_map<uint32_t, float> prediction_cache;
  scaler_t scaler_;
};

}  // namespace cuopt
