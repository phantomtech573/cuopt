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

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <string>

#include <mip_heuristics/logger.hpp>
#include <utilities/determinism_log.hpp>
#include <utilities/macros.cuh>

#include "producer_sync.hpp"
#include "timer.hpp"
#include "work_unit_scheduler.hpp"

namespace cuopt {

inline double read_work_unit_scale_env_or_default(const char* env_name, double default_value)
{
  const char* env_value = std::getenv(env_name);
  if (env_value == nullptr || env_value[0] == '\0') { return default_value; }

  errno                     = 0;
  char* end_ptr             = nullptr;
  const double parsed_value = std::strtod(env_value, &end_ptr);
  const bool valid_value    = errno == 0 && end_ptr != env_value && *end_ptr == '\0' &&
                           std::isfinite(parsed_value) && parsed_value > 0.0;
  cuopt_assert(valid_value, "Invalid work-unit scale env var");
  return parsed_value;
}

struct work_limit_context_t {
  double global_work_units_elapsed{0.0};
  double total_sync_time{0.0};  // Total time spent waiting at sync barriers (seconds)
  bool deterministic{false};
  work_unit_scheduler_t* scheduler{nullptr};
  producer_sync_t* producer_sync{nullptr};
  std::string name;
  std::unique_ptr<std::atomic<double>> producer_work_units_elapsed{
    std::make_unique<std::atomic<double>>(0.0)};
  double producer_progress_scale{
    read_work_unit_scale_env_or_default("CUOPT_GPU_HEUR_WORK_UNIT_SCALE", 1.0)};

  work_limit_context_t(const std::string& name) : name(name) {}

  work_limit_context_t(const work_limit_context_t& other)
    : global_work_units_elapsed(other.global_work_units_elapsed),
      total_sync_time(other.total_sync_time),
      deterministic(other.deterministic),
      scheduler(other.scheduler),
      producer_sync(other.producer_sync),
      name(other.name),
      producer_work_units_elapsed(std::make_unique<std::atomic<double>>(
        other.producer_work_units_elapsed->load(std::memory_order_acquire))),
      producer_progress_scale(other.producer_progress_scale)
  {
  }

  work_limit_context_t(work_limit_context_t&& other) noexcept
    : global_work_units_elapsed(other.global_work_units_elapsed),
      total_sync_time(other.total_sync_time),
      deterministic(other.deterministic),
      scheduler(other.scheduler),
      producer_sync(other.producer_sync),
      name(std::move(other.name)),
      producer_work_units_elapsed(std::make_unique<std::atomic<double>>(
        other.producer_work_units_elapsed->load(std::memory_order_acquire))),
      producer_progress_scale(other.producer_progress_scale)
  {
  }

  work_limit_context_t& operator=(const work_limit_context_t& other)
  {
    if (this == &other) { return *this; }
    global_work_units_elapsed   = other.global_work_units_elapsed;
    total_sync_time             = other.total_sync_time;
    deterministic               = other.deterministic;
    scheduler                   = other.scheduler;
    producer_sync               = other.producer_sync;
    name                        = other.name;
    producer_work_units_elapsed = std::make_unique<std::atomic<double>>(
      other.producer_work_units_elapsed->load(std::memory_order_acquire));
    producer_progress_scale = other.producer_progress_scale;
    return *this;
  }

  work_limit_context_t& operator=(work_limit_context_t&& other) noexcept
  {
    if (this == &other) { return *this; }
    global_work_units_elapsed   = other.global_work_units_elapsed;
    total_sync_time             = other.total_sync_time;
    deterministic               = other.deterministic;
    scheduler                   = other.scheduler;
    producer_sync               = other.producer_sync;
    name                        = std::move(other.name);
    producer_work_units_elapsed = std::make_unique<std::atomic<double>>(
      other.producer_work_units_elapsed->load(std::memory_order_acquire));
    producer_progress_scale = other.producer_progress_scale;
    return *this;
  }

  double current_work() const noexcept { return global_work_units_elapsed; }

  double current_producer_work() const noexcept { return current_work() * producer_progress_scale; }

  std::atomic<double>* producer_progress_ptr() noexcept
  {
    return producer_work_units_elapsed.get();
  }

  void attach_producer_sync(producer_sync_t* producer_sync_)
  {
    producer_sync = producer_sync_;
    producer_work_units_elapsed->store(current_producer_work(), std::memory_order_release);
    if (producer_progress_scale != 1.0) {
      CUOPT_DETERMINISM_LOG_DEBUG(
        "[%s] Using producer work-unit scale %f", name.c_str(), producer_progress_scale);
    }
  }

  void detach_producer_sync() noexcept { producer_sync = nullptr; }

  void set_current_work(double total_work, bool notify_producer = true)
  {
    if (!deterministic) return;
    cuopt_assert(total_work + 1e-12 >= global_work_units_elapsed,
                 "Deterministic work progress must be monotonic");
    global_work_units_elapsed = total_work;
    producer_work_units_elapsed->store(current_producer_work(), std::memory_order_release);
    if (notify_producer && producer_sync != nullptr) { producer_sync->notify_progress(); }
  }

  void record_work_sync_on_horizon(double work)
  {
    if (!deterministic) return;
    cuopt_assert(std::isfinite(work), "Recorded work must be finite");
    cuopt_assert(work >= 0.0, "Recorded work must be non-negative");
    const double total_work = global_work_units_elapsed + work;
    set_current_work(total_work, false);
    if (scheduler) { scheduler->on_work_recorded(*this, total_work); }
    if (producer_sync != nullptr) { producer_sync->notify_progress(); }
  }

  void record_work(double work) { record_work_sync_on_horizon(work); }
};

}  // namespace cuopt
