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
#include <string>

#include <mip_heuristics/logger.hpp>

#include "timer.hpp"
#include "work_limit_context.hpp"

namespace cuopt {

// In determinism mode, relies on a work limit accumulator; otherwise rely on a timer
// in non-determinism mode: 1s = 1wu
// In deterministic mode, all timers share a global work units counter (via work_limit_context_t),
// and each timer records the snapshot at construction time to determine when its own limit is
// reached.
class work_limit_timer_t {
 public:
  work_limit_timer_t()
    : deterministic(false), work_limit(0), timer(0), work_context(nullptr), work_units_at_start(0)
  {
  }

  // Constructor taking work limit context reference (for deterministic mode)
  work_limit_timer_t(work_limit_context_t& context, double work_limit_)
    : deterministic(context.deterministic),
      work_limit(work_limit_),
      timer(work_limit_),
      work_context(&context),
      work_units_at_start(context.deterministic ? context.global_work_units_elapsed : 0)
  {
  }

  // Compatibility constructor for non-deterministic contexts.
  explicit work_limit_timer_t(double time_limit_)
    : deterministic(false),
      work_limit(time_limit_),
      timer(time_limit_),
      work_context(nullptr),
      work_units_at_start(0)
  {
  }

  bool check_limit(const char* caller = __builtin_FUNCTION(),
                   const char* file   = __builtin_FILE(),
                   int line           = __builtin_LINE()) const noexcept
  {
    // TODO check against global timer

    if (deterministic) {
      if (!work_context) { return false; }
      // Check if global work has exceeded our budget (snapshot + limit)
      double elapsed_since_start = work_context->global_work_units_elapsed - work_units_at_start;
      bool finished_now          = elapsed_since_start >= work_limit;
      if (finished_now && !finished) {
        finished                   = true;
        double actual_elapsed_time = timer.elapsed_time();
        // 10% timing error
        if (work_limit > 0 && abs(actual_elapsed_time - work_limit) / work_limit > 0.10) {
          CUOPT_LOG_ERROR(
            "%s:%d: %s(): Work limit timer finished with a large discrepancy: %fs for %fwu "
            "(global: %f, start: %f)",
            file,
            line,
            caller,
            actual_elapsed_time,
            work_limit,
            work_context->global_work_units_elapsed,
            work_units_at_start);
        }
      }
      return finished;
    } else {
      return timer.check_time_limit();
    }
  }

  // in determinism mode, add the work units to the global work limit accumulator
  void record_work(double work_units,
                   const char* caller = __builtin_FUNCTION(),
                   const char* file   = __builtin_FILE(),
                   int line           = __builtin_LINE())
  {
    if (deterministic && work_context) {
      CUOPT_LOG_DEBUG("%s:%d: %s(): Recorded %f work units in %fs, total %f",
                      file,
                      line,
                      caller,
                      work_units,
                      timer.elapsed_time(),
                      work_context->global_work_units_elapsed);
      work_context->record_work_sync_on_horizon(work_units);
    }
  }

  double remaining_units() const noexcept
  {
    if (deterministic) {
      if (!work_context) { return work_limit; }
      double elapsed_since_start = work_context->global_work_units_elapsed - work_units_at_start;
      return work_limit - elapsed_since_start;
    } else {
      return timer.remaining_time();
    }
  }

  double remaining_time() const noexcept { return remaining_units(); }

  double elapsed_time() const noexcept
  {
    if (deterministic) {
      return work_context->global_work_units_elapsed - work_units_at_start;
    } else {
      return timer.elapsed_time();
    }
  }

  bool check_time_limit(const char* caller = __builtin_FUNCTION(),
                        const char* file   = __builtin_FILE(),
                        int line           = __builtin_LINE()) const noexcept
  {
    return check_limit(caller, file, line);
  }

  bool check_half_time() const noexcept
  {
    if (deterministic) {
      if (!work_context) { return false; }
      double elapsed_since_start = work_context->global_work_units_elapsed - work_units_at_start;
      return elapsed_since_start >= work_limit / 2;
    } else {
      return timer.check_half_time();
    }
  }

  double clamp_remaining_time(double desired_time) const noexcept
  {
    return std::min<double>(desired_time, remaining_time());
  }

  double get_time_limit() const noexcept
  {
    if (deterministic) {
      return work_limit;
    } else {
      return timer.get_time_limit();
    }
  }

  void print_debug(std::string msg) const
  {
    if (deterministic) {
      printf("%s work_limit: %f remaining_work: %f elapsed_work: %f \n",
             msg.c_str(),
             work_limit,
             remaining_time(),
             elapsed_time());
    } else {
      timer.print_debug(msg);
    }
  }

  timer_t timer;
  double work_limit{};
  mutable bool finished{false};
  bool deterministic{false};
  // Pointer to work limit context (shared across all timers in deterministic mode)
  work_limit_context_t* work_context{nullptr};
  // Snapshot of global work units when this timer was created
  double work_units_at_start{0};
};
}  // namespace cuopt
