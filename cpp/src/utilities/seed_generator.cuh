/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once
#include <raft/random/rng_device.cuh>
#include <utilities/cuda_helpers.cuh>

namespace cuopt {

// TODO: should be thread local?
class seed_generator {
  static int64_t seed_;

 public:
  template <typename seed_t>
  static void set_seed(seed_t seed)
  {
#ifdef BENCHMARK
    seed_ = std::random_device{}();
#else
    seed_ = static_cast<int64_t>(seed);
#endif
  }
  template <typename arg0, typename arg1, typename... args>
  static void set_seed(arg0 seed0, arg1 seed1, args... seeds)
  {
    set_seed(seed1 + ((seed0 + seed1) * (seed0 + seed1 + 1) / 2), seeds...);
  }

#if SEED_GENERATOR_DEBUG
  static int64_t get_seed(const char* caller = __builtin_FUNCTION(),
                          const char* file   = __builtin_FILE(),
                          int line           = __builtin_LINE())
  {
    printf("&&&&&&& SEED CALLED BY %s:%d: %s() ***\n", file, line, caller);
    return seed_++;
  }
#else
  static int64_t get_seed() { return seed_++; }
#endif

  static int64_t peek_seed() { return seed_; }

 public:
  seed_generator(seed_generator const&) = delete;
  void operator=(seed_generator const&) = delete;
};

}  // namespace cuopt
