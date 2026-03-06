/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/linear_programming/constants.h>

#define MIP_INSTANTIATE_FLOAT  CUOPT_INSTANTIATE_FLOAT
#define MIP_INSTANTIATE_DOUBLE CUOPT_INSTANTIATE_DOUBLE

namespace cuopt::linear_programming::detail {

inline constexpr bool is_deterministic_mode(int determinism_mode)
{
  return determinism_mode == CUOPT_MODE_DETERMINISTIC ||
         determinism_mode == CUOPT_MODE_DETERMINISTIC_GPU_HEURISTICS;
}

inline constexpr bool is_gpu_heuristics_deterministic_mode(int determinism_mode)
{
  return determinism_mode == CUOPT_MODE_DETERMINISTIC_GPU_HEURISTICS;
}

}  // namespace cuopt::linear_programming::detail
