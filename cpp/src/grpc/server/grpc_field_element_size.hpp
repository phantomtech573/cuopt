/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

// Codegen target: this file maps ArrayFieldId enum values to their element byte sizes.
// A future version of cpp/codegen/generate_conversions.py can produce this from
// a problem_arrays section in field_registry.yaml.

#pragma once

#ifdef CUOPT_ENABLE_GRPC

#include <cstdint>
#include "cuopt_remote.pb.h"

inline int64_t array_field_element_size(cuopt::remote::ArrayFieldId field_id)
{
  switch (field_id) {
    case cuopt::remote::FIELD_A_INDICES:
    case cuopt::remote::FIELD_A_OFFSETS:
    case cuopt::remote::FIELD_Q_INDICES:
    case cuopt::remote::FIELD_Q_OFFSETS: return 4;
    case cuopt::remote::FIELD_ROW_TYPES:
    case cuopt::remote::FIELD_VARIABLE_TYPES:
    case cuopt::remote::FIELD_VARIABLE_NAMES:
    case cuopt::remote::FIELD_ROW_NAMES: return 1;
    default: return 8;
  }
}

#endif  // CUOPT_ENABLE_GRPC
