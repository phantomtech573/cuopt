/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#if !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-overflow"  // ignore boost error for pip wheel build
#endif
#include <papilo/Config.hpp>
#include <papilo/core/PresolveMethod.hpp>
#include <papilo/core/Problem.hpp>
#include <papilo/core/ProblemUpdate.hpp>
#if !defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace cuopt::linear_programming::detail {

template <typename f_t>
class BigMIndicatorAggregation : public papilo::PresolveMethod<f_t> {
 public:
  BigMIndicatorAggregation() : papilo::PresolveMethod<f_t>()
  {
    this->setName("bigm_indicator_aggregation");
    this->setType(papilo::PresolverType::kIntegralCols);
    this->setTiming(papilo::PresolverTiming::kMedium);
  }

  papilo::PresolveStatus execute(const papilo::Problem<f_t>& problem,
                                 const papilo::ProblemUpdate<f_t>& problemUpdate,
                                 const papilo::Num<f_t>& num,
                                 papilo::Reductions<f_t>& reductions,
                                 const papilo::Timer& timer,
                                 int& reason_of_infeasibility) override;

 private:
  bool is_binary_or_implied(int col,
                            const papilo::Flags<papilo::ColFlag>* col_flags,
                            const f_t* lower_bounds,
                            const f_t* upper_bounds) const
  {
    if (!col_flags[col].test(papilo::ColFlag::kIntegral) &&
        !col_flags[col].test(papilo::ColFlag::kImplInt))
      return false;
    if (col_flags[col].test(papilo::ColFlag::kLbInf)) return false;
    if (col_flags[col].test(papilo::ColFlag::kUbInf)) return false;
    return lower_bounds[col] == 0.0 && upper_bounds[col] == 1.0;
  }
};

}  // namespace cuopt::linear_programming::detail
