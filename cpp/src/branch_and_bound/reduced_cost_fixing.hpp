/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/presolve.hpp>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
i_t find_reduced_cost_fixings(const lp_problem_t<i_t, f_t>& lp,
                              const std::vector<f_t>& reduced_costs,
                              const std::vector<variable_type_t>& var_types,
                              f_t obj,
                              f_t upper_bound,
                              std::vector<f_t>& lower_bounds,
                              std::vector<f_t>& upper_bounds,
                              std::vector<bool>& bounds_changed,
                              const simplex_solver_settings_t<i_t, f_t>& settings)
{
  const f_t threshold   = 100.0 * settings.integer_tol;
  const f_t weaken      = settings.integer_tol;
  const f_t fixed_tol   = settings.fixed_tol;
  i_t num_improved      = 0;
  i_t num_fixed         = 0;
  i_t num_cols_to_check = reduced_costs.size();  // Reduced costs will be smaller than the original
                                                 // problem because we have added slacks for cuts

  bounds_changed.assign(lp.num_cols, false);

  for (i_t j = 0; j < num_cols_to_check; j++) {
    if (std::isfinite(reduced_costs[j]) && std::abs(reduced_costs[j]) > threshold) {
      const f_t lower_j            = lp.lower[j];
      const f_t upper_j            = lp.upper[j];
      const f_t abs_gap            = upper_bound - obj;
      f_t reduced_cost_upper_bound = upper_j;
      f_t reduced_cost_lower_bound = lower_j;
      if (lower_j > -inf && reduced_costs[j] > 0) {
        const f_t new_upper_bound = lower_j + abs_gap / reduced_costs[j];
        reduced_cost_upper_bound  = var_types[j] == variable_type_t::INTEGER
                                      ? std::floor(new_upper_bound + weaken)
                                      : new_upper_bound;
        if (reduced_cost_upper_bound < upper_j && var_types[j] == variable_type_t::INTEGER) {
          ++num_improved;
          upper_bounds[j]   = reduced_cost_upper_bound;
          bounds_changed[j] = true;
        }
      }
      if (upper_j < inf && reduced_costs[j] < 0) {
        const f_t new_lower_bound = upper_j + abs_gap / reduced_costs[j];
        reduced_cost_lower_bound  = var_types[j] == variable_type_t::INTEGER
                                      ? std::ceil(new_lower_bound - weaken)
                                      : new_lower_bound;
        if (reduced_cost_lower_bound > lower_j && var_types[j] == variable_type_t::INTEGER) {
          ++num_improved;
          lower_bounds[j]   = reduced_cost_lower_bound;
          bounds_changed[j] = true;
        }
      }
      if (var_types[j] == variable_type_t::INTEGER &&
          reduced_cost_upper_bound <= reduced_cost_lower_bound + fixed_tol) {
        ++num_fixed;
      }
    }
  }

  if (num_fixed > 0 || num_improved > 0) {
    settings.log.debug(
      "Reduced costs: Found %d improved bounds and %d fixed variables\n", num_improved, num_fixed);
  }
  return num_fixed;
}

}  // namespace cuopt::linear_programming::dual_simplex