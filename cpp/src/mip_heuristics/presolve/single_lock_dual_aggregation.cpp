/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "single_lock_dual_aggregation.hpp"

#include <mip_heuristics/mip_constants.hpp>
#include <utilities/logger.hpp>

#include <algorithm>
#include <vector>

namespace cuopt::linear_programming::detail {

template <typename f_t>
papilo::PresolveStatus SingleLockDualAggregation<f_t>::execute(
  const papilo::Problem<f_t>& problem,
  const papilo::ProblemUpdate<f_t>& problemUpdate,
  const papilo::Num<f_t>& num,
  papilo::Reductions<f_t>& reductions,
  const papilo::Timer& timer,
  int& reason_of_infeasibility)
{
  const auto& constraint_matrix = problem.getConstraintMatrix();
  const auto& lhs_values        = constraint_matrix.getLeftHandSides();
  const auto& rhs_values        = constraint_matrix.getRightHandSides();
  const auto& row_flags         = constraint_matrix.getRowFlags();
  const auto& domains           = problem.getVariableDomains();
  const auto& col_flags         = domains.flags;
  const auto& lower_bounds      = domains.lower_bounds;
  const auto& upper_bounds      = domains.upper_bounds;
  const auto& objective         = problem.getObjective().coefficients;

  const int nrows   = constraint_matrix.getNRows();
  const int ncols   = problem.getNCols();
  const double tlim = problemUpdate.getPresolveOptions().tlim;
  const f_t tol     = num.getFeasTol();

  // =========================================================================
  // Step 1: Lock Counting Sweep — O(nnz)
  //
  // Lock rules:
  //   <= row: positive coeff = up-lock, negative coeff = down-lock
  //   >= row: positive coeff = down-lock, negative coeff = up-lock
  //   =  row: any nonzero coeff = both
  //
  // Branch-on-row-sense is hoisted outside the inner loop.
  // =========================================================================

  std::vector<int> up_locks(ncols, 0);
  std::vector<int> down_locks(ncols, 0);
  std::vector<int> first_up_lock_row(ncols, -1);
  std::vector<int> first_down_lock_row(ncols, -1);

  for (int row = 0; row < nrows; ++row) {
    if (this->is_time_exceeded(timer, tlim)) return papilo::PresolveStatus::kUnchanged;
    if (row_flags[row].test(papilo::RowFlag::kRedundant)) continue;

    bool lhs_inf = row_flags[row].test(papilo::RowFlag::kLhsInf);
    bool rhs_inf = row_flags[row].test(papilo::RowFlag::kRhsInf);

    auto row_coeff   = constraint_matrix.getRowCoefficients(row);
    const int* cols  = row_coeff.getIndices();
    const f_t* vals  = row_coeff.getValues();
    const int length = row_coeff.getLength();

    if (!lhs_inf && !rhs_inf) {
      // Equality or range: every nonzero is both an up-lock and a down-lock
      for (int j = 0; j < length; ++j) {
        int col = cols[j];
        if (up_locks[col]++ == 0) first_up_lock_row[col] = row;
        if (down_locks[col]++ == 0) first_down_lock_row[col] = row;
      }
    } else if (lhs_inf && !rhs_inf) {
      // <= row: positive coeff = up-lock, negative coeff = down-lock
      for (int j = 0; j < length; ++j) {
        int col = cols[j];
        if (vals[j] > tol) {
          if (up_locks[col]++ == 0) first_up_lock_row[col] = row;
        } else if (vals[j] < -tol) {
          if (down_locks[col]++ == 0) first_down_lock_row[col] = row;
        }
      }
    } else if (!lhs_inf && rhs_inf) {
      // >= row: positive coeff = down-lock, negative coeff = up-lock
      for (int j = 0; j < length; ++j) {
        int col = cols[j];
        if (vals[j] > tol) {
          if (down_locks[col]++ == 0) first_down_lock_row[col] = row;
        } else if (vals[j] < -tol) {
          if (up_locks[col]++ == 0) first_up_lock_row[col] = row;
        }
      }
    }
  }

  // =========================================================================
  // Step 2: Candidate Identification — O(ncols)
  //
  // Upward candidate:   lb=0, ub>0 finite, c<=0, up_locks==1.
  //   Solver wants x_j high; one constraint blocks it.
  //   Probe proves y=0 => x_j=0. Substitution: x_j = ub_j * y.
  //
  // Downward candidate: binary, c>=0, down_locks==1.
  //   Solver wants x_j=0; one constraint blocks it.
  //   Probe proves y=1 => x_j=1. Substitution: x_j = y.
  //
  // Both directions share the same candidate structure: (col, locking_row).
  // =========================================================================

  struct candidate_t {
    int col;
    int lock_row;
    bool is_upward;  // true: up-lock, substitute x=U*y; false: down-lock, substitute x=y
  };
  std::vector<candidate_t> candidates;
  candidates.reserve(std::min(ncols, nrows));

  for (int col = 0; col < ncols; ++col) {
    if (this->is_time_exceeded(timer, tlim)) return papilo::PresolveStatus::kUnchanged;
    if (col_flags[col].test(papilo::ColFlag::kFixed, papilo::ColFlag::kSubstituted)) continue;

    // Upward: lb=0, ub>0 finite, c<=0, up_locks==1
    if (up_locks[col] == 1 && objective[col] <= tol &&
        is_valid_detail(col, col_flags.data(), lower_bounds.data(), upper_bounds.data())) {
      candidates.push_back({col, first_up_lock_row[col], true});
    }
    // Downward: binary, c>=0, down_locks==1
    else if (down_locks[col] == 1 && objective[col] >= -tol &&
             is_binary_or_implied(
               col, col_flags.data(), lower_bounds.data(), upper_bounds.data())) {
      candidates.push_back({col, first_down_lock_row[col], false});
    }
  }

  if (candidates.empty()) return papilo::PresolveStatus::kUnchanged;

  // =========================================================================
  // Step 3: Localized Mini-Probing — O(nnz) total
  //
  // Candidates are sorted by locking row so that A_min/A_max are computed
  // once per row, then all candidates in that row are tested in O(1) each.
  //
  // For an upward candidate (positive coeff in <= row, or negative in >= row):
  //   Fix y=0, x_j=ub_j. If min-activity > rhs (<=) or max-activity < lhs
  //   (>=), proven y=0 => x_j=0.
  //
  // For a downward candidate (positive coeff in >= row, or negative in <= row):
  //   Fix y=1, x_j=0. If min-activity > rhs (<=) or max-activity < lhs
  //   (>=), proven y=1 => x_j=1.
  // =========================================================================

  std::sort(candidates.begin(), candidates.end(), [](const candidate_t& a, const candidate_t& b) {
    return a.lock_row < b.lock_row;
  });

  struct substitution_t {
    int cand;
    int master;
    f_t factor;  // ub for upward, 1 for downward
  };
  std::vector<substitution_t> substitutions;

  auto it = candidates.begin();
  while (it != candidates.end()) {
    if (this->is_time_exceeded(timer, tlim)) break;

    int r = it->lock_row;
    if (r < 0) {
      ++it;
      continue;
    }

    // Find all candidates sharing this locking row
    auto row_end =
      std::find_if(it, candidates.end(), [r](const candidate_t& c) { return c.lock_row != r; });

    // Compute A_min and A_max once for this row
    auto row_coeff   = constraint_matrix.getRowCoefficients(r);
    const int* cols  = row_coeff.getIndices();
    const f_t* vals  = row_coeff.getValues();
    const int length = row_coeff.getLength();

    bool has_lhs = !row_flags[r].test(papilo::RowFlag::kLhsInf);
    bool has_rhs = !row_flags[r].test(papilo::RowFlag::kRhsInf);

    f_t A_min = 0, A_max = 0;
    bool can_reach_neg_inf = false, can_reach_pos_inf = false;

    for (int j = 0; j < length; ++j) {
      int col     = cols[j];
      f_t coef    = vals[j];
      bool lb_inf = col_flags[col].test(papilo::ColFlag::kLbInf);
      bool ub_inf = col_flags[col].test(papilo::ColFlag::kUbInf);

      if (coef > 0) {
        if (lb_inf)
          can_reach_neg_inf = true;
        else
          A_min += coef * lower_bounds[col];
        if (ub_inf)
          can_reach_pos_inf = true;
        else
          A_max += coef * upper_bounds[col];
      } else {
        if (ub_inf)
          can_reach_neg_inf = true;
        else
          A_min += coef * upper_bounds[col];
        if (lb_inf)
          can_reach_pos_inf = true;
        else
          A_max += coef * lower_bounds[col];
      }
    }

    // Probe each candidate locked by this row
    for (auto cand_it = it; cand_it != row_end; ++cand_it) {
      int cand       = cand_it->col;
      bool is_upward = cand_it->is_upward;

      // Find cand's coefficient in this row
      f_t cand_coeff = 0;
      for (int j = 0; j < length; ++j) {
        if (cols[j] == cand) {
          cand_coeff = vals[j];
          break;
        }
      }

      // Precompute cand's original contribution to A_min/A_max
      f_t orig_cand_min, orig_cand_max;
      if (cand_coeff > 0) {
        orig_cand_min = cand_coeff * lower_bounds[cand];
        orig_cand_max = cand_coeff * upper_bounds[cand];
      } else {
        orig_cand_min = cand_coeff * upper_bounds[cand];
        orig_cand_max = cand_coeff * lower_bounds[cand];
      }

      // Test value for the candidate under the probe assumption
      f_t cand_test_val     = is_upward ? upper_bounds[cand] : f_t{0};
      f_t cand_test_contrib = cand_coeff * cand_test_val;

      // Try each binary variable in the row as a potential master y
      for (int j = 0; j < length; ++j) {
        int y_col = cols[j];
        if (y_col == cand) continue;
        if (col_flags[y_col].test(papilo::ColFlag::kFixed, papilo::ColFlag::kSubstituted)) continue;
        if (!is_binary_or_implied(
              y_col, col_flags.data(), lower_bounds.data(), upper_bounds.data()))
          continue;

        f_t y_coef     = vals[j];
        f_t orig_y_min = std::min(f_t{0}, y_coef);
        f_t orig_y_max = std::max(f_t{0}, y_coef);

        // Master test value: y=0 for upward, y=1 for downward
        f_t y_test_val     = is_upward ? f_t{0} : f_t{1};
        f_t y_test_contrib = y_coef * y_test_val;
        f_t test_contrib   = cand_test_contrib + y_test_contrib;

        bool proven = false;

        // Check if fixing (cand=test_val, y=y_test_val) violates either bound.
        // <= bound violated if min-possible-activity > rhs
        if (!proven && has_rhs && !can_reach_neg_inf) {
          f_t probed_min = A_min - orig_cand_min - orig_y_min + test_contrib;
          if (probed_min > rhs_values[r] + tol) proven = true;
        }
        // >= bound violated if max-possible-activity < lhs
        if (!proven && has_lhs && !can_reach_pos_inf) {
          f_t probed_max = A_max - orig_cand_max - orig_y_max + test_contrib;
          if (probed_max < lhs_values[r] - tol) proven = true;
        }

        if (proven) {
          f_t factor = is_upward ? upper_bounds[cand] : f_t{1};
          substitutions.push_back({cand, y_col, factor});
          break;
        }
      }
    }

    it = row_end;
  }

  if (substitutions.empty()) return papilo::PresolveStatus::kUnchanged;

  // =========================================================================
  // Step 4: Substitution — replaceCol(j, y, factor, 0)
  // =========================================================================

  CUOPT_LOG_INFO("Single-lock dual aggregation: %d candidates, %d substitutions",
                 (int)candidates.size(),
                 (int)substitutions.size());

  papilo::PresolveStatus status = papilo::PresolveStatus::kUnchanged;

  for (const auto& s : substitutions) {
    if (this->is_time_exceeded(timer, tlim)) break;
    reductions.replaceCol(s.cand, s.master, s.factor, f_t{0});
    status = papilo::PresolveStatus::kReduced;
  }

  return status;
}

#define INSTANTIATE(F_TYPE) template class SingleLockDualAggregation<F_TYPE>;

#if MIP_INSTANTIATE_FLOAT
INSTANTIATE(float)
#endif

#if MIP_INSTANTIATE_DOUBLE
INSTANTIATE(double)
#endif

#undef INSTANTIATE

}  // namespace cuopt::linear_programming::detail
