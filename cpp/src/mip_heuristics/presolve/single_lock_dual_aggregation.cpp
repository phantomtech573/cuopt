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
      for (int j = 0; j < length; ++j) {
        int col = cols[j];
        if (up_locks[col]++ == 0) first_up_lock_row[col] = row;
        if (down_locks[col]++ == 0) first_down_lock_row[col] = row;
      }
    } else if (lhs_inf && !rhs_inf) {
      for (int j = 0; j < length; ++j) {
        int col = cols[j];
        if (vals[j] > tol) {
          if (up_locks[col]++ == 0) first_up_lock_row[col] = row;
        } else if (vals[j] < -tol) {
          if (down_locks[col]++ == 0) first_down_lock_row[col] = row;
        }
      }
    } else if (!lhs_inf && rhs_inf) {
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
  // =========================================================================

  struct candidate_t {
    int col;
    int lock_row;
    bool is_upward;
  };
  std::vector<candidate_t> candidates;
  candidates.reserve(std::min(ncols, nrows));

  for (int col = 0; col < ncols; ++col) {
    if (this->is_time_exceeded(timer, tlim)) return papilo::PresolveStatus::kUnchanged;
    if (col_flags[col].test(papilo::ColFlag::kFixed, papilo::ColFlag::kSubstituted)) continue;

    if (up_locks[col] == 1 && objective[col] <= tol &&
        is_valid_detail(col, col_flags.data(), lower_bounds.data(), upper_bounds.data())) {
      candidates.push_back({col, first_up_lock_row[col], true});
    } else if (down_locks[col] == 1 && objective[col] >= -tol &&
               is_binary_or_implied(
                 col, col_flags.data(), lower_bounds.data(), upper_bounds.data())) {
      candidates.push_back({col, first_down_lock_row[col], false});
    }
  }

  if (candidates.empty()) return papilo::PresolveStatus::kUnchanged;

  // =========================================================================
  // Step 3: Localized Mini-Probing with Top-2 Extrema — O(K + L) per row
  //
  // Candidates are sorted by locking row. For each row group, a single O(L)
  // sweep computes A_min, A_max, and the Top-2 binary master candidates.
  // Each candidate is then evaluated in O(1) using the precomputed extrema.
  //
  // Master selection depends on direction and row sense:
  //
  //   | Lock type            | Upward (y=0)       | Downward (y=1)     |
  //   |----------------------|--------------------|--------------------|
  //   | <= row (pos coeff)   | most negative a_y  | —                  |
  //   | >= row (neg coeff)   | most positive a_y  | —                  |
  //   | >= row (pos coeff)   | —                  | most negative a_y  |
  //   | <= row (neg coeff)   | —                  | most positive a_y  |
  //
  // Intuition: we pick the master whose removal from the activity bound
  // creates the largest "swing" toward constraint violation.
  //
  //   <= check: probed_min = A_min - orig_cand_min - orig_y_min + test
  //     The "swing" from y is delta = -min(0, a_y) for y=0, or
  //     a_y - min(0, a_y) for y=1. Maximized by most negative (y=0)
  //     or most positive (y=1) a_y respectively.
  //
  //   >= check: probed_max = A_max - orig_cand_max - orig_y_max + test
  //     The "swing" is -max(0, a_y) for y=0, or a_y - max(0, a_y) for y=1.
  //     Minimized by most positive (y=0) or most negative (y=1) a_y.
  // =========================================================================

  std::sort(candidates.begin(), candidates.end(), [](const candidate_t& a, const candidate_t& b) {
    return a.lock_row < b.lock_row;
  });

  // Top-2 tracker: keeps the two best binary variable indices by coefficient.
  struct top2_t {
    int idx1 = -1, idx2 = -1;
    f_t val1 = 0, val2 = 0;

    void update(int idx, f_t val, bool want_min)
    {
      if (want_min) {
        if (idx1 == -1 || val < val1) {
          idx2 = idx1;
          val2 = val1;
          idx1 = idx;
          val1 = val;
        } else if (idx2 == -1 || val < val2) {
          idx2 = idx;
          val2 = val;
        }
      } else {
        if (idx1 == -1 || val > val1) {
          idx2 = idx1;
          val2 = val1;
          idx1 = idx;
          val1 = val;
        } else if (idx2 == -1 || val > val2) {
          idx2 = idx;
          val2 = val;
        }
      }
    }

    int best(int exclude) const { return (idx1 != exclude) ? idx1 : idx2; }
  };

  struct substitution_t {
    int cand;
    int master;
    f_t factor;
  };
  std::vector<substitution_t> substitutions;
  std::vector<f_t> dense_row_vals(ncols, f_t{0});

  auto cand_it = candidates.begin();
  while (cand_it != candidates.end()) {
    if (this->is_time_exceeded(timer, tlim)) break;

    int r = cand_it->lock_row;
    if (r < 0) {
      ++cand_it;
      continue;
    }

    auto row_end = cand_it;
    while (row_end != candidates.end() && row_end->lock_row == r)
      ++row_end;

    // --- Single O(L) sweep over this row ---
    auto row_coeff   = constraint_matrix.getRowCoefficients(r);
    const int* cols  = row_coeff.getIndices();
    const f_t* vals  = row_coeff.getValues();
    const int length = row_coeff.getLength();

    bool has_lhs = !row_flags[r].test(papilo::RowFlag::kLhsInf);
    bool has_rhs = !row_flags[r].test(papilo::RowFlag::kRhsInf);

    f_t A_min = 0, A_max = 0;
    bool can_reach_neg_inf = false, can_reach_pos_inf = false;

    top2_t neg_y;  // most negative binary coefficients (for <= with y=0, >= with y=1)
    top2_t pos_y;  // most positive binary coefficients (for >= with y=0, <= with y=1)

    for (int j = 0; j < length; ++j) {
      int col     = cols[j];
      f_t coef    = vals[j];
      bool lb_inf = col_flags[col].test(papilo::ColFlag::kLbInf);
      bool ub_inf = col_flags[col].test(papilo::ColFlag::kUbInf);

      dense_row_vals[col] = coef;

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

      if (col_flags[col].test(papilo::ColFlag::kFixed, papilo::ColFlag::kSubstituted)) continue;
      if (!is_binary_or_implied(col, col_flags.data(), lower_bounds.data(), upper_bounds.data()))
        continue;

      neg_y.update(col, coef, true);
      pos_y.update(col, coef, false);
    }

    // Build a coefficient lookup for candidates in this row.
    // This avoids O(L) scans per candidate; instead O(L) once + O(1) per cand.
    // We temporarily store cand->coeff in a local array indexed by position
    // in the candidate group (which is small).
    bool use_leq_check = has_rhs && !can_reach_neg_inf;
    bool use_geq_check = has_lhs && !can_reach_pos_inf;

    // Helper: evaluate a (cand_coeff, y_col, y_coef) triple.
    auto evaluate = [&](f_t cand_coeff, int cand, bool is_upward, int y_col, f_t y_coef) -> bool {
      if (y_col < 0 || y_col == cand) return false;

      f_t orig_cand_min =
        (cand_coeff > 0) ? cand_coeff * lower_bounds[cand] : cand_coeff * upper_bounds[cand];
      f_t orig_cand_max =
        (cand_coeff > 0) ? cand_coeff * upper_bounds[cand] : cand_coeff * lower_bounds[cand];
      f_t cand_test = cand_coeff * (is_upward ? upper_bounds[cand] : f_t{0});

      f_t orig_y_min = std::min(f_t{0}, y_coef);
      f_t orig_y_max = std::max(f_t{0}, y_coef);
      f_t y_test     = y_coef * (is_upward ? f_t{0} : f_t{1});
      f_t test       = cand_test + y_test;

      if (use_leq_check) {
        f_t probed_min = A_min - orig_cand_min - orig_y_min + test;
        if (probed_min > rhs_values[r] + tol) return true;
      }
      if (use_geq_check) {
        f_t probed_max = A_max - orig_cand_max - orig_y_max + test;
        if (probed_max < lhs_values[r] - tol) return true;
      }
      return false;
    };

    // Retrieve the coefficient and value for a top2 entry, handling the
    // fallback when the best entry is the candidate itself.
    auto pick_master = [](const top2_t& t, int exclude) -> std::pair<int, f_t> {
      if (t.idx1 != exclude) return {t.idx1, t.val1};
      return {t.idx2, t.val2};
    };

    // --- O(1) evaluation per candidate ---
    for (auto ci = cand_it; ci != row_end; ++ci) {
      int cand       = ci->col;
      bool is_upward = ci->is_upward;

      f_t cand_coeff = dense_row_vals[cand];

      bool proven    = false;
      int master_col = -1;

      // <= check: upward wants neg_y, downward wants pos_y
      if (!proven && use_leq_check) {
        auto [y, yc] = pick_master(is_upward ? neg_y : pos_y, cand);
        if (evaluate(cand_coeff, cand, is_upward, y, yc)) {
          proven     = true;
          master_col = y;
        }
      }
      // >= check: upward wants pos_y, downward wants neg_y
      if (!proven && use_geq_check) {
        auto [y, yc] = pick_master(is_upward ? pos_y : neg_y, cand);
        if (evaluate(cand_coeff, cand, is_upward, y, yc)) {
          proven     = true;
          master_col = y;
        }
      }

      if (proven) {
        f_t factor = is_upward ? upper_bounds[cand] : f_t{1};
        substitutions.push_back({cand, master_col, factor});
      }
    }

    // O(L) cleanup: reset dense array for the next row
    for (int j = 0; j < length; ++j)
      dense_row_vals[cols[j]] = f_t{0};

    cand_it = row_end;
  }

  if (substitutions.empty()) return papilo::PresolveStatus::kUnchanged;

  // =========================================================================
  // Step 4: Substitution — replaceCol(cand, master, factor, 0)
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
