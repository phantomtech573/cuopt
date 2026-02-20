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

// Single-Lock Dual Aggregation
//
// For a binary variable x with exactly one "up-lock" (one constraint preventing
// it from increasing), we try to prove an implication y=0 => x=0 via activity
// bounds on the locking row. If additionally the row is non-binding when y=1
// (no capacity competition), we can substitute x = y, eliminating a variable.
//
// Symmetric logic applies for "down-lock" candidates (one constraint preventing
// decrease), proving y=1 => x=1.

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
  // Step 1: Lock Counting — O(nnz)
  //
  // An "up-lock" on column j means a constraint prevents j from increasing:
  //   - a_j > 0 in a <= row, or a_j < 0 in a >= row.
  // "Down-lock" is the reverse. Equality rows lock both directions.
  // We record the row index of the first lock; a second lock invalidates it.
  // =========================================================================

  std::vector<int> up_locks(ncols, 0);
  std::vector<int> down_locks(ncols, 0);
  std::vector<int> up_lock_row(ncols, -1);
  std::vector<int> down_lock_row(ncols, -1);

  for (int row = 0; row < nrows; ++row) {
    if (this->is_time_exceeded(timer, tlim)) return papilo::PresolveStatus::kUnchanged;
    if (row_flags[row].test(papilo::RowFlag::kRedundant)) continue;

    bool lhs_inf = row_flags[row].test(papilo::RowFlag::kLhsInf);
    bool rhs_inf = row_flags[row].test(papilo::RowFlag::kRhsInf);

    auto row_coeff   = constraint_matrix.getRowCoefficients(row);
    const int* cols  = row_coeff.getIndices();
    const f_t* vals  = row_coeff.getValues();
    const int length = row_coeff.getLength();

    // record the index of the locking row.
    // if more than one lock exists, mark the col as excluded from the search
    // (as we cannot safely reduce in this case)
    auto record_up = [&](int col) {
      if (up_locks[col]++ == 0)
        up_lock_row[col] = row;
      else
        up_lock_row[col] = -1;
    };
    auto record_down = [&](int col) {
      if (down_locks[col]++ == 0)
        down_lock_row[col] = row;
      else
        down_lock_row[col] = -1;
    };

    if (!lhs_inf && !rhs_inf) {
      // equality: locks both directions
      for (int j = 0; j < length; ++j) {
        record_up(cols[j]);
        record_down(cols[j]);
      }
    } else if (lhs_inf && !rhs_inf) {
      // <= row: positive coeff locks up, negative locks down
      for (int j = 0; j < length; ++j) {
        if (vals[j] > tol)
          record_up(cols[j]);
        else if (vals[j] < -tol)
          record_down(cols[j]);
      }
    } else if (!lhs_inf && rhs_inf) {
      // >= row: positive coeff locks down, negative locks up
      for (int j = 0; j < length; ++j) {
        if (vals[j] > tol)
          record_down(cols[j]);
        else if (vals[j] < -tol)
          record_up(cols[j]);
      }
    }
  }

  // =========================================================================
  // Step 2: Candidate Identification — O(ncols)
  //
  // Upward candidates: binary, single up-lock, c <= 0 (objective doesn't
  // penalize increase — needed so x pushes against the lock or is indifferent).
  // Downward: symmetric with single down-lock, c >= 0.
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
    if (!is_binary_or_implied(col, col_flags.data(), lower_bounds.data(), upper_bounds.data()))
      continue;
    // Skip singletons (one nonzero): PaPILO's stuffing presolver handles
    // these, and our replaceCol on the same column in the same round creates
    // a deferred transaction that cascades into false infeasibility when
    // stuffing's fix is applied first.
    if (constraint_matrix.getColumnCoefficients(col).getLength() <= 1) continue;

    if (up_locks[col] == 1 && objective[col] <= tol)
      candidates.push_back({col, up_lock_row[col], true});
    else if (down_locks[col] == 1 && objective[col] >= -tol)
      candidates.push_back({col, down_lock_row[col], false});
  }

  if (candidates.empty()) return papilo::PresolveStatus::kUnchanged;

  // =========================================================================
  // Step 3: Mini-Probing — O(L + K) per row
  //
  // For each locking row (L nonzeros, K candidates), we prove implications by
  // fixing two variables and checking if the row's activity bounds are violated:
  //   - Fix candidate x to its "bad" bound (ub for upward, lb for downward)
  //   - Fix master y to its "unfavorable" bound (0 for upward, 1 for downward)
  //   - If the resulting minimum (LEQ) or maximum (GEQ) activity exceeds the
  //     row's bound, the combination is infeasible, proving y_unfav => x_safe.
  //
  // The master y is the binary variable in the row whose coefficient best
  // amplifies the violation. We track the top-2 most extreme coefficients
  // (neg_y for most negative, pos_y for most positive) so that if the
  // candidate itself is the top-1 extremum, we can fall back to top-2.
  // This keeps master selection O(1) per candidate instead of O(L).
  //
  // Candidates are sorted by lock_row so all K candidates sharing a row are
  // processed together in a single O(L) scan, yielding O(L+K) per row group.
  //
  // dense_row_vals[] is an ncols-sized scratch array giving O(1) coefficient
  // lookup by column index; populated and cleaned per row in O(L).
  // =========================================================================

  std::sort(candidates.begin(), candidates.end(), [](const candidate_t& a, const candidate_t& b) {
    return a.lock_row < b.lock_row;
  });

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
  };

  struct substitution_t {
    int cand, master;
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

    // advance row_end to the first candidate with a different lock_row
    auto row_end = cand_it;
    while (row_end != candidates.end() && row_end->lock_row == r)
      ++row_end;

    auto row_coeff   = constraint_matrix.getRowCoefficients(r);
    const int* cols  = row_coeff.getIndices();
    const f_t* vals  = row_coeff.getValues();
    const int length = row_coeff.getLength();

    bool has_lhs = !row_flags[r].test(papilo::RowFlag::kLhsInf);
    bool has_rhs = !row_flags[r].test(papilo::RowFlag::kRhsInf);

    // A_min / A_max: tightest possible activity of the row over all variable bounds
    f_t A_min = 0, A_max = 0;
    bool can_reach_neg_inf = false, can_reach_pos_inf = false;
    top2_t neg_y, pos_y;

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
      if (lower_bounds[col] == upper_bounds[col]) continue;

      neg_y.update(col, coef, true);
      pos_y.update(col, coef, false);
    }

    // LEQ probe needs finite A_min; GEQ probe needs finite A_max
    bool use_leq_check = has_rhs && !can_reach_neg_inf;
    bool use_geq_check = has_lhs && !can_reach_pos_inf;

    // Probe: replace cand and y's min/max contributions with their fixed test
    // values, then check if the resulting activity violates the row bound.
    auto evaluate = [&](f_t cand_coeff, int cand, bool is_upward, int y_col, f_t y_coef) -> bool {
      if (y_col < 0 || y_col == cand) return false;
      f_t orig_cand_min =
        (cand_coeff > 0) ? cand_coeff * lower_bounds[cand] : cand_coeff * upper_bounds[cand];
      f_t orig_cand_max =
        (cand_coeff > 0) ? cand_coeff * upper_bounds[cand] : cand_coeff * lower_bounds[cand];
      f_t cand_test  = cand_coeff * (is_upward ? upper_bounds[cand] : f_t{0});
      f_t orig_y_min = std::min(f_t{0}, y_coef);
      f_t orig_y_max = std::max(f_t{0}, y_coef);
      f_t y_test     = y_coef * (is_upward ? f_t{0} : f_t{1});
      f_t test       = cand_test + y_test;

      if (use_leq_check) {
        // minimum activity with cand at bad bound and y at unfavorable bound
        f_t probed_min = A_min - orig_cand_min - orig_y_min + test;
        if (probed_min > rhs_values[r] + tol) return true;
      }
      if (use_geq_check) {
        f_t probed_max = A_max - orig_cand_max - orig_y_max + test;
        if (probed_max < lhs_values[r] - tol) return true;
      }
      return false;
    };

    // Return the best master from the top-2 tracker, skipping cand itself.
    auto pick_master = [](const top2_t& t, int exclude) -> std::pair<int, f_t> {
      if (t.idx1 != exclude) return {t.idx1, t.val1};
      return {t.idx2, t.val2};
    };

    for (auto ci = cand_it; ci != row_end; ++ci) {
      int cand       = ci->col;
      bool is_upward = ci->is_upward;
      f_t cand_coeff = dense_row_vals[cand];

      bool proven    = false;
      int master_col = -1;

      // For LEQ upward: y=0 zeroes out y's contribution, so the best master
      // is the one with the most negative coefficient (maximizes probed_min).
      // For LEQ downward: y=1 adds y's coefficient, so pick the most positive.
      if (use_leq_check) {
        auto [y, yc] = pick_master(is_upward ? neg_y : pos_y, cand);
        if (evaluate(cand_coeff, cand, is_upward, y, yc)) {
          proven     = true;
          master_col = y;
        }
      }
      if (!proven && use_geq_check) {
        auto [y, yc] = pick_master(is_upward ? pos_y : neg_y, cand);
        if (evaluate(cand_coeff, cand, is_upward, y, yc)) {
          proven     = true;
          master_col = y;
        }
      }

      // The probe proves a one-directional implication (e.g. y=0 => x=0).
      // The substitution x=y also asserts the reverse (y=1 => x=1), which is
      // only safe if forcing x to its bound doesn't starve other variables of
      // capacity in the locking row. Verify the row becomes globally non-binding
      // when y is in its favorable state.
      if (proven) {
        f_t y_coef_val    = dense_row_vals[master_col];
        f_t fav_y_val     = is_upward ? f_t{1} : f_t{0};
        f_t fav_y_contrib = y_coef_val * fav_y_val;

        if (has_rhs) {
          if (can_reach_pos_inf) {
            proven = false;
          } else {
            f_t fav_max = A_max - std::max(f_t{0}, y_coef_val) + fav_y_contrib;
            if (fav_max > rhs_values[r] + tol) proven = false;
          }
        }
        if (proven && has_lhs) {
          if (can_reach_neg_inf) {
            proven = false;
          } else {
            f_t fav_min = A_min - std::min(f_t{0}, y_coef_val) + fav_y_contrib;
            if (fav_min < lhs_values[r] - tol) proven = false;
          }
        }
      }

      if (proven) {
        f_t factor = is_upward ? upper_bounds[cand] : f_t{1};
        substitutions.push_back({cand, master_col, factor});
      }
    }

    for (int j = 0; j < length; ++j)
      dense_row_vals[cols[j]] = f_t{0};
    cand_it = row_end;
  }

  if (substitutions.empty()) return papilo::PresolveStatus::kUnchanged;

  CUOPT_LOG_INFO("Single-lock dual aggregation: %d candidates, %d substitutions",
                 (int)candidates.size(),
                 (int)substitutions.size());

  for (const auto& s : substitutions) {
    reductions.replaceCol(s.cand, s.master, s.factor, f_t{0});
  }

  return papilo::PresolveStatus::kReduced;
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
