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
#include <array>
#include <vector>

namespace cuopt::linear_programming::detail {

static constexpr int MAX_LOCKS = 3;

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
  // Records up to MAX_LOCKS lock rows per variable per direction.
  // =========================================================================

  std::vector<int> up_locks(ncols, 0);
  std::vector<int> down_locks(ncols, 0);
  // up_lock_rows[col] stores up to MAX_LOCKS row indices
  std::vector<std::array<int, MAX_LOCKS>> up_lock_rows(ncols);
  std::vector<std::array<int, MAX_LOCKS>> down_lock_rows(ncols);
  for (int i = 0; i < ncols; ++i) {
    up_lock_rows[i].fill(-1);
    down_lock_rows[i].fill(-1);
  }

  auto record_up_lock = [&](int col, int row) {
    int k = up_locks[col]++;
    if (k < MAX_LOCKS) up_lock_rows[col][k] = row;
  };
  auto record_down_lock = [&](int col, int row) {
    int k = down_locks[col]++;
    if (k < MAX_LOCKS) down_lock_rows[col][k] = row;
  };

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
        record_up_lock(cols[j], row);
        record_down_lock(cols[j], row);
      }
    } else if (lhs_inf && !rhs_inf) {
      for (int j = 0; j < length; ++j) {
        if (vals[j] > tol)
          record_up_lock(cols[j], row);
        else if (vals[j] < -tol)
          record_down_lock(cols[j], row);
      }
    } else if (!lhs_inf && rhs_inf) {
      for (int j = 0; j < length; ++j) {
        if (vals[j] > tol)
          record_down_lock(cols[j], row);
        else if (vals[j] < -tol)
          record_up_lock(cols[j], row);
      }
    }
  }

  // =========================================================================
  // Step 1.5: Shadow Lock Elimination — O(nnz) total
  //
  // For variables with 2 or 3 up/down locks, check if the extra locks are
  // "shadow" locks: the constraint is satisfied for all values of x_j
  // given the current bounds of the other variables.
  //
  // A row is a shadow lock for column j if the row's max activity WITHOUT
  // j's contribution already satisfies the bound. For a <= row with
  // positive a_j: if A_max_without_j <= rhs, the row can never be violated
  // by j, so it's not a real lock.
  // =========================================================================

  // effective_up_lock_row[col]: the single "real" up-lock row after shadows removed
  std::vector<int> effective_up_lock_row(ncols, -1);
  std::vector<int> effective_up_locks(ncols, 0);
  std::vector<int> effective_down_lock_row(ncols, -1);
  std::vector<int> effective_down_locks(ncols, 0);

  auto is_shadow_lock = [&](int col, int row) -> bool {
    if (row < 0) return true;
    if (row_flags[row].test(papilo::RowFlag::kRedundant)) return true;

    auto rc          = constraint_matrix.getRowCoefficients(row);
    const int* rcols = rc.getIndices();
    const f_t* rvals = rc.getValues();
    const int rlen   = rc.getLength();

    bool has_rhs = !row_flags[row].test(papilo::RowFlag::kRhsInf);
    bool has_lhs = !row_flags[row].test(papilo::RowFlag::kLhsInf);

    // Compute max activity of the row WITHOUT col's contribution
    f_t max_without = 0;
    f_t min_without = 0;
    bool max_inf = false, min_inf = false;

    for (int j = 0; j < rlen; ++j) {
      if (rcols[j] == col) continue;
      f_t c       = rvals[j];
      bool lb_inf = col_flags[rcols[j]].test(papilo::ColFlag::kLbInf);
      bool ub_inf = col_flags[rcols[j]].test(papilo::ColFlag::kUbInf);

      if (c > 0) {
        if (ub_inf)
          max_inf = true;
        else
          max_without += c * upper_bounds[rcols[j]];
        if (lb_inf)
          min_inf = true;
        else
          min_without += c * lower_bounds[rcols[j]];
      } else {
        if (lb_inf)
          max_inf = true;
        else
          max_without += c * lower_bounds[rcols[j]];
        if (ub_inf)
          min_inf = true;
        else
          min_without += c * upper_bounds[rcols[j]];
      }
    }

    // A row is a shadow lock for col iff col CANNOT cause a violation:
    // even with col at its worst bound AND all other variables at THEIR
    // worst bounds (maximizing the chance of violation), the constraint
    // is still satisfied.
    //
    // For <= row with positive col coeff (up-lock):
    //   shadow iff max_without + a_col * ub_col <= rhs + tol
    //   (even worst-case other vars + col at max doesn't exceed rhs)
    //
    // For >= row with negative col coeff (up-lock):
    //   shadow iff min_without + a_col * ub_col >= lhs - tol
    //   (even worst-case other vars + col at max doesn't go below lhs)

    f_t col_coeff = 0;
    for (int j = 0; j < rlen; ++j) {
      if (rcols[j] == col) {
        col_coeff = rvals[j];
        break;
      }
    }

    // <= row, positive coeff: shadow if max(others) + a*ub <= rhs
    if (has_rhs && col_coeff > tol && !max_inf) {
      if (max_without + col_coeff * upper_bounds[col] <= rhs_values[row] + tol) return true;
    }
    // >= row, negative coeff: shadow if min(others) + a*ub >= lhs
    if (has_lhs && col_coeff < -tol && !min_inf) {
      if (min_without + col_coeff * upper_bounds[col] >= lhs_values[row] - tol) return true;
    }

    return false;
  };

  for (int col = 0; col < ncols; ++col) {
    if (up_locks[col] >= 1 && up_locks[col] <= MAX_LOCKS) {
      int real_count = 0;
      int real_row   = -1;
      for (int k = 0; k < up_locks[col]; ++k) {
        if (!is_shadow_lock(col, up_lock_rows[col][k])) {
          ++real_count;
          real_row = up_lock_rows[col][k];
        }
      }
      effective_up_locks[col]    = real_count;
      effective_up_lock_row[col] = (real_count == 1) ? real_row : -1;
    } else {
      effective_up_locks[col]    = up_locks[col];
      effective_up_lock_row[col] = -1;
    }

    if (down_locks[col] >= 1 && down_locks[col] <= MAX_LOCKS) {
      int real_count = 0;
      int real_row   = -1;
      for (int k = 0; k < down_locks[col]; ++k) {
        if (!is_shadow_lock(col, down_lock_rows[col][k])) {
          ++real_count;
          real_row = down_lock_rows[col][k];
        }
      }
      effective_down_locks[col]    = real_count;
      effective_down_lock_row[col] = (real_count == 1) ? real_row : -1;
    } else {
      effective_down_locks[col]    = down_locks[col];
      effective_down_lock_row[col] = -1;
    }
  }

  // =========================================================================
  // Step 2: Candidate Identification — O(ncols)
  //
  // Uses effective locks (after shadow elimination).
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

    if (effective_up_locks[col] == 1 && objective[col] <= tol &&
        is_valid_detail(col, col_flags.data(), lower_bounds.data(), upper_bounds.data())) {
      candidates.push_back({col, effective_up_lock_row[col], true});
    } else if (effective_down_locks[col] == 1 && objective[col] >= -tol &&
               is_binary_or_implied(
                 col, col_flags.data(), lower_bounds.data(), upper_bounds.data())) {
      candidates.push_back({col, effective_down_lock_row[col], false});
    }
  }

  if (candidates.empty()) return papilo::PresolveStatus::kUnchanged;

  // =========================================================================
  // Step 3: Localized Mini-Probing with Top-2 Extrema — O(K + L) per row
  //
  // Extension 1 (Continuous Masters): For 2-variable rows, extract the VUB
  // algebraically: x_j = (-a_y/a_j) * y + rhs/a_j, where y can be any
  // variable (binary or continuous). No probing needed.
  //
  // For rows with 3+ variables: standard Top-2 probing with binary masters.
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
    int cand;
    int master;
    f_t factor;
    f_t offset;
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

    auto row_coeff   = constraint_matrix.getRowCoefficients(r);
    const int* cols  = row_coeff.getIndices();
    const f_t* vals  = row_coeff.getValues();
    const int length = row_coeff.getLength();

    bool has_lhs = !row_flags[r].test(papilo::RowFlag::kLhsInf);
    bool has_rhs = !row_flags[r].test(papilo::RowFlag::kRhsInf);

    // --- Extension 1: 2-variable row algebraic VUB extraction ---
    // For a 2-variable <= row: a_j * x_j + a_y * y <= b
    // with a_j > 0 (up-lock for x_j): x_j <= (-a_y/a_j) * y + b/a_j
    // For a 2-variable >= row: a_j * x_j + a_y * y >= b
    // with a_j < 0 (up-lock for x_j): x_j <= (-a_y/a_j) * y + b/a_j  (dividing by negative flips)
    if (length == 2) {
      for (auto ci = cand_it; ci != row_end; ++ci) {
        int cand = ci->col;
        if (!ci->is_upward) continue;  // algebraic VUB only for upward

        int y_col     = -1;
        f_t cand_coef = 0, y_coef = 0;
        for (int j = 0; j < 2; ++j) {
          if (cols[j] == cand)
            cand_coef = vals[j];
          else {
            y_col  = cols[j];
            y_coef = vals[j];
          }
        }

        if (y_col < 0) continue;
        if (col_flags[y_col].test(papilo::ColFlag::kFixed, papilo::ColFlag::kSubstituted)) continue;
        if (std::abs(cand_coef) < tol) continue;

        f_t factor, offset;
        if (has_rhs && !has_lhs && cand_coef > tol) {
          // <= row: x_j <= (-y_coef / cand_coef) * y + rhs / cand_coef
          factor = -y_coef / cand_coef;
          offset = rhs_values[r] / cand_coef;
        } else if (has_lhs && !has_rhs && cand_coef < -tol) {
          // >= row: divide by negative cand_coef flips the inequality
          factor = -y_coef / cand_coef;
          offset = lhs_values[r] / cand_coef;
        } else {
          continue;
        }

        // Verify the substitution respects x_j's bounds for all feasible y
        f_t y_lb = col_flags[y_col].test(papilo::ColFlag::kLbInf) ? -1e20 : lower_bounds[y_col];
        f_t y_ub = col_flags[y_col].test(papilo::ColFlag::kUbInf) ? 1e20 : upper_bounds[y_col];
        f_t xj_at_ylb = factor * y_lb + offset;
        f_t xj_at_yub = factor * y_ub + offset;
        f_t xj_min    = std::min(xj_at_ylb, xj_at_yub);
        f_t xj_max    = std::max(xj_at_ylb, xj_at_yub);

        if (xj_min < lower_bounds[cand] - tol) continue;
        if (xj_max > upper_bounds[cand] + tol) continue;

        substitutions.push_back({cand, y_col, factor, offset});
      }

      // Clean up and advance (no dense array used for 2-var rows)
      cand_it = row_end;
      continue;
    }

    // --- Standard probing for rows with 3+ variables ---
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

      neg_y.update(col, coef, true);
      pos_y.update(col, coef, false);
    }

    bool use_leq_check = has_rhs && !can_reach_neg_inf;
    bool use_geq_check = has_lhs && !can_reach_pos_inf;

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
        f_t probed_min = A_min - orig_cand_min - orig_y_min + test;
        if (probed_min > rhs_values[r] + tol) return true;
      }
      if (use_geq_check) {
        f_t probed_max = A_max - orig_cand_max - orig_y_max + test;
        if (probed_max < lhs_values[r] - tol) return true;
      }
      return false;
    };

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

      if (!proven && use_leq_check) {
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

      if (proven) {
        f_t factor = is_upward ? upper_bounds[cand] : f_t{1};
        substitutions.push_back({cand, master_col, factor, f_t{0}});
      }
    }

    for (int j = 0; j < length; ++j)
      dense_row_vals[cols[j]] = f_t{0};
    cand_it = row_end;
  }

  if (substitutions.empty()) return papilo::PresolveStatus::kUnchanged;

  // =========================================================================
  // Step 4: Substitution
  // =========================================================================

  CUOPT_LOG_INFO("Single-lock dual aggregation: %d candidates, %d substitutions",
                 (int)candidates.size(),
                 (int)substitutions.size());

  papilo::PresolveStatus status = papilo::PresolveStatus::kUnchanged;

  for (const auto& s : substitutions) {
    if (this->is_time_exceeded(timer, tlim)) break;
    reductions.replaceCol(s.cand, s.master, s.factor, s.offset);
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
