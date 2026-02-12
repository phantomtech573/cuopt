/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "bigm_indicator_aggregation.hpp"

#include <mip_heuristics/mip_constants.hpp>
#include <utilities/logger.hpp>

#include <vector>

namespace cuopt::linear_programming::detail {

template <typename f_t>
papilo::PresolveStatus BigMIndicatorAggregation<f_t>::execute(
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

  const int nrows = constraint_matrix.getNRows();
  const int ncols = problem.getNCols();

  // Phase 1: Detect big-M indicator constraints.
  //
  // Pattern: -M * y + x_1 + x_2 + ... + x_K <= 0   (M >= K)
  //   - Exactly one variable y with negative coefficient -M
  //   - All other K variables with coefficient exactly +1
  //   - M >= K (big-M at least as large as detail count; need not be tight)
  //   - All variables binary or implied binary
  //   - Row is <= 0 (lhs = -inf, rhs = 0)

  struct bigm_info_t {
    int row;
    int master_col;
    std::vector<int> detail_cols;
  };
  std::vector<bigm_info_t> bigm_rows;

  // detail_col -> (bigm_index, master_col); only keep details in exactly 1 big-M
  std::vector<int> detail_bigm_count(ncols, 0);
  std::vector<int> detail_master(ncols, -1);

  for (int row = 0; row < nrows; ++row) {
    if (row_flags[row].test(papilo::RowFlag::kRedundant)) continue;

    // Must be <= 0: rhs finite and == 0, lhs infinite
    if (!row_flags[row].test(papilo::RowFlag::kLhsInf)) continue;
    if (row_flags[row].test(papilo::RowFlag::kRhsInf)) continue;
    if (!num.isZero(rhs_values[row])) continue;

    auto row_coeff   = constraint_matrix.getRowCoefficients(row);
    const int* cols  = row_coeff.getIndices();
    const f_t* vals  = row_coeff.getValues();
    const int length = row_coeff.getLength();

    if (length < 2) continue;

    int master_col      = -1;
    f_t master_coeff    = 0;
    int n_positive_ones = 0;
    bool valid          = true;

    for (int j = 0; j < length; ++j) {
      int col  = cols[j];
      f_t coef = vals[j];

      if (col_flags[col].test(papilo::ColFlag::kFixed, papilo::ColFlag::kSubstituted)) {
        valid = false;
        break;
      }
      if (!is_binary_or_implied(col, col_flags.data(), lower_bounds.data(), upper_bounds.data())) {
        valid = false;
        break;
      }

      if (coef < -0.5) {
        if (master_col != -1) {
          valid = false;
          break;
        }
        master_col   = col;
        master_coeff = coef;
      } else if (num.isEq(coef, f_t{1})) {
        ++n_positive_ones;
      } else {
        valid = false;
        break;
      }
    }

    if (!valid || master_col == -1) continue;
    // M >= K: the big-M coefficient must be at least as large as the detail count
    // This ensures that when y = 1, all K details can simultaneously be 1.
    if (master_coeff > -(f_t)n_positive_ones + num.getFeasTol()) continue;

    bigm_info_t info;
    info.row        = row;
    info.master_col = master_col;
    info.detail_cols.reserve(n_positive_ones);
    for (int j = 0; j < length; ++j) {
      if (cols[j] != master_col) info.detail_cols.push_back(cols[j]);
    }
    bigm_rows.push_back(std::move(info));
  }

  if (bigm_rows.empty()) return papilo::PresolveStatus::kUnchanged;

  // Build detail -> master mapping (only for details in exactly 1 big-M)
  int detail_set_size = 0;
  for (size_t bi = 0; bi < bigm_rows.size(); ++bi) {
    for (int d : bigm_rows[bi].detail_cols) {
      detail_bigm_count[d]++;
      detail_master[d] = bigm_rows[bi].master_col;
      ++detail_set_size;
    }
  }

  // Phase 2: Safety check per detail.
  //
  // The substitution x_i = y is valid when, for any optimal (x*, y*),
  // setting x_i = y_{m(i)} doesn't worsen the objective or violate any
  // constraint. This holds when:
  //   - c(x_i) <= 0 (non-positive cost: activating the detail is free or beneficial)
  //   - Every non-big-M constraint containing x_i satisfies one of:
  //     (a) >= constraint with coeff(x_i) > 0: substitution increases LHS, helping >=
  //     (b) <= constraint with coeff(x_i) < 0: substitution decreases LHS, helping <=
  //   - Unsafe cases (substitution makes constraint harder to satisfy):
  //     (c) >= constraint with coeff(x_i) < 0
  //     (d) <= constraint with coeff(x_i) > 0
  //     (e) equality constraint (either direction could violate)
  //     (f) range constraint with positive coeff (tightens the <= side)

  std::vector<bool> bigm_row_flag(nrows, false);
  for (const auto& bm : bigm_rows) {
    bigm_row_flag[bm.row] = true;
  }

  std::vector<bool> detail_safe(ncols, false);
  int n_safe = 0;

  for (int col = 0; col < ncols; ++col) {
    if (detail_bigm_count[col] != 1) continue;

    auto col_coeff    = constraint_matrix.getColumnCoefficients(col);
    const int* rows   = col_coeff.getIndices();
    const f_t* cvals  = col_coeff.getValues();
    const int col_len = col_coeff.getLength();

    // Detail must have non-positive objective cost (zero or beneficial)
    if (objective[col] > num.getFeasTol()) continue;

    bool safe = true;
    for (int k = 0; k < col_len; ++k) {
      int r  = rows[k];
      f_t cv = cvals[k];

      if (row_flags[r].test(papilo::RowFlag::kRedundant)) continue;

      if (bigm_row_flag[r]) continue;

      bool lhs_inf = row_flags[r].test(papilo::RowFlag::kLhsInf);
      bool rhs_inf = row_flags[r].test(papilo::RowFlag::kRhsInf);

      // >= constraint (lhs finite, rhs infinite) with positive coeff: safe
      if (!lhs_inf && rhs_inf && cv > 0) continue;

      // <= constraint (lhs infinite, rhs finite) with negative coeff: safe
      if (lhs_inf && !rhs_inf && cv < 0) continue;

      // Anything else (equality, range, or wrong-sign coefficient) is unsafe
      safe = false;
      break;
    }

    if (safe) {
      detail_safe[col] = true;
      ++n_safe;
    }
  }

  if (n_safe == 0) return papilo::PresolveStatus::kUnchanged;

  CUOPT_LOG_INFO(
    "BigM indicator aggregation: %d big-M constraints detected, %d detail variables "
    "substituted out of %d candidates",
    (int)bigm_rows.size(),
    n_safe,
    (int)detail_set_size);

  // Phase 3: Substitute safe details via replaceCol.
  //
  // replaceCol(detail, master, 1.0, 0.0) declares detail = master.
  // PaPILO substitutes the detail out of all constraints and stores
  // the relationship for postsolve.

  papilo::PresolveStatus status = papilo::PresolveStatus::kUnchanged;

  for (int col = 0; col < ncols; ++col) {
    if (!detail_safe[col]) continue;
    int master = detail_master[col];
    reductions.replaceCol(col, master, f_t{1}, f_t{0});
    status = papilo::PresolveStatus::kReduced;
  }

  // Big-M rows are NOT explicitly marked redundant here. PaPILO postpones
  // replaceCol substitutions to apply them last, so any markRowRedundant
  // would execute before the substitutions and prematurely remove the
  // big-M constraints. After all substitutions are applied, the big-M rows
  // become tautologies (0 <= 0) and PaPILO's own redundancy detection
  // will remove them in a subsequent presolve round.

  return status;
}

#define INSTANTIATE(F_TYPE) template class BigMIndicatorAggregation<F_TYPE>;

#if MIP_INSTANTIATE_FLOAT
INSTANTIATE(float)
#endif

#if MIP_INSTANTIATE_DOUBLE
INSTANTIATE(double)
#endif

#undef INSTANTIATE

}  // namespace cuopt::linear_programming::detail
