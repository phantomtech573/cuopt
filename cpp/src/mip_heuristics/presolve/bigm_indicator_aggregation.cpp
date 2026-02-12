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

  const int nrows   = constraint_matrix.getNRows();
  const int ncols   = problem.getNCols();
  const double tlim = problemUpdate.getPresolveOptions().tlim;

  // Phase 1: Detect big-M indicator constraints and build detail-master mapping.
  //
  // Standard big-M (y=0 => x_i=0):
  //   -M*y + a_1*x_1 + ... + a_K*x_K <= 0   with M >= sum(a_i), a_i > 0
  //   Row form: lhs=-inf, rhs=0, one negative coeff (master), rest positive (details).
  //
  // Also detected in normalized form (Case B):
  //   a_1*x_1 + ... + a_K*x_K - M*y >= 0   (one positive master, rest negative details)
  //   Equivalent to: sum(|a_i|*x_i) <= M*y.
  //
  // The implication y=0 => x_i=0 holds for any positive a_i, since
  // sum(a_i * x_i) <= 0 with a_i > 0 and x_i >= 0 forces all x_i = 0.
  //
  // All variables must be binary or implied binary.

  std::vector<int> detail_bigm_count(ncols, 0);
  std::vector<int> detail_master(ncols, -1);
  std::vector<bool> bigm_row_flag(nrows, false);
  int n_bigm_rows     = 0;
  int detail_set_size = 0;

  for (int row = 0; row < nrows; ++row) {
    if (this->is_time_exceeded(timer, tlim)) return papilo::PresolveStatus::kUnchanged;
    if (row_flags[row].test(papilo::RowFlag::kRedundant)) continue;

    bool lhs_inf_row = row_flags[row].test(papilo::RowFlag::kLhsInf);
    bool rhs_inf_row = row_flags[row].test(papilo::RowFlag::kRhsInf);
    if (lhs_inf_row && rhs_inf_row) continue;
    if (!lhs_inf_row && !rhs_inf_row) continue;

    auto row_coeff   = constraint_matrix.getRowCoefficients(row);
    const int* cols  = row_coeff.getIndices();
    const f_t* vals  = row_coeff.getValues();
    const int length = row_coeff.getLength();

    if (length < 2) continue;

    int neg_count = 0;
    int neg_last  = -1;
    f_t sum_neg   = 0;
    int pos_count = 0;
    int pos_last  = -1;
    f_t sum_pos   = 0;
    bool valid    = true;

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

      if (coef < -num.getFeasTol()) {
        neg_last = col;
        ++neg_count;
        sum_neg += coef;
      } else if (coef > num.getFeasTol()) {
        pos_last = col;
        ++pos_count;
        sum_pos += coef;
      } else {
        valid = false;
        break;
      }
    }
    if (!valid) continue;

    bool is_leq = lhs_inf_row && !rhs_inf_row;
    bool is_geq = !lhs_inf_row && rhs_inf_row;
    f_t bound   = is_leq ? rhs_values[row] : lhs_values[row];

    int master_col = -1;
    bool detected  = false;

    // Case A: one negative (master), rest positive (details), <= 0
    if (neg_count == 1 && pos_count >= 1 && is_leq && num.isZero(bound)) {
      f_t M = -sum_neg;
      if (M + num.getFeasTol() >= sum_pos) {
        master_col = neg_last;
        detected   = true;
      }
    }
    // Case B: one positive (master), rest negative (details), >= 0
    // Normalized: sum(|a_i|*x_i) <= M*y
    else if (pos_count == 1 && neg_count >= 1 && is_geq && num.isZero(bound)) {
      f_t M = sum_pos;
      if (M + num.getFeasTol() >= -sum_neg) {
        master_col = pos_last;
        detected   = true;
      }
    }

    if (!detected || master_col == -1) continue;

    bigm_row_flag[row] = true;
    ++n_bigm_rows;
    for (int j = 0; j < length; ++j) {
      if (cols[j] != master_col) {
        detail_bigm_count[cols[j]]++;
        detail_master[cols[j]] = master_col;
        ++detail_set_size;
      }
    }
  }

  if (n_bigm_rows == 0) return papilo::PresolveStatus::kUnchanged;

  // Phase 2: Safety check per detail (row-major pass).
  //
  // The substitution x_i = y INCREASES x_i from 0 to 1 when y=1. Safe when:
  //   (a) >= constraint with coeff > 0: increasing x helps
  //   (b) <= constraint with coeff < 0: increasing x helps
  //   Objective: c(x_i) <= 0 (increasing x is free or beneficial)

  std::vector<bool> detail_unsafe(ncols, false);

  for (int row = 0; row < nrows; ++row) {
    if (this->is_time_exceeded(timer, tlim)) return papilo::PresolveStatus::kUnchanged;
    if (row_flags[row].test(papilo::RowFlag::kRedundant)) continue;
    if (bigm_row_flag[row]) continue;

    bool lhs_inf = row_flags[row].test(papilo::RowFlag::kLhsInf);
    bool rhs_inf = row_flags[row].test(papilo::RowFlag::kRhsInf);
    bool is_geq  = !lhs_inf && rhs_inf;
    bool is_leq  = lhs_inf && !rhs_inf;

    auto row_coeff   = constraint_matrix.getRowCoefficients(row);
    const int* cols  = row_coeff.getIndices();
    const f_t* vals  = row_coeff.getValues();
    const int length = row_coeff.getLength();

    for (int j = 0; j < length; ++j) {
      int col = cols[j];
      if (detail_bigm_count[col] != 1) continue;
      if (detail_unsafe[col]) continue;

      f_t cv         = vals[j];
      bool safe_here = (is_geq && cv > 0) || (is_leq && cv < 0);
      if (!safe_here) detail_unsafe[col] = true;
    }
  }

  int n_safe = 0;
  for (int col = 0; col < ncols; ++col) {
    if (detail_bigm_count[col] != 1) continue;
    if (detail_unsafe[col]) continue;
    if (objective[col] > num.getFeasTol()) continue;
    ++n_safe;
  }

  if (n_safe == 0) return papilo::PresolveStatus::kUnchanged;

  CUOPT_LOG_INFO(
    "BigM indicator aggregation: %d big-M constraints detected, %d detail variables "
    "substituted out of %d candidates",
    n_bigm_rows,
    n_safe,
    detail_set_size);

  // Phase 3: Substitute safe details via replaceCol.
  //
  // replaceCol(detail, master, 1.0, 0.0) declares detail = master.
  // PaPILO substitutes the detail out of all constraints and stores
  // the relationship for postsolve.

  if (this->is_time_exceeded(timer, tlim)) return papilo::PresolveStatus::kUnchanged;

  papilo::PresolveStatus status = papilo::PresolveStatus::kUnchanged;

  for (int col = 0; col < ncols; ++col) {
    if (detail_bigm_count[col] != 1) continue;
    if (detail_unsafe[col]) continue;
    if (objective[col] > num.getFeasTol()) continue;
    if (this->is_time_exceeded(timer, tlim)) break;
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
