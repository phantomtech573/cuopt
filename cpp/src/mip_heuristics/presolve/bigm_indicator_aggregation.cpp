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
  //   -M*y + a_1*x_1 + ... + a_K*x_K <= 0   with M >= sum(a_i * U_i), a_i > 0
  //   Row form: lhs=-inf, rhs=0, one negative coeff (master), rest positive (details).
  //
  // Also detected in normalized form (Case B):
  //   a_1*x_1 + ... + a_K*x_K - M*y >= 0   (one positive master, rest negative details)
  //   Equivalent to: sum(|a_i|*x_i) <= M*y.
  //
  // The implication y=0 => x_i=0 holds for any positive a_i, since
  // sum(a_i * x_i) <= 0 with a_i > 0 and x_i >= 0 forces all x_i = 0.
  //
  // The master must be binary. Details can be binary, integer, or continuous
  // with lb=0 and finite ub > 0. The substitution is x_i = U_i * y where
  // U_i = upper_bound[x_i].
  //
  // The M-check uses sum(a_i * U_i) instead of sum(a_i) to ensure all
  // details can simultaneously be at their upper bounds when y = 1.

  // Per-detail flags: STANDARD means y=0 => x=0 detected, REVERSE means
  // y=1 => x=1 detected. If both are set (SANDWICH), the substitution x=y
  // is exact and requires no cost or constraint-sign check.
  static constexpr uint8_t FLAG_STANDARD = 1;
  static constexpr uint8_t FLAG_REVERSE  = 2;
  static constexpr uint8_t FLAG_SANDWICH = FLAG_STANDARD | FLAG_REVERSE;

  std::vector<int> detail_bigm_count(ncols, 0);
  std::vector<int> detail_master(ncols, -1);
  std::vector<uint8_t> detail_flags(ncols, 0);
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

    // Scan: partition into one master (binary) and detail candidates.
    // Track both sum(|coeff|) and sum(|coeff| * U_i) for the M-check.
    int neg_count        = 0;
    int neg_last         = -1;
    f_t sum_neg_coeff_ub = 0;
    int pos_count        = 0;
    int pos_last         = -1;
    f_t sum_pos_coeff_ub = 0;
    bool valid           = true;

    for (int j = 0; j < length; ++j) {
      int col  = cols[j];
      f_t coef = vals[j];

      if (col_flags[col].test(papilo::ColFlag::kFixed, papilo::ColFlag::kSubstituted)) {
        valid = false;
        break;
      }

      if (coef < -num.getFeasTol()) {
        neg_last = col;
        ++neg_count;
        sum_neg_coeff_ub += coef * upper_bounds[col];
      } else if (coef > num.getFeasTol()) {
        pos_last = col;
        ++pos_count;
        sum_pos_coeff_ub += coef * upper_bounds[col];
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
    // Master must be binary; details must have lb=0, finite ub > 0.
    if (neg_count == 1 && pos_count >= 1 && is_leq && num.isZero(bound)) {
      if (is_binary_or_implied(
            neg_last, col_flags.data(), lower_bounds.data(), upper_bounds.data())) {
        f_t M = -sum_neg_coeff_ub;  // |master_coeff| * 1 (master is binary, ub=1)
        // Verify all details are eligible
        bool details_ok = true;
        for (int j = 0; j < length && details_ok; ++j) {
          if (cols[j] != neg_last)
            details_ok =
              is_valid_detail(cols[j], col_flags.data(), lower_bounds.data(), upper_bounds.data());
        }
        if (details_ok && M + num.getFeasTol() >= sum_pos_coeff_ub) {
          master_col = neg_last;
          detected   = true;
        }
      }
    }
    // Case B: one positive (master), rest negative (details), >= 0
    // Normalized: sum(|a_i|*x_i) <= M*y
    else if (pos_count == 1 && neg_count >= 1 && is_geq && num.isZero(bound)) {
      if (is_binary_or_implied(
            pos_last, col_flags.data(), lower_bounds.data(), upper_bounds.data())) {
        f_t M           = sum_pos_coeff_ub;  // master_coeff * 1 (master is binary, ub=1)
        bool details_ok = true;
        for (int j = 0; j < length && details_ok; ++j) {
          if (cols[j] != pos_last)
            details_ok =
              is_valid_detail(cols[j], col_flags.data(), lower_bounds.data(), upper_bounds.data());
        }
        if (details_ok && M + num.getFeasTol() >= -sum_neg_coeff_ub) {
          master_col = pos_last;
          detected   = true;
        }
      }
    }

    if (!detected || master_col == -1) continue;

    bigm_row_flag[row] = true;
    ++n_bigm_rows;
    for (int j = 0; j < length; ++j) {
      if (cols[j] != master_col) {
        detail_bigm_count[cols[j]]++;
        detail_master[cols[j]] = master_col;
        detail_flags[cols[j]] |= FLAG_STANDARD;
        ++detail_set_size;
      }
    }
  }

  if (n_bigm_rows == 0) return papilo::PresolveStatus::kUnchanged;

  // Sandwich detection: scan for pairwise reverse constraints x_i - y >= 0
  // (equivalently x_i >= y, i.e., y=1 => x_i=1) where x_i is already a
  // standard detail of master y. Both x_i and y must be binary.
  // These are 2-variable rows: +1*x_i - 1*y >= 0 (lhs=0, rhs=inf).
  // When both standard and reverse hold, the feasible set is {(0,0),(1,1)},
  // making x_i = y an exact primal reduction (no cost/constraint check needed).
  for (int row = 0; row < nrows; ++row) {
    if (this->is_time_exceeded(timer, tlim)) break;
    if (row_flags[row].test(papilo::RowFlag::kRedundant)) continue;
    if (bigm_row_flag[row]) continue;

    bool lhs_inf_row = row_flags[row].test(papilo::RowFlag::kLhsInf);
    bool rhs_inf_row = row_flags[row].test(papilo::RowFlag::kRhsInf);

    auto row_coeff   = constraint_matrix.getRowCoefficients(row);
    const int* cols  = row_coeff.getIndices();
    const f_t* vals  = row_coeff.getValues();
    const int length = row_coeff.getLength();

    if (length != 2) continue;

    // Look for x - y >= 0 or y - x <= 0 (two equivalent forms)
    int col_a = cols[0], col_b = cols[1];
    f_t ca = vals[0], cb = vals[1];

    // Both must be binary
    if (!is_binary_or_implied(col_a, col_flags.data(), lower_bounds.data(), upper_bounds.data()))
      continue;
    if (!is_binary_or_implied(col_b, col_flags.data(), lower_bounds.data(), upper_bounds.data()))
      continue;
    if (col_flags[col_a].test(papilo::ColFlag::kFixed, papilo::ColFlag::kSubstituted)) continue;
    if (col_flags[col_b].test(papilo::ColFlag::kFixed, papilo::ColFlag::kSubstituted)) continue;

    // Identify: which variable has coeff +1 (detail) and which has -1 (master)?
    // For >= 0:  +1*x - 1*y >= 0  means x >= y  (y=1 => x=1)
    // For <= 0:  -1*x + 1*y <= 0  means y <= x  (y=1 => x=1)
    int detail_col = -1, master_candidate = -1;

    if (!lhs_inf_row && rhs_inf_row && num.isZero(lhs_values[row])) {
      // >= 0: positive coeff is the detail, negative is the master
      if (num.isEq(ca, f_t{1}) && num.isEq(cb, f_t{-1})) {
        detail_col       = col_a;
        master_candidate = col_b;
      } else if (num.isEq(cb, f_t{1}) && num.isEq(ca, f_t{-1})) {
        detail_col       = col_b;
        master_candidate = col_a;
      }
    } else if (lhs_inf_row && !rhs_inf_row && num.isZero(rhs_values[row])) {
      // <= 0: negative coeff is the detail, positive is the master
      if (num.isEq(ca, f_t{-1}) && num.isEq(cb, f_t{1})) {
        detail_col       = col_a;
        master_candidate = col_b;
      } else if (num.isEq(cb, f_t{-1}) && num.isEq(ca, f_t{1})) {
        detail_col       = col_b;
        master_candidate = col_a;
      }
    }

    if (detail_col == -1) continue;

    // Only mark as reverse if this detail already has a standard big-M
    // with the SAME master.
    if ((detail_flags[detail_col] & FLAG_STANDARD) &&
        detail_master[detail_col] == master_candidate) {
      detail_flags[detail_col] |= FLAG_REVERSE;
    }
  }

  papilo::PresolveStatus status = papilo::PresolveStatus::kUnchanged;
  int n_reverse_tightened       = 0;

  // Reverse big-M tightening: detect multi-variable reverse rows
  //   sum(x_i) - K*y >= 0   (K = count, all binary, all coeffs +1)
  // and tighten them to equalities when all details have c(x_i) >= 0
  // and decreasing x_i is safe in all non-big-M constraints.
  //
  // Proof that tightening to equality preserves OPT:
  //   When y=1: sum = K (forced by binary + tight K). Equality holds.
  //   When y=0: sum >= 0. Since c(x_i) >= 0 and decreasing is safe,
  //     setting all x_i=0 is at least as good. So sum=0=K*y at optimum.
  // Therefore sum(x_i) = K*y at every optimal solution.
  //
  // Unlike replaceCol, this is safe under partial application: the equality
  // is a hard constraint that PaPILO's Substitution presolver can exploit
  // to eliminate variables at its own pace.

  // First, collect reverse big-M rows and their details.
  struct reverse_bigm_t {
    int row;
    int master_col;
    std::vector<int> detail_cols;
  };
  std::vector<reverse_bigm_t> reverse_rows;

  for (int row = 0; row < nrows; ++row) {
    if (this->is_time_exceeded(timer, tlim)) break;
    if (row_flags[row].test(papilo::RowFlag::kRedundant)) continue;
    if (bigm_row_flag[row]) continue;

    bool lhs_inf_row = row_flags[row].test(papilo::RowFlag::kLhsInf);
    bool rhs_inf_row = row_flags[row].test(papilo::RowFlag::kRhsInf);
    if (lhs_inf_row && rhs_inf_row) continue;
    if (!lhs_inf_row && !rhs_inf_row) continue;

    auto row_coeff   = constraint_matrix.getRowCoefficients(row);
    const int* cols  = row_coeff.getIndices();
    const f_t* vals  = row_coeff.getValues();
    const int length = row_coeff.getLength();

    if (length < 2) continue;

    // Detect: one negative (master), rest positive (details), >= 0
    // OR: one positive (master), rest negative (details), <= 0
    // Both require K = count, all detail |coeffs| = 1, master coeff = -K or +K.
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

    // Case: one negative (master -K), rest positive +1, >= 0
    if (neg_count == 1 && pos_count >= 1 && is_geq && num.isZero(bound)) {
      if (num.isEq(-sum_neg, (f_t)pos_count) && num.isEq(sum_pos, (f_t)pos_count)) {
        master_col = neg_last;
        detected   = true;
      }
    }
    // Case: one positive (master +K), rest negative -1, <= 0
    else if (pos_count == 1 && neg_count >= 1 && is_leq && num.isZero(bound)) {
      if (num.isEq(sum_pos, (f_t)neg_count) && num.isEq(-sum_neg, (f_t)neg_count)) {
        master_col = pos_last;
        detected   = true;
      }
    }

    if (!detected || master_col == -1) continue;

    reverse_bigm_t rbm;
    rbm.row        = row;
    rbm.master_col = master_col;
    for (int j = 0; j < length; ++j)
      if (cols[j] != master_col) rbm.detail_cols.push_back(cols[j]);
    reverse_rows.push_back(std::move(rbm));
  }

  // For each reverse big-M row, check if ALL its details satisfy:
  //   (a) c(x_i) >= 0 (non-negative cost: decreasing x is free or beneficial)
  //   (b) In every non-big-M, non-reverse-big-M constraint: decreasing x is safe
  //       (positive coeff in <=, negative coeff in >=)
  // If so, tighten the row from >= to equality.
  if (!reverse_rows.empty()) {
    // Build a set of reverse row indices for skipping in the safety scan
    std::vector<bool> reverse_row_flag(nrows, false);
    for (const auto& rbm : reverse_rows)
      reverse_row_flag[rbm.row] = true;

    // Per-column "decrease is unsafe" flag
    std::vector<bool> decrease_unsafe(ncols, false);

    for (int row = 0; row < nrows; ++row) {
      if (this->is_time_exceeded(timer, tlim)) break;
      if (row_flags[row].test(papilo::RowFlag::kRedundant)) continue;
      if (bigm_row_flag[row]) continue;
      if (reverse_row_flag[row]) continue;

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
        if (decrease_unsafe[col]) continue;
        f_t cv         = vals[j];
        bool safe_here = (is_leq && cv > 0) || (is_geq && cv < 0);
        if (!safe_here) decrease_unsafe[col] = true;
      }
    }

    n_reverse_tightened = 0;
    for (const auto& rbm : reverse_rows) {
      bool all_safe = true;
      for (int d : rbm.detail_cols) {
        if (decrease_unsafe[d] || objective[d] < -num.getFeasTol()) {
          all_safe = false;
          break;
        }
      }
      if (!all_safe) continue;

      papilo::TransactionGuard tg{reductions};
      reductions.lockRow(rbm.row);
      reductions.changeRowRHS(rbm.row, f_t{0});
      ++n_reverse_tightened;
      status = papilo::PresolveStatus::kReduced;
    }
  }

  // Phase 2: Safety check per detail (row-major pass).
  //
  // The substitution x_i = U_i * y INCREASES x_i from 0 to U_i when y=1.
  // Safe when:
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
      if (detail_flags[col] == FLAG_SANDWICH) continue;  // exact reduction, no check needed
      if (detail_unsafe[col]) continue;

      f_t cv         = vals[j];
      bool safe_here = (is_geq && cv > 0) || (is_leq && cv < 0);
      if (!safe_here) detail_unsafe[col] = true;
    }
  }

  int n_safe     = 0;
  int n_sandwich = 0;
  for (int col = 0; col < ncols; ++col) {
    if (detail_bigm_count[col] != 1) continue;
    if (detail_flags[col] == FLAG_SANDWICH) {
      ++n_safe;
      ++n_sandwich;
      continue;
    }
    if (detail_unsafe[col]) continue;
    if (objective[col] > num.getFeasTol()) continue;
    ++n_safe;
  }

  if (n_safe == 0) return status;

  CUOPT_LOG_INFO(
    "BigM indicator aggregation: %d big-M constraints detected, %d detail variables "
    "substituted (%d sandwich), %d reverse rows tightened to equality, out of %d candidates",
    n_bigm_rows,
    n_safe,
    n_sandwich,
    n_reverse_tightened,
    detail_set_size);

  // Phase 3: Substitute safe details via replaceCol.
  //
  // replaceCol(detail, master, U_i, 0) declares detail = U_i * master,
  // where U_i = upper_bound[detail]. For binary details, U_i = 1.
  // PaPILO substitutes the detail out of all constraints and stores
  // the relationship for postsolve.

  if (this->is_time_exceeded(timer, tlim)) return status;

  for (int col = 0; col < ncols; ++col) {
    if (detail_bigm_count[col] != 1) continue;
    bool is_sandwich = (detail_flags[col] == FLAG_SANDWICH);
    if (!is_sandwich) {
      if (detail_unsafe[col]) continue;
      if (objective[col] > num.getFeasTol()) continue;
    }
    if (this->is_time_exceeded(timer, tlim)) break;
    int master = detail_master[col];
    f_t U      = upper_bounds[col];
    reductions.replaceCol(col, master, U, f_t{0});
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
