/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <vector>

namespace cuopt::mps_parser {

/**
 * @brief Symmetrize a CSR matrix by computing A + A^T
 *
 * Given a CSR matrix A, computes the symmetric matrix H = A + A^T.
 * Diagonal entries are doubled (A[i,i] + A[i,i] = 2*A[i,i]).
 * Off-diagonal entries are summed (H[i,j] = A[i,j] + A[j,i]).
 *
 * @tparam i_t Integer type for indices
 * @tparam f_t Floating point type for values
 *
 * @param[in] in_values   CSR values array
 * @param[in] in_indices  CSR column indices array
 * @param[in] in_offsets  CSR row offsets array (size = n_rows + 1)
 * @param[in] n_rows      Number of rows (and columns, assuming square matrix)
 * @param[out] out_values   Output CSR values
 * @param[out] out_indices  Output CSR column indices
 * @param[out] out_offsets  Output CSR row offsets
 */
template <typename i_t, typename f_t>
void symmetrize_csr(const f_t* in_values,
                    const i_t* in_indices,
                    const i_t* in_offsets,
                    i_t n_rows,
                    std::vector<f_t>& out_values,
                    std::vector<i_t>& out_indices,
                    std::vector<i_t>& out_offsets)
{
  // Optimized 3-pass algorithm
  // Pass 1: Count entries per row in A + A^T
  std::vector<i_t> row_counts(n_rows, 0);
  for (i_t i = 0; i < n_rows; ++i) {
    for (i_t p = in_offsets[i]; p < in_offsets[i + 1]; ++p) {
      i_t j = in_indices[p];
      row_counts[i]++;
      if (i != j) { row_counts[j]++; }
    }
  }

  // Build temporary offsets via prefix sum
  std::vector<i_t> temp_offsets(n_rows + 1);
  temp_offsets[0] = 0;
  for (i_t i = 0; i < n_rows; ++i) {
    temp_offsets[i + 1] = temp_offsets[i] + row_counts[i];
  }

  i_t total_entries = temp_offsets[n_rows];
  std::vector<i_t> temp_indices(total_entries);
  std::vector<f_t> temp_values(total_entries);

  // Pass 2: Fill entries directly
  std::vector<i_t> row_pos = temp_offsets;  // Copy for tracking insertion positions

  for (i_t i = 0; i < n_rows; ++i) {
    for (i_t p = in_offsets[i]; p < in_offsets[i + 1]; ++p) {
      i_t j = in_indices[p];
      f_t x = in_values[p];

      // Add entry (i, j) with value 2x for diagonal, x for off-diagonal
      temp_indices[row_pos[i]] = j;
      temp_values[row_pos[i]]  = (i == j) ? (2 * x) : x;
      row_pos[i]++;

      // Add transpose entry (j, i) if off-diagonal
      if (i != j) {
        temp_indices[row_pos[j]] = i;
        temp_values[row_pos[j]]  = x;
        row_pos[j]++;
      }
    }
  }

  // Pass 3: Deduplicate and build final CSR
  std::vector<i_t> workspace(n_rows, -1);
  out_offsets.resize(n_rows + 1);
  out_indices.resize(total_entries);
  out_values.resize(total_entries);

  i_t nz = 0;
  for (i_t i = 0; i < n_rows; ++i) {
    i_t row_start_out = nz;
    out_offsets[i]    = row_start_out;

    for (i_t p = temp_offsets[i]; p < temp_offsets[i + 1]; ++p) {
      i_t j = temp_indices[p];
      f_t x = temp_values[p];

      if (workspace[j] >= row_start_out) {
        out_values[workspace[j]] += x;
      } else {
        workspace[j]    = nz;
        out_indices[nz] = j;
        out_values[nz]  = x;
        nz++;
      }
    }
  }

  out_offsets[n_rows] = nz;
  out_indices.resize(nz);
  out_values.resize(nz);
}

/**
 * @brief Symmetrize a CSR matrix using std::vector (convenience overload)
 */
template <typename i_t, typename f_t>
void symmetrize_csr(const std::vector<f_t>& in_values,
                    const std::vector<i_t>& in_indices,
                    const std::vector<i_t>& in_offsets,
                    std::vector<f_t>& out_values,
                    std::vector<i_t>& out_indices,
                    std::vector<i_t>& out_offsets)
{
  i_t n_rows = static_cast<i_t>(in_offsets.size()) - 1;
  symmetrize_csr(in_values.data(),
                 in_indices.data(),
                 in_offsets.data(),
                 n_rows,
                 out_values,
                 out_indices,
                 out_offsets);
}

}  // namespace cuopt::mps_parser
