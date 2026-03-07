/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/basis_updates.hpp>
#include <dual_simplex/bb_event.hpp>
#include <dual_simplex/bounds_strengthening.hpp>
#include <dual_simplex/initial_basis.hpp>
#include <dual_simplex/mip_node.hpp>
#include <dual_simplex/types.hpp>
#include <utilities/work_limit_timer.hpp>

#include <cmath>
#include <limits>
#include <memory>
#include <queue>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {

// Queued pseudo-cost update for BSP determinism
// Updates are collected during horizon, then applied in deterministic order at sync
template <typename i_t, typename f_t>
struct pseudo_cost_update_t {
  i_t variable;
  rounding_direction_t direction;
  f_t delta;      // change_in_obj / frac
  double vt;      // virtual time when update occurred (for deterministic ordering)
  int worker_id;  // for tie-breaking in sort
};

// Per-worker state for BSP (Bulk Synchronous Parallel) branch-and-bound
template <typename i_t, typename f_t>
struct bb_worker_state_t {
  int worker_id{0};

  // ==========================================================================
  // Plunging data structures (matching explore_subtree strategy)
  // ==========================================================================

  // Plunge stack: depth-first path through the tree
  // - Front = next node to process (LIFO)
  // - Max size = 2 (current node's sibling only)
  // - When branching with sibling on stack, sibling is moved to backlog
  std::deque<mip_node_t<i_t, f_t>*> plunge_stack;

  // Backlog: nodes "plugged" when branching - candidates for load balancing
  // When branching with a sibling on the plunge stack, that sibling moves here.
  // At horizon sync, backlog nodes participate in redistribution.
  std::vector<mip_node_t<i_t, f_t>*> backlog;

  // Current node being processed (may be paused at horizon boundary)
  mip_node_t<i_t, f_t>* current_node{nullptr};

  // Last node that was solved (for basis warm-start detection)
  // If next node's parent == last_solved_node, we can reuse basis
  mip_node_t<i_t, f_t>* last_solved_node{nullptr};

  // Worker's virtual time clock (cumulative work units)
  double clock{0.0};

  // Current horizon boundaries (for BSP sync)
  double horizon_start{0.0};
  double horizon_end{0.0};

  // Creation sequence counter - cumulative across horizons for unique identity
  // Each node created by this worker gets (worker_id, next_creation_seq++)
  int32_t next_creation_seq{0};

  // Events generated during this horizon
  bb_event_batch_t<i_t, f_t> events;

  // Event sequence counter for deterministic tie-breaking
  int event_sequence{0};

  // LP problem copy for this worker (bounds modified per node)
  std::unique_ptr<lp_problem_t<i_t, f_t>> leaf_problem;

  // Basis factorization state
  std::unique_ptr<basis_update_mpf_t<i_t, f_t>> basis_factors;

  // Bounds strengthening (node presolver)
  std::unique_ptr<bounds_strengthening_t<i_t, f_t>> node_presolver;

  // Working vectors for basis
  std::vector<i_t> basic_list;
  std::vector<i_t> nonbasic_list;

  // Work unit context for this worker
  work_limit_context_t work_context;

  // Whether basis needs recomputation for next node
  bool recompute_bounds_and_basis{true};

  // Per-horizon statistics (reset each horizon)
  i_t nodes_processed_this_horizon{0};
  double work_units_this_horizon{0.0};

  // Cumulative statistics (across all horizons)
  i_t total_nodes_processed{0};
  i_t total_nodes_pruned{0};
  i_t total_nodes_branched{0};
  i_t total_nodes_infeasible{0};
  i_t total_integer_solutions{0};
  i_t total_nodes_assigned{0};  // via load balancing
  double total_work_units{0.0};

  // Timing statistics (in seconds)
  double total_runtime{0.0};       // Total time spent doing actual work
  double total_barrier_wait{0.0};  // Total time spent waiting at horizon sync barriers
  double total_nowork_time{0.0};   // Total time spent with no nodes to work on
  double horizon_finish_time{
    0.0};  // Timestamp when worker finished current horizon (for barrier wait calc)

  // Worker-local upper bound for BSP determinism (prevents cross-worker pruning races)
  f_t local_upper_bound{std::numeric_limits<f_t>::infinity()};

  // Queued integer solutions found during this horizon (merged at sync)
  struct queued_integer_solution_t {
    f_t objective;
    std::vector<f_t> solution;
    i_t depth;
  };
  std::vector<queued_integer_solution_t> integer_solutions;

  // Queued pseudo-cost updates (applied to global pseudo-costs at sync)
  std::vector<pseudo_cost_update_t<i_t, f_t>> pseudo_cost_updates;

  // Pseudo-cost snapshot for deterministic variable selection
  // Initialized from global pseudo-costs at horizon start, then updated locally
  // during the horizon as the worker processes nodes. This allows within-horizon
  // learning while maintaining determinism (each worker's updates are sequential).
  // At sync, all workers' updates are merged into global pseudo-costs, and new
  // snapshots are taken at the next horizon start.
  std::vector<f_t> pc_sum_up_snapshot;
  std::vector<f_t> pc_sum_down_snapshot;
  std::vector<i_t> pc_num_up_snapshot;
  std::vector<i_t> pc_num_down_snapshot;

  // Constructor
  explicit bb_worker_state_t(int id)
    : worker_id(id), work_context("BB_Worker_" + std::to_string(id))
  {
  }

  // Initialize worker with problem data
  void initialize(const lp_problem_t<i_t, f_t>& original_lp,
                  const csr_matrix_t<i_t, f_t>& Arow,
                  const std::vector<variable_type_t>& var_types,
                  i_t refactor_frequency,
                  bool deterministic)
  {
    // Create copy of LP problem for this worker
    leaf_problem = std::make_unique<lp_problem_t<i_t, f_t>>(original_lp);

    // Initialize basis factors
    const i_t m   = leaf_problem->num_rows;
    basis_factors = std::make_unique<basis_update_mpf_t<i_t, f_t>>(m, refactor_frequency);

    // Initialize bounds strengthening
    std::vector<char> row_sense;
    node_presolver =
      std::make_unique<bounds_strengthening_t<i_t, f_t>>(*leaf_problem, Arow, row_sense, var_types);

    // Initialize working vectors
    basic_list.resize(m);
    nonbasic_list.clear();

    // Configure work context
    work_context.deterministic = deterministic;
  }

  // Reset for new horizon
  void reset_for_horizon(double horizon_start, double horizon_end, f_t global_upper_bound)
  {
    // Reset clock to horizon_start for consistent VT timestamps across workers
    clock = horizon_start;
    events.clear();
    events.horizon_start         = horizon_start;
    events.horizon_end           = horizon_end;
    event_sequence               = 0;
    nodes_processed_this_horizon = 0;
    work_units_this_horizon      = 0.0;
    // Also sync work_context to match clock for consistent tracking
    work_context.set_current_work(horizon_start, false);
    // Note: next_creation_seq is NOT reset - it's cumulative for unique identity

    // Initialize worker-local upper bound from global (for BSP determinism)
    local_upper_bound = global_upper_bound;

    // Clear queued updates from previous horizon
    integer_solutions.clear();
    pseudo_cost_updates.clear();
  }

  // Queue a pseudo-cost update for global sync AND apply it to local snapshot immediately
  // This allows within-horizon learning while maintaining determinism:
  // - Local snapshot updates are sequential within each worker (deterministic)
  // - Global updates are merged at sync in sorted (VT, worker_id) order (deterministic)
  void queue_pseudo_cost_update(i_t variable, rounding_direction_t direction, f_t delta)
  {
    // Queue for global sync at horizon end
    pseudo_cost_updates.push_back({variable, direction, delta, clock, worker_id});

    // Also apply to local snapshot immediately for better variable selection
    if (direction == rounding_direction_t::DOWN) {
      pc_sum_down_snapshot[variable] += delta;
      pc_num_down_snapshot[variable]++;
    } else {
      pc_sum_up_snapshot[variable] += delta;
      pc_num_up_snapshot[variable]++;
    }
  }

  // Variable selection using snapshot (for BSP determinism)
  // Returns the best variable to branch on based on pseudo-cost scores
  i_t variable_selection_from_snapshot(const std::vector<i_t>& fractional,
                                       const std::vector<f_t>& solution) const
  {
    const i_t num_fractional = fractional.size();
    if (num_fractional == 0) return -1;

    // Compute averages from snapshot
    i_t num_initialized_down = 0;
    i_t num_initialized_up   = 0;
    f_t pseudo_cost_down_avg = 0;
    f_t pseudo_cost_up_avg   = 0;

    const i_t n = pc_sum_down_snapshot.size();
    for (i_t j = 0; j < n; ++j) {
      if (pc_num_down_snapshot[j] > 0) {
        ++num_initialized_down;
        if (std::isfinite(pc_sum_down_snapshot[j])) {
          pseudo_cost_down_avg += pc_sum_down_snapshot[j] / pc_num_down_snapshot[j];
        }
      }
      if (pc_num_up_snapshot[j] > 0) {
        ++num_initialized_up;
        if (std::isfinite(pc_sum_up_snapshot[j])) {
          pseudo_cost_up_avg += pc_sum_up_snapshot[j] / pc_num_up_snapshot[j];
        }
      }
    }
    if (num_initialized_down > 0) {
      pseudo_cost_down_avg /= num_initialized_down;
    } else {
      pseudo_cost_down_avg = 1.0;
    }
    if (num_initialized_up > 0) {
      pseudo_cost_up_avg /= num_initialized_up;
    } else {
      pseudo_cost_up_avg = 1.0;
    }

    // Compute scores
    std::vector<f_t> score(num_fractional);
    for (i_t k = 0; k < num_fractional; ++k) {
      const i_t j = fractional[k];
      f_t pc_down = (pc_num_down_snapshot[j] != 0)
                      ? pc_sum_down_snapshot[j] / pc_num_down_snapshot[j]
                      : pseudo_cost_down_avg;
      f_t pc_up   = (pc_num_up_snapshot[j] != 0) ? pc_sum_up_snapshot[j] / pc_num_up_snapshot[j]
                                                 : pseudo_cost_up_avg;
      constexpr f_t eps = 1e-6;
      const f_t f_down  = solution[j] - std::floor(solution[j]);
      const f_t f_up    = std::ceil(solution[j]) - solution[j];
      score[k]          = std::max(f_down * pc_down, eps) * std::max(f_up * pc_up, eps);
    }

    // Select variable with maximum score
    i_t branch_var = fractional[0];
    f_t max_score  = score[0];
    for (i_t k = 1; k < num_fractional; ++k) {
      if (score[k] > max_score) {
        max_score  = score[k];
        branch_var = fractional[k];
      }
    }

    return branch_var;
  }

  // ==========================================================================
  // Node enqueueing methods
  // ==========================================================================

  // Add a node to the plunge stack, assigning BSP identity if not already set
  // Used for initial node assignment and when starting a new plunge from backlog
  void enqueue_node(mip_node_t<i_t, f_t>* node)
  {
    // Assign BSP identity if not already set
    if (!node->has_bsp_identity()) {
      node->origin_worker_id = worker_id;
      node->creation_seq     = next_creation_seq++;
    }
    plunge_stack.push_front(node);
  }

  // Add a node that already has BSP identity (from load balancing or initial distribution)
  // Goes to plunge stack front for immediate processing
  void enqueue_node_with_identity(mip_node_t<i_t, f_t>* node)
  {
    assert(node->has_bsp_identity() &&
           "Node must have BSP identity for enqueue_node_with_identity");
    plunge_stack.push_front(node);
  }

  // Add children after branching with proper plunging behavior:
  // 1. If plunge stack has a sibling, move it to backlog (plugging)
  // 2. Push both children to plunge stack with preferred child on top
  // Returns the child that was placed on top (to be explored first)
  mip_node_t<i_t, f_t>* enqueue_children_for_plunge(mip_node_t<i_t, f_t>* down_child,
                                                    mip_node_t<i_t, f_t>* up_child,
                                                    rounding_direction_t preferred_direction)
  {
    // PLUGGING: If plunge stack has a sibling from previous branch, move it to backlog
    if (!plunge_stack.empty()) {
      mip_node_t<i_t, f_t>* sibling = plunge_stack.back();
      plunge_stack.pop_back();
      backlog.push_back(sibling);
    }

    // Assign BSP identity to children
    if (!down_child->has_bsp_identity()) {
      down_child->origin_worker_id = worker_id;
      down_child->creation_seq     = next_creation_seq++;
    }
    if (!up_child->has_bsp_identity()) {
      up_child->origin_worker_id = worker_id;
      up_child->creation_seq     = next_creation_seq++;
    }

    // Push children - preferred child on top (front) for immediate exploration
    mip_node_t<i_t, f_t>* first_child;
    if (preferred_direction == rounding_direction_t::UP) {
      plunge_stack.push_front(down_child);  // Second to explore
      plunge_stack.push_front(up_child);    // First to explore (on top)
      first_child = up_child;
    } else {
      plunge_stack.push_front(up_child);    // Second to explore
      plunge_stack.push_front(down_child);  // First to explore (on top)
      first_child = down_child;
    }

    return first_child;
  }

  // ==========================================================================
  // Node dequeueing methods
  // ==========================================================================

  // Get next node to process using plunging strategy:
  // 1. Resume paused node if any
  // 2. Pop from plunge stack (depth-first continuation)
  // 3. Fall back to backlog (best-first from plugged nodes)
  mip_node_t<i_t, f_t>* dequeue_node()
  {
    // 1. Resume paused node if any
    if (current_node != nullptr) {
      mip_node_t<i_t, f_t>* node = current_node;
      current_node               = nullptr;
      return node;
    }

    // 2. Prefer plunge stack (depth-first continuation)
    if (!plunge_stack.empty()) {
      mip_node_t<i_t, f_t>* node = plunge_stack.front();
      plunge_stack.pop_front();
      return node;
    }

    // 3. Fall back to backlog - select best node (lowest lower_bound)
    if (!backlog.empty()) {
      auto best_it =
        std::min_element(backlog.begin(),
                         backlog.end(),
                         [](const mip_node_t<i_t, f_t>* a, const mip_node_t<i_t, f_t>* b) {
                           // Best-first: prefer lower bound
                           if (a->lower_bound != b->lower_bound) {
                             return a->lower_bound < b->lower_bound;
                           }
                           // Deterministic tie-breaking by BSP identity
                           if (a->origin_worker_id != b->origin_worker_id) {
                             return a->origin_worker_id < b->origin_worker_id;
                           }
                           return a->creation_seq < b->creation_seq;
                         });
      mip_node_t<i_t, f_t>* node = *best_it;
      backlog.erase(best_it);
      return node;
    }

    return nullptr;
  }

  // ==========================================================================
  // Queue state queries
  // ==========================================================================

  // Check if worker has work available
  bool has_work() const
  {
    return current_node != nullptr || !plunge_stack.empty() || !backlog.empty();
  }

  // Get number of nodes in worker's queues (including paused node)
  size_t queue_size() const
  {
    return plunge_stack.size() + backlog.size() + (current_node != nullptr ? 1 : 0);
  }

  // Get number of nodes in plunge stack only
  size_t plunge_stack_size() const { return plunge_stack.size(); }

  // Get number of nodes in backlog only
  size_t backlog_size() const { return backlog.size(); }

  // ==========================================================================
  // Load balancing support
  // ==========================================================================

  // Extract all nodes from worker (for load balancing)
  // Returns nodes in arbitrary order - caller should sort if deterministic order needed
  std::vector<mip_node_t<i_t, f_t>*> extract_all_nodes()
  {
    std::vector<mip_node_t<i_t, f_t>*> nodes;
    nodes.reserve(queue_size());

    // Include paused node if any
    if (current_node != nullptr) {
      nodes.push_back(current_node);
      current_node = nullptr;
    }

    // Extract from plunge stack
    for (auto* node : plunge_stack) {
      nodes.push_back(node);
    }
    plunge_stack.clear();

    // Extract from backlog
    for (auto* node : backlog) {
      nodes.push_back(node);
    }
    backlog.clear();

    return nodes;
  }

  // Extract only backlog nodes (for redistribution at horizon sync)
  // Plunge stack nodes stay with worker for locality
  std::vector<mip_node_t<i_t, f_t>*> extract_backlog_nodes()
  {
    std::vector<mip_node_t<i_t, f_t>*> nodes = std::move(backlog);
    backlog.clear();
    return nodes;
  }

  // Clear all queues without returning nodes (use with caution)
  void clear_queue()
  {
    current_node = nullptr;
    plunge_stack.clear();
    backlog.clear();
  }

  // Record an event
  void record_event(bb_event_t<i_t, f_t> event)
  {
    event.event_sequence = event_sequence++;
    events.add(std::move(event));
  }

  // Pause current node processing at horizon boundary
  void pause_current_node(mip_node_t<i_t, f_t>* node, double accumulated_vt)
  {
    node->accumulated_vt = accumulated_vt;
    node->bsp_state      = bsp_node_state_t::PAUSED;
    current_node         = node;

    record_event(
      bb_event_t<i_t, f_t>::make_paused(clock, worker_id, node->node_id, 0, accumulated_vt));
  }

  // Record node branching event
  void record_branched(
    mip_node_t<i_t, f_t>* node, i_t down_child_id, i_t up_child_id, i_t branch_var, f_t branch_val)
  {
    record_event(bb_event_t<i_t, f_t>::make_branched(clock,
                                                     worker_id,
                                                     node->node_id,
                                                     0,
                                                     down_child_id,
                                                     up_child_id,
                                                     node->lower_bound,
                                                     branch_var,
                                                     branch_val));
  }

  // Record integer solution found
  void record_integer_solution(mip_node_t<i_t, f_t>* node, f_t objective)
  {
    record_event(
      bb_event_t<i_t, f_t>::make_integer_solution(clock, worker_id, node->node_id, 0, objective));
  }

  // Record node fathomed
  void record_fathomed(mip_node_t<i_t, f_t>* node, f_t lower_bound)
  {
    record_event(
      bb_event_t<i_t, f_t>::make_fathomed(clock, worker_id, node->node_id, 0, lower_bound));
  }

  // Record node infeasible
  void record_infeasible(mip_node_t<i_t, f_t>* node)
  {
    record_event(bb_event_t<i_t, f_t>::make_infeasible(clock, worker_id, node->node_id, 0));
  }

  // Record numerical error
  void record_numerical(mip_node_t<i_t, f_t>* node)
  {
    record_event(bb_event_t<i_t, f_t>::make_numerical(clock, worker_id, node->node_id, 0));
  }

  // Update clock with work units
  void advance_clock(double work_units)
  {
    clock += work_units;
    work_units_this_horizon += work_units;
    total_work_units += work_units;
    work_context.record_work(work_units);
  }

  // Track node processed (called when a node LP solve completes)
  void track_node_processed()
  {
    ++nodes_processed_this_horizon;
    ++total_nodes_processed;
  }

  // Track node branched
  void track_node_branched() { ++total_nodes_branched; }

  // Track node pruned (fathomed due to bound)
  void track_node_pruned() { ++total_nodes_pruned; }

  // Track node infeasible
  void track_node_infeasible() { ++total_nodes_infeasible; }

  // Track integer solution found
  void track_integer_solution() { ++total_integer_solutions; }

  // Track node assigned via load balancing
  void track_node_assigned() { ++total_nodes_assigned; }
};

// Container for all worker states in BSP B&B
template <typename i_t, typename f_t>
class bb_worker_pool_t {
 public:
  bb_worker_pool_t() = default;

  // Initialize pool with specified number of workers
  void initialize(int num_workers,
                  const lp_problem_t<i_t, f_t>& original_lp,
                  const csr_matrix_t<i_t, f_t>& Arow,
                  const std::vector<variable_type_t>& var_types,
                  i_t refactor_frequency,
                  bool deterministic)
  {
    workers_.clear();
    workers_.reserve(num_workers);
    for (int i = 0; i < num_workers; ++i) {
      workers_.emplace_back(i);
      workers_.back().initialize(original_lp, Arow, var_types, refactor_frequency, deterministic);
    }
  }

  // Get worker by ID
  bb_worker_state_t<i_t, f_t>& operator[](int worker_id) { return workers_[worker_id]; }

  const bb_worker_state_t<i_t, f_t>& operator[](int worker_id) const { return workers_[worker_id]; }

  // Get number of workers
  int size() const { return static_cast<int>(workers_.size()); }

  // Reset all workers for new horizon
  void reset_for_horizon(double horizon_start, double horizon_end, f_t global_upper_bound)
  {
    for (auto& worker : workers_) {
      worker.reset_for_horizon(horizon_start, horizon_end, global_upper_bound);
    }
  }

  // Collect all events from all workers into a single sorted batch
  bb_event_batch_t<i_t, f_t> collect_and_sort_events()
  {
    bb_event_batch_t<i_t, f_t> all_events;
    for (auto& worker : workers_) {
      for (auto& event : worker.events.events) {
        all_events.add(std::move(event));
      }
      worker.events.clear();
    }
    all_events.sort_for_replay();
    return all_events;
  }

  // Check if any worker has work
  bool any_has_work() const
  {
    for (const auto& worker : workers_) {
      if (worker.has_work()) return true;
    }
    return false;
  }

  // Get total queue size across all workers
  size_t total_queue_size() const
  {
    size_t total = 0;
    for (const auto& worker : workers_) {
      total += worker.queue_size();
    }
    return total;
  }

  // Iterator support
  auto begin() { return workers_.begin(); }
  auto end() { return workers_.end(); }
  auto begin() const { return workers_.begin(); }
  auto end() const { return workers_.end(); }

 private:
  std::vector<bb_worker_state_t<i_t, f_t>> workers_;
};

}  // namespace cuopt::linear_programming::dual_simplex
