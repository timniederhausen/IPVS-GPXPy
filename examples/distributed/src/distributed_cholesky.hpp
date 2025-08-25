﻿#pragma once

#include "gprat/scheduler.hpp"

#include "distributed_tile.hpp"
#include "scheduling.hpp"

GPRAT_NS_BEGIN

struct tiled_scheduler_sma : tiled_scheduler_distributed
{
    using tiled_scheduler_distributed::tiled_scheduler_distributed;

    std::size_t num_localities = localities_.size();
};

struct tiled_scheduler_cyclic : tiled_scheduler_distributed
{
    using tiled_scheduler_distributed::tiled_scheduler_distributed;

    std::size_t num_localities = localities_.size();
};

namespace schedule
{

constexpr std::size_t
covariance_tile(const tiled_scheduler_sma &sched, std::size_t n_tiles, std::size_t row, std::size_t col)
{
    return (row + col) % sched.num_localities;
}

constexpr std::size_t
cross_covariance_tile(const tiled_scheduler_sma &sched, std::size_t n_tiles, std::size_t row, std::size_t col)
{
    return (row + col) % sched.num_localities;
}

constexpr std::size_t alpha_tile(const tiled_scheduler_sma &sched, std::size_t n_tiles, std::size_t i) { return 0; }

constexpr std::size_t prediction_tile(const tiled_scheduler_sma &sched, std::size_t n_tiles, std::size_t i)
{
    return 0;
}

constexpr std::size_t
t_cross_covariance_tile(const tiled_scheduler_sma &sched, std::size_t n_tiles, std::size_t row, std::size_t col)
{
    return (row + col) % sched.num_localities;
}

constexpr std::size_t
prior_K_tile(const tiled_scheduler_sma &sched, std::size_t n_tiles, std::size_t row, std::size_t col)
{
    return (row + col) % sched.num_localities;
}

constexpr std::size_t
K_inv_tile(const tiled_scheduler_sma &sched, std::size_t n_tiles, std::size_t row, std::size_t col)
{
    return (row + col) % sched.num_localities;
}

constexpr std::size_t
K_grad_v_tile(const tiled_scheduler_sma &sched, std::size_t n_tiles, std::size_t row, std::size_t col)
{
    return (row + col) % sched.num_localities;
}

constexpr std::size_t
K_grad_l_tile(const tiled_scheduler_sma &sched, std::size_t n_tiles, std::size_t row, std::size_t col)
{
    return (row + col) % sched.num_localities;
}

constexpr std::size_t uncertainty_tile(const tiled_scheduler_sma &sched, std::size_t n_tiles, std::size_t i)
{
    return i % sched.num_localities;
}

constexpr std::size_t inter_alpha_tile(const tiled_scheduler_sma &sched, std::size_t n_tiles, std::size_t i)
{
    return i % sched.num_localities;
}

constexpr std::size_t diag_tile(const tiled_scheduler_sma &sched, std::size_t n_tiles, std::size_t i) {
    return i % sched.num_localities;
  }

constexpr std::size_t cholesky_potrf(const tiled_scheduler_sma &sched, std::size_t n_tiles, std::size_t k)
{
    return (2 * k) % sched.num_localities;
}

constexpr std::size_t cholesky_syrk(const tiled_scheduler_sma &sched, std::size_t n_tiles, std::size_t m)
{
    return (2 * m) % sched.num_localities;
}

constexpr std::size_t cholesky_trsm(const tiled_scheduler_sma &sched, std::size_t n_tiles, std::size_t k, std::size_t m)
{
    return (k + m) % sched.num_localities;
}

constexpr std::size_t
cholesky_gemm(const tiled_scheduler_sma &sched, std::size_t n_tiles, std::size_t k, std::size_t m, std::size_t n)
{
    return (m + n) % sched.num_localities;
}

constexpr std::size_t solve_trsv(const tiled_scheduler_sma &sched, std::size_t n_tiles, std::size_t k)
{
    return (2 * k) % sched.num_localities;
}

constexpr std::size_t solve_trsm(const tiled_scheduler_sma &sched, std::size_t n_tiles, std::size_t k)
{
    return (2 * k) % sched.num_localities;
}

constexpr std::size_t solve_gemv(const tiled_scheduler_sma &sched, std::size_t n_tiles, std::size_t k, std::size_t m)
{
    return (k + m) % sched.num_localities;
}

constexpr std::size_t
solve_matrix_trsm(const tiled_scheduler_sma &sched, std::size_t n_tiles, std::size_t c, std::size_t k)
{
    return (k + c) % sched.num_localities;
}

constexpr std::size_t
solve_matrix_gemm(const tiled_scheduler_sma &sched, std::size_t n_tiles, std::size_t c, std::size_t k, std::size_t m)
{
    return (k + m) % sched.num_localities;
}

constexpr std::size_t multiply_gemv(const tiled_scheduler_sma &sched, std::size_t n_tiles, std::size_t k, std::size_t m)
{
    return (k + m) % sched.num_localities;
}

constexpr std::size_t k_rank_dot_diag_syrk(const tiled_scheduler_sma &sched, std::size_t n_tiles, std::size_t k)
{
    return (2 * k) % sched.num_localities;
}

constexpr std::size_t
k_rank_gemm(const tiled_scheduler_sma &sched, std::size_t n_tiles, std::size_t c, std::size_t k, std::size_t m)
{
    return (k + m) % sched.num_localities;
}

constexpr std::size_t vector_axpy(const tiled_scheduler_sma &sched, std::size_t n_tiles, std::size_t k)
{
    return (2 * k) % sched.num_localities;
}

constexpr std::size_t get_diagonal(const tiled_scheduler_sma &sched, std::size_t n_tiles, std::size_t k)
{
    return (2 * k) % sched.num_localities;
}

constexpr std::size_t compute_loss(const tiled_scheduler_sma &sched, std::size_t n_tiles, std::size_t k)
{
    return (2 * k) % sched.num_localities;
}

// ==========================

constexpr std::size_t
covariance_tile(const tiled_scheduler_cyclic &sched, std::size_t n_tiles, std::size_t row, std::size_t col)
{
    return (row * n_tiles + col) % sched.num_localities;
}

constexpr std::size_t
cross_covariance_tile(const tiled_scheduler_cyclic &sched, std::size_t n_tiles, std::size_t row, std::size_t col)
{
    return 0;
}

constexpr std::size_t alpha_tile(const tiled_scheduler_cyclic &sched, std::size_t n_tiles, std::size_t i) { return 0; }

constexpr std::size_t prediction_tile(const tiled_scheduler_cyclic &sched, std::size_t n_tiles, std::size_t i)
{
    return 0;
}

constexpr std::size_t cholesky_potrf(const tiled_scheduler_cyclic &sched, std::size_t n_tiles, std::size_t k)
{
    return (k * n_tiles + k) % sched.num_localities;
}

constexpr std::size_t cholesky_syrk(const tiled_scheduler_cyclic &sched, std::size_t n_tiles, std::size_t m)
{
    return (m * n_tiles + m) % sched.num_localities;
}

constexpr std::size_t
cholesky_trsm(const tiled_scheduler_cyclic &sched, std::size_t n_tiles, std::size_t k, std::size_t m)
{
    return (m * n_tiles + k) % sched.num_localities;
}

constexpr std::size_t
cholesky_gemm(const tiled_scheduler_cyclic &sched, std::size_t n_tiles, std::size_t k, std::size_t m, std::size_t n)
{
    return (m * n_tiles + n) % sched.num_localities;
}

constexpr std::size_t solve_trsv(const tiled_scheduler_cyclic &sched, std::size_t n_tiles, std::size_t k) { return 0; }

constexpr std::size_t solve_trsm(const tiled_scheduler_cyclic &sched, std::size_t n_tiles, std::size_t k) { return 0; }

constexpr std::size_t solve_gemv(const tiled_scheduler_cyclic &sched, std::size_t n_tiles, std::size_t k, std::size_t m)
{
    return 0;
}

constexpr std::size_t
solve_matrix_trsm(const tiled_scheduler_cyclic &sched, std::size_t n_tiles, std::size_t c, std::size_t k)
{
    return 0;
}

constexpr std::size_t
solve_matrix_gemm(const tiled_scheduler_cyclic &sched, std::size_t n_tiles, std::size_t c, std::size_t k, std::size_t m)
{
    return 0;
}

constexpr std::size_t
multiply_gemv(const tiled_scheduler_cyclic &sched, std::size_t n_tiles, std::size_t k, std::size_t m)
{
    return 0;
}

constexpr std::size_t k_rank_dot_diag_syrk(const tiled_scheduler_cyclic &sched, std::size_t n_tiles, std::size_t k)
{
    return 0;
}

constexpr std::size_t
k_rank_gemm(const tiled_scheduler_cyclic &sched, std::size_t n_tiles, std::size_t c, std::size_t k, std::size_t m)
{
    return 0;
}

constexpr std::size_t vector_axpy(const tiled_scheduler_cyclic &sched, std::size_t n_tiles, std::size_t k) { return 0; }

constexpr std::size_t get_diagonal(const tiled_scheduler_cyclic &sched, std::size_t n_tiles, std::size_t k)
{
    return 0;
}

constexpr std::size_t compute_loss(const tiled_scheduler_cyclic &sched, std::size_t n_tiles, std::size_t k)
{
    return 0;
}

}  // namespace schedule

GPRAT_NS_END
