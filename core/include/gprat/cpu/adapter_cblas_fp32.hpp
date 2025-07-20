#ifndef GPRAT_CPU_ADAPTER_CBLAS_FP32_HPP
#define GPRAT_CPU_ADAPTER_CBLAS_FP32_HPP

#pragma once

#include "gprat/detail/config.hpp"
#include "gprat/tile_data.hpp"

#include <hpx/future.hpp>
#include <vector>

GPRAT_NS_BEGIN

// Constants that are compatible with CBLAS
typedef enum BLAS_TRANSPOSE { Blas_no_trans = 111, Blas_trans = 112 } BLAS_TRANSPOSE;

typedef enum BLAS_SIDE { Blas_left = 141, Blas_right = 142 } BLAS_SIDE;

typedef enum BLAS_ALPHA { Blas_add = 1, Blas_substract = -1 } BLAS_ALPHA;

// typedef enum BLAS_UPLO { Blas_upper = 121,
//                          Blas_lower = 122 } BLAS_UPLO;

// typedef enum BLAS_ORDERING { Blas_row_major = 101,
//                              Blas_col_major = 102 } BLAS_ORDERING;

// BLAS level 3 operations

/**
 * @brief FP32 In-place Cholesky decomposition of A
 * @param A matrix to be factorized
 * @param N matrix dimension
 * @return factorized, lower triangular matrix f_L
 */
mutable_tile_data<float> potrf(const mutable_tile_data<float>& A, int N);

/**
 * @brief FP32 In-place solve L(^T) * X = A or X * L(^T) = A where L lower triangular
 * @param L Cholesky factor matrix
 * @param A right hand side matrix
 * @param N first dimension
 * @param M second dimension
 * @return solution matrix f_X
 */
mutable_tile_data<float> trsm(const const_tile_data<float>& L,
                   const mutable_tile_data<float>& A,
                   int N,
                   int M,
                   BLAS_TRANSPOSE transpose_L,
                   BLAS_SIDE side_L);

/**
 * @brief FP32 Symmetric rank-k update: A = A - B * B^T
 * @param A Base matrix
 * @param B Symmetric update matrix
 * @param N matrix dimension
 * @return updated matrix f_A
 */
mutable_tile_data<float> syrk(const mutable_tile_data<float>& A, const const_tile_data<float>& B, int N);

/**
 * @brief FP32 General matrix-matrix multiplication: C = C - A(^T) * B(^T)
 * @param C Base matrix
 * @param B Right update matrix
 * @param A Left update matrix
 * @param N first matrix dimension
 * @param M second matrix dimension
 * @param K third matrix dimension
 * @param transpose_A transpose left matrix
 * @param transpose_B transpose right matrix
 * @return updated matrix f_X
*/
mutable_tile_data<float>
gemm(const const_tile_data<float>& A,
     const const_tile_data<float>& B,
     const mutable_tile_data<float>& C,
     int N,
     int M,
     int K,
     BLAS_TRANSPOSE transpose_A,
     BLAS_TRANSPOSE transpose_B);

// BLAS level 2 operations

/**
 * @brief FP32 In-place solve L(^T) * x = a where L lower triangular
 * @param L Cholesky factor matrix
 * @param a right hand side vector
 * @param N matrix dimension
 * @param transpose_L transpose Cholesky factor
 * @return solution vector f_x
 */
mutable_tile_data<float> trsv(const const_tile_data<float>& L, const mutable_tile_data<float>& a, int N, BLAS_TRANSPOSE transpose_L);

/**
 * @brief FP32 General matrix-vector multiplication: b = b - A(^T) * a
 * @param f_A update matrix
 * @param f_a update vector
 * @param f_b base vector
 * @param N matrix dimension
 * @param alpha add or substract update to base vector
 * @param transpose_A transpose update matrix
 * @return updated vector f_b
*/
mutable_tile_data<float> gemv(const const_tile_data<float>& A,
                   const const_tile_data<float>& a,
                   const mutable_tile_data<float>& b,
                   int N,
                   int M,
                   BLAS_ALPHA alpha,
                   BLAS_TRANSPOSE transpose_A);

/**
 * @brief FP32 Vector update with diagonal SYRK: r = r + diag(A^T * A)
 * @param A update matrix
 * @param r base vector
 * @param N first matrix dimension
 * @param M second matrix dimension
 * @return updated vector f_r
*/
mutable_tile_data<float> dot_diag_syrk(const const_tile_data<float>& A, const mutable_tile_data<float>& r, int N, int M);

/**
 * @brief FP32 Vector update with diagonal GEMM: r = r + diag(A * B)
 * @param A first update matrix
 * @param B second update matrix
 * @param r base vector
 * @param N first matrix dimension
 * @param M second matrix dimension
 * @return updated vector f_r
 */
mutable_tile_data<float> dot_diag_gemm(const const_tile_data<float>& A, const const_tile_data<float>& B, const mutable_tile_data<float>& r, int N, int M);

// BLAS level 1 operations

/**
 * @brief FP32 AXPY: y - x
 * @param f_y left vector
 * @param f_x right vector
 * @param N vector length
 * @return y - x
 */
mutable_tile_data<float> axpy(const mutable_tile_data<float>& y, const const_tile_data<float>& x, int N);

/**
 * @brief FP32 Dot product: a * b
 * @param f_a left vector
 * @param f_b right vector
 * @param N vector length
 * @return f_a * f_b
 */
float dot(std::span<const float> a, std::span<const float> b, int N);

GPRAT_NS_END

#endif
