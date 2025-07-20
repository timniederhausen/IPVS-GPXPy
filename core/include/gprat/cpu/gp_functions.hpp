#ifndef GPRAT_CPU_GP_FUNCTIONS_HPP
#define GPRAT_CPU_GP_FUNCTIONS_HPP

#pragma once

#include "gprat/detail/config.hpp"
#include "gprat/hyperparameters.hpp"
#include "gprat/kernels.hpp"

#include <vector>

GPRAT_NS_BEGIN

namespace cpu
{

/**
 * @brief Perform Cholesky decompositon (+Assebmly)
 *
 * @param training_input The training input data
 * @param hyperparameters The kernel hyperparameters
 *
 * @param n_tiles The number of training tiles
 * @param n_tile_size The size of each training tile
 * @param n_regressors The number of regressors
 *
 * @return The tiled Cholesky factor
 */
std::vector<std::vector<double>>
cholesky(const std::vector<double> &training_input,
         const SEKParams &sek_params,
         int n_tiles,
         int n_tile_size,
         int n_regressors);

/**
 * @brief Compute the predictions without uncertainties.
 *
 * @param training_input The training input data
 * @param training_output The raining output data
 * @param test_input The test input data
 * @param hyperparameters The kernel hyperparameters
 * @param n_tiles The number of training tiles
 * @param n_tile_size The size of each training tile
 * @param m_tiles The number of test tiles
 * @param m_tile_size The size of each test tile
 * @param n_regressors The number of regressors
 *
 * @return A vector containing the predictions
 */
std::vector<double>
predict(const std::vector<double> &training_input,
        const std::vector<double> &training_output,
        const std::vector<double> &test_input,
        const SEKParams &sek_params,
        int n_tiles,
        int n_tile_size,
        int m_tiles,
        int m_tile_size,
        int n_regressors);

/**
 * @brief Compute the predictions with uncertainties.
 *
 * @param training_input The training input data
 * @param training_output The raining output data
 * @param test_input The test input data
 * @param hyperparameters The kernel hyperparameters
 * @param n_tiles The number of training tiles
 * @param n_tile_size The size of each training tile
 * @param m_tiles The number of test tiles
 * @param m_tile_size The size of each test tile
 * @param n_regressors The number of regressors
 *
 * @return A vector containing the prediction vector and the uncertainty vector
 */
std::vector<std::vector<double>> predict_with_uncertainty(
    const std::vector<double> &training_input,
    const std::vector<double> &training_output,
    const std::vector<double> &test_input,
    const SEKParams &sek_params,
    int n_tiles,
    int n_tile_size,
    int m_tiles,
    int m_tile_size,
    int n_regressors);

/**
 * @brief Compute the predictions with full covariance matrix.
 *
 * @param training_input The training input data
 * @param training_output The raining output data
 * @param test_input The test input data
 * @param hyperparameters The kernel hyperparameters
 * @param n_tiles The number of training tiles
 * @param n_tile_size The size of each training tile
 * @param m_tiles The number of test tiles
 * @param m_tile_size The size of each test tile
 * @param n_regressors The number of regressors
 *
 * @return A vector containing the prediction vector and the full posterior covariance matrix
 */
std::vector<std::vector<double>> predict_with_full_cov(
    const std::vector<double> &training_input,
    const std::vector<double> &training_output,
    const std::vector<double> &test_data,
    const SEKParams &sek_params,
    int n_tiles,
    int n_tile_size,
    int m_tiles,
    int m_tile_size,
    int n_regressors);

/**
 * @brief Compute loss for given data and Gaussian process model
 *
 * @param training_input The training input data
 * @param training_output The raining output data
 * @param hyperparameters The kernel hyperparameters
 * @param n_tiles The number of training tiles
 * @param n_tile_size The size of each training tile
 * @param n_regressors The number of regressors
 *
 * @return The loss
 */
double compute_loss(const std::vector<double> &training_input,
                    const std::vector<double> &training_output,
                    const SEKParams &sek_params,
                    int n_tiles,
                    int n_tile_size,
                    int n_regressors);

/**
 * @brief Perform optimization for a given number of iterations
 *
 * @param training_input The training input data
 * @param training_output The raining output data
 *
 * @param n_tiles The number of training tiles
 * @param n_tile_size The size of each training tile
 * @param n_regressors The number of regressors
 *
 * @param hyperparams The Adam optimizer hyperparameters
 * @param hyperparameters The kernel hyperparameters
 * @param trainable_params The vector containing a bool wheather to train a hyperparameter
 *
 * @return A vector containing the loss values of each iteration
 */
std::vector<double>
optimize(const std::vector<double> &training_input,
         const std::vector<double> &training_output,
         int n_tiles,
         int n_tile_size,
         int n_regressors,
         const AdamParams &adam_params,
         SEKParams &sek_params,
         std::vector<bool> trainable_params);

/**
 * @brief Perform a single optimization step
 *
 * @param training_input The training input data
 * @param training_output The raining output data
 *
 * @param n_tiles The number of training tiles
 * @param n_tile_size The size of each training tile
 * @param n_regressors The number of regressors
 *
 * @param hyperparams The Adam optimizer hyperparameters
 * @param hyperparameters The kernel hyperparameters
 * @param trainable_params The vector containing a bool wheather to train a hyperparameter
 *
 * @param iter The current optimization iteration
 *
 * @return The loss value
 */
double optimize_step(const std::vector<double> &training_input,
                     const std::vector<double> &training_output,
                     int n_tiles,
                     int n_tile_size,
                     int n_regressors,
                     AdamParams &adam_params,
                     SEKParams &sek_params,
                     std::vector<bool> trainable_params,
                     int iter);

}  // end of namespace cpu

GPRAT_NS_END

#endif
