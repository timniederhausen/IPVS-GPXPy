#ifndef GPRAT_GPKERNELS_HPP
#define GPRAT_GPKERNELS_HPP

#pragma once

#include "gprat/detail/config.hpp"

#include <cstddef>
#include <vector>

GPRAT_NS_BEGIN

/**
 * @brief Squared Exponential Kernel Parameters
 */
struct SEKParams
{
    /**
     * @brief Lengthscale: variance of training output
     *
     * Sometimes denoted with index 0.
     */
    double lengthscale;

    /**
     * @brief Vertical Lengthscale: standard deviation of training input
     *
     * Sometimes denoted with index 1.
     */
    double vertical_lengthscale;

    /**
     * @brief Noise Variance: small value
     *
     * Sometimes denoted with index 2.
     */
    double noise_variance;

    std::vector<double> m_T;

    std::vector<double> w_T;

    /**
     * @brief Construct a new SEKParams object
     *
     * @param lengthscale Lengthscale: variance of training output
     * @param vertical_lengthscale Vertical Lengthscale: standard deviation
     * of training input
     * @param noise_variance Noise Variance: small value
     */
    SEKParams(double lengthscale_, double vertical_lengthscale_, double noise_variance_);

    /**
     * @brief Return the number of parameters
     */
    std::size_t size();

    /**
     * @brief Sets the parameter at the given index
     *
     * Index 0: lengthscale
     * Index 1: vertical_lengthscale
     * Index 2: noise_variance
     *
     * @param index Index of the parameter to set
     * @param value Value to set
     */
    void set_param(std::size_t index, double value);

    /**
     * @brief Returns the parameter at the given index
     *
     * Index 0: lengthscale
     * Index 1: vertical_lengthscale
     * Index 2: noise_variance
     *
     * @param index Index of the parameter to get
     */
    const double &get_param(std::size_t index) const;
};

GPRAT_NS_END

#endif
