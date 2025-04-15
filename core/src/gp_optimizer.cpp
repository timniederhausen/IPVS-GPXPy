#include "../include/gp_optimizer.hpp"

#include "../include/adapter_cblas_fp64.hpp"
#include <numbers>
#include <numeric>

///////////////////////////////////////////////////////////
// Parameter constraints
double to_constrained(double parameter, bool noise)
{
    if (noise)
    {
        return log(1.0 + exp(parameter)) + 1e-6;
    }
    else
    {
        return log(1.0 + exp(parameter));
    }
}

double to_unconstrained(double parameter, bool noise)
{
    if (noise)
    {
        return log(exp(parameter - 1e-6) - 1.0);
    }
    else
    {
        return log(exp(parameter) - 1.0);
    }
}

double compute_sigmoid(double parameter) { return 1.0 / (1.0 + exp(-parameter)); }

/////////////////////////////////////////////////////////
// Tile generation
double compute_covariance_distance(std::size_t i_global,
                                   std::size_t j_global,
                                   std::size_t n_regressors,
                                   const gprat_hyper::SEKParams &sek_params,
                                   const std::vector<double> &i_input,
                                   const std::vector<double> &j_input)
{
    // -0.5*lengthscale^2*(z_i-z_j)^2
    double distance = 0.0;
    double z_ik_minus_z_jk;

    for (std::size_t k = 0; k < n_regressors; k++)
    {
        z_ik_minus_z_jk = i_input[i_global + k] - j_input[j_global + k];
        distance += z_ik_minus_z_jk * z_ik_minus_z_jk;
    }
    return -0.5 / (sek_params.lengthscale * sek_params.lengthscale) * distance;
}

std::vector<double> gen_tile_distance(
    std::size_t row,
    std::size_t col,
    std::size_t N,
    std::size_t n_regressors,
    const gprat_hyper::SEKParams &sek_params,
    const std::vector<double> &input)
{
    std::size_t i_global, j_global;
    // Preallocate memory
    std::vector<double> tile;
    tile.reserve(N * N);
    for (std::size_t i = 0; i < N; i++)
    {
        i_global = N * row + i;
        for (std::size_t j = 0; j < N; j++)
        {
            j_global = N * col + j;
            // compute covariance function
            tile.push_back(compute_covariance_distance(i_global, j_global, n_regressors, sek_params, input, input));
        }
    }
    return tile;
}

std::vector<double> gen_tile_covariance_with_distance(
    std::size_t row,
    std::size_t col,
    std::size_t N,
    const gprat_hyper::SEKParams &sek_params,
    const std::vector<double> &distance)
{
    std::size_t i_global, j_global;
    double covariance;
    // Preallocate required memory
    std::vector<double> tile;
    tile.reserve(N * N);
    for (std::size_t i = 0; i < N; i++)
    {
        i_global = N * row + i;
        for (std::size_t j = 0; j < N; j++)
        {
            j_global = N * col + j;
            // compute covariance function
            covariance = sek_params.vertical_lengthscale * exp(distance[i * N + j]);
            if (i_global == j_global)
            {
                // noise variance on diagonal
                covariance += sek_params.noise_variance;
            }
            tile.push_back(covariance);
        }
    }
    return tile;
}

std::vector<double>
gen_tile_grad_v(std::size_t N, const gprat_hyper::SEKParams &sek_params, const std::vector<double> &distance)
{
    // Preallocate required memory
    std::vector<double> tile;
    tile.reserve(N * N);
    double hyperparam_der = compute_sigmoid(to_unconstrained(sek_params.vertical_lengthscale, false));
    for (std::size_t i = 0; i < N; i++)
    {
        for (std::size_t j = 0; j < N; j++)
        {
            // compute derivative
            tile.push_back(exp(distance[i * N + j]) * hyperparam_der);
        }
    }
    return tile;
}

std::vector<double>
gen_tile_grad_l(std::size_t N, const gprat_hyper::SEKParams &sek_params, const std::vector<double> &distance)
{
    // Preallocate required memory
    std::vector<double> tile;
    tile.reserve(N * N);
    double hyperparam_der = compute_sigmoid(to_unconstrained(sek_params.lengthscale, false));
    double factor = -2.0 * sek_params.vertical_lengthscale / sek_params.lengthscale;
    for (std::size_t i = 0; i < N; i++)
    {
        for (std::size_t j = 0; j < N; j++)
        {
            // compute derivative
            tile.push_back(factor * distance[i * N + j] * exp(distance[i * N + j]) * hyperparam_der);
        }
    }
    return tile;
}

/////////////////////////////////////////////////////////////////////////
// Adam
double update_first_moment(double gradient, double m_T, double beta_1)
{
    return beta_1 * m_T + (1.0 - beta_1) * gradient;
}

double update_second_moment(double gradient, double v_T, double beta_2)
{
    return beta_2 * v_T + (1.0 - beta_2) * gradient * gradient;
}

double adam_step(const double unconstrained_hyperparam,
                 const gprat_hyper::AdamParams &adam_params,
                 double m_T,
                 double v_T,
                 std::size_t iter)
{
    // Compute decay rate
    double beta1_T = pow(adam_params.beta1, static_cast<double>(iter + 1));
    double beta2_T = pow(adam_params.beta2, static_cast<double>(iter + 1));

    // Option 1:
    // double mhat = m_T / (1.0 - beta1_T[iter]);
    // double vhat = v_T / (1.0 - beta2_T[iter]);
    // return unconstrained_hyperparam - adam_params.learning_rate * mhat / (sqrt(vhat)
    // + adam_params.epsilon);

    // Option 2:
    double nu_T = adam_params.learning_rate * sqrt(1.0 - beta2_T) / (1.0 - beta1_T);
    return unconstrained_hyperparam - nu_T * m_T / (sqrt(v_T) + adam_params.epsilon);
}

/////////////////////////////////////////////////////////////////////////
// Loss
double compute_loss(const std::vector<double> &K_diag_tile,
                    const std::vector<double> &alpha_tile,
                    const std::vector<double> &y_tile,
                    std::size_t N)
{
    // l = y^T * alpha + \sum_i^N log(L_ii^2)
    double l;
    // Compute y^T * alpha
    l = dot(y_tile, alpha_tile, static_cast<int>(N));
    // Compute \sum_i^N log(L_ii^2)
    for (std::size_t i = 0; i < N; i++)
    {
        double diag_value = K_diag_tile[i * N + i];
        l += log(diag_value * diag_value);
    }
    return l;
}

double add_losses(const std::vector<double> &losses, std::size_t N, std::size_t n_tiles)
{
    // 0.5 * \sum losses + const
    double l = 0.0;
    double Nn = static_cast<double>(N * n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        // Add the squared difference to the error
        l += losses[i];
    }

    l += Nn * log(2.0 * std::numbers::pi);
    return 0.5 * l / Nn;  // why /Nn?
}

/////////////////////////////////////////////////////////////////////////
// Gradient
double compute_gradient(double trace, double dot, std::size_t N, std::size_t n_tiles)
{
    return 0.5 / static_cast<double>(N * n_tiles) * (trace - dot);
}

double compute_trace(const std::vector<double> &diagonal, double trace)
{
    return trace + std::reduce(diagonal.begin(), diagonal.end());
}

double compute_dot(const std::vector<double> &vector_T, const std::vector<double> &vector, double result)
{
    return result + dot(vector_T, vector, static_cast<int>(vector.size()));
}

double compute_trace_diag(const std::vector<double> &tile, double trace, std::size_t N)
{
    double local_trace = 0.0;
    for (std::size_t i = 0; i < N; ++i)
    {
        local_trace += tile[i * N + i];
    }
    return trace + local_trace;
}
