#include "gprat_c.hpp"
#include "utils_c.hpp"
#include <chrono>
#include <fstream>
#include <iostream>

int main(int argc, char *argv[])
{
    namespace po = hpx::program_options;
    po::options_description desc("Allowed options");
    // clang-format off
    desc.add_options()
        ("help", "produce help message")
        ("train_x_path", po::value<std::string>()->default_value("../../../data/data_1024/training_input.txt"), "training data (x)")
        ("train_y_path", po::value<std::string>()->default_value("../../../data/data_1024/training_output.txt"), "training data (y)")
        ("test_path", po::value<std::string>()->default_value("../../../data/data_1024/test_input.txt"), "test data")
        ("tiles", po::value<std::size_t>()->default_value(16), "tiles per dimension")
        ("regressors", po::value<std::size_t>()->default_value(8), "num regressors")
        ("start-cores", po::value<std::size_t>()->default_value(2), "num CPUs to start with")
        ("end-cores", po::value<std::size_t>()->default_value(4), "num CPUs to end with")
        ("start", po::value<std::size_t>()->default_value(512), "Starting number of training samples")
        ("end", po::value<std::size_t>()->default_value(1024), "End number of training samples")
        ("step", po::value<std::size_t>()->default_value(2), "Increment of training samples")
        ("loop", po::value<std::size_t>()->default_value(2), "Number of iterations to be performed for each number of training samples")
        ("opt_iter", po::value<int>()->default_value(1), "Number of optimization iterations*/")
    ;
    // clang-format on

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.contains("help"))
    {
        std::cout << desc << "\n";
        return 1;
    }

    /////////////////////
    /////// configuration
    std::size_t START = vm["start"].as<std::size_t>();
    std::size_t END = vm["end"].as<std::size_t>();
    std::size_t STEP = vm["step"].as<std::size_t>();
    std::size_t LOOP = vm["loop"].as<std::size_t>();
    const int OPT_ITER = vm["opt_iter"].as<int>();

    int n_test = 1024;
    const std::size_t N_CORES = vm["end-cores"].as<std::size_t>();
    const std::size_t n_tiles = vm["tiles"].as<std::size_t>();
    const std::size_t n_reg = vm["regressors"].as<std::size_t>();

    std::string train_path = vm["train_x_path"].as<std::string>();
    std::string out_path = vm["train_y_path"].as<std::string>();
    std::string test_path = vm["test_path"].as<std::string>();

    bool use_gpu =
        utils::compiled_with_cuda() && gprat::gpu_count() > 0 && argc > 1 && std::strcmp(argv[1], "--use_gpu") == 0;

    for (std::size_t core = vm["start-cores"].as<std::size_t>(); core <= N_CORES; core = core * 2)
    {
        // Create new argc and argv to include the --hpx:threads argument
        std::vector<std::string> args(argv, argv + argc);
        if (use_gpu)
        {
            args.erase(args.begin() + 1);
        }
        args.push_back("--hpx:threads=" + std::to_string(core));

        // Convert the arguments to char* array
        std::vector<char *> cstr_args;
        for (auto &arg : args)
        {
            cstr_args.push_back(const_cast<char *>(arg.c_str()));
        }

        int new_argc = static_cast<int>(cstr_args.size());
        char **new_argv = cstr_args.data();

        for (std::size_t start = START; start <= END; start = start * STEP)
        {
            int n_train = static_cast<int>(start);
            for (std::size_t l = 0; l < LOOP; l++)
            {
                // Compute tile sizes and number of predict tiles
                int tile_size = utils::compute_train_tile_size(n_train, n_tiles);
                auto result = utils::compute_test_tiles(n_test, n_tiles, tile_size);
                /////////////////////
                ///// hyperparams
                gprat_hyper::AdamParams hpar = { 0.1, 0.9, 0.999, 1e-8, OPT_ITER };

                /////////////////////
                ////// data loading
                gprat::GP_data training_input(train_path, n_train, n_reg);
                gprat::GP_data training_output(out_path, n_train, n_reg);
                gprat::GP_data test_input(test_path, n_test, n_reg);

                auto start_total = std::chrono::high_resolution_clock::now();

                std::chrono::duration<double> init_time;
                std::chrono::duration<double> cholesky_time;
                std::chrono::duration<double> opt_time;
                std::chrono::duration<double> pred_uncer_time;
                std::chrono::duration<double> pred_full_cov_time;
                std::chrono::duration<double> pred_time;
                std::vector<bool> trainable = { true, true, true };
                std::string target;

                if (!use_gpu)
                {
                    target = "cpu";

                    /////////////////////
                    ///// GP
                    auto start_init = std::chrono::high_resolution_clock::now();
                    gprat::GP gp_cpu(training_input.data,
                                     training_output.data,
                                     n_tiles,
                                     tile_size,
                                     n_reg,
                                     { 1.0, 1.0, 0.1 },
                                     trainable);
                    auto end_init = std::chrono::high_resolution_clock::now();
                    init_time = end_init - start_init;

                    // Initialize HPX with the new arguments, don't run hpx_main
                    utils::start_hpx_runtime(new_argc, new_argv);

                    // Measure the time taken to execute gp.cholesky();
                    auto start_cholesky = std::chrono::high_resolution_clock::now();
                    std::vector<std::vector<double>> choleksy_cpu = gp_cpu.cholesky();
                    auto end_cholesky = std::chrono::high_resolution_clock::now();
                    cholesky_time = end_cholesky - start_cholesky;

                    // Measure the time taken to execute gp.optimize(hpar);
                    auto start_opt = std::chrono::high_resolution_clock::now();
                    std::vector<double> losses = gp_cpu.optimize(hpar);
                    auto end_opt = std::chrono::high_resolution_clock::now();
                    opt_time = end_opt - start_opt;

                    auto start_pred_uncer = std::chrono::high_resolution_clock::now();
                    std::vector<std::vector<double>> sum_cpu =
                        gp_cpu.predict_with_uncertainty(test_input.data, result.first, result.second);
                    auto end_pred_uncer = std::chrono::high_resolution_clock::now();
                    pred_uncer_time = end_pred_uncer - start_pred_uncer;

                    auto start_pred_full_cov = std::chrono::high_resolution_clock::now();
                    std::vector<std::vector<double>> full_cpu =
                        gp_cpu.predict_with_full_cov(test_input.data, result.first, result.second);
                    auto end_pred_full_cov = std::chrono::high_resolution_clock::now();
                    pred_full_cov_time = end_pred_full_cov - start_pred_full_cov;

                    auto start_pred = std::chrono::high_resolution_clock::now();
                    std::vector<double> pred_cpu = gp_cpu.predict(test_input.data, result.first, result.second);
                    auto end_pred = std::chrono::high_resolution_clock::now();
                    pred_time = end_pred - start_pred;
                }
                else
                {
                    target = "gpu";

                    auto start_init = std::chrono::high_resolution_clock::now();
                    gprat::GP gp_gpu(
                        training_input.data,
                        training_output.data,
                        n_tiles,
                        tile_size,
                        n_reg,
                        { 1.0, 1.0, 0.1 },
                        trainable,
                        0,
                        2);
                    auto end_init = std::chrono::high_resolution_clock::now();
                    init_time = end_init - start_init;

                    // Initialize HPX with the new arguments, don't run hpx_main
                    utils::start_hpx_runtime(new_argc, new_argv);

                    auto start_cholesky = std::chrono::high_resolution_clock::now();
                    std::vector<std::vector<double>> choleksy_gpu = gp_gpu.cholesky();
                    auto end_cholesky = std::chrono::high_resolution_clock::now();
                    cholesky_time = end_cholesky - start_cholesky;

                    // NOTE: optimization is not implemented for GPU
                    opt_time = std::chrono::seconds(-1);

                    auto start_pred_uncer = std::chrono::high_resolution_clock::now();
                    std::vector<std::vector<double>> sum_gpu =
                        gp_gpu.predict_with_uncertainty(test_input.data, result.first, result.second);
                    auto end_pred_uncer = std::chrono::high_resolution_clock::now();
                    pred_uncer_time = end_pred_uncer - start_pred_uncer;

                    auto start_pred_full_cov = std::chrono::high_resolution_clock::now();
                    std::vector<std::vector<double>> full_gpu =
                        gp_gpu.predict_with_full_cov(test_input.data, result.first, result.second);
                    auto end_pred_full_cov = std::chrono::high_resolution_clock::now();
                    pred_full_cov_time = end_pred_full_cov - start_pred_full_cov;

                    auto start_pred = std::chrono::high_resolution_clock::now();
                    std::vector<double> pred_gpu = gp_gpu.predict(test_input.data, result.first, result.second);
                    auto end_pred = std::chrono::high_resolution_clock::now();
                    pred_time = end_pred - start_pred;
                }

                // Stop the HPX runtime
                utils::stop_hpx_runtime();

                auto end_total = std::chrono::high_resolution_clock::now();
                auto total_time = end_total - start_total;

                // Save parameters and times to a .txt file with a header
                std::ofstream outfile("output.csv", std::ios::app);  // Append mode
                if (outfile.tellp() == 0)
                {
                    // If file is empty, write the header
                    outfile << "Target,Cores,N_train,N_test,N_tiles,N_regressor,Opt_iter,Total_time,Init_time,Cholesky_"
                               "time,Opt_time,Pred_Uncer_time,Pred_Full_time,Pred_time,N_loop\n";
                }
                outfile << target << "," << core << "," << n_train << "," << n_test << "," << n_tiles << "," << n_reg
                        << "," << OPT_ITER << "," << total_time.count() << "," << init_time.count() << ","
                        << cholesky_time.count() << "," << opt_time.count() << "," << pred_uncer_time.count() << ","
                        << pred_full_cov_time.count() << "," << pred_time.count() << "," << l << "\n";
                outfile.close();
            }
        }
    }

    return 0;
}
