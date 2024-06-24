#include <gaussian_process>
#include <iostream>
#include <chrono>
#include <fstream>
// #include <boost/program_options.hpp>

int main(int argc, char *argv[])
{
    /////////////////////
    /////// configuration
    int START = 100;
    int END = 300;
    int STEP = 100;
    int LOOP = 10;
    const int OPT_ITER = 3;

    int n_test = 700;
    const int N_CORES = 2; // Set this to the number of threads
    const int n_tiles = 10;
    const int n_reg = 100;

    std::string train_path = "/home/maksim/simtech/thesis/GPPPy_hpx/src/data/training/training_input.txt";
    std::string out_path = "/home/maksim/simtech/thesis/GPPPy_hpx/src/data/training/training_output.txt";
    std::string test_path = "/home/maksim/simtech/thesis/GPPPy_hpx/src/data/test/test_input.txt";

    for (std::size_t core = 1; core <= pow(2, N_CORES); core = core * 2)
    {
        // Create new argc and argv to include the --hpx:threads argument
        std::vector<std::string> args(argv, argv + argc);
        args.push_back("--hpx:threads=" + std::to_string(core));

        // Convert the arguments to char* array
        std::vector<char *> cstr_args;
        for (auto &arg : args)
        {
            cstr_args.push_back(const_cast<char *>(arg.c_str()));
        }

        int new_argc = static_cast<int>(cstr_args.size());
        char **new_argv = cstr_args.data();
        
        for (std::size_t start = START; start <= END; start = start + STEP)
        {
            int n_train = start;
            for (std::size_t l = 0; l < 10; l++)
            {
                auto start_total = std::chrono::high_resolution_clock::now();

                // Compute tile sizes and number of predict tiles
                int tile_size = utils::compute_train_tile_size(n_train, n_tiles);
                auto result = utils::compute_test_tiles(n_test, n_tiles, tile_size);
                /////////////////////
                ///// hyperparams
                std::vector<double> M = {0.0, 0.0, 0.0};
                gpppy_hyper::Hyperparameters hpar = {0.1, 0.9, 0.999, 1e-8, OPT_ITER, M};

                /////////////////////
                ////// data loading
                gpppy::GP_data training_input(train_path, n_train);
                gpppy::GP_data training_output(out_path, n_train);
                gpppy::GP_data test_input(test_path, n_test);

                /////////////////////
                ///// GP
                auto start_init = std::chrono::high_resolution_clock::now();
                std::vector<bool> trainable = {false, false, true};
                gpppy::GP gp(training_input.data, training_output.data, n_tiles, tile_size, 1.0, 1.0, 0.1, n_reg, trainable);
                auto end_init = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> init_time = end_init - start_init;

                // Initialize HPX with the new arguments, don't run hpx_main
                utils::start_hpx_runtime(new_argc, new_argv);

                // Measure the time taken to execute gp.optimize(hpar);
                auto start_opt = std::chrono::high_resolution_clock::now();
                std::vector<double> losses = gp.optimize(hpar);
                auto end_opt = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> opt_time = end_opt - start_opt;

                auto start_pred = std::chrono::high_resolution_clock::now();
                std::vector<std::vector<double>> sum = gp.predict(test_input.data, result.first, result.second);
                auto end_pred = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> pred_time = end_pred - start_pred;

                // Stop the HPX runtime
                utils::stop_hpx_runtime();

                auto end_total = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> total_time = end_total - start_total;

                // Save parameters and times to a .txt file with a header
                std::ofstream outfile("results.txt", std::ios::app); // Append mode
                if (outfile.tellp() == 0)
                {
                    // If file is empty, write the header
                    outfile << "Cores,N_train,N_test,N_tiles,N_regressor,Opt_iter,Total_time,Init_time,Opt_time, Predict_time, N_loop\n";
                }
                outfile << core << "," << n_train << "," << n_test << "," << n_tiles << "," << n_reg << ","
                        << OPT_ITER << "," << total_time.count() << "," << init_time.count() << "," << opt_time.count() << ","
                        << pred_time.count() << "," << l << "\n";
                outfile.close();
            }
        }
    }
    return 0;
}
