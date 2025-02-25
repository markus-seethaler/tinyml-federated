#ifndef HYPER_PARAMETER_OPTIMIZER_H
#define HYPER_PARAMETER_OPTIMIZER_H

#include <vector>
#include <string>
#include <memory>
#include <limits>
#include "DataPreprocessor/DataPreprocessor.h"
#include "FederatedClient/FederatedClient.h"

struct HyperParams {
    std::vector<size_t> topology;
    float learning_rate;
    size_t samples_per_round;
    float client_fraction;

    // For tracking results
    int rounds_to_success;
    float final_accuracy;
    float final_loss;

    std::string to_string() const;
};

class SuccessTracker {
public:
    static constexpr size_t REQUIRED_CONSECUTIVE_ROUNDS = 20;
    static constexpr float ACCURACY_THRESHOLD = 0.90f;
    static constexpr float LOSS_THRESHOLD = 0.3f;

    void reset();
    bool update(int current_round, float accuracy, float loss);
    int get_rounds_to_success() const;

private:
    size_t accuracy_streak = 0;
    size_t loss_streak = 0;
    int rounds_to_success = std::numeric_limits<int>::max();
};

class HyperParameterOptimizer {
public:
    HyperParameterOptimizer(const std::string& data_path = "../data", 
                           uint32_t seed = 42);
    
    // Run the optimization process
    std::vector<HyperParams> run_optimization();
    
    // Set parameters for optimization
    void set_max_rounds(int max_rounds) { max_fl_rounds = max_rounds; }
    void set_num_clients(size_t num_clients) { num_clients = num_clients; }
    void set_quick_search(bool quick) { quick_search = quick; }
    
private:
    // Generate grid of parameter combinations to test
    std::vector<HyperParams> generate_param_grid();
    
    // Evaluate a single configuration
    bool evaluate_configuration(HyperParams& params, const std::string& metrics_file = "hyperparam_metrics.csv");
    
    // Helper struct for tracking metrics during training
    struct TrainingMetrics {
        std::vector<std::vector<float>> predictions;
        std::vector<std::vector<float>> targets;
    };
    
    // Helper method for training clients
    TrainingMetrics train_clients_online(
        const std::vector<size_t>& selected_clients,
        std::vector<std::unique_ptr<FederatedClient>>& clients,
        std::shared_ptr<DataPreprocessor> preprocessor,
        float learning_rate,
        size_t samples_per_client);
    
    // Member variables
    std::string data_path;
    uint32_t seed;
    int max_fl_rounds = 600;
    size_t num_clients = 100;
    bool quick_search = false;
};

#endif // HYPER_PARAMETER_OPTIMIZER_H