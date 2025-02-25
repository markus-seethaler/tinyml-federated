#include "HPO/HyperParameterOptimizer.h"
#include "DataLoader/DataLoader.h"
#include "Metrics/Metrics.h"
#include "FederatedServer/FederatedServer.h"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>

std::string HyperParams::to_string() const {
    std::stringstream ss;
    ss << "Topology: [";
    for (size_t i = 0; i < topology.size(); i++) {
        ss << topology[i];
        if (i < topology.size() - 1)
            ss << ", ";
    }
    ss << "], LR: " << learning_rate
       << ", Samples/Round: " << samples_per_round
       << ", Client Fraction: " << client_fraction;
    return ss.str();
}

void SuccessTracker::reset() {
    accuracy_streak = 0;
    loss_streak = 0;
    rounds_to_success = std::numeric_limits<int>::max();
}

bool SuccessTracker::update(int current_round, float accuracy, float loss) {
    // Update accuracy streak
    if (accuracy >= ACCURACY_THRESHOLD) {
        accuracy_streak++;
    } else {
        accuracy_streak = 0;
    }

    // Update loss streak
    if (loss <= LOSS_THRESHOLD) {
        loss_streak++;
    } else {
        loss_streak = 0;
    }

    // Check if both conditions are met
    if (accuracy_streak >= REQUIRED_CONSECUTIVE_ROUNDS &&
        loss_streak >= REQUIRED_CONSECUTIVE_ROUNDS) {
        if (rounds_to_success == std::numeric_limits<int>::max()) {
            rounds_to_success = current_round - REQUIRED_CONSECUTIVE_ROUNDS + 1;
        }
        return true;
    }
    return false;
}

int SuccessTracker::get_rounds_to_success() const {
    return rounds_to_success;
}

HyperParameterOptimizer::HyperParameterOptimizer(const std::string& data_path, uint32_t seed)
    : data_path(data_path), seed(seed) {
}

std::vector<HyperParams> HyperParameterOptimizer::generate_param_grid() {
    std::vector<HyperParams> grid;

    // Define parameter ranges
    std::vector<std::vector<size_t>> topologies;
    std::vector<float> learning_rates;
    std::vector<size_t> samples_per_round;
    std::vector<float> client_fractions;

    if (quick_search) {
        // Reduced parameter space for quick search
        topologies = {
            {11, 15, 3}, {11, 30, 3}, {11, 60, 3}
        };
        learning_rates = {0.3f, 0.75f};
        samples_per_round = {10, 20};
        client_fractions = {0.2f, 0.4f};
    } else {
        // Full parameter space
        topologies = {
            {11, 10, 3}, {11, 15, 3}, {11, 20, 3}, 
            {11, 30, 3}, {11, 60, 3}, {11, 40, 20, 3}
        };
        learning_rates = {0.1f, 0.3f, 0.5f, 0.75f, 1.0f};
        samples_per_round = {5, 10, 15, 20, 25};
        client_fractions = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
    }

    // Create all combinations
    for (const auto& topology : topologies) {
        for (float lr : learning_rates) {
            for (size_t samples : samples_per_round) {
                for (float fraction : client_fractions) {
                    grid.push_back({
                        topology, lr, samples, fraction,
                        std::numeric_limits<int>::max(), // rounds_to_success
                        0.0f,                            // final_accuracy
                        0.0f                             // final_loss
                    });
                }
            }
        }
    }

    return grid;
}

HyperParameterOptimizer::TrainingMetrics HyperParameterOptimizer::train_clients_online(
    const std::vector<size_t>& selected_clients,
    std::vector<std::unique_ptr<FederatedClient>>& clients,
    std::shared_ptr<DataPreprocessor> preprocessor,
    float learning_rate,
    size_t samples_per_client) {

    TrainingMetrics metrics;

    // Train only selected clients
    for (size_t i = 0; i < samples_per_client; i++) {
        // Each selected client gets a different sample
        for (size_t client_idx : selected_clients) {
            TrainingSample sample = preprocessor->get_next_training_sample(client_idx);

            // Get prediction before training
            auto prediction = clients[client_idx]->predict(sample.features);

            // Train on sample
            clients[client_idx]->train_on_sample(sample.features, sample.target, learning_rate);

            // Store predictions and targets
            metrics.predictions.push_back(prediction);
            metrics.targets.push_back(sample.target);
        }
    }

    return metrics;
}

bool HyperParameterOptimizer::evaluate_configuration(
    HyperParams& params,
    const std::string& metrics_file) {

    try {
        // Load dataset
        DataLoader loader(data_path);
        auto dataset = loader.load_dataset("motion_metadata.csv");

        // Prepare data
        auto preprocessor = std::make_shared<DataPreprocessor>(seed);
        preprocessor->prepare_dataset(dataset);

        // Initialize components
        FederatedServer server(seed);
        std::vector<std::unique_ptr<FederatedClient>> clients;

        // Initialize clients with current topology
        for (size_t i = 0; i < num_clients; i++) {
            clients.push_back(std::make_unique<FederatedClient>(
                params.topology, preprocessor, seed + i));
        }

        // Get test set
        auto test_samples = preprocessor->get_test_set();
        if (test_samples.empty()) {
            throw std::runtime_error("No test samples available");
        }

        // Success tracking
        SuccessTracker tracker;

        // Open metrics file
        std::ofstream metrics_file_stream(metrics_file, std::ios::app);
        metrics_file_stream << "Round,Config,Accuracy,TestLoss,TrainingLoss\n";

        // Training loop
        for (int round = 0; round < max_fl_rounds; round++) {
            // Select clients
            auto selected_clients = server.select_clients(
                clients.size(), params.client_fraction);

            // Train selected clients
            auto training_metrics = train_clients_online(
                selected_clients, clients, preprocessor,
                params.learning_rate, params.samples_per_round);

            float training_loss = Metrics::cross_entropy_loss(
                training_metrics.predictions, training_metrics.targets);

            // Average weights
            std::vector<std::vector<float>> client_weights;
            for (size_t client_idx : selected_clients) {
                client_weights.push_back(clients[client_idx]->get_weights());
            }
            auto averaged_weights = server.average_weights(client_weights);

            // Update all clients
            for (auto& client : clients) {
                client->set_weights(averaged_weights);
            }

            // Evaluate
            std::vector<std::vector<float>> test_predictions;
            std::vector<std::vector<float>> test_targets;
            for (const auto& test_sample : test_samples) {
                test_predictions.push_back(
                    clients[0]->predict(test_sample.features));
                test_targets.push_back(test_sample.target);
            }

            float test_loss = Metrics::cross_entropy_loss(
                test_predictions, test_targets);
            float test_accuracy = Metrics::accuracy(
                test_predictions, test_targets);

            // Log metrics
            metrics_file_stream << round << ","
                                << params.to_string() << ","
                                << test_accuracy << ","
                                << test_loss << ","
                                << training_loss << "\n";

            // Update success tracker
            bool success = tracker.update(round, test_accuracy, test_loss);

            // Store final metrics
            params.final_accuracy = test_accuracy;
            params.final_loss = test_loss;

            if (success) {
                params.rounds_to_success = tracker.get_rounds_to_success();
                return true;
            }
        }

        return false;
    }
    catch (const std::exception& e) {
        std::cerr << "Error evaluating configuration: " << e.what() << "\n";
        return false;
    }
}

std::vector<HyperParams> HyperParameterOptimizer::run_optimization() {
    auto param_grid = generate_param_grid();
    std::cout << "Generated " << param_grid.size() << " configurations to test\n";

    std::vector<HyperParams> successful_configs;

    for (auto& params : param_grid) {
        std::cout << "\nTesting configuration:\n"
                  << params.to_string() << "\n";

        if (evaluate_configuration(params)) {
            successful_configs.push_back(params);
            std::cout << "Success! Rounds needed: "
                      << params.rounds_to_success << "\n";
        }
        else {
            std::cout << "Did not meet success criteria\n";
        }
    }

    // Sort successful configurations by rounds to success
    std::sort(successful_configs.begin(), successful_configs.end(),
              [](const HyperParams& a, const HyperParams& b) {
                  return a.rounds_to_success < b.rounds_to_success;
              });

    // Print results
    std::cout << "\n=== Results ===\n";
    std::cout << "Successful configurations: "
              << successful_configs.size() << "/"
              << param_grid.size() << "\n\n";

    if (!successful_configs.empty()) {
        std::cout << "Best configuration:\n"
                  << successful_configs[0].to_string() << "\n"
                  << "Rounds to success: "
                  << successful_configs[0].rounds_to_success << "\n"
                  << "Final accuracy: "
                  << (successful_configs[0].final_accuracy * 100.0f) << "%\n"
                  << "Final loss: "
                  << successful_configs[0].final_loss << "\n";
                  
        // Save best configuration to a JSON file
        std::ofstream best_config_file("best_config.json");
        best_config_file << "{\n";
        best_config_file << "  \"topology\": [";
        
        for (size_t i = 0; i < successful_configs[0].topology.size(); i++) {
            best_config_file << successful_configs[0].topology[i];
            if (i < successful_configs[0].topology.size() - 1) {
                best_config_file << ", ";
            }
        }
        
        best_config_file << "],\n";
        best_config_file << "  \"learning_rate\": " << successful_configs[0].learning_rate << ",\n";
        best_config_file << "  \"samples_per_round\": " << successful_configs[0].samples_per_round << ",\n";
        best_config_file << "  \"client_fraction\": " << successful_configs[0].client_fraction << ",\n";
        best_config_file << "  \"rounds_to_success\": " << successful_configs[0].rounds_to_success << ",\n";
        best_config_file << "  \"final_accuracy\": " << successful_configs[0].final_accuracy << ",\n";
        best_config_file << "  \"final_loss\": " << successful_configs[0].final_loss << "\n";
        best_config_file << "}\n";
    }

    return successful_configs;
}