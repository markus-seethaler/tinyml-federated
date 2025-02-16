#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <limits>
#include <Metrics/Metrics.h>
#include <DataLoader/DataLoader.h>
#include <DataPreprocessor/DataPreprocessor.h>
#include <FederatedServer/FederatedServer.h>
#include <FederatedClient/FederatedClient.h>

struct TrainingMetrics
{
    std::vector<std::vector<float>> predictions;
    std::vector<std::vector<float>> targets;
};

struct HyperParams
{
    std::vector<size_t> topology;
    float learning_rate;
    size_t samples_per_round;
    float client_fraction;

    // For tracking results
    int rounds_to_success;
    float final_accuracy;
    float final_loss;

    std::string to_string() const
    {
        std::stringstream ss;
        ss << "Topology: [";
        for (size_t i = 0; i < topology.size(); i++)
        {
            ss << topology[i];
            if (i < topology.size() - 1)
                ss << ", ";
        }
        ss << "], LR: " << learning_rate
           << ", Samples/Round: " << samples_per_round
           << ", Client Fraction: " << client_fraction;
        return ss.str();
    }
};

class SuccessTracker
{
public:
    static constexpr size_t REQUIRED_CONSECUTIVE_ROUNDS = 20;
    static constexpr float ACCURACY_THRESHOLD = 0.90f;
    static constexpr float LOSS_THRESHOLD = 0.3f;

    void reset()
    {
        accuracy_streak = 0;
        loss_streak = 0;
        rounds_to_success = std::numeric_limits<int>::max();
    }

    bool update(int current_round, float accuracy, float loss)
    {
        // Update accuracy streak
        if (accuracy >= ACCURACY_THRESHOLD)
        {
            accuracy_streak++;
        }
        else
        {
            accuracy_streak = 0;
        }

        // Update loss streak
        if (loss <= LOSS_THRESHOLD)
        {
            loss_streak++;
        }
        else
        {
            loss_streak = 0;
        }

        // Check if both conditions are met
        if (accuracy_streak >= REQUIRED_CONSECUTIVE_ROUNDS &&
            loss_streak >= REQUIRED_CONSECUTIVE_ROUNDS)
        {
            if (rounds_to_success == std::numeric_limits<int>::max())
            {
                rounds_to_success = current_round - REQUIRED_CONSECUTIVE_ROUNDS + 1;
            }
            return true;
        }
        return false;
    }

    int get_rounds_to_success() const
    {
        return rounds_to_success;
    }

private:
    size_t accuracy_streak = 0;
    size_t loss_streak = 0;
    int rounds_to_success = std::numeric_limits<int>::max();
};

class HyperParamTuner
{
public:
    static std::vector<HyperParams> generate_param_grid()
    {
        std::vector<HyperParams> grid;

        // Define parameter ranges
        std::vector<std::vector<size_t>> topologies = {
            {11, 10, 3}, {11, 15, 3}, {11, 20, 3}, {11, 30, 3}, {11, 60, 3}};

        std::vector<float> learning_rates = {0.3f, 0.5f, 0.75f};
        std::vector<size_t> samples_per_round = {5, 10, 15, 20};
        std::vector<float> client_fractions = {0.1, 0.2f, 0.3f, 0.4f};

        // Create all combinations
        for (const auto &topology : topologies)
        {
            for (float lr : learning_rates)
            {
                for (size_t samples : samples_per_round)
                {
                    for (float fraction : client_fractions)
                    {
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

private:
    static TrainingMetrics train_clients_online(
        const std::vector<size_t> &selected_clients,
        std::vector<std::unique_ptr<FederatedClient>> &clients,
        std::shared_ptr<DataPreprocessor> preprocessor,
        float learning_rate,
        size_t samples_per_client)
    {

        TrainingMetrics metrics;

        // Train only selected clients
        for (size_t i = 0; i < samples_per_client; i++)
        {
            // Each selected client gets a different sample
            for (size_t client_idx : selected_clients)
            {
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

public:
    static bool evaluate_configuration(
        HyperParams &params,
        const std::string &metrics_file = "hyperparam_metrics.csv")
    {

        try
        {
            // Load dataset
            DataLoader loader("../data");
            auto dataset = loader.load_dataset("motion_metadata.csv");

            // Prepare data
            auto preprocessor = std::make_shared<DataPreprocessor>(42);
            preprocessor->prepare_dataset(dataset);

            // Initialize components
            const size_t NUM_CLIENTS = 100;
            const int MAX_FL_ROUNDS = 600; // Maximum rounds before giving up

            FederatedServer server(42);
            std::vector<std::unique_ptr<FederatedClient>> clients;

            // Initialize clients with current topology
            for (size_t i = 0; i < NUM_CLIENTS; i++)
            {
                clients.push_back(std::make_unique<FederatedClient>(
                    params.topology, preprocessor, 42));
            }

            // Get test set
            auto test_samples = preprocessor->get_test_set();
            if (test_samples.empty())
            {
                throw std::runtime_error("No test samples available");
            }

            // Success tracking
            SuccessTracker tracker;

            // Open metrics file
            std::ofstream metrics_file_stream(metrics_file, std::ios::app);
            metrics_file_stream << "Round,Config,Accuracy,TestLoss,TrainingLoss\n";

            // Training loop
            for (int round = 0; round < MAX_FL_ROUNDS; round++)
            {
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
                for (size_t client_idx : selected_clients)
                {
                    client_weights.push_back(clients[client_idx]->get_weights());
                }
                auto averaged_weights = server.average_weights(client_weights);

                // Update all clients
                for (auto &client : clients)
                {
                    client->set_weights(averaged_weights);
                }

                // Evaluate
                std::vector<std::vector<float>> test_predictions;
                std::vector<std::vector<float>> test_targets;
                for (const auto &test_sample : test_samples)
                {
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
                bool success = tracker.update(round, test_accuracy, training_loss);

                // Store final metrics
                params.final_accuracy = test_accuracy;
                params.final_loss = training_loss;

                if (success)
                {
                    params.rounds_to_success = tracker.get_rounds_to_success();
                    return true;
                }
            }

            return false;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error evaluating configuration: " << e.what() << "\n";
            return false;
        }
    }
};

int mainHPO()
{
    auto param_grid = HyperParamTuner::generate_param_grid();
    std::cout << "Generated " << param_grid.size() << " configurations to test\n";

    std::vector<HyperParams> successful_configs;

    for (auto &params : param_grid)
    {
        std::cout << "\nTesting configuration:\n"
                  << params.to_string() << "\n";

        if (HyperParamTuner::evaluate_configuration(params))
        {
            successful_configs.push_back(params);
            std::cout << "Success! Rounds needed: "
                      << params.rounds_to_success << "\n";
        }
        else
        {
            std::cout << "Did not meet success criteria\n";
        }
    }

    // Sort successful configurations by rounds to success
    std::sort(successful_configs.begin(), successful_configs.end(),
              [](const HyperParams &a, const HyperParams &b)
              {
                  return a.rounds_to_success < b.rounds_to_success;
              });

    // Print results
    std::cout << "\n=== Results ===\n";
    std::cout << "Successful configurations: "
              << successful_configs.size() << "/"
              << param_grid.size() << "\n\n";

    if (!successful_configs.empty())
    {
        std::cout << "Best configuration:\n"
                  << successful_configs[0].to_string() << "\n"
                  << "Rounds to success: "
                  << successful_configs[0].rounds_to_success << "\n"
                  << "Final accuracy: "
                  << (successful_configs[0].final_accuracy * 100.0f) << "%\n"
                  << "Final loss: "
                  << successful_configs[0].final_loss << "\n";
    }

    return 0;
}