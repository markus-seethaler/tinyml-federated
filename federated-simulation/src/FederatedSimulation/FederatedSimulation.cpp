#include "FederatedSimulation/FederatedSimulation.h"
#include "Metrics/Metrics.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>

FederatedSimulation::FederatedSimulation(const std::string& data_path, uint32_t seed)
    : data_path(data_path), seed(seed) {
}

FederatedSimulation::TrainingMetrics FederatedSimulation::train_clients_online(
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

            // Get prediction before training (for loss calculation)
            auto prediction = clients[client_idx]->predict(sample.features);

            // Train on sample
            clients[client_idx]->train_on_sample(sample.features, sample.target, learning_rate);

            // Store predictions and targets for loss calculation
            metrics.predictions.push_back(prediction);
            metrics.targets.push_back(sample.target);
        }
    }

    return metrics;
}

float FederatedSimulation::evaluate_test_set(
    FederatedClient& client,
    const std::vector<TrainingSample>& test_set) {
    
    std::vector<std::vector<float>> predictions;
    std::vector<std::vector<float>> targets;

    // Get predictions for all test samples
    for (const auto& test_sample : test_set) {
        predictions.push_back(client.predict(test_sample.features));
        targets.push_back(test_sample.target);
    }

    return Metrics::accuracy(predictions, targets);
}

void FederatedSimulation::write_metrics_to_csv(
    const std::string& filename,
    int round,
    float accuracy,
    float test_loss,
    float training_loss) {
    
    bool file_exists = std::ifstream(filename).good();

    std::ofstream file;
    if (!file_exists) {
        file.open(filename);
        file << "Round,Accuracy,TestLoss,TrainingLoss\n";
    } else {
        file.open(filename, std::ios_base::app);
    }

    file << round << ","
         << std::fixed << std::setprecision(4)
         << (accuracy * 100.0f) << ","
         << test_loss << ","
         << training_loss << "\n";

    file.close();
}

void FederatedSimulation::print_final_evaluation(
    FederatedClient& client,
    const std::vector<TrainingSample>& test_set) {
    
    float test_accuracy = evaluate_test_set(client, test_set);
    std::cout << "\nFinal Test Set Evaluation:" << std::endl;
    std::cout << "Accuracy: " << (test_accuracy * 100.0f) << "%" << std::endl;

    // Get predictions for confusion matrix
    std::vector<std::vector<float>> predictions;
    std::vector<std::vector<float>> targets;
    for (const auto& test_sample : test_set) {
        predictions.push_back(client.predict(test_sample.features));
        targets.push_back(test_sample.target);
    }

    // Calculate and print confusion matrix
    auto conf_matrix = Metrics::confusion_matrix(predictions, targets);
    Metrics::print_confusion_matrix(conf_matrix);

    // Calculate and print F1 scores
    auto f1_scores = Metrics::f1_scores(conf_matrix);
    std::cout << "\nF1 Scores per class:" << std::endl;
    for (size_t i = 0; i < f1_scores.size(); i++) {
        std::cout << "Class " << i << ": " << f1_scores[i] << std::endl;
    }
    
    // Calculate and print ROC AUC scores
    auto auc_scores = Metrics::roc_auc(predictions, targets);
    std::cout << "\nROC AUC Scores per class:" << std::endl;
    for (size_t i = 0; i < auc_scores.size(); i++) {
        std::cout << "Class " << i << ": " << auc_scores[i] << std::endl;
    }
}

void FederatedSimulation::run_simulation() {
    try {
        // Load dataset
        DataLoader loader(data_path);
        auto dataset = loader.load_dataset("motion_metadata.csv");
        std::cout << "Loaded " << dataset.size() << " samples\n\n";

        // Prepare data for training
        auto preprocessor = std::make_shared<DataPreprocessor>(seed);
        preprocessor->prepare_dataset(dataset);

        // Create federated components
        FederatedServer server(seed);
        std::vector<std::unique_ptr<FederatedClient>> clients;

        // Initialize clients
        for (size_t i = 0; i < num_clients; i++) {
            clients.push_back(std::make_unique<FederatedClient>(topology, preprocessor, seed + i));
        }

        // Remove existing metrics file if it exists
        std::remove(metrics_file.c_str());

        // Get test samples for evaluation
        auto test_samples = preprocessor->get_test_set();
        if (test_samples.empty()) {
            throw std::runtime_error("No test samples available");
        }

        std::cout << "\nStarting federated learning with:" << std::endl;
        std::cout << "  Clients: " << num_clients << std::endl;
        std::cout << "  Client Fraction: " << client_fraction << std::endl;
        std::cout << "  Samples Per Round: " << samples_per_round << std::endl;
        std::cout << "  Learning Rate: " << learning_rate << std::endl;
        std::cout << "  Rounds: " << fl_rounds << std::endl;
        
        std::cout << "  Topology: [";
        for (size_t i = 0; i < topology.size(); i++) {
            std::cout << topology[i];
            if (i < topology.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;

        // Federated Learning Rounds
        for (int round = 0; round < fl_rounds; round++) {
            std::cout << "\n=== Federated Learning Round " << (round + 1) << " ===\n";

            // Select subset of clients for this round
            auto selected_clients = server.select_clients(clients.size(), client_fraction);
            std::cout << "Selected " << selected_clients.size() << " clients for this round\n";

            // Local training on selected clients
            std::cout << "\nLocal training with " << samples_per_round
                      << " samples per client...\n";

            // Train selected clients
            auto training_metrics = train_clients_online(
                selected_clients, clients, preprocessor,
                learning_rate, samples_per_round);

            // Calculate training loss
            float training_loss = Metrics::cross_entropy_loss(
                training_metrics.predictions,
                training_metrics.targets);

            // Collect weights only from selected clients
            std::vector<std::vector<float>> client_weights;
            for (size_t client_idx : selected_clients) {
                client_weights.push_back(clients[client_idx]->get_weights());
            }

            // Average weights from selected clients
            auto averaged_weights = server.average_weights(client_weights);

            // Update ALL clients with averaged weights
            for (auto& client : clients) {
                client->set_weights(averaged_weights);
            }

            // Calculate test metrics
            std::vector<std::vector<float>> test_predictions;
            std::vector<std::vector<float>> test_targets;

            for (const auto& test_sample : test_samples) {
                test_predictions.push_back(clients[0]->predict(test_sample.features));
                test_targets.push_back(test_sample.target);
            }

            float test_loss = Metrics::cross_entropy_loss(test_predictions, test_targets);
            float test_accuracy = Metrics::accuracy(test_predictions, test_targets);

            // Write to CSV
            write_metrics_to_csv(metrics_file, round + 1, test_accuracy, test_loss, training_loss);

            // Display metrics
            std::cout << "Round " << (round + 1) << " metrics:\n"
                      << "  Training Loss: " << training_loss << "\n"
                      << "  Test Loss: " << test_loss << "\n"
                      << "  Test Accuracy: " << (test_accuracy * 100.0f) << "%\n";
        }

        // After FL rounds complete
        std::cout << "\nPerforming final evaluation..." << std::endl;
        print_final_evaluation(*clients[0], test_samples);
        
        std::cout << "\nFederated learning simulation complete." << std::endl;
        std::cout << "Results saved to " << metrics_file << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        throw;
    }
}