#ifndef FEDERATED_SIMULATION_H
#define FEDERATED_SIMULATION_H

#include <vector>
#include <memory>
#include <string>
#include "DataLoader/DataLoader.h"
#include "DataPreprocessor/DataPreprocessor.h"
#include "FederatedClient/FederatedClient.h"
#include "FederatedServer/FederatedServer.h"

class FederatedSimulation {
public:
    FederatedSimulation(const std::string& data_path = "../data", 
                       uint32_t seed = 42);
    
    // Configuration setters
    void set_learning_rate(float lr) { learning_rate = lr; }
    void set_num_clients(size_t clients) { num_clients = clients; }
    void set_client_fraction(float fraction) { client_fraction = fraction; }
    void set_samples_per_round(size_t samples) { samples_per_round = samples; }
    void set_fl_rounds(int rounds) { fl_rounds = rounds; }
    void set_topology(const std::vector<size_t>& topo) { topology = topo; }
    void set_metrics_file(const std::string& file) { metrics_file = file; }
    
    // Run the simulation
    void run_simulation();
    
private:
    // Helper struct for tracking metrics during training
    struct TrainingMetrics {
        std::vector<std::vector<float>> predictions;
        std::vector<std::vector<float>> targets;
    };
    
    // Helper methods
    TrainingMetrics train_clients_online(
        const std::vector<size_t>& selected_clients,
        std::vector<std::unique_ptr<FederatedClient>>& clients,
        std::shared_ptr<DataPreprocessor> preprocessor,
        float learning_rate,
        size_t samples_per_client);
    
    float evaluate_test_set(
        FederatedClient& client,
        const std::vector<TrainingSample>& test_set);
    
    void write_metrics_to_csv(
        const std::string& filename,
        int round,
        float accuracy,
        float test_loss,
        float training_loss);
    
    void print_final_evaluation(
        FederatedClient& client,
        const std::vector<TrainingSample>& test_set);
    
    // Member variables
    std::string data_path;
    uint32_t seed;
    
    // Configuration parameters
    size_t num_clients = 100;
    float client_fraction = 0.3f;
    size_t samples_per_round = 20;
    float learning_rate = 0.75f;
    int fl_rounds = 200;
    std::vector<size_t> topology = {11, 15, 3};
    std::string metrics_file = "federated_metrics.csv";
};

#endif // FEDERATED_SIMULATION_H