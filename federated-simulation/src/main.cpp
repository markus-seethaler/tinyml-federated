#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include "FederatedSimulation/FederatedSimulation.h"
#include "HPO/HyperParameterOptimizer.h"
#include <algorithm>

// Helper function to parse command line arguments
bool getCmdOption(const std::vector<std::string>& args, const std::string& option, std::string& value) {
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == option && i + 1 < args.size()) {
            value = args[i + 1];
            return true;
        }
    }
    return false;
}

bool cmdOptionExists(const std::vector<std::string>& args, const std::string& option) {
    return std::find(args.begin(), args.end(), option) != args.end();
}

std::vector<size_t> parseTopology(const std::string& topologyStr) {
    std::vector<size_t> topology;
    std::stringstream ss(topologyStr);
    std::string item;
    
    while (std::getline(ss, item, ',')) {
        topology.push_back(std::stoul(item));
    }
    
    return topology;
}

void printUsage() {
    std::cout << "Usage: SmartBikeLockSimulation [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --hpo                 Run hyperparameter optimization\n";
    std::cout << "  --quick-search        Run a quicker hyperparameter search with reduced parameter space\n";
    std::cout << "  --rounds <N>          Set number of federated learning rounds (default: 200)\n";
    std::cout << "  --clients <N>         Set number of clients (default: 100)\n";
    std::cout << "  --samples <N>         Set samples per round (default: 20)\n";
    std::cout << "  --lr <rate>           Set learning rate (default: 0.75)\n";
    std::cout << "  --fraction <f>        Set client fraction (default: 0.3)\n";
    std::cout << "  --topology <layers>   Set neural network topology (default: 11,15,3)\n";
    std::cout << "                        Format: comma-separated layer sizes, e.g., 11,20,3\n";
    std::cout << "  --data-path <path>    Set path to data directory (default: ../data)\n";
    std::cout << "  --metrics <file>      Set metrics output file (default: federated_metrics.csv)\n";
    std::cout << "  --seed <N>            Set random seed (default: 42)\n";
    std::cout << "  --help                Display this help message\n";
}

int main(int argc, char* argv[]) {
    std::vector<std::string> args(argv + 1, argv + argc);
    
    // Check for help option first
    if (cmdOptionExists(args, "--help") || cmdOptionExists(args, "-h")) {
        printUsage();
        return 0;
    }
    
    // Default values
    std::string dataPath = "../data";
    uint32_t seed = 42;
    int rounds = 200;
    size_t numClients = 100;
    size_t samplesPerRound = 20;
    float learningRate = 0.75f;
    float clientFraction = 0.3f;
    std::vector<size_t> topology = {11, 15, 3};
    std::string metricsFile = "federated_metrics.csv";
    
    // Parse command line arguments
    std::string value;
    if (getCmdOption(args, "--data-path", value)) dataPath = value;
    if (getCmdOption(args, "--seed", value)) seed = std::stoul(value);
    if (getCmdOption(args, "--rounds", value)) rounds = std::stoi(value);
    if (getCmdOption(args, "--clients", value)) numClients = std::stoul(value);
    if (getCmdOption(args, "--samples", value)) samplesPerRound = std::stoul(value);
    if (getCmdOption(args, "--lr", value)) learningRate = std::stof(value);
    if (getCmdOption(args, "--fraction", value)) clientFraction = std::stof(value);
    if (getCmdOption(args, "--metrics", value)) metricsFile = value;
    
    if (getCmdOption(args, "--topology", value)) {
        topology = parseTopology(value);
        if (topology.size() < 2) {
            std::cerr << "Error: Topology must have at least input and output layers.\n";
            return 1;
        }
    }
    
    // Check which mode to run
    bool runHPO = cmdOptionExists(args, "--hpo");
    bool quickSearch = cmdOptionExists(args, "--quick-search");
    
    try {
        if (runHPO) {
            std::cout << "Running Hyperparameter Optimization\n";
            
            HyperParameterOptimizer optimizer(dataPath, seed);
            optimizer.set_max_rounds(rounds);
            optimizer.set_num_clients(numClients);
            optimizer.set_quick_search(quickSearch);
            
            optimizer.run_optimization();
        } else {
            std::cout << "Running Standard Federated Learning Simulation\n";
            
            FederatedSimulation simulation(dataPath, seed);
            simulation.set_fl_rounds(rounds);
            simulation.set_num_clients(numClients);
            simulation.set_samples_per_round(samplesPerRound);
            simulation.set_learning_rate(learningRate);
            simulation.set_client_fraction(clientFraction);
            simulation.set_topology(topology);
            simulation.set_metrics_file(metricsFile);
            
            simulation.run_simulation();
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}