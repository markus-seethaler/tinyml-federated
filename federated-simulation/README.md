# Federated Simulation

This folder contains the simulation environment for testing federated learning approaches for the TinyML-Federated smart bike lock project.

## Overview

The federated simulation environment allows researchers and developers to:

1. **Simulate Multiple Devices**: Create virtual clients that mimic Arduino Nano 33 BLE devices
2. **Test Different FL Configurations**: Experiment with various hyperparameters
3. **Evaluate Performance**: Analyze convergence rates, accuracy, and communication efficiency
4. **Optimize Hyperparameters**: Find optimal settings for real-world deployment
5. **Benchmark Different Architectures**: Compare various network topologies and configurations

## Components

The simulation environment consists of the following key components:

### Core Components
- **Neural Network**: Lightweight implementation of a feedforward neural network
- **Feature Extractor**: Extracts frequency domain and statistical features from raw accelerometer data
- **Data Loader**: Loads and manages motion data from CSV files
- **Data Preprocessor**: Normalizes data and prepares it for training

### Federated Learning Components
- **Federated Client**: Simulates Arduino clients with local training capabilities
- **Federated Server**: Implements model aggregation using Federated Averaging (FedAvg)
- **Federated Simulation**: Orchestrates the federated learning process
- **Hyperparameter Optimizer**: Performs grid search to find optimal configurations

### Evaluation Components
- **Metrics**: Calculates accuracy, loss, confusion matrix, and F1 scores

## Build Instructions

### Requirements

- CMake (version 3.15 or higher)
- C++ compiler with C++17 support
- FFTW3 library for FFT computation (https://www.fftw.org/)

### Building

1. Create a build directory:
   ```bash
   mkdir build && cd build
   ```

2. Configure with CMake:
   ```bash
   cmake ..
   ```

3. Build the project:
   ```bash
   make
   ```

## Usage

The simulation supports two main modes:

### 1. Standard Federated Learning Simulation

Run the simulation with default parameters:
```bash
./SmartBikeLockSimulation
```

This will:
- Load the dataset from the `/data` directory
- Create 100 simulated client devices
- Run 200 rounds of federated learning
- Log metrics to `federated_metrics.csv`
- Output a confusion matrix and F1 scores at the end

### 2. Hyperparameter Optimization

Run hyperparameter optimization to find the best configuration:
```bash
./SmartBikeLockSimulation --hpo
```

This will:
- Test multiple network topologies, learning rates, and other parameters
- Track which configurations meet the success criteria
- Log detailed metrics to `hyperparam_metrics.csv`
- Output the best configuration found

## Command Line Options

- `--hpo`: Run hyperparameter optimization
- `--rounds <N>`: Set the number of federated learning rounds (default: 200)
- `--clients <N>`: Set the number of clients (default: 100)
- `--samples <N>`: Set the number of samples per round (default: 20)
- `--lr <rate>`: Set the learning rate (default: 0.75)
- `--fraction <f>`: Set the client fraction (default: 0.3)
- `--topology <layers>`: Set the neural network topology (default: 11,15,3)
- `--data-path <path>`: Set the path to the data directory (default: ../data)

## Data Format

The simulation expects data in the same format as produced by the data collection server:

- A `motion_metadata.csv` file with sample metadata
- Individual CSV files in a `motion_data` subdirectory containing accelerometer readings

An example dataset is delivered with the repository.

## Output Files

The simulation produces the following output files:

- `federated_metrics.csv`: Contains accuracy and loss metrics for each round
- `hyperparam_metrics.csv`: Contains metrics for each hyperparameter configuration tested
- `best_config.json`: Contains the best hyperparameter configuration found

## Customization

You can customize the simulation by:

1. Modifying the network architecture in `main.cpp`
2. Adjusting the hyperparameter ranges in `HyperParameterOptimizer.cpp`
3. Implementing new aggregation algorithms in `FederatedServer.cpp`
4. Adding new feature extraction methods in `FeatureExtractor.cpp`

## Examples

### Example 1: Basic Simulation

```bash
./SmartBikeLockSimulation --rounds 100 --clients 50 --lr 0.5
```

### Example 2: Custom Network Topology

```bash
./SmartBikeLockSimulation --topology 11,25,10,3
```

### Example 3: Quick Hyperparameter Search

```bash
./SmartBikeLockSimulation --hpo --quick-search
```

## Troubleshooting

- **"Could not open metadata file"**: Ensure your data path is correct
- **"Failed to initialize FFTW"**: Check FFTW3 installation
- **Memory issues**: Reduce the number of clients or samples
- **Slow performance**: Try reducing the network size or number of training rounds